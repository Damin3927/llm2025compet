# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
grpo.py をベースに必要に応じて DPO の処理に書き換える形で実装
"""

import logging
import os
import sys
from dataclasses import dataclass, field
import yaml
import datetime

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.get_data import get_data_from_config

def load_config_from_yaml(file_path: str) -> DataConfig:
    """
    Loads dataset configurations from a YAML file.
    """
    print(f"⚙️  Loading configuration from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Map the list of dicts from YAML to a list of DatasetClass dataclass instances
    dataset_configs = [DatasetClass(**item) for item in data['neko_sft_datasets']]
    return DataConfig(datasets=dataset_configs)

def get_dataconfig():
    parser = argparse.ArgumentParser(description="Load and combine datasets from a YAML configuration file.")
    parser.add_argument(
        "--dataconfig",
        type=str,
        required=True,
        help="Path to the YAML data configuration file."
    )
    args, _ = parser.parse_known_args()
    return load_config_from_yaml(args.dataconfig)

# TRL の DPO 用ライブラリ
from trl import ORPOConfig, ORPOTrainer, ModelConfig, TrlParser, ScriptArguments, get_peft_config

logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args, data_config):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load Dataset
    ################
    logger.info("*** Loading dataset ***")
    if data_config is None:
        logger.info(f"Loading dataset: {script_args.dataset_name}")
        dataset = datasets.load_dataset(script_args.dataset_name, script_args.dataset_config)
    else:
        dataset = get_data_from_config(data_config)
    # dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    logger.info("*** Loading Tokenizer ***")
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Format into pariwise preference
    def format_pref(example, tokenizer=tokenizer):
        # todo: 本学習の際にはキーを "question" "preferred_output", "non_preferred_output" に書き換える
        return {
            "prompt": example["question"],
            "chosen": example["preferred_output"],
            "rejected": example["non_preferred_output"]
        }
        """return {
            "prompt": example["question"],
            "chosen": example["chosen_response"]+tokenizer.eos_token,
            "rejected": example["rejected_response"]+tokenizer.eos_token
        }"""
    
    dataset["train"] = dataset["train"].map(format_pref, remove_columns=dataset["train"].column_names)

    #############################
    # Initialize the ORPO trainer
    #############################
    logger.info("*** Initialize DPO Trainer ***")
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        #callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        logger.info(f"last checkpoint: {last_checkpoint}")
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    transformers.logging.set_verbosity_info()
    #if training_args.push_to_hub:
    logger.info("Pushing to hub...")
    trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, ORPOConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True, fail_with_unknown_args=False)
    data_config = get_dataconfig()
    print(f"Data Configration loaded: {data_config}")
    main(script_args, training_args, model_args, data_config)
