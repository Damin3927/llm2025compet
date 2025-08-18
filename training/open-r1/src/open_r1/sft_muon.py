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
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig, DatasetClass, DataConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from open_r1.get_datas import get_datas_from_config

import argparse
import yaml
import torch
import datetime


import deepspeed  # NEW: for set_z3_leaf_modules
print(f"DeepSpeed version: {deepspeed.__version__}")

# flash_muonオプティマイザをインポート
from open_r1.muon import Muon
FLASH_MUON_AVAILABLE = True
print("flash_muon successfully imported.")


# NEW: try to import the concrete Qwen3 MoE block class (falls back to name match)
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock as _QwenSparseMoeBlock,
    )
    print(f"Using Qwen3 Sparse MoE Block: {_QwenSparseMoeBlock.__name__}")
except Exception:
    _QwenSparseMoeBlock = None
    print("Falling back to generic Sparse MoE Block.")



logger = logging.getLogger(__name__)
print(f"Last modified time is 2025-0817-0500-JST")

def load_config_from_yaml(file_path: str) -> DataConfig:
    """Loads dataset configurations from a YAML file."""
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
    args, unknown = parser.parse_known_args()
    return load_config_from_yaml(args.dataconfig)

class HybridOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        # Trainerがスケジューラを作成するためにparam_groupsを必要とする
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        for opt, state in zip(self.optimizers, state_dict):
            opt.load_state_dict(state)

def main(script_args, training_args, model_args, data_config: DataConfig):
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

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Data configuration: {data_config}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    if data_config is None:
        dataset = get_dataset(script_args)
    else:
        dataset = get_datas_from_config(data_config)
    print(f"Loaded dataset: {dataset}")
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    # NEW: MoE × ZeRO-3 安定化（leaf module 指定）
    if getattr(model.config, "model_type", "") == "qwen3_moe":
        print("[MoE] Detected model_type='qwen3_moe'. Applying ZeRO-3 leaf module setting...")

        # （任意）ルータのロジットを有効化したい場合はコメントアウトを外す
        # if hasattr(model.config, "output_router_logits"):
        #     model.config.output_router_logits = True
        #     print("[MoE] Enabled output_router_logits=True")

        if _QwenSparseMoeBlock is not None:
            deepspeed.utils.set_z3_leaf_modules(model, [_QwenSparseMoeBlock])
            print("[MoE] Set ZeRO-3 leaf module: Qwen3MoeSparseMoeBlock (direct import)")
        else:
            print("[MoE] Direct import of Qwen3MoeSparseMoeBlock failed; falling back to name scan...")
            set_flag = False
            for m in model.modules():
                if "SparseMoeBlock" in m.__class__.__name__:
                    deepspeed.utils.set_z3_leaf_modules(model, [m.__class__])
                    print(f"[MoE] Set ZeRO-3 leaf module by name: {m.__class__.__name__}")
                    set_flag = True
                    break
            if not set_flag:
                print("[MoE][WARN] Could not find a SparseMoeBlock; ZeRO-3 leaf NOT set (collectives may hang).")

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
        
    # オプティマイザの作成
    custom_optimizer = None
    if FLASH_MUON_AVAILABLE:
        logger.info("Creating custom Hybrid Muon/AdamW optimizer.")
        
        try:
            body_module = model.model
            embed_module = model.get_input_embeddings()
            head_module = model.get_output_embeddings()
        except AttributeError:
            logger.warning("Could not find standard model attributes. Falling back to a simple parameter split.")
            body_module, embed_module, head_module = model, torch.nn.Module(), torch.nn.Module()

        muon_params = [p for p in body_module.parameters() if p.ndim >= 2]
        adamw_params = (
            [p for p in body_module.parameters() if p.ndim < 2]
            + list(embed_module.parameters())
            + list(head_module.parameters())
        )
        
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        logger.info(f"Initializing Muon with rank={rank} and world_size={world_size}")
        
        optimizer_muon = Muon(
            muon_params, 
            lr=0.02, 
            momentum=0.95,
            rank=rank,                 # rank を渡す
            world_size=world_size      # world_size を渡す
        )
        adam_kwargs = {
            "lr": training_args.learning_rate,
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
            "weight_decay": training_args.weight_decay,
        }
        optimizer_adamw = torch.optim.AdamW(adamw_params, **adam_kwargs)
        
        logger.info(f"Muon optimizer initialized for {len(muon_params)} parameter groups.")
        logger.info(f"AdamW optimizer initialized for {len(adamw_params)} parameter groups with args: {adam_kwargs}")

        custom_optimizer = HybridOptimizer([optimizer_muon, optimizer_adamw])
    
    optimizers = (custom_optimizer, None) if custom_optimizer else (None, None)


    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        optimizers=optimizers,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
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
    if data_config is not None:
        kwargs["dataset_name"] = [dataset.name for dataset in data_config.datasets]
    
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
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, unknown_args = parser.parse_args_and_config(return_remaining_strings = True, fail_with_unknown_args = False)
    data_config = get_dataconfig()
    print(f"Data configuration loaded: {data_config}")
    main(script_args, training_args, model_args, data_config)
