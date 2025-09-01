# Copyright 2025 The HuggingFace Team.
# Licensed under the Apache License, Version 2.0

import logging
import os
import sys
from typing import Optional

import socket
import torch
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

# DeepSpeed は任意: 無い環境でも動くようにガード
try:
    from deepspeed import DeepSpeedEngine  # type: ignore
except Exception:
    DeepSpeedEngine = None  # noqa: N816

logger = logging.getLogger(__name__)

# ---- distributed + cuda sanity (put BEFORE accelerate init) ----
host = socket.gethostname()
rank = os.environ.get("RANK")
lrank = int(os.environ.get("LOCAL_RANK", "0"))
wsize = os.environ.get("WORLD_SIZE")
maddr = os.environ.get("MASTER_ADDR")
mport = os.environ.get("MASTER_PORT")

# env snapshot
print(
    "[Env@Worker]",
    f"host={host}", f"RANK={rank}", f"LOCAL_RANK={lrank}",
    f"WORLD_SIZE={wsize}", f"MASTER_ADDR={maddr}", f"MASTER_PORT={mport}",
    file=sys.stderr,
    flush=True,
)

# pin device early to silence NCCL warnings
if torch.cuda.is_available():
    torch.cuda.set_device(lrank)
    cur_dev = torch.cuda.current_device()
    dev_cnt = torch.cuda.device_count()
    dev_name = torch.cuda.get_device_name(cur_dev)
else:
    cur_dev = None
    dev_cnt = 0
    dev_name = None

print(
    "[CudaSanity]",
    f"host={host}", f"LOCAL_RANK={lrank}",
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
    f"device_count={dev_cnt}", f"current_device={cur_dev}", f"device_name={dev_name}",
    file=sys.stderr,
    flush=True,
)
# ---------------------------------------------------------------


def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig):
    # Seed
    set_seed(training_args.seed)

    # Logging
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
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Resume checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
        
    # W&B
    if "wandb" in (training_args.report_to or []):
        init_wandb_training(training_args)

    # Dataset
    dataset = get_dataset(script_args)

    # ---- decide columns ----
    # prompt column comes from config.py (GRPOScriptArguments)
    qcol = script_args.dataset_prompt_column or "prompt"
    # solution column: do NOT add new CLI args; infer from dataset
    # prefer `solution`, else fallback to `answer`
    preferred_solution_cols = ["solution", "answer"]
    logger.info(f"Using prompt column = '{qcol}', solution candidates = {preferred_solution_cols}")

    # clean up stray chat columns
    def _drop_messages_if_exist(ds):
        if "messages" in ds.column_names:
            ds = ds.remove_columns("messages")
        return ds

    dataset = {split: _drop_messages_if_exist(dataset[split]) for split in dataset}

    # build prompt & ensure `solution` field exists for rewards
    def _prepare(example):
        if qcol not in example:
            raise ValueError(f"Dataset Question Field Error: '{qcol}' is not in the example.")
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example[qcol]})

        # choose solution column
        sol = None
        if "solution" in example and example["solution"] is not None:
            sol = example["solution"]
        else:
            for c in preferred_solution_cols:
                if c in example and example[c] is not None:
                    sol = example[c]
                    break
        if sol is None:
            raise ValueError(
                f"Dataset Solution Field Error: none of {preferred_solution_cols} found in the example."
            )
        return {"prompt": prompt, "solution": sol}

    for split in list(dataset.keys()):
        dataset[split] = dataset[split].map(_prepare, desc=f"prepare/{split}")

    # Tokenizer & Model
    tokenizer = get_tokenizer(model_args, training_args)
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    logger.info(
        f"Tokenizer check: eos='{tokenizer.eos_token}'(id={tokenizer.eos_token_id}), "
        f"pad='{tokenizer.pad_token}'(id={tokenizer.pad_token_id}), "
        f"chat_template={'set' if getattr(tokenizer, 'chat_template', None) else 'none'}"
    )

    # Rewards
    reward_funcs = get_reward_funcs(script_args)  # accuracy_reward は dataset['solution'] を参照できる

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # ---- DeepSpeed config snapshot (optional) ----
    def debug_deepspeed_config(trainer_obj):
        if DeepSpeedEngine is None:
            print("[DS-DEBUG] deepspeed not available; skipping snapshot", flush=True)
            return
        eng = trainer_obj.model_wrapped
        if not isinstance(eng, DeepSpeedEngine):
            print("[DS-DEBUG] model_wrapped is not DeepSpeedEngine:", type(eng), flush=True)
            return
        cfg = eng.config
        zc = getattr(cfg, "zero_config", {})
        print("[DS-DEBUG] ---- DeepSpeed Config Snapshot ----", flush=True)
        print("[DS-DEBUG] zero_enabled:", getattr(cfg, "zero_enabled", None), flush=True)
        print("[DS-DEBUG] stage:", zc.get("stage"), flush=True)
        print("[DS-DEBUG] offload_param:", zc.get("offload_param"), flush=True)
        print("[DS-DEBUG] offload_optimizer:", zc.get("offload_optimizer"), flush=True)
        print("[DS-DEBUG] nvme_path exists:", os.path.isdir(zc.get("offload_param", {}).get("nvme_path", "")), flush=True)
        print("[DS-DEBUG] overlap_comm:", zc.get("overlap_comm"), flush=True)
        print("[DS-DEBUG] contiguous_gradients:", zc.get("contiguous_gradients"), flush=True)
        print("[DS-DEBUG] aio:", zc.get("aio"), flush=True)

    debug_deepspeed_config(trainer)

    # Train
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "dataset_name": getattr(script_args, "dataset_name", None),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Eval
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # ここで **新しい dataclass を渡さない**（衝突回避）
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
