# coding=utf-8
# UnslothとSFTTrainerでの動作確認用スクリプト

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from transformers import set_seed, TrainingArguments
from trl import SFTTrainer, TrlParser
from unsloth import FastLanguageModel

# open-r1のカスタム設定クラスを利用
from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2-1.5B-Instruct")
    model_revision: str = field(default="main")
    trust_remote_code: bool = field(default=True)
    use_unsloth_dtype: Optional[str] = field(default=None)

def main(script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelArguments):
    set_seed(training_args.seed)

    # --- 1. モデルとトークナイザーの読み込み ---
    unsloth_dtype = None
    if model_args.use_unsloth_dtype:
        unsloth_dtype = getattr(torch, model_args.use_unsloth_dtype)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        dtype=unsloth_dtype,
        load_in_4bit=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. LoRAの設定 ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        random_state=training_args.seed,
        task_type="CAUSAL_LM",
    )

    # --- 3. データセットの準備 ---
    dataset = get_dataset(script_args)
    # SFT用に'text'列を作成する簡単な処理を追加
    # (実際のデータ形式に合わせて調整が必要な場合があります)
    def format_for_sft(example):
        # この部分はデータセットの形式に合わせて調整してください
        # 例: {"prompt": "...", "chosen": "..."} -> "<s>[INST]...[/INST]...</s>"
        prompt = example.get(script_args.dataset_prompt_column, "")
        chosen = example.get(script_args.dataset_chosen_column, "")
        return {"text": f"<s>[INST] {prompt} [/INST] {chosen} </s>"}

    dataset = dataset.map(format_for_sft)

    # --- 4. SFTTrainerの初期化 ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
    )

    # --- 5. 学習の実行 ---
    trainer.train()
    print("SFTによる動作確認が完了しました！")

if __name__ == "__main__":
    # open-r1のSFTConfigとScriptArgumentsを使用
    parser = TrlParser((ScriptArguments, SFTConfig, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
