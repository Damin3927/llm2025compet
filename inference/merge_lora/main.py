import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from peft import PeftModel
from huggingface_hub import HfApi, login


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f}{units[i]}"


essential_env = [
    "HF_HUB_ENABLE_HF_TRANSFER",
    "TRANSFORMERS_OFFLINE",
]


def pick_dtype(name: str):
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def detect_quantized_config(cfg: PretrainedConfig) -> bool:
    # Heuristic: many quantized checkpoints attach quantization_config or bnb flags.
    if getattr(cfg, "quantization_config", None) is not None:
        return True
    for k in (
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_quant_type",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
    ):
        if hasattr(cfg, k):
            return True
    return False


def tiny_forward(model) -> None:
    """Tiny forward pass to sanity-check shapes/devices only."""
    try:
        with torch.inference_mode():
            vocab_size = int(getattr(model.config, "vocab_size", 32000))
            # Infer device from first parameter; fallback to CPU
            device = torch.device("cpu")
            for param in model.parameters():
                device = param.device
                break

            x = torch.randint(0, vocab_size, (1, 8), device=device)
            _ = model(input_ids=x)
            print("[INFO] Smoke test passed")
    except Exception as e:
        print(f"[WARN] tiny forward failed (non-fatal): {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="merge-lora",
        description=(
            "Merge one or more PEFT-LoRA adapters into a base CausalLM. "
            "Supports huge models via device_map='auto' and offload_folder."
        ),
    )
    p.add_argument(
        "--base-model", required=True, help="Base model repo_id or local path"
    )
    p.add_argument(
        "--adapters",
        required=True,
        nargs="+",
        help=(
            "One or more LoRA adapters (repo_id or local path). If multiple are provided, "
            "they are merged sequentially in the given order."
        ),
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for the merged model"
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="bf16 recommended",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help="Device map for loading (e.g., auto, cpu). Use 'auto' to utilize available GPUs with offload.",
    )
    p.add_argument(
        "--offload-dir",
        default=os.environ.get("OFFLOAD_DIR", "offload_dir"),
        help="Directory for CPU/SSD offload shards (created if missing)",
    )
    p.add_argument(
        "--max-shard-size",
        default=os.environ.get("MAX_SHARD_SIZE", "5GB"),
        help='Max safetensors shard size (e.g., "5GB", "10GB"). Larger -> fewer files, less I/O overhead.',
    )
    p.add_argument("--revision", default=None, help="Model revision/branch if needed")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable if model uses custom code",
    )
    p.add_argument(
        "--allow-quantized-merge",
        action="store_true",
        help="Allow merging into a quantized base (NOT recommended). Default: abort if detected.",
    )
    p.add_argument(
        "--save-tokenizer", action="store_true", help="Also save tokenizer to out-dir"
    )
    p.add_argument(
        "--smoke-test", action="store_true", help="Run tiny forward before/after merge"
    )
    p.add_argument("--print-plan", action="store_true", help="Print plan JSON and exit")

    # Hugging Face upload options
    p.add_argument(
        "--upload-to-hub",
        action="store_true",
        help="Upload merged model to Hugging Face Hub",
    )
    p.add_argument(
        "--hub-repo-id",
        default=None,
        help="Hugging Face repository ID (required if uploading)",
    )
    p.add_argument(
        "--hub-token", default=None, help="Hugging Face token (or set HF_TOKEN env var)"
    )
    p.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the uploaded repository private",
    )
    return p.parse_args()


def upload_to_huggingface(
    model_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = False
) -> None:
    """Upload merged model to Hugging Face Hub."""
    print(f"[INFO] Uploading to Hugging Face Hub: {repo_id}")

    # Get token from environment if not provided
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "[ERROR] No Hugging Face token provided. Set HF_TOKEN env var or use --hub-token",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify required files exist for vLLM compatibility
    required_files = ["config.json"]
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(
            f"[ERROR] Missing required files for vLLM compatibility: {missing_files}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check for model files (safetensors or pytorch_model.bin)
    has_safetensors = any(model_dir.glob("*.safetensors"))
    has_pytorch = any(model_dir.glob("pytorch_model*.bin"))

    if not (has_safetensors or has_pytorch):
        print(
            "[ERROR] No model weight files found (*.safetensors or pytorch_model*.bin)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Login to Hugging Face
        login(token=hf_token)
        api = HfApi()

        # Create repository if it doesn't exist
        print(f"[INFO] Creating/updating repository: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model",  # Explicitly specify this is a model repository
        )

        # Upload all files in the model directory
        print("[INFO] Uploading model files...")
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            commit_message="Upload merged LoRA model via merge_lora tool",
            ignore_patterns=[".git*", "__pycache__", "*.pyc", ".DS_Store"],
        )

        print(f"[INFO] Successfully uploaded to: https://huggingface.co/{repo_id}")
        print(f"[INFO] Model is ready for use with vLLM:")
        print(f"       vllm serve {repo_id}")

    except Exception as e:
        print(f"[ERROR] Failed to upload to Hugging Face: {e}", file=sys.stderr)
        raise


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.offload_dir).mkdir(parents=True, exist_ok=True)

    dtype = pick_dtype(args.dtype)

    plan = {
        "base_model": args.base_model,
        "adapters": args.adapters,
        "out_dir": str(out_dir.resolve()),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "offload_dir": args.offload_dir,
        "max_shard_size": args.max_shard_size,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "allow_quantized_merge": args.allow_quantized_merge,
        "save_tokenizer": args.save_tokenizer,
        "upload_to_hub": args.upload_to_hub,
        "hub_repo_id": args.hub_repo_id,
        "hub_private": args.hub_private,
        "torch_version": torch.__version__,
        "env": {k: os.environ.get(k) for k in essential_env},
    }

    if args.print_plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    # Validate upload arguments
    if args.upload_to_hub and not args.hub_repo_id:
        print(
            "[ERROR] --hub-repo-id is required when --upload-to-hub is specified",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[INFO] Merge plan:")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    # Config probe to detect quantized base
    print("[INFO] Loading base model config…")
    cfg = AutoConfig.from_pretrained(
        args.base_model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if detect_quantized_config(cfg) and not args.allow_quantized_merge:
        print(
            "[ERROR] Detected a quantized base (4/8bit) in config. "
            "Merging directly is unsafe. Load a 16-bit/bf16 base or pass --allow-quantized-merge to force.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Load base model with offload to handle huge checkpoints
    print("[INFO] Loading base model weights (this may take a long time)…")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        offload_folder=args.offload_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # Attach and merge adapters sequentially
    model = base
    for i, adapter in enumerate(args.adapters):
        print(f"[INFO] Attaching LoRA adapter {i+1}/{len(args.adapters)}: {adapter}")
        try:
            model = PeftModel.from_pretrained(model, adapter)
        except Exception as e:
            print(f"[ERROR] Failed to load adapter {adapter}: {e}", file=sys.stderr)
            sys.exit(1)

        if args.smoke_test:
            print("[INFO] Running tiny forward (pre-merge)…")
            tiny_forward(model)

        print("[INFO] Merging adapter into base weights…")
        try:
            merge_fn = getattr(model, "merge_and_unload", None)
            if merge_fn is None:
                raise RuntimeError(
                    "merge_and_unload() not found on model. Ensure PEFT PeftModel was applied."
                )
            model = merge_fn()
        except Exception as e:
            print(f"[ERROR] Failed to merge adapter {adapter}: {e}", file=sys.stderr)
            sys.exit(1)

    if args.smoke_test:
        print("[INFO] Running tiny forward (post-merge)…")
        tiny_forward(model)

    print("[INFO] Saving merged model (sharded safetensors)…")
    try:
        model.save_pretrained(
            out_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        print(f"[INFO] Model saved to: {out_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}", file=sys.stderr)
        sys.exit(1)

    if args.save_tokenizer:
        try:
            print("[INFO] Saving tokenizer…")
            tok = AutoTokenizer.from_pretrained(
                args.base_model,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
            )
            tok.save_pretrained(out_dir)
            print(f"[INFO] Tokenizer saved to: {out_dir}")
        except Exception as e:
            print(f"[WARN] Failed to save tokenizer (non-fatal): {e}", file=sys.stderr)

    # Verify vLLM compatibility
    config_file = out_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            print(
                f"[INFO] Model architecture: {config.get('architectures', ['Unknown'])}"
            )
            print(f"[INFO] Model is ready for vLLM inference")
        except Exception as e:
            print(f"[WARN] Could not read config.json: {e}", file=sys.stderr)
    else:
        print(
            "[WARN] config.json not found - may affect vLLM compatibility",
            file=sys.stderr,
        )

    with open(out_dir / "MERGE_METADATA.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    total = 0
    for p in out_dir.glob("*.safetensors"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    print(
        f"[INFO] Done. Total tensor bytes (safetensors): {human_bytes(total)} | dir: {out_dir}"
    )

    # Upload to Hugging Face Hub if requested
    if args.upload_to_hub:
        upload_to_huggingface(
            model_dir=out_dir,
            repo_id=args.hub_repo_id,
            token=args.hub_token,
            private=args.hub_private,
        )


if __name__ == "__main__":
    main()
