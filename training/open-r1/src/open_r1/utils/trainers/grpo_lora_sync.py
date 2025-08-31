import logging
import copy
import torch

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from trl import GRPOTrainer

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Colocate モードで vLLM エンジンと同期する Mixin
# --------------------------------------------------------

class ColocateVLLMLoRASyncMixin:
    """
    Mixin to inject a LoRA-adapted model directly into vLLM colocate engine.
    Ensures inference for reward calculation uses the updated LoRA parameters.
    """

    def init_vllm_engine(self):
        logger.info("[Colocate] Initializing vLLM engine with training model (LoRA-injected)")
        
        self.vllm_engine = AsyncLLMEngine.from_model(
            model=self.model,
            tokenizer=self.tokenizer,
            tensor_parallel_size=self.args.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.max_prompt_length + self.args.max_completion_length,
        )

    def vllm_generate(self, prompts):
        completions = self.vllm_engine.generate(prompts)

        logger.info(f"[vllm_generate] prompts[0]: {prompts[0] if prompts else 'EMPTY'}")

        if completions and hasattr(completions[0], "outputs") and completions[0].outputs:
            logger.info(f"[vllm_generate] completions[0]: {completions[0].outputs[0].text.strip()}")
        else:
            logger.warning("[vllm_generate] No completions returned or unexpected format.")
        return completions

# --------------------------------------------------------
# GRPOTrainer 拡張版：LoRA + colocate vLLM + frozen ref_model
# --------------------------------------------------------

class GRPOTrainerWithLoRASync(ColocateVLLMLoRASyncMixin, GRPOTrainer):
    """
    拡張 GRPOTrainer:
    - LoRA モデルでの方策訓練
    - deepcopy による固定方策の保持（旧ポリシーとして）
    - colocate モードでの vLLM 推論
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # colocate モードでの vLLM エンジン初期化
        if getattr(self.args, "use_vllm", False) and getattr(self.args, "vllm_mode", None) == "colocate":
            self.init_vllm_engine()

        # ===== 固定の旧方策として ref_model を deepcopy で保持 =====
        logger.info("[RefModel] Freezing initial policy (deepcopy)")
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def generate_completions(self, prompts):
        """
        使用可能なら vLLM で推論。なければ fallback。
        """
        if hasattr(self, "vllm_engine"):
            logger.info("[generate_completions] using vLLM engine for generation")
            return self.vllm_generate(prompts)
        else:
            logger.info("[generate_completions] using HF model.generate fallback")
            return self.model.generate(**prompts)

    def compute_logprobs(self, model, input_ids, attention_mask):
        """
        任意のモデル（policy or ref）から logp を取得
        """
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logp = -outputs.loss  # より厳密には tokenwise gather が望ましい
        return logp

    def compute_policy_ratio(self, input_ids, attention_mask):
        """
        方策比（新方策 / 旧方策）を計算
        """
        logp_policy = self.compute_logprobs(self.model, input_ids, attention_mask)
        logp_ref = self.compute_logprobs(self.ref_model, input_ids, attention_mask)
        return torch.exp(logp_policy - logp_ref)
