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
        return self.vllm_engine.generate(prompts)

# --------------------------------------------------------
# GRPOTrainer 拡張版：LoRA + colocate vLLM + frozen ref_model
# --------------------------------------------------------

class GRPOTrainerWithLoRASync(ColocateVLLMLoRASyncMixin, GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # colocate vLLM起動
        if getattr(self.args, "use_vllm", False) and getattr(self.args, "vllm_mode", None) == "colocate":
            self.init_vllm_engine()

        # === 参照モデルを deepcopy で固定 ===
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def generate_completions(self, prompts):
        if hasattr(self, "vllm_engine"):
            return self.vllm_generate(prompts)
        else:
            return self.model.generate(**prompts)

    def compute_logprobs(self, model, input_ids, attention_mask):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logp = -outputs.loss
        return logp

    def compute_policy_ratio(self, input_ids, attention_mask):
        logp_policy = self.compute_logprobs(self.model, input_ids, attention_mask)
        logp_ref = self.compute_logprobs(self.ref_model, input_ids, attention_mask)
        return torch.exp(logp_policy - logp_ref)
