import logging
from vllm.engine.async_llm_engine import AsyncLLMEngine
from trl import GRPOTrainer

logger = logging.getLogger(__name__)

class ColocateVLLMLoRASyncMixin:
    """
    Mixin to inject a LoRA-adapted model directly into vLLM colocate engine.
    Ensures inference for reward calculation uses the updated LoRA parameters.
    """

    def init_vllm_engine(self):
        logger.info("[Colocate] Initializing vLLM engine with training model (LoRA-injected)")
        
        # Assume self.model is already a PEFT (LoRA) wrapped model
        self.vllm_engine = AsyncLLMEngine.from_model(
            model=self.model,
            tokenizer=self.tokenizer,
            tensor_parallel_size=self.args.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.max_prompt_length + self.args.max_completion_length,
        )

    def vllm_generate(self, prompts):
        """
        Generates completions via colocated vLLM engine using the LoRA-updated model.
        """
        return self.vllm_engine.generate(prompts)

# ------------------ Inject into GRPOTrainer ------------------ #

class GRPOTrainerWithLoRASync(ColocateVLLMLoRASyncMixin, GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if getattr(self.args, "use_vllm", False) and getattr(self.args, "vllm_mode", None) == "colocate":
            self.init_vllm_engine()

        # === 追加: 固定参照モデルの読み込み ===
        self.ref_model = self.load_ref_model()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def generate_completions(self, prompts):
        if hasattr(self, "vllm_engine"):
            return self.vllm_generate(prompts)
        else:
            return self.model.generate(**prompts)  # fallback (slow)

    def compute_logprobs(self, model, input_ids, attention_mask):
        # 任意のモデル（self.model または self.ref_model）で logp を取得
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logp = -outputs.loss  # または logits から log_softmax → gather
        return logp

    def compute_policy_ratio(self, input_ids, attention_mask):
        logp_policy = self.compute_logprobs(self.model, input_ids, attention_mask)
        logp_ref = self.compute_logprobs(self.ref_model, input_ids, attention_mask)
        return torch.exp(logp_policy - logp_ref)

    def load_ref_model(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            torch_dtype=self.args.torch_dtype,
            trust_remote_code=self.args.trust_remote_code,
        )
        ref_model = PeftModel.from_pretrained(base, self.args.ref_lora_path)
        return ref_model

# ------------------ Usage ------------------ #
# In grpo.py or trainer init:
# from your_module import GRPOTrainerWithLoRASync
# trainer = GRPOTrainerWithLoRASync(...)
