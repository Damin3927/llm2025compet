from typing import TypedDict, NotRequired
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion

class Param(TypedDict):
    temperature: float
    max_tokens: int
    top_p: NotRequired[float]


DEFAULT_PARAMS: dict[str, Param] = {
    # https://huggingface.co/Qwen/Qwen3-32B#best-practices
    "Qwen/Qwen3-32B": {
        "temperature": 0.6,
        "max_tokens": 32_768,
        "top_p": 0.95,
    },
    "Qwen/Qwen3-235B-A22B": {
        "temperature": 0.6,
        "max_tokens": 32_768,
        "top_p": 0.95,
    },

    # https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
    "deepseek/DeepSeek-R1-0528": {
        "temperature": 0.6,
        "max_tokens": 32_768,
    }
}


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = OpenAI(
            api_key="token-abc123",
            base_url=base_url,
            timeout=86400,
            max_retries=3,
        )

    def generate_msg(
            self,
            model: str,
            messages: list[ChatCompletionMessageParam],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None
    ) -> ChatCompletion:
        default_param = DEFAULT_PARAMS.get(model, {
            "temperature": 0.6,
            "max_tokens": 32_768,
        })
        param: Param = {
            "temperature": temperature or default_param["temperature"],
            "max_tokens": max_tokens or default_param["max_tokens"],
            "top_p": default_param.get("top_p", 0.95)
        }

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=param["temperature"],
            max_completion_tokens=param["max_tokens"],
            top_p=param.get("top_p"),
            stream=False,
            extra_body={
                "enable_thinking": True,
            },
        )

        return completion
