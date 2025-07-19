import os
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
res = llm.generate("What is the capital of France?")

print("==" * 20)
print(res[0].outputs[0].text)
