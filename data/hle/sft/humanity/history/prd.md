data: ./humanity/history/results/history.json
data keys:
- question
- answer

expected output format:
```json
[{
    "id": "reasonmed_<index>",
    "question": "<question>",
    "output": "<think>...</think><final_answer>",
    "answer": "<final_answer>"
},...]
```

aim: I need to transfer the existing data to the expected output format.
output: need to be generated from the question and answer pair as natural reasoning process with LLM
answer: need to be generized by summarizing the original answer with LLM

action items:
* write a python script using vllm to achieve the aim, and a bash script to run it with slurm
* write a test python script with openrouter