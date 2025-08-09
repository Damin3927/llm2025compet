python 1_evaluate_cot.py \
    --dataset ./dataset/sample.jsonl \
    --evaluator-models deepseek/deepseek-r1-0528:free,moonshotai/kimi-k2:free,qwen/qwen3-235b-a22b:free,z-ai/glm-4.5-air:free \
    --eval-concurrency 4 \
    --ids 0,1,2,3