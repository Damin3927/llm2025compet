import json
import os
from vllm import LLM, SamplingParams

# === 設定 ===
INPUT_JSON = "adversarial_benign_v2.jsonl"
basename = os.path.basename(INPUT_JSON).split('.jsonl')[0]
OUTPUT_JSONL = str(basename)+"_qwen3-32B.jsonl"
NUM_THREADS = 4  # 使用できる GPU メモリに応じて調整

# === モデル初期化 ===
llm = LLM(
    model="Qwen/Qwen3-32B",
    trust_remote_code=True,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.80,
    max_model_len=8192,
    max_num_seqs=1
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=32768
)

# === データ読み込みと既出ID取得 ===
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

done_ids = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["id"])
            except:
                continue

print(f"再開処理: すでに {len(done_ids)} 件完了済み")

# === 推論ループ ===
for item in data:
    if item["id"] in done_ids:
        continue

    prompt = f"""Problem: {item['question']}

Answer: {item['answer']}

Task: Produce sentences of step-by-step reasoning that logically lead to the given Answer.

Guideline:
- Begin the very first reasoning sentence with "To".  
- Subsequent sentences should not begin with "To"; instead, write them as straightforward statements.  
- Be precise; no fluff, apologies, or meta commentary.  

<think>"""

    try:
        result = llm.generate(prompt, sampling_params=sampling_params)[0].outputs[0].text
        think_text = result.split("</think>")[0].strip()
        answer = item["answer"].strip()

        item["output"] = f"<think>{think_text}</think>{answer}"

        # 保存（即時ディスク反映）
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        done_ids.add(item["id"])
        print(f"ID {item['id']} 完了")

    except Exception as e:
        print(f"ID {item['id']} 失敗: {e}")
