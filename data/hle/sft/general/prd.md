### PRD — **Natural-Language Reasoning Generator for StrategyQA**

---

#### 1. Goal

Transform each record in the **zen-E/StrategyQA\_CoT\_GPT4o** dataset into a JSON object whose **output** field contains a clear, readable reasoning chain plus the final answer.

---

#### 2. Input

| Column     | Description                              |
| ---------- | ---------------------------------------- |
| `question` | StrategyQA question (string)             |
| `answer`   | Gold label (`true` / `false`)            |
| `cot`      | Chain-of-thought text produced by GPT-4o |

---

#### 3. Output Schema

```json
{
  "id": "StrategyQA_<index>",
  "question": "<original question>",
  "output": "<think>…human-like reasoning…</think>\n\n<answer>",
  "answer": "<true|false>"
}
```

* **`id`** – prefix `StrategyQA_` followed by the zero-based row index.
* **`question`** and **`answer`** – copied verbatim from the source row.
* **`output`** – two parts:

  1. `<think>…</think>` block holding the rewritten reasoning (see §4).
  2. A blank line, then the single token `true` or `false` that matches **`answer`**.

---

#### 4. Reasoning-Rewrite Rules

| Requirement       | Notes                                                                  |
| ----------------- | ---------------------------------------------------------------------- |
| **Faithful**      | Do not add facts or alter the logical path.                            |
| **Readable**      | Write in complete sentences; avoid jargon and “Step 1/2…” scaffolding. |
| **Concise**       | 1–3 short paragraphs; aim for ≤ 120 words.                             |
| **Neutral tone**  | No personal opinions or filler phrases.                                |
| **Tag integrity** | Keep the `<think>` wrapper exactly; no extra markup inside or outside. |

---

#### 5. Processing Steps for Each Row

1. **Copy columns**: `question`, `answer`.
2. **Rewrite** `cot` following §4 with qwen3.
3. **Assemble** the JSON object using the schema in §3.
4. **Repeat** until all rows are converted.
5. **Emit** a single top-level JSON array containing every object in order.

---

#### 6. Example

Source row

```json
{
  "question": "Were the 1900 Olympics held in Europe?",
  "answer": "true",
  "cot": "The 1900 Olympics took place in Paris, France, which is in Europe. Therefore the answer is true."
}
```

Converted output

```json
{
  "id": "StrategyQA_0",
  "question": "Were the 1900 Olympics held in Europe?",
  "output": "<think>The 1900 Olympic Games were hosted in Paris, France—an unquestionably European city. That places the event within Europe.</think>\n\ntrue",
  "answer": "true"
}
```

---

#### 7. Delivery

1. Make a python script to transformed data as a `.json` file using LLM with vllm, and a bash file to call the python file in slurm.
2. Make a pythopn script to test the data creation using LLM with openrouter.