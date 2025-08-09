## PRD: Convert `reasonmed_raw_selected` (train split) to SFT JSON Format

### Objective

Transform the `train` split of the `reasonmed_raw_selected` config from the private dataset `neko-llm/CoT_Medicine` into a JSON format suitable for SFT-style training, including chain-of-thought (CoT) reasoning markup.

---

### Input Specification

**Source:**

* Dataset: `neko-llm/CoT_Medicine`
* Config: `reasonmed_raw_selected`
* Split: `train`

**Columns:**

* `instruction`: a natural language prompt that typically includes both question and explanation.
* `output`: model-generated answer text (to be discarded or replaced).

---

### Target Format

Each row will be converted into a dictionary with the following structure:

```json
{
    "id": "reasonmed_<index>",
    "question": "<instruction>",
    "output": "<think>...</think><final_answer>",
    "answer": "<final_answer>"
}
```

---

### Transformation Rules

| Target Field | Source Field  | Transformation                                                                                                       |
| ------------ | ------------- | -------------------------------------------------------------------------------------------------------------------- |
| `id`         | row index     | Format as `reasonmed_<index>`, starting from 0                                                                       |
| `question`   | `instruction` | Copy directly                                                                                                        |
| `output`     | `instruction` | Convert to CoT format by extracting explanation, wrapping it in `<think>...</think>`, and appending the final answer |
| `answer`     | `instruction` | Extract only the final answer (no reasoning)                                                                         |

**Requirements:**

* If any row has null or missing values in `instruction`, skip it or log a warning.
* The `output` field generation (i.e., CoT formatting) will be performed using an open-source LLM and run via a Slurm-based GPU job.
* The existing `output` column in the dataset is not used.

---

### Output Path

Write the final JSON file to:

```
~/explore/data/hle/sft/medical/results/reasonmed_cot.json
```

---

### Deliverables

1. **Python script** to:

   * Load the `reasonmed_raw_selected` split
   * Extract and format the `instruction` column into `question`, `output`, and `answer` fields
   * Save the resulting data as a JSON list to the specified path

2. **Slurm-compatible bash script** to:

   * Launch the transformation process on a GPU node
   * Use an open-source LLM (e.g., Mistral or Qwen) to generate the `<think>...</think>` content for `output`

3. **Final JSON file** ready for SFT ingestion