## PRD: Convert `medcalc_raw_selected` (train split) to SFT JSON Format

### Objective

Transform the `train` split of the `medcalc_raw_selected` config from the private dataset repo `neko-llm/CoT_Medicine` into a structured JSON format suitable for SFT-style training.

### Input

Source:

* Dataset: `neko-llm/CoT_Medicine`
* Config: `medcalc_raw_selected`
* Split: `train`

Columns in this dataset:

* `Row Number`
* `Calculator ID`
* `Calculator Name`
* `Category` (e.g., `"physical"`)
* `Output Type` (e.g., `"decimal"`)
* `Note ID` (e.g., `"usmle-4974"`)
* `Note Type` (e.g., `"Extracted"`)
* `Patient Note`
* `Question`
* `Relevant Entities`
* `Ground Truth Answer`
* `Lower Limit`
* `Upper Limit`
* `Ground Truth Explanation`

### Output

A JSON file consisting of a list of dict entries, each with the following format:

```json
{
    "id": "medcalc_1",
    "question": "<Patient Note>\n<Question>\n<Relevant Entities>",
    "output": "<think><converted Ground Truth Explanation></think><final_answer>",
    "answer": "<Ground Truth Answer> (<Lower Limit> ~ <Upper Limit>)"
}
```

### Field Mapping & Transformation Logic

| Target Field | Source Field(s)                                     | Description                                                                                                                                                                           |
| ------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`         | implicit                                            | Format as `medcalc_{idx}`, where `idx` is the row index starting from 1                                                                                                               |
| `question`   | `Patient Note`, `Question`, `Relevant Entities`     | Concatenate with newline separators (`\n`). If `Relevant Entities` is empty, omit it                                                                                                  |
| `output`     | `Ground Truth Explanation` + `Ground Truth Answer`  | Transform explanation into chain-of-thought format by wrapping with `<think>...</think>`, then append the final answer (same as `Ground Truth Answer`) directly after the closing tag. PLEASE USE vllm + Qwen3-32b to do the transfer to ensure the format is suitable for training reasoning models |
| `answer`     | `Ground Truth Answer`, `Lower Limit`, `Upper Limit` | Format as: `<answer> (<lower> ~ <upper>)`. Use string `"NA"` if lower/upper bound is missing                                                                                          |

**Note**: 
- If any of the required fields are missing or null, skip the row or raise a warning during processing.
- for transfering the output column, please leave the place to use open source llm with slurm.

### Output Path

Write the resulting JSON to:

```
~/explore/data/hle/sft/medical/results/medcalc_cot.json
```

### Deliverables

* A Python script that performs this transformation.
* a bash script to run the code with slurm, and use gpu
* The output JSON file, ready for downstream SFT ingestion.
