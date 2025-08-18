### Product Requirements Document

**Project:** ChemPile Reasoning — QA Pair Extraction

---

#### 1. Objective

Extract question–answer pairs with their full reasoning traces from the **`jablonkagroup/chempile-reasoning`** dataset (config `claude-3.5-distilled-spectral-reasoning-default`, split `train`) and emit them in a JSON format ready for downstream evaluation.

---

#### 2. Input Data

| Field                 | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| `prompt`              | Natural-language question (may include extra context)            |
| `extracted_reasoning` | Model-generated chain-of-thought that ends with the final answer |

---

#### 3. Processing Steps

1. **Question extraction**

   * Use the full text in `prompt` without alterations.

2. **Reasoning wrapping**

   * Surround `extracted_reasoning` with `<think>` … `</think>` tags.

3. **Answer identification**

   * Run the *Qwen-3* model served through **vLLM** on `extracted_reasoning` to locate the definitive answer line.
   * Strip whitespace and enclosing punctuation.

4. **Record assembly**

   * Build an object with:

     ```jsonc
     {
       "id": "chempile_<row_index>",
       "question": "<prompt>",
       "output": "<think><reasoning></think>[final answer]",
       "answer": "[final answer]"
     }
     ```
   * `<row_index>` is the zero-based row number from the dataset.

---

#### 4. Output

* A single JSON array where each element follows the schema above.
* File must be UTF-8 encoded and newline-terminated.

---

#### 5. Acceptance Criteria

| Requirement            | Test                                                                        |
| ---------------------- | --------------------------------------------------------------------------- |
| **Schema compliance**  | Every object contains exactly the four required keys.                       |
| **Answer correctness** | `answer` string matches the answer portion inside `output`.                 |
| **Tag integrity**      | Exactly one `<think>` opening and `</think>` closing tag per record.        |
| **Row coverage**       | One output object per row in the `train` split; no omissions or duplicates. |

---

#### 6. Performance & Limits

* Pipeline must process the entire split in ≤ 4 GB peak RAM.
* End-to-end runtime target on an A10G GPU: ≤ 45 minutes for 20 K rows.

---

#### 7. Deliverables

1. A python script to achive the function
2. A bash script to run the python script in slurm
3. a test file to fast test 3 examples with openrouter
