import os
import pandas as pd
import requests
from datasets import Dataset
from tqdm import tqdm
import time
import json
import re

# --- Constants ---
# ★★★ New dataset URL ★★★
JSONL_URL = "https://raw.githubusercontent.com/tana114/vllm-api-structured/main/project/olym/data/TP_TO_maths_en_COMP.jsonl"
# NUM_PROBLEMS = 3 # Number of problems to process

# --- OpenRouter API Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
YOUR_SITE_URL = "http://localhost"
APP_NAME = "Math Problem Evolver"

# --- Hugging Face Upload Settings ---
OUTPUT_DATASET_ID = "Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free"

# ★★★ Updated Prompt Template ★★★
UPWARD_EVOLUTION_PROMPT_TEMPLATE = """
You are an expert in creating complex mathematical problems. Your task is to rewrite the given instruction to make it more challenging. When the given instruction is a "proof" question, you must transform it into a difficult "computational" question that requires a specific answer, as well as into an even more difficult problem.

#Instruction#
{problem}

Follow these steps precisely.
Step 1: Understand the core concept and structure of the "#Instruction#". Identify the key elements such as variables, conditions, participants, actions, or processes that can be manipulated to increase complexity. Also, recognize the theme of the instruction and ensure it remains consistent throughout the evolution. Finally, analyze the "#Instruction#" to determine its type ("proof problem" vs. "computational problem")

Step 2: Formulate a comprehensive plan to increment the complexity of the "#Instruction#" based on the identified elements in Step 1. The plan should involve modifying or expanding at least three components from the list. It is crucial to ensure that all components in the instruction are logically interconnected and that the complexity increase is coherent and justified. The plan should avoid introducing variables or conditions without clear criteria for determining their values or without contributing to the overall complexity. In this step, consider adding more real-world constraints and dependencies between variables to make the problem more challenging. And you can also add more constraints, concretizing, increasing reasoning. 
- If it is a proof problem: Your primary goal is to transform it into a non-trivial computational problem that is HARDER than the original proof. Do not simplify the problem by merely choosing a small or trivial base case. Instead, introduce complex parameters, ask for a specific but difficult-to-calculate value (e.g., an optimal value, a count of a large set, a specific term in a complex sequence), or combine concepts from the proof with another area of mathematics.
- If it is already a computational problem: Your goal is to increase its difficulty by making it more abstract, adding more variables, or introducing new, challenging constraints.

Step 3: Implement the plan step by step to create the "#Rewritten Instruction#". Ensure the rewritten instruction maintains a logical sequence and avoids ambiguity or confusion. If additional variables or conditions are introduced, provide clear and unambiguous methods or criteria for determining their values. The "#Rewritten Instruction#" should not exceed the original "#Instruction#" by more than 30 words to ensure readability and comprehension.

Step 4: Review the "#Rewritten Instruction#" thoroughly to identify any unreasonable elements or inconsistencies. Make sure the "#Rewritten Instruction#" is a more complex version of the "#Instruction#". and that it accurately reflects the intended increase in complexity. Adjust any part of the instruction that may lead to misunderstanding or ambiguity, and provide the "#Finally Rewritten Instruction#" without any supplementary explanation.
Please reply strictly in the following format:

Step 1
#Elements Identified#:
...
Step 2
#Plan#:
...
Step 3
#Rewritten Instruction#:
...
Step 4
#Finally Rewritten Instruction#:
...
"""

def get_problems_from_jsonl(url: str):
    """Downloads and parses a .jsonl file from a URL."""
    print(f"🔄 Downloading dataset from {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        lines = response.text.strip().splitlines()
        print(f"✅ Download complete. Found {len(lines)} problems.")
        
        problems = []
        for line in lines:
            try:
                data = json.loads(line)
                # We will use 'question' as the problem text
                if 'id' in data and 'question' in data:
                    problems.append(data)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping a line due to JSON parsing error: {line}")
        
        print(f"✅ Successfully parsed {len(problems)} problems.")
        return pd.DataFrame(problems)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading the dataset: {e}")
        return None

def evolve_problem_with_openrouter(problem_text: str) -> tuple[str, str]:
    """Calls the OpenRouter API to evolve a problem statement."""
    # This function remains the same as the previous version
    if not OPENROUTER_API_KEY:
        return "failure", "❌ OPENROUTER_API_KEY environment variable not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": APP_NAME,
        "Content-Type": "application/json"
    }
    prompt = UPWARD_EVOLUTION_PROMPT_TEMPLATE.format(problem=problem_text)
    data = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    last_error = ""

    for attempt in range(3):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            json_response = response.json()

            if 'choices' in json_response and len(json_response['choices']) > 0:
                content = json_response['choices'][0]['message']['content']
                return "success", content.strip()
            else:
                last_error = f"❌ API response missing valid content. Response: {json_response}"

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [402, 429]:
                try:
                    error_details = e.response.json().get('error', {}).get('message', '')
                except json.JSONDecodeError:
                    error_details = e.response.text
                final_error_message = f"❌ Possible credit exhaustion or rate limit ({e.response.status_code}): {error_details}"
                return "failure", final_error_message
            else:
                last_error = f"❌ HTTP Error: {e}"
        except Exception as e:
            last_error = f"❌ An unknown error occurred: {e}"
        time.sleep(1)
    return "failure", last_error

def parse_final_instruction(response_text: str) -> str:
    """Extracts the final rewritten instruction from the full API response."""
    match = re.search(r'#Finally Rewritten Instruction#\s*:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'#Finally Rewritten Instruction#\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Extraction Failed"


def main():
    """Main execution function to process all problems with resume capability."""
    # --- Configuration ---
    output_filename = "evolved_math_problems.csv"

    # --- Resume Logic ---
    processed_ids = []
    if os.path.exists(output_filename):
        print(f"📄 Found existing results file: '{output_filename}'.")
        try:
            existing_df = pd.read_csv(output_filename)
            if 'id' in existing_df.columns:
                processed_ids = existing_df['id'].dropna().tolist()
                print(f"✅ Found {len(processed_ids)} previously processed problems. They will be skipped.")
        except pd.errors.EmptyDataError:
            print("⚠️ Existing results file is empty. Starting fresh.")
        except Exception as e:
            print(f"⚠️ Could not read existing file, starting fresh. Error: {e}")

    # --- Data Loading and Filtering ---
    all_problems_df = get_problems_from_jsonl(JSONL_URL)
    
    if all_problems_df is None or all_problems_df.empty:
        print("No problems were loaded from the source. Exiting.")
        return

    # Filter out any problems that have already been processed (for resuming)
    if processed_ids:
        problems_to_process_df = all_problems_df[~all_problems_df['id'].isin(processed_ids)]
    else:
        # If no processed problems, use the full dataframe
        problems_to_process_df = all_problems_df

    if problems_to_process_df.empty:
        print("✅ All problems have already been processed. Nothing to do.")
        return

    # --- Main Processing Loop ---
    batch_results = []
    total_to_process = len(problems_to_process_df)
    
    print(f"\n🚀 Starting upward evolution for all {total_to_process} remaining problems...")
    print(f"💾 Results will be appended to '{output_filename}' in batches of 5.")
    print("⚠️ This will take a very long time and may incur significant API costs.")

    for i, (index, row) in enumerate(problems_to_process_df.iterrows()):
        print(f"\n--- Processing problem {i + 1}/{total_to_process} (Original ID: {row.get('id', 'N/A')}) ---")
        
        original_problem = row['question']
        start_time = time.time()
        status, evolved_response = evolve_problem_with_openrouter(original_problem)
        end_time = time.time()
        processing_time = end_time - start_time
        
        evolved_problem = ""
        if status == 'success':
            evolved_problem = parse_final_instruction(evolved_response)
        
        batch_results.append({
            "id": row.get('id', 'N/A'),
            "original_problem": original_problem,
            "evolved_problem": evolved_problem,
            "evolved_response": evolved_response,
            "status": status,
            "processing_time_seconds": round(processing_time, 2),
            "original_solution": str(row.get('solution', 'N/A'))
        })
        
        if (i + 1) % 5 == 0 or (i + 1) == total_to_process:
            print(f"💾 Saving batch of {len(batch_results)} results to CSV...")
            temp_df = pd.DataFrame(batch_results)
            temp_df.to_csv(
                output_filename,
                mode='a',
                header=not os.path.exists(output_filename) or os.path.getsize(output_filename) == 0,
                index=False,
                encoding='utf-8-sig'
            )
            batch_results.clear()

    print(f"\n✅ All local processing complete. Final results are in '{output_filename}'.")

    # --- Final Upload ---
    try:
        print(f"🚀 Uploading final dataset to Hugging Face Hub: '{OUTPUT_DATASET_ID}'...")
        final_df = pd.read_csv(output_filename)
        hf_dataset = Dataset.from_pandas(final_df)
        hf_dataset.push_to_hub(repo_id=OUTPUT_DATASET_ID, private=True)
        print(f"✅ Successfully uploaded dataset to '{OUTPUT_DATASET_ID}'.")
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face Hub: {e}")

    print("\n--- Processing complete ---")

if __name__ == "__main__":
    main()