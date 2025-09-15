"""
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang
This file has been modified for the user's specific workflow.
Final Version as of 2025-08-06.
"""

import os
import sys
import json
import requests
import argparse
import re
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = "http://localhost:3000"
APP_NAME = "IMO Solver Test Final"

_log_file = None
original_print = print
def log_print(*args, **kwargs):
    original_print(*args, **kwargs)
    if _log_file is not None:
        message = ' '.join(str(arg) for arg in args)
        _log_file.write(message + '\n')
        _log_file.flush()
print = log_print
def set_log_file(log_file_path):
    global _log_file
    if log_file_path:
        try: _log_file = open(log_file_path, 'w', encoding='utf-8'); return True
        except Exception as e: print(f"Error opening log file {log_file_path}: {e}"); return False
    return True
def close_log_file():
    global _log_file
    if _log_file is not None: _log_file.close(); _log_file = None

step1_prompt = """
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

*   **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
*   **Location:** "By interchanging the limit and the integral, we get..."
    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.

"""


verification_remider = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

def get_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: print("Error: OPENROUTER_API_KEY not set."); sys.exit(1)
    return api_key

def read_file_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: print(f"Error reading file '{filepath}': {e}"); sys.exit(1)

def build_request_payload(system_prompt, question_prompt, history=None):
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    if history: messages.extend(history)
    messages.append({"role": "user", "content": question_prompt})
    return {"model": MODEL_NAME, "messages": messages, "temperature": 0.1}

def send_api_request(api_key, payload):
    headers = {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER, "X-Title": APP_NAME,
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}", file=sys.stderr)
        print(f"Raw API Response: {e.response.text if e.response else 'No response'}", file=sys.stderr)
        return None

def extract_text_from_response(response_data):
    if response_data is None: return None
    try: return response_data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error: Could not extract text from API response: {e}", file=sys.stderr)
        print(f"Full API Response: {json.dumps(response_data, indent=2)}", file=sys.stderr)
        return None

def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    if not solution: return ''
    idx = solution.find(marker)
    if idx == -1: return ''
    return solution[idx + len(marker):].strip() if after else solution[:idx].strip()

def verify_solution(problem_statement, solution, history):
    dsol = extract_detailed_solution(solution)
    verification_prompt = f"### Problem ###\n\n{problem_statement}\n\n### Solution ###\n\n{dsol}\n\n{verification_remider}"
    payload = build_request_payload(verification_system_prompt, verification_prompt)
    res = send_api_request(get_api_key(), payload)
    verification_output = extract_text_from_response(res)
    if verification_output is None: return "API_ERROR", "no"
    
    check_prompt = f"Response in 'yes' or 'no'. Is this statement saying the solution is correct, or does not contain a critical error or major justification gap?\n\n---\n\n{verification_output}"
    payload_check = build_request_payload("", check_prompt)
    res_check = send_api_request(get_api_key(), payload_check)
    is_good_text = extract_text_from_response(res_check)
    if is_good_text is None: return "API_ERROR", "no"

    bug_report = ""
    if "yes" not in is_good_text.lower():
        bug_report = extract_detailed_solution(verification_output, "Detailed Verification", False)
    return bug_report, is_good_text

def check_if_solution_claimed_complete(solution):
    if not solution: return False
    check_prompt = f'Is this text claiming the solution is complete? Response in "yes" or "no".\n\n---\n{solution}'
    payload = build_request_payload("", check_prompt)
    res = send_api_request(get_api_key(), payload)
    response_text = extract_text_from_response(res)
    if response_text is None: return False
    return "yes" in response_text.lower()

def run_agent_process(problem_statement):
    max_initial_retries = 3
    solution = None
    history = []

    for attempt in range(max_initial_retries):
        print(f"\n>>>>>>> Initial solution attempt {attempt + 1}/{max_initial_retries}...")
        history = []
        
        print(">>>>>>> Step 1: Initial Solution Generation")
        payload1 = build_request_payload(step1_prompt, problem_statement)
        res1 = send_api_request(get_api_key(), payload1)
        output1 = extract_text_from_response(res1)
        if output1 is None:
            print(f">>>>>>> API error during initial generation on attempt {attempt + 1}. Retrying...")
            continue
        history.extend([{"role": "user", "content": problem_statement}, {"role": "assistant", "content": output1}])
        
        print(">>>>>>> Step 2: Self Improvement")
        payload2 = build_request_payload(step1_prompt, self_improvement_prompt, history=history)
        res2 = send_api_request(get_api_key(), payload2)
        current_solution = extract_text_from_response(res2)
        if current_solution is None:
            print(f">>>>>>> API error during self-improvement on attempt {attempt + 1}. Retrying...")
            continue
        
        if check_if_solution_claimed_complete(current_solution):
            print(">>>>>>> Solution is claimed to be complete. Proceeding to verification.")
            solution = current_solution
            history.extend([{"role": "user", "content": self_improvement_prompt}, {"role": "assistant", "content": solution}])
            break
        else:
            print(f">>>>>>> Solution not claimed complete on attempt {attempt + 1}. Retrying...")

    if solution is None:
        print(">>>>>>> Failed to generate a complete initial solution after several retries.")
        return None

    error_count, correct_count = 0, 0
    for i in range(10):
        print(f"\n--- Verification Iteration {i+1}, Consecutive Corrects: {correct_count}, Consecutive Errors: {error_count} ---")
        bug_report, good_verify = verify_solution(problem_statement, solution, history)

        if "API_ERROR" in bug_report:
            print(">>>>>>> API error during verification. Terminating this agent run.")
            return None

        if "yes" in good_verify.lower():
            correct_count += 1
            error_count = 0
            if correct_count >= 3:
                print("\n🎉🎉🎉 Found a correct solution after multiple verifications.")
                return solution
        else:
            correct_count = 0
            error_count += 1
            if error_count >= 5:
                print("\n❌❌❌ Failed after multiple errors.")
                return None
            
            print(">>>>>>> Verification failed. Correcting...")
            correction_prompt_full = f"{correction_prompt}\n\n### Bug Report\n\n{bug_report}"
            history.append({"role": "user", "content": correction_prompt_full})
            payload_correct = build_request_payload(step1_prompt, "", history=history)
            res_correct = send_api_request(get_api_key(), payload_correct)
            solution = extract_text_from_response(res_correct)
            if solution is None:
                print(">>>>>>> API error during correction. Terminating this agent run.")
                return None
            history.append({"role": "assistant", "content": solution})
    
    print("\n❌❌❌ Reached max verification iteration limit.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent (OpenRouter Version)')
    parser.add_argument('problem_file', help='Path to the problem statement file')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--signal_file', type=str, required=True, help='File to write the successful solution to')
    args = parser.parse_args()

    if args.log and not set_log_file(args.log): sys.exit(1)
    
    problem_statement = read_file_content(args.problem_file)
    
    try:
        sol = run_agent_process(problem_statement)
        if sol is not None:
            print(f"\n✅✅✅ Found a correct solution in this run.")
            with open(args.signal_file, "w", encoding='utf-8') as f:
                f.write(sol)
    except Exception as e:
        print(f">>>>>>> Critical error in agent run: {e}", file=sys.stderr)
        
    close_log_file()