# run_solver.py

import subprocess
import os
import sys
import time
import argparse
import json
import traceback
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset

def get_existing_ids(filepath):
    """
    JSONLファイルから既存のすべての問題IDをセットとして読み取る。
    """
    if not os.path.exists(filepath):
        return set()
    
    ids = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'id' in data:
                    ids.add(data['id'])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids

def fetch_problems(start_index=1, num_to_fetch=3):
    """
    Hugging Face Hubから指定された範囲の数学問題をストリーミングで取得する
    """
    print(f"📚 Fetching {num_to_fetch} problems from Hugging Face Hub, starting from problem #{start_index}...")
    try:
        dataset = load_dataset("Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free", split='train', streaming=True)
        
        problems = []
        from itertools import islice
        start_position = start_index - 1
        dataset_slice = islice(dataset, start_position, start_position + num_to_fetch)

        for i, example in enumerate(dataset_slice):
            problems.append({
                "id": start_index + i,
                "problem": example['evolved_problem']
            })
        
        print(f"✅ Successfully fetched {len(problems)} problems.")
        if not problems:
            print("Warning: No problems were fetched. Check start_index and dataset size.")
        return problems
    except Exception as e:
        print(f"❌ Failed to fetch problems. An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

def run_single_agent_instance(agent_id, problem_id, problem_file, log_dir):
    log_file = os.path.join(log_dir, f"problem_{problem_id}_agent_{agent_id:02d}.log")
    signal_file = f"SUCCESS_SOLUTION_{problem_id}_{agent_id}.txt"
    cmd = [sys.executable, "solver_agent.py", problem_file, "--log", log_file, "--signal_file", signal_file]
    
    print(f"🚀 Starting agent {agent_id} for problem {problem_id}...")
    try:
        subprocess.run(cmd, timeout=1800, check=False)
        if os.path.exists(signal_file):
            with open(signal_file, 'r', encoding='utf-8') as f:
                solution_text = f.read()
            os.remove(signal_file)
            return (problem_id, agent_id, True, solution_text)
        return (problem_id, agent_id, False, None)
    except subprocess.TimeoutExpired:
        print(f"⌛ Agent {agent_id} for problem {problem_id} timed out.")
        return (problem_id, agent_id, False, None)
    except Exception as e:
        print(f"💥 Agent {agent_id} for problem {problem_id} failed: {e}", file=sys.stderr)
        return (problem_id, agent_id, False, None)

def solve_problem_in_parallel(problem_id, problem_text, num_agents, log_dir):
    problem_file = f"problem_{problem_id}.txt"
    with open(problem_file, "w", encoding="utf-8") as f: f.write(problem_text)
    print("\n" + "="*60)
    print(f"🔬 Solving Problem ID: {problem_id} with {num_agents} parallel agents...")
    solution_text = None
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = {executor.submit(run_single_agent_instance, i, problem_id, problem_file, log_dir): i for i in range(num_agents)}
        for future in as_completed(futures):
            try:
                prob_id, agent_id, success, result_text = future.result()
                if success:
                    solution_text = result_text
                    print(f"\n🎉🎉🎉 Agent {agent_id} FOUND A SOLUTION for Problem ID: {prob_id}! Shutting down other agents.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            except Exception as e:
                 print(f"A worker process for problem {problem_id} failed: {e}", file=sys.stderr)
    os.remove(problem_file)
    return solution_text

def extract_final_answer(solution_text):
    if not solution_text: return ""
    patterns = [r'\\boxed\{(.+?)\}', r'Final Answer:\s*(.*)', r'Verdict:\s*(.*)']
    for pattern in patterns:
        match = re.search(pattern, solution_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def main():
    parser = argparse.ArgumentParser(description='Solve math problems, skipping any problems that already exist in the output file.')
    parser.add_argument('--start_problem', type=int, default=1, help='The starting problem id to fetch.')
    parser.add_argument('--num_problems', type=int, default=3, help='The number of problems to attempt in this run.')
    parser.add_argument('--num_agents', type=int, default=3, help='Number of parallel agents per problem.')
    parser.add_argument('--output_file', type=str, default='results.jsonl', help='Output file for results in JSON Lines (.jsonl) format.')
    args = parser.parse_args()

    start_problem_id = args.start_problem
    print(f"🚀 Starting session. Attempting to solve {args.num_problems} problems, starting from ID {start_problem_id}.")
    
    existing_ids = get_existing_ids(args.output_file)
    if existing_ids:
        print(f"ℹ️ Found {len(existing_ids)} existing problems in '{args.output_file}'. Will skip them if they are in the requested range.")
        
    problems = fetch_problems(start_problem_id, args.num_problems)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.time()
    solved_count = 0
    skipped_count = 0

    for problem in problems:
        if problem['id'] in existing_ids:
            print(f"⏭️ Skipping problem ID {problem['id']} as it already exists in the results file.")
            skipped_count += 1
            continue
        
        solution_text = solve_problem_in_parallel(problem['id'], problem['problem'], args.num_agents, log_dir)
        
        if solution_text:
            final_answer = extract_final_answer(solution_text)
            # ★ 修正点1: 成功した場合のJSON構造を変更
            result_json = {
                "id": problem['id'], 
                "question": problem['problem'], 
                "output": None,  # 常にnull (PythonではNone)
                "answer": final_answer,
                "solution": solution_text
            }
            solved_count += 1
        else:
            print(f"\n❌ No solution found for Problem ID: {problem['id']}.")
            # ★ 修正点2: 失敗した場合のJSON構造も統一
            result_json = {
                "id": problem['id'], 
                "question": problem['problem'], 
                "output": None, # 常にnull (PythonではNone)
                "answer": "NO_SOLUTION_FOUND",
                "solution": "NO_SOLUTION_FOUND"
            }
        
        with open(args.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_json) + '\n')
        print(f"💾 Result for problem ID {problem['id']} saved to {args.output_file}")

    print("\n" + "#"*60)
    print("### FINAL SUMMARY ###")
    print(f"Problems attempted in this run: {len(problems)}")
    print(f"Problems skipped (already solved): {skipped_count}")
    print(f"Problems newly solved: {solved_count}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Results are saved in: {os.path.abspath(args.output_file)}")
    print("#"*60)

if __name__ == "__main__":
    main()