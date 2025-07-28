#!/usr/bin/env python3
"""
Standalone script to generate harder math problems with better prompts.
This version is fully decoupled from other files.
"""

import json
import argparse
import asyncio
import aiohttp
import re
import time
from dataclasses import dataclass
from typing import Optional
import sympy as sp

# Load API key
with open("./keys.json", "r") as f:
    keys = json.load(f)
OPENROUTER_API_KEY = keys["openrouter"]
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model configurations
WEAK_MODEL = "qwen/qwen3-8b:free"
STRONG_MODEL = "qwen/qwen3-235b-a22b-2507:free"
GENERATOR_MODEL = "openai/gpt-4o"

@dataclass
class MathProblem:
    """Simple data class for math problems."""
    question: str
    answer: str
    topic: str
    difficulty: str = "olympiad"
    solution_steps: Optional[str] = None

# Better prompts for harder problems
HARD_PROBLEM_PROMPTS = {
    "algebra": """Generate an advanced algebra problem for math olympiad level.

Requirements:
1. The problem should involve polynomials, functional equations, or inequalities
2. The answer should be a specific numerical value (not a simple integer)
3. The solution should require at least 3-4 non-trivial steps
4. Avoid problems that reduce to simple factoring or quadratic formula
5. The answer should be in \\boxed{} format
6. Make the answer a fraction, irrational number, or large integer (>1000)

Example types:
- Find the sum of all real roots of a complex polynomial equation
- Functional equations with specific value computations
- Systems of equations with parameters

Output format:
QUESTION: [The problem statement]
SOLUTION: [Detailed step-by-step solution]
ANSWER: \\boxed{[numerical answer]}""",

    "number_theory": """Generate a competition-level number theory problem.

Requirements:
1. Focus on less common topics: quadratic residues, Pell equations, or advanced divisibility
2. The answer should be a large integer (>100) or involve modular arithmetic
3. Avoid simple "count the divisors" or "find GCD" problems
4. The solution should require clever insights, not just computation
5. Format the answer in \\boxed{}

Example types:
- Find the last three digits of a large power
- Count solutions to Diophantine equations with constraints
- Problems involving digit sums and divisibility conditions

Output format:
QUESTION: [The problem statement]
SOLUTION: [Detailed step-by-step solution]
ANSWER: \\boxed{[numerical answer]}""",

    "combinatorics": """Generate an olympiad-level combinatorics problem.

Requirements:
1. The problem should involve advanced counting with constraints
2. Avoid simple permutation/combination formulas
3. The answer should be a specific number (not a formula)
4. Include non-standard constraints or recursive structures
5. The answer should typically be a large number (>1000)
6. Format answer in \\boxed{}

Example types:
- Count colorings with adjacency constraints
- Recursive sequences with combinatorial interpretation
- Paths on graphs with special properties

Output format:
QUESTION: [The problem statement]
SOLUTION: [Detailed step-by-step solution]
ANSWER: \\boxed{[numerical answer]}"""
}

async def make_api_call(prompt: str, model: str = GENERATOR_MODEL) -> Optional[str]:
    """Make an API call to generate or evaluate content."""
    start_time = time.time()
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7 if model == GENERATOR_MODEL else 0.2,
        "max_tokens": 1500 if model == GENERATOR_MODEL else 2000
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as response:
                elapsed = time.time() - start_time
                if response.status == 200:
                    data = await response.json()
                    print(f"[API] {model}: {elapsed:.2f}s", end=" ", flush=True)
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    print(f"\n[API ERROR] Status {response.status} for model {model} after {elapsed:.2f}s: {error_text[:200]}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"API call error after {elapsed:.2f}s: {e}")
    
    return None

def parse_problem(content: str, topic: str) -> Optional[MathProblem]:
    """Parse generated content into a MathProblem."""
    try:
        # Extract components
        question_match = re.search(r'QUESTION:\s*(.*?)(?=SOLUTION:|$)', content, re.DOTALL)
        solution_match = re.search(r'SOLUTION:\s*(.*?)(?=ANSWER:|$)', content, re.DOTALL)
        answer_match = re.search(r'\\boxed\{([^}]+)\}', content)
        
        if not answer_match:
            # Try alternative formats
            answer_match = re.search(r'Answer:\s*([^\n]+)', content)
            if not answer_match:
                answer_match = re.search(r'=\s*([^\n]+)$', content, re.MULTILINE)
        
        if question_match and answer_match:
            return MathProblem(
                question=question_match.group(1).strip(),
                answer=answer_match.group(1).strip(),
                topic=topic,
                difficulty="olympiad",
                solution_steps=solution_match.group(1).strip() if solution_match else None
            )
    except Exception as e:
        print(f"Parse error: {e}")
    
    return None

async def generate_harder_problem(topic: str) -> Optional[MathProblem]:
    """Generate a harder problem for the given topic."""
    start_time = time.time()
    prompt = HARD_PROBLEM_PROMPTS.get(topic, HARD_PROBLEM_PROMPTS["algebra"])
    content = await make_api_call(prompt)
    
    if content:
        result = parse_problem(content, topic)
        elapsed = time.time() - start_time
        print(f" | Total: {elapsed:.2f}s")
        return result
    else:
        elapsed = time.time() - start_time
        print(f" | Failed after {elapsed:.2f}s")
    return None

async def evaluate_problem_once(problem: MathProblem, model: str, debug: bool = False) -> tuple[bool, Optional[str]]:
    """Evaluate a problem once with the given model. Returns (success, response)."""
    prompt = f"{problem.question}\n\nPlease solve this step-by-step and put your final answer in \\boxed{{}}."
    response = await make_api_call(prompt, model)
    
    if response:
        success = check_answer(response, problem.answer, debug=debug)
        return success, response
    return False, None

def check_answer(response: str, correct_answer: str, debug: bool = False) -> bool:
    """Check if the model's answer matches the correct answer."""
    # Extract answer from response
    answer_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if not answer_match:
        if debug:
            print(f"[DEBUG] No \\boxed{{}} found in response")
        return False
    
    model_answer = answer_match.group(1).strip()
    
    # Try exact match first
    if model_answer == correct_answer:
        return True
    
    # Try numerical comparison
    try:
        # Use sympy for symbolic comparison
        model_expr = sp.sympify(model_answer)
        correct_expr = sp.sympify(correct_answer)
        
        if debug:
            print(f"[DEBUG] Model expr: {model_expr}, Correct expr: {correct_expr}")
        
        # Check if expressions are equal
        if sp.simplify(model_expr - correct_expr) == 0:
            return True
        
        # For numerical values, check if they're close
        if model_expr.is_number and correct_expr.is_number:
            diff = abs(float(model_expr) - float(correct_expr))
            if debug:
                print(f"[DEBUG] Numerical diff: {diff}")
            return diff < 1e-6
    except Exception as e:
        if debug:
            print(f"[DEBUG] SymPy error: {e}")
    
    return False

async def evaluate_problem_full(problem: MathProblem, num_attempts: int = 16):
    """Fully evaluate a problem with both models."""
    eval_start_time = time.time()
    print(f"\nEvaluating: {problem.question[:100]}...")
    print(f"Answer: \\boxed{{{problem.answer}}}")
    
    # Evaluate with weak model
    weak_start_time = time.time()
    weak_successes = 0
    weak_sample_answers = []
    for i in range(num_attempts):
        success, response = await evaluate_problem_once(problem, WEAK_MODEL)
        if success:
            weak_successes += 1
        # Collect first 3 responses for analysis
        if i < 3 and response:
            answer_match = re.search(r'\\boxed\{([^}]+)\}', response)
            weak_sample_answers.append(answer_match.group(1).strip() if answer_match else "NO ANSWER")
        print(f"\rWeak model progress: {i+1}/{num_attempts}", end="")
    
    weak_elapsed = time.time() - weak_start_time
    weak_rate = weak_successes / num_attempts
    print(f"\nWeak model (7B): {weak_successes}/{num_attempts} = {weak_rate:.3f} | Time: {weak_elapsed:.1f}s")
    print(f"Sample weak answers: {weak_sample_answers}")
    
    # Evaluate with strong model
    strong_start_time = time.time()
    strong_successes = 0
    strong_sample_answers = []
    for i in range(num_attempts):
        # Enable debug for first attempt
        success, response = await evaluate_problem_once(problem, STRONG_MODEL, debug=(i == 0))
        if success:
            strong_successes += 1
        # Collect first 3 responses for analysis
        if i < 3 and response:
            answer_match = re.search(r'\\boxed\{([^}]+)\}', response)
            strong_sample_answers.append(answer_match.group(1).strip() if answer_match else "NO ANSWER")
        print(f"\rStrong model progress: {i+1}/{num_attempts}", end="")
    
    strong_elapsed = time.time() - strong_start_time
    strong_rate = strong_successes / num_attempts
    print(f"\nStrong model (32B): {strong_successes}/{num_attempts} = {strong_rate:.3f} | Time: {strong_elapsed:.1f}s")
    print(f"Sample strong answers: {strong_sample_answers}")
    
    eval_total_time = time.time() - eval_start_time
    print(f"Total evaluation time: {eval_total_time:.1f}s")
    
    # Analyze filtering criteria
    print("\nFiltering Analysis:")
    keep = True
    reasons = []
    
    if weak_rate >= 15/16:
        keep = False
        reasons.append(f"TOO EASY: Weak model solved {weak_rate:.3f} >= 0.938")
    
    if strong_rate == 0:
        keep = False
        reasons.append("TOO HARD/NOISY: Strong model never solved it")
    
    if weak_rate > strong_rate:
        keep = False
        reasons.append(f"ANOMALOUS: Weak ({weak_rate:.3f}) > Strong ({strong_rate:.3f})")
    
    difficulty_gap = strong_rate - weak_rate
    if difficulty_gap <= 6/16 and strong_rate >= 0.75:
        keep = False
        reasons.append(f"SMALL GAP: Gap ({difficulty_gap:.3f}) <= 0.375 and Strong >= 0.75")
    
    if keep:
        print("✅ KEPT: Problem meets all criteria")
    else:
        print("❌ FILTERED OUT:")
        for reason in reasons:
            print(f"   - {reason}")
    
    return {
        "problem": problem,
        "weak_rate": weak_rate,
        "strong_rate": strong_rate,
        "kept": keep,
        "reasons": reasons,
        "weak_sample_answers": weak_sample_answers,
        "strong_sample_answers": strong_sample_answers
    }

async def generate_harder_problems(num_problems: int = 3, n_attempts: int = 16, topics: list[str] = ["algebra", "number_theory", "combinatorics"]):
    """
    Generate harder problems with specialized prompts.
    Args:
        num_problems: Number of problems to generate.
        n_attempts: Number of attempts to evaluate the problem for each model.
        topics: List of topics to generate problems for.
    """
    script_start_time = time.time()
    print("=== Standalone Harder Problem Generator ===\n")
    
    # Generate problems
    generation_start_time = time.time()
    print(f"1. Generating {num_problems} harder problems with specialized prompts...\n")
    
    problems = []
    
    for topic in topics:
        print(f"Generating {topic} problem...", end=" ")
        problem = await generate_harder_problem(topic)
        
        if problem:
            problems.append(problem)
            print(f"✓ Generated successfully")
            print(f"  Q: {problem.question[:100]}...")
            print(f"  A: \\boxed{{{problem.answer}}}")
        else:
            print(f"✗ Failed to generate")
    
    generation_elapsed = time.time() - generation_start_time
    print(f"\n⏱️  Generation phase completed in {generation_elapsed:.1f}s")
    
    # Save initial problems
    if problems:
        initial_data = []
        for i, p in enumerate(problems, 1):
            initial_data.append({
                "id": i,
                "topic": p.topic,
                "difficulty": p.difficulty,
                "question": p.question,
                "answer": p.answer,
                "solution": p.solution_steps
            })
        
        with open("harder_problems_initial.json", "w") as f:
            json.dump(initial_data, f, indent=2)
        
        print(f"\n✓ Saved {len(problems)} problems to harder_problems_initial.json")
        
        # Evaluate problems
        evaluation_start_time = time.time()
        print("\n" + "="*60)
        print("2. EVALUATING PROBLEMS (this will take several minutes)")
        print("="*60)
        
        results = []
        for i, problem in enumerate(problems, 1):
            print(f"\n[Problem {i}/{len(problems)}]")
            result = await evaluate_problem_full(problem, n_attempts)
            results.append(result)
        
        evaluation_elapsed = time.time() - evaluation_start_time
        
        # Summary
        kept_count = sum(1 for r in results if r["kept"])
        print(f"\n{'='*60}")
        print(f"SUMMARY: {kept_count}/{len(problems)} problems passed filtering")
        print(f"Acceptance rate: {kept_count/len(problems)*100:.1f}%")
        print(f"⏱️  Evaluation phase completed in {evaluation_elapsed:.1f}s")
        print(f"⏱️  Average time per problem: {evaluation_elapsed/len(problems):.1f}s")
        
        # Save evaluation results
        eval_data = {
            "total_problems": len(problems),
            "kept_problems": kept_count,
            "acceptance_rate": kept_count/len(problems) if problems else 0,
            "details": [
                {
                    "question": r["problem"].question,
                    "answer": r["problem"].answer,
                    "weak_rate": r["weak_rate"],
                    "strong_rate": r["strong_rate"],
                    "kept": r["kept"],
                    "reasons": r["reasons"],
                    "weak_sample_answers": r["weak_sample_answers"],
                    "strong_sample_answers": r["strong_sample_answers"]
                }
                for r in results
            ]
        }
        
        with open("harder_problems_evaluation.json", "w") as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"\n✓ Saved evaluation results to harder_problems_evaluation.json")
    
    # Show example problems that should work
    print("\n" + "="*60)
    print("3. EXAMPLE PROBLEMS THAT SHOULD WORK BETTER:")
    print("="*60)
    
    examples = [
        {
            "topic": "algebra",
            "question": "Let P(x) = x^4 - 4x^3 + 6x^2 - 4x + k. Find the value of k such that P(x) has exactly two distinct real roots.",
            "answer": "\\frac{5}{4}",
            "why": "Requires analysis of polynomial behavior, not simple factoring"
        },
        {
            "topic": "number_theory", 
            "question": "Find the remainder when 7^{7^7} is divided by 1000.",
            "answer": "343",
            "why": "Uses Euler's theorem and modular arithmetic, not brute force"
        },
        {
            "topic": "combinatorics",
            "question": "In how many ways can we place non-attacking rooks on a 6×6 chessboard with the constraint that no rook is on the main diagonal?",
            "answer": "265",
            "why": "Derangement problem, not simple permutation"
        }
    ]
    
    for ex in examples:
        print(f"\n{ex['topic'].upper()}:")
        print(f"Q: {ex['question']}")
        print(f"A: {ex['answer']}")
        print(f"Why it works: {ex['why']}")
    
    # Final timing summary
    script_total_time = time.time() - script_start_time
    print(f"\n{'='*60}")
    print(f"⏱️  TOTAL SCRIPT TIME: {script_total_time:.1f}s ({script_total_time/60:.1f} minutes)")
    if problems:
        print(f"⏱️  Generation: {generation_elapsed:.1f}s ({generation_elapsed/script_total_time*100:.1f}%)")
        if 'evaluation_elapsed' in locals():
            print(f"⏱️  Evaluation: {evaluation_elapsed:.1f}s ({evaluation_elapsed/script_total_time*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate harder math problems")
    parser.add_argument("--num_problems", type=int, default=3, help="Number of problems to generate")
    parser.add_argument("--n_attempts", type=int, default=16, help="Number of attempts to evaluate the problem for each model")
    parser.add_argument("--topics", type=str, nargs="+", default=["algebra", "number_theory", "combinatorics"], help="Topics to generate problems for")
    args = parser.parse_args()
    print(args)

    asyncio.run(generate_harder_problems(args.num_problems, args.n_attempts, args.topics))