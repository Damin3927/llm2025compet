# Generate Harder Problems Script

A standalone script for generating high-quality math problems following the Reasoning360 paper methodology. This script creates challenging math problems that pass strict difficulty filtering criteria.

## Overview

This script implements the Reasoning360 dataset construction methodology to generate math problems that are:
- Challenging but solvable
- Well-calibrated in difficulty
- Suitable for training reasoning models

## Features

- **Specialized Problem Generation**: Custom prompts for algebra, number theory, and combinatorics
- **Dual-Model Evaluation**: Tests problems with both weak (8B) and strong (235B) models
- **Automatic Filtering**: Removes problems that are too easy, too hard, or anomalous
- **Detailed Analytics**: Provides sample answers and rejection reasons
- **Standalone Operation**: No dependencies on other project files

## Requirements

- Python 3.7+
- Dependencies:
  ```bash
  pip install aiohttp sympy
  ```
- OpenRouter API key in `../keys.json`:
  ```json
  {
    "openrouter": "your-api-key-here"
  }
  ```

## Usage

Basic usage:
```bash
python generate_harder_problems.py
```

With custom parameters:
```bash
# Generate 5 problems with 8 evaluation attempts each
python generate_harder_problems.py --num_problems 5 --n_attempts 8

# Generate only algebra and number theory problems
python generate_harder_problems.py --topics algebra number_theory

# Full example
python generate_harder_problems.py --num_problems 10 --n_attempts 16 --topics algebra combinatorics
```

### Command Line Arguments

- `--num_problems`: Number of problems to generate (default: 3)
- `--n_attempts`: Number of evaluation attempts per model (default: 16)
- `--topics`: List of topics to generate (default: algebra number_theory combinatorics)

## How It Works

### 1. Problem Generation
The script uses GPT-4o with specialized prompts for each topic:
- **Algebra**: Polynomials, functional equations, inequalities
- **Number Theory**: Modular arithmetic, Diophantine equations, advanced divisibility
- **Combinatorics**: Complex counting problems with constraints

### 2. Evaluation Process
Each problem is evaluated multiple times with two models:
- **Weak Model** (qwen3-8b): Baseline performance
- **Strong Model** (qwen3-235b): Advanced performance

### 3. Filtering Criteria
Problems are filtered based on four criteria:
1. **Too Easy**: Weak model solves ≥15/16 times (≥93.8%)
2. **Too Hard/Noisy**: Strong model never solves it (0/16)
3. **Anomalous**: Weak model outperforms strong model
4. **Small Gap**: Difficulty gap ≤6/16 AND strong model ≥75%

### 4. Debug Features
The script includes debugging output showing:
- Sample answers from both models
- API error messages
- Answer extraction and comparison details

## Output Files

1. **harder_problems_initial.json**: All generated problems
   ```json
   {
     "id": 1,
     "topic": "algebra",
     "question": "...",
     "answer": "\\frac{5}{4}",
     "solution": "..."
   }
   ```

2. **harder_problems_evaluation.json**: Detailed evaluation results
   ```json
   {
     "total_problems": 3,
     "kept_problems": 1,
     "acceptance_rate": 0.333,
     "details": [
       {
         "question": "...",
         "weak_rate": 0.125,
         "strong_rate": 0.625,
         "kept": true,
         "reasons": [],
         "weak_sample_answers": ["4.25", "17/4", "NO ANSWER"],
         "strong_sample_answers": ["17/4", "17/4", "4.25"]
       }
     ]
   }
   ```

## Model Configuration

Current models (using free tiers):
- **Weak Model**: `qwen/qwen3-8b:free`
- **Strong Model**: `qwen/qwen3-235b-a22b-2507:free`
- **Generator Model**: `openai/gpt-4o`

Temperature settings:
- Generation: 0.7 (balanced creativity)
- Evaluation: 0.2 (consistent reasoning)

## Troubleshooting

### Low Acceptance Rate
If most problems are filtered out:
1. Check that models are properly configured
2. Verify API key has access to all models
3. Consider adjusting temperature settings
4. Review generated problems for common patterns

### Anomalous Behavior
If weak model frequently outperforms strong model:
1. Check model names are correct
2. Verify answer formats match exactly
3. Review sample answers for parsing issues
4. Consider lowering evaluation temperature

### API Errors
Common issues:
- Rate limiting: Add delays between requests
- Invalid model names: Check OpenRouter documentation
- Quota exceeded: Check API usage limits

## Example Output

```
=== Standalone Harder Problem Generator ===

1. Generating 3 harder problems with specialized prompts...

Generating algebra problem...
✓ Generated successfully
  Q: Let f(x) = x^4 - 4x^3 + 6x^2 - 4x + k. Find k such that f(x) has exactly two distinct real roots...
  A: \boxed{\frac{5}{4}}

============================================================
2. EVALUATING PROBLEMS (this will take several minutes)
============================================================

Evaluating: Let f(x) = x^4 - 4x^3 + 6x^2 - 4x + k...
Answer: \boxed{\frac{5}{4}}
[DEBUG] Strong model response exists: True
[DEBUG] Response length: 1842
[DEBUG] Found \boxed: True
[DEBUG] Extracted answer: \frac{5}{4}
[DEBUG] Expected answer: \frac{5}{4}

Weak model (8B): 2/16 = 0.125
Sample weak answers: ['5/4', 'NO ANSWER', '1.25']

Strong model (235B): 14/16 = 0.875
Sample strong answers: ['\frac{5}{4}', '\frac{5}{4}', '\frac{5}{4}']

Filtering Analysis:
✅ KEPT: Problem meets all criteria
```

## Related Files

- `math_qa_generator_demo.py`: Original implementation
- `debug_math_qa_generator.py`: Debug version with detailed analysis
- `initial_problems_debug.json`: Example of problematic generations