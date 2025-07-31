#!/usr/bin/env python3
"""
Convert reasonmed_raw_selected (train split) to SFT JSON format.
Based on PRD requirements for transforming neko-llm/CoT_Medicine dataset.
"""

import json
import os
import argparse
import time
import re
from pathlib import Path
from datasets import load_dataset
import warnings
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_hf_token():
    """Load Hugging Face token from keys.json"""
    try:
        with open("../../keys.json", "r") as f:
            keys = json.load(f)
            return keys["llm"]
    except Exception as e:
        print(f"Error loading HF token: {e}")
        return None


def load_reasonmd_dataset(hf_token):
    """Load the reasonmd dataset from Hugging Face."""
    print("Loading dataset: neko-llm/CoT_Medicine, config: reasonmed_raw_selected, split: train")
    dataset = load_dataset("neko-llm/CoT_Medicine", "reasonmed_raw_selected", split="train", token=hf_token)
    print(f"Loaded {len(dataset)} rows")
    return dataset


def extract_answer_from_output(llm_client, sampling_params, question, output):
    """
    Extract the final answer from the output text.
    """

    prompt = f"""You are given a medical question and reasoning output. Based on the question and reasoning output, extract the final answer as a concise string.
    The final answer should be a single letter (A, B, C, D) or a specific value/term.

    Question:
    {question}

    Reasoning output:
    {output}

    IMPORTANT: Return ONLY the final answer choice (e.g., 'A', 'B', 'C', 'D' or the specific answer value). 
    Do not include any explanations, reasoning, or additional text.
    Do not use backticks, quotes, or any formatting.

    Final answer:"""

    # Try to extract after </think>
    final_answer = call_vllm_model(prompt, llm_client, sampling_params)
    
    # Clean up the answer - extract just the first letter or value
    if final_answer:
        # Remove any backticks, quotes, or extra formatting
        cleaned_answer = final_answer.strip().strip('`').strip('"').strip("'").strip()
        # Take only the first part if there are multiple lines or explanations
        cleaned_answer = cleaned_answer.split('\n')[0].split('.')[0].strip()
        
        # Try to extract just the answer letter if the model included extra text
        # Look for patterns like "The answer is D" or "Answer: D" or just "D"
        import re
        answer_patterns = [
            r'^([A-D])$',  # Just a single letter
            r'answer[:\s]+([A-D])',  # "Answer: D" or "Answer D"
            r'the\s+answer\s+is\s+([A-D])',  # "The answer is D"
            r'final\s+answer[:\s]+([A-D])',  # "Final answer: D"
            r'([A-D])\s*$',  # Letter at the end
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, cleaned_answer, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, return the cleaned answer as is
        return cleaned_answer
    
    return final_answer
    


def transform_row(llm_client, sampling_params, row, idx, debug=False):
    """Transform a single row according to PRD specifications."""
    
    # Check for required fields
    if "output" not in row or row["output"] is None:
        warnings.warn(f"Row {idx}: Missing 'output' field, skipping row")
        return None

    question = str(row["instruction"]).strip()
    if not question:
        warnings.warn(f"Row {idx}: Empty 'instruction' field, skipping row")
        return None

    output = str(row["output"]).strip()
    if not output:
        warnings.warn(f"Row {idx}: Empty 'output' field, skipping row")
        return None
    
    # Extract answer from output
    answer = extract_answer_from_output(llm_client, sampling_params, question, output)

    # validate the answer format
    is_valid, validation_msg = basic_answer_format_check(answer, debug)
    if not is_valid:
        warnings.warn(f"Row {idx}: {validation_msg}, skipping row")
        return None
    
    return {
        "id": f"reasonmed_{idx}",
        "question": question,
        "output": f"<think>{output}</think>\n\n{answer}",
        "answer": answer
    }


def validate_output_format(output_text):
    """Validate the output format matches <think>...</think><final_answer> pattern."""
    if not output_text.startswith('<think>'):
        return False, "Output doesn't start with <think>"
    
    if '</think>' not in output_text:
        return False, "Output doesn't contain </think>"
    
    parts = output_text.split('</think>')
    if len(parts) < 2 or not parts[1].strip():
        return False, "No final answer after </think>"
    
    return True, "Valid format"


def call_vllm_model(prompt, llm_client, sampling_params):
    """Call VLLM model with the given prompt"""
    try:
        outputs = llm_client.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"VLLM call failed: {e}")
        return None


def basic_answer_format_check(answer, debug=False):
    """Basic answer format check without LLM."""
    if not answer or len(answer.strip()) == 0:
        return False, "Empty answer"
    
    answer = answer.strip()
    if debug:
        print("generated answer: ", answer)
    # Check for single letter answers (A, B, C, D)
    if re.match(r'^[A-D]$', answer, re.IGNORECASE):
        return True, "Valid single letter answer"
    
    # Check for reasonable length and content
    if len(answer) < 2:
        return False, "Answer too short"
    
    if len(answer) > 300:
        return False, "Answer too long"
    
    # Check for common invalid phrases
    invalid_phrases = [
        "unable to extract",
        "cannot determine",
        "not clear",
        "unclear",
        "no answer"
    ]
    
    for phrase in invalid_phrases:
        if phrase in answer.lower():
            return False, f"Contains invalid phrase: {phrase}"
    
    return True, "Basic format check passed"


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert reasonmd dataset to SFT format')
    parser.add_argument('--model', default='Qwen/Qwen3-32B', help='Model name for VLLM transformation')
    parser.add_argument('--test-mode', action='store_true', help='Process only first 5 entries for testing')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size for VLLM')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging output')
    args = parser.parse_args()
    
    # Load HF token
    hf_token = load_hf_token()
    if not hf_token:
        print("Error: Hugging Face token not found. Please check keys.json")
        return
    
    # Load dataset
    dataset = load_reasonmd_dataset(hf_token)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=8192)
    sampling_params = SamplingParams(
        temperature=0.1,  # Lower temperature for more consistent outputs
        max_tokens=50,    # Much shorter max tokens since we only need a single letter/word
        top_p=0.9,
        stop=["\n", ".", "```"]  # Stop at newlines, periods, or code blocks
    )
    
    # Apply test mode if requested
    if args.test_mode:
        print("TEST MODE: Processing only first 5 entries")
        dataset = dataset.select(range(5))
    
    # Transform data
    transformed_data = []
    skipped_count = 0
    
    print("Transforming data...")
    for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Processing rows"):
        transformed_row = transform_row(llm, sampling_params, row, idx, args.debug)
        if transformed_row is not None:
            transformed_data.append(transformed_row)
        else:
            skipped_count += 1
    
    print(f"Transformation complete: {len(transformed_data)} rows processed, {skipped_count} rows skipped")
    
    # Create output directory if it doesn't exist
    if args.test_mode:
        output_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_cot_test.json"
    else:
        output_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_cot.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    print(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(transformed_data)} entries to {output_path}")
    
    # Print sample entry for verification
    if transformed_data:
        print("\nSample entry:")
        print(json.dumps(transformed_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Test command examples:
# python convert_reasonmd.py --test-mode --use-llm --debug  # Test with 5 entries and debug output
# python convert_reasonmd.py --use-llm --model "Qwen/Qwen3-32Bt"  # Full run with VLLM
# python convert_reasonmd.py  # Quick run without LLM (uses extracted answers)