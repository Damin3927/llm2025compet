#!/usr/bin/env python3
"""
Convert medcalc_raw_selected (train split) to SFT JSON format.
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
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm


with open("/home/qian.niu/explore/data/hle/sft/keys.json", "r") as f:
    keys = json.load(f)
    hf_token = keys["llm"]

def load_medcalc_dataset():
    """Load the medcalc dataset from Hugging Face."""
    print("Loading dataset: neko-llm/CoT_Medicine, config: medcalc_raw_selected, split: train")
    dataset = load_dataset("neko-llm/CoT_Medicine", "medcalc_raw_selected", split="train", token=hf_token)
    print(f"Loaded {len(dataset)} rows")
    return dataset


def transform_row(row, idx):
    """Transform a single row according to PRD specifications."""
    
    # Check for required fields
    required_fields = ["Patient Note", "Question", "Ground Truth Answer", "Ground Truth Explanation"]
    for field in required_fields:
        if field not in row or row[field] is None or str(row[field]).strip() == "":
            warnings.warn(f"Row {idx}: Missing required field '{field}', skipping row")
            return None
    
    # Build the question field
    question_parts = [str(row["Patient Note"]).strip()]
    question_parts.append(str(row["Question"]).strip())
    
    # Add Relevant Entities if not empty
    if "Relevant Entities" in row and row["Relevant Entities"] is not None:
        entities = str(row["Relevant Entities"]).strip()
        if entities:
            question_parts.append(entities)
    
    question = "\n".join(question_parts)
    
    # Build the output field - store raw explanation for LLM processing
    explanation = str(row["Ground Truth Explanation"]).strip()
    ground_truth = str(row["Ground Truth Answer"]).strip()
    
    # Store raw explanation for LLM transformation
    raw_explanation = explanation
    raw_answer = ground_truth
    
    # Build the answer field
    lower_limit = row.get("Lower Limit", "NA")
    upper_limit = row.get("Upper Limit", "NA")
    
    # Handle missing limits
    if lower_limit is None or str(lower_limit).strip() == "":
        lower_limit = "NA"
    else:
        lower_limit = str(lower_limit).strip()
        
    if upper_limit is None or str(upper_limit).strip() == "":
        upper_limit = "NA"
    else:
        upper_limit = str(upper_limit).strip()
    
    answer = f"{ground_truth} ({lower_limit} ~ {upper_limit})"
    
    return {
        "id": f"medcalc_{idx}",
        "question": question,
        "raw_explanation": raw_explanation,
        "raw_answer": raw_answer,
        "answer": answer
    }


def validate_output_format(output_text):
    """Validate the output format matches <think>...</think><final_answer> pattern."""
    # Check if it starts with <think> and contains </think>
    if not output_text.startswith('<think>'):
        return False, "Output doesn't start with <think>"
    
    if '</think>' not in output_text:
        return False, "Output doesn't contain </think>"
    
    # Check if there's content after </think>
    parts = output_text.split('</think>')
    if len(parts) < 2 or not parts[1].strip():
        return False, "No final answer after </think>"
    
    return True, "Valid format"


def transform_explanations_with_llm(data_entries, model_name="Qwen/Qwen2.5-32B-Instruct", debug=False):
    """Transform explanations using vllm + Qwen3-32B to proper CoT format."""
    
    print(f"Initializing LLM model: {model_name}")
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        max_model_len=4096,  # Reduced from default 40960 to fit in GPU memory
        dtype="float16",  # Use half precision to save memory
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=3000,
        stop=None,
        repetition_penalty=1.1
    )
    
    # Create prompts for transformation
    prompts = []
    for entry in data_entries:
        prompt = f"""
        You are an expert in medical science. Your task is to transform medical explanations into clear, structured chain-of-thought reasoning.
        TASK: Based on the given medical explanation and ground truth answer, please describe the reasoning process that leads to the expected answer.
        INPUTS:
        Medical explanation: {entry['raw_explanation']}
        Ground truth answer: {entry['raw_answer']}
        Please only output the reasoning process, no other text.
        """
        prompts.append(prompt)
    
    print(f"Processing {len(prompts)} explanations with LLM...")
    
    # Track timing for each transformation
    transformation_times = []
    format_errors = 0
    
    # Process in batches with progress bar
    batch_size = 16  # Reduced batch size to fit in GPU memory
    for i in tqdm(range(0, len(prompts), batch_size), desc="LLM Processing"):
        batch_start = time.time()
        batch_prompts = prompts[i:i+batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        batch_time = time.time() - batch_start
        
        # Update entries with transformed output
        for j, output in enumerate(batch_outputs):
            entry_idx = i + j
            row_start_time = time.time()
            
            # Get the raw output
            transformed_text = output.outputs[0].text.strip()
            
            # Debug output - show first few characters
            if debug:
                print(f"\n--- Entry {entry_idx + 1} Debug ---")
                print(f"Raw LLM output (first 200 chars): {repr(transformed_text[:200])}")
                print(f"Full length: {len(transformed_text)}")
            
            # Validate format
            is_valid, validation_msg = validate_output_format(transformed_text)
            response = f"<think>{transformed_text}</think>{entry['raw_answer']}"
            
            data_entries[entry_idx]['output'] = response
            
            # Record timing for this row
            row_time = time.time() - row_start_time
            transformation_times.append({
                'entry_id': data_entries[entry_idx]['id'],
                'processing_time': row_time,
                'batch_time_per_item': batch_time / len(batch_outputs)
            })
            
            # Remove temporary fields
            del data_entries[entry_idx]['raw_explanation']
            del data_entries[entry_idx]['raw_answer']
    
    # Print timing statistics
    if transformation_times:
        avg_time = sum(t['processing_time'] for t in transformation_times) / len(transformation_times)
        total_time = sum(t['processing_time'] for t in transformation_times)
        print(f"\nLLM Transformation completed:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per entry: {avg_time:.4f}s")
        print(f"Format errors encountered: {format_errors}")
        
        # Save timing data
        timing_path = Path.home() / "explore/data/hle/sft/medical/results/medcalc_timing.json"
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(timing_path, 'w') as f:
            json.dump(transformation_times, f, indent=2)
        print(f"Timing data saved to: {timing_path}")
    
    return data_entries


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert medcalc dataset to SFT format')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM to transform explanations')
    parser.add_argument('--model', default='Qwen/Qwen2.5-32B-Instruct', help='Model name for LLM transformation')
    parser.add_argument('--test-mode', action='store_true', help='Process only first 5 entries for testing')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging output')
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_medcalc_dataset()
    
    # Apply test mode if requested
    if args.test_mode:
        print("TEST MODE: Processing only first 5 entries")
        dataset = dataset.select(range(5))
    
    # Transform data
    transformed_data = []
    skipped_count = 0
    
    print("Transforming data...")
    for idx, row in tqdm(enumerate(dataset, 1), total=len(dataset), desc="Processing rows"):
        transformed_row = transform_row(row, idx)
        if transformed_row is not None:
            transformed_data.append(transformed_row)
        else:
            skipped_count += 1
    
    print(f"Transformation complete: {len(transformed_data)} rows processed, {skipped_count} rows skipped")
    
    # Transform explanations with LLM if requested
    if args.use_llm:
        print("Transforming explanations with LLM...")
        transformed_data = transform_explanations_with_llm(transformed_data, args.model, args.debug)
    else:
        print("Skipping LLM transformation. Use --use-llm flag to enable.")
        # Add placeholder output for non-LLM mode
        for entry in transformed_data:
            entry['output'] = f"<think>{entry['raw_explanation']}</think>{entry['raw_answer']}"
            del entry['raw_explanation']
            del entry['raw_answer']
    
    # Create output directory if it doesn't exist
    if args.test_mode:
        output_path = Path.home() / "explore/data/hle/sft/medical/results/medcalc_cot_test.json"
    else:
        output_path = Path.home() / "explore/data/hle/sft/medical/results/medcalc_cot.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate final format of all entries
    print("\nValidating final format...")
    validation_errors = []
    required_fields = ['id', 'question', 'output', 'answer']
    
    for i, entry in enumerate(transformed_data):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                validation_errors.append(f"Entry {i+1}: Missing field '{field}'")
        
        # Validate output format
        if 'output' in entry:
            is_valid, msg = validate_output_format(entry['output'])
            if not is_valid:
                validation_errors.append(f"Entry {i+1}: {msg}")
        
        # Validate ID format
        if 'id' in entry and not entry['id'].startswith('medcalc_'):
            validation_errors.append(f"Entry {i+1}: Invalid ID format '{entry['id']}'")
    
    if validation_errors:
        print(f"\nValidation errors found ({len(validation_errors)} total):")
        for error in validation_errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors")
    else:
        print("✓ All entries passed format validation")
    
    # Save to JSON
    print(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(transformed_data)} entries to {output_path}")
    
    # Print sample entry for verification
    if transformed_data:
        print("\nSample entry:")
        print(json.dumps(transformed_data[0], indent=2, ensure_ascii=False))
    
    # Save validation report
    validation_report = {
        'total_entries': len(transformed_data),
        'validation_errors': len(validation_errors),
        'error_details': validation_errors
    }
    report_path = Path.home() / "explore/data/hle/sft/medical/results/medcalc_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()

# Test command examples:
# python convert_medcalc.py --test-mode --use-llm --debug  # Test with 5 entries and debug output
# python convert_medcalc.py --use-llm --model "Qwen/Qwen2.5-32B-Instruct"  # Full run with LLM
# python convert_medcalc.py  # Quick run without LLM (uses fallback formatting)