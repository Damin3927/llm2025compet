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
        with open("/home/qian.niu/explore/data/hle/sft/keys.json", "r") as f:
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


def extract_answer_from_output(output):
    """Extract the final answer from the output text."""
    
    # Common patterns for answers in medical reasoning
    patterns = [
        r"(?:The (?:correct )?answer is|Answer:|The answer:|Final answer:)\s*([^\n.]+)",
        r"(?:Therefore|Thus|Hence|So),?\s*([^\n.]+)",
        r"(?:In conclusion|To conclude),?\s*([^\n.]+)",
        r"(?:The diagnosis is|Diagnosis:)\s*([^\n.]+)",
        r"(?:The treatment is|Treatment:)\s*([^\n.]+)",
        r"(?:The patient (?:has|should))\s*([^\n.]+)",
    ]
    
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Return the last match (usually the final conclusion)
            answer = matches[-1].strip()
            # Clean up the answer
            answer = re.sub(r'^[^\w]*', '', answer)  # Remove leading non-word chars
            answer = re.sub(r'[^\w\s]*$', '', answer)  # Remove trailing non-word chars
            if len(answer) > 10 and len(answer) < 200:  # Reasonable answer length
                return answer
    
    # Fallback: take last sentence
    sentences = re.split(r'[.!?]+', output)
    if len(sentences) > 1:
        last_sentence = sentences[-2].strip()  # -2 because last is usually empty after split
        if len(last_sentence) > 10 and len(last_sentence) < 200:
            return last_sentence
    
    # Final fallback: take first reasonable sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence) < 300:
            return sentence
    
    return "Unable to extract answer"


def transform_row(row, idx):
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
    answer = extract_answer_from_output(output)
    
    return {
        "id": f"reasonmed_{idx}",
        "question": question,
        "raw_output": output,  # Keep for LLM processing
        "extracted_answer": answer
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


def transform_with_llm(data_entries, model_name="Qwen/Qwen3-32B", tp=1, debug=False):
    """Transform instructions using VLLM to proper CoT format."""
    
    # Initialize VLLM model
    try:
        llm_client = LLM(model=model_name, tensor_parallel_size=tp, max_model_len=4096)
        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=3000,
            top_p=0.95
        )
    except Exception as e:
        print(f"Error initializing VLLM model: {e}")
        return data_entries
    
    print(f"Using VLLM with model: {model_name}")
    print(f"Processing {len(data_entries)} instructions with VLLM...")
    
    # Track timing for each transformation
    transformation_times = []
    format_errors = 0
    
    # Process each entry individually
    for idx, entry in tqdm(enumerate(data_entries), total=len(data_entries), desc="API Processing"):
        row_start_time = time.time()
        
        # Create prompt for transformation
        prompt = f"""You are a medical reasoning expert. Your task is to analyze a medical instruction and create a structured chain-of-thought reasoning process.

TASK: Given a medical instruction that contains both reasoning and a conclusion, extract the reasoning process and the final answer.

INSTRUCTION:
{entry['raw_output']}

REQUIREMENTS:
1. Identify the reasoning steps in the instruction
2. Structure them as clear, logical steps
3. Extract the final answer/conclusion

Please provide only the reasoning process (no other text):"""
        
        # Call VLLM model
        reasoning_text = call_vllm_model(prompt, llm_client, sampling_params)
        processing_time = time.time() - row_start_time
        
        if reasoning_text is None:
            if debug:
                print(f"Entry {idx}: API call failed, using fallback")
            reasoning_text = entry['raw_output']
        
        if debug:
            print(f"\n--- Entry {idx + 1} Debug ---")
            print(f"Raw LLM reasoning (first 200 chars): {repr(reasoning_text[:200])}")
            print(f"Full length: {len(reasoning_text)}")
        
        # Construct the final output with <think> tags
        final_answer = entry['extracted_answer']
        output_text = f"<think>{reasoning_text}</think>{final_answer}"
        
        # Validate format
        is_valid, validation_msg = validate_output_format(output_text)
        
        if not is_valid:
            if debug:
                print(f"Validation failed: {validation_msg}")
            format_errors += 1
            # Use fallback format
            output_text = f"<think>{entry['raw_output']}</think>{final_answer}"
        elif debug:
            print("✓ Format validation passed")
        
        # Update entry
        entry['output'] = output_text
        entry['answer'] = final_answer
        
        # Record timing
        transformation_times.append({
            'entry_id': entry['id'],
            'processing_time': processing_time,
            'reasoning_length': len(reasoning_text),
            'format_valid': is_valid
        })
        
        # Remove temporary fields
        del entry['raw_output']
        del entry['extracted_answer']
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Print timing statistics
    if transformation_times:
        avg_time = sum(t['processing_time'] for t in transformation_times) / len(transformation_times)
        total_time = sum(t['processing_time'] for t in transformation_times)
        successful_conversions = sum(1 for t in transformation_times if t['format_valid'])
        
        print(f"\nVLLM Transformation completed:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per entry: {avg_time:.2f}s")
        print(f"Successful conversions: {successful_conversions}/{len(transformation_times)}")
        print(f"Format errors encountered: {format_errors}")
        
        # Save timing data
        timing_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_timing.json"
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(timing_path, 'w') as f:
            json.dump(transformation_times, f, indent=2)
        print(f"Timing data saved to: {timing_path}")
    
    return data_entries


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert reasonmd dataset to SFT format')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM to transform instructions')
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
    
    # Apply test mode if requested
    if args.test_mode:
        print("TEST MODE: Processing only first 5 entries")
        dataset = dataset.select(range(5))
    
    # Transform data
    transformed_data = []
    skipped_count = 0
    
    print("Transforming data...")
    for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Processing rows"):
        transformed_row = transform_row(row, idx)
        if transformed_row is not None:
            transformed_data.append(transformed_row)
        else:
            skipped_count += 1
    
    print(f"Transformation complete: {len(transformed_data)} rows processed, {skipped_count} rows skipped")
    
    # Transform with LLM if requested
    if args.use_llm:
        print("Transforming instructions with LLM...")
        transformed_data = transform_with_llm(transformed_data, args.model, args.tp, args.debug)
    else:
        print("Skipping LLM transformation. Use --use-llm flag to enable.")
        # Add placeholder output for non-LLM mode
        for entry in transformed_data:
            entry['output'] = f"<think>{entry['raw_output']}</think>{entry['extracted_answer']}"
            entry['answer'] = entry['extracted_answer']
            del entry['raw_output']
            del entry['extracted_answer']
    
    # Create output directory if it doesn't exist
    if args.test_mode:
        output_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_cot_test.json"
    else:
        output_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_cot.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate final format of all entries
    print("\nValidating final format...")
    validation_errors = []
    required_fields = ['id', 'question', 'output', 'answer']
    
    for i, entry in enumerate(transformed_data):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                validation_errors.append(f"Entry {i}: Missing field '{field}'")
        
        # Validate output format
        if 'output' in entry:
            is_valid, msg = validate_output_format(entry['output'])
            if not is_valid:
                validation_errors.append(f"Entry {i}: {msg}")
        
        # Validate ID format
        if 'id' in entry and not entry['id'].startswith('reasonmd_'):
            validation_errors.append(f"Entry {i}: Invalid ID format '{entry['id']}'")
    
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
    report_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()

# Test command examples:
# python convert_reasonmd.py --test-mode --use-llm --debug  # Test with 5 entries and debug output
# python convert_reasonmd.py --use-llm --model "Qwen/Qwen3-32Bt"  # Full run with VLLM
# python convert_reasonmd.py  # Quick run without LLM (uses extracted answers)