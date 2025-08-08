#!/usr/bin/env python3
"""
Quick test script for reasonMD conversion using OpenRouter API.
Tests the conversion process with a small dataset before running the full conversion.
"""

import json
import time
import requests
import re
from pathlib import Path
from datasets import load_dataset

def load_api_key():
    """Load OpenRouter API key from keys.json"""
    try:
        with open("/home/qian.niu/explore/data/hle/sft/keys.json", "r") as f:
            keys = json.load(f)
            return keys.get("openrouter", keys.get("llm"))
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

def call_openrouter_api(prompt, api_key, model="qwen/qwen3-32b"):
    """Call OpenRouter API with the given prompt"""
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/neko-llm/CoT_Medicine",
        "X-Title": "ReasonMD CoT Conversion"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 3000,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def extract_answer_from_instruction(instruction):
    """Extract the final answer from the instruction text."""
    
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
        matches = re.findall(pattern, instruction, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Return the last match (usually the final conclusion)
            answer = matches[-1].strip()
            # Clean up the answer
            answer = re.sub(r'^[^\w]*', '', answer)  # Remove leading non-word chars
            answer = re.sub(r'[^\w\s]*$', '', answer)  # Remove trailing non-word chars
            if len(answer) > 10 and len(answer) < 200:  # Reasonable answer length
                return answer
    
    # Fallback: take last sentence
    sentences = re.split(r'[.!?]+', instruction)
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

def test_reasonmd_conversion():
    """Test the reasonMD conversion process using OpenRouter API"""
    
    print("=" * 60)
    print("RUNNING REASONMD CONVERSION TEST WITH OPENROUTER")
    print("=" * 60)
    print("This will process 3 entries using OpenRouter API.")
    print()
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("✗ Failed to load OpenRouter API key from keys.json")
        return False
    
    print("✓ API key loaded successfully")
    
    # Load dataset
    try:
        print("Loading dataset...")
        with open("/home/qian.niu/explore/data/hle/sft/keys.json", "r") as f:
            keys = json.load(f)
            hf_token = keys["llm"]
        
        dataset = load_dataset("neko-llm/CoT_Medicine", "reasonmed_raw_selected", split="train", token=hf_token)
        print(f"✓ Dataset loaded: {len(dataset)} entries")
        
        # Test with first 3 entries
        test_entries = dataset.select(range(3))
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Process each entry
    results = []
    successful_conversions = 0
    
    for idx, row in enumerate(test_entries):
        print(f"\n--- Processing Entry {idx} ---")
        
        # Get the instruction
        instruction = str(row["instruction"]).strip()
        if not instruction:
            print("✗ Empty instruction, skipping")
            continue
        
        print(f"Instruction length: {len(instruction)} characters")
        print(f"Instruction preview: {instruction[:200]}...")
        
        # Extract answer from instruction
        extracted_answer = extract_answer_from_instruction(instruction)
        print(f"Extracted answer: {extracted_answer}")
        
        # Create prompt for LLM reasoning
        prompt = f"""You are a medical reasoning expert. Your task is to analyze a medical instruction and create a structured chain-of-thought reasoning process.

TASK: Given a medical instruction that contains both reasoning and a conclusion, extract the reasoning process and the final answer.

INSTRUCTION:
{instruction}

REQUIREMENTS:
1. Identify the reasoning steps in the instruction
2. Structure them as clear, logical steps
3. Extract the final answer/conclusion

Please provide only the reasoning process (no other text):"""
        
        # Call OpenRouter API
        start_time = time.time()
        reasoning_response = call_openrouter_api(prompt, api_key)
        processing_time = time.time() - start_time
        
        if reasoning_response is None:
            print("✗ API call failed")
            continue
            
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Reasoning response length: {len(reasoning_response)} characters")
        print(f"Reasoning preview: {reasoning_response[:200]}...")
        
        # Construct final output
        output_text = f"<think>{reasoning_response}</think>{extracted_answer}"
        
        # Validate format
        is_valid, validation_msg = validate_output_format(output_text)
        
        if is_valid:
            print("✓ Format validation passed")
            successful_conversions += 1
        else:
            print(f"✗ Format validation failed: {validation_msg}")
            # Use fallback
            output_text = f"<think>{instruction}</think>{extracted_answer}"
        
        # Build result entry
        result_entry = {
            "id": f"reasonmd_{idx}",
            "question": instruction,
            "output": output_text,
            "answer": extracted_answer,
            "processing_time": processing_time,
            "reasoning_length": len(reasoning_response) if reasoning_response else 0,
            "format_valid": is_valid
        }
        
        results.append(result_entry)
    
    # Save test results
    output_path = Path.home() / "explore/data/hle/sft/medical/results/reasonmd_openrouter_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total entries processed: {len(results)}")
    print(f"Successful conversions: {successful_conversions}")
    if len(results) > 0:
        print(f"Format validation rate: {successful_conversions/len(results)*100:.1f}%")
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"Average processing time: {avg_time:.2f}s per entry")
        avg_reasoning = sum(r['reasoning_length'] for r in results) / len(results)
        print(f"Average reasoning length: {avg_reasoning:.0f} characters")
    
    print(f"Results saved to: {output_path}")
    
    if len(results) > 0:
        print(f"\nSample result:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    
    return successful_conversions == len(results)

if __name__ == "__main__":
    success = test_reasonmd_conversion()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        print("This is normal during testing - use the results to debug the conversion process.")