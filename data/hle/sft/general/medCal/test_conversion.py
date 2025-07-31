#!/usr/bin/env python3
"""
Quick test script to debug LLM output issues using OpenRouter API.
Run this first to test with a small dataset before running the full conversion.
"""

import json
import time
import requests
from pathlib import Path
from datasets import load_dataset

def load_api_key():
    """Load OpenRouter API key from keys.json"""
    try:
        with open("/home/qian.niu/explore/data/hle/sft/keys.json", "r") as f:
            keys = json.load(f)
            return keys.get("openrouter", keys.get("llm"))  # Try openrouter first, fallback to llm
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
        "X-Title": "MedCalc CoT Conversion"
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

def test_openrouter_conversion():
    """Test the conversion process using OpenRouter API"""
    
    print("=" * 60)
    print("RUNNING MEDCALC CONVERSION TEST WITH OPENROUTER")
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
        
        dataset = load_dataset("neko-llm/CoT_Medicine", "medcalc_raw_selected", split="train", token=hf_token)
        print(f"✓ Dataset loaded: {len(dataset)} entries")
        
        # Test with first 3 entries
        test_entries = dataset.select(range(3))
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Process each entry
    results = []
    successful_conversions = 0
    
    for idx, row in enumerate(test_entries, 1):
        print(f"\n--- Processing Entry {idx} ---")
        
        # Build the prompt
        explanation = str(row["Ground Truth Explanation"]).strip()
        answer = str(row["Ground Truth Answer"]).strip()
        
        prompt = f"""
        You are an expert in medical science. Your task is to transform medical explanations into clear, structured chain-of-thought reasoning.
        TASK: Based on the given medical explanation and ground truth answer, please describe the reasoning process that leads to the expected answer.
        INPUTS:
        Medical explanation: {explanation}
        Ground truth answer: {answer}
        Please only output the reasoning process, no other text.
        """
        
        print(f"Original answer: {answer}")
        
        # Call OpenRouter API
        start_time = time.time()
        response = call_openrouter_api(prompt, api_key)
        processing_time = time.time() - start_time
        
        if response is None:
            print("✗ API call failed")
            continue
            
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Response length: {len(response)} characters")
        print(f"Raw response: {repr(response)}")
        
        response = f"<think>{response}</think>{answer}"
        
        print("  Applied fix, re-validating...")
        is_valid, _ = validate_output_format(response)
        print(f"  Fixed version valid: {is_valid}")
        
        # Build result entry
        result_entry = {
            "id": f"medcalc_{idx}",
            "question": f"{row['Patient Note']}\n{row['Question']}",
            "output": response,
            "answer": f"{answer} ({row.get('Lower Limit', 'NA')} ~ {row.get('Upper Limit', 'NA')})",
            "explanation": explanation,
            "processing_time": processing_time,
            "format_valid": is_valid
        }
        
        results.append(result_entry)
    
    # Save test results
    output_path = Path.home() / "explore/data/hle/sft/medical/results/medcalc_openrouter_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total entries processed: {len(results)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Format validation rate: {successful_conversions/len(results)*100:.1f}%")
    print(f"Results saved to: {output_path}")
    
    if len(results) > 0:
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"Average processing time: {avg_time:.2f}s per entry")
        
        print(f"\nSample result:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    
    return successful_conversions == len(results)

if __name__ == "__main__":
    success = test_openrouter_conversion()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        print("This is normal during testing - use the results to debug the conversion process.")