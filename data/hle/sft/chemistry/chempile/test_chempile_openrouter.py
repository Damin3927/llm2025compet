#!/usr/bin/env python3
"""
Test script for ChemPile QA extraction using OpenRouter API

Tests the extraction pipeline with 3 examples using OpenRouter instead of vLLM.
"""

import json
import os
import requests
from datasets import load_dataset
import logging
import time
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_answer_format(answer: str) -> bool:
    """
    Validate that the extracted answer has a reasonable format.
    """
    if not answer or len(answer.strip()) == 0:
        return False
    
    # Check if answer is too long (likely contains explanation)
    if len(answer) > 200:
        return False
    
    # Check for common unwanted patterns
    unwanted_patterns = [
        r"the answer is",
        r"therefore",
        r"in conclusion",
        r"based on",
        r"according to"
    ]
    
    answer_lower = answer.lower()
    for pattern in unwanted_patterns:
        if re.search(pattern, answer_lower):
            return False
    
    return True

def call_openrouter_api(prompt: str, api_key: str, model: str = "qwen/qwen3-32b", max_retries: int = 3) -> Optional[str]:
    """
    Generic function to call OpenRouter API with retry logic and validation.
    
    Args:
        prompt: The prompt to send to the API
        api_key: OpenRouter API key
        model: Model to use for the API call
        max_retries: Maximum number of retry attempts
        
    Returns:
        The response content or None if all attempts failed
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 4096
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"API call attempt {attempt + 1}/{max_retries}")
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # Parse response and validate JSON structure
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise ValueError("Failed to get valid JSON response after all retries")
            
            # Validate response structure
            if not response_data.get("choices") or len(response_data["choices"]) == 0:
                logger.warning(f"Empty choices in response on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ValueError("No choices in API response")
            
            # Extract content
            message_content = response_data["choices"][0].get("message", {}).get("content")
            if not message_content:
                logger.warning(f"Empty message content on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ValueError("Empty message content in API response")
            
            content = message_content.strip()
            logger.info(f"Successfully received API response: '{content[:100]}...'")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise ValueError(f"API request failed after {max_retries} attempts: {e}")
        
        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                raise ValueError(f"API call failed after {max_retries} attempts: {e}")
    
    return None

def extract_answer_with_openrouter(question: str, reasoning_text: str, api_key: str, 
                                  custom_prompt: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
    """
    Use OpenRouter API to identify the definitive answer line from reasoning text.
    Includes retry logic and validation for robust extraction.
    
    Args:
        question: The original question
        reasoning_text: The reasoning text to extract answer from
        api_key: OpenRouter API key
        custom_prompt: Optional custom prompt template. If None, uses default prompt.
        max_retries: Maximum number of retry attempts
        
    Returns:
        Extracted answer or None if extraction failed
    """
    # Use custom prompt if provided, otherwise use default
    if custom_prompt is None:
        prompt = f"""Extract the final answer from the following question and reasoning text. Return only the answer without any additional explanation or formatting.

Question:
{question}

Reasoning text:
{reasoning_text}

Please only return the choice of correct answer, no other text, no explanation."""
    else:
        prompt = custom_prompt.format(question=question, reasoning_text=reasoning_text)
    
    # Make initial API call
    response_content = call_openrouter_api(prompt, api_key, max_retries=max_retries)
    
    if response_content is None:
        logger.error("Failed to get response from API")
        return None
    
    # Process the response for answer extraction
    for attempt in range(max_retries):
        try:
            logger.info(f"Answer processing attempt {attempt + 1}/{max_retries}")
            
            # Strip whitespace and enclosing punctuation
            answer = response_content.strip().strip('.,;:!?').strip()
            
            # Validate answer format
            if not validate_answer_format(answer):
                logger.warning(f"Invalid answer format on attempt {attempt + 1}: '{answer}'")
                if attempt < max_retries - 1:
                    # Try a more specific prompt for retry
                    retry_prompt = f"""{prompt}

IMPORTANT: Return ONLY the final answer choice (e.g., 'A', 'B', 'C', 'D' or the specific answer value). Do not include any explanations, reasoning, or additional text."""
                    response_content = call_openrouter_api(retry_prompt, api_key, max_retries=1)
                    if response_content is None:
                        continue
                else:
                    logger.warning(f"Using potentially invalid answer after all retries: '{answer}'")
                    break
            else:
                logger.info(f"Successfully extracted answer: '{answer}'")
                return answer
                
        except Exception as e:
            logger.warning(f"Error processing answer on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None
            continue
    
    # Return the final answer even if validation failed
    return answer if 'answer' in locals() else None

def test_extraction():
    """
    Test the extraction pipeline with 3 examples from the dataset.
    """
    # Check for API key
    with open("../../keys.json", "r") as f:
        keys = json.load(f)
    api_key = keys["openrouter"]
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        return
    
    logger.info("Loading ChemPile dataset...")
    
    # Load the dataset
    dataset = load_dataset(
        "jablonkagroup/chempile-reasoning",
        "claude-3.5-distilled-spectral-reasoning-default",
        split="train"
    )
    
    # Take first 3 examples
    test_samples = dataset.select(range(3))
    
    results = []
    
    for idx, row in enumerate(test_samples):
        try:
            logger.info(f"Processing sample {idx + 1}/3")
            
            # Extract components
            question = row["prompt"]
            reasoning = row["extracted_reasoning"].strip()
            
            # Wrap reasoning with think tags
            wrapped_reasoning = f"<think>{reasoning}</think>"
            
            # Extract answer using OpenRouter with retry logic
            answer = extract_answer_with_openrouter(question, reasoning, api_key)
            
            # Check if extraction failed
            if answer is None:
                logger.error(f"Failed to extract answer for sample {idx + 1}, skipping")
                continue

            
            
            # Construct output format
            output = f"{wrapped_reasoning}{answer}"
            
            # Create record
            record = {
                "id": f"chempile_{idx}",
                "question": question,
                "output": output,
                "answer": answer
            }
            results.append(record)
            
            # Print for debugging
            print(f"\n--- Sample {idx + 1} ---")
            print(f"Question: {question[:100]}...")
            print(f"Answer: {answer}")
            print(f"Think tags present: {'<think>' in output and '</think>' in output}")
                
        except Exception as e:
            logger.error(f"Error processing sample {idx + 1}: {e}")
            # Add to results with error indicator for debugging
            error_record = {
                "id": f"chempile_{idx}_error",
                "question": row.get("prompt", "Unknown"),
                "output": f"<think>Error during processing: {str(e)}</think>ERROR",
                "answer": "ERROR",
                "error": str(e)
            }
            results.append(error_record)
            continue
    
    # Save test results
    output_file = "chempile_test_results.json"
    logger.info(f"Saving {len(results)} test records to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write('\n')
    
    # Validate test results
    validate_test_output(output_file)

def validate_test_output(file_path: str) -> bool:
    """
    Validate the test output file against acceptance criteria.
    """
    logger.info(f"Validating test output: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_keys = {"id", "question", "output", "answer"}
        
        for i, record in enumerate(data):
            # Check schema compliance
            if not isinstance(record, dict) or set(record.keys()) != required_keys:
                logger.error(f"Test record {i} schema validation failed")
                return False
            
            # Check tag integrity
            output = record["output"]
            think_open = output.count("<think>")
            think_close = output.count("</think>")
            
            if think_open != 1 or think_close != 1:
                logger.error(f"Test record {i} think tag validation failed")
                return False
            
            # Check answer correctness
            answer = record["answer"]
            if answer not in output:
                logger.error(f"Test record {i} answer validation failed")
                return False
        
        logger.info(f"Test validation passed for {len(data)} records")
        return True
        
    except Exception as e:
        logger.error(f"Test validation failed: {e}")
        return False

if __name__ == "__main__":
    print("ChemPile QA Extraction Test")
    print("=" * 40)
    print("This script tests the extraction pipeline with 3 examples using OpenRouter.")
    print("Make sure to set your OPENROUTER_API_KEY environment variable.")
    print()
    
    test_extraction()