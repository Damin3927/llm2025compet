#!/usr/bin/env python3
"""
ChemPile QA Pair Extraction Script

Extract question-answer pairs with full reasoning traces from the 
jablonkagroup/chempile-reasoning dataset and format them for downstream evaluation.
"""

import json
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
import re
import gc
import logging
import time
from typing import List, Dict, Any, Optional

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

def call_vllm_api(llm: LLM, prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Generic function to call vLLM with retry logic and validation.
    
    Args:
        llm: The vLLM model instance
        prompt: The prompt to send to the model
        max_retries: Maximum number of retry attempts
        
    Returns:
        The response content or None if all attempts failed
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
        stop=["\n", "Answer:", "Final answer:", "Question:"]
    )
    
    for attempt in range(max_retries):
        try:
            logger.info(f"vLLM call attempt {attempt + 1}/{max_retries}")
            
            outputs = llm.generate([prompt], sampling_params)
            
            if not outputs or len(outputs) == 0:
                logger.warning(f"Empty outputs on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Short delay before retry
                    continue
                else:
                    raise ValueError("No outputs from vLLM")
            
            if not outputs[0].outputs or len(outputs[0].outputs) == 0:
                logger.warning(f"Empty output texts on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise ValueError("No output text from vLLM")
            
            content = outputs[0].outputs[0].text.strip()
            
            if not content:
                logger.warning(f"Empty content on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise ValueError("Empty content from vLLM")
            
            logger.info(f"Successfully received vLLM response: '{content[:100]}...'")
            return content
            
        except Exception as e:
            logger.warning(f"vLLM call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise ValueError(f"vLLM call failed after {max_retries} attempts: {e}")
    
    return None

def extract_answer_with_qwen(llm: LLM, question: str, reasoning_text: str, 
                           custom_prompt: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
    """
    Use Qwen model to identify the definitive answer line from reasoning text.
    Includes retry logic and validation for robust extraction.
    
    Args:
        llm: The vLLM model instance
        question: The original question
        reasoning_text: The reasoning text to extract answer from
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

Please only return the choice of correct answer, no other text, no explanation.

Final answer:"""
    else:
        prompt = custom_prompt.format(question=question, reasoning_text=reasoning_text)
    
    # Make initial vLLM call
    response_content = call_vllm_api(llm, prompt, max_retries=max_retries)
    
    if response_content is None:
        logger.error("Failed to get response from vLLM")
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

IMPORTANT: Return ONLY the final answer choice (e.g., 'A', 'B', 'C', 'D' or the specific answer value). Do not include any explanations, reasoning, or additional text.

Final answer:"""
                    response_content = call_vllm_api(llm, retry_prompt, max_retries=1)
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

def process_dataset(model_name: str = "Qwen/Qwen2.5-3B-Instruct", output_file: str = "chempile_qa_pairs.json") -> None:
    """
    Process the ChemPile dataset and extract QA pairs.
    """
    logger.info("Loading ChemPile dataset...")
    
    # Load the dataset
    dataset = load_dataset(
        "jablonkagroup/chempile-reasoning",
        "claude-3.5-distilled-spectral-reasoning-default",
        split="train"
    )
    
    logger.info(f"Loaded {len(dataset)} samples from dataset")
    
    # Initialize vLLM model with tensor parallelism
    logger.info(f"Initializing vLLM with model: {model_name}, tp=2")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        trust_remote_code=True,
        tensor_parallel_size=2
    )
    
    results = []
    
    for idx, row in enumerate(dataset):
        try:
            # Extract components
            question = row["prompt"]
            reasoning = row["extracted_reasoning"].strip()
            
            # Wrap reasoning with think tags
            wrapped_reasoning = f"<think>{reasoning}</think>"
            
            # Extract answer using Qwen with retry logic
            answer = extract_answer_with_qwen(llm, question, reasoning)
            
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
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples")
                
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
    
    # Save results
    logger.info(f"Saving {len(results)} records to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Newline-terminated as required
    
    logger.info("Processing complete!")

def validate_output(file_path: str) -> bool:
    """
    Validate the output file against acceptance criteria.
    """
    logger.info(f"Validating output file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("Output is not a JSON array")
            return False
        
        required_keys = {"id", "question", "output", "answer"}
        
        for i, record in enumerate(data):
            # Check schema compliance
            if not isinstance(record, dict) or set(record.keys()) != required_keys:
                logger.error(f"Record {i} does not have exactly the four required keys")
                return False
            
            # Check tag integrity
            output = record["output"]
            think_open = output.count("<think>")
            think_close = output.count("</think>")
            
            if think_open != 1 or think_close != 1:
                logger.error(f"Record {i} does not have exactly one opening and closing think tag")
                return False
            
            # Check answer correctness
            answer = record["answer"]
            if answer not in output:
                logger.error(f"Record {i} answer '{answer}' not found in output")
                return False
        
        logger.info(f"Validation passed for {len(data)} records")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract QA pairs from ChemPile dataset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", 
                       help="Model name for answer extraction")
    parser.add_argument("--output", default="chempile_qa_pairs.json",
                       help="Output JSON file path")
    parser.add_argument("--validate", action="store_true",
                       help="Validate output file after processing")
    
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(args.model, args.output)
    
    # Validate if requested
    if args.validate:
        validate_output(args.output)

if __name__ == "__main__":
    main()