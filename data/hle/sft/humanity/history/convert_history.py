#!/usr/bin/env python3
import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(input_file: str) -> List[Dict[str, str]]:
    """Load the original history data from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_reasoning_prompt(question: str, original_answer: str) -> str:
    """Create a prompt to generate reasoning process."""
    return f"""Given this question and answer, please describe the reasoning process that leads to the expected answer. 

Question: {question}

Original Answer: {original_answer}

please only output the reasoning process, no other text.
"""

def create_summary_prompt(question: str, original_answer: str) -> str:
    """Create a prompt to generate a summarized answer."""
    return f"""Q: {question}
A: {original_answer}

Short answer:"""

def process_with_vllm(data: List[Dict[str, str]], model_name: str = "meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size: int = 2) -> List[Dict[str, Any]]:
    """Process the data using vLLM to generate reasoning and summaries."""
    logger.info(f"Initializing vLLM with model: {model_name}")
    
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=4096
    )
    answer_params = SamplingParams(
        temperature=0.0,
        top_p=0.1,
        max_tokens=50,
        stop=["\n", ".", "!", "?"]
    )
    
    results = []
    
    for idx, item in enumerate(data):
        logger.info(f"Processing item {idx + 1}/{len(data)}")
        
        question = item['question']
        original_answer = item['answer']
        
        # Generate reasoning process
        reasoning_prompt = create_reasoning_prompt(question, original_answer)
        reasoning_outputs = llm.generate([reasoning_prompt], sampling_params)
        reasoning_response = reasoning_outputs[0].outputs[0].text.strip()
        
        # Generate summary
        summary_prompt = create_summary_prompt(question, original_answer)
        summary_outputs = llm.generate([summary_prompt], answer_params)
        summary_response = summary_outputs[0].outputs[0].text.strip()
        print(summary_response)

        result = {
            "id": f"history_{idx}",
            "question": question,
            "output": f"<think>{reasoning_response}</think>\n\n" + summary_response,
            "answer": summary_response
        }
        
        results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save the processed results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert history data to reasoning format using vLLM")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Model name for vLLM")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size for vLLM")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    logger.info("Loading input data...")
    data = load_data(args.input)
    
    if args.limit:
        data = data[:args.limit]
        logger.info(f"Processing limited dataset: {len(data)} samples")
    
    logger.info(f"Processing {len(data)} samples...")
    results = process_with_vllm(data, args.model, args.tp)
    
    logger.info("Saving results...")
    save_results(results, args.output)
    
    logger.info("Conversion completed successfully!")

if __name__ == "__main__":
    main()