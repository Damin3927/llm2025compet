#!/usr/bin/env python3
"""
Transform StrategyQA dataset using vLLM for reasoning rewriting.
Based on PRD requirements for Natural-Language Reasoning Generator.
"""

import json
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are tasked with rewriting chain-of-thought reasoning to be more readable and natural while maintaining faithfulness to the original logic.

Original reasoning: {cot}

Requirements:
- Keep the same logical path and facts
- Write in complete sentences, avoid "Step 1/2" scaffolding  
- Be concise (1-3 short paragraphs, ≤120 words)
- Use neutral tone, no personal opinions
- Make it readable and human-like

Rewrite the reasoning:"""

def load_strategyqa_dataset() -> List[Dict[str, Any]]:
    """Load the StrategyQA CoT GPT4o dataset."""
    logger.info("Loading zen-E/StrategyQA_CoT_GPT4o dataset...")
    dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o", split="train")
    return list(dataset)

def rewrite_reasoning_batch(llm: LLM, cot_texts: List[str], sampling_params: SamplingParams) -> List[str]:
    """Rewrite multiple CoT texts using vLLM batch processing."""
    prompts = [REWRITE_PROMPT.format(cot=cot) for cot in cot_texts]
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def transform_record(record: Dict[str, Any], index: int, rewritten_reasoning: str) -> Dict[str, Any]:
    """Transform a single record according to the output schema."""
    return {
        "id": f"StrategyQA_{index}",
        "question": record["question"],
        "output": f"<think>{rewritten_reasoning}</think>\n\n{record['answer']}",
        "answer": record["answer"]
    }

def main():
    parser = argparse.ArgumentParser(description="Transform StrategyQA dataset with vLLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to use for rewriting")
    parser.add_argument("--output", default="strategyqa_transformed.json", help="Output JSON file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_strategyqa_dataset()
    logger.info(f"Loaded {len(dataset)} records")
    
    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=None
    )
    
    transformed_records = []
    
    # Process in batches
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i + args.batch_size]
        batch_cots = [record["cot"] for record in batch]
        
        logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(dataset) + args.batch_size - 1)//args.batch_size}")
        
        # Rewrite reasoning for the batch
        rewritten_batch = rewrite_reasoning_batch(llm, batch_cots, sampling_params)
        
        # Transform records
        for j, (record, rewritten) in enumerate(zip(batch, rewritten_batch)):
            transformed_record = transform_record(record, i + j, rewritten)
            transformed_records.append(transformed_record)
    
    # Save output
    logger.info(f"Saving {len(transformed_records)} transformed records to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(transformed_records, f, indent=2, ensure_ascii=False)
    
    logger.info("Transformation complete!")

if __name__ == "__main__":
    main()