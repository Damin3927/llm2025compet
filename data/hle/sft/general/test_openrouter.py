#!/usr/bin/env python3
"""
Test script for StrategyQA transformation using OpenRouter API.
Based on PRD requirements for Natural-Language Reasoning Generator.
"""

import json
import os
import requests
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
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

class OpenRouterClient:
    def __init__(self, api_key: str, model: str = "qwen/qwen3-32b"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.1) -> str:
        """Generate text using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

def load_sample_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """Load a small sample of the StrategyQA dataset for testing."""
    logger.info(f"Loading {num_samples} sample records from zen-E/StrategyQA_CoT_GPT4o...")
    dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o", split="train")
    return list(dataset.select(range(min(num_samples, len(dataset)))))

def transform_record(record: Dict[str, Any], index: int, rewritten_reasoning: str) -> Dict[str, Any]:
    """Transform a single record according to the output schema."""
    return {
        "id": f"StrategyQA_{index}",
        "question": record["question"],
        "output": f"<think>{rewritten_reasoning}</think>\n\n{record['answer']}",
        "answer": record["answer"]
    }

def test_transformation(client: OpenRouterClient, sample_data: List[Dict[str, Any]], 
                       max_tokens: int, temperature: float) -> List[Dict[str, Any]]:
    """Test the transformation process on sample data."""
    transformed_records = []
    
    for i, record in enumerate(sample_data):
        logger.info(f"Processing record {i+1}/{len(sample_data)}: {record['question'][:50]}...")
        
        # Generate rewritten reasoning
        prompt = REWRITE_PROMPT.format(cot=record["cot"])
        try:
            rewritten_reasoning = client.generate(prompt, max_tokens, temperature)
            logger.info(f"Original CoT length: {len(record['cot'])} chars")
            logger.info(f"Rewritten length: {len(rewritten_reasoning)} chars")
            
            # Transform record
            transformed_record = transform_record(record, i, rewritten_reasoning)
            transformed_records.append(transformed_record)
            
            # Print example for verification
            if i == 0:
                logger.info("First transformed example:")
                logger.info(json.dumps(transformed_record, indent=2))
                
        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            continue
    
    return transformed_records

def main():
    parser = argparse.ArgumentParser(description="Test StrategyQA transformation with OpenRouter")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="qwen/qwen3-32b", help="Model to use")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--output", default="test_strategyqa_openrouter.json", help="Output JSON file")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Get API key
    with open("../keys.json", "r") as f:
        api_key = json.load(f)["openrouter"]
    if not api_key:
        logger.error("Please provide OpenRouter API key via --api-key or OPENROUTER_API_KEY env var")
        return
    
    # Load sample data
    sample_data = load_sample_data(args.num_samples)
    logger.info(f"Loaded {len(sample_data)} sample records")
    
    # Initialize OpenRouter client
    client = OpenRouterClient(api_key, args.model)
    logger.info(f"Testing with model: {args.model}")
    
    # Test transformation
    transformed_records = test_transformation(
        client, sample_data, args.max_tokens, args.temperature
    )
    
    if transformed_records:
        # Save output
        logger.info(f"Saving {len(transformed_records)} transformed records to {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(transformed_records, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"Test complete! Successfully transformed {len(transformed_records)}/{len(sample_data)} records")
        
        # Show stats
        avg_original_length = sum(len(record["cot"]) for record in sample_data) / len(sample_data)
        logger.info(f"Average original CoT length: {avg_original_length:.1f} characters")
    else:
        logger.error("No records were successfully transformed")

if __name__ == "__main__":
    main()