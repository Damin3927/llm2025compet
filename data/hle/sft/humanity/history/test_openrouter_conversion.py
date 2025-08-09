#!/usr/bin/env python3
import json
import argparse
import requests
import os
from typing import List, Dict, Any
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: str = None, model: str = "qwen/qwen3-32b"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY env var or pass api_key parameter.")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "History Conversion Test",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Generate text using OpenRouter API."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            logger.error(f"Response: {response.text}")
            raise

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
    return f"""Please summarize the original answer to the shortest possible answer that directly answer the question, dont include any other text:

Question: {question}

Original Answer: {original_answer}
"""

def process_with_openrouter(data: List[Dict[str, str]], client: OpenRouterClient) -> List[Dict[str, Any]]:
    """Process the data using OpenRouter to generate reasoning and summaries."""
    logger.info(f"Processing data with OpenRouter model: {client.model}")
    
    results = []
    
    for idx, item in enumerate(data):
        logger.info(f"Processing item {idx + 1}/{len(data)}")
        
        question = item['question']
        original_answer = item['answer']
        
        try:
            # Generate reasoning process
            reasoning_prompt = create_reasoning_prompt(question, original_answer)
            logger.info("Generating reasoning...")
            reasoning_response = client.generate(reasoning_prompt)
            reasoning_response = f"<think>{reasoning_response}</think>"
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
            # Generate summary
            summary_prompt = create_summary_prompt(question, original_answer)
            logger.info("Generating summary...")
            summary_response = client.generate(summary_prompt)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
            # Format the output
            reasoning_response += "\n\n" + summary_response
                
            result = {
                "id": f"history_{idx}",
                "question": question,
                "output": reasoning_response,
                "answer": summary_response
            }
            
            results.append(result)
            logger.info(f"Successfully processed item {idx + 1}")
            
        except Exception as e:
            logger.error(f"Failed to process item {idx + 1}: {e}")
            # Continue with next item instead of failing completely
            continue
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save the processed results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

def create_sample_data() -> List[Dict[str, str]]:
    """Create sample test data for testing."""
    return [
        {
            "question": "What were the main causes of World War I?",
            "answer": "World War I was caused by a complex web of factors including militarism, alliances, imperialism, and nationalism. The immediate trigger was the assassination of Archduke Franz Ferdinand in 1914, but underlying tensions had been building for years due to competing imperial interests, an arms race, and a complex system of alliances that divided Europe into opposing camps."
        },
        {
            "question": "Who was Napoleon Bonaparte?",
            "answer": "Napoleon Bonaparte was a French military general and political leader who rose to prominence during the French Revolution. He became Emperor of the French in 1804 and conquered much of continental Europe through the Napoleonic Wars. He is remembered for his military genius, legal reforms including the Napoleonic Code, and his eventual defeat and exile."
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Test history data conversion using OpenRouter API")
    parser.add_argument("--input", default="./results/history.json", help="Input JSON file path (optional, will use sample data if not provided)")
    parser.add_argument("--output", default="./results/test_output.json", help="Output JSON file path")
    parser.add_argument("--model", default="qwen/qwen3-32b", help="Model name for OpenRouter")
    parser.add_argument("--limit", type=int, default=3, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    try:
        # Initialize OpenRouter client
        with open("../../keys.json", "r") as f:
            keys = json.load(f)

        client = OpenRouterClient(api_key=keys["openrouter"], model=args.model)
        logger.info(f"Initialized OpenRouter client with model: {args.model}")
        
        # Load or create test data
        if args.input:
            logger.info("Loading input data...")
            data = load_data(args.input)
        else:
            logger.info("Using sample test data...")
            data = create_sample_data()
        
        if args.limit:
            data = data[:args.limit]
            logger.info(f"Processing limited dataset: {len(data)} samples")
        
        logger.info(f"Processing {len(data)} samples...")
        results = process_with_openrouter(data, client)
        
        if results:
            logger.info("Saving results...")
            save_results(results, args.output)
            logger.info(f"Conversion completed successfully! Processed {len(results)} items.")
        else:
            logger.error("No results generated. Check your API key and network connection.")
            
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())