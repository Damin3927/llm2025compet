#!/usr/bin/env python3
"""
Think tag generation script for DPO datasets
Adds <think> tags to preferred outputs to improve reasoning transparency
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class ThinkTagGenerator:
    def __init__(self, api_key: str, model_name: str, apply_to_non_preferred: bool = False):
        """Initialize think tag generator with API credentials"""
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_name = model_name
        self.apply_to_non_preferred = apply_to_non_preferred

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM to generate thinking process"""
        system_prompt = "Generate a concise thinking process for the given question and answer. Output in English only."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else None
                
        except Exception as e:
            logging.error(f"API error: {e}")
            return None

    def _generate_think(self, question: str, answer: str, is_preferred: bool = True) -> str:
        """Generate think tag content based on question and answer"""
        if is_preferred:
            prompt = f"Question: {question}\nAnswer: {answer}\n\nGenerate thinking process for why this is appropriate:"
        else:
            prompt = f"Question: {question}\nAnswer: {answer}\n\nGenerate thinking process for why this is problematic:"
        
        result = self._call_llm(prompt)
        if result:
            return result
        
        # Fallback responses when API fails
        if is_preferred:
            return "This response follows safety guidelines and provides appropriate information."
        else:
            return "This response violates safety guidelines and should be avoided."

    def _has_think_tag(self, text: str) -> bool:
        """Check if text already contains think tags"""
        return isinstance(text, str) and "<think>" in text and "</think>" in text

    def _add_think_tag(self, text: str, think_content: str) -> str:
        """Add think tag to the beginning of text"""
        if not think_content:
            return text
        return f"<think>{think_content}</think>{text}"

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process single dataset item to add think tags"""
        if not isinstance(item, dict) or 'question' not in item or 'preferred_output' not in item:
            return item
            
        result = item.copy()
        question = item["question"]
        preferred_output = item["preferred_output"]
        
        # Add think tag to preferred output if missing
        if not self._has_think_tag(preferred_output):
            think_content = self._generate_think(question, preferred_output, is_preferred=True)
            result["preferred_output"] = self._add_think_tag(preferred_output, think_content)
        
        # Optionally add think tag to non-preferred output
        if self.apply_to_non_preferred and 'non_preferred_output' in item:
            non_preferred_output = item["non_preferred_output"]
            if not self._has_think_tag(non_preferred_output):
                think_content = self._generate_think(question, non_preferred_output, is_preferred=False)
                result["non_preferred_output"] = self._add_think_tag(non_preferred_output, think_content)
        
        return result


def setup_logging(log_dir: str, start_index: int, end_index: int):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{log_dir}/think_tags_{start_index}-{end_index}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                       handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def load_existing_ids(output_file: str) -> set:
    existing_ids = set()
    if Path(output_file).exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(data['id'])
                except json.JSONDecodeError:
                    pass
    return existing_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--end_index", type=int, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model_name", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset_name", type=str, default="argo11/DNA_DPO_hh-rlhf")
    parser.add_argument("--apply_to_non_preferred", action="store_true")
    
    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = f"think_tagged_{args.start_index}-{args.end_index-1}.jsonl"
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return 1
    
    setup_logging(args.log_dir, args.start_index, args.end_index)
    
    try:
        dataset = load_dataset(args.dataset_name, split="train")
        generator = ThinkTagGenerator(api_key, args.model_name, args.apply_to_non_preferred)
        existing_ids = load_existing_ids(args.output_file)
        
        processed_count = 0
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output_file, 'a', encoding='utf-8') as f:
            try:
                dataset_len = len(dataset)
            except:
                dataset_len = args.end_index
                
            for i in tqdm(range(args.start_index, min(args.end_index, dataset_len)), desc="Processing"):
                try:
                    item = dataset[i]
                    # Convert to dict safely
                    if hasattr(item, 'keys'):
                        item_dict = dict(item)
                    else:
                        item_dict = item
                    
                    if not isinstance(item_dict, dict):
                        continue
                        
                    item_id = item_dict.get('id', f'index_{i}')
                    
                    if item_id in existing_ids:
                        continue
                    
                    result = generator.process_item(item_dict)
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Error processing item {i}: {e}")
        
        logging.info(f"Processed {processed_count} items")
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
