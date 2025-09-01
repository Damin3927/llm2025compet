#!/usr/bin/env python3
"""
DPO dataset V2 generator
Creates improved DPO training data by fixing problematic response categories
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DPOV2Generator:
    def __init__(self):
        """Initialize generator with API credentials and target categories"""
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        self.api_base = "https://openrouter.ai/api/v1"
        self.model = "meta-llama/llama-3.1-70b-instruct"
        
        # Categories that need improved responses
        self.target_categories = {
            'disclaimer': ['Causing material harm by disseminating misinformation e.g. in medicine or law'],
            'refusal': [
                'Compromise privacy by leaking or inferring private information (person/individual)',
                'Nudging or advising users to perform unethical or unsafe actions'
            ]
        }
        
        self.output_dir = Path("processed")
        self.output_dir.mkdir(exist_ok=True)

    def load_source_dataset(self) -> pd.DataFrame:
        """Load source dataset from Hugging Face"""
        dataset = load_dataset("neko-llm/dna_dpo_hh-rlhf")['train']
        return pd.DataFrame(list(dataset))

    def filter_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to only target categories that need fixing"""
        target_categories = []
        for categories in self.target_categories.values():
            target_categories.extend(categories)
        return df[df['type_of_harm'].isin(target_categories)].copy()

    def call_openrouter_api(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API to generate improved responses"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"API error: {e}")
            return None

    def generate_chosen_response(self, question: str, type_of_harm: str) -> Optional[str]:
        """Generate improved chosen response based on harm category"""
        if type_of_harm in self.target_categories['disclaimer']:
            prompt = f"Generate a safe response with disclaimer for: {question}"
        else:
            prompt = f"Generate a polite refusal for: {question}"
        
        return self.call_openrouter_api(prompt)

    def process_row(self, row: pd.Series) -> Optional[Dict]:
        """Process single row to create DPO training pair"""
        try:
            question = row['question']
            type_of_harm = row['type_of_harm']
            original_preferred = row['preferred_output']
            
            # Generate improved chosen response
            new_chosen = self.generate_chosen_response(question, type_of_harm)
            if not new_chosen:
                return None
            
            # Use original preferred as rejected (since it was problematic)
            return {
                'prompt': question,
                'chosen': new_chosen,
                'rejected': original_preferred
            }
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            return None

    def generate_dataset(self):
        """Main dataset generation pipeline"""
        try:
            df = self.load_source_dataset()
            filtered_df = self.filter_target_data(df)
            
            if len(filtered_df) == 0:
                logger.warning("No target data found")
                return
            
            processed_data = []
            
            # Process each row to create DPO pairs
            for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing"):
                result = self.process_row(row)
                if result:
                    processed_data.append(result)
            
            # Save results to JSONL file
            output_file = self.output_dir / "dataset_v2.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Generated {len(processed_data)} items -> {output_file}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

def main():
    try:
        generator = DPOV2Generator()
        generator.generate_dataset()
        logger.info("Dataset generation completed")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
