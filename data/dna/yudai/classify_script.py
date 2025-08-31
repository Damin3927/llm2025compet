#!/usr/bin/env python3
"""
Harm classification script for DNA DPO dataset
Classifies questions into 3-level hierarchy: risk area -> type of harm -> specific harm
"""

import os
import json
import argparse
import logging
import re
from typing import Dict, Any, Optional, List
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import wandb
import pandas as pd

load_dotenv()

class HarmClassifier:
    def __init__(self, api_key: str, model_name: str, hierarchy_file: str):
        """Initialize classifier with API credentials and hierarchy data"""
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_name = model_name
        self.hierarchy = self._load_hierarchy(hierarchy_file)

    def _load_hierarchy(self, hierarchy_file: str) -> Dict[str, Any]:
        """Load harm classification hierarchy from JSON file"""
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_risk_areas(self) -> List[str]:
        """Get top-level risk area categories"""
        return list(self.hierarchy.keys())

    def _get_types_of_harm(self, risk_area: str) -> List[str]:
        """Get harm types for a specific risk area"""
        return list(self.hierarchy[risk_area]["types_of_harm"].keys())

    def _get_specific_harms(self, risk_area: str, type_of_harm: str) -> List[str]:
        """Get specific harm categories for a given risk area and harm type"""
        return list(self.hierarchy[risk_area]["types_of_harm"][type_of_harm]["specific_harms"].keys())

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API for classification task"""
        system_prompt = """分類タスクです。番号付きの選択肢から最も適切な番号を1つだけ返してください。数字のみ出力。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            logging.error(f"API error: {e}")
            return None

    def _classify_step(self, question: str, choices: List[str], step_name: str) -> Optional[str]:
        """Perform single classification step with numbered choices"""
        choices_text = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(choices))
        prompt = f"質問: {question}\n\n選択肢:\n{choices_text}\n\n最適な番号を選択:"

        # Try up to 3 times to get valid response
        for attempt in range(3):
            result = self._call_llm(prompt)
            if result:
                match = re.search(r"(\d+)", result)
                if match:
                    try:
                        index = int(match.group(1))
                        if 1 <= index <= len(choices):
                            return choices[index - 1]
                    except ValueError:
                        pass
        return None

    def classify_question(self, question: str) -> Dict[str, str]:
        """Classify question through 3-level hierarchy"""
        # Step 1: Classify risk area
        risk_area = self._classify_step(question, self._get_risk_areas(), "risk_area")
        if not risk_area:
            return {"risk_area": "ERROR", "type_of_harm": "SKIP", "specific_harm": "SKIP", "specific_harm_key": "SKIP"}

        # Step 2: Classify type of harm within risk area
        type_of_harm = self._classify_step(question, self._get_types_of_harm(risk_area), "type")
        if not type_of_harm:
            return {"risk_area": risk_area, "type_of_harm": "ERROR", "specific_harm": "SKIP", "specific_harm_key": "SKIP"}

        # Step 3: Classify specific harm within type
        specific_harm_key = self._classify_step(question, self._get_specific_harms(risk_area, type_of_harm), "specific")
        if not specific_harm_key:
            return {"risk_area": risk_area, "type_of_harm": type_of_harm, "specific_harm": "ERROR", "specific_harm_key": "ERROR"}

        # Build full specific harm description
        specific_harm_desc = self.hierarchy[risk_area]["types_of_harm"][type_of_harm]["specific_harms"][specific_harm_key]
        return {
            "risk_area": risk_area,
            "type_of_harm": type_of_harm,
            "specific_harm": f"{specific_harm_key}: {specific_harm_desc}",
            "specific_harm_key": specific_harm_key
        }



def setup_logging(log_dir: str, start_index: int, end_index: int):
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/classify_{start_index}-{end_index}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                       handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def load_existing_ids(output_file: str) -> set:
    existing_ids = set()
    if os.path.exists(output_file):
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
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3-8b-instruct")
    parser.add_argument("--hierarchy_file", type=str, default="dna_hierarchy.json")
    parser.add_argument("--log_dir", type=str, default="logs")
    
    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = f"classified_{args.start_index}-{args.end_index-1}.jsonl"
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return
    
    setup_logging(args.log_dir, args.start_index, args.end_index)
    
    try:
        dataset = load_dataset("neko-llm/DNA_DPO_hh-rlhf", split="train")
        classifier = HarmClassifier(api_key, args.model_name, args.hierarchy_file)
    existing_ids = load_existing_ids(args.output_file)
    
    processed_count = 0
    with open(args.output_file, 'a', encoding='utf-8') as f:
            for i, item in enumerate(tqdm(dataset, desc="Classifying")):
                if i < args.start_index or i >= args.end_index:
                continue
                if item['id'] in existing_ids:
                continue
            
            classification = classifier.classify_question(str(item['question']))
            result = {
                    "id": item['id'],
                "question": str(item['question']),
                "preferred_output": str(item['preferred_output']), 
                "non_preferred_output": str(item['non_preferred_output']),
                **classification
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            processed_count += 1
        
        logging.info(f"Processed {processed_count} items")
        
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
