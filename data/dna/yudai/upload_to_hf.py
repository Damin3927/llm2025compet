#!/usr/bin/env python3
"""
Hugging Face dataset upload script
Uploads processed JSONL files to Hugging Face Hub
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login

load_dotenv()


class HuggingFaceUploader:
    def __init__(self, api_key: str, dataset_name: str):
        """Initialize uploader with API credentials"""
        self.api_key = api_key
        self.dataset_name = dataset_name
        
        try:
            login(token=api_key)
            logging.info("Logged in to Hugging Face")
        except Exception as e:
            logging.error(f"Login error: {e}")
            raise
    
    def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return data
    
    def _merge_files(self, file_patterns: List[str]) -> List[Dict[str, Any]]:
        """Merge multiple JSONL files matching patterns"""
        all_data = []
        
        for pattern in file_patterns:
            if "*" in pattern:
                import glob
                matching_files = glob.glob(pattern)
            else:
                matching_files = [pattern] if Path(pattern).exists() else []
            
            for file_path in matching_files:
                try:
                    data = self._load_jsonl_file(file_path)
                    all_data.extend(data)
                    logging.info(f"Loaded {len(data)} items from {file_path}")
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {e}")
        
        return all_data
    
    def upload_data(self, file_patterns: List[str], commit_message: Optional[str] = None) -> bool:
        """Upload merged data to Hugging Face Hub"""
        try:
            data = self._merge_files(file_patterns)
            
            if not data:
                logging.error("No data found")
                return False
            
            # Create dataset and upload
            dataset = Dataset.from_list(data)
            
            if not commit_message:
                commit_message = f"Update dataset - {len(data)} items"
            
            dataset.push_to_hub(self.dataset_name, commit_message=commit_message, private=False)
            logging.info(f"Uploaded {len(data)} items to {self.dataset_name}")
            return True
            
        except Exception as e:
            logging.error(f"Upload error: {e}")
            return False


def main():
    """Main CLI interface for dataset upload"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--dataset-name", type=str, default="neko-llm/dna_dpo_hh-rlhf")
    parser.add_argument("--commit-message", type=str)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("Error: HUGGINGFACE_API_KEY not set")
        return 1
    
    # Verify files exist before starting upload
    existing_files = []
    for pattern in args.files:
        if "*" in pattern:
            import glob
            matching_files = glob.glob(pattern)
        else:
            matching_files = [pattern] if Path(pattern).exists() else []
        existing_files.extend(matching_files)
    
    if not existing_files:
        print("No files found")
        return 1
    
    try:
        uploader = HuggingFaceUploader(api_key, args.dataset_name)
        success = uploader.upload_data(args.files, args.commit_message)
        
        if success:
            print(f"✅ Upload completed: https://huggingface.co/datasets/{args.dataset_name}")
            return 0
        else:
            print("❌ Upload failed")
            return 1
            
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
