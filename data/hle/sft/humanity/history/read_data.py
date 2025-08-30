#!/usr/bin/env python3
"""
Script to load and filter the nielsprovos/world-history-1500-qa dataset by difficulty.
"""

import os
import json
import argparse
from datasets import load_dataset
import pandas as pd
from typing import List, Optional, Dict


def load_history_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the nielsprovos/world-history-1500-qa dataset from Hugging Face.
    
    Args:
        split: Dataset split to load ('train', 'test', 'validation')
        
    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading dataset: nielsprovos/world-history-1500-qa (split: {split})")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("nielsprovos/world-history-1500-qa", split=split)
        
        # Convert to pandas DataFrame for easier manipulation
        df = dataset.to_pandas()
        
        print(f"Successfully loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")

        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def transfer_data_format(df: pd.DataFrame) -> Dict:
    """
    Transfer the data format to json
    """
    results = []
    for row in df.iloc[0]["qa_pairs"]:
        question = row["question"]
        answer = row["answer"]
        results.append({
            "question": question,
            "answer": answer
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Filter legal dataset by difficulty")
    parser.add_argument("--split", default="train", choices=["train"],
                       help="Dataset split to load")
    parser.add_argument("--difficulty", type=int, default=4,
                       help="Difficulty higher than the values will be included")
    parser.add_argument("--output", default="./results",
                       help="Output file path")
    parser.add_argument("--show-sample", action="store_true",
                       help="Show a sample of the filtered data")
    
    args = parser.parse_args()
    
    # Load the dataset
    df = load_history_dataset(args.split)
    
    if df is None:
        return

    # transfer data
    transferred_data = transfer_data_format(df)
    
    # Show sample if requested
    if args.show_sample and len(df) > 0:
        print("\nSample of filtered data:")
        print(df.head())
    
    # Save filtered data
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(transferred_data, f, indent=4)
        
    return transferred_data


if __name__ == "__main__":
    main()
