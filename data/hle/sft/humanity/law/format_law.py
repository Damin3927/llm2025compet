#!/usr/bin/env python3
"""
Script to load and filter the moremilk/CoT_Legal_Issues_And_Laws dataset by difficulty.
"""

import os
import json
import argparse
from datasets import load_dataset
import pandas as pd
from typing import List, Optional, Dict


def load_legal_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the moremilk/CoT_Legal_Issues_And_Laws dataset from Hugging Face.
    
    Args:
        split: Dataset split to load ('train', 'test', 'validation')
        
    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading dataset: moremilk/CoT_Legal_Issues_And_Laws (split: {split})")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("moremilk/CoT_Legal_Issues_And_Laws", split=split)
        
        # Convert to pandas DataFrame for easier manipulation
        df = dataset.to_pandas()
        
        print(f"Successfully loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        # Display unique values in difficulty column if it exists
        df['difficulty'] = df['metadata'].apply(lambda x: int(x['difficulty']))

        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def filter_by_difficulty(df: pd.DataFrame, difficulty_value: int, num_samples: int) -> pd.DataFrame:
    """
    Filter the dataset by difficulty values.
    
    Args:
        df: Input DataFrame
        difficulty_values: List of difficulty values to include
        num_samples: max number of samples to select
    Returns:
        Filtered DataFrame
    """
    if 'difficulty' not in df.columns:
        print("Error: 'difficulty' column not found in dataset")
        return df
    
    # Filter by difficulty
    filtered_df = df.loc[df['difficulty'] >= difficulty_value]
    filtered_df = filtered_df.sample(n=num_samples)
    
    print(f"Filtered dataset: {len(filtered_df)} samples (from {len(df)} total)")
    print(f"Selected difficulty >= {difficulty_value}")
    print(f"Filtered difficulty distribution:\n{filtered_df['difficulty'].value_counts()}")
    
    return filtered_df

def transfer_data_format(df: pd.DataFrame) -> Dict:
    """
    Transfer the data format to the format of the SFT dataset.
    The format of the SFT dataset is:
    {
        "id": "law_{index}",
        "question": "What is the capital of France?",
        "output": "<think>Reasoning</think>\n<answer>Answer</answer>",
        "answer": "Answer"
    }

    """
    results = []
    for index, row in df.iterrows():
        id = f"law_{index}"
        question = row["question"]
        output = f"<think>{row['metadata']['reasoning']}</think>\n{row['answer']}"
        answer = row["answer"]
        results.append({
            "id": id,
            "question": question,
            "output": output,
            "answer": answer
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Filter legal dataset by difficulty")
    parser.add_argument("--split", default="train", choices=["train"],
                       help="Dataset split to load")
    parser.add_argument("--difficulty", type=int, default=4,
                       help="Difficulty higher than the values will be included")
    parser.add_argument("--output", default="../results/law",
                       help="Output file path")
    parser.add_argument("--show-sample", action="store_true",
                       help="Show a sample of the filtered data")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of samples to select")
    args = parser.parse_args()
    
    # Load the dataset
    df = load_legal_dataset(args.split)
    
    if df is None:
        return
    
    # Filter by difficulty
    filtered_df = filter_by_difficulty(df, args.difficulty, args.num_samples)

    # transfer data
    transferred_data = transfer_data_format(filtered_df)
    
    # Show sample if requested
    if args.show_sample and len(filtered_df) > 0:
        print("\nSample of filtered data:")
        print(filtered_df.head())
    
    # Save filtered data
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "law.json"), "w") as f:
        json.dump(transferred_data, f, indent=4)
        
    return transferred_data


if __name__ == "__main__":
    main()
