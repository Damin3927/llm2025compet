#!/usr/bin/env python3
"""
Mixture of Thoughts Dataset Processing Script

This script processes the Mixture of Thoughts dataset splits (code, math, science)
and converts them into structured JSON files with question, solution, and answer fields.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


def extract_solution_and_answer(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract solution from <think>...</think> tags and answer from content after </think>.
    
    Args:
        content: Assistant message content
        
    Returns:
        Tuple of (solution, answer) where either can be None if not found
    """
    # Look for <think>...</think> block
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, content, re.DOTALL)
    
    solution = None
    answer = None
    
    if think_match:
        solution = think_match.group(1).strip()
        # Extract content after </think> tag
        think_end_pattern = r'</think>\s*(.*)'
        answer_match = re.search(think_end_pattern, content, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        solution = f"<think>{solution}</think>\n\n{answer}"
    return solution, answer


def process_conversation(messages: List[Dict], split_name: str, index: int) -> Optional[Dict]:
    """
    Process a single conversation into the required format.
    
    Args:
        messages: List of message dictionaries
        split_name: Name of the dataset split (code, math, science)
        index: Row index in the split
        
    Returns:
        Processed conversation dict or None if invalid
    """
    # Validate we have at least one user and one assistant message
    if len(messages) < 2:
        return None
    
    user_msg = None
    assistant_msg = None
    
    # Find first user and assistant messages
    for msg in messages:
        if msg.get('role') == 'user' and user_msg is None:
            user_msg = msg
        elif msg.get('role') == 'assistant' and assistant_msg is None:
            assistant_msg = msg
        
        # Stop once we have both
        if user_msg and assistant_msg:
            break
    
    # Skip if either is missing
    if not user_msg or not assistant_msg:
        return None
    
    question = user_msg.get('content', '').strip()
    assistant_content = assistant_msg.get('content', '')
    
    solution, answer = extract_solution_and_answer(assistant_content)
    
    return {
        'id': f'MoT_{split_name}_{index}',
        'question': question,
        'output': solution,
        'answer': answer
    }


def process_split(split_name: str, output_dir: Path) -> int:
    """
    Process a single dataset split.
    
    Args:
        split_name: Name of the split to process
        output_dir: Base output directory
        
    Returns:
        Number of conversations processed
    """
    print(f"Processing {split_name} split...")
    
    # Load the dataset split
    dataset = load_dataset("open-r1/Mixture-of-Thoughts", split_name, split="train")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_conversations = []
    processed_count = 0

    for index, row in enumerate(dataset):
        # Handle case where messages might be a string (JSON) or already parsed
        messages_raw = row.get('messages', [])

        if isinstance(messages_raw, str):
            try:
                messages = json.loads(messages_raw)
            except json.JSONDecodeError:
                print(f"Skipped {split_name} row {index}: invalid JSON in messages field")
                continue
        else:
            messages = messages_raw
        
        # Process the conversation
        processed_conv = process_conversation(messages, split_name, index)
        
        if processed_conv:
            processed_conversations.append(processed_conv)
            processed_count += 1
        else:
            print(f"Skipped {split_name} row {index}: invalid conversation structure")
    
    # Save all conversations to a single JSON file
    filename = f"MoT_{split_name}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(processed_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {processed_count} conversations from {split_name} split")
    print(f"Saved to: {filepath}")
    return processed_count


def main():
    """Main processing function."""
    # Define splits to process (excluding 'total')
    splits = ['code', 'math', 'science']
    
    # Set output directory
    output_dir = Path('./processed_mot_data')
    output_dir.mkdir(exist_ok=True)
    
    total_processed = 0
    
    for split_name in splits:
        split_path = output_dir / split_name
        split_path.mkdir(exist_ok=True)
        try:
            count = process_split(split_name, split_path)
            total_processed += count
        except Exception as e:
            print(f"Error processing split {split_name}: {e}")
            continue
    
    print(f"\nTotal conversations processed: {total_processed}")
    print(f"Output saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()