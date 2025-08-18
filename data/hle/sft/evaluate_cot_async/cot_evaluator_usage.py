#!/usr/bin/env python3
"""
Example usage of the CoTEvaluationProcessor class

This script demonstrates how to use the CoTEvaluationProcessor class
to evaluate CoT (Chain of Thought) quality in your data processing scripts.
"""

import json
from pathlib import Path
from cot_evaluator import CoTEvaluationProcessor

async def main():
    """Example usage of the CoTEvaluationProcessor class"""
    
    # Initialize the processor
    # You can customize the API key, model, and other parameters
    processor = CoTEvaluationProcessor(
        # api_key="your_api_key_here",  # Optional: will use environment variable if not provided
        # model="deepseek/deepseek-r1-0528:free",  # Optional: will use default if not provided
        # temperature=0.0,  # Optional: will use default if not provided
        # max_tokens=8000,  # Optional: will use default if not provided
    )
    
    # Example 1: Evaluate a single item
    print("=== Example 1: Evaluating a single item ===")
    
    sample_item = {
        "id": "example_1",
        "question": "What is 2 + 2?",
        "output": "<think>I need to add 2 and 2. This is a simple arithmetic problem. 2 + 2 = 4.</think>",
        "answer": "4"
    }
    
    try:
        result = await processor.evaluate_single_item(sample_item)
        print(f"Grade: {result['grade']}")
        print(f"Score: {result['score']:.2f}/10")
        print(f"Evaluation success: {result['evaluation'].get('evaluation_success', True)}")
    except Exception as e:
        print(f"Error evaluating single item: {e}")
    
    # Example 2: Evaluate a dataset
    print("\n=== Example 2: Evaluating a dataset ===")
    
    # Create a sample dataset
    sample_dataset = [
        {
            "id": "1",
            "question": "What is 3 + 5?",
            "output": "<think>I need to add 3 and 5. This is basic addition. 3 + 5 = 8.</think>",
            "answer": "8"
        },
        {
            "id": "2", 
            "question": "What is 10 - 4?",
            "output": "<think>I need to subtract 4 from 10. 10 - 4 = 6.</think>",
            "answer": "6"
        }
    ]
    
    # Save sample dataset to file
    sample_file = Path("sample_dataset.jsonl")
    with open(sample_file, 'w') as f:
        for item in sample_dataset:
            json.dump(item, f)
            f.write('\n')
    
    try:
        # Evaluate the dataset
        stats = await processor.evaluate_dataset(
            dataset_path=str(sample_file),
            output_file="./results/evaluated_sample_dataset.jsonl",
            # ids="1,2",  # Optional: evaluate specific IDs
            # evaluator_models="deepseek/deepseek-r1-0528:free",  # Optional: specify models
            # eval_concurrency=4,  # Optional: specify concurrency
            # output_format="jsonl",  # Optional: specify output format
            # skip_existing=True,  # Optional: skip already evaluated items
        )
        
        print(f"Evaluation completed!")
        print(f"Evaluated: {stats['evaluated_count']} items")
        print(f"Skipped: {stats['skipped_count']} items")
        print(f"Average score: {stats['average_score']:.2f}/10")
        print(f"Grade distribution: {stats['grade_counts']}")
        print(f"Results saved to: {stats['output_file']}")
        
    except Exception as e:
        print(f"Error evaluating dataset: {e}")
    
    # Example 3: Using multiple models for evaluation
    print("\n=== Example 3: Using multiple models ===")
    
    try:
        # This would use multiple models for more robust evaluation
        # Note: This requires multiple model access and may incur additional costs
        multi_model_stats = await processor.evaluate_dataset(
            dataset_path=str(sample_file),
            output_file="./results/multi_model_evaluated.jsonl",
            evaluator_models="deepseek/deepseek-r1-0528:free,moonshotai/kimi-k2:free",  # Multiple models
            eval_concurrency=2,  # Lower concurrency for multiple models
        )
        
        print(f"Multi-model evaluation completed!")
        print(f"Results saved to: {multi_model_stats['output_file']}")
        
    except Exception as e:
        print(f"Error with multi-model evaluation: {e}")
        print("This might be due to model availability or API access")
    
    # Clean up sample files
    if sample_file.exists():
        sample_file.unlink()
    
    print("\n=== Example completed! ===")
    print("You can now use CoTEvaluationProcessor in your own scripts!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
