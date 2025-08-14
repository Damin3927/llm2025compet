#!/usr/bin/env python3
"""
Example usage of the CoTRegenerator module.

This script demonstrates various ways to use the CoTRegenerator class for
regenerating Chain of Thought reasoning with single and multiple models.

Usage:
    python regenerate_example.py
"""

import asyncio
import json
from pathlib import Path
from cot_regenerator import CoTRegenerator, ModelConfig
from cot_evaluator import CoTEvaluator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_1_basic_single_model():
    """Example 1: Basic single model regeneration."""
    print("\n" + "="*60)
    print("Example 1: Basic Single Model Regeneration")
    print("="*60)
    
    # Initialize regenerator with default model
    regenerator = CoTRegenerator()
    
    # Sample data
    question = "What is 15% of 240?"
    answer = "36"
    previous_cot = """<think>
To find 15% of 240, I'll multiply 240 by 0.15.
240 × 0.15 = 36
</think>"""
    
    evaluation_details = {
        "grade": "C",
        "strengths": ["Correct calculation"],
        "weaknesses": [
            "No explanation of percentage concept",
            "No verification of result",
            "Missing alternative methods"
        ],
        "improvement_suggestions": [
            "Explain what percentage means",
            "Show multiple solution methods",
            "Verify the answer"
        ],
        "learning_value_scores": {
            "method_explanation": 3,
            "step_by_step": 4,
            "verification": 2,
            "common_mistakes": 2,
            "domain_insight": 3,
            "metacognitive": 2
        }
    }
    
    # Regenerate
    new_cot = regenerator.regenerate_single(
        question, answer, previous_cot, evaluation_details
    )
    
    if new_cot:
        print(f"\nOriginal CoT Grade: {evaluation_details['grade']}")
        print(f"\nRegenerated CoT:\n{new_cot[:500]}...")  # Show first 500 chars
        print("\n✅ Regeneration successful!")
    else:
        print("\n❌ Regeneration failed!")


def example_2_custom_models():
    """Example 2: Using custom model configurations."""
    print("\n" + "="*60)
    print("Example 2: Custom Model Configurations")
    print("="*60)
    
    # Create custom model configurations
    models = [
        ModelConfig(
            model_name="deepseek/deepseek-r1-0528:free",
            temperature=0.3,
            max_tokens=30000
        ),
        ModelConfig(
            model_name="meta-llama/llama-3-8b-instruct:free",
            temperature=0.2,
            max_tokens=25000
        )
    ]
    
    # Initialize regenerator with custom models
    regenerator = CoTRegenerator(models=models)
    
    print(f"Configured models:")
    for config in regenerator.model_configs:
        print(f"  - {config.model_name} (temp={config.temperature})")
    
    print("\n✅ Custom models configured successfully!")


async def example_3_async_multi_model():
    """Example 3: Async multi-model regeneration with best selection."""
    print("\n" + "="*60)
    print("Example 3: Async Multi-Model Regeneration")
    print("="*60)
    
    # Initialize with multiple models
    models = [
        "deepseek/deepseek-r1-0528:free",
        "meta-llama/llama-3-8b-instruct:free",
        "google/gemma-2-9b-it:free"
    ]
    
    regenerator = CoTRegenerator(models=models)
    
    # Sample problem
    question = "If a train travels 120 miles in 2 hours, what is its average speed?"
    answer = "60 mph"
    previous_cot = "The train goes 120 miles in 2 hours, so speed is 120/2 = 60 mph."
    
    evaluation_details = {
        "grade": "C",
        "strengths": ["Correct calculation"],
        "weaknesses": [
            "No explanation of average speed concept",
            "No units explanation",
            "No verification"
        ],
        "improvement_suggestions": [
            "Explain the formula for average speed",
            "Discuss units",
            "Verify the answer makes sense"
        ],
        "learning_value_scores": {
            "method_explanation": 3,
            "step_by_step": 3,
            "verification": 1,
            "common_mistakes": 2,
            "domain_insight": 2,
            "metacognitive": 2
        }
    }
    
    print(f"Testing with {len(models)} models in parallel...")
    print(f"Models: {', '.join(models)}")
    
    # Regenerate with all models and select best
    result = await regenerator.regenerate_multi_async(
        question, answer, previous_cot, evaluation_details
    )
    
    if result:
        print(f"\n✅ Best model: {result['best_model']}")
        print(f"Best grade: {result['best_grade']}")
        print(f"Best score: {result['best_score']:.2f}")
        print(f"Models tried: {', '.join(result['all_models_tried'])}")
        print(f"\nBest regenerated CoT (first 400 chars):\n{result['best_cot'][:400]}...")
    else:
        print("\n❌ All models failed!")


async def example_4_get_all_results():
    """Example 4: Get all regeneration results for comparison."""
    print("\n" + "="*60)
    print("Example 4: Get All Regeneration Results")
    print("="*60)
    
    models = [
        "deepseek/deepseek-r1-0528:free",
        "meta-llama/llama-3-8b-instruct:free"
    ]
    
    regenerator = CoTRegenerator(models=models)
    
    question = "What is 25% of 80?"
    answer = "20"
    previous_cot = "25% of 80 = 80 * 0.25 = 20"
    
    evaluation_details = {
        "grade": "D",
        "strengths": ["Correct answer"],
        "weaknesses": [
            "Too brief",
            "No explanation",
            "No verification"
        ],
        "improvement_suggestions": [
            "Explain percentage calculation",
            "Show alternative methods"
        ],
        "learning_value_scores": {
            "method_explanation": 1,
            "step_by_step": 2,
            "verification": 0,
            "common_mistakes": 0,
            "domain_insight": 1,
            "metacognitive": 0
        }
    }
    
    # Get all results
    all_results = await regenerator.regenerate_multi_async(
        question, answer, previous_cot, evaluation_details,
        return_all=True
    )
    
    if all_results:
        print(f"\nReceived {len(all_results)} results:")
        for i, result in enumerate(all_results, 1):
            print(f"\n{i}. Model: {result['model']}")
            print(f"   Grade: {result.get('grade', 'N/A')}")
            print(f"   Score: {result.get('score', 0):.2f}")
            print(f"   CoT preview: {result['cot'][:100]}...")
    else:
        print("\n❌ No results received!")


def example_5_dataset_regeneration():
    """Example 5: Regenerate an entire dataset."""
    print("\n" + "="*60)
    print("Example 5: Dataset Regeneration")
    print("="*60)
    
    # Check if sample dataset exists
    sample_dataset = Path("results/evaluated_sample_dataset.jsonl")
    
    if not sample_dataset.exists():
        print(f"Sample dataset not found at {sample_dataset}")
        print("Creating a minimal sample dataset...")
        
        # Create sample data
        sample_data = [
            {
                "id": "1",
                "question": "What is 10% of 50?",
                "answer": "5",
                "output": "<think>10% of 50 = 50 * 0.1 = 5</think>5",
                "metadata": {
                    "cot_history": [{
                        "timestamp": "2024-01-01T00:00:00",
                        "output": "<think>10% of 50 = 50 * 0.1 = 5</think>5",
                        "evaluation": {
                            "grade": "C",
                            "strengths": ["Correct"],
                            "weaknesses": ["Too brief"],
                            "improvement_suggestions": ["Add explanation"],
                            "learning_value_scores": {
                                "method_explanation": 2,
                                "step_by_step": 3,
                                "verification": 1,
                                "common_mistakes": 1,
                                "domain_insight": 2,
                                "metacognitive": 1
                            }
                        }
                    }]
                }
            }
        ]
        
        # Save sample dataset
        sample_dataset.parent.mkdir(exist_ok=True)
        with open(sample_dataset, 'w') as f:
            for item in sample_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    # Initialize regenerator
    regenerator = CoTRegenerator(
        models=["deepseek/deepseek-r1-0528:free"]
    )
    
    output_path = Path("results/regenerated_dataset.jsonl")
    
    print(f"Input dataset: {sample_dataset}")
    print(f"Output dataset: {output_path}")
    print("Regenerating items with grade C or below...")
    
    # Regenerate dataset
    stats = regenerator.regenerate_dataset(
        dataset_path=sample_dataset,
        output_path=output_path,
        grade_threshold="C",
        use_async=False  # Use single model for simplicity
    )
    
    print(f"\n✅ Dataset regeneration complete!")
    print(f"Statistics:")
    print(f"  - Total items: {stats['total_items']}")
    print(f"  - Items regenerated: {stats['items_to_regenerate']}")
    print(f"  - Successful: {stats['successful']}")
    print(f"  - Failed: {stats['failed']}")
    print(f"  - Improved: {stats['improved']}")


def example_6_integration_with_existing_code():
    """Example 6: Using the backwards-compatible function."""
    print("\n" + "="*60)
    print("Example 6: Backwards Compatibility")
    print("="*60)
    
    # Import the legacy function
    from cot_regenerator import regenerate_cot
    import openai
    
    # This mimics the old usage pattern
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set. Skipping this example.")
        return
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    question = "What is 5 + 3?"
    answer = "8"
    previous_cot = "5 + 3 = 8"
    
    evaluation_details = {
        "grade": "D",
        "strengths": ["Correct"],
        "weaknesses": ["No explanation"],
        "improvement_suggestions": ["Explain addition"],
        "learning_value_scores": {}
    }
    
    # Use the legacy function
    new_cot = regenerate_cot(
        client, question, answer, previous_cot, evaluation_details
    )
    
    if new_cot:
        print("✅ Legacy function works!")
        print(f"Regenerated CoT preview: {new_cot[:200]}...")
    else:
        print("❌ Legacy function failed!")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CoT Regenerator Module - Usage Examples")
    print("="*60)
    
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  Warning: OPENROUTER_API_KEY not set in environment.")
        print("Please set it in .env file or as environment variable.")
        print("Some examples may fail without it.")
    
    # Run synchronous examples
    example_1_basic_single_model()
    example_2_custom_models()
    
    # Run async examples
    print("\nRunning async examples...")
    asyncio.run(example_3_async_multi_model())
    asyncio.run(example_4_get_all_results())
    
    # Run dataset example
    example_5_dataset_regeneration()
    
    # Run compatibility example
    example_6_integration_with_existing_code()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()