#!/usr/bin/env python3
"""
Comprehensive usage examples for the CoTRegenerator module.

This script demonstrates practical use cases for regenerating Chain of Thought
reasoning, including single model, multi-model async, and dataset processing.

Usage:
    python cot_regenerator_usage.py
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

# Import the regenerator and evaluator
from cot_regenerator import CoTRegenerator, ModelConfig
from cot_evaluator import CoTEvaluator, CoTEvaluationProcessor

# For demonstration purposes
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# EXAMPLE 1: Simple Single Model Regeneration
# ============================================================================

def example_1_simple_regeneration():
    """
    Most basic usage - regenerate a single CoT with default model.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Single Model Regeneration")
    print("="*70)
    
    # Initialize with default settings
    regenerator = CoTRegenerator()
    
    # Problem that needs better explanation
    question = "A car travels 120 miles in 2 hours. What is its average speed?"
    answer = "60 mph"
    
    # Original CoT (too brief)
    previous_cot = "Speed = distance/time = 120/2 = 60 mph"
    
    # Evaluation feedback
    evaluation_details = {
        "grade": "C",
        "strengths": ["Correct formula", "Correct calculation"],
        "weaknesses": [
            "No explanation of average speed concept",
            "No unit analysis",
            "No verification of answer"
        ],
        "improvement_suggestions": [
            "Explain what average speed means",
            "Show unit conversions explicitly",
            "Verify the answer makes sense"
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
    
    print(f"\nOriginal CoT (Grade {evaluation_details['grade']}):")
    print(f"  '{previous_cot}'")
    print(f"\nWeaknesses identified:")
    for weakness in evaluation_details['weaknesses']:
        print(f"  - {weakness}")
    
    print("\nRegenerating with improvements...")
    
    # Regenerate
    new_cot = regenerator.regenerate_single(
        question=question,
        answer=answer,
        previous_cot=previous_cot,
        evaluation_details=evaluation_details
    )
    
    if new_cot:
        print(f"\n✅ Regeneration successful!")
        print(f"\nImproved CoT:")
        print("-" * 50)
        # Show first 500 chars for brevity
        preview = new_cot[:500] + "..." if len(new_cot) > 500 else new_cot
        print(preview)
        print("-" * 50)
    else:
        print("\n❌ Regeneration failed")


# ============================================================================
# EXAMPLE 2: Multi-Model Async Regeneration with Comparison
# ============================================================================

async def example_2_multi_model_comparison():
    """
    Use multiple models to regenerate and compare their outputs.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Model Async Regeneration with Comparison")
    print("="*70)
    
    # Configure multiple models (only use available ones)
    models = [
        "deepseek/deepseek-r1-0528:free",
        "z-ai/glm-4.5-air:free"
    ]
    
    regenerator = CoTRegenerator(models=models)
    
    # Math problem needing improvement
    question = "What is 25% of 80?"
    answer = "20"
    previous_cot = "25% = 0.25, so 0.25 × 80 = 20"
    
    evaluation_details = {
        "grade": "D",
        "strengths": ["Correct answer"],
        "weaknesses": [
            "No explanation of percentage concept",
            "No alternative methods shown",
            "Too brief for learning",
            "No verification"
        ],
        "improvement_suggestions": [
            "Explain what 25% means conceptually",
            "Show multiple solution methods",
            "Add verification step"
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
    
    print(f"\nTesting {len(models)} models in parallel...")
    print(f"Original CoT Grade: {evaluation_details['grade']}")
    
    # Get all results to compare
    all_results = await regenerator.regenerate_multi_async(
        question=question,
        answer=answer,
        previous_cot=previous_cot,
        evaluation_details=evaluation_details,
        return_all=True  # Get all results for comparison
    )
    
    if all_results:
        print(f"\n📊 Comparison of {len(all_results)} regenerated CoTs:")
        print("-" * 50)
        
        for i, result in enumerate(all_results, 1):
            print(f"\n{i}. Model: {result['model']}")
            print(f"   Grade: {result.get('grade', 'N/A')}")
            print(f"   Score: {result.get('score', 0):.2f}/10")
            
            # Show snippet of CoT
            cot_preview = result['cot'][:150] + "..." if len(result['cot']) > 150 else result['cot']
            print(f"   Preview: {cot_preview}")
        
        # Best result
        print("\n" + "="*50)
        print(f"🏆 Best Result: {all_results[0]['model']}")
        print(f"   Grade: {all_results[0].get('grade', 'N/A')}")
        print(f"   Score: {all_results[0].get('score', 0):.2f}/10")
    else:
        print("\n❌ All models failed to regenerate")


# ============================================================================
# EXAMPLE 3: Custom Model Configurations
# ============================================================================

def example_3_custom_configurations():
    """
    Configure models with different parameters for specific use cases.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Model Configurations")
    print("="*70)
    
    # Create custom configurations for different purposes
    models = [
        # More creative responses
        ModelConfig(
            model_name="deepseek/deepseek-r1-0528:free",
            temperature=0.5,  # Higher temperature for creativity
            max_tokens=50000,
            retry_attempts=3
        ),

        # Balanced approach
        ModelConfig(
            model_name="google/gemma-2-9b-it:free",
            temperature=0.3,
            max_tokens=40000,
            retry_attempts=3
        )
    ]
    
    regenerator = CoTRegenerator(models=models)
    
    print("\nConfigured models:")
    for i, config in enumerate(regenerator.model_configs, 1):
        print(f"{i}. {config.model_name}")
        print(f"   - Temperature: {config.temperature}")
        print(f"   - Max Tokens: {config.max_tokens}")
        print(f"   - Retry Attempts: {config.retry_attempts}")
    
    print("\n✅ Custom configurations ready for use!")


# ============================================================================
# EXAMPLE 4: Batch Dataset Processing
# ============================================================================

def example_4_dataset_processing():
    """
    Process an entire dataset, regenerating low-quality CoTs.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Dataset Processing")
    print("="*70)
    
    # Create sample dataset
    sample_data = [
        {
            "id": "1",
            "question": "What is 10% of 50?",
            "answer": "5",
            "output": "<think>10% of 50 = 50 × 0.1 = 5</think>5",
            "metadata": {
                "cot_history": [{
                    "timestamp": "2024-01-01T00:00:00",
                    "output": "<think>10% of 50 = 50 × 0.1 = 5</think>5",
                    "evaluation": {
                        "grade": "C",
                        "strengths": ["Correct calculation"],
                        "weaknesses": ["No explanation", "Too brief"],
                        "improvement_suggestions": ["Explain percentage", "Add verification"],
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
        },
        {
            "id": "2",
            "question": "What is 2 + 2?",
            "answer": "4",
            "output": "<think>Adding 2 + 2: First, I recognize this as a basic addition problem. When we add 2 + 2, we're combining two groups of 2 items. This gives us 4 total items.</think>4",
            "metadata": {
                "cot_history": [{
                    "timestamp": "2024-01-01T00:00:00",
                    "output": "<think>Adding 2 + 2: First, I recognize this as a basic addition problem. When we add 2 + 2, we're combining two groups of 2 items. This gives us 4 total items.</think>4",
                    "evaluation": {
                        "grade": "B",
                        "strengths": ["Good explanation", "Clear reasoning"],
                        "weaknesses": [],
                        "improvement_suggestions": [],
                        "learning_value_scores": {
                            "method_explanation": 7,
                            "step_by_step": 7,
                            "verification": 5,
                            "common_mistakes": 5,
                            "domain_insight": 6,
                            "metacognitive": 6
                        }
                    }
                }]
            }
        },
        {
            "id": "3",
            "question": "What is 15 × 4?",
            "answer": "60",
            "output": "<think>15 × 4 = 60</think>60",
            "metadata": {
                "cot_history": [{
                    "timestamp": "2024-01-01T00:00:00",
                    "output": "<think>15 × 4 = 60</think>60",
                    "evaluation": {
                        "grade": "D",
                        "strengths": ["Correct answer"],
                        "weaknesses": ["No process shown", "No explanation", "No verification"],
                        "improvement_suggestions": ["Show multiplication steps", "Explain method", "Verify result"],
                        "learning_value_scores": {
                            "method_explanation": 0,
                            "step_by_step": 1,
                            "verification": 0,
                            "common_mistakes": 0,
                            "domain_insight": 0,
                            "metacognitive": 0
                        }
                    }
                }]
            }
        }
    ]
    
    # Save sample dataset
    dataset_path = Path("sample_dataset_for_regeneration.jsonl")
    with open(dataset_path, 'w') as f:
        for item in sample_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Created sample dataset: {dataset_path}")
    print(f"Total items: {len(sample_data)}")
    
    # Analyze grades
    grade_counts = {}
    for item in sample_data:
        grade = item['metadata']['cot_history'][0]['evaluation']['grade']
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    
    print("\nGrade distribution:")
    for grade in sorted(grade_counts.keys()):
        print(f"  Grade {grade}: {grade_counts[grade]} items")
    
    # Initialize regenerator
    regenerator = CoTRegenerator(
        models=["deepseek/deepseek-r1-0528:free"]  # Single model for speed
    )
    
    output_path = Path("regenerated_dataset.jsonl")
    
    print(f"\nRegenerating items with grade C or below...")
    print(f"Output will be saved to: {output_path}")
    
    # Process dataset
    stats = regenerator.regenerate_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        grade_threshold="C",  # Regenerate C and below
        use_async=False  # Single model, no need for async
    )
    
    print("\n📊 Regeneration Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Items to regenerate: {stats['items_to_regenerate']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Improved: {stats['improved']}")
    
    # Clean up
    if dataset_path.exists():
        dataset_path.unlink()
    if output_path.exists():
        output_path.unlink()


# ============================================================================
# EXAMPLE 5: Integration with Evaluation Pipeline
# ============================================================================

async def example_5_evaluation_integration():
    """
    Complete pipeline: Evaluate → Regenerate → Re-evaluate
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Evaluation → Regeneration → Re-evaluation Pipeline")
    print("="*70)
    
    # Initialize components
    evaluator = CoTEvaluator()
    regenerator = CoTRegenerator(
        models=["deepseek/deepseek-r1-0528:free"]
    )
    
    # Sample problem
    question = "If a rectangle has length 8 cm and width 5 cm, what is its area?"
    answer = "40 cm²"
    original_cot = "Length × Width = 8 × 5 = 40 cm²"
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Original CoT: {original_cot}")
    
    # Step 1: Initial evaluation
    print("\n📝 Step 1: Initial Evaluation")
    print("-" * 30)
    
    # Create proper client for evaluator
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        import openai
        evaluator.llm_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        initial_eval = evaluator.evaluate(question, original_cot, answer)
        
        print(f"Initial Grade: {initial_eval['grade']}")
        print(f"Initial Score: {initial_eval['score']:.2f}/10")
        
        if initial_eval['evaluation'].get('weaknesses'):
            print("Weaknesses:")
            for weakness in initial_eval['evaluation']['weaknesses'][:3]:
                print(f"  - {weakness}")
        
        # Step 2: Regenerate if grade is low
        if initial_eval['grade'] in ['C', 'D']:
            print("\n🔄 Step 2: Regenerating with Multiple Models")
            print("-" * 30)
            
            # Prepare evaluation details for regeneration
            evaluation_details = {
                "grade": initial_eval['grade'],
                "strengths": initial_eval['evaluation'].get('strengths', []),
                "weaknesses": initial_eval['evaluation'].get('weaknesses', []),
                "improvement_suggestions": initial_eval['evaluation'].get('improvement_suggestions', []),
                "learning_value_scores": initial_eval['evaluation'].get('learning_value_scores', {})
            }
            
            # Regenerate with multiple models
            result = await regenerator.regenerate_multi_async(
                question=question,
                answer=answer,
                previous_cot=original_cot,
                evaluation_details=evaluation_details
            )
            
            if result:
                print(f"Best Model: {result['best_model']}")
                print(f"Predicted Grade: {result['best_grade']}")
                print(f"Predicted Score: {result['best_score']:.2f}/10")
                
                # Step 3: Re-evaluate the regenerated CoT
                print("\n✅ Step 3: Re-evaluation of Regenerated CoT")
                print("-" * 30)
                
                new_eval = evaluator.evaluate(question, result['best_cot'], answer)
                
                print(f"New Grade: {new_eval['grade']}")
                print(f"New Score: {new_eval['score']:.2f}/10")
                
                # Compare improvement
                print("\n📈 Improvement Summary:")
                print(f"  Grade: {initial_eval['grade']} → {new_eval['grade']}")
                print(f"  Score: {initial_eval['score']:.2f} → {new_eval['score']:.2f}")
                
                improvement = new_eval['score'] - initial_eval['score']
                if improvement > 0:
                    print(f"  ✅ Improved by {improvement:.2f} points!")
                else:
                    print(f"  ⚠️ No improvement achieved")
        else:
            print(f"\n✅ Grade {initial_eval['grade']} is already good - no regeneration needed")
    else:
        print("⚠️ API key not found - skipping evaluation")


# ============================================================================
# EXAMPLE 6: Error Handling and Robustness
# ============================================================================

async def example_6_error_handling():
    """
    Demonstrate error handling and recovery mechanisms.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling and Robustness")
    print("="*70)
    
    # Test with potentially problematic inputs
    test_cases = [
        {
            "name": "Empty evaluation details",
            "question": "What is 5 + 3?",
            "answer": "8",
            "previous_cot": "5 + 3 = 8",
            "evaluation_details": {}  # Empty details
        },
        {
            "name": "Very long question",
            "question": "A " * 500 + "What is the sum?",  # Very long question
            "answer": "Unknown",
            "previous_cot": "Cannot determine",
            "evaluation_details": {"grade": "D", "weaknesses": ["Unclear"]}
        },
        {
            "name": "Special characters",
            "question": "What is 2²³ × π ÷ √16?",
            "answer": "2π",
            "previous_cot": "2²³ × π ÷ √16 = 8π ÷ 4 = 2π",
            "evaluation_details": {
                "grade": "C",
                "weaknesses": ["Needs more steps"],
                "improvement_suggestions": ["Break down the calculation"]
            }
        }
    ]
    
    regenerator = CoTRegenerator(
        models=["deepseek/deepseek-r1-0528:free"]
    )
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Attempt regeneration
            result = regenerator.regenerate_single(
                question=test_case['question'][:100],  # Truncate if too long
                answer=test_case['answer'],
                previous_cot=test_case['previous_cot'],
                evaluation_details=test_case['evaluation_details']
            )
            
            if result:
                print(f"✅ Regeneration successful")
                print(f"   Output length: {len(result)} characters")
            else:
                print(f"⚠️ Regeneration returned None (API might have failed)")
                
        except Exception as e:
            print(f"❌ Error occurred: {type(e).__name__}: {str(e)[:100]}")
    
    print("\n" + "="*70)
    print("Error handling test complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Run all examples sequentially.
    """
    print("\n" + "="*70)
    print("CoT REGENERATOR - COMPREHENSIVE USAGE EXAMPLES")
    print("="*70)
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  WARNING: OPENROUTER_API_KEY not set")
        print("Please set it in .env file or as environment variable")
        print("Some examples may fail without it\n")
    
    # Run examples
    try:
        # Synchronous examples
        print("\n▶️ Running synchronous examples...")
        example_1_simple_regeneration()
        example_3_custom_configurations()
        example_4_dataset_processing()
        
        # Asynchronous examples
        print("\n▶️ Running asynchronous examples...")
        await example_2_multi_model_comparison()
        await example_5_evaluation_integration()
        await example_6_error_handling()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. CoTRegenerator can work with single or multiple models")
    print("2. Async operations enable parallel processing for better performance")
    print("3. Integration with CoTEvaluator creates a complete improvement pipeline")
    print("4. Batch processing enables dataset-wide improvements")
    print("5. Robust error handling ensures reliability in production")
    print("\nFor more details, check the module documentation and source code.")


if __name__ == "__main__":
    asyncio.run(main())