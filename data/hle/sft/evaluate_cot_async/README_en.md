# CoT Evaluation & Regeneration System

A comprehensive system for evaluating and improving Chain of Thought (CoT) quality in training data. This module provides a flexible and reusable way to assess the educational value of CoT examples and regenerate high-quality CoTs based on evaluation feedback.

## Features

### Evaluation Features
- **Class-based design**: Easy to import and use in other scripts
- **Flexible configuration**: Customizable API keys, models, and parameters
- **Multiple evaluation modes**: Single item evaluation and batch dataset processing
- **Multi-model support**: Use multiple models for more robust evaluation
- **Progress tracking**: Built-in progress bars and detailed statistics
- **Multiple format support**: JSONL, JSON, and Hugging Face datasets
- **Resume capability**: Skip already evaluated items

### Regeneration Features (New)
- **Async multi-model regeneration**: Regenerate CoTs with multiple models simultaneously
- **Automatic best selection**: Automatically select the best result based on evaluation scores
- **Flexible model configuration**: Configure each model with its own parameters
- **Batch processing**: Support for regenerating entire datasets
- **Backwards compatibility**: Maintain compatibility with existing code

## Installation

Make sure you have the required dependencies:

```bash
pip install openai tqdm python-dotenv
# Optional: for Hugging Face datasets support
pip install datasets
```

Set up your API key:

```bash
# Set environment variable
export OPENROUTER_API_KEY="your_api_key_here"

# Or create a .env file
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Basic Usage

#### Evaluation

```python
from cot_evaluator import CoTEvaluationProcessor

# Initialize the processor
processor = CoTEvaluationProcessor()

# Evaluate a single item
item = {
    "question": "What is 2 + 2?",
    "output": "<think>I need to add 2 and 2. 2 + 2 = 4.</think>",
    "answer": "4"
}

result = processor.evaluate_single_item(item)
print(f"Grade: {result['grade']}, Score: {result['score']:.2f}/10")

# Evaluate a dataset
stats = processor.evaluate_dataset(
    dataset_path="your_dataset.jsonl",
    output_file="evaluated_dataset.jsonl"
)
print(f"Evaluated {stats['evaluated_count']} items")
```

#### Regeneration (New)

```python
from cot_regenerator import CoTRegenerator
import asyncio

# Single model regeneration
regenerator = CoTRegenerator()
new_cot = regenerator.regenerate_single(
    question="What is 15% of 240?",
    answer="36",
    previous_cot="15% of 240 = 240 × 0.15 = 36",
    evaluation_details={
        "grade": "C",
        "weaknesses": ["No explanation", "No verification"],
        "improvement_suggestions": ["Explain percentage concept", "Verify result"]
    }
)

# Async multi-model regeneration (automatically selects best result)
regenerator = CoTRegenerator(
    models=["model1", "model2", "model3"]
)

best_result = asyncio.run(regenerator.regenerate_multi_async(
    question, answer, previous_cot, evaluation_details
))

print(f"Best model: {best_result['best_model']}")
print(f"Best CoT: {best_result['best_cot']}")

### Advanced Configuration

```python
# Customize the processor
processor = CoTEvaluationProcessor(
    api_key="your_custom_api_key",
    model="deepseek/deepseek-r1-0528:free",
    temperature=0.1,
    max_tokens=4000
)

# Use multiple models for evaluation
stats = processor.evaluate_dataset(
    dataset_path="dataset.jsonl",
    evaluator_models="model1,model2,model3",
    eval_concurrency=3,
    output_format="json",
    skip_existing=False
)
```

## API Reference

### CoTEvaluationProcessor

The main class for CoT evaluation.

### CoTRegenerator (New)

Class for regenerating CoTs based on evaluation feedback.

#### Constructor

```python
CoTRegenerator(
    models: Optional[Union[str, List[str], List[ModelConfig]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    default_api_base: str = "https://openrouter.ai/api/v1",
    system_prompt: Optional[str] = None
)
```

**Parameters:**
- `models`: Single model name, list of model names, or list of ModelConfig objects
- `api_keys`: Dictionary of API keys (e.g., {"openrouter": "key", "anthropic": "key"})
- `default_api_base`: Default API base URL
- `system_prompt`: Custom system prompt for regeneration

#### Methods

##### regenerate_single()

Regenerate CoT using a single model.

```python
regenerate_single(
    question: str,
    answer: str,
    previous_cot: str,
    evaluation_details: Dict[str, Any],
    model_config: Optional[ModelConfig] = None
) -> Optional[str]
```

**Parameters:**
- `question`: The problem/question
- `answer`: The expected answer
- `previous_cot`: The previous CoT reasoning
- `evaluation_details`: Evaluation details from the evaluator
- `model_config`: Optional specific model configuration to use

**Returns:**
- Regenerated CoT text (without `<think>` tags) or None if failed

##### regenerate_multi_async()

Regenerate CoT asynchronously with multiple models and select the best result.

```python
async regenerate_multi_async(
    question: str,
    answer: str,
    previous_cot: str,
    evaluation_details: Dict[str, Any],
    return_all: bool = False
) -> Union[Dict[str, Any], List[Dict[str, Any]]]
```

**Parameters:**
- `question`: The problem/question
- `answer`: The expected answer
- `previous_cot`: The previous CoT reasoning
- `evaluation_details`: Evaluation details from the evaluator
- `return_all`: If True, return all results; if False, return only the best

**Returns:**
- If `return_all` is False: Dictionary with best result and metadata
- If `return_all` is True: List of all results with metadata

##### regenerate_dataset()

Regenerate CoTs for an entire dataset.

```python
regenerate_dataset(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    grade_threshold: str = "C",
    specific_ids: Optional[List[str]] = None,
    use_async: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `dataset_path`: Path to the input JSONL dataset
- `output_path`: Path for output (defaults to input path)
- `grade_threshold`: Regenerate items with this grade or lower
- `specific_ids`: Optional list of specific IDs to regenerate
- `use_async`: Whether to use async multi-model regeneration

**Returns:**
- Dictionary with regeneration statistics

### ModelConfig (New)

Configuration class for specific models.

```python
ModelConfig(
    model_name: str,
    api_base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 40000,
    retry_attempts: int = 3
)
```

**Parameters:**
- `model_name`: Model name
- `api_base_url`: API base URL
- `api_key`: API key (uses environment variable if not provided)
- `temperature`: Generation temperature
- `max_tokens`: Maximum tokens
- `retry_attempts`: Number of retry attempts

### CoTEvaluationProcessor Details

#### Constructor

```python
CoTEvaluationProcessor(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8000
)
```

**Parameters:**
- `api_key`: OpenAI API key (uses environment variable if not provided)
- `base_url`: API base URL (defaults to OpenRouter)
- `model`: Model name to use for evaluation
- `temperature`: Generation temperature (0.0 for consistent evaluation)
- `max_tokens`: Maximum tokens for evaluation responses

#### Methods

##### evaluate_single_item()

Evaluate a single CoT item.

```python
evaluate_single_item(
    item: Dict,
    model_names: Optional[List[str]] = None,
    concurrency: int = 4
) -> Dict
```

**Parameters:**
- `item`: Dictionary with `question`, `output`, and `answer` fields
- `model_names`: List of model names for multi-model evaluation
- `concurrency`: Number of concurrent evaluations

**Returns:**
- Dictionary containing grade, score, and detailed evaluation

##### evaluate_dataset()

Evaluate an entire dataset.

```python
evaluate_dataset(
    dataset_path: str,
    output_file: Optional[str] = None,
    ids: Optional[str] = None,
    evaluator_models: Optional[str] = None,
    eval_concurrency: int = 4,
    output_format: str = 'jsonl',
    skip_existing: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `dataset_path`: Path to dataset file or Hugging Face dataset
- `output_file`: Output file path (overwrites input if not specified)
- `ids`: Comma-separated list of IDs to evaluate
- `evaluator_models`: Comma-separated list of model names
- `eval_concurrency`: Number of concurrent evaluations
- `output_format`: Output format ('jsonl' or 'json')
- `skip_existing`: Whether to skip already evaluated items

**Returns:**
- Dictionary with evaluation statistics

## Data Format

### Input Format

Each item should have the following structure:

```json
{
    "id": "unique_id",
    "question": "The problem or question",
    "output": "<think>Chain of thought reasoning</think>",
    "answer": "Expected answer"
}
```

### Output Format

Evaluation results are stored in the `metadata.cot_history` field:

```json
{
    "metadata": {
        "cot_history": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "output": "<think>...</think>",
                "evaluation": {
                    "grade": "A",
                    "score": 8.5,
                    "timestamp": "2024-01-01T12:00:00",
                    "passed_requirements": {
                        "independence": true,
                        "logical_completeness": true,
                        "correctness": true,
                        "answer_reached": true
                    },
                    "learning_value_scores": {
                        "method_explanation": 8,
                        "step_by_step": 9,
                        "verification": 7,
                        "common_mistakes": 8,
                        "domain_insight": 9,
                        "metacognitive": 8
                    },
                    "strengths": ["Clear step-by-step reasoning", "Good explanation"],
                    "weaknesses": ["Could provide more verification"],
                    "improvement_suggestions": ["Add alternative solution methods"]
                }
            }
        ]
    }
}
```

## Evaluation Criteria

The evaluation assesses CoT quality based on:

### Mandatory Requirements (Grade D if any fail)
1. **Independence**: No external references
2. **Logical Completeness**: Connected reasoning without leaps
3. **Correctness**: Proper application of methods
4. **Answer Reached**: Arrives at correct/reasonable answer

### Learning Value (0-10 scale)
1. **Method Selection Explanation**: Why this method was chosen
2. **Step-by-Step Derivation**: Clear, detailed steps
3. **Verification and Checking**: Validates results
4. **Handling Common Mistakes**: Points out error-prone areas
5. **Domain Insight**: Explains meaning and interpretation
6. **Metacognitive Elements**: Shows reasoning process

### Grading Scale
- **A (8.0+)**: Excellent training data
- **B (6.0-7.9)**: Good training data
- **C (4.0-5.9)**: Acceptable training data
- **D (<4.0)**: Poor training data

## Examples

See the sample files for complete examples:

### Evaluation Examples (`example_usage.py`)
- Single item evaluation
- Dataset evaluation
- Multi-model evaluation
- Custom configuration

### Regeneration Examples (`regenerate_example.py`)
- Single model regeneration
- Custom model configuration
- Async multi-model regeneration
- Getting all results for comparison
- Full dataset regeneration
- Backwards compatibility with existing code

## Command Line Usage

The original command-line interface is still available:

```bash
python cot_evaluator.py --dataset your_dataset.jsonl --output-file evaluated.jsonl
```

## Integration with Other Scripts

### Evaluation and Regeneration Integration

```python
# In your data processing pipeline
from cot_evaluator import CoTEvaluationProcessor
from cot_regenerator import CoTRegenerator
import asyncio

async def process_training_data():
    evaluator = CoTEvaluationProcessor()
    regenerator = CoTRegenerator(
        models=["model1", "model2", "model3"]
    )
    
    # Load your data
    data = load_your_data()
    
    for item in data:
        # Evaluate CoT quality
        if needs_evaluation(item):
            result = evaluator.evaluate_single_item(item)
            
            if result['grade'] in ['A', 'B']:
                # High quality - use as is
                add_to_training_set(item)
            elif result['grade'] == 'C':
                # Room for improvement - try regeneration
                evaluation_details = {
                    "grade": result['grade'],
                    "strengths": result['evaluation']['strengths'],
                    "weaknesses": result['evaluation']['weaknesses'],
                    "improvement_suggestions": result['evaluation']['improvement_suggestions'],
                    "learning_value_scores": result['evaluation']['learning_value_scores']
                }
                
                # Regenerate with multiple models and select best
                best = await regenerator.regenerate_multi_async(
                    item['question'],
                    item['answer'],
                    item['output'],
                    evaluation_details
                )
                
                if best and best['best_grade'] > result['grade']:
                    # Improvement successful
                    item['output'] = f"<think>{best['best_cot']}</think>{item['answer']}"
                    add_to_training_set(item)
                else:
                    # No improvement - use original
                    add_to_training_set(item)
            else:
                # Grade D - reject
                reject_item(item)

# Run
asyncio.run(process_training_data())
```

### Importing from Other Folders

```python
# When using from other folders
import sys
sys.path.append('/path/to/evaluate_cot_multiagent')

from evaluate_cot_multiagent import CoTRegenerator, CoTEvaluator

# Or direct import
from evaluate_cot_multiagent.cot_regenerator import CoTRegenerator
from evaluate_cot_multiagent.cot_evaluator import CoTEvaluator
```

## Error Handling

The processor includes robust error handling:
- API failures with retry logic
- Invalid JSON responses
- Missing required fields
- File I/O errors

Check the returned statistics for evaluation success rates and error counts.

## Performance Tips

- Use `skip_existing=True` to resume interrupted evaluations
- Adjust `eval_concurrency` based on your API rate limits
- Use multiple models for more robust evaluation (but increases cost)
- Consider batching large datasets