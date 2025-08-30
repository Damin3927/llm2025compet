# Hugging Face Datasets Processing Pipeline

This script provides a comprehensive pipeline for processing multiple Hugging Face datasets with CoT evaluation and regeneration capabilities.

## Features

- **Batch Processing**: Process multiple datasets with different configurations
- **Organized Output**: Saves results in organized folder structure (`dataset_name/config/`)
- **Async Processing**: Parallel model execution for better performance
- **Comprehensive Evaluation**: Uses CoTEvaluator for quality assessment
- **Smart Regeneration**: Regenerates only low-quality CoTs based on grade threshold
- **Detailed Statistics**: Comprehensive reporting and analytics
- **Flexible Configuration**: JSON-based configuration with processing options

## Quick Start

### 1. Create Example Configuration

```bash
python process_hf_datasets.py --create-example-config
```

This creates `datasets_config.json` with example configurations.

### 2. Test with Limited Data

```bash
python process_hf_datasets.py --config datasets_config.json --max-items 10 --output-dir ./test_output
```

### 3. Full Processing

```bash
python process_hf_datasets.py --config datasets_config.json --output-dir ./processed_datasets
```

## Configuration File Format

### Basic Structure

```json
{
  "datasets": [
    {
      "dataset_name": "gsm8k",
      "configs": ["main", "socratic"],
      "split": "train",
      "question_field": "question",
      "answer_field": "answer",
      "output_field": "output",
      "description": "Grade school math problems"
    }
  ],
  "processing_options": {
    "max_items_per_dataset": 1000,
    "grade_threshold": "C",
    "evaluator_models": ["deepseek/deepseek-r1-0528:free"],
    "regenerator_models": [
      "deepseek/deepseek-r1-0528:free",
      "meta-llama/llama-3-8b-instruct:free"
    ],
    "use_async_regeneration": true
  }
}
```

### Dataset Configuration Fields

- **`dataset_name`**: Hugging Face dataset identifier
- **`configs`**: List of dataset configurations to process
- **`split`**: Dataset split (train, test, validation)
- **`question_field`**: Field name containing the question/problem
- **`answer_field`**: Field name containing the expected answer
- **`output_field`**: Field name containing the CoT reasoning
- **`description`**: Optional description for documentation

### Processing Options

- **`max_items_per_dataset`**: Limit items per dataset (for testing)
- **`grade_threshold`**: Regenerate items with this grade or lower (A/B/C/D)
- **`evaluator_models`**: Models for evaluation
- **`regenerator_models`**: Models for regeneration
- **`use_async_regeneration`**: Enable async multi-model regeneration

## Command Line Options

```bash
python process_hf_datasets.py [OPTIONS]

Options:
  --config PATH                 Path to JSON configuration file (required)
  --output-dir PATH            Output directory [default: ./processed_datasets]
  --evaluator-models MODELS    Comma-separated list of evaluation models
  --regenerator-models MODELS  Comma-separated list of regeneration models
  --grade-threshold GRADE      Regenerate threshold [A|B|C|D, default: C]
  --max-items N                Maximum items per dataset (for testing)
  --no-async                   Disable async multi-model regeneration
  --create-example-config      Create example config and exit
```

## Output Structure

```
processed_datasets/
├── processing_statistics.json     # Detailed statistics
├── summary_report.txt             # Human-readable report
├── dataset1/
│   ├── config1/
│   │   ├── processed_20240101_120000.json
│   │   └── processed_20240101_120000.jsonl
│   └── config2/
│       ├── processed_20240101_130000.json
│       └── processed_20240101_130000.jsonl
└── dataset2/
    └── default/
        ├── processed_20240101_140000.json
        └── processed_20240101_140000.jsonl
```

## Output Data Format

Each processed item includes:

```json
{
  "id": "0",
  "question": "What is 2 + 2?",
  "answer": "4",
  "output": "<think>Improved reasoning here...</think>4",
  "metadata": {
    "original_data": { ... },
    "cot_history": [
      {
        "timestamp": "2024-01-01T12:00:00",
        "output": "<think>Original reasoning</think>4",
        "evaluation": {
          "grade": "C",
          "strengths": ["Correct answer"],
          "weaknesses": ["Too brief"],
          "improvement_suggestions": ["Add explanation"],
          "learning_value_scores": { ... }
        }
      },
      {
        "timestamp": "2024-01-01T12:01:00",
        "output": "<think>Improved reasoning here...</think>4",
        "evaluation": null,
        "regeneration_metadata": {
          "model": "deepseek/deepseek-r1-0528:free",
          "predicted_grade": "B",
          "predicted_score": 7.2
        }
      }
    ]
  }
}
```

## Example Configurations

### Mathematics Datasets

```json
{
  "datasets": [
    {
      "dataset_name": "gsm8k",
      "configs": ["main", "socratic"],
      "split": "train",
      "question_field": "question",
      "answer_field": "answer",
      "output_field": "output"
    },
    {
      "dataset_name": "math_qa",
      "configs": ["default"],
      "split": "train", 
      "question_field": "Problem",
      "answer_field": "correct",
      "output_field": "Rationale"
    },
    {
      "dataset_name": "aqua_rat",
      "configs": ["raw"],
      "split": "train",
      "question_field": "question",
      "answer_field": "correct", 
      "output_field": "rationale"
    }
  ]
}
```

### Science Datasets

```json
{
  "datasets": [
    {
      "dataset_name": "sciq",
      "configs": ["default"],
      "split": "train",
      "question_field": "question",
      "answer_field": "correct_answer",
      "output_field": "explanation"
    },
    {
      "dataset_name": "arc",
      "configs": ["ARC-Challenge", "ARC-Easy"],
      "split": "train",
      "question_field": "question",
      "answer_field": "answerKey",
      "output_field": "explanation"
    }
  ]
}
```

## Processing Workflow

1. **Load Configuration**: Parse JSON config file
2. **Dataset Loading**: Load each dataset/config combination
3. **Item Preparation**: Format items for evaluation
4. **Evaluation**: Assess CoT quality with grading
5. **Regeneration**: Improve low-quality items (if grade ≤ threshold)
6. **Best Selection**: Choose best regenerated version (multi-model)
7. **Save Results**: Organized folder structure with metadata
8. **Statistics**: Comprehensive reporting and analytics

## Performance Tips

- **Start Small**: Use `--max-items 100` for initial testing
- **Parallel Processing**: Use multiple regenerator models for better results
- **Grade Threshold**: Adjust based on your quality requirements
- **Async Mode**: Enable for multi-model regeneration efficiency
- **Resource Management**: Monitor API usage and rate limits

## Error Handling

The script includes robust error handling:
- **Individual Item Failures**: Don't stop entire processing
- **API Failures**: Retry logic with exponential backoff
- **Missing Fields**: Graceful handling with warnings
- **Dataset Loading**: Continue with other datasets if one fails

## Monitoring Progress

- **Progress Bars**: Real-time progress for each dataset
- **Logging**: Detailed logs saved to `dataset_processing.log`
- **Statistics**: Live updates on processing stats
- **Error Tracking**: All errors logged with item IDs

## Example Usage

### Basic Usage

```bash
# Create example config
python process_hf_datasets.py --create-example-config

# Test with small dataset
python process_hf_datasets.py --config datasets_config.json --max-items 50

# Full processing
python process_hf_datasets.py --config datasets_config.json
```

### Advanced Usage

```bash
# Multi-model regeneration
python process_hf_datasets.py \
  --config datasets_config.json \
  --regenerator-models "deepseek/deepseek-r1-0528:free,meta-llama/llama-3-8b-instruct:free" \
  --grade-threshold B

# High-quality filtering
python process_hf_datasets.py \
  --config datasets_config.json \
  --grade-threshold A \
  --evaluator-models "deepseek/deepseek-r1-0528:free,gpt-4"
```

## Requirements

- Python 3.8+
- `datasets` library for Hugging Face datasets
- `openai` for API interactions
- `tqdm` for progress bars
- `python-dotenv` for environment variables
- API key in environment: `OPENROUTER_API_KEY`

## Installation

```bash
pip install datasets openai tqdm python-dotenv
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

2. **Dataset Not Found**
   - Verify dataset name and config in Hugging Face
   - Check if dataset requires authentication

3. **Field Not Found**
   - Inspect dataset structure first
   - Use `dataset.column_names` to see available fields

4. **Out of Memory**
   - Use `--max-items` to limit processing
   - Process datasets individually

5. **API Rate Limits**
   - Reduce concurrency
   - Add delays between requests

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python process_hf_datasets.py --config datasets_config.json
```

This comprehensive script provides a complete solution for processing Hugging Face datasets with CoT evaluation and regeneration capabilities.