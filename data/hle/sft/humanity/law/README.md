# Legal Dataset Formatter

## Overview

This tool processes the [CoT_Legal_Issues_And_Laws dataset](https://huggingface.co/datasets/moremilk/CoT_Legal_Issues_And_Laws) from Hugging Face, filtering legal questions by difficulty level and transforming them into a structured SFT (Supervised Fine-Tuning) format with explicit reasoning traces.

## Processing Pipeline
[flow_chart](https://claude.ai/public/artifacts/6a50ba0b-e156-44ea-bb5c-921a336f57ad)

### Step-by-Step Processing Pipeline Description

1. **Load Dataset**  
   The process begins by loading the `moremilk/CoT_Legal_Issues_And_Laws` dataset from Hugging Face using the `format_law.py` script.

2. **Convert to DataFrame**  
   The raw dataset is converted into a pandas DataFrame for easier manipulation and analysis.

3. **Extract Metadata**  
   Metadata fields, such as difficulty and reasoning, are extracted from each data entry.

4. **Parse Difficulty Levels**  
   The script parses the difficulty level from the metadata, which is used to determine the complexity of each legal question.

5. **Filter by Difficulty**  
   Each entry is evaluated:  
   - If the difficulty is **below** the specified threshold, the entry is **excluded** from further processing.  
   - If the difficulty **meets or exceeds** the threshold, the entry is **included** for the next steps.

6. **Transform Format**  
   Included entries are prepared for format transformation to match the SFT (Supervised Fine-Tuning) requirements.

7. **Extract Components**  
   For each included entry, the following components are extracted:  
   - **Question Field**: The original legal question.  
   - **Reasoning from Metadata**: The step-by-step legal reasoning process.  
   - **Answer Field**: The final answer to the legal question.

8. **Build Structure**  
   The extracted components are combined to form a structured data entry.

9. **Create SFT Entry**  
   For each structured entry:  
   - **Generate ID**: Assign a unique identifier in the format `law_{index}`.  
   - **Format Output**: Combine the reasoning (enclosed in `<think>...</think>`) and the answer into a single output field.

10. **Combine Fields**  
    The ID, question, formatted output, and answer are assembled into the final SFT entry.

11. **Add to Results**  
    The completed SFT entry is added to the results list.

12. **Log Statistics**  
    For excluded entries (those below the difficulty threshold), statistics are logged for reporting and analysis.

13. **Iterate Over All Rows**  
    The process repeats for each row in the dataset until all entries have been processed.

14. **Save JSON**  
    Once all entries are processed, the results are saved as a JSON file (`law.json`) in the `results/law/` directory.

This pipeline ensures that only legal questions of sufficient difficulty are included, and that each entry is transformed into a consistent, structured format suitable for supervised fine-tuning tasks.

## Features

- **Difficulty-based Filtering**: Select legal questions based on complexity level
- **Metadata Extraction**: Parse difficulty and reasoning from dataset metadata
- **Format Transformation**: Convert to structured SFT format with think tags
- **Batch Processing**: Efficiently process entire dataset splits
- **Statistics Reporting**: Display filtering results and data distribution

## Dataset Information

### Source
- **Repository**: Hugging Face - `moremilk/CoT_Legal_Issues_And_Laws`
- **Type**: Legal reasoning dataset with Chain-of-Thought explanations
- **Split**: Training set (default)

### Data Fields
- **question**: Legal question or scenario
- **answer**: Final answer to the legal question
- **metadata**: Contains difficulty level and reasoning process
  - **difficulty**: Integer value (1-5) indicating question complexity
  - **reasoning**: Step-by-step legal reasoning

## Output Format

### SFT Structure
```json
{
  "id": "law_0",
  "question": "What are the legal implications of...",
  "output": "<think>Legal reasoning process...</think>\nFinal legal answer",
  "answer": "Final legal answer"
}
```

### Field Descriptions
- **id**: Unique identifier with format `law_{index}`
- **question**: Original legal question from dataset
- **output**: Combined reasoning (in think tags) and answer
- **answer**: Standalone final answer

## Usage

### Basic Usage
```bash
# Process with default settings (difficulty >= 4)
python format_law.py
```

### Custom Difficulty Filtering
```bash
# Include only high difficulty questions (5 and above)
python format_law.py --difficulty 5

# Include all questions (difficulty >= 1)
python format_law.py --difficulty 1
```

### Advanced Options
```bash
python format_law.py \
  --split train \
  --difficulty 3 \
  --output ../custom_results/law \
  --show-sample
```

### Command Line Arguments
- `--split`: Dataset split to load (default: "train")
- `--difficulty`: Minimum difficulty threshold (default: 4)
  - 1: Very Easy
  - 2: Easy
  - 3: Medium
  - 4: Hard
  - 5: Very Hard
- `--output`: Output directory path (default: "../results/law")
- `--show-sample`: Display sample of filtered data

## Processing Steps

### 1. Dataset Loading
- Loads the CoT_Legal_Issues_And_Laws dataset from Hugging Face
- Converts to pandas DataFrame for efficient manipulation
- Displays dataset statistics and column information

### 2. Metadata Extraction
- Parses the metadata field to extract difficulty levels
- Creates a dedicated difficulty column for filtering

### 3. Difficulty Filtering
- Filters questions based on the specified difficulty threshold
- Retains only questions meeting or exceeding the threshold
- Reports filtering statistics and distribution

### 4. Format Transformation
- Iterates through filtered entries
- Extracts question, reasoning, and answer components
- Formats reasoning with `<think>` tags
- Combines components into SFT structure

### 5. Output Generation
- Creates output directory if it doesn't exist
- Saves formatted data as JSON with proper indentation
- Preserves all legal reasoning for training purposes

## Output Statistics

The tool provides detailed statistics including:
- Total samples loaded
- Number of samples after filtering
- Difficulty distribution of filtered data
- Sample preview (when requested)

## Requirements

### Dependencies
```bash
pip install datasets pandas
```

### System Requirements
- Python 3.7 or higher
- Internet connection for dataset download
- Sufficient disk space for dataset cache

## File Structure
```
law/
├── format_law.py       # Main processing script
├── README.md          # This documentation
└── results/           # Output directory (created automatically)
    └── law/
        └── law.json   # Formatted output file
```

## Example Output

### Input (from dataset)
```json
{
  "question": "Is a verbal contract legally binding?",
  "answer": "Yes, verbal contracts can be legally binding...",
  "metadata": {
    "difficulty": 4,
    "reasoning": "To determine if a verbal contract is legally binding, we need to consider..."
  }
}
```

### Output (SFT format)
```json
{
  "id": "law_42",
  "question": "Is a verbal contract legally binding?",
  "output": "<think>To determine if a verbal contract is legally binding, we need to consider...</think>\nYes, verbal contracts can be legally binding...",
  "answer": "Yes, verbal contracts can be legally binding..."
}
```

## Error Handling

The script includes robust error handling for:
- Dataset loading failures
- Missing metadata fields
- Invalid difficulty values
- File system operations

## Performance Considerations

- **Memory Usage**: DataFrame operations are memory-efficient
- **Processing Speed**: Vectorized operations for filtering
- **Output Size**: JSON formatting with indentation for readability

## Development

### Extending the Tool

To customize the processing:

1. **Add Custom Filters**: Modify `filter_by_difficulty()` for additional criteria
2. **Change Output Format**: Update `transfer_data_format()` for different structures
3. **Add Preprocessing**: Insert data cleaning steps before transformation
4. **Include Additional Metadata**: Extract more fields from the metadata

### Testing
```python
# Test with a small subset
python format_law.py --difficulty 4 --show-sample
```

## Troubleshooting

### Common Issues

1. **Dataset Download Fails**
   ```bash
   # Check internet connection
   # Verify Hugging Face availability
   # Clear cache if corrupted
   rm -rf ~/.cache/huggingface/datasets/moremilk___cot_legal_issues_and_laws
   ```

2. **Memory Issues**
   - Process in batches by modifying the script
   - Use a machine with more RAM
   - Filter more aggressively

3. **Missing Difficulty Values**
   - The script handles missing values gracefully
   - Check metadata structure if errors occur

## License

This tool is provided for educational and research purposes. Please refer to the original [CoT_Legal_Issues_And_Laws dataset](https://huggingface.co/datasets/moremilk/CoT_Legal_Issues_And_Laws) for licensing information regarding the data itself.