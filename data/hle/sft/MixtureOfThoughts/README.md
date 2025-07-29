# Mixture of Thoughts Dataset Processing

This directory contains a script to process the [Mixture of Thoughts dataset](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) according to the specifications in `prd.md`.

## Usage

1. Install required dependencies:
```bash
pip install datasets
```

2. Run the processing script:
```bash
python process_mot_dataset.py
```

## Output

The script will create a `processed_mot_data/` directory with one JSON file per split:
- `MoT_code.json` - Contains all processed code-related conversations
- `MoT_math.json` - Contains all processed math-related conversations  
- `MoT_science.json` - Contains all processed science-related conversations

Each file contains an array of conversation objects with the format:
```json
[
  {
    "id": "MoT_code_0",
    "question": "...",
    "solution": "...",
    "answer": "..."
  },
  {
    "id": "MoT_code_1",
    "question": "...",
    "solution": "...",
    "answer": "..."
  }
]
```

## Features

- Processes only the first user-assistant message pair from each conversation
- Extracts thinking process from `<think>...</think>` tags as the solution
- Extracts final response after `</think>` as the answer
- Handles malformed conversations gracefully by skipping them
- Provides progress feedback during processing