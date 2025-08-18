# SFT Data Processing Pipeline

This repository contains comprehensive tools and scripts for processing diverse datasets for Supervised Fine-Tuning (SFT). The pipeline covers multiple domains including mathematics, science, medicine, law, history, chemistry, and general reasoning.

## 📁 Directory Structure

```
data/hle/sft/
├── 🧮 MixtureOfThoughts/     # Mixture of Thoughts dataset processing
├── 📐 OpenMath/              # OpenMath reasoning and filtering
├── 🧬 biology/               # Biology Tree of Thoughts data
├── ⚗️ chemistry/             # ChemPile chemistry Q&A extraction
├── 💭 general/               # General reasoning (StrategyQA, MedCalc)
├── 📚 humanity/              # Humanities data (history, law)
├── 🏥 medical/               # Medical reasoning (ReasonMD)
├── 📊 results/               # Processed output directory
├── 🔧 Core Scripts/          # Main processing tools
│   ├── OpenMathReasoningFiltering.py      # LLM-based filtering
│   ├── OpenMathReasoningFiltering_bylabel.py  # Label-based filtering
│   ├── generateFromSeed.py               # Solution generation from seeds
│   ├── upload_data.py                    # HuggingFace upload tool
│   ├── difficulty_scorer.py              # Multi-metric difficulty assessment
│   ├── length_selector.py                # Length-based data selection
│   └── merge_datasets.py                 # Dataset merging utility
└── 🚀 Bash Scripts/         # Automation scripts
    ├── run_filter.sh                      # LLM filtering SLURM script
    ├── run_label_filter.sh                # Label filtering SLURM script
    ├── run_length_selector.sh             # Data selection SLURM script
    └── run_upload_data.sh                 # Upload SLURM script
```

## 🎯 Dataset Coverage

### Mathematics & Science
- **MixtureOfThoughts**: 348K samples across code, math, and science with explicit reasoning traces
- **OpenMath**: 3.9M mathematical problems with difficulty filtering (CoT and GenSelect splits)
- **Biology**: Tree of Thoughts reasoning for biological questions
- **Chemistry**: ChemPile dataset with cleaned chemistry Q&A pairs

### Humanities & Medicine
- **History**: Historical Q&A with LLM-generated reasoning chains
- **Law**: Legal questions filtered by difficulty (1-5 scale) from CoT_Legal_Issues_And_Laws
- **Medical/ReasonMD**: Medical reasoning with VLLM-extracted concise answers
- **General**: StrategyQA and MedCalc for general reasoning and medical calculations

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv hfenv
source hfenv/bin/activate  # Linux/Mac
# or
hfenv\Scripts\activate     # Windows

# Install dependencies
pip install torch transformers datasets huggingface-hub tqdm pandas pyarrow
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2. API Keys Configuration

```bash
# Copy and edit the keys template
cp keys.json.example keys.json

# Add your Hugging Face token
{
  "llm": "your_huggingface_token_here"
}
```

## 🔄 Complete Processing Workflow

### Step 1: Data Selection & Filtering

Choose one of three approaches based on your dataset size and requirements:

#### Option A: Fast Length-Based Selection (Recommended for large datasets)
```bash
python length_selector.py \
    --input "dataset-name" \
    --total_samples 10000 \
    --output "selected_seeds.json"
```

#### Option B: Precision Difficulty-Based Selection (For small-medium datasets)
```bash
python difficulty_scorer.py \
    --input "dataset-name" \
    --output "difficulty_scores.json" \
    --max_samples 50000
```

#### Option C: Label-Based Ultra-Fast Filtering (For pre-labeled data)
```bash
python OpenMathReasoningFiltering_bylabel.py \
    --filter-by-pass-rate 0.1 \
    --save-per-iteration 10000
```

### Step 2: Domain-Specific Processing

Process data according to domain requirements:

#### Mathematics (MixtureOfThoughts)
```bash
cd MixtureOfThoughts
python process_mot_dataset.py
# Output: processed_mot_data/ with MoT_code.json, MoT_math.json, MoT_science.json
```

#### OpenMath Reasoning
```bash
cd OpenMath
python OpenMathReasoningFiltering.py \
    --inference-model Qwen/Qwen3-8B \
    --filter-by-pass-rate 0.3
```

#### Medical (ReasonMD)
```bash
cd medical/reasonMD
# Step 1: Select data
python reasonmd_selector.py --target_samples 1000
# Step 2: Convert with VLLM
python convert_reasonmd.py
```

#### Law
```bash
cd humanity/law
python format_law.py --difficulty 4 --num_samples 200
```

#### History
```bash
cd humanity/history
python convert_history.py --input input.json --output output.json
```

#### Chemistry
```bash
cd chemistry/chempile
bash run_chempile_extraction.sh
python remove_error_items.py
```

### Step 3: Data Augmentation (Optional)
```bash
# Generate new solutions from seed problems
python generateFromSeed.py \
    --model Qwen/Qwen3-32B \
    --input_file selected_seeds.json \
    --output_file expanded_solutions.json \
    --max_tokens 4096
```

### Step 4: Dataset Upload to Hugging Face
```bash
# Convert to Parquet and upload
python upload_data.py \
    --dataset_path ./results/processed_dataset \
    --repo_id your-username/sft-dataset-name \
    --create_dataset_card
```

## 📊 Unified Output Format

All processed datasets follow a consistent schema:

```json
{
  "id": "{dataset}_{index}",
  "question": "The original question or problem",
  "output": "<think>Step-by-step reasoning process</think>\nFinal answer",
  "answer": "Concise final answer"
}
```

### Field Descriptions
- **id**: Unique identifier with dataset prefix
- **question**: Original question/problem from the dataset
- **output**: Combined reasoning (in `<think>` tags) and answer
- **answer**: Standalone final answer for validation

## 🛠️ Core Processing Scripts

### 📊 Difficulty Scorer (`difficulty_scorer.py`)
Evaluates question difficulty using multiple metrics:
- Average log probability of gold answers
- Ensemble accuracy across multiple models
- IRT difficulty parameter β

```bash
python difficulty_scorer.py \
    --input "dataset-name" \
    --output "scores.json" \
    --max_samples 10000
```

### 📏 Length Selector (`length_selector.py`)
Fast data selection based on answer length distribution:
- Half-Gaussian distribution (favors longer answers)
- Dynamic binning based on actual data
- Streaming processing for large datasets

```bash
python length_selector.py \
    --input "dataset-name" \
    --total_samples 5000 \
    --num_bins 6 \
    --curve_sharpness 3.0
```

### 🌱 Seed Generator (`generateFromSeed.py`)
Generates new solutions from seed problems:
```bash
python generateFromSeed.py \
    --model Qwen/Qwen3-32B \
    --input_file seeds.json \
    --output_file generated.json
```

### 📤 Upload Tool (`upload_data.py`)
Converts JSON to Parquet and uploads to Hugging Face:
```bash
python upload_data.py \
    --dataset_path ./results \
    --repo_id username/dataset \
    --create_dataset_card
```

## 🖥️ SLURM Scripts for HPC

For cluster/HPC environments:

```bash
# Label-based filtering
sbatch run_label_filter.sh

# LLM-based filtering
sbatch run_filter.sh

# Length-based selection
sbatch run_length_selector.sh

# Upload to Hugging Face
sbatch run_upload_data.sh
```

## 📋 Workflow Examples

### Complete Multi-Domain SFT Dataset
```bash
# 1. Process mathematics
cd MixtureOfThoughts && python process_mot_dataset.py
cd ../OpenMath && python OpenMathReasoningFiltering_bylabel.py

# 2. Process sciences
cd ../biology && python ToT/transfer_data.py
cd ../chemistry && bash chempile/run_chempile_extraction.sh

# 3. Process humanities
cd ../humanity/law && python format_law.py
cd ../history && python convert_history.py

# 4. Process medical
cd ../../medical/reasonMD && python reasonmd_selector.py && python convert_reasonmd.py

# 5. Merge and upload
cd ../..
python merge_datasets.py --input_dirs results/* --output merged_dataset
python upload_data.py --dataset_path merged_dataset --repo_id username/complete-sft
```

### Quick Domain-Specific Dataset
```bash
# For mathematics only
cd OpenMath
python OpenMathReasoningFiltering_bylabel.py --filter-by-pass-rate 0.1
cd ..
python upload_data.py --dataset_path OpenMath/results --repo_id username/math-sft
```

## 🔧 Configuration Tips

### For Large Datasets (>100K samples)
- Use label-based or length-based filtering
- Set appropriate `--save-per-iteration` values
- Process in parallel with percentage ranges

### For Multi-GPU Systems
- Use tensor parallelism: `--inference-tp 2`
- Increase batch sizes: `--vllm-batch-size 32`
- Monitor GPU memory with `nvidia-smi`

### For Memory Optimization
- Process datasets in chunks
- Use streaming where available
- Keep Parquet files separate (don't merge)

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch sizes
--inference-batch-size 1 --vllm-batch-size 16

# Use tensor parallelism
--inference-tp 2
```

### CUDA Errors
```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_FLASH_ATTENTION=1
```

### Upload Failures
```bash
# Verify HF token
cat keys.json

# Check repository permissions
# Ensure write access to the repository
```

## 📚 Dataset Sources

- **MixtureOfThoughts**: `open-r1/Mixture-of-Thoughts`
- **OpenMath**: `nvidia/OpenMathReasoning`
- **Law**: `moremilk/CoT_Legal_Issues_And_Laws`
- **ReasonMD**: `lingshu-medical-mllm/ReasonMed`, `neko-llm/CoT_Medicine`
- **Custom**: History, Biology ToT, Chemistry ChemPile

## 📋 Requirements

- Python 3.8+
- PyTorch with CUDA support
- vLLM for efficient inference
- Hugging Face Hub account
- GPU with 8GB+ VRAM (recommended)
- 32GB+ RAM for large datasets

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review script help: `python script.py --help`
3. Monitor resource usage during processing
4. Verify environment setup
5. Contact @Junyu Liu on Slack

## 📝 License

This pipeline is provided for research and educational purposes. Please refer to individual dataset licenses for data usage rights.