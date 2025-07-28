# difficulty_scorer.py
# Built for GPU‑ or CPU‑only workstations that cannot load models larger than 8 b parameters.
# Modified to process models one at a time for GPU memory efficiency.
# It scores 100 000 QA pairs for hardness using three signals:
#   1.   − average log‑probability of the gold answer under one small LLM (primary_model)
#   2.   Fraction of a small‑model ensemble that answers correctly (ensemble_accuracy)
#   3.   Item difficulty β from a 1‑PL IRT fit to the ensemble response matrix
# Final score = z(−logprob) + z(1 − ensemble_accuracy) + z(β)
# Author: OpenAI o3

import argparse, json, os, sys
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset

# Optional: install with  `pip install py-irt`  (uses PyTorch under the hood)
try:
    from pyirt import irt
    PY_IRT_AVAILABLE = True
except ImportError:
    PY_IRT_AVAILABLE = False
print(f"PY_IRT_AVAILABLE: {PY_IRT_AVAILABLE}")

###############################################################################
# Helper functions
###############################################################################

def normalise(text: str) -> str:
    """Lower-case strip and compress whitespace."""
    return " ".join(text.strip().lower().split())


def compare_answers(generated: str, gold: str) -> bool:
    """Compare generated answer with gold answer using multiple strategies."""
    # Normalize both answers
    gen_norm = normalise(generated)
    gold_norm = normalise(gold)
    
    # Exact match after normalization
    if gen_norm == gold_norm:
        return True
    
    # Check if generated answer contains the gold answer
    if gold_norm in gen_norm:
        return True
    
    # Check if gold answer contains the generated answer (for shorter generations)
    if gen_norm in gold_norm and len(gen_norm) > 0:
        return True
    
    # For numeric answers, try to extract and compare numbers
    import re
    gold_numbers = re.findall(r'-?\d+\.?\d*', gold_norm)
    gen_numbers = re.findall(r'-?\d+\.?\d*', gen_norm)
    
    if gold_numbers and gen_numbers:
        # Compare first number found in each
        try:
            gold_num = float(gold_numbers[0])
            gen_num = float(gen_numbers[0])
            # Allow small floating point differences
            return abs(gold_num - gen_num) < 1e-6
        except ValueError:
            pass
    
    return False


def average_logprob(model, tokenizer, prompt: str, answer: str, device: torch.device, max_length: int = 2048) -> float:
    """Return the mean log‑probability (natural log) of *answer* tokens given prompt."""

    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Check sequence length and truncate if necessary
        prompt_len = prompt_ids["input_ids"].size(1)
        answer_len = answer_ids["input_ids"].size(1)
        total_len = prompt_len + answer_len
        
        if total_len > max_length:
            # Truncate answer if total length exceeds limit
            available_answer_len = max_length - prompt_len
            if available_answer_len <= 0:
                # If prompt itself is too long, truncate prompt and keep small answer
                available_prompt_len = max_length - min(answer_len, 256)
                prompt_ids["input_ids"] = prompt_ids["input_ids"][:, -available_prompt_len:]
                answer_ids["input_ids"] = answer_ids["input_ids"][:, :min(answer_len, 256)]
            else:
                answer_ids["input_ids"] = answer_ids["input_ids"][:, :available_answer_len]
        
        input_ids = torch.cat([prompt_ids["input_ids"], answer_ids["input_ids"]], dim=1)
        attention_mask = torch.ones_like(input_ids, device=device)
        labels = input_ids.clone()
        # Mask prompt tokens so loss covers only answer tokens
        labels[:, : prompt_ids["input_ids"].size(1)] = -100
        
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.loss is the mean NLL over **unmasked** tokens (i.e. answer tokens)
            return -outputs.loss.item()  # positive log‑probability (mean per token)
        except torch.cuda.OutOfMemoryError:
            # If still OOM, try with even shorter sequences
            torch.cuda.empty_cache()
            shorter_max = max_length // 2
            return average_logprob(model, tokenizer, prompt, answer, device, shorter_max)


def generate_answer(model, tokenizer, question: str, device: torch.device, max_new_tokens: int = 1024, max_input_length: int = 1024) -> str:
    prompt = f"Question: {question}\nAnswer:"
    
    # Truncate input if too long
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens)
    
    try:
        with torch.no_grad():
            generated = model.generate(**input_ids, generation_config=gen_cfg)
        generated_text = tokenizer.decode(generated[0][input_ids["input_ids"].size(1):], skip_special_tokens=True)
        return generated_text.strip()
    except torch.cuda.OutOfMemoryError:
        # If OOM, try with shorter generation
        torch.cuda.empty_cache()
        shorter_gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens // 2)
        with torch.no_grad():
            generated = model.generate(**input_ids, generation_config=shorter_gen_cfg)
        generated_text = tokenizer.decode(generated[0][input_ids["input_ids"].size(1):], skip_special_tokens=True)
        return generated_text.strip()


def zscore(series: np.ndarray) -> np.ndarray:
    """Compute z-scores with robust handling of edge cases."""
    mu = series.mean()
    sigma = series.std()
    
    # Handle edge cases more robustly
    if sigma < 1e-10:  # Essentially zero variance
        # Return zeros if all values are the same
        return np.zeros_like(series)
    
    # Check for any infinite or NaN values
    z_scores = (series - mu) / sigma
    
    # Replace any infinite values with appropriate bounds
    z_scores = np.clip(z_scores, -10.0, 10.0)
    
    # Replace any NaN values with zero
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    
    return z_scores

###############################################################################
# Data loading utilities
###############################################################################

def get_data_sections_info(input_source: str, dataset_spec: str = "train", max_samples_per_iteration: int = 10000, token: str = None) -> dict:
    """
    Get information about data sections for processing.
    
    Returns:
        dict: Information about data sections including total sections, type, etc.
    """
    if not os.path.exists(input_source):
        # Hugging Face dataset
        print(f"Analyzing Hugging Face dataset: {input_source}", file=sys.stderr)
        # Parse dataset_spec: can be "split" or "config:split"
        if ":" in dataset_spec:
            dataset_config, split = dataset_spec.split(":", 1)
        else:
            # If no colon, treat as split name (config will be determined later if needed)
            dataset_config = None
            split = dataset_spec
        
        # For streaming datasets, we'll estimate sections based on max_samples_per_iteration
        # We can't know the exact size without iterating through everything
        return {
            'type': 'huggingface',
            'source': input_source,
            'dataset_config': dataset_config,
            'split': split,
            'token': token,
            'max_samples_per_iteration': max_samples_per_iteration,
            'sections': 'streaming'  # We'll process until no more data
        }
    
    elif os.path.isdir(input_source):
        # Directory of JSON files
        json_files = [f for f in os.listdir(input_source) if f.endswith('.json')]
        json_files.sort()  # Ensure consistent ordering
        print(f"Found {len(json_files)} JSON files in directory: {input_source}", file=sys.stderr)
        return {
            'type': 'json_directory',
            'source': input_source,
            'json_files': json_files,
            'sections': len(json_files)
        }
    
    else:
        # Single JSON file
        print(f"Single JSON file: {input_source}", file=sys.stderr)
        return {
            'type': 'json_file',
            'source': input_source,
            'sections': 1
        }


def load_qa_data_section(data_info: dict, section_idx: int, id_field: str = "id", 
                        question_field: str = "question", answer_field: str = "answer", 
                        max_samples_remaining: int = None) -> tuple:
    """
    Load QA data for a specific section with unique ID generation.
    
    Args:
        data_info: Data information from get_data_sections_info
        section_idx: Index of the section to load
        id_field: Field name for ID in the dataset
        question_field: Field name for question in the dataset  
        answer_field: Field name for answer in the dataset
        max_samples_remaining: Maximum number of samples to load (for HF datasets with total limit)
    
    Returns:
        tuple: (ids, questions, answers, has_more_data)
        has_more_data: Boolean indicating if there might be more sections
    """
    ids, questions, answers = [], [], []
    has_more_data = False
    
    if data_info['type'] == 'huggingface':
        # Load from Hugging Face dataset
        input_source = data_info['source']
        dataset_config = data_info['dataset_config']
        split = data_info['split']
        token = data_info.get('token')
        max_samples_per_iteration = data_info['max_samples_per_iteration']
        
        # Limit samples for this section if max_samples_remaining is specified
        if max_samples_remaining is not None:
            samples_to_load = min(max_samples_per_iteration, max_samples_remaining)
            print(f"Loading HF dataset section {section_idx}: {input_source} (limited to {samples_to_load} samples)", file=sys.stderr)
        else:
            samples_to_load = max_samples_per_iteration
            print(f"Loading HF dataset section {section_idx}: {input_source}", file=sys.stderr)
        
        # Load dataset and handle IterableDatasetDict vs IterableDataset
        if dataset_config:
            dataset = load_dataset(input_source, dataset_config, split=split, streaming=True, token=token)
        else:
            dataset = load_dataset(input_source, split, streaming=True, token=token)
        
        # For IterableDataset, we need to handle sectioning differently
        # Instead of skip/take, we'll iterate and track position
        start_idx = section_idx * max_samples_per_iteration
        
        items_loaded = 0
        has_more_data = False
        
        try:
            # Iterate through dataset and track position manually
            for global_idx, item in enumerate(dataset):
                # Skip items until we reach our section start
                if global_idx < start_idx:
                    continue
                
                # Stop if we've loaded enough samples for this section
                if items_loaded >= samples_to_load:
                    has_more_data = True  # More data available beyond this section
                    break
                
                # Debug: check item type and structure
                if items_loaded == 0:  # Only log for first item
                    print(f"Item type: {type(item)}", file=sys.stderr)
                    if isinstance(item, dict):
                        print(f"Available fields: {list(item.keys())}", file=sys.stderr)
                    else:
                        print(f"Item content preview: {str(item)[:200]}...", file=sys.stderr)
                
                # Handle different item types
                if isinstance(item, str):
                    # Item is a string, might be JSON
                    try:
                        item = json.loads(item)
                    except:
                        print(f"Error: Dataset item is a string but not valid JSON: {item[:100]}...", file=sys.stderr)
                        continue
                
                if not isinstance(item, dict):
                    print(f"Error: Dataset item is not a dictionary: {type(item)}", file=sys.stderr)
                    continue
                
                # Check if required fields exist
                if question_field not in item:
                    print(f"Error: Question field '{question_field}' not found in item. Available: {list(item.keys())}", file=sys.stderr)
                    continue
                
                if answer_field not in item:
                    print(f"Error: Answer field '{answer_field}' not found in item. Available: {list(item.keys())}", file=sys.stderr)
                    continue
                
                # Generate unique ID across sections
                if id_field in item and item[id_field]:
                    original_id = str(item[id_field])
                    unique_id = f"hf_s{section_idx}_{original_id}"
                else:
                    unique_id = f"hf_s{section_idx}_item_{items_loaded}"
                
                ids.append(unique_id)
                questions.append(str(item[question_field]))
                answers.append(str(item[answer_field]))
                items_loaded += 1
            else:
                # Loop completed without break - no more data available
                has_more_data = False
            
        except Exception as e:
            print(f"Finished loading HF dataset at section {section_idx}: {e}", file=sys.stderr)
            has_more_data = False
        
        print(f"Loaded {len(ids)} samples from HF dataset section {section_idx}", file=sys.stderr)
    
    elif data_info['type'] == 'json_directory':
        # Load from directory of JSON files
        json_files = data_info['json_files']
        if section_idx >= len(json_files):
            print(f"Section {section_idx} exceeds available JSON files", file=sys.stderr)
            return ids, questions, answers, False
        
        json_file = json_files[section_idx]
        json_file_path = os.path.join(data_info['source'], json_file)
        print(f"Loading JSON file {section_idx + 1}/{len(json_files)}: {json_file}", file=sys.stderr)
        
        with open(json_file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        
        # Handle both list of objects and single object with a list field
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and len(data) == 1:
            # If dict has one key, assume it contains the list
            key = next(iter(data))
            items = data[key] if isinstance(data[key], list) else [data]
        else:
            # Single object, wrap in list
            items = [data]
        
        for i, obj in enumerate(items):
            # Generate unique ID across sections (files)
            if id_field in obj and obj[id_field]:
                original_id = str(obj[id_field])
                unique_id = f"json_f{section_idx}_{original_id}"
            else:
                unique_id = f"json_f{section_idx}_item_{i}"
            
            ids.append(unique_id)
            questions.append(str(obj[question_field]))
            answers.append(str(obj[answer_field]))
        
        has_more_data = (section_idx + 1 < len(json_files))
        print(f"Loaded {len(ids)} samples from JSON file section {section_idx}", file=sys.stderr)
    
    elif data_info['type'] == 'json_file':
        # Load from single JSON file (only one section)
        if section_idx > 0:
            print(f"Single JSON file has only one section", file=sys.stderr)
            return ids, questions, answers, False
        
        json_file_path = data_info['source']
        print(f"Loading single JSON file: {json_file_path}", file=sys.stderr)
        
        with open(json_file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        
        # Handle both list of objects and single object with a list field
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and len(data) == 1:
            # If dict has one key, assume it contains the list
            key = next(iter(data))
            items = data[key] if isinstance(data[key], list) else [data]
        else:
            # Single object, wrap in list
            items = [data]
        
        for i, obj in enumerate(items):
            # Generate unique ID (single file, so no section prefix needed)
            if id_field in obj and obj[id_field]:
                ids.append(str(obj[id_field]))
            else:
                ids.append(f"item_{i}")
            
            questions.append(str(obj[question_field]))
            answers.append(str(obj[answer_field]))
        
        has_more_data = False
        print(f"Loaded {len(ids)} samples from single JSON file", file=sys.stderr)
    
    return ids, questions, answers, has_more_data

###############################################################################
# File I/O utilities for intermediate results
###############################################################################

def save_intermediate_results(results_dir: str, model_name: str, model_idx: int, section_idx: int, data: dict):
    """Save intermediate results for a single model and section."""
    os.makedirs(results_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    filename = f"{results_dir}/model_{model_idx:02d}_{safe_model_name}_section_{section_idx:03d}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved results for {model_name} section {section_idx} to {filename}", file=sys.stderr)


def load_all_intermediate_results(results_dir: str, num_models: int) -> tuple:
    """Load all intermediate results from all sections and reconstruct arrays."""
    import glob
    
    # Get all result files and organize by model and section
    all_files = glob.glob(f"{results_dir}/model_*_section_*.pkl")
    if not all_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    # Organize files by model and section
    model_sections = {}
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Extract model_idx and section_idx from filename
        # Format: model_{model_idx:02d}_{safe_model_name}_section_{section_idx:03d}.pkl
        parts = filename.split('_')
        model_idx = int(parts[1])
        section_idx = int(parts[-1].split('.')[0])  # Last part before .pkl
        
        if model_idx not in model_sections:
            model_sections[model_idx] = {}
        model_sections[model_idx][section_idx] = filepath
    
    print(f"Found results for {len(model_sections)} models", file=sys.stderr)
    
    # Load and concatenate data from all sections
    all_logprobs = []
    all_ensemble_responses = {model_idx: [] for model_idx in range(num_models)}
    all_ids = []
    
    # Get all unique section indices across all models
    all_section_indices = set()
    for model_idx in model_sections:
        all_section_indices.update(model_sections[model_idx].keys())
    all_section_indices = sorted(all_section_indices)
    
    print(f"Processing {len(all_section_indices)} sections: {all_section_indices}", file=sys.stderr)
    
    # Process each section
    for section_idx in all_section_indices:
        print(f"Loading section {section_idx}...", file=sys.stderr)
        
        section_logprobs = None
        section_responses = {}
        section_ids = None
        
        # Load data for each model in this section
        for model_idx in range(num_models):
            if model_idx not in model_sections or section_idx not in model_sections[model_idx]:
                print(f"Warning: Missing data for model {model_idx}, section {section_idx}", file=sys.stderr)
                continue
            
            filepath = model_sections[model_idx][section_idx]
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # Collect responses for this model in this section
            section_responses[model_idx] = data['responses']
            
            # For primary model, also collect logprobs and IDs
            if model_idx == 0:
                if 'logprobs' in data:
                    section_logprobs = data['logprobs']
                if 'ids' in data:
                    section_ids = data['ids']
        
        # Add section data to overall collections
        if section_logprobs is not None:
            all_logprobs.extend(section_logprobs)
        
        if section_ids is not None:
            all_ids.extend(section_ids)
        
        # Add responses for each model
        for model_idx in section_responses:
            all_ensemble_responses[model_idx].extend(section_responses[model_idx])
    
    # Convert to numpy arrays
    total_items = len(all_logprobs) if all_logprobs else max(len(responses) for responses in all_ensemble_responses.values() if responses)
    
    if not all_logprobs:
        all_logprobs = np.zeros(total_items, dtype=np.float32)
    else:
        all_logprobs = np.array(all_logprobs, dtype=np.float32)
    
    # Reconstruct ensemble matrix
    ensemble_correct = np.zeros((num_models, total_items), dtype=np.float32)
    for model_idx in range(num_models):
        if model_idx in all_ensemble_responses and all_ensemble_responses[model_idx]:
            responses = np.array(all_ensemble_responses[model_idx], dtype=np.float32)
            ensemble_correct[model_idx, :len(responses)] = responses
    
    print(f"Loaded {total_items} total items from all sections", file=sys.stderr)
    return all_logprobs, ensemble_correct, all_ids


def cleanup_intermediate_files(results_dir: str):
    """Remove intermediate result files."""
    import glob
    files = glob.glob(f"{results_dir}/model_*_section_*.pkl")
    for file in files:
        try:
            os.remove(file)
            print(f"Cleaned up {file}", file=sys.stderr)
        except OSError:
            pass

###############################################################################
# IRT utilities
###############################################################################

def fit_irt_matrix(matrix: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Fit 1‑PL (Rasch) difficulty β values for each item using `py‑irt`.
    Returns β (len = n_items). If py‑irt is unavailable, falls back to centred logit of accuracy."""

    n_models, n_items = matrix.shape

    if not PY_IRT_AVAILABLE:
        # Fallback: difficulty = logit(1 − mean accuracy), centred
        acc = matrix.mean(axis=0)
        # Use adaptive epsilon based on data distribution to avoid extreme values
        eps = max(1e-6, np.std(acc) * 0.01)
        # Clip accuracy to avoid extreme logit values
        acc_clipped = np.clip(acc, eps, 1 - eps)
        beta = np.log((1 - acc_clipped) / acc_clipped)
        # Center the beta values
        beta = beta - beta.mean()
        return beta

    # Build list of observations for irt.fit
    observations = []
    for subj_idx in range(n_models):
        for item_idx in range(n_items):
            if np.isnan(matrix[subj_idx, item_idx]):
                continue
            observations.append({
                "subject_id": f"m{subj_idx}",
                "item_id": f"q{item_idx}",
                "response": int(matrix[subj_idx, item_idx])
            })

    config = irt.config.IRTRunnerConfig(model_type="rasch", num_epochs=max_iter)
    runner = irt.runners.IRTRunner(config)
    runner.fit(observations)

    item_params = runner.get_item_params()
    beta = np.array([item_params[f"q{i}"]["beta"] for i in range(n_items)], dtype=np.float32)
    return beta

def process_data(args, hf_token):
    all_models = [args.primary_model] + args.ensemble_models

    # Get data sections information
    data_info = get_data_sections_info(
        input_source=args.input,
        dataset_spec=args.dataset_spec,
        max_samples_per_iteration=args.max_samples_per_iteration,
        token=hf_token
    )
    
    print(f"Data type: {data_info['type']}", file=sys.stderr)
    if data_info['type'] in ['json_directory', 'json_file']:
        print(f"Number of sections: {data_info['sections']}", file=sys.stderr)
    
    # Track total samples processed for HF datasets with max_samples limit
    total_samples_processed = 0
    max_samples = args.max_samples
    if max_samples is not None:
        print(f"Maximum samples limit: {max_samples}", file=sys.stderr)

    def process_data_section(section_idx):
        nonlocal total_samples_processed
        
        print(f"\n=== Processing Section {section_idx} ===", file=sys.stderr)
        
        # Calculate remaining samples for HF datasets with limit
        max_samples_remaining = None
        if data_info['type'] == 'huggingface' and max_samples is not None:
            max_samples_remaining = max_samples - total_samples_processed
            if max_samples_remaining <= 0:
                print(f"Maximum samples limit ({max_samples}) reached, stopping", file=sys.stderr)
                return False
        
        # Load dataset section with unique IDs
        ids, questions, answers, has_more_data = load_qa_data_section(
            data_info=data_info,
            section_idx=section_idx,
            id_field=args.id_field,
            question_field=args.question_field,
            answer_field=args.answer_field,
            max_samples_remaining=max_samples_remaining
        )
        
        if not ids:
            print(f"No data found in section {section_idx}", file=sys.stderr)
            return False
        
        n_items = len(ids)
        total_samples_processed += n_items
        
        print(f"Processing {n_items} items with {len(all_models)} models...", file=sys.stderr)
        if max_samples is not None:
            print(f"Total samples processed so far: {total_samples_processed}/{max_samples}", file=sys.stderr)
        
        ###########################################################################
        # Process each model individually for this section
        ###########################################################################
        
        for model_idx, model_name in enumerate(all_models):
            print(f"\n--- Processing Model {model_idx + 1}/{len(all_models)}: {model_name} (Section {section_idx}) ---", file=sys.stderr)
            
            # Clear GPU cache before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model and tokenizer with error handling
            print("Loading tokenizer and model...", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            # Determine model loading parameters based on user args
            dtype = torch.float32 if args.use_float32 else (torch.float16 if args.device.startswith("cuda") else torch.float32)
            
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True
            }
            
            if args.disable_flash_attention:
                model_kwargs["attn_implementation"] = "eager"
                model_kwargs["use_flash_attention_2"] = False
            
            # Try loading model with user-specified settings first, then fallback
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(args.device).eval()
                print(f"Successfully loaded {model_name}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to load {model_name} with user settings: {e}", file=sys.stderr)
                print("Trying with fallback settings...", file=sys.stderr)
                
                # Fallback: disable flash attention and use float32
                fallback_kwargs = {
                    "torch_dtype": torch.float32,
                    "trust_remote_code": True,
                    "attn_implementation": "eager",
                    "use_flash_attention_2": False
                }
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs).to(args.device).eval()
                    print(f"Successfully loaded {model_name} with fallback settings", file=sys.stderr)
                except Exception as e2:
                    print(f"All loading attempts failed for {model_name}: {e2}", file=sys.stderr)
                    print("Skipping this model...", file=sys.stderr)
                    continue
            
            # Storage for this model's results in this section
            model_responses = np.zeros(n_items, dtype=np.float32)
            model_logprobs = None
            
            if model_idx == 0:  # Primary model - compute logprobs
                print("Computing log-probabilities...", file=sys.stderr)
                model_logprobs = np.zeros(n_items, dtype=np.float32)
                
                for i, (q, a) in enumerate(zip(questions, answers)):
                    prompt = f"Question: {q}\nAnswer:"
                    model_logprobs[i] = average_logprob(model, tokenizer, prompt, a, args.device, args.max_sequence_length)
                    if (i + 1) % 500 == 0:
                        print(f"  Logprobs: {i+1}/{n_items} done", file=sys.stderr)
            
            # Generate answers and check correctness for all models
            print("Generating answers...", file=sys.stderr)
            for i, (q, gold) in enumerate(zip(questions, answers)):
                gen = generate_answer(model, tokenizer, q, args.device, args.max_new_tokens, args.max_input_length)
                model_responses[i] = int(compare_answers(gen, gold))
                if (i + 1) % 500 == 0:
                    print(f"  Generation: {i+1}/{n_items} done", file=sys.stderr)
            
            # Save intermediate results for this section
            result_data = {
                'model_name': model_name,
                'model_idx': model_idx,
                'section_idx': section_idx,
                'responses': model_responses,
                'ids': ids  # Save IDs for this section
            }
            if model_logprobs is not None:
                result_data['logprobs'] = model_logprobs
                
            save_intermediate_results(args.temp_dir, model_name, model_idx, section_idx, result_data)
            
            # Clean up model from memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Completed processing {model_name} for section {section_idx}", file=sys.stderr)
        
        return has_more_data

    ###########################################################################
    # Process all data sections
    ###########################################################################
    section_idx = 0
    while True:
        has_more_data = process_data_section(section_idx)
        section_idx += 1
        
        # Stop conditions
        if data_info['type'] == 'json_file' and section_idx >= 1:
            break
        elif data_info['type'] == 'json_directory' and section_idx >= data_info['sections']:
            break
        elif data_info['type'] == 'huggingface' and not has_more_data:
            break
        elif data_info['type'] == 'huggingface' and max_samples is not None and total_samples_processed >= max_samples:
            print(f"Reached maximum samples limit ({max_samples}), stopping processing", file=sys.stderr)
            break
        
        print(f"Completed section {section_idx - 1}, continuing to next section...", file=sys.stderr)

    ###########################################################################
    # Load all results and perform final analysis
    ###########################################################################
    
    print("\n=== Loading all results for final analysis ===", file=sys.stderr)
    logprobs, ensemble_correct, all_ids = load_all_intermediate_results(args.temp_dir, len(all_models))
    
    total_items = len(all_ids)
    print(f"Total items across all sections: {total_items}", file=sys.stderr)
    
    # compute per‑item accuracy (fraction of models correct)
    acc = ensemble_correct.mean(axis=0)

    ###########################################################################
    # IRT difficulty on the response matrix
    ###########################################################################
    print("Fitting IRT...", file=sys.stderr)
    beta = fit_irt_matrix(ensemble_correct)

    ###########################################################################
    # Composite score
    ###########################################################################
    final_score = zscore(-logprobs) + zscore(1 - acc) + zscore(beta)

    ###########################################################################
    # Write json output
    ###########################################################################
    print("Writing output...", file=sys.stderr)
    output_data = []
    for i in range(total_items):
        output_data.append({
            "id": all_ids[i],
            "avg_logprob": float(logprobs[i]),
            "ensemble_acc": float(acc[i]),
            "irt_beta": float(beta[i]),
            "difficulty_z": float(final_score[i])
        })
    
    with open(args.output, "w", encoding="utf8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {args.output}", file=sys.stderr)

    # Clean up intermediate files if requested
    if args.cleanup:
        print("Cleaning up intermediate files...", file=sys.stderr)
        cleanup_intermediate_files(args.temp_dir)

###############################################################################
# Main
###############################################################################

def main(hf_token):
    parser = argparse.ArgumentParser(description="Rank QA pairs by difficulty using sub‑8B models")
    parser.add_argument("--input", required=True, 
                        help="Path to input JSON file, directory of JSON files, OR Hugging Face dataset name")
    parser.add_argument("--output", required=True, help="JSON path for difficulty scores")
    
    # Data loading options
    parser.add_argument("--dataset_spec", default="cot:train",
                        help="Dataset specification: 'split' or 'config:split' for HF datasets (default: cot:train, the cot subset, train split)")
    parser.add_argument("--id_field", default="id",
                        help="Field name for ID in the dataset (default: id)")
    parser.add_argument("--question_field", default="question",
                        help="Field name for question in the dataset (default: question)")
    parser.add_argument("--answer_field", default="answer",
                        help="Field name for answer in the dataset (default: answer)")
    parser.add_argument("--max_samples_per_iteration", type=int, default=10000,
                        help="Maximum number of samples to process per iteration for HF datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to include in the scoring process (only for HF datasets)")
    
    # Model options
    parser.add_argument("--primary_model", default="microsoft/Phi-4-mini-reasoning",
                        help="Model name for log-probability scoring (≤8B)")
    parser.add_argument("--ensemble_models", nargs="*", default=[
                        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        "meta-llama/Llama-3.1-8B"],
                        help="List of additional model names (each ≤8B)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=2048,
                        help="Maximum sequence length for model inputs (reduce if OOM)")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input length for generation (reduce if OOM)")
    parser.add_argument("--disable_flash_attention", action="store_true",
                        help="Disable flash attention for models (use if encountering flash_attn errors)")
    parser.add_argument("--use_float32", action="store_true",
                        help="Use float32 precision instead of float16 (more stable but slower)")
    parser.add_argument("--temp_dir", default="./temp_difficulty_results", 
                        help="Directory to store intermediate results")
    parser.add_argument("--cleanup", action="store_true", 
                        help="Clean up intermediate files after completion")
    args = parser.parse_args()

    process_data(args, hf_token)
    print("Done.", file=sys.stderr)

###############################################################################

if __name__ == "__main__":
    with open("./keys.json", "r") as f:
        keys = json.load(f)
    hf_token = keys["llm"]
    main(hf_token)
