#!/usr/bin/env python3
"""
Fast data selector based on answer length with Gaussian-like distribution.
Selects more long answers, fewer short answers for SFT training.
Supports streaming data processing for large datasets.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
import numpy as np
from datasets import load_dataset
import random


def calculate_length_distribution(lengths, total_samples, min_length_samples=10):
    """
    Calculate sampling probabilities with Gaussian-like distribution.
    Longer answers get higher probability, shorter answers get lower probability.
    """
    min_len, max_len = min(lengths), max(lengths)
    
    # Create length bins
    num_bins = min(50, len(set(lengths)))  # Max 50 bins
    bins = np.linspace(min_len, max_len, num_bins)
    
    # Count items in each bin
    bin_counts = defaultdict(int)
    length_to_bin = {}
    
    for length in lengths:
        bin_idx = np.digitize(length, bins) - 1
        bin_idx = max(0, min(bin_idx, num_bins - 1))
        bin_counts[bin_idx] += 1
        length_to_bin[length] = bin_idx
    
    # Create Gaussian-like weights (peak at longer lengths)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers = np.append(bin_centers, bins[-1])  # Handle last bin
    
    # Weight function: higher weight for longer answers
    # Use exponential function shifted to favor longer answers
    normalized_positions = (bin_centers - min_len) / (max_len - min_len) if max_len > min_len else np.ones_like(bin_centers)
    weights = np.exp(2 * normalized_positions)  # Exponential favoring longer answers
    
    # Calculate samples per bin
    total_weight = sum(weights[i] * min(bin_counts[i], 1) for i in range(len(weights)) if bin_counts[i] > 0)
    
    samples_per_bin = {}
    for i in range(len(weights)):
        if bin_counts[i] > 0:
            target_samples = max(min_length_samples, int(total_samples * weights[i] / total_weight))
            samples_per_bin[i] = min(target_samples, bin_counts[i])
    
    # Create length-to-probability mapping
    length_probs = {}
    for length in lengths:
        bin_idx = length_to_bin[length]
        if bin_idx in samples_per_bin:
            prob = samples_per_bin[bin_idx] / bin_counts[bin_idx]
            length_probs[length] = min(1.0, prob)
        else:
            length_probs[length] = 0.0
    
    return length_probs


class StreamingLengthSelector:
    """Streaming data selector using reservoir sampling with length-based weighting."""
    
    def __init__(self, target_samples, answer_field, sample_size_for_stats=1000, num_bins=6, curve_sharpness=2.0):
        self.target_samples = target_samples
        self.answer_field = answer_field
        self.sample_size_for_stats = sample_size_for_stats
        self.num_bins = num_bins
        self.curve_sharpness = curve_sharpness
        
        # For length distribution estimation
        self.length_samples = []
        self.length_probs = {}
        self.stats_collected = False
        
        # Dynamic bin ranges (will be calculated from data)
        self.bin_edges = None
        self.bin_labels = []
        
        # Reservoir sampling buckets
        self.buckets = defaultdict(list)  # length_bucket -> [items]
        self.bucket_sizes = {}  # length_bucket -> target_size
        
        random.seed(42)
        np.random.seed(42)
    
    def _create_dynamic_bins(self):
        """Create bin ranges based on actual data distribution with open-ended first and last bins."""
        if not self.length_samples:
            return
        
        lengths = np.array(self.length_samples)
        min_len, max_len = lengths.min(), lengths.max()
        
        print(f"Creating dynamic bins from {len(lengths)} samples", file=sys.stderr)
        print(f"Length range: {min_len} - {max_len}", file=sys.stderr)
        
        if self.num_bins <= 2:
            # Special case: if only 1-2 bins, just create open-ended bins
            if self.num_bins == 1:
                self.bin_edges = np.array([0, float('inf')])
                self.bin_labels = ["all"]
            else:  # num_bins == 2
                median_len = np.median(lengths)
                self.bin_edges = np.array([0, median_len, float('inf')])
                self.bin_labels = [f"0-{int(median_len)-1}", f"{int(median_len)}+"]
        else:
            # For 3+ bins: create open-ended first and last bins
            # Use quantiles for the middle bins only
            # Skip the 0% and 100% quantiles, use intermediate quantiles
            middle_bins = self.num_bins - 2  # Exclude first and last bins
            if middle_bins > 0:
                # Create quantiles for middle bins (exclude 0% and 100%)
                quantiles = np.linspace(100/(self.num_bins-1), 100*(self.num_bins-2)/(self.num_bins-1), middle_bins + 1)
                middle_edges = np.percentile(lengths, quantiles)
                
                # Ensure unique edges
                middle_edges = np.unique(middle_edges)
                
                # Create full bin edges: [0, middle_edges..., inf]
                self.bin_edges = np.concatenate([[0], middle_edges, [float('inf')]])
            else:
                # Fallback if something goes wrong
                self.bin_edges = np.array([0, np.median(lengths), float('inf')])
            
            # Ensure we have the right number of bins
            while len(self.bin_edges) - 1 < self.num_bins:
                # Add more edges if needed
                mid_point = np.median(lengths)
                self.bin_edges = np.concatenate([[0], np.linspace(lengths.min(), lengths.max(), self.num_bins-1), [float('inf')]])
                break
            
            # Create bin labels with open-ended first and last bins
            self.bin_labels = []
            for i in range(len(self.bin_edges) - 1):
                if i == 0:
                    # First bin: open-ended start
                    end = int(self.bin_edges[i + 1]) - 1
                    self.bin_labels.append(f"0-{end}")
                elif i == len(self.bin_edges) - 2:
                    # Last bin: open-ended end
                    start = int(self.bin_edges[i])
                    self.bin_labels.append(f"{start}+")
                else:
                    # Middle bins: normal range
                    start = int(self.bin_edges[i])
                    end = int(self.bin_edges[i + 1]) - 1
                    self.bin_labels.append(f"{start}-{end}")
        
        print(f"Created {len(self.bin_edges)-1} open-ended bins:", file=sys.stderr)
        print(f"Bin edges: {self.bin_edges}", file=sys.stderr)
        for i, label in enumerate(self.bin_labels):
            # Count items in each bin
            if i == 0:
                # First bin: everything from 0 to first edge
                count = np.sum(lengths < self.bin_edges[i + 1])
            elif i == len(self.bin_edges) - 2:
                # Last bin: everything from last edge to infinity
                count = np.sum(lengths >= self.bin_edges[i])
            else:
                # Middle bins: normal range
                count = np.sum((lengths >= self.bin_edges[i]) & (lengths < self.bin_edges[i + 1]))
            
            pct = 100 * count / len(lengths)
            print(f"  Bin {i}: {label} chars ({count} samples, {pct:.1f}%)", file=sys.stderr)
    
    def _get_length_bucket(self, length):
        """Map length to bucket for sampling using dynamic bins with open-ended first/last bins."""
        if self.bin_edges is None:
            return 0  # Default bucket if bins not created yet
        
        # Handle open-ended bins
        num_buckets = len(self.bin_edges) - 1
        
        # Debug: For very first few calls, print the logic
        debug_call = hasattr(self, '_debug_bucket_calls')
        if not debug_call:
            self._debug_bucket_calls = 0
        
        if self._debug_bucket_calls < 5:
            print(f"DEBUG bucket assignment: length={length}, bin_edges={self.bin_edges}, num_buckets={num_buckets}", file=sys.stderr)
        
        # First bin: everything less than second edge (0 to bin_edges[1])
        if length < self.bin_edges[1]:
            bucket = 0
            if self._debug_bucket_calls < 5:
                print(f"  -> First bin (bucket 0): {length} < {self.bin_edges[1]}", file=sys.stderr)
        # Last bin: everything >= second-to-last edge
        elif length >= self.bin_edges[-2]:
            bucket = num_buckets - 1
            if self._debug_bucket_calls < 5:
                print(f"  -> Last bin (bucket {bucket}): {length} >= {self.bin_edges[-2]}", file=sys.stderr)
        else:
            # Middle bins: find which middle bin this belongs to
            bucket = num_buckets - 1  # Default to last bucket
            for i in range(1, num_buckets - 1):
                if self.bin_edges[i] <= length < self.bin_edges[i + 1]:
                    bucket = i
                    if self._debug_bucket_calls < 5:
                        print(f"  -> Middle bin (bucket {bucket}): {self.bin_edges[i]} <= {length} < {self.bin_edges[i + 1]}", file=sys.stderr)
                    break
        
        self._debug_bucket_calls += 1
        return bucket
    
    def _calculate_bucket_sizes(self):
        """Calculate target sizes for each bucket based on length distribution."""
        print(f"_calculate_bucket_sizes called. Length samples: {len(self.length_samples)}, bin_edges: {self.bin_edges}", file=sys.stderr)
        
        if not self.length_samples:
            print("No length samples available!", file=sys.stderr)
            return
            
        if self.bin_edges is not None:
            print("Bin edges already exist, skipping bin creation", file=sys.stderr)
        else:
            print("Creating dynamic bins...", file=sys.stderr)
            
        # Create dynamic bins based on data
        try:
            self._create_dynamic_bins()
            print(f"After bin creation: bin_edges = {self.bin_edges}", file=sys.stderr)
        except Exception as e:
            print(f"Error in _create_dynamic_bins: {e}", file=sys.stderr)
            return
        
        if self.bin_edges is None:
            print("ERROR: bin_edges is still None after _create_dynamic_bins!", file=sys.stderr)
            return
        
        # Create bucket counts from samples
        print("Calculating bucket counts from samples...", file=sys.stderr)
        bucket_counts = defaultdict(int)
        try:
            for i, length in enumerate(self.length_samples[:10]):  # Debug first 10
                bucket = self._get_length_bucket(length)
                print(f"  Sample {i}: length={length} -> bucket={bucket}", file=sys.stderr)
            
            for length in self.length_samples:
                bucket = self._get_length_bucket(length)
                bucket_counts[bucket] += 1
                
        except Exception as e:
            print(f"Error calculating bucket counts: {e}", file=sys.stderr)
            return
        
        print(f"Raw bucket counts: {dict(bucket_counts)}", file=sys.stderr)
        
        # Half-Gaussian distribution: longest items most frequent, shortest least frequent
        # This creates a curve that peaks at the longest items and decreases toward shorter items
        num_buckets = len(self.bin_edges) - 1
        bucket_weights = {}
        
        for bucket in range(num_buckets):
            # Normalized position: 0 (shortest) to 1 (longest)
            normalized_pos = bucket / (num_buckets - 1) if num_buckets > 1 else 0.5
            
            # Half-Gaussian curve: f(x) = exp(-k*(1-x)^2) where x=0 is shortest, x=1 is longest
            # This creates a curve that peaks at x=1 (longest) and decays toward x=0 (shortest)
            k = self.curve_sharpness  # Controls the sharpness of the curve (higher k = sharper peak)
            gaussian_weight = np.exp(-k * (1 - normalized_pos) ** 2)
            
            # Scale to ensure minimum weight is reasonable (avoid zero weights)
            min_weight = 0.1
            max_weight = 2.0
            scaled_weight = min_weight + (max_weight - min_weight) * gaussian_weight
            
            bucket_weights[bucket] = scaled_weight
            
        print(f"Half-Gaussian weights (shortest->longest): {[f'{bucket_weights[i]:.2f}' for i in range(num_buckets)]}", file=sys.stderr)
        
        # Calculate target samples per bucket
        total_weight = sum(bucket_weights.get(bucket, 1.0) * min(count, 1) 
                          for bucket, count in bucket_counts.items())
        
        min_samples_per_bucket = max(10, self.target_samples // (num_buckets * 2))
        
        print(f"Starting bucket size calculation...", file=sys.stderr)
        for bucket, count in bucket_counts.items():
            print(f"  Processing bucket {bucket} with count {count}", file=sys.stderr)
            if count > 0:
                weight = bucket_weights.get(bucket, 1.0)
                target = max(min_samples_per_bucket, 
                           int(self.target_samples * weight / total_weight))
                final_target = min(target, count * 10)  # Don't exceed reasonable limits
                self.bucket_sizes[bucket] = final_target
                print(f"    Bucket {bucket}: weight={weight}, target={target}, final={final_target}", file=sys.stderr)
        
        print(f"Bucket counts from samples: {dict(bucket_counts)}", file=sys.stderr)
        print(f"Bucket weights: {bucket_weights}", file=sys.stderr)
        print(f"Total weight: {total_weight}", file=sys.stderr)
        print(f"Min samples per bucket: {min_samples_per_bucket}", file=sys.stderr)
        print(f"Final bucket target sizes: {dict(self.bucket_sizes)}", file=sys.stderr)
    
    def process_item(self, item, item_idx):
        """Process a single item from the stream."""
        if self.answer_field not in item:
            if item_idx < 10:  # Debug first few items
                print(f"Item {item_idx}: Missing field '{self.answer_field}', available: {list(item.keys())}", file=sys.stderr)
            return
        
        answer = str(item[self.answer_field])
        length = len(answer)
        
        # Collect samples for statistics (first N items)
        if not self.stats_collected and len(self.length_samples) < self.sample_size_for_stats:
            self.length_samples.append(length)
            
            # Once we have enough samples, calculate bucket sizes
            if len(self.length_samples) >= self.sample_size_for_stats:
                print(f"About to calculate bucket sizes...", file=sys.stderr)
                self._calculate_bucket_sizes()
                self.stats_collected = True
                print(f"Length distribution calculated from {len(self.length_samples)} samples", file=sys.stderr)
                print(f"Length range: {min(self.length_samples)} - {max(self.length_samples)}", file=sys.stderr)
                print(f"Stats collection completed. Bucket sizes: {dict(self.bucket_sizes)}", file=sys.stderr)
        
        # Skip if we haven't collected stats yet
        if not self.stats_collected:
            return
        
        bucket = self._get_length_bucket(length)
        target_size = self.bucket_sizes.get(bucket, 0)
        
        # Debug info for first few items after stats collection
        if item_idx < self.sample_size_for_stats + 10:
            print(f"Item {item_idx}: length={length}, bucket={bucket}, target_size={target_size}", file=sys.stderr)
        
        if target_size == 0:
            if item_idx < self.sample_size_for_stats + 10:
                print(f"Item {item_idx}: Skipping because target_size=0 for bucket {bucket}", file=sys.stderr)
            return
        
        # Reservoir sampling for this bucket
        bucket_items = self.buckets[bucket]
        
        if len(bucket_items) < target_size:
            # Bucket not full, add item
            bucket_items.append(item)
            # if item_idx < self.sample_size_for_stats + 20:
            #     print(f"Item {item_idx}: Added to bucket {bucket} (now {len(bucket_items)}/{target_size})", file=sys.stderr)
        else:
            # Bucket full, randomly replace with proper reservoir sampling probability
            # The probability of replacing should be target_size / (number of items seen for this bucket)
            # For simplicity, we'll use item_idx as an approximation
            items_seen_for_bucket = item_idx - self.sample_size_for_stats + 1
            if items_seen_for_bucket > 0:
                replace_prob = target_size / items_seen_for_bucket
                if random.random() < replace_prob:
                    replace_idx = random.randint(0, target_size - 1)
                    bucket_items[replace_idx] = item
                    if item_idx < self.sample_size_for_stats + 20:
                        print(f"Item {item_idx}: Replaced item in bucket {bucket} at index {replace_idx}", file=sys.stderr)
    
    def get_selected_items(self):
        """Get final selected items from all buckets."""
        selected = []
        for items in self.buckets.values():
            selected.extend(items)
        return selected
    
    def print_selection_summary(self):
        """Print detailed summary of selection process."""
        print(f"\n=== Selection Summary ===", file=sys.stderr)
        print(f"Stats collected: {self.stats_collected}", file=sys.stderr)
        print(f"Sample size for stats: {self.sample_size_for_stats}", file=sys.stderr)
        print(f"Length samples collected: {len(self.length_samples)}", file=sys.stderr)
        
        if self.bin_edges is not None:
            print(f"Bin edges: {self.bin_edges}", file=sys.stderr)
            print(f"Number of bins: {len(self.bin_edges) - 1}", file=sys.stderr)
        else:
            print("Bin edges: None (not created)", file=sys.stderr)
        
        print(f"Bucket sizes targets: {dict(self.bucket_sizes)}", file=sys.stderr)
        
        total_selected = 0
        for bucket_id in sorted(self.buckets.keys()):
            items = self.buckets[bucket_id]
            target = self.bucket_sizes.get(bucket_id, 0)
            print(f"Bucket {bucket_id}: {len(items)}/{target} items", file=sys.stderr)
            total_selected += len(items)
        
        print(f"Total selected: {total_selected}", file=sys.stderr)
        print(f"========================", file=sys.stderr)


def select_data_hf_streaming(dataset_name, dataset_config, split, answer_field, total_samples, token=None, shuffle=False, sample_size_for_stats=1000, num_bins=6, curve_sharpness=2.0):
    """Select data from Hugging Face dataset using streaming."""
    print(f"Loading dataset: {dataset_name}", file=sys.stderr)
    
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True, token=token)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True, token=token)
    
    # Shuffle dataset if requested
    if shuffle:
        print("Shuffling dataset...", file=sys.stderr)
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
    
    # Initialize streaming selector
    selector = StreamingLengthSelector(total_samples, answer_field, sample_size_for_stats, num_bins, curve_sharpness)
    
    iteration_mode = "random" if shuffle else "sequential"
    print(f"Processing streaming data ({iteration_mode} order)...", file=sys.stderr)
    
    # Process items one by one
    for item_idx, item in enumerate(dataset):
        selector.process_item(item, item_idx)
        
        if (item_idx + 1) % 10000 == 0:
            current_selected = len(selector.get_selected_items())
            # Show bucket status
            bucket_status = {}
            for bucket_id, items in selector.buckets.items():
                bucket_status[bucket_id] = len(items)
            print(f"Processed {item_idx + 1} items, selected {current_selected} total", file=sys.stderr)
            print(f"  Bucket status: {bucket_status}", file=sys.stderr)
            print(f"  Bucket targets: {dict(selector.bucket_sizes)}", file=sys.stderr)
        
        # Optional: stop early if we have enough data and have seen enough diversity
        if item_idx > 100000 and len(selector.get_selected_items()) >= total_samples:
            print(f"Early stopping at {item_idx + 1} items", file=sys.stderr)
            break
    
    selected_items = selector.get_selected_items()
    print(f"Final selection: {len(selected_items)} items", file=sys.stderr)
    
    # Print detailed summary for debugging
    selector.print_selection_summary()
    
    return selected_items


def select_data_json(file_path, answer_field, total_samples):
    """Select data from JSON file."""
    print(f"Loading JSON file: {file_path}", file=sys.stderr)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and len(data) == 1:
        key = next(iter(data))
        items = data[key] if isinstance(data[key], list) else [data]
    else:
        items = [data]
    
    # Calculate lengths
    lengths = []
    valid_items = []
    
    for item in items:
        if answer_field not in item:
            continue
        answer = str(item[answer_field])
        length = len(answer)
        lengths.append(length)
        valid_items.append(item)
    
    print(f"Found {len(valid_items)} items", file=sys.stderr)
    print(f"Length range: {min(lengths)} - {max(lengths)}", file=sys.stderr)
    
    # Calculate selection probabilities
    length_probs = calculate_length_distribution(lengths, total_samples)
    
    # Select items
    selected_items = []
    np.random.seed(42)
    
    for item, length in zip(valid_items, lengths):
        prob = length_probs.get(length, 0.0)
        if np.random.random() < prob:
            selected_items.append(item)
    
    print(f"Selected {len(selected_items)} items", file=sys.stderr)
    return selected_items


def print_length_stats(items, answer_field):
    """Print statistics about selected data lengths."""
    lengths = [len(str(item[answer_field])) for item in items]
    
    print(f"\nSelected data statistics:", file=sys.stderr)
    print(f"Total items: {len(items)}", file=sys.stderr)
    print(f"Length stats:", file=sys.stderr)
    print(f"  Min: {min(lengths)}", file=sys.stderr)
    print(f"  Max: {max(lengths)}", file=sys.stderr)
    print(f"  Mean: {np.mean(lengths):.1f}", file=sys.stderr)
    print(f"  Median: {np.median(lengths):.1f}", file=sys.stderr)
    print(f"  Std: {np.std(lengths):.1f}", file=sys.stderr)
    
    # Show length distribution using dynamic quantile-based bins
    if lengths:
        # Create 8 bins based on quantiles for display
        quantiles = [0, 10, 25, 40, 60, 75, 90, 100]
        bin_edges = np.percentile(lengths, quantiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        print(f"\nLength distribution (quantile-based):", file=sys.stderr)
        for i in range(len(bin_edges) - 1):
            bin_start = int(bin_edges[i])
            bin_end = int(bin_edges[i + 1])
            
            if i == len(bin_edges) - 2:  # Last bin
                count = sum(1 for l in lengths if l >= bin_start)
                label = f"{bin_start}+"
            else:
                count = sum(1 for l in lengths if bin_start <= l < bin_end)
                label = f"{bin_start}-{bin_end-1}"
            
            pct = 100 * count / len(lengths)
            print(f"  {label}: {count} ({pct:.1f}%)", file=sys.stderr)


def main(hf_token):
    parser = argparse.ArgumentParser(description="Select data by answer length with Gaussian-like distribution")
    parser.add_argument("--input", required=True,
                        help="Input: HF dataset name or JSON file path")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path")
    parser.add_argument("--answer_field", default="answer",
                        help="Field name for answer (default: answer)")
    parser.add_argument("--total_samples", type=int, default=1000,
                        help="Target number of samples to select")
    
    # HF dataset options
    parser.add_argument("--dataset_config",
                        help="Dataset config for HF datasets")
    parser.add_argument("--split", default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle dataset before processing (random order vs sequential)")
    parser.add_argument("--sample_size_for_stats", type=int, default=1000,
                        help="Number of samples to analyze for creating dynamic bins (default: 1000)")
    parser.add_argument("--num_bins", type=int, default=6,
                        help="Number of length bins to create (default: 6)")
    parser.add_argument("--curve_sharpness", type=float, default=2.0,
                        help="Sharpness of half-Gaussian curve (higher = sharper peak at longest items, default: 2.0)")
    
    args = parser.parse_args()
    
    # Determine input type and select data
    if os.path.exists(args.input):
        # JSON file
        selected_items = select_data_json(args.input, args.answer_field, args.total_samples)
    else:
        # HF dataset - use streaming version
        selected_items = select_data_hf_streaming(
            args.input, args.dataset_config, args.split, 
            args.answer_field, args.total_samples, hf_token, args.shuffle,
            args.sample_size_for_stats, args.num_bins, args.curve_sharpness
        )
    
    # Print statistics
    print_length_stats(selected_items, args.answer_field)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(selected_items)} selected items to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    with open("./keys.json", "r") as f:
        keys = json.load(f)
        hf_token = keys["llm"]
    main(hf_token)