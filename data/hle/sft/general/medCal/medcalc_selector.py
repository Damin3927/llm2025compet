#!/usr/bin/env python3
"""
Data selector for MedCalc-Bench dataset with half-gaussian distribution favoring longer explanations.
Based on reasonmd_selector.py but adapted for MedCalc-Bench dataset structure.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
import numpy as np
from datasets import load_dataset
import random


class MedCalcLengthSelector:
    """Length-based selector for MedCalc-Bench dataset favoring longer explanations."""
    
    def __init__(self, target_samples=500, sample_size_for_stats=2000, num_bins=6, curve_sharpness=3.0):
        self.target_samples = target_samples
        self.sample_size_for_stats = sample_size_for_stats
        self.num_bins = num_bins
        self.curve_sharpness = curve_sharpness
        
        # For length distribution estimation
        self.length_samples = []
        self.stats_collected = False
        
        # Dynamic bin ranges
        self.bin_edges = None
        self.bin_labels = []
        
        # Reservoir sampling buckets
        self.buckets = defaultdict(list)  # bucket_id -> [items]
        self.bucket_sizes = {}  # bucket_id -> target_size
        
        random.seed(42)
        np.random.seed(42)
    
    def _create_dynamic_bins(self):
        """Create bin ranges based on actual data distribution."""
        if not self.length_samples:
            return
        
        lengths = np.array(self.length_samples)
        min_len, max_len = lengths.min(), lengths.max()
        
        print(f"Creating {self.num_bins} bins from {len(lengths)} samples", file=sys.stderr)
        print(f"Length range: {min_len} - {max_len}", file=sys.stderr)
        
        if self.num_bins <= 2:
            if self.num_bins == 1:
                self.bin_edges = np.array([0, float('inf')])
                self.bin_labels = ["all"]
            else:
                median_len = np.median(lengths)
                self.bin_edges = np.array([0, median_len, float('inf')])
                self.bin_labels = [f"0-{int(median_len)-1}", f"{int(median_len)}+"]
        else:
            # Create quantile-based bins with open-ended first and last bins
            middle_bins = self.num_bins - 2
            if middle_bins > 0:
                quantiles = np.linspace(100/(self.num_bins-1), 100*(self.num_bins-2)/(self.num_bins-1), middle_bins + 1)
                middle_edges = np.percentile(lengths, quantiles)
                middle_edges = np.unique(middle_edges)
                self.bin_edges = np.concatenate([[0], middle_edges, [float('inf')]])
            else:
                self.bin_edges = np.array([0, np.median(lengths), float('inf')])
            
            # Create bin labels
            self.bin_labels = []
            for i in range(len(self.bin_edges) - 1):
                if i == 0:
                    end = int(self.bin_edges[i + 1]) - 1
                    self.bin_labels.append(f"0-{end}")
                elif i == len(self.bin_edges) - 2:
                    start = int(self.bin_edges[i])
                    self.bin_labels.append(f"{start}+")
                else:
                    start = int(self.bin_edges[i])
                    end = int(self.bin_edges[i + 1]) - 1
                    self.bin_labels.append(f"{start}-{end}")
        
        print(f"Bin edges: {self.bin_edges}", file=sys.stderr)
        for i, label in enumerate(self.bin_labels):
            if i == 0:
                count = np.sum(lengths < self.bin_edges[i + 1])
            elif i == len(self.bin_edges) - 2:
                count = np.sum(lengths >= self.bin_edges[i])
            else:
                count = np.sum((lengths >= self.bin_edges[i]) & (lengths < self.bin_edges[i + 1]))
            
            pct = 100 * count / len(lengths)
            print(f"  Bin {i}: {label} chars ({count} samples, {pct:.1f}%)", file=sys.stderr)
    
    def _get_length_bucket(self, length):
        """Map length to bucket using dynamic bins."""
        if self.bin_edges is None:
            return 0
        
        num_buckets = len(self.bin_edges) - 1
        
        # First bin: everything less than second edge
        if length < self.bin_edges[1]:
            return 0
        # Last bin: everything >= second-to-last edge
        elif length >= self.bin_edges[-2]:
            return num_buckets - 1
        else:
            # Middle bins
            for i in range(1, num_buckets - 1):
                if self.bin_edges[i] <= length < self.bin_edges[i + 1]:
                    return i
            return num_buckets - 1
    
    def _calculate_bucket_sizes(self):
        """Calculate target sizes for each bucket with half-gaussian distribution."""
        if not self.length_samples:
            return
        
        self._create_dynamic_bins()
        
        if self.bin_edges is None:
            return
        
        # Calculate bucket counts from samples
        bucket_counts = defaultdict(int)
        for length in self.length_samples:
            bucket = self._get_length_bucket(length)
            bucket_counts[bucket] += 1
        
        print(f"Raw bucket counts: {dict(bucket_counts)}", file=sys.stderr)
        
        # Half-Gaussian distribution: longest items get highest weight
        num_buckets = len(self.bin_edges) - 1
        bucket_weights = {}
        
        for bucket in range(num_buckets):
            # Normalized position: 0 (shortest) to 1 (longest)
            normalized_pos = bucket / (num_buckets - 1) if num_buckets > 1 else 0.5
            
            # Half-Gaussian: peaks at longest (x=1), decays toward shortest (x=0)
            k = self.curve_sharpness
            gaussian_weight = np.exp(-k * (1 - normalized_pos) ** 2)
            
            # Scale weights to reasonable range
            min_weight = 0.2
            max_weight = 3.0
            scaled_weight = min_weight + (max_weight - min_weight) * gaussian_weight
            
            bucket_weights[bucket] = scaled_weight
        
        print(f"Half-Gaussian weights (shortest->longest): {[f'{bucket_weights[i]:.2f}' for i in range(num_buckets)]}", file=sys.stderr)
        
        # Calculate target samples per bucket
        total_weight = sum(bucket_weights.get(bucket, 1.0) * min(count, 1) 
                          for bucket, count in bucket_counts.items())
        
        min_samples_per_bucket = max(10, self.target_samples // (num_buckets * 3))
        
        for bucket, count in bucket_counts.items():
            if count > 0:
                weight = bucket_weights.get(bucket, 1.0)
                target = max(min_samples_per_bucket, 
                           int(self.target_samples * weight / total_weight))
                final_target = min(target, count * 2)  # Don't exceed reasonable limits
                self.bucket_sizes[bucket] = final_target
        
        print(f"Final bucket target sizes: {dict(self.bucket_sizes)}", file=sys.stderr)
    
    def process_item(self, item, item_idx):
        """Process a single item from the stream."""
        # Check required fields
        if 'Ground Truth Explanation' not in item:
            return
        
        explanation_text = str(item['Ground Truth Explanation'])
        length = len(explanation_text)
        
        # Collect samples for statistics
        if not self.stats_collected and len(self.length_samples) < self.sample_size_for_stats:
            self.length_samples.append(length)
            
            if len(self.length_samples) >= self.sample_size_for_stats:
                self._calculate_bucket_sizes()
                self.stats_collected = True
                print(f"Statistics collected from {len(self.length_samples)} samples", file=sys.stderr)
        
        # Skip if we haven't collected stats yet
        if not self.stats_collected:
            return
        
        bucket = self._get_length_bucket(length)
        target_size = self.bucket_sizes.get(bucket, 0)
        
        if target_size == 0:
            return
        
        # Reservoir sampling for this bucket
        bucket_items = self.buckets[bucket]
        
        if len(bucket_items) < target_size:
            bucket_items.append(item)
        else:
            # Reservoir sampling: replace with probability target_size / items_seen
            items_seen_for_bucket = item_idx - self.sample_size_for_stats + 1
            if items_seen_for_bucket > 0:
                replace_prob = target_size / items_seen_for_bucket
                if random.random() < replace_prob:
                    replace_idx = random.randint(0, target_size - 1)
                    bucket_items[replace_idx] = item
    
    def get_selected_items(self):
        """Get final selected items from all buckets."""
        selected = []
        for items in self.buckets.values():
            selected.extend(items)
        return selected
    
    def print_selection_summary(self):
        """Print summary of selection process."""
        print(f"\n=== Selection Summary ===", file=sys.stderr)
        print(f"Target samples: {self.target_samples}", file=sys.stderr)
        print(f"Stats sample size: {self.sample_size_for_stats}", file=sys.stderr)
        print(f"Number of bins: {self.num_bins}", file=sys.stderr)
        print(f"Curve sharpness: {self.curve_sharpness}", file=sys.stderr)
        
        if self.bin_edges is not None:
            print(f"Bin edges: {self.bin_edges}", file=sys.stderr)
        
        total_selected = 0
        for bucket_id in sorted(self.buckets.keys()):
            items = self.buckets[bucket_id]
            target = self.bucket_sizes.get(bucket_id, 0)
            label = self.bin_labels[bucket_id] if bucket_id < len(self.bin_labels) else f"bucket_{bucket_id}"
            print(f"Bin {bucket_id} ({label}): {len(items)}/{target} items", file=sys.stderr)
            total_selected += len(items)
        
        print(f"Total selected: {total_selected}", file=sys.stderr)
        print(f"========================", file=sys.stderr)


def select_medcalc_data(target_samples=500, sample_size_for_stats=2000, num_bins=6, curve_sharpness=3.0, hf_token=None):
    """Select data from MedCalc-Bench dataset with half-gaussian distribution."""
    print(f"Loading MedCalc-Bench dataset...", file=sys.stderr)
    
    dataset = load_dataset("ncbi/MedCalc-Bench-v1.0", streaming=True, token=hf_token)
    train_dataset = dataset['train']
    
    # Initialize selector
    selector = MedCalcLengthSelector(target_samples, sample_size_for_stats, num_bins, curve_sharpness)
    
    print(f"Processing streaming data...", file=sys.stderr)
    
    # Process items
    for item_idx, item in enumerate(train_dataset):
        selector.process_item(item, item_idx)
        
        if (item_idx + 1) % 2000 == 0:
            current_selected = len(selector.get_selected_items())
            print(f"Processed {item_idx + 1} items, selected {current_selected} total", file=sys.stderr)
        
        # Stop early if we have enough data and have processed enough items
        if item_idx > 10000 and len(selector.get_selected_items()) >= target_samples:
            print(f"Early stopping at {item_idx + 1} items", file=sys.stderr)
            break
    
    selected_items = selector.get_selected_items()
    print(f"Final selection: {len(selected_items)} items", file=sys.stderr)
    
    selector.print_selection_summary()
    return selected_items


def print_length_stats(items):
    """Print statistics about selected data lengths."""
    lengths = [len(str(item['Ground Truth Explanation'])) for item in items]
    
    print(f"\nSelected data statistics:", file=sys.stderr)
    print(f"Total items: {len(items)}", file=sys.stderr)
    print(f"Explanation length stats:", file=sys.stderr)
    print(f"  Min: {min(lengths)}", file=sys.stderr)
    print(f"  Max: {max(lengths)}", file=sys.stderr)
    print(f"  Mean: {np.mean(lengths):.1f}", file=sys.stderr)
    print(f"  Median: {np.median(lengths):.1f}", file=sys.stderr)
    print(f"  Std: {np.std(lengths):.1f}", file=sys.stderr)
    
    # Show length distribution
    if lengths:
        quantiles = [0, 10, 25, 40, 60, 75, 90, 100]
        bin_edges = np.percentile(lengths, quantiles)
        bin_edges = np.unique(bin_edges)
        
        print(f"\nLength distribution:", file=sys.stderr)
        for i in range(len(bin_edges) - 1):
            bin_start = int(bin_edges[i])
            bin_end = int(bin_edges[i + 1])
            
            if i == len(bin_edges) - 2:
                count = sum(1 for l in lengths if l >= bin_start)
                label = f"{bin_start}+"
            else:
                count = sum(1 for l in lengths if bin_start <= l < bin_end)
                label = f"{bin_start}-{bin_end-1}"
            
            pct = 100 * count / len(lengths)
            print(f"  {label}: {count} ({pct:.1f}%)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Select MedCalc-Bench data with half-gaussian distribution favoring longer explanations")
    parser.add_argument("--output", default="medcalc_selected.json",
                        help="Output JSON file path (default: medcalc_selected.json)")
    parser.add_argument("--target_samples", type=int, default=500,
                        help="Target number of samples to select (default: 500)")
    parser.add_argument("--sample_size_for_stats", type=int, default=2000,
                        help="Number of samples to analyze for creating bins (default: 2000)")
    parser.add_argument("--num_bins", type=int, default=6,
                        help="Number of length bins to create (default: 6)")
    parser.add_argument("--curve_sharpness", type=float, default=3.0,
                        help="Sharpness of half-Gaussian curve (higher = sharper peak at longest, default: 3.0)")
    
    args = parser.parse_args()
    
    # Load HF token
    try:
        with open("../keys.json", "r") as f:
            keys = json.load(f)
            hf_token = keys["llm"]
    except FileNotFoundError:
        print("Warning: keys.json not found, proceeding without HF token", file=sys.stderr)
        hf_token = None
    
    # Select data
    selected_items = select_medcalc_data(
        args.target_samples, args.sample_size_for_stats, 
        args.num_bins, args.curve_sharpness, hf_token
    )
    
    # Print statistics
    print_length_stats(selected_items)
    
    # Save results (keep original structure)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(selected_items)} selected items to {args.output}", file=sys.stderr)
    
    # Print sample structure
    if selected_items:
        print(f"Sample output structure:", file=sys.stderr)
        sample_item = selected_items[0]
        print(f"  Fields: {list(sample_item.keys())}", file=sys.stderr)
        for field in ["Calculator Name", "Question", "Ground Truth Explanation"]:
            if field in sample_item:
                field_content = str(sample_item[field])
                preview = field_content[:100] + "..." if len(field_content) > 100 else field_content
                print(f"  {field}: {preview}", file=sys.stderr)


if __name__ == "__main__":
    main()