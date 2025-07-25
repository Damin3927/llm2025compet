"""
OpenMathReasoning Dataset Label-Based Filtering

This script performs efficient dataset filtering using pre-computed labels instead of LLM inference.
The filtering process includes:

1. Primary Filter: Select only problems with 'has_answer_extracted' label
   - Ensures we only process problems with valid mathematical answers
   
2. Difficulty Filter: Filter by pass rate threshold (default: <= 0.3)
   - Lower pass rates indicate harder problems
   - Uses pre-computed 'pass_rate_72b_tir' from 72B model evaluation
   
3. Complexity Ranking: Sort by pass_rate_72b_tir in ascending order, and then by the length of generated_solution in descending order
   - Longer solutions typically indicate more complex mathematical reasoning
   - Helps prioritize challenging problems for training

Benefits over LLM-based filtering:
- Much faster execution (no model inference required)
- Deterministic results (no sampling variance)
- Lower computational cost
- Preserves original problem-solution pairs

Output: Filtered and ranked dataset saved in JSON format for downstream training

Author: Junyu
Date: 2025-07-25
"""


#%%
import os
import datasets
import torch
import json
from tqdm import tqdm
import argparse

# hard coding the dataset size
cot_dataset_size = 3.3e6
genselect_dataset_size = 5.66e5

def parse_args():
    parser = argparse.ArgumentParser(description='OpenMathReasoning Filtering Script by label')
    
    # Run configuration
    parser.add_argument('--filter-by-pass-rate', type=float, default=0.2,
                      help='pass rate threshold for filtering the dataset')
    parser.add_argument('--save-per-iteration', type=int, default=10000,
                      help='save the dataset every N iterations')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='batch size for filtering the dataset')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
print(args)


# create the output directory
output_dir = "./results"
filtered_dataset_dir = f"{output_dir}/filtered_dataset"
os.makedirs(filtered_dataset_dir, exist_ok=True)


#%%
def filter_dataset_by_correctness(dataset, filter_by_pass_rate, save_per_iteration, save_dir):
    # filter the dataset by the pass rate
    filtered_dataset = dataset.filter(lambda x: x['problem_type'] == 'has_answer_extracted')

    save_files = sorted(os.listdir(save_dir))
    if len(save_files) == 0:
        iter_number = 0
    else:
        # find the last save file
        last_save_file = save_files[-1]
        # find the iter number from the save file name
        iter_number = int(last_save_file.split("_")[-1].split(".")[0])

    i = 1
    filtered_dataset_collection = []
    for data_batch in filtered_dataset.iter(batch_size=args.batch_size):
        if i <= iter_number:
            if i % save_per_iteration == 0:
                print(f"Skipping {i} batches")
            i += 1
            continue
        # data_batch is a dictionary of lists, convert it to a list of dictionaries
        data_batch = [{k: v[j] for k, v in data_batch.items()} for j in range(len(data_batch['problem']))]
        for data in data_batch:
            if 'n/a' in data['pass_rate_72b_tir'] or float(data['pass_rate_72b_tir']) > filter_by_pass_rate:
                continue
            filtered_dataset_collection.append(data)

        # save the filtered dataset
        if i % save_per_iteration == 0:
            # sort the filtered dataset collection by pass_rate_72b_tir in ascending order, and then by the length of generated_solution in descending order
            filtered_dataset_collection.sort(key=lambda x: (1-float(x['pass_rate_72b_tir']), len(x['generated_solution'])), reverse=True)
            with open(f"{save_dir}/filtered_dataset_{i}.json", "w") as f:
                json.dump(filtered_dataset_collection, f, indent=4, ensure_ascii=False)
            filtered_dataset_collection = []
            print(f"Saved filtered dataset {i} batches")
        i += 1


 #%%
 # load and filter the dataset
if __name__ == "__main__":
    cot_result_dir = f"{filtered_dataset_dir}/cot"
    if not os.path.exists(cot_result_dir):
        os.makedirs(cot_result_dir, exist_ok=True)
    cot_dataset = datasets.load_dataset("nvidia/OpenMathReasoning", split='cot', streaming=True)
    filtered_dataset = filter_dataset_by_correctness(cot_dataset, args.filter_by_pass_rate, args.save_per_iteration, cot_result_dir)
    del cot_dataset

    genselect_result_dir = f"{filtered_dataset_dir}/genselect"
    if not os.path.exists(genselect_result_dir):
        os.makedirs(genselect_result_dir, exist_ok=True)
    genselect_dataset = datasets.load_dataset("nvidia/OpenMathReasoning", split='genselect', streaming=True)
    filtered_dataset = filter_dataset_by_correctness(genselect_dataset, args.filter_by_pass_rate, args.save_per_iteration, genselect_result_dir)
    del genselect_dataset