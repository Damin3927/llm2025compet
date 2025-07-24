#%%
"""
this is the script to score the difficulty of the questions
we only use the cot and genselect splits since tool use is not allowed
process:
1. load the dataset
2. filter the dataset to only include the cot and genselect splits
3. score the difficulty of the questions with an inference LLM n times
4. check the correctness of the answers with a judgement LLM
5. score the questions difficulty based on the correct hit rate
6. save the dataset
"""


#%%
import os
import 
import datasets
import torch
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"

# hard coding the dataset size
cot_dataset_size = 3.3e6
genselect_dataset_size = 5.66e5

def parse_args():
    parser = argparse.ArgumentParser(description='OpenMathReasoning Filtering Script')
    
    # Run configuration
    parser.add_argument('--run-index', type=int, default=1,
                      help='Run index for multiple script executions')
    parser.add_argument('--filter-by-pass-rate', type=float, default=0.3,
                      help='pass rate threshold for filtering the dataset')
    
    # Inference model parameters
    parser.add_argument('--inference-model', type=str, default="Qwen/Qwen3-8B",
                      help='Model to use for answering questions')
    parser.add_argument('--inference-temperature', type=float, default=0.3,
                      help='Temperature for inference sampling')
    parser.add_argument('--inference-max-tokens', type=int, default=4096,
                      help='Maximum tokens to generate during inference')
    parser.add_argument('--inference-max-model-len', type=int, default=8192,
                      help='Maximum model input length for inference')
    parser.add_argument('--inference-batch-size', type=int, default=4,
                      help='Batch size for inference')
    parser.add_argument('--inference-tp', type=int, default=1,
                      help='Tensor parallel size for inference')
    parser.add_argument('--inference-pp', type=int, default=1,
                      help='Pipeline parallel size for inference')
    parser.add_argument('--save-per-batch', type=int, default=100,
                      help='Save results every N batches')
    
    # Judgment model parameters
    parser.add_argument('--judgement-model', type=str, default="Qwen/Qwen3-8B",
                      help='Model to use for judging answers')
    parser.add_argument('--judgement-temperature', type=float, default=0.0,
                      help='Temperature for judgment sampling')
    parser.add_argument('--judgement-max-tokens', type=int, default=50,
                      help='Maximum tokens to generate during judgment')
    parser.add_argument('--judgement-max-model-len', type=int, default=8192,
                      help='Maximum model input length for judgment')
    parser.add_argument('--judgement-batch-size', type=int, default=4,
                      help='Batch size for judgment')
    parser.add_argument('--judgement-tp', type=int, default=1,
                      help='Tensor parallel size for judgment')
    parser.add_argument('--judgement-pp', type=int, default=1,
                      help='Pipeline parallel size for judgment')
    parser.add_argument('--judge-only-by-answer', action='store_true',
                      help='If set, only judge the answers without reasoning')
    # start from percentage
    parser.add_argument('--start-from-percentage', type=float, default=0,
                      help='Start from the percentage (0.5 = 50%) of the dataset, so we can run the script separately')
    parser.add_argument('--end-at-percentage', type=float, default=1.0,
                      help='End at the percentage (1.0 = 100%) of the dataset, so we can run the script separately')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
print(args)

# Assign parameters from command line arguments
run_index = args.run_index

inference_model = args.inference_model
inference_temperature = args.inference_temperature
inference_max_tokens = args.inference_max_tokens
inference_batch_size = args.inference_batch_size
inference_tp, inference_pp = args.inference_tp, args.inference_pp
save_per_batch = args.save_per_batch

judgement_model = args.judgement_model
judgement_temperature = args.judgement_temperature
judgement_max_tokens = args.judgement_max_tokens
judgement_batch_size = args.judgement_batch_size
judgement_tp, judgement_pp = args.judgement_tp, args.judgement_pp
judge_only_by_answer = args.judge_only_by_answer


# create the output directory
output_dir = "./results"
inference_dir = f"{output_dir}/inference/run_{run_index}" # save temporary inference results
judgement_dir = f"{output_dir}/judgement/run_{run_index}" # save temporary judgement results
os.makedirs(inference_dir, exist_ok=True)
os.makedirs(judgement_dir, exist_ok=True)

#%% prompts
# write a prompt for the inference LLM to answer the questions three times
inference_cot_prompt = (
    "You are a highly skilled mathematician known for clear and rigorous reasoning.\n"
    "Given the following math question, provide a step-by-step analysis of your thought process, followed by the final answer.\n"
    "Question:\n"
    "{question}\n"
    "Please respond with only your reasoning steps and the final answer. Do not include any extraneous text or explanations outside your solution."
)

inference_genselect_prompt = (
    "You are a highly skilled mathematician known for clear and rigorous reasoning.\n"
    "You are given a math question along with several candidate answers.\n"
    "Analyze each candidate solution, explain your reasoning, and then state which candidate is correct as your final answer.\n"
    "Question and candidate solutions:\n"
    "{question}\n"
    "Please respond with only your analysis and the final answer. The final answer must be one of the provided candidate solutions. Do not include any extraneous text."
)

judgement_prompt = "Reply with only 'yes' if both are correct, or 'no' if either is incorrect. Do not include any other text.\n"
if judge_only_by_answer:
    judgement_prompt = "Reply with only 'yes' if the answer is correct, or 'no' if it is incorrect. Do not include any other text.\n"

judgement_cot_prompt = (
    "You are a mathematics expert tasked with evaluating a user's solution.\n"
    "You will be given a question, the correct answer, and the user's solution (including their reasoning and final answer).\n"
    "Determine if BOTH the reasoning and the final answer in the user's solution are correct.\n"
    "{judgement_prompt_1}"
    "Question:\n"
    "{question}\n"
    "Correct answer:\n"
    "{correct_answer}\n"
    "User's solution:\n"
    "{solution}\n"
    "{judgement_prompt_2}"
)

judgement_genselect_prompt = (
    "You are a mathematics expert tasked with evaluating a user's solution.\n"
    "You will be given a question with candidate solutions, the correct answer, and the user's analysis and final answer.\n"
    "Determine if BOTH the reasoning and the final answer in the user's solution are correct.\n"
    "{judgement_prompt_1}"
    "Question and candidate solutions:\n"
    "{question}\n"
    "Correct answer:\n"
    "{correct_answer}\n"
    "User's solution:\n"
    "{solution}\n"
    "{judgement_prompt_2}"
)

#%%


def vllm_inference(llm, prompts, temperature=0.3, max_tokens=1024):
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,  # number of completions per prompt
    )

    # Batched inference
    results = []
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        # output.outputs[0].text contains the generated text
        results.append(output.outputs[0].text)
    return results

def vllm_judgement(llm, prompts, temperature=0.1, max_tokens=1024):
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,  # number of completions per prompt
    )

    # Batched inference
    results = []
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        # output.outputs[0].text contains the generated text
        results.append(output.outputs[0].text)
    return results

# inference the whole dataset
def inference(llm, inf_dataset, inference_batch_size, save_per_batch, inference_temperature, inference_max_tokens, inference_prompt, inference_dir, dataset_size):
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    inference_collection = []
    start_from_batch_index = int(dataset_size * args.start_from_percentage // inference_batch_size)
    end_at_batch_index = int(dataset_size * args.end_at_percentage // inference_batch_size)
    print(f"Inferencing from batch {start_from_batch_index} to batch {end_at_batch_index}")

    i = 1
    for data_batch in tqdm(inf_dataset.iter(batch_size=inference_batch_size), desc="Inferencing"):
        if i < start_from_batch_index:
            i += 1
            continue
        if i >= end_at_batch_index:
            break

        if i % 100 == 0:
            print(f"Inferencing {i} batches")
        
        if i % (save_per_batch) == 0 and os.path.exists(f"{inference_dir}/inference_{i}.json"):
            print(f"Inference {i} batches already exists, skipping")
            i += 1
            continue

        inference_prompts = [inference_prompt.format(question=question) for question in data_batch["problem"]]
        inference_results = vllm_inference(llm, inference_prompts, inference_temperature, inference_max_tokens)
        # add the inference results to data_batch, make a new column called 'inference'
        data_batch['inference'] = inference_results
        # data_batch is a dictionary, convert it to a list of dictionaries
        actual_batch_size = len(data_batch['problem'])  # Get actual size of current batch
        data_batch = [{k: v[j] for k, v in data_batch.items()} for j in range(actual_batch_size)]
        inference_collection.extend(data_batch)

        # save the temporary inference results
        if i % (save_per_batch) == 0 and inference_collection:  # Only save if we have data to save
            # save as pandas dataframe
            with open(f"{inference_dir}/inference_{i}.json", "w") as f:
                json.dump(inference_collection, f)
            inference_collection = []
            print(f"Saved inference {i} batches")
        i += 1

    # Save any remaining batches at the end
    if inference_collection:
        with open(f"{inference_dir}/inference_{i}.json", "w") as f:
            json.dump(inference_collection, f)
        print(f"Saved final inference batch {i}")

def judgement(jud_model, judgement_batch_size, judgement_temperature, judgement_max_tokens, judgement_prompt, inference_dir, judgement_dir):
    if not os.path.exists(judgement_dir):
        os.makedirs(judgement_dir)

    for inf_filename in tqdm(os.listdir(inference_dir), desc="Judging"):
        judgement_filename = inf_filename.replace("inference", "judgement")
        if os.path.exists(f"{judgement_dir}/{judgement_filename}"):
            print(f"Judgement {inf_filename} already exists, skipping")
            continue
        print(f"Judging {inf_filename}")

        judgement_collection = []
        with open(f"{inference_dir}/{inf_filename}", "r") as f:
            inf_results = json.load(f)
        num_rows = len(inf_results)
        for index in range(0, num_rows, judgement_batch_size):
            batch = inf_results[index:index+judgement_batch_size]
            question = [item['problem'] for item in batch]
            correct_answer = [item['generated_solution'] for item in batch]
            solution = [item['inference'] for item in batch]
            judgement_prompts = [judgement_prompt.format(judgement_prompt_1=judgement_prompt, judgement_prompt_2=judgement_prompt, question=q, correct_answer=ca, solution=s) for q, ca, s in zip(question, correct_answer, solution)]
            judgement_results = vllm_judgement(jud_model, judgement_prompts, judgement_temperature, judgement_max_tokens)
            for i, item in enumerate(batch):
                item['judgement'] = judgement_results[i]
            judgement_collection.extend(batch)
        with open(f"{judgement_dir}/{judgement_filename}", "w") as f:
            json.dump(judgement_collection, f)
            print(f"Saved judgement {inf_filename}")
        

#%%
# inferece the questions
# load inference model to use vllm
llm = LLM(
    model=inference_model,
    tensor_parallel_size=inference_tp,
    pipeline_parallel_size=inference_pp,
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    max_model_len=args.inference_max_model_len,  # Reduced from default 40960 to fit in GPU memory
    dtype="float16",  # Use half precision to save memory
)

cot_dataset = datasets.load_dataset("nvidia/OpenMathReasoning", split='cot', streaming=True)
# filter the inf_dataset by the problem_type column to be has_answer_extracted
cot_dataset = cot_dataset.filter(lambda x: x['problem_type'] == 'has_answer_extracted' and x['pass_rate_72b_tir'].isnumeric() and float(x['pass_rate_72b_tir']) < args.filter_by_pass_rate)
inference(llm, cot_dataset, inference_batch_size, save_per_batch, inference_temperature, inference_max_tokens, inference_cot_prompt, inference_dir + "/cot", cot_dataset_size)
# release the cot dataset
del cot_dataset

genselect_dataset = datasets.load_dataset("nvidia/OpenMathReasoning", split='genselect', streaming=True)
# filter the inf_dataset by the problem_type column to be has_answer_extracted
genselect_dataset = genselect_dataset.filter(lambda x: x['problem_type'] == 'has_answer_extracted' and x['pass_rate_72b_tir'].isnumeric() and float(x['pass_rate_72b_tir']) < args.filter_by_pass_rate)
inference(llm, genselect_dataset, inference_batch_size, save_per_batch, inference_temperature, inference_max_tokens, inference_genselect_prompt, inference_dir + "/genselect", genselect_dataset_size)
# release the genselect dataset
del genselect_dataset

# clear the inference model
del llm
torch.cuda.empty_cache()

#%%
llm = LLM(
    model=judgement_model,
    tensor_parallel_size=judgement_tp,
    pipeline_parallel_size=judgement_pp,
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    max_model_len=args.judgement_max_model_len,  # Reduced from default 40960 to fit in GPU memory
    dtype="float16",  # Use half precision to save memory
)

judgement(llm, judgement_batch_size, judgement_temperature, judgement_max_tokens, judgement_cot_prompt, inference_dir + "/cot", judgement_dir + "/cot")
judgement(llm, judgement_batch_size, judgement_temperature, judgement_max_tokens, judgement_genselect_prompt, inference_dir + "/genselect", judgement_dir + "/genselect")





