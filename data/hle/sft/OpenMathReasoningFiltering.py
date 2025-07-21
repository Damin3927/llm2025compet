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
import datasets
import torch
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"

run_index = 1 # we want to run the script multiple times, so we need to save the results with a run_index

inference_model = "Qwen/Qwen3-8B" # TODO: change to a model you want to use to answer the questions
inference_temperature = 0.3 # lower temperature means more deterministic
inference_max_tokens = 4096 # max tokens to generate
inference_batch_size = 4
inference_tp, inference_pp, inference_dp = 1, 1, 1 # tensor parallel, pipeline parallel, data parallel
save_per_batch = 1 # save per `save_per_batch` batches

judgement_model = "Qwen/Qwen3-8B" # TODO: change to a model you want to use to judge the answers
judgement_temperature = 0. # lower temperature means more deterministic
judgement_max_tokens = 50
judgement_batch_size = 4
judgement_tp, judgement_pp, judgement_dp = 1, 1, 1 # tensor parallel, pipeline parallel, data parallel
judge_only_by_answer = True # if True, only judge the answers, if False, judge the reasoning and the answer

# hard coding the dataset size
cot_dataset_size = 3.3e9
genselect_dataset_size = 5.66e5
# start from percentage
start_from_percentage = 0 # start from the percentage (0.5 = 50%) of the dataset, so we can run the script separately
end_at_percentage = 1.0 # end at the percentage (1.0 = 100%) of the dataset, so we can run the script separately


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
    import pdb; pdb.set_trace()
    for output in outputs:
        # output.outputs[0].text contains the generated text
        results.append(output.outputs[0].text)
    return results

# inference the whole dataset
def inference(inf_dataset, inference_batch_size, save_per_batch, inference_temperature, inference_max_tokens, inference_prompt, inference_dir, dataset_size):
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    inference_collection = []
    start_from_batch_index = int(dataset_size * start_from_percentage // inference_batch_size)
    end_at_batch_index = int(dataset_size * end_at_percentage // inference_batch_size)

    # filter the inf_dataset by the problem_type column to be has_answer_extracted
    inf_dataset = inf_dataset.filter(lambda x: x['problem_type'] == 'has_answer_extracted')
    i = 1
    for data_batch in tqdm(inf_dataset.iter(batch_size=inference_batch_size), desc="Inferencing"):
        if i < start_from_batch_index:
            i += 1
            continue
        if i >= end_at_batch_index:
            break
        
        if i % (save_per_batch) == 0 and os.path.exists(f"{inference_dir}/inference_{i}.json"):
            i += 1
            continue

        inference_prompts = [inference_prompt.format(question=question) for question in data_batch["problem"]]
        inference_results = vllm_inference(llm, inference_prompts, inference_temperature, inference_max_tokens)
        # add the inference results to data_batch, make a new column called 'inference'
        data_batch['inference'] = inference_results
        # data_batch is a dictionary, convert it to a list of dictionaries
        data_batch = [{k: v[j] for k, v in data_batch.items()} for j in range(inference_batch_size)]
        inference_collection.extend(data_batch)

        # save the temporary inference results
        if i % (save_per_batch) == 0:
            # save as pandas dataframe
            with open(f"{inference_dir}/inference_{i}.json", "w") as f:
                json.dump(inference_collection, f)
            inference_collection.clear()
        i += 1

def judgement(jud_model, judgement_batch_size, judgement_temperature, judgement_max_tokens, judgement_prompt, inference_dir, judgement_dir):
    if not os.path.exists(judgement_dir):
        os.makedirs(judgement_dir)

    for inf_filename in tqdm(os.listdir(inference_dir)):
        judgement_filename = inf_filename.replace("inference", "judgement")
        if os.path.exists(f"{judgement_dir}/{judgement_filename}"):
            continue

        judgement_collection = []
        with open(f"{inference_dir}/{inf_filename}", "r") as f:
            inf_results = json.load(f)
        num_rows = len(inf_results)
        for index in tqdm(range(0, num_rows, judgement_batch_size), desc="Judging"):
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
        

#%%
# inferece the questions
# load inference model to use vllm
llm = LLM(model=inference_model, tensor_parallel_size=inference_tp, pipeline_parallel_size=inference_pp, gpu_memory_utilization=0.95)

cot_dataset = datasets.load_dataset("nvidia/OpenMathReasoning", split='cot', streaming=True)
inference(cot_dataset, inference_batch_size, save_per_batch, inference_temperature, inference_max_tokens, inference_cot_prompt, inference_dir + "/cot", cot_dataset_size)
# release the cot dataset
del cot_dataset

genselect_dataset = datasets.load_dataset("nvidia/OpenMathReasoning", split='genselect', streaming=True)
inference(genselect_dataset, inference_batch_size, save_per_batch, inference_temperature, inference_max_tokens, inference_genselect_prompt, inference_dir + "/genselect", genselect_dataset_size)
# release the genselect dataset
del genselect_dataset

# clear the inference model
del llm
torch.cuda.empty_cache()

#%%
llm = LLM(model=judgement_model, tensor_parallel_size=judgement_tp, pipeline_parallel_size=judgement_pp, gpu_memory_utilization=0.95)

judgement(llm, judgement_batch_size, judgement_temperature, judgement_max_tokens, judgement_cot_prompt, inference_dir + "/cot", judgement_dir + "/cot")
judgement(llm, judgement_batch_size, judgement_temperature, judgement_max_tokens, judgement_genselect_prompt, inference_dir + "/genselect", judgement_dir + "/genselect")





