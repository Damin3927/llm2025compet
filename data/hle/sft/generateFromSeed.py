import os
import torch
from vllm import LLM, SamplingParams
import json
from typing import List, Dict, Tuple

def setup_llm(model_name: str = "Qwen/Qwen3-8B", max_model_len: int = 1024) -> LLM:
    """
    Initialize the LLM using vllm.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        LLM: Initialized LLM instance
    """
    # Set environment variables to fix networking issues
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,  # Required for some models like Qwen
        dtype="bfloat16",  # Use bfloat16 for better memory efficiency
        max_model_len=max_model_len,
        gpu_memory_utilization=0.95
    )
    return llm

def create_prompt(seed_question: str, seed_answer: str) -> str:
    """
    Create a prompt for generating new math QA pairs based on a seed pair.
    
    Args:
        seed_question: The seed math question
        seed_answer: The corresponding answer
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""
    Based on the following math question and answer pair, generate a new, similar but different math question and its detailed solution.
    Original Question: {seed_question}
    Original Answer: {seed_answer}
    Generate a new question that tests similar concepts but with different numbers, context, or slight variations in the problem structure.
    Please provide the new question and answer in the following format. Do not include any other text.
    ```json
    {{
        "question": "New Question",
        "answer": "New Answer"
    }}
    ```
    """
    return prompt

def parse_generation(generated_text: str) -> Tuple[str, str]:
    """
    Parse the generated text to extract the question and answer.
    
    Args:
        generated_text: Raw generated text from the model
        
    Returns:
        Tuple[str, str]: Extracted (question, answer) pair
    """
    # Split on common answer indicators
    json_str = generated_text.replace("```json", "").replace("```", "").strip()
    import pdb; pdb.set_trace()
    json_data = json.loads(json_str)
    question = json_data["question"]
    answer = json_data["answer"]
    return question, answer

def generate_qa_pairs(
    llm: LLM,
    seed_question: str,
    seed_answer: str,
    num_pairs: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 512
) -> List[Dict[str, str]]:
    """
    Generate new question-answer pairs based on a seed pair.
    
    Args:
        llm: Initialized LLM instance
        seed_question: The seed math question
        seed_answer: The corresponding answer
        num_pairs: Number of new pairs to generate
        temperature: Sampling temperature (higher = more creative)
        max_tokens: Maximum tokens to generate
        
    Returns:
        List[Dict[str, str]]: List of generated QA pairs
    """
    prompt = create_prompt(seed_question, seed_answer)
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_pairs
    )
    
    # Generate responses
    outputs = llm.generate([prompt], sampling_params)
    
    # Process and format the outputs
    generated_pairs = []
    for output in outputs[0].outputs:
        question, answer = parse_generation(output.text)
        generated_pairs.append({
            "question": question,
            "answer": answer
        })
    
    return generated_pairs

def save_qa_pairs(pairs: List[Dict[str, str]], output_file: str):
    """
    Save generated QA pairs to a JSON file.
    
    Args:
        pairs: List of generated QA pairs
        output_file: Path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)

if __name__ == "__main__":
    res_dir = "results/generateFromSeed"
    os.makedirs(res_dir, exist_ok=True)

    # Example usage
    seed_question = "If a triangle has angles measuring 30°, 60°, and 90°, what is the ratio of its shortest to its longest side?"
    seed_answer = "The ratio is 1:2. In a 30-60-90 triangle, if the shortest side (opposite to 30°) is x, then the hypotenuse (opposite to 90°) is 2x, and the remaining side is x√3."
    
    # Initialize the LLM
    llm = setup_llm("Qwen/Qwen3-8B", max_model_len=2048)
    
    # Generate new QA pairs
    generated_pairs = generate_qa_pairs(
        llm=llm,
        seed_question=seed_question,
        seed_answer=seed_answer,
        num_pairs=3
    )
    
    # Save the generated pairs
    save_qa_pairs(generated_pairs, os.path.join(res_dir, "generated_math_qa_pairs.json"))
