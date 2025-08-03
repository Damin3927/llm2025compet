# ref: https://github.com/matsuolab/llm_bridge_prod/blob/master/eval_hle/hle_benchmark/vllm_predictions.py

from argparse import ArgumentParser
from typing import Sequence, TypedDict, Dict, Any, Optional
import os
import json
import asyncio
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm_asyncio
from inference.utils import VLLMClient


SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


class QuestionType(TypedDict):
    id: str
    question: str
    image: str  # "" if not multi-modal
    answer_type: str



def format_message(question: QuestionType):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(client: VLLMClient, model: str, question: QuestionType, max_retries: int = 3):
    messages = format_message(question)
    
    for attempt in range(max_retries):
        try:
            response = client.generate_msg(model=model, messages=messages)
            content = response.choices[0].message.content
            if response.choices[0].finish_reason == "length":
                print(f"Warning: Response for question {question['id']} was truncated due to length")
                return None
            
            tokens = json.loads(response.usage.model_dump_json()) if response.usage else {}
            # Handle reasoning_content if available
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
            if reasoning_content:
                tokens["reasoning_content"] = reasoning_content
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1} for question {question['id']}: {e}")
            if attempt == max_retries - 1:  # Last attempt
                return None
            await asyncio.sleep(1)  # Brief delay before retry
            continue
        
        if content is None:  # failed
            if attempt == max_retries - 1:
                return None
            continue
        
        return question["id"], content, tokens
    
    return None


def save_to_hf_dataset(predictions: Dict[str, Any], model_name: str, dataset_name: Optional[str] = None, 
                      push_to_hub: bool = False, hf_token: Optional[str] = None) -> Dataset:
    """
    Save predictions to a HuggingFace Dataset.
    
    Args:
        predictions: Dictionary of predictions
        model_name: Name of the model used
        dataset_name: Name for the dataset (if pushing to hub)
        push_to_hub: Whether to push to HuggingFace Hub
        hf_token: HuggingFace token for authentication
    
    Returns:
        The created Dataset object
    """
    # Convert predictions dict to dataset format
    data = {
        'id': [],
        'model': [],
        'response': [],
        'usage': []
    }
    
    for question_id, pred in predictions.items():
        data['id'].append(question_id)
        data['model'].append(pred['model'])
        data['response'].append(pred['response'])
        data['usage'].append(json.dumps(pred['usage']))  # Store usage as JSON string
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    
    # Save locally first
    local_path = f"predictions/hle_{os.path.basename(model_name)}_dataset"
    os.makedirs("predictions", exist_ok=True)
    dataset.save_to_disk(local_path)
    print(f"Dataset saved locally to: {local_path}")
    
    # Optionally push to hub
    if push_to_hub and dataset_name:
        try:
            dataset.push_to_hub(dataset_name, token=hf_token)
            print(f"Dataset pushed to HuggingFace Hub: {dataset_name}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")
            print("Dataset is still saved locally.")
    
    return dataset


def load_existing_predictions(model_name: str) -> Dict[str, Any]:
    """
    Load existing predictions from both JSON and HF Dataset formats.
    """
    predictions = {}
    
    # Try to load from JSON first (backward compatibility)
    json_path = f"predictions/hle_{os.path.basename(model_name)}.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            predictions.update(json.load(f))
        print(f"Loaded {len(predictions)} existing predictions from JSON")
    
    # Try to load from HF Dataset
    dataset_path = f"predictions/hle_{os.path.basename(model_name)}_dataset"
    if os.path.exists(dataset_path):
        try:
            existing_dataset = Dataset.load_from_disk(dataset_path)
            for i in range(len(existing_dataset)):
                row = existing_dataset[i]
                predictions[row['id']] = {
                    'model': row['model'],
                    'response': row['response'],
                    'usage': json.loads(row['usage'])
                }
            print(f"Loaded {len(existing_dataset)} existing predictions from HF Dataset")
        except Exception as e:
            print(f"Failed to load existing dataset: {e}")
    
    return predictions


async def attempt_all(client: VLLMClient, model: str, num_workers: int, questions: Sequence[QuestionType]):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(client, model, question)
            return content
            
    semaphore = asyncio.Semaphore(num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results

def main():
    arg_parser = ArgumentParser(description="Run inference on HLE dataset")
    arg_parser.add_argument("--model", type=str, required=True, help="Model name to use for inference")
    arg_parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL for the vLLM API")
    arg_parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    arg_parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    arg_parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for failed API calls")
    arg_parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name to push results")
    arg_parser.add_argument("--push_to_hub", action="store_true", help="Push results to HuggingFace Hub")
    arg_parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for authentication")
    arg_parser.add_argument("--batch_size", type=int, default=50, help="Batch size for saving intermediate results")

    args = arg_parser.parse_args()

    client = VLLMClient(base_url=args.base_url)

    # Load dataset
    dataset = load_dataset("cais/hle", split="test")
    dataset = dataset.filter(lambda item: item['image'] == "")
    
    # Convert to list of questions using the original approach but fixed
    questions = []
    for item in dataset:
        question_dict = {
            'id': item['id'],
            'question': item['question'], 
            'image': item['image'],
            'answer_type': item['answer_type']
        }
        questions.append(question_dict)
    
    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[:args.max_samples]
    
    # Create predictions directory
    os.makedirs("predictions", exist_ok=True)

    # Load existing predictions (with fail-safe recovery)
    predictions = load_existing_predictions(args.model)
    
    # Filter out questions that already have predictions
    questions = [q for q in questions if q["id"] not in predictions]
    print(f"Processing {len(questions)} new questions (skipping {len(predictions)} existing predictions)")

    if not questions:
        print("All questions already have predictions. Creating final dataset...")
        save_to_hf_dataset(predictions, args.model, args.dataset_name, args.push_to_hub, args.hf_token)
        return

    # Process questions in batches for fail-safe intermediate saving
    batch_size = args.batch_size
    total_processed = 0
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
        
        # Create a modified attempt_all function with retry logic
        async def attempt_all_with_retries(client: VLLMClient, model: str, num_workers: int, questions: Sequence[QuestionType], max_retries: int):
            async def bound_func(question):
                async with semaphore:
                    content = await attempt_question(client, model, question, max_retries)
                    return content
                    
            semaphore = asyncio.Semaphore(num_workers)
            async with semaphore:
                tasks = [bound_func(q) for q in questions]
                results = await tqdm_asyncio.gather(*tasks)
            return results
        
        # Run inference on current batch
        results = asyncio.run(attempt_all_with_retries(
            client, 
            model=args.model, 
            num_workers=args.num_workers, 
            questions=batch,
            max_retries=args.max_retries
        ))

        # Process results and update predictions
        batch_success = 0
        for result in results:
            if result is None:  # API call failed
                continue
            unique_id, response, usage = result
            predictions[unique_id] = {
                "model": args.model,
                "response": response,
                "usage": usage
            }
            batch_success += 1
        
        total_processed += batch_success
        print(f"Batch completed: {batch_success}/{len(batch)} successful inferences")
        
        # Save intermediate results (fail-safe)
        try:
            save_to_hf_dataset(predictions, args.model, args.dataset_name, False, args.hf_token)
            print(f"Intermediate results saved. Total: {len(predictions)} predictions")
        except Exception as e:
            print(f"Warning: Failed to save intermediate results: {e}")
            # Fallback to JSON
            try:
                output_filepath = f"predictions/hle_{os.path.basename(args.model)}.json"
                with open(output_filepath, "w") as f:
                    json.dump(predictions, f, indent=4)
                print(f"Fallback: Saved to JSON at {output_filepath}")
            except Exception as json_e:
                print(f"Critical: Failed to save even to JSON: {json_e}")

    print(f"Inference completed! Total successful: {total_processed}/{len(questions)}")
    
    # Final save with option to push to hub
    try:
        dataset_obj = save_to_hf_dataset(predictions, args.model, args.dataset_name, args.push_to_hub, args.hf_token)
        print(f"Final dataset created with {len(dataset_obj)} records")
    except Exception as e:
        print(f"Error creating final dataset: {e}")
        # Ensure we at least have a JSON backup
        output_filepath = f"predictions/hle_{os.path.basename(args.model)}.json"
        with open(output_filepath, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"Results saved to JSON backup: {output_filepath}")


if __name__ == "__main__":
    main()
