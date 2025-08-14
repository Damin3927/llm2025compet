# ref: https://github.com/matsuolab/llm_bridge_prod/blob/master/eval_hle/hle_benchmark/vllm_predictions.py

from argparse import ArgumentParser
from typing import Sequence, TypedDict, Dict, Any, Optional, cast, List
import os
import json
import asyncio
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
from inference import wait_until_vllm_up, AsyncVLLMClient


SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


class QuestionType(TypedDict):
    id: str
    question: str
    image: str  # "" if not multi-modal
    answer_type: str


def format_message(question: QuestionType):
    answer_type = question["answer_type"]
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == "exact_match" else SYSTEM_MC
    question_text = question["question"]

    text_content = dict(type="text", text=question_text)
    if question["image"]:  # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question["image"]))
        content = [text_content, image_content]
    else:
        content = [text_content]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return messages


async def attempt_question(
    client: AsyncVLLMClient, model: str, question: QuestionType, max_retries: int = 3
):
    messages = format_message(question)

    for attempt in range(max_retries):
        try:
            response = await client.generate_msg(model=model, messages=messages)
            content = response.choices[0].message.content
            if response.choices[0].finish_reason == "length":
                print(
                    f"Warning: Response for question {question['id']} was truncated due to length"
                )
                return None

            tokens = (
                json.loads(response.usage.model_dump_json()) if response.usage else {}
            )
            # Handle reasoning_content if available
            reasoning_content = getattr(
                response.choices[0].message, "reasoning_content", None
            )
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


def json_path_for_model(model_name: str) -> str:
    os.makedirs("predictions", exist_ok=True)
    return f"predictions/hle_{os.path.basename(model_name)}.json"


def load_predictions_json(path: str) -> Dict[str, Dict[str, Any]]:
    """Load predictions JSON dict keyed by id. Returns empty dict if file missing."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
        else:
            print(f"Warning: Predictions file is not a dict: {path}. Ignoring.")
            return {}
    except Exception as e:
        print(f"Warning: Failed to read predictions JSON '{path}': {e}")
        return {}


def save_predictions_json(path: str, predictions: Dict[str, Dict[str, Any]]):
    """Write the entire predictions dict to a JSON file (pretty-printed)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# Single-node worker pool removed; unified on multi-node implementation below.


async def run_worker_pool_multi(
    clients: Sequence[AsyncVLLMClient],
    model: str,
    questions: Sequence[QuestionType],
    num_workers_per_node: int,
    max_retries: int,
    predictions: Dict[str, Dict[str, Any]],
    save_path: str,
    flush_every: int,
) -> int:
    """Run a multi-node worker pool.

    Creates a shared queue and spawns `num_workers_per_node` workers per client.
    Each worker pulls from the same queue and sends requests via its assigned client.

    Returns the number of successful inferences.
    """
    total_workers = len(clients) * num_workers_per_node
    queue: asyncio.Queue[Optional[QuestionType]] = asyncio.Queue()
    for q in questions:
        await queue.put(q)
    # Add sentinels for clean shutdown (one per worker)
    for _ in range(total_workers):
        await queue.put(None)

    success_count = 0
    pending_writes = 0
    lock = asyncio.Lock()  # guard updates and saves

    with tqdm(total=len(questions), desc="Infer (multi-node)", unit="q") as pbar:
        async def worker_with_client(client: AsyncVLLMClient, _idx: int):
            nonlocal success_count
            nonlocal pending_writes
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                result = await attempt_question(client, model, item, max_retries)
                if result is not None:
                    unique_id, response, usage = result
                    async with lock:
                        predictions[unique_id] = {
                            "model": model,
                            "response": response,
                            "usage": usage,
                        }
                        success_count += 1
                        pending_writes += 1
                        if pending_writes >= flush_every:
                            save_predictions_json(save_path, predictions)
                            pending_writes = 0
                pbar.update(1)
                queue.task_done()

        tasks: List[asyncio.Task] = []
        for client_idx, client in enumerate(clients):
            for w in range(num_workers_per_node):
                tasks.append(asyncio.create_task(worker_with_client(client, client_idx * num_workers_per_node + w)))

        await queue.join()
        await asyncio.gather(*tasks)

    return success_count


def main():
    arg_parser = ArgumentParser(description="Run inference on HLE dataset")
    arg_parser.add_argument(
        "--model", type=str, required=True, help="Model name to use for inference"
    )
    arg_parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the vLLM API",
    )
    arg_parser.add_argument(
        "--base_urls",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Space- or comma-separated list of base URLs for multi-node vLLM servers. "
            "Example: --base_urls http://node1:8000 http://node2:8000"
        ),
    )
    arg_parser.add_argument(
        "--num_nodes",
        type=int,
        default=None,
        help=(
            "Number of nodes. When --base_urls is provided, this must match its length. "
            "If --base_urls is omitted, defaults to 1."
        ),
    )
    arg_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (per node when using --base_urls)",
    )
    arg_parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    arg_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API calls",
    )
    arg_parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name to push results",
    )
    arg_parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push results to HuggingFace Hub after inference",
    )
    arg_parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for authentication",
    )
    arg_parser.add_argument(
        "--flush_every",
        type=int,
        default=20,
        help="Flush predictions JSON to disk after this many new results (and at end)",
    )
    # Upload-only mode
    arg_parser.add_argument(
        "--upload_only",
        action="store_true",
    help="Only upload an existing predictions JSON file to the HuggingFace Hub; skip inference",
    )
    arg_parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Path to an existing predictions JSON to upload. Defaults to predictions/hle_<model>.json",
    )

    args = arg_parser.parse_args()

    # Parse base URLs for single- or multi-node
    parsed_base_urls: List[str] = []
    if args.base_urls:
        # Flatten and split on commas if needed
        for token in args.base_urls:
            for part in str(token).split(","):
                part = part.strip()
                if part:
                    parsed_base_urls.append(part)
        if args.num_nodes is not None and args.num_nodes != len(parsed_base_urls):
            raise ValueError(
                f"--num_nodes ({args.num_nodes}) must match number of --base_urls ({len(parsed_base_urls)})."
            )
    else:
        parsed_base_urls = [args.base_url]
        if args.num_nodes not in (None, 1):
            raise ValueError(
                "--num_nodes > 1 requires --base_urls to specify each node's base URL."
            )

    # Create vLLM clients per base URL
    clients: List[AsyncVLLMClient] = [AsyncVLLMClient(base_url=f"{u}/v1") for u in parsed_base_urls]

    # Resolve predictions JSON path
    output_json = args.input_json or json_path_for_model(args.model)

    # Upload-only mode
    if args.upload_only:
        if not args.dataset_name:
            raise ValueError("--dataset_name is required in --upload_only mode")
        if not os.path.exists(output_json):
            raise FileNotFoundError(f"Input JSON not found: {output_json}")
        print(
            f"Uploading predictions JSON '{output_json}' to HuggingFace Hub dataset '{args.dataset_name}'..."
        )
        try:
            # Read JSON dict and convert to HF dataset rows
            preds = load_predictions_json(output_json)
            rows = {
                "id": list(preds.keys()),
                "model": [preds[k].get("model") for k in preds.keys()],
                "response": [preds[k].get("response") for k in preds.keys()],
                "usage": [json.dumps(preds[k].get("usage", {})) for k in preds.keys()],
            }
            ds: HFDataset = HFDataset.from_dict(rows)
            ds.push_to_hub(args.dataset_name, token=args.hf_token)
            print("Upload completed.")
        except Exception as e:
            raise RuntimeError(f"Failed to upload dataset: {e}") from e
        return

    # Load dataset
    dataset = cast(HFDataset, load_dataset("cais/hle", split="test"))
    dataset = cast(HFDataset, dataset.filter(lambda item: item["image"] == ""))

    # Convert to list of questions using the original approach but fixed
    questions = []
    for i in range(len(dataset)):
        item = cast(Dict[str, Any], dataset[i])
        question_dict = {
            "id": item["id"],
            "question": item["question"],
            "image": item["image"],
            "answer_type": item["answer_type"],
        }
        questions.append(question_dict)

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[: args.max_samples]

    # Create predictions directory and load existing predictions JSON (for resume)
    os.makedirs("predictions", exist_ok=True)
    predictions: Dict[str, Dict[str, Any]] = load_predictions_json(output_json)
    existing_ids = set(predictions.keys())

    # Filter out questions that already have predictions based on JSONL IDs
    questions = [q for q in questions if q["id"] not in existing_ids]
    print(
        f"Processing {len(questions)} new questions (skipping {len(existing_ids)} existing predictions)"
    )

    if not questions:
        print("All questions already have predictions.")
        if args.push_to_hub and args.dataset_name:
            print("Pushing existing predictions JSON to HuggingFace Hub...")
            try:
                rows = {
                    "id": list(predictions.keys()),
                    "model": [predictions[k].get("model") for k in predictions.keys()],
                    "response": [predictions[k].get("response") for k in predictions.keys()],
                    "usage": [json.dumps(predictions[k].get("usage", {})) for k in predictions.keys()],
                }
                ds: HFDataset = HFDataset.from_dict(rows)
                ds.push_to_hub(args.dataset_name, token=args.hf_token)
                print("Upload completed.")
            except Exception as e:
                print(f"Failed to push to hub: {e}")
        return

    # Wait until the vLLM server(s) are up
    if len(parsed_base_urls) == 1:
        print("Waiting for vLLM Server to be up...")
        wait_until_vllm_up(parsed_base_urls[0])
    else:
        print(f"Waiting for {len(parsed_base_urls)} vLLM Servers to be up...")
        for u in parsed_base_urls:
            print(f" - {u}")
            wait_until_vllm_up(u)

    # Run multi-node worker pool (works for single-node as well)
    print(
        f"Running inference across {len(clients)} node(s) with {args.num_workers} workers per node"
    )
    total_success = asyncio.run(
        run_worker_pool_multi(
            clients=clients,
            model=args.model,
            questions=questions,
            num_workers_per_node=args.num_workers,
            max_retries=args.max_retries,
            predictions=predictions,
            save_path=output_json,
            flush_every=args.flush_every,
        )
    )

    print(f"Inference completed! Total successful: {total_success}/{len(questions)}")

    # Final save to ensure all results are persisted
    try:
        save_predictions_json(output_json, predictions)
        print(f"Predictions saved to {output_json}")
    except Exception as e:
        print(f"Warning: failed to save predictions JSON: {e}")

    # Final upload to the hub if requested
    if args.push_to_hub and args.dataset_name:
        print("Pushing results JSON to HuggingFace Hub...")
        try:
            rows = {
                "id": list(predictions.keys()),
                "model": [predictions[k].get("model") for k in predictions.keys()],
                "response": [predictions[k].get("response") for k in predictions.keys()],
                "usage": [json.dumps(predictions[k].get("usage", {})) for k in predictions.keys()],
            }
            ds: HFDataset = HFDataset.from_dict(rows)
            ds.push_to_hub(args.dataset_name, token=args.hf_token)
            print("Upload completed.")
        except Exception as e:
            print(f"Failed to push to hub: {e}")


if __name__ == "__main__":
    main()
