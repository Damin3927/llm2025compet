#!/usr/bin/env python3
"""
Hugging Face Dataset Processing Pipeline

Reads a JSON configuration file containing Hugging Face datasets,
evaluates CoT quality, regenerates low-quality items, and saves
results in an organized folder structure.

Usage:
    python process_hf_datasets.py --config datasets_config.json --output-dir ./processed_datasets
"""
import sys
sys.path.append("../")
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

# Import dataset handling
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Error: datasets library not installed. Install with: pip install datasets")
    exit(1)

# Import our modules
from evaluate_cot_async import CoTEvaluationProcessor, CoTRegenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HFDatasetProcessor:
    """Process multiple Hugging Face datasets with evaluation and regeneration."""
    
    def __init__(
        self,
        output_dir: str = "./processed_datasets",
        evaluator_models: Optional[List[str]] = None,
        regenerator_models: Optional[List[str]] = None,
        grade_threshold: str = "C",
        max_items_per_dataset: Optional[int] = None,
        use_async_regeneration: bool = True,
        skip_existing: bool = False
    ):
        """
        Initialize the dataset processor.
        
        Args:
            output_dir: Base directory for saving processed datasets
            evaluator_models: Models to use for evaluation
            regenerator_models: Models to use for regeneration
            grade_threshold: Regenerate items with this grade or lower
            max_items_per_dataset: Maximum items to process per dataset (for testing)
            use_async_regeneration: Whether to use async multi-model regeneration
            skip_existing: Skip datasets that have already been processed
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = CoTEvaluationProcessor()
        self.evaluator_models = evaluator_models or ["deepseek/deepseek-r1-0528:free"]
        
        # Initialize regenerator
        self.regenerator = CoTRegenerator(
            models=regenerator_models or ["deepseek/deepseek-r1-0528:free"]
        )
        
        self.grade_threshold = grade_threshold
        self.max_items = max_items_per_dataset
        self.use_async_regeneration = use_async_regeneration
        self.skip_existing = skip_existing
        
        # Statistics tracking
        self.global_stats = {
            "total_datasets": 0,
            "total_items_processed": 0,
            "total_items_evaluated": 0,
            "total_items_regenerated": 0,
            "total_improvements": 0,
            "total_skipped": 0,
            "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
            "errors": []
        }
    
    def load_config(self, config_path: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load dataset configuration from JSON file.
        
        Expected format:
        {
            "datasets": [
                {
                    "dataset_name": "gsm8k",
                    "configs": ["main", "socratic"],
                    "split": "train",
                    "question_field": "question",
                    "answer_field": "answer",
                    "output_field": "output"
                },
                ...
            ],
            "processing_options": {
                "max_items_per_dataset": 1000,
                "grade_threshold": "C",
                ...
            }
        }
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Handle both formats: list or nested with "datasets" key
        if isinstance(config, list):
            datasets = config
            processing_options = {}
        else:
            datasets = config.get("datasets", [])
            processing_options = config.get("processing_options", {})
        
        logger.info(f"Loaded configuration with {len(datasets)} datasets")
        return datasets, processing_options
    
    def prepare_item_for_evaluation(
        self,
        item: Dict[str, Any],
        question_field: str,
        answer_field: str,
        output_field: str,
        item_id: Any
    ) -> Dict[str, Any]:
        """
        Prepare a dataset item for evaluation.
        
        Args:
            item: Raw item from dataset
            question_field: Field name for question
            answer_field: Field name for answer
            output_field: Field name for CoT output
            item_id: Unique identifier for the item
        
        Returns:
            Formatted item ready for evaluation
        """
        return {
            "id": str(item_id),
            "question": str(item.get(question_field, "")),
            "answer": str(item.get(answer_field, "")),
            "output": str(item.get(output_field, "")),
            "metadata": {
                "original_data": item,
                "cot_history": []
            }
        }
    
    async def process_single_item(
        self,
        item: Dict[str, Any],
        regenerate: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single item: evaluate and optionally regenerate.
        
        Args:
            item: Item to process
            regenerate: Whether to regenerate low-quality items
        
        Returns:
            Processed item with evaluation and possibly regeneration
        """
        try:
            # Step 1: Evaluate
            eval_result = await self.evaluator.evaluate_single_item(
                item,
                model_names=self.evaluator_models,
                concurrency=2
            )
            
            # Update item with evaluation
            if "metadata" not in item:
                item["metadata"] = {}
            if "cot_history" not in item["metadata"]:
                item["metadata"]["cot_history"] = []
            
            item["metadata"]["cot_history"].append({
                "timestamp": datetime.now().isoformat(),
                "output": item["output"],
                "evaluation": eval_result["evaluation"]
            })
            
            # Update grade statistics
            grade = eval_result.get("grade", "D")
            self.global_stats["grade_distribution"][grade] += 1
            
            # Step 2: Regenerate if needed
            if regenerate and self._should_regenerate(grade):
                evaluation_details = {
                    "grade": grade,
                    "strengths": eval_result["evaluation"].get("strengths", []),
                    "weaknesses": eval_result["evaluation"].get("weaknesses", []),
                    "improvement_suggestions": eval_result["evaluation"].get("improvement_suggestions", []),
                    "learning_value_scores": eval_result["evaluation"].get("learning_value_scores", {})
                }
                
                if self.use_async_regeneration and len(self.regenerator.model_configs) > 1:
                    # Async multi-model regeneration
                    regen_result = await self.regenerator.regenerate_multi_async(
                        question=item["question"],
                        answer=item["answer"],
                        previous_cot=item["output"],
                        evaluation_details=evaluation_details
                    )
                    
                    if regen_result:
                        new_output = f"<think>{regen_result['best_cot']}</think>{item['answer']}"
                        item["output"] = new_output
                        
                        # Add regeneration to history
                        item["metadata"]["cot_history"].append({
                            "timestamp": datetime.now().isoformat(),
                            "output": new_output,
                            "evaluation": None,
                            "regeneration_metadata": {
                                "model": regen_result["best_model"],
                                "predicted_grade": regen_result["best_grade"],
                                "predicted_score": regen_result["best_score"]
                            }
                        })
                        
                        self.global_stats["total_items_regenerated"] += 1
                        
                        # Check if improved
                        if regen_result["best_grade"] > grade:
                            self.global_stats["total_improvements"] += 1
                else:
                    # Single model regeneration
                    new_cot = self.regenerator.regenerate_single(
                        question=item["question"],
                        answer=item["answer"],
                        previous_cot=item["output"],
                        evaluation_details=evaluation_details
                    )
                    
                    if new_cot:
                        new_output = f"<think>{new_cot}</think>{item['answer']}"
                        item["output"] = new_output
                        
                        # Add regeneration to history
                        item["metadata"]["cot_history"].append({
                            "timestamp": datetime.now().isoformat(),
                            "output": new_output,
                            "evaluation": None,
                            "regeneration_metadata": {
                                "model": self.regenerator.model_configs[0].model_name
                            }
                        })
                        
                        self.global_stats["total_items_regenerated"] += 1
            
            return item
            
        except Exception as e:
            logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
            self.global_stats["errors"].append({
                "item_id": item.get("id", "unknown"),
                "error": str(e)
            })
            return item
    
    def _should_regenerate(self, grade: str) -> bool:
        """Check if an item should be regenerated based on grade."""
        grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
        return grade_order.get(grade, 0) <= grade_order.get(self.grade_threshold, 0)
    
    async def process_dataset_config(
        self,
        dataset_name: str,
        config: str,
        split: str,
        question_field: str,
        answer_field: str,
        output_field: str
    ) -> Dict[str, Any]:
        """
        Process a single dataset configuration.
        
        Returns:
            Statistics for this dataset configuration
        """
        # Check if dataset already exists and should be skipped
        if self.skip_existing and self._check_dataset_exists(dataset_name, config):
            logger.info(f"Skipping {dataset_name}/{config} - already processed")
            return {
                "dataset_name": dataset_name,
                "config": config,
                "split": split,
                "items_processed": 0,
                "items_evaluated": 0,
                "items_regenerated": 0,
                "skipped": True,
                "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
                "errors": []
            }
        
        logger.info(f"Processing {dataset_name} (config: {config}, split: {split})")
        
        stats = {
            "dataset_name": dataset_name,
            "config": config,
            "split": split,
            "items_processed": 0,
            "items_evaluated": 0,
            "items_regenerated": 0,
            "skipped": False,
            "grade_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
            "errors": []
        }
        
        try:
            # Load dataset
            logger.info(f"Loading dataset {dataset_name}/{config}...")
            dataset = load_dataset(dataset_name, config, split=split)
            
            # Limit items if specified
            if self.max_items:
                dataset = dataset.select(range(min(self.max_items, len(dataset))))
            
            logger.info(f"Loaded {len(dataset)} items")
            
            # Prepare items for processing
            processed_items = []
            
            # Process items with progress bar
            for idx, raw_item in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}/{config}")):
                # Prepare item
                item = self.prepare_item_for_evaluation(
                    raw_item,
                    question_field,
                    answer_field,
                    output_field,
                    item_id=idx
                )
                
                # Process (evaluate and regenerate)
                processed_item = await self.process_single_item(item, regenerate=True)
                processed_items.append(processed_item)
                
                # Update statistics
                stats["items_processed"] += 1
                if processed_item.get("metadata", {}).get("cot_history"):
                    stats["items_evaluated"] += 1
                    
                    # Check if regenerated
                    if len(processed_item["metadata"]["cot_history"]) > 1:
                        stats["items_regenerated"] += 1
                    
                    # Update grade distribution
                    first_eval = processed_item["metadata"]["cot_history"][0]
                    if first_eval.get("evaluation"):
                        grade = first_eval["evaluation"].get("grade", "D")
                        stats["grade_distribution"][grade] += 1
            
            # Save processed dataset
            output_path = self._get_output_path(dataset_name, config)
            self._save_processed_dataset(processed_items, output_path)
            
            logger.info(f"Saved {len(processed_items)} items to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}/{config}: {e}")
            stats["errors"].append(str(e))
        
        return stats
    
    def _get_output_path(self, dataset_name: str, config: str) -> Path:
        """Get output path for a dataset configuration."""
        # Create folder structure: output_dir/dataset_name/config/
        dataset_dir = self.output_dir / dataset_name / config
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{timestamp}.json"
        
        return dataset_dir / filename
    
    def _check_dataset_exists(self, dataset_name: str, config: str) -> bool:
        """Check if a dataset has already been processed."""
        dataset_dir = self.output_dir / dataset_name / config
        
        if not dataset_dir.exists():
            return False
        
        # Check for any processed JSON or JSONL files
        json_files = list(dataset_dir.glob("processed_*.json"))
        jsonl_files = list(dataset_dir.glob("processed_*.jsonl"))
        
        return len(json_files) > 0 or len(jsonl_files) > 0
    
    def _save_processed_dataset(self, items: List[Dict[str, Any]], output_path: Path):
        """Save processed dataset to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        # Also save as JSONL for compatibility
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in items:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    async def process_all_datasets(self, config_path: str) -> Dict[str, Any]:
        """
        Process all datasets from configuration file.
        
        Args:
            config_path: Path to JSON configuration file
        
        Returns:
            Global statistics
        """
        # Load configuration
        datasets_config, processing_options = self.load_config(config_path)
        
        # Override settings from config file if provided
        if processing_options:
            if "max_items_per_dataset" in processing_options:
                self.max_items = processing_options["max_items_per_dataset"]
            if "grade_threshold" in processing_options:
                self.grade_threshold = processing_options["grade_threshold"]
            if "use_async_regeneration" in processing_options:
                self.use_async_regeneration = processing_options["use_async_regeneration"]
        
        logger.info(f"Starting processing of {len(datasets_config)} datasets")
        
        # Process each dataset
        dataset_stats = []
        
        for dataset_config in datasets_config:
            dataset_name = dataset_config["dataset_name"]
            configs = dataset_config.get("configs", ["main"])
            split = dataset_config.get("split", "train")
            question_field = dataset_config.get("question_field", "question")
            answer_field = dataset_config.get("answer_field", "answer")
            output_field = dataset_config.get("output_field", "output")
            
            # Process each configuration
            for config in configs:
                stats = await self.process_dataset_config(
                    dataset_name=dataset_name,
                    config=config,
                    split=split,
                    question_field=question_field,
                    answer_field=answer_field,
                    output_field=output_field
                )
                
                dataset_stats.append(stats)
                
                # Update global statistics
                if stats.get("skipped", False):
                    self.global_stats["total_skipped"] += 1
                    logger.info(f"Dataset {dataset_name}/{config} was skipped")
                else:
                    self.global_stats["total_datasets"] += 1
                    self.global_stats["total_items_processed"] += stats["items_processed"]
                    self.global_stats["total_items_evaluated"] += stats["items_evaluated"]
                    self.global_stats["total_items_regenerated"] += stats["items_regenerated"]
        
        # Save statistics
        self._save_statistics(dataset_stats)
        
        return self.global_stats
    
    def _save_statistics(self, dataset_stats: List[Dict[str, Any]]):
        """Save processing statistics."""
        stats_path = self.output_dir / "processing_statistics.json"
        
        statistics = {
            "timestamp": datetime.now().isoformat(),
            "global_stats": self.global_stats,
            "dataset_stats": dataset_stats
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved statistics to {stats_path}")
        
        # Also create a summary report
        self._create_summary_report(dataset_stats)
    
    def _create_summary_report(self, dataset_stats: List[Dict[str, Any]]):
        """Create a human-readable summary report."""
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATASET PROCESSING SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Global statistics
            f.write("GLOBAL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Datasets Processed: {self.global_stats['total_datasets']}\n")
            if self.global_stats.get('total_skipped', 0) > 0:
                f.write(f"Total Datasets Skipped: {self.global_stats['total_skipped']}\n")
            f.write(f"Total Items Processed: {self.global_stats['total_items_processed']}\n")
            f.write(f"Total Items Evaluated: {self.global_stats['total_items_evaluated']}\n")
            f.write(f"Total Items Regenerated: {self.global_stats['total_items_regenerated']}\n")
            f.write(f"Total Improvements: {self.global_stats['total_improvements']}\n")
            f.write(f"Error Count: {len(self.global_stats['errors'])}\n\n")
            
            # Grade distribution
            f.write("GRADE DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            total_graded = sum(self.global_stats['grade_distribution'].values())
            for grade in ['A', 'B', 'C', 'D']:
                count = self.global_stats['grade_distribution'][grade]
                percentage = (count / total_graded * 100) if total_graded > 0 else 0
                f.write(f"Grade {grade}: {count:5d} ({percentage:5.1f}%)\n")
            f.write("\n")
            
            # Per-dataset statistics
            f.write("PER-DATASET STATISTICS\n")
            f.write("-"*40 + "\n")
            for stats in dataset_stats:
                f.write(f"\nDataset: {stats['dataset_name']}/{stats['config']}\n")
                if stats.get('skipped', False):
                    f.write("  Status: SKIPPED (already processed)\n")
                    continue
                f.write(f"  Split: {stats['split']}\n")
                f.write(f"  Items Processed: {stats['items_processed']}\n")
                f.write(f"  Items Evaluated: {stats['items_evaluated']}\n")
                f.write(f"  Items Regenerated: {stats['items_regenerated']}\n")
                
                # Grade distribution for this dataset
                total = sum(stats['grade_distribution'].values())
                if total > 0:
                    f.write("  Grade Distribution:\n")
                    for grade in ['A', 'B', 'C', 'D']:
                        count = stats['grade_distribution'][grade]
                        percentage = (count / total * 100) if total > 0 else 0
                        f.write(f"    {grade}: {count:3d} ({percentage:5.1f}%)\n")
                
                if stats['errors']:
                    f.write(f"  Errors: {len(stats['errors'])}\n")
            
            # Errors summary
            if self.global_stats['errors']:
                f.write("\n" + "="*70 + "\n")
                f.write("ERROR SUMMARY\n")
                f.write("-"*40 + "\n")
                for error in self.global_stats['errors'][:10]:  # Show first 10 errors
                    f.write(f"Item {error['item_id']}: {error['error']}\n")
                if len(self.global_stats['errors']) > 10:
                    f.write(f"... and {len(self.global_stats['errors']) - 10} more errors\n")
        
        logger.info(f"Saved summary report to {report_path}")


def create_example_config(output_path: str = "example_datasets_config.json"):
    """Create an example configuration file."""
    example_config = {
        "datasets": [
            {
                "dataset_name": "gsm8k",
                "configs": ["main"],
                "split": "train",
                "question_field": "question",
                "answer_field": "answer",
                "output_field": "output",
                "description": "Grade school math problems"
            },
            {
                "dataset_name": "math_qa",
                "configs": ["default"],
                "split": "train",
                "question_field": "Problem",
                "answer_field": "correct",
                "output_field": "Rationale",
                "description": "Math word problems"
            }
        ],
        "processing_options": {
            "max_items_per_dataset": 100,
            "grade_threshold": "C",
            "evaluator_models": ["deepseek/deepseek-r1-0528:free"],
            "regenerator_models": ["deepseek/deepseek-r1-0528:free"],
            "use_async_regeneration": False
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Created example configuration file: {output_path}")
    return output_path


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process Hugging Face datasets with CoT evaluation and regeneration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_datasets",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--evaluator-models",
        type=str,
        help="Comma-separated list of models for evaluation"
    )
    parser.add_argument(
        "--regenerator-models",
        type=str,
        help="Comma-separated list of models for regeneration"
    )
    parser.add_argument(
        "--grade-threshold",
        type=str,
        default="C",
        choices=["A", "B", "C", "D"],
        help="Regenerate items with this grade or lower"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum items to process per dataset (for testing)"
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Disable async multi-model regeneration"
    )
    parser.add_argument(
        "--create-example-config",
        action="store_true",
        help="Create an example configuration file and exit"
    )
    parser.add_argument(
        "--skip-existing",
        type=str,
        default="false",
        help="Skip datasets that have already been processed (true/false, default: false)"
    )
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_example_config:
        config_path = create_example_config()
        print("\nExample usage:")
        print(f"  python {__file__} --config {config_path} --max-items 10")
        return
    
    # Check for config file
    if not args.config:
        print("Error: --config is required (or use --create-example-config)")
        parser.print_help()
        return
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY not set in environment")
        print("Please set it in .env file or environment variable")
        return
    
    # Parse model lists
    evaluator_models = None
    if args.evaluator_models:
        evaluator_models = [m.strip() for m in args.evaluator_models.split(",")]
    
    regenerator_models = None
    if args.regenerator_models:
        regenerator_models = [m.strip() for m in args.regenerator_models.split(",")]
    
    # Parse skip_existing argument
    skip_existing = args.skip_existing.lower() in ['true', 'yes', '1', 't']
    
    # Initialize processor
    processor = HFDatasetProcessor(
        output_dir=args.output_dir,
        evaluator_models=evaluator_models,
        regenerator_models=regenerator_models,
        grade_threshold=args.grade_threshold,
        max_items_per_dataset=args.max_items,
        use_async_regeneration=not args.no_async,
        skip_existing=skip_existing
    )
    
    # Process all datasets
    logger.info("Starting dataset processing pipeline...")
    start_time = time.time()
    
    global_stats = await processor.process_all_datasets(args.config)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    print(f"\nGlobal Statistics:")
    print(f"  Datasets processed: {global_stats['total_datasets']}")
    if global_stats.get('total_skipped', 0) > 0:
        print(f"  Datasets skipped: {global_stats['total_skipped']}")
    print(f"  Items processed: {global_stats['total_items_processed']}")
    print(f"  Items evaluated: {global_stats['total_items_evaluated']}")
    print(f"  Items regenerated: {global_stats['total_items_regenerated']}")
    print(f"  Improvements: {global_stats['total_improvements']}")
    print(f"  Errors: {len(global_stats['errors'])}")
    print(f"\nGrade Distribution:")
    for grade in ['A', 'B', 'C', 'D']:
        print(f"  Grade {grade}: {global_stats['grade_distribution'][grade]}")
    
    print(f"\nDetailed reports saved in: {args.output_dir}")
    print("  - processing_statistics.json")
    print("  - summary_report.txt")


if __name__ == "__main__":
    try:
        # Try to run with asyncio.run() first
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # If there's already an event loop running, get it and run the coroutine
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            # Re-raise other RuntimeErrors
            raise