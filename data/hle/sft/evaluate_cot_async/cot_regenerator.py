#!/usr/bin/env python3
"""
CoT Regenerator Module

A reusable module for regenerating Chain of Thought (CoT) reasoning based on evaluation feedback.
Supports asynchronous multi-model regeneration and automatic selection of the best result.

Usage:
    from cot_regenerator import CoTRegenerator
    
    regenerator = CoTRegenerator(
        models=["deepseek/deepseek-r1-0528:free", "claude-3-5-sonnet-20241022"],
        api_keys={"openrouter": "key1", "anthropic": "key2"}
    )
    
    # Single model regeneration
    result = regenerator.regenerate_single(question, answer, previous_cot, evaluation_details)
    
    # Multi-model async regeneration with best selection
    best_result = await regenerator.regenerate_multi_async(
        question, answer, previous_cot, evaluation_details
    )
"""

import json
import os
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for a specific model."""
    
    def __init__(
        self,
        model_name: str,
        api_base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 40000,
        retry_attempts: int = 3
    ):
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts


class CoTRegenerator:
    """Regenerates CoT reasoning based on evaluation feedback."""
    
    # Default system prompt for regeneration
    DEFAULT_SYSTEM_PROMPT = """You are a problem-solving expert who learns from feedback and has the ability to improve solutions.

Improvement guidelines:
1. Focus on improving weaknesses identified in the evaluation
2. Maintain aspects identified as strengths
3. Ensure logical completeness and independence
4. Strengthen step-by-step derivation and verification
5. Explicitly address common mistakes
6. Add metacognitive elements (e.g., why you chose certain methods)

Important notes:
- Generate as a completely independent solution (avoid references to previous attempts)
- Avoid external references (e.g., "according to textbooks", "generally speaking")
- Divide the thought process into clear steps
- Always verify the validity of results
- Always reach the final answer"""
    
    def __init__(
        self,
        models: Optional[Union[str, List[str], List[ModelConfig]]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        default_api_base: str = "https://openrouter.ai/api/v1",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the CoT Regenerator.
        
        Args:
            models: Single model name, list of model names, or list of ModelConfig objects
            api_keys: Dictionary of API keys (e.g., {"openrouter": "key", "anthropic": "key"})
            default_api_base: Default API base URL
            system_prompt: Custom system prompt for regeneration
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.api_keys = api_keys or {}
        self.default_api_base = default_api_base
        
        # Initialize model configurations
        self.model_configs = self._initialize_models(models)
        
        # Initialize evaluator for scoring regenerated CoTs
        try:
            from cot_evaluator import CoTEvaluator
            self.evaluator = CoTEvaluator()
        except ImportError:
            logger.warning("CoTEvaluator not available. Best result selection will use simple heuristics.")
            self.evaluator = None
    
    def _initialize_models(self, models: Optional[Union[str, List[str], List[ModelConfig]]]) -> List[ModelConfig]:
        """Initialize model configurations."""
        if models is None:
            # Default model
            return [ModelConfig("deepseek/deepseek-r1-0528:free")]
        
        if isinstance(models, str):
            return [ModelConfig(models)]
        
        configs = []
        for model in models:
            if isinstance(model, ModelConfig):
                configs.append(model)
            elif isinstance(model, str):
                configs.append(ModelConfig(model))
            else:
                raise ValueError(f"Invalid model configuration: {model}")
        
        return configs
    
    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        previous_cot: str,
        evaluation_details: Dict[str, Any]
    ) -> str:
        """Build the user prompt for regeneration."""
        # Extract evaluation details
        previous_grade = evaluation_details.get('grade', 'Unknown')
        strengths = evaluation_details.get('strengths', [])
        weaknesses = evaluation_details.get('weaknesses', [])
        improvement_suggestions = evaluation_details.get('improvement_suggestions', [])
        learning_value_scores = evaluation_details.get('learning_value_scores', {})
        
        # Format text sections
        strengths_text = "\n".join(f"• {s}" for s in strengths) if strengths else "• None identified"
        weaknesses_text = "\n".join(f"• {w}" for w in weaknesses) if weaknesses else "• None identified"
        suggestions_text = "\n".join(f"• {s}" for s in improvement_suggestions) if improvement_suggestions else "• None provided"
        
        # Build detailed scores
        scores_lines = []
        if learning_value_scores:
            scores_lines.append(f"• Method explanation: {learning_value_scores.get('method_explanation', 0)}/10")
            scores_lines.append(f"• Step-by-step derivation: {learning_value_scores.get('step_by_step', 0)}/10")
            scores_lines.append(f"• Verification and checking: {learning_value_scores.get('verification', 0)}/10")
            scores_lines.append(f"• Handling common mistakes: {learning_value_scores.get('common_mistakes', 0)}/10")
            scores_lines.append(f"• Domain-specific insight: {learning_value_scores.get('domain_insight', 0)}/10")
            scores_lines.append(f"• Metacognitive elements: {learning_value_scores.get('metacognitive', 0)}/10")
        scores_text = "\n".join(scores_lines) if scores_lines else "• No scores available"
        
        # Extract CoT content (remove <think> tags if present)
        if '<think>' in previous_cot and '</think>' in previous_cot:
            start = previous_cot.find('<think>') + len('<think>')
            end = previous_cot.find('</think>')
            previous_cot_content = previous_cot[start:end].strip()
        else:
            previous_cot_content = previous_cot
        
        return f"""Problem:
{question}

Expected answer:
{answer}

===============================================================================
Previous solution:
{previous_cot_content}

===============================================================================
Evaluation results:

Grade: {previous_grade}

Strengths (to maintain):
{strengths_text}

Weaknesses (to improve):
{weaknesses_text}

Detailed scores:
{scores_text}

Improvement suggestions:
{suggestions_text}

===============================================================================

Based on the feedback above, generate an improved solution.
Maintain the strengths while focusing on improving the weaknesses and low-scoring areas."""
    
    def regenerate_single(
        self,
        question: str,
        answer: str,
        previous_cot: str,
        evaluation_details: Dict[str, Any],
        model_config: Optional[ModelConfig] = None
    ) -> Optional[str]:
        """
        Regenerate CoT using a single model.
        
        Args:
            question: The problem/question
            answer: The expected answer
            previous_cot: The previous CoT reasoning
            evaluation_details: Evaluation details from the evaluator
            model_config: Optional specific model configuration to use
        
        Returns:
            Regenerated CoT text (without <think> tags) or None if failed
        """
        config = model_config or self.model_configs[0]
        user_prompt = self._build_user_prompt(question, answer, previous_cot, evaluation_details)
        
        # Create client
        client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url
        )
        
        # Retry logic
        for attempt in range(config.retry_attempts):
            try:
                response = client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < config.retry_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Error with {config.model_name}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {config.retry_attempts} attempts with {config.model_name}: {e}")
                    return None
    
    async def _regenerate_single_async(
        self,
        question: str,
        answer: str,
        previous_cot: str,
        evaluation_details: Dict[str, Any],
        model_config: ModelConfig
    ) -> Tuple[str, Optional[str]]:
        """
        Async version of single model regeneration.
        
        Returns:
            Tuple of (model_name, regenerated_cot)
        """
        user_prompt = self._build_user_prompt(question, answer, previous_cot, evaluation_details)
        
        # Create async client
        client = AsyncOpenAI(
            api_key=model_config.api_key,
            base_url=model_config.api_base_url
        )
        
        try:
            # Retry logic
            for attempt in range(model_config.retry_attempts):
                try:
                    response = await client.chat.completions.create(
                        model=model_config.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
                    )
                    return (model_config.model_name, response.choices[0].message.content)
                except Exception as e:
                    if attempt < model_config.retry_attempts - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Error with {model_config.model_name}: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {model_config.retry_attempts} attempts with {model_config.model_name}: {e}")
                        return (model_config.model_name, None)
        finally:
            # Ensure client is properly closed
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Error closing client for {model_config.model_name}: {e}")
    
    async def regenerate_multi_async(
        self,
        question: str,
        answer: str,
        previous_cot: str,
        evaluation_details: Dict[str, Any],
        return_all: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Regenerate CoT using multiple models asynchronously and select the best result.
        
        Args:
            question: The problem/question
            answer: The expected answer
            previous_cot: The previous CoT reasoning
            evaluation_details: Evaluation details from the evaluator
            return_all: If True, return all results; if False, return only the best
        
        Returns:
            If return_all is False: Dictionary with best result and metadata
            If return_all is True: List of all results with metadata
        """
        # Create async tasks for all models
        tasks = [
            self._regenerate_single_async(question, answer, previous_cot, evaluation_details, config)
            for config in self.model_configs
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Process results
        all_results = []
        for model_name, cot in results:
            if cot is not None:
                result = {
                    "model": model_name,
                    "cot": cot,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Evaluate if evaluator is available
                if self.evaluator:
                    try:
                        eval_result = await self._evaluate_cot_async(question, cot, answer)
                        result["evaluation"] = eval_result
                        result["score"] = eval_result.get("score", 0)
                        result["grade"] = eval_result.get("grade", "D")
                    except Exception as e:
                        logger.warning(f"Failed to evaluate CoT from {model_name}: {e}")
                        result["score"] = 0
                        result["grade"] = "D"
                else:
                    # Simple heuristic scoring based on length and structure
                    result["score"] = self._simple_score(cot)
                    result["grade"] = self._score_to_grade(result["score"])
                
                all_results.append(result)
        
        if not all_results:
            logger.error("All models failed to regenerate CoT")
            return None if not return_all else []
        
        # Sort by score (highest first)
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        if return_all:
            return all_results
        else:
            # Return the best result
            best = all_results[0]
            return {
                "best_model": best["model"],
                "best_cot": best["cot"],
                "best_score": best.get("score", 0),
                "best_grade": best.get("grade", "Unknown"),
                "all_models_tried": [r["model"] for r in all_results],
                "timestamp": datetime.now().isoformat()
            }
    
    async def _evaluate_cot_async(self, question: str, cot: str, answer: str) -> Dict[str, Any]:
        """Evaluate a CoT asynchronously."""
        if not self.evaluator:
            return {}
        
        # Run evaluation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.evaluator.evaluate,
                question,
                cot,
                answer
            )
        return result
    
    def _simple_score(self, cot: str) -> float:
        """Simple heuristic scoring when evaluator is not available."""
        score = 0.0
        
        # Check for step-by-step structure
        if any(marker in cot.lower() for marker in ["step", "first", "then", "finally"]):
            score += 2.0
        
        # Check for verification
        if any(word in cot.lower() for word in ["verify", "check", "confirm"]):
            score += 1.5
        
        # Check for explanation
        if any(word in cot.lower() for word in ["because", "therefore", "since", "thus"]):
            score += 1.5
        
        # Length bonus (reasonable length)
        word_count = len(cot.split())
        if 200 <= word_count <= 2000:
            score += 2.0
        elif 100 <= word_count < 200:
            score += 1.0
        
        # Check for mathematical/logical markers
        if any(marker in cot for marker in ["=", "→", "∴", "∵"]):
            score += 1.0
        
        # Cap at 10
        return min(score, 10.0)
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 8:
            return "A"
        elif score >= 6:
            return "B"
        elif score >= 4:
            return "C"
        else:
            return "D"
    
    def regenerate_dataset(
        self,
        dataset_path: Path,
        output_path: Optional[Path] = None,
        grade_threshold: str = "C",
        specific_ids: Optional[List[str]] = None,
        use_async: bool = True
    ) -> Dict[str, Any]:
        """
        Regenerate CoTs for an entire dataset.
        
        Args:
            dataset_path: Path to the input JSONL dataset
            output_path: Path for output (defaults to input path)
            grade_threshold: Regenerate items with this grade or lower
            specific_ids: Optional list of specific IDs to regenerate
            use_async: Whether to use async multi-model regeneration
        
        Returns:
            Dictionary with regeneration statistics
        """
        from pathlib import Path
        import json
        
        # Load dataset
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        output_path = output_path or dataset_path
        
        # Grade order for comparison
        grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        threshold_value = grade_order.get(grade_threshold, 0)
        
        # Filter items to regenerate
        items_to_regenerate = []
        for item in data:
            if not item.get('metadata', {}).get('cot_history'):
                continue
            
            latest = item['metadata']['cot_history'][-1]
            if not latest.get('evaluation'):
                continue
            
            item_id = str(item.get('id', ''))
            grade = latest['evaluation'].get('grade', 'D')
            
            if specific_ids:
                if item_id in specific_ids:
                    items_to_regenerate.append(item)
            elif grade_order.get(grade, 0) <= threshold_value:
                items_to_regenerate.append(item)
        
        # Statistics
        stats = {
            "total_items": len(data),
            "items_to_regenerate": len(items_to_regenerate),
            "successful": 0,
            "failed": 0,
            "improved": 0
        }
        
        # Process each item
        for item in items_to_regenerate:
            latest = item['metadata']['cot_history'][-1]
            previous_cot = latest.get('output', '')
            evaluation_details = latest['evaluation']
            
            try:
                if use_async and len(self.model_configs) > 1:
                    # Multi-model async regeneration
                    result = asyncio.run(self.regenerate_multi_async(
                        item['question'],
                        item['answer'],
                        previous_cot,
                        evaluation_details
                    ))
                    
                    if result:
                        new_cot = result['best_cot']
                        new_output = f"<think>{new_cot}</think>{item['answer']}"
                        
                        # Add to history
                        new_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "output": new_output,
                            "evaluation": None,
                            "model": result['best_model'],
                            "regeneration_metadata": {
                                "models_tried": result['all_models_tried'],
                                "best_score": result['best_score'],
                                "best_grade": result['best_grade']
                            }
                        }
                        item['metadata']['cot_history'].append(new_entry)
                        item['output'] = new_output
                        
                        stats['successful'] += 1
                        if result['best_grade'] > evaluation_details.get('grade', 'D'):
                            stats['improved'] += 1
                else:
                    # Single model regeneration
                    new_cot = self.regenerate_single(
                        item['question'],
                        item['answer'],
                        previous_cot,
                        evaluation_details
                    )
                    
                    if new_cot:
                        new_output = f"<think>{new_cot}</think>{item['answer']}"
                        
                        # Add to history
                        new_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "output": new_output,
                            "evaluation": None,
                            "model": self.model_configs[0].model_name
                        }
                        item['metadata']['cot_history'].append(new_entry)
                        item['output'] = new_output
                        
                        stats['successful'] += 1
                
            except Exception as e:
                logger.error(f"Failed to regenerate item {item.get('id')}: {e}")
                stats['failed'] += 1
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        return stats


# Convenience function for backwards compatibility
def regenerate_cot(
    client,
    question: str,
    answer: str,
    previous_cot: str,
    evaluation_details: Dict[str, Any]
) -> Optional[str]:
    """
    Legacy function for compatibility with existing code.
    Uses the CoTRegenerator class internally.
    """
    regenerator = CoTRegenerator()
    return regenerator.regenerate_single(question, answer, previous_cot, evaluation_details)