#!/usr/bin/env python3
"""
Shared evaluation utility functions for alignment methods.
"""

import json
import logging
from typing import Callable, Dict, List

from vllm import LLM, SamplingParams
from .drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str] = None,
    reward_fn: Callable[[str, str], Dict[str, float]] = None,
    output_path: str = None
) -> List[Dict]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    
    if ground_truths is None:
        logger.error("ground_truths must be provided for evaluation")
        return []
    
    # Use r1_zero_reward_fn as default for MATH evaluation
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn
        logger.info("Using default r1_zero_reward_fn for MATH evaluation")
    
    assert len(prompts) == len(ground_truths)
    
    logger.info(f"Evaluating {len(prompts)} examples...")
    
    # Generate responses
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    for i, (prompt, gt, output) in enumerate(zip(prompts, ground_truths, outputs)):
        response = output.outputs[0].text.strip()
        
        # Compute rewards
        rewards = reward_fn(response, gt)
        
        # Extract answer from ground truth (last number after ####)
        if "####" in gt:
            gt_answer = gt.split("####")[-1].strip()
        else:
            gt_answer = gt
        
        result = {
            "example_id": i,
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "ground_truth_answer": gt_answer,
            "format_reward": rewards.get("format_reward", 0.0),
            "answer_reward": rewards.get("answer_reward", 0.0),
            "reward": rewards.get("reward", 0.0),
        }
        results.append(result)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(prompts)} examples")
    
    # Serialize results to disk if output_path is provided
    if output_path:
        save_results_to_disk(results, output_path)
    
    return results


def save_results_to_disk(results: List[Dict], output_path: str) -> None:
    """Save evaluation results to disk in JSONL format."""
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Results saved to {output_path}")


def load_model(model_name: str = "qwen_math_15b"):
    """Load a model using the configuration."""
    from .config import MODEL_CONFIGS
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    try:
        model = LLM(
            model=config["path"],
            trust_remote_code=config["trust_remote_code"],
            dtype=config["dtype"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
        )
        logger.info(f"Model {model_name} loaded successfully from {config['path']}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def create_math_eval_sampling_params():
    """Create sampling parameters optimized for MATH evaluation."""
    from .config import DEFAULT_EVAL_SAMPLING_PARAMS
    
    sampling_params = SamplingParams(
        temperature=DEFAULT_EVAL_SAMPLING_PARAMS["temperature"],
        top_p=DEFAULT_EVAL_SAMPLING_PARAMS["top_p"],
        max_tokens=DEFAULT_EVAL_SAMPLING_PARAMS["max_tokens"],
        stop=DEFAULT_EVAL_SAMPLING_PARAMS["stop"],
        include_stop_str_in_output=DEFAULT_EVAL_SAMPLING_PARAMS["include_stop_str_in_output"],
    )
    
    logger.info("Created MATH evaluation sampling parameters with </answer> stop token")
    return sampling_params


def evaluate_math(
    vllm_model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    output_path: str = None
) -> List[Dict]:
    """
    Evaluate a language model on MATH problems using r1_zero_reward_fn.
    
    This function automatically uses the correct sampling parameters and reward function
    for MATH evaluation as specified in the assignment.
    """
    sampling_params = create_math_eval_sampling_params()
    
    return evaluate_vllm(
        vllm_model=vllm_model,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
        reward_fn=r1_zero_reward_fn,  # Use the specified reward function
        output_path=output_path
    )


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute evaluation metrics from results."""
    
    if not results:
        return {}
    
    total = len(results)
    
    # Count correct cases
    correct_format_answer = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0)
    correct_format_wrong_answer = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0)
    wrong_format = sum(1 for r in results if r["format_reward"] == 0.0)
    
    # Calculate percentages
    format_accuracy = (correct_format_answer + correct_format_wrong_answer) / total * 100
    answer_accuracy = correct_format_answer / total * 100
    overall_accuracy = correct_format_answer / total * 100
    
    return {
        "total_examples": total,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,
        "overall_accuracy": overall_accuracy,
        "correct_format_and_answer": correct_format_answer,
        "correct_format_wrong_answer": correct_format_wrong_answer,
        "wrong_format": wrong_format,
    }
