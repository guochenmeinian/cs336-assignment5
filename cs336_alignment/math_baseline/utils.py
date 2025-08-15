#!/usr/bin/env python3
"""
Utility functions for GSM8K evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def load_gsm8k_data(data_path: str) -> List[Dict]:
    """Load GSM8K data from JSONL file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    return data


def load_r1_zero_prompt() -> str:
    """Load the r1_zero prompt template."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "r1_zero.prompt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_r1_zero_prompt(question: str) -> str:
    """Format a question using the r1_zero prompt template."""
    prompt_template = load_r1_zero_prompt()
    return prompt_template.format(question=question)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
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
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary statistics
    format_rewards = [r["format_reward"] for r in results]
    answer_rewards = [r["answer_reward"] for r in results]
    total_rewards = [r["reward"] for r in results]
    
    print(f"\n==== EVALUATION SUMMARY ====")
    print(f"Total examples: {len(results)}")
    print(f"Format reward = 1.0: {sum(1 for r in format_rewards if r == 1.0)} ({sum(1 for r in format_rewards if r == 1.0) / len(format_rewards) * 100:.2f}%)")
    print(f"Answer reward = 1.0: {sum(1 for r in answer_rewards if r == 1.0)} ({sum(1 for r in answer_rewards if r == 1.0) / len(answer_rewards) * 100:.2f}%)")
    print(f"Total reward = 1.0: {sum(1 for r in total_rewards if r == 1.0)} ({sum(1 for r in total_rewards if r == 1.0) / len(total_rewards) * 100:.2f}%)")
    
    # Categorize results
    correct_format_answer = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0]
    correct_format_wrong_answer = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]
    wrong_format = [r for r in results if r["format_reward"] == 0.0]
    
    print(f"\n1) Correct format & answer: {len(correct_format_answer)} ({len(correct_format_answer) / len(results) * 100:.2f}%)")
    print(f"2) Correct format, wrong answer: {len(correct_format_wrong_answer)} ({len(correct_format_wrong_answer) / len(results) * 100:.2f}%)")
    print(f"3) Wrong format: {len(wrong_format)} ({len(wrong_format) / len(results) * 100:.2f}%)")
