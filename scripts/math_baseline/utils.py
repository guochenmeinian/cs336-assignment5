#!/usr/bin/env python3
"""
Utilities for the math_baseline problem.
Includes:
- load_gsm8k_data
- load_r1_zero_prompt
- format_r1_zero_prompt
- evaluate_vllm
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    # scripts/math_baseline/utils.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def load_gsm8k_data(data_path: str) -> List[Dict]:
    """Load GSM8K examples from JSONL file."""
    examples: List[Dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    logger.info("Loaded %d examples from %s", len(examples), data_path)
    return examples


def load_r1_zero_prompt() -> str:
    """Load r1_zero prompt template from cs336_alignment/prompts/r1_zero.prompt."""
    prompt_path = _project_root() / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    if not prompt_path.exists():
        logger.warning("r1_zero prompt not found at %s; using fallback template", prompt_path)
        return (
            "A conversation between User and Assistant. The User asks a question, and the Assistant "
            "solves it. The Assistant first thinks about the reasoning process in the mind and then "
            "provides the User with the answer. The reasoning process is enclosed within <think> </think> "
            "and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning "
            "process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>"
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def format_r1_zero_prompt(question: str) -> str:
    """Format a question using the r1_zero prompt template."""
    prompt_template = load_r1_zero_prompt()
    return prompt_template.format(question=question)


def _sampling_params_to_dict(sp: SamplingParams) -> Dict[str, Any]:
    return {
        "temperature": sp.temperature,
        "top_p": sp.top_p,
        "max_tokens": sp.max_tokens,
        "stop": getattr(sp, "stop", None),
    }


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
    *,
    model_name_or_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    Returns metrics dict.
    """
    logger.info("Generating responses for %d prompts...", len(prompts))
    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)

    results: List[Dict] = []
    total_correct_format_answer = 0
    total_correct_format_wrong_answer = 0
    total_wrong_format = 0

    for i, (output, ground_truth) in enumerate(zip(raw_outputs, ground_truths)):
        response = output.outputs[0].text.strip()
        rewards = reward_fn(response, ground_truth)

        format_reward = rewards.get("format_reward", 0.0)
        answer_reward = rewards.get("answer_reward", 0.0)

        if format_reward == 1.0 and answer_reward == 1.0:
            total_correct_format_answer += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            total_correct_format_wrong_answer += 1
        elif format_reward == 0.0:
            total_wrong_format += 1

        result = {
            "example_id": i,
            "prompt": prompts[i],
            "response": response,
            "ground_truth": ground_truth,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "total_reward": rewards.get("reward", 0.0),
            "rewards": rewards,
        }
        if model_name_or_path is not None:
            result["model_name_or_path"] = model_name_or_path
            result["sampling_params"] = _sampling_params_to_dict(eval_sampling_params)

        results.append(result)

    total_examples = len(results)
    metrics = {
        "total_examples": total_examples,
        "correct_format_and_answer": total_correct_format_answer,
        "correct_format_wrong_answer": total_correct_format_wrong_answer,
        "wrong_format": total_wrong_format,
        "format_accuracy": (total_correct_format_answer + total_correct_format_wrong_answer)
        / total_examples
        if total_examples
        else 0.0,
        "answer_accuracy": (total_correct_format_answer / total_examples) if total_examples else 0.0,
        "overall_accuracy": (total_correct_format_answer / total_examples) if total_examples else 0.0,
    }

    logger.info("Evaluation Results: %s", metrics)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Results saved to %s", out_path)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics
