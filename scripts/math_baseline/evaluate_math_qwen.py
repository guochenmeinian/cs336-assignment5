#!/usr/bin/env python3
"""
Evaluate Qwen 2.5 Math 1.5B zero-shot performance on GSM8K dataset.
- Loads test data
- Formats prompts with r1_zero
- Generates outputs with VLLM
- Computes metrics using r1_zero_reward_fn
- Serializes per-example results and aggregate metrics
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import reusable code from the package
from cs336_alignment.math_baseline import (
    evaluate_vllm,
    format_r1_zero_prompt,
    load_gsm8k_data,
)

# Reward fn from alignment grader (preferred); fallback if unavailable
try:
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn  # type: ignore
except Exception:  # pragma: no cover
    def r1_zero_reward_fn(response, ground_truth, fast=True):  # type: ignore
        if "</think> <answer>" in response and "</answer>" in response:
            model_answer = response.split("<answer>")[-1].replace("</answer>", "").strip()
            if isinstance(ground_truth, (float, int)):
                ground_truth = str(ground_truth)
            is_correct = model_answer.strip() == str(ground_truth).strip()
            return {"format_reward": 1.0, "answer_reward": 1.0 if is_correct else 0.0, "reward": 1.0 if is_correct else 0.0}
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 Math 1.5B on GSM8K (zero-shot)")
    parser.add_argument("--model-path", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data-path", type=str, default="/data/a5-alignment/gsm8k/test.jsonl")
    parser.add_argument("--output-path", type=str, default=str(PROJECT_ROOT / "results/math_baseline/qwen_gsm8k_baseline.jsonl"))
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    if not os.path.exists(args.model_path):
        logger.error("Model path does not exist: %s", args.model_path)
        sys.exit(1)
    if not os.path.exists(args.data_path):
        logger.error("Data path does not exist: %s", args.data_path)
        sys.exit(1)

    logger.info("Loading GSM8K test data from %s", args.data_path)
    examples = load_gsm8k_data(args.data_path)

    # Extract questions and ground truths from GSM8K format
    questions = [ex["question"] for ex in examples]
    ground_truths = [ex["answer"] for ex in examples]

    logger.info("Formatting prompts with r1_zero template")
    prompts = [format_r1_zero_prompt(q) for q in questions]

    logger.info("Loading VLLM model from %s", args.model_path)
    model = LLM(model=args.model_path, tensor_parallel_size=args.num_gpus, trust_remote_code=True, max_model_len=2048)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>", "\n\n", "User:", "Assistant:"],
    )

    logger.info("Starting evaluation...")
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=args.output_path,
    )
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
