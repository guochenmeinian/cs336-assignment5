#!/usr/bin/env python3
"""
Script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on GSM8K.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from vllm import LLM, SamplingParams

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_gsm8k_data,
    format_r1_zero_prompt,
    evaluate_vllm,
)

# Import reward function from drgrpo_grader
sys.path.append(str(Path(__file__).parent.parent))
from drgrpo_grader import r1_zero_reward_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    
    # Configuration
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    validation_data_path = "../../data/gsm8k_split/validation.jsonl"
    output_path = "gsm8k_baseline_results.jsonl"
    
    # Sampling parameters for evaluation
    eval_sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for evaluation
        max_tokens=512,
        stop=["User:", "Human:", "Assistant:"],  # Stop at conversation boundaries
    )
    
    logger.info(f"Loading model: {model_name}")
    
    # Load the model
    try:
        model = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",  # Use bfloat16 for efficiency
            gpu_memory_utilization=0.9,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load validation data
    logger.info(f"Loading validation data from {validation_data_path}")
    validation_data = load_gsm8k_data(validation_data_path)
    
    # Format prompts using r1_zero template
    logger.info("Formatting prompts...")
    prompts = []
    ground_truths = []
    
    for example in validation_data:
        question = example["question"]
        answer = example["answer"]
        
        # Format prompt using r1_zero template
        prompt = format_r1_zero_prompt(question)
        prompts.append(prompt)
        ground_truths.append(answer)
    
    logger.info(f"Formatted {len(prompts)} prompts")
    
    # Evaluate the model
    logger.info("Starting evaluation...")
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=eval_sampling_params,
        output_path=output_path,
    )
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
