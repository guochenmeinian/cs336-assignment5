#!/usr/bin/env python3
"""
Evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH dataset.

This script implements the requirements from the assignment:
1. Load MATH validation examples
2. Format them as string prompts using the r1_zero prompt
3. Generate outputs for each example using vLLM
4. Calculate evaluation metrics using the r1_zero_reward_fn
5. Serialize examples, model generations, and evaluation scores to disk
"""

import logging
from pathlib import Path

from ..shared.math_evaluation_utils import evaluate_vllm, load_model
from ..shared.config import EVALUATIONS_DIR, MATH_TEST_PATH, DEFAULT_EVAL_SAMPLING_PARAMS
from ..shared.math_data_utils import load_math_data, format_r1_zero_prompt
from ..shared.drgrpo_grader import r1_zero_reward_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run Qwen 2.5 Math 1.5B zero-shot evaluation on MATH test dataset."""
    
    # Configuration
    model_name = "qwen_math_15b"
    test_data_path = str(MATH_TEST_PATH)
    output_path = EVALUATIONS_DIR / "math_baseline.jsonl"
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use default sampling parameters from config
    from vllm import SamplingParams
    eval_sampling_params = SamplingParams(**DEFAULT_EVAL_SAMPLING_PARAMS)
    
    logger.info(f"Loading Qwen 2.5 Math 1.5B model: {model_name}")
    
    # Load the model using shared utility
    try:
        model = load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    test_data = load_math_data(test_data_path)
    
    # Format prompts using r1_zero template
    logger.info("Formatting prompts using r1_zero template...")
    prompts = []
    ground_truths = []
    
    for example in test_data:
        question = example["question"]
        answer = example["answer"]
        
        # Extract pure answer from <answer>...</answer> format (MATH dataset)
        if "<answer>" in answer and "</answer>" in answer:
            pure_answer = answer.split("<answer>")[-1].replace("</answer>", "").strip()
        else:
            # This should not happen with our MATH dataset
            logger.warning(f"Unexpected answer format: {answer[:100]}...")
            pure_answer = answer
        
        # Format prompt using r1_zero template
        prompt = format_r1_zero_prompt(question)
        prompts.append(prompt)
        ground_truths.append(pure_answer)
    
    logger.info(f"Formatted {len(prompts)} prompts")
    
    # Evaluate the model using shared evaluate_vllm function
    logger.info("Starting evaluation...")
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        ground_truths=ground_truths,
        output_path=str(output_path),
    )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {output_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
