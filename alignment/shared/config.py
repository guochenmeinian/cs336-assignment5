#!/usr/bin/env python3
"""
Configuration file for alignment experiments.
Centralizes model paths and result folder paths.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model paths (located in autodl-tmp)
QWEN_MATH_15B_PATH = "/root/autodl-tmp/Qwen/Qwen2.5-Math-1.5B"
LLAMA_31_8B_PATH = "/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B"
LLAMA_33_70B_INSTRUCT_PATH = "/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B"

# Data paths
GSM8K_VALIDATION_PATH = PROJECT_ROOT / "data" / "gsm8k" / "validation.jsonl"
GSM8K_TEST_PATH = PROJECT_ROOT / "data" / "gsm8k" / "test.jsonl"
GSM8K_TRAIN_PATH = PROJECT_ROOT / "data" / "gsm8k" / "train.jsonl"

# Results directory (outside the module, as requested)
RESULTS_DIR = PROJECT_ROOT / "results"

# Default sampling parameters for evaluation
DEFAULT_EVAL_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}

# Model configurations
MODEL_CONFIGS = {
    "qwen_math_15b": {
        "path": QWEN_MATH_15B_PATH,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
    },
    "llama_31_8b": {
        "path": LLAMA_31_8B_PATH,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
    },
    "llama_33_70b_instruct": {
        "path": LLAMA_33_70B_INSTRUCT_PATH,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
    },
}
