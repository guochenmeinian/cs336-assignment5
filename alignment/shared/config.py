#!/usr/bin/env python3
"""
Configuration file for alignment experiments.
Centralizes wandb settings and filesystem paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables with defaults
def get_env(key: str, default: str) -> str:
    """Get environment variable with default value."""
    return os.getenv(key, default)

def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with default value."""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')

# Model paths (located in autodl-tmp)
QWEN_MATH_15B_PATH = "/root/autodl-tmp/Qwen/Qwen2.5-Math-1.5B"
LLAMA_31_8B_PATH = "/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B"
LLAMA_33_70B_INSTRUCT_PATH = "/root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B"

# Data paths
GSM8K_VALIDATION_PATH = PROJECT_ROOT / "data" / "gsm8k" / "validation.jsonl"
GSM8K_TEST_PATH = PROJECT_ROOT / "data" / "gsm8k" / "test.jsonl"
GSM8K_TRAIN_PATH = PROJECT_ROOT / "data" / "gsm8k" / "train.jsonl"

# Math data directory (统一管理)
MATH_DATA_DIR = PROJECT_ROOT / "data" / "math"
MATH_TRAIN_PATH = MATH_DATA_DIR / "train.jsonl"
MATH_VALIDATION_PATH = MATH_DATA_DIR / "validation.jsonl"
MATH_TEST_PATH = MATH_DATA_DIR / "test.jsonl"

# Evaluations directory (outside the module, as requested)
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"

# Models directory (outside the module, for trained models)
MODEL_DIR = PROJECT_ROOT / "models"

# Wandb Configuration
WANDB_PROJECT = get_env("WANDB_PROJECT", "cs336-math")
WANDB_ENTITY = get_env("WANDB_ENTITY", None)
WANDB_ENABLED = get_env_bool("WANDB_ENABLED", False)

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
        "gpu_memory_utilization": 0.5, # 4090D can only handle ~0.5
    },
    "llama_31_8b": {
        "path": LLAMA_31_8B_PATH,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.7,
    },
    "llama_33_70b_instruct": {
        "path": LLAMA_33_70B_INSTRUCT_PATH,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.7,
    },
}

# SFT training constants
SFT_LOG_EVERY = 50
SFT_EVAL_BATCH_SIZE = 2
SFT_MAX_NEW_TOKENS = 128
