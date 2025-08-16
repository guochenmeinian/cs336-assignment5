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

# Converted GSM8K data paths (with <think> format)
GSM8K_CONVERTED_TRAIN_PATH = PROJECT_ROOT / "data" / "gsm8k_converted" / "train_converted.jsonl"
GSM8K_CONVERTED_VALIDATION_PATH = PROJECT_ROOT / "data" / "gsm8k_converted" / "validation_converted.jsonl"
GSM8K_CONVERTED_TEST_PATH = PROJECT_ROOT / "data" / "gsm8k_converted" / "test_converted.jsonl"

# Results directory (outside the module, as requested)
RESULTS_DIR = PROJECT_ROOT / "results"

# Wandb Configuration
WANDB_PROJECT = get_env("WANDB_PROJECT", "cs336-math")
WANDB_ENTITY = get_env("WANDB_ENTITY", None)
WANDB_NAME = get_env("WANDB_NAME", None)
WANDB_TAGS = get_env("WANDB_TAGS", "alignment").split(",") if get_env("WANDB_TAGS", "") else ["alignment"]
WANDB_ENABLED = get_env_bool("WANDB_ENABLED", True)

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

# SFT-specific configurations
SFT_DEFAULT_CONFIG = {
    "model_id": "qwen_math_15b",  # 使用MODEL_CONFIGS中的key
    "lr": 2e-5,
    "batch_size": 2,
    "grad_accum": 8,
    "max_steps": 2000,
    "max_grad_norm": 1.0,
    "bf16": True,
    "amp": False,
    "use_flash_attention": True,
    "torch_dtype": "bfloat16",
    "seed": 42,
}

# SFT training constants
SFT_LOG_EVERY = 50
SFT_EVAL_BATCH_SIZE = 2
SFT_MAX_NEW_TOKENS = 128
