#!/usr/bin/env python3
"""
Run Expert Iteration on GSM8K dataset.
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .trainer import EITrainer
from .config import EIConfig

# Import shared utilities
from ..shared.config import RESULTS_DIR, GSM8K_TRAIN_PATH
from ..shared.math_data_utils import format_r1_zero_prompt
from ..shared.wandb_config import log_metrics, log_generation_examples

# Import reward function
try:
    from ..shared.drgrpo_grader import r1_zero_reward_fn as reward_fn
except Exception:
    import re
    _BOX = re.compile(r"\\boxed\{([^}]*)\}")
    def _final(x:str):
        m = list(_BOX.finditer(x or ""))
        if m: return m[-1].group(1).strip()
        nums = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", x or "")
        return nums[-1].strip() if nums else None
    def reward_fn(resp:str, gt:str)->dict:
        p=_final(resp); g=_final(gt)
        ok = 1.0 if (p is not None and g is not None and p==g) else 0.0
        return {"reward": ok, "format_reward": 1.0, "answer_reward": ok}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("math_ei")


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def convert_gsm8k_to_questions(rows: List[Dict[str, str]]) -> List[str]:
    """Convert GSM8K format to question list for EI."""
    questions = []
    for row in rows:
        question = row["question"]
        # Format question using r1_zero template
        formatted_question = format_r1_zero_prompt(question)
        questions.append(formatted_question)
    return questions


def main():
    """Run Expert Iteration on GSM8K dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / "ei_qwen_math_15b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 配置 ===
    cfg = EIConfig(
        # Use shared wandb settings
        wandb_enabled=True,
        wandb_project="cs336-ei-math",
        wandb_name="qwen2.5-math-1.5b-ei",
        wandb_tags=["ei", "math", "qwen2.5"],
        # Memory optimization
        use_flash_attention=True,
        torch_dtype="bfloat16",
        # EI parameters
        n_ei_steps=3,
        G=8,
        sampling_temperature=1.0,
        sampling_max_tokens=512,
        sampling_min_tokens=4,
        # Training parameters
        batch_size=2,
        grad_accum=8,
        max_steps=1000,  # Per EI iteration
        lr=2e-5,
    )

    # === 数据 ===
    logger.info("Loading GSM8K training data...")
    train_rows = load_jsonl(GSM8K_TRAIN_PATH)
    questions = convert_gsm8k_to_questions(train_rows)
    logger.info(f"Loaded {len(questions)} questions")

    # === 训练器 ===
    logger.info("Initializing EI trainer...")
    trainer = EITrainer(cfg)

    # === 运行Expert Iteration ===
    logger.info("Starting Expert Iteration...")
    trainer.run_expert_iteration(questions, reward_fn)

    # === 保存最终权重 ===
    logger.info("Saving final model...")
    trainer.save(str(out_dir))
    logger.info(f"Saved to {out_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
