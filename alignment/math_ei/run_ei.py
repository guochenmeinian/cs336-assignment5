#!/usr/bin/env python3
"""
Run Expert Iteration on MATH dataset.
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
from ..shared.config import MODEL_DIR, MATH_TRAIN_PATH
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


def convert_math_to_questions_and_answers(rows: List[Dict[str, str]]) -> tuple[List[str], List[str]]:
    """Convert MATH format to question and answer lists for EI."""
    questions = []
    answers = []
    for row in rows:
        question = row["question"]
        answer = row["answer"]
        # Format question using r1_zero template
        formatted_question = format_r1_zero_prompt(question)
        questions.append(formatted_question)
        answers.append(answer)
    return questions, answers


def main():
    """Run Expert Iteration on MATH dataset."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = MODEL_DIR / "ei_qwen_math_15b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Configuration ===
    # Use default config from math_ei/config.py
    cfg = EIConfig()
    
    # Only override wandb settings
    cfg.wandb_enabled = False

    # === Data ===
    logger.info("Loading MATH training data...")
    train_rows = load_jsonl(MATH_TRAIN_PATH)
    questions, answers = convert_math_to_questions_and_answers(train_rows)
    logger.info(f"Loaded {len(questions)} questions and {len(answers)} answers")

    # === Trainer ===
    logger.info("Initializing EI trainer...")
    trainer = EITrainer(cfg)

    # === Run Expert Iteration ===
    logger.info("Starting Expert Iteration...")
    trainer.run_expert_iteration(questions, answers, reward_fn)

    # === Save final weights ===
    logger.info("Saving final model...")
    trainer.save(str(out_dir))
    logger.info(f"Saved to {out_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
