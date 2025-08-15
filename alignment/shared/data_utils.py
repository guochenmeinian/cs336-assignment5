#!/usr/bin/env python3
"""
Shared data utility functions for alignment methods.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

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


def ensure_directories(*dirs: Path) -> None:
    """Ensure directories exist."""
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True, parents=True)
