#!/usr/bin/env python3
"""
Analyze GSM8K evaluation results and provide insights on model performance.
Groups results into reward categories and prints examples and commentary.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cs336_alignment.math_baseline import load_results, analyze_results, print_analysis

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GSM8K evaluation results")
    parser.add_argument("--results-path", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    if not Path(args.results_path).exists():
        logger.error("Results file does not exist: %s", args.results_path)
        return

    results = load_results(args.results_path)
    analysis = analyze_results(results)
    print_analysis(analysis)

    out_path = Path(args.results_path).with_name(Path(args.results_path).stem + "_analysis.json")
    out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Analysis saved to %s", out_path)


if __name__ == "__main__":
    main()
