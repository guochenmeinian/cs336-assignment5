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
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_results(results_path: str) -> List[Dict]:
    results: List[Dict] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    logger.info("Loaded %d results from %s", len(results), results_path)
    return results


def analyze_results(results: List[Dict]) -> Dict:
    correct_format_answer: List[Dict] = []
    correct_format_wrong_answer: List[Dict] = []
    wrong_format: List[Dict] = []

    for r in results:
        fr = r.get("format_reward", 0.0)
        ar = r.get("answer_reward", 0.0)
        if fr == 1.0 and ar == 1.0:
            correct_format_answer.append(r)
        elif fr == 1.0 and ar == 0.0:
            correct_format_wrong_answer.append(r)
        elif fr == 0.0:
            wrong_format.append(r)

    total = len(results)
    return {
        "total_examples": total,
        "correct_format_and_answer": {
            "count": len(correct_format_answer),
            "percentage": (len(correct_format_answer) / total * 100) if total else 0.0,
            "examples": correct_format_answer[:10],
        },
        "correct_format_wrong_answer": {
            "count": len(correct_format_wrong_answer),
            "percentage": (len(correct_format_wrong_answer) / total * 100) if total else 0.0,
            "examples": correct_format_wrong_answer[:10],
        },
        "wrong_format": {
            "count": len(wrong_format),
            "percentage": (len(wrong_format) / total * 100) if total else 0.0,
            "examples": wrong_format[:10],
        },
    }


def print_analysis(analysis: Dict) -> None:
    total = analysis["total_examples"]
    print("\n==== GSM8K EVALUATION RESULTS ANALYSIS ====")
    print(f"Total examples: {total}")
    print(
        "1) Correct format & answer: {count} ({pct:.2f}%)".format(
            count=analysis["correct_format_and_answer"]["count"],
            pct=analysis["correct_format_and_answer"]["percentage"],
        )
    )
    print(
        "2) Correct format, wrong answer: {count} ({pct:.2f}%)".format(
            count=analysis["correct_format_wrong_answer"]["count"],
            pct=analysis["correct_format_wrong_answer"]["percentage"],
        )
    )
    print(
        "3) Wrong format: {count} ({pct:.2f}%)".format(
            count=analysis["wrong_format"]["count"],
            pct=analysis["wrong_format"]["percentage"],
        )
    )

    # Brief commentary templates
    if analysis["wrong_format"]["count"] >= 10:
        print("\nWrong format (≥10 cases): Likely an output-formatting issue from the base model, not the parser.")
        print("Model often fails to produce <think>...</think><answer>...</answer> as required by r1_zero.")
    if analysis["correct_format_wrong_answer"]["count"] >= 10:
        print("\nCorrect format but wrong answer (≥10 cases): Likely math reasoning errors rather than parsing issues.")


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
