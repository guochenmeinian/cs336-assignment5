#!/usr/bin/env python3
"""
Script to analyze GSM8K evaluation results and provide detailed analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> List[Dict]:
    """Load evaluation results from JSONL file."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze evaluation results and categorize them."""
    
    # Categorize results
    correct_format_answer = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0]
    correct_format_wrong_answer = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]
    wrong_format = [r for r in results if r["format_reward"] == 0.0]
    
    total = len(results)
    
    analysis = {
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
    
    return analysis


def print_detailed_analysis(analysis: Dict) -> None:
    """Print detailed analysis with examples."""
    
    total = analysis["total_examples"]
    print("\n" + "="*60)
    print("GSM8K EVALUATION RESULTS ANALYSIS")
    print("="*60)
    print(f"Total examples: {total}")
    
    # Category 1: Correct format & answer
    cat1 = analysis["correct_format_and_answer"]
    print(f"\n1) Correct format & answer: {cat1['count']} ({cat1['percentage']:.2f}%)")
    
    # Category 2: Correct format, wrong answer
    cat2 = analysis["correct_format_wrong_answer"]
    print(f"\n2) Correct format, wrong answer: {cat2['count']} ({cat2['percentage']:.2f}%)")
    
    # Category 3: Wrong format
    cat3 = analysis["wrong_format"]
    print(f"\n3) Wrong format: {cat3['count']} ({cat3['percentage']:.2f}%)")
    
    # Analysis of wrong format cases
    if cat3['count'] >= 10:
        print(f"\n{'='*60}")
        print("ANALYSIS OF WRONG FORMAT CASES (≥10 examples)")
        print("="*60)
        print("This suggests the issue is likely with the base model's output formatting,")
        print("not the parser. The model often fails to produce the required")
        print("<think>...</think><answer>...</answer> format as specified by r1_zero.")
        
        print(f"\nExamples of wrong format responses:")
        for i, example in enumerate(cat3['examples'][:5]):
            print(f"\nExample {i+1}:")
            print(f"Question: {example['prompt'].split('User: ')[-1].split('\\nAssistant:')[0]}")
            print(f"Response: {example['response'][:200]}...")
    
    # Analysis of correct format but wrong answer cases
    if cat2['count'] >= 10:
        print(f"\n{'='*60}")
        print("ANALYSIS OF CORRECT FORMAT BUT WRONG ANSWER CASES (≥10 examples)")
        print("="*60)
        print("This suggests the issue is with math reasoning rather than parsing.")
        print("The model can follow the format but makes mathematical errors.")
        
        print(f"\nExamples of correct format but wrong answer:")
        for i, example in enumerate(cat2['examples'][:5]):
            print(f"\nExample {i+1}:")
            print(f"Question: {example['prompt'].split('User: ')[-1].split('\\nAssistant:')[0]}")
            print(f"Response: {example['response'][:200]}...")
            print(f"Ground truth: {example['ground_truth_answer']}")
    
    # Overall performance summary
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*60)
    format_accuracy = (cat1['count'] + cat2['count']) / total * 100
    answer_accuracy = cat1['count'] / total * 100
    
    print(f"Format accuracy: {format_accuracy:.2f}%")
    print(f"Answer accuracy: {answer_accuracy:.2f}%")
    print(f"Overall accuracy (both format and answer): {cat1['percentage']:.2f}%")


def main():
    """Main analysis function."""
    
    results_path = "gsm8k_baseline_results.jsonl"
    
    if not Path(results_path).exists():
        logger.error(f"Results file {results_path} not found. Please run evaluation first.")
        return
    
    # Load results
    results = load_results(results_path)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print detailed analysis
    print_detailed_analysis(analysis)


if __name__ == "__main__":
    main()
