#!/usr/bin/env python3
"""
通用模型分析脚本 - 可以分析任何评估结果
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from .config import EVALUATIONS_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_evaluation_results(results_path: str) -> Dict:
    """分析评估结果"""
    
    # 加载结果
    results = []
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        logger.error("没有找到评估结果")
        return {}
    
    # 统计信息
    total_samples = len(results)
    format_correct = sum(1 for r in results if r.get("format_reward", 0) >= 0.5)
    answer_correct = sum(1 for r in results if r.get("answer_reward", 0) >= 0.5)
    overall_correct = sum(1 for r in results if r.get("reward", 0) >= 0.5)
    
    # 计算准确率
    format_accuracy = format_correct / total_samples if total_samples > 0 else 0.0
    answer_accuracy = answer_correct / total_samples if total_samples > 0 else 0.0
    overall_accuracy = overall_correct / total_samples if total_samples > 0 else 0.0
    
    # 计算平均分数
    format_scores = [r.get("format_reward", 0) for r in results]
    answer_scores = [r.get("answer_reward", 0) for r in results]
    overall_scores = [r.get("reward", 0) for r in results]
    
    avg_format_score = sum(format_scores) / len(format_scores) if format_scores else 0.0
    avg_answer_score = sum(answer_scores) / len(answer_scores) if answer_scores else 0.0
    avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    
    # 分析错误案例
    format_errors = [r for r in results if r.get("format_reward", 0) < 0.5]
    answer_errors = [r for r in results if r.get("answer_reward", 0) < 0.5]
    
    analysis = {
        "total_samples": total_samples,
        "format_correct": format_correct,
        "answer_correct": answer_correct,
        "overall_correct": overall_correct,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,
        "overall_accuracy": overall_accuracy,
        "avg_format_score": avg_format_score,
        "avg_answer_score": avg_answer_score,
        "avg_overall_score": avg_overall_score,
        "format_error_count": len(format_errors),
        "answer_error_count": len(answer_errors),
        "format_error_rate": len(format_errors) / total_samples if total_samples > 0 else 0.0,
        "answer_error_rate": len(answer_errors) / total_samples if total_samples > 0 else 0.0,
    }
    
    return analysis


def print_analysis(analysis: Dict, model_name: str = "Unknown Model") -> None:
    """打印分析结果"""
    
    print("="*80)
    print(f"模型评估结果分析: {model_name}")
    print("="*80)
    
    print(f"总样本数: {analysis['total_samples']}")
    print()
    
    print("准确率统计:")
    print(f"  格式正确: {analysis['format_correct']}/{analysis['total_samples']} ({analysis['format_accuracy']:.1%})")
    print(f"  答案正确: {analysis['answer_correct']}/{analysis['total_samples']} ({analysis['answer_accuracy']:.1%})")
    print(f"  完全正确: {analysis['overall_correct']}/{analysis['total_samples']} ({analysis['overall_accuracy']:.1%})")
    print()
    
    print("平均分数:")
    print(f"  格式分数: {analysis['avg_format_score']:.3f}")
    print(f"  答案分数: {analysis['avg_answer_score']:.3f}")
    print(f"  总体分数: {analysis['avg_overall_score']:.3f}")
    print()
    
    print("错误分析:")
    print(f"  格式错误: {analysis['format_error_count']} ({analysis['format_error_rate']:.1%})")
    print(f"  答案错误: {analysis['answer_error_count']} ({analysis['answer_error_rate']:.1%})")
    print("="*80)


def main():
    """主函数 - 用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分析模型评估结果")
    parser.add_argument("results_path", help="评估结果文件路径")
    parser.add_argument("--model-name", "-n", help="模型名称（用于显示）")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.results_path).exists():
        logger.error(f"结果文件不存在: {args.results_path}")
        exit(1)
    
    # 分析结果
    analysis = analyze_evaluation_results(args.results_path)
    
    if analysis:
        # 如果没有指定模型名称，从文件名推断
        if not args.model_name:
            args.model_name = Path(args.results_path).stem.replace("_evaluation", "")
        
        # 打印分析结果
        print_analysis(analysis, args.model_name)
        
        # 保存分析结果
        output_path = Path(args.results_path).parent / f"{Path(args.results_path).stem}_analysis.json"
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"分析结果保存到: {output_path}")
    else:
        logger.error("分析失败")
        exit(1)


if __name__ == "__main__":
    main()
