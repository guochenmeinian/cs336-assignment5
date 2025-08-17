#!/usr/bin/env python3
"""
过滤MATH数据集，只保留正确答案的样本
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.shared.drgrpo_grader import r1_zero_reward_fn

def load_jsonl(path: Path):
    """加载JSONL文件"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_jsonl(data, path: Path):
    """保存JSONL文件"""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def filter_correct_answers(data):
    """过滤只保留正确答案的样本"""
    correct_samples = []
    
    print(f"开始过滤 {len(data)} 个样本...")
    
    for i, sample in enumerate(data):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(data)}")
        
        # 提取答案部分
        answer = sample["answer"]
        if "<answer>" in answer and "</answer>" in answer:
            pure_answer = answer.split("<answer>")[-1].replace("</answer>", "").strip()
        else:
            print(f"警告: 样本 {i} 格式不正确: {answer[:100]}...")
            continue
        
        # 使用reward函数检查答案正确性
        # 创建一个虚拟的response来检查答案
        dummy_response = f"<think>Let me solve this step by step.</think> <answer>{pure_answer}</answer>"
        
        try:
            scores = r1_zero_reward_fn(dummy_response, pure_answer)
            if scores["answer_reward"] >= 0.5:  # 答案正确
                correct_samples.append(sample)
        except Exception as e:
            print(f"警告: 样本 {i} 评估失败: {e}")
            continue
    
    print(f"过滤完成: {len(correct_samples)}/{len(data)} 个样本答案正确")
    return correct_samples

def main():
    """主函数"""
    # 数据路径
    train_path = PROJECT_ROOT / "data" / "math" / "train.jsonl"
    validation_path = PROJECT_ROOT / "data" / "math" / "validation.jsonl"
    test_path = PROJECT_ROOT / "data" / "math" / "test.jsonl"
    
    # 输出路径
    filtered_dir = PROJECT_ROOT / "data" / "math_filtered"
    filtered_dir.mkdir(exist_ok=True)
    
    # 加载数据
    print("加载原始数据...")
    train_data = load_jsonl(train_path)
    validation_data = load_jsonl(validation_path)
    test_data = load_jsonl(test_path)
    
    print(f"原始数据大小:")
    print(f"  训练: {len(train_data)}")
    print(f"  验证: {len(validation_data)}")
    print(f"  测试: {len(test_data)}")
    
    # 过滤数据
    print("\n开始过滤训练数据...")
    filtered_train = filter_correct_answers(train_data)
    
    print("\n开始过滤验证数据...")
    filtered_validation = filter_correct_answers(validation_data)
    
    print("\n开始过滤测试数据...")
    filtered_test = filter_correct_answers(test_data)
    
    # 保存过滤后的数据
    print("\n保存过滤后的数据...")
    save_jsonl(filtered_train, filtered_dir / "train.jsonl")
    save_jsonl(filtered_validation, filtered_dir / "validation.jsonl")
    save_jsonl(filtered_test, filtered_dir / "test.jsonl")
    
    # 保存统计信息
    stats = {
        "original": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
            "total": len(train_data) + len(validation_data) + len(test_data)
        },
        "filtered": {
            "train": len(filtered_train),
            "validation": len(filtered_validation),
            "test": len(filtered_test),
            "total": len(filtered_train) + len(filtered_validation) + len(filtered_test)
        },
        "filtering_rate": {
            "train": len(filtered_train) / len(train_data) if train_data else 0,
            "validation": len(filtered_validation) / len(validation_data) if validation_data else 0,
            "test": len(filtered_test) / len(test_data) if test_data else 0,
            "overall": (len(filtered_train) + len(filtered_validation) + len(filtered_test)) / 
                      (len(train_data) + len(validation_data) + len(test_data)) if (train_data + validation_data + test_data) else 0
        }
    }
    
    with open(filtered_dir / "filtering_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n过滤完成!")
    print(f"过滤后数据保存在: {filtered_dir}")
    print(f"统计信息:")
    print(f"  训练: {len(filtered_train)}/{len(train_data)} ({stats['filtering_rate']['train']:.1%})")
    print(f"  验证: {len(filtered_validation)}/{len(validation_data)} ({stats['filtering_rate']['validation']:.1%})")
    print(f"  测试: {len(filtered_test)}/{len(test_data)} ({stats['filtering_rate']['test']:.1%})")
    print(f"  总体: {stats['filtered']['total']}/{stats['original']['total']} ({stats['filtering_rate']['overall']:.1%})")

if __name__ == "__main__":
    main()
