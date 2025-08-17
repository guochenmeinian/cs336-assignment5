#!/usr/bin/env python3
"""
SFT实验脚本 - 支持不同数据集大小和参数配置
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.math_sft.config import SFTConfig
from alignment.math_sft.run_sft import main as run_sft_experiment
from alignment.shared.config import MATH_TRAIN_PATH, MATH_VALIDATION_PATH
import json
import random

def load_jsonl(path: Path):
    """加载JSONL文件"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def create_subset_dataset(full_data, subset_size):
    """创建指定大小的数据集子集"""
    if subset_size >= len(full_data):
        return full_data
    
    # 随机采样
    random.seed(42)  # 固定随机种子
    indices = random.sample(range(len(full_data)), subset_size)
    return [full_data[i] for i in indices]

def run_experiment(experiment_name, dataset_size, lr, batch_size, max_steps=2000):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"数据集大小: {dataset_size}")
    print(f"学习率: {lr}")
    print(f"批量大小: {batch_size}")
    print(f"最大步数: {max_steps}")
    print(f"{'='*60}")
    
    # 创建实验配置
    config = SFTConfig(
        lr=lr,
        batch_size=batch_size,
        grad_accum=8,
        max_steps=max_steps,
        max_grad_norm=1.0,
        bf16=True,
        amp=False,
        use_flash_attention=True,
        torch_dtype="bfloat16",
        seed=42,
        device="cuda:0",
        wandb_enabled=True,
        wandb_name=f"sft_{experiment_name}",
        wandb_tags=["sft", "qwen", "math", "15b", f"size_{dataset_size}"],
    )
    
    # 创建临时数据集文件
    temp_train_path = PROJECT_ROOT / "data" / "math" / f"train_temp_{dataset_size}.jsonl"
    temp_val_path = PROJECT_ROOT / "data" / "math" / f"validation_temp_{dataset_size}.jsonl"
    
    # 加载完整数据
    full_train_data = load_jsonl(MATH_TRAIN_PATH)
    full_val_data = load_jsonl(MATH_VALIDATION_PATH)
    
    # 创建子集
    subset_train = create_subset_dataset(full_train_data, dataset_size)
    subset_val = create_subset_dataset(full_val_data, min(1000, dataset_size))
    
    # 保存临时文件
    with open(temp_train_path, "w") as f:
        for item in subset_train:
            f.write(json.dumps(item) + "\n")
    
    with open(temp_val_path, "w") as f:
        for item in subset_val:
            f.write(json.dumps(item) + "\n")
    
    try:
        # 运行实验
        # 注意：这里需要修改run_sft.py来支持自定义数据路径
        print(f"实验配置已准备，请手动运行:")
        print(f"python -m alignment.math_sft.run_sft")
        print(f"临时数据文件:")
        print(f"  训练: {temp_train_path}")
        print(f"  验证: {temp_val_path}")
        
    except Exception as e:
        print(f"实验运行失败: {e}")
    finally:
        # 清理临时文件
        if temp_train_path.exists():
            temp_train_path.unlink()
        if temp_val_path.exists():
            temp_val_path.unlink()

def main():
    """运行所有实验"""
    
    # 实验配置
    experiments = [
        # (实验名称, 数据集大小, 学习率, 批量大小, 最大步数)
        ("size_128", 128, 2e-5, 2, 1000),
        ("size_256", 256, 2e-5, 2, 1500),
        ("size_512", 512, 2e-5, 2, 2000),
        ("size_1024", 1024, 2e-5, 2, 2000),
        ("size_full", 6792, 2e-5, 2, 2000),
        
        # 不同学习率实验
        ("lr_1e5", 6792, 1e-5, 2, 2000),
        ("lr_5e5", 6792, 5e-5, 2, 2000),
        ("lr_1e4", 6792, 1e-4, 2, 2000),
        
        # 不同批量大小实验
        ("bs_1", 6792, 2e-5, 1, 2000),
        ("bs_4", 6792, 2e-5, 4, 2000),
        ("bs_8", 6792, 2e-5, 8, 2000),
    ]
    
    print("SFT实验配置")
    print("="*60)
    
    for i, (name, size, lr, bs, steps) in enumerate(experiments):
        print(f"{i+1:2d}. {name:15s} | 大小: {size:4d} | LR: {lr:.1e} | BS: {bs} | 步数: {steps}")
    
    print("\n选择要运行的实验:")
    print("1. 运行所有实验")
    print("2. 运行数据集大小实验 (128, 256, 512, 1024, full)")
    print("3. 运行学习率实验 (1e-5, 2e-5, 5e-5, 1e-4)")
    print("4. 运行批量大小实验 (1, 2, 4, 8)")
    print("5. 运行单个实验")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == "1":
        # 运行所有实验
        for name, size, lr, bs, steps in experiments:
            run_experiment(name, size, lr, bs, steps)
    
    elif choice == "2":
        # 数据集大小实验
        size_experiments = [exp for exp in experiments if exp[0].startswith("size_")]
        for name, size, lr, bs, steps in size_experiments:
            run_experiment(name, size, lr, bs, steps)
    
    elif choice == "3":
        # 学习率实验
        lr_experiments = [exp for exp in experiments if exp[0].startswith("lr_")]
        for name, size, lr, bs, steps in lr_experiments:
            run_experiment(name, size, lr, bs, steps)
    
    elif choice == "4":
        # 批量大小实验
        bs_experiments = [exp for exp in experiments if exp[0].startswith("bs_")]
        for name, size, lr, bs, steps in bs_experiments:
            run_experiment(name, size, lr, bs, steps)
    
    elif choice == "5":
        # 单个实验
        print("\n可用实验:")
        for i, (name, size, lr, bs, steps) in enumerate(experiments):
            print(f"{i+1:2d}. {name}")
        
        exp_idx = int(input("请输入实验编号: ")) - 1
        if 0 <= exp_idx < len(experiments):
            name, size, lr, bs, steps = experiments[exp_idx]
            run_experiment(name, size, lr, bs, steps)
        else:
            print("无效的实验编号")
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
