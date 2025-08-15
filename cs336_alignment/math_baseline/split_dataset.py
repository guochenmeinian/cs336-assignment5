#!/usr/bin/env python3
"""
Script to split GSM8K dataset into train/validation/test sets.
"""

import json
import random
from pathlib import Path

def split_gsm8k_dataset():
    """Split GSM8K dataset into train/validation/test sets."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load train data
    train_data = []
    with open("../../data/gsm8k/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))
    
    # Load test data
    test_data = []
    with open("../../data/gsm8k/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"Original train examples: {len(train_data)}")
    print(f"Original test examples: {len(test_data)}")
    
    # Shuffle train data
    random.shuffle(train_data)
    
    # We want test to have exactly 1000 examples
    # If original test has more than 1000, we'll take first 1000
    # If original test has less than 1000, we'll supplement from train
    target_test_size = 1000
    
    if len(test_data) >= target_test_size:
        # Take first 1000 from original test
        final_test_data = test_data[:target_test_size]
        # Add remaining test examples back to train
        train_data.extend(test_data[target_test_size:])
    else:
        # Need to supplement test with examples from train
        needed_from_train = target_test_size - len(test_data)
        final_test_data = test_data + train_data[:needed_from_train]
        train_data = train_data[needed_from_train:]
    
    # Take 1000 from remaining train for validation
    validation_size = 1000
    validation_data = train_data[:validation_size]
    new_train_data = train_data[validation_size:]
    
    print(f"New train examples: {len(new_train_data)}")
    print(f"New validation examples: {len(validation_data)}")
    print(f"New test examples: {len(final_test_data)}")
    
    # Create output directory
    output_dir = Path("../../data/gsm8k_split")
    output_dir.mkdir(exist_ok=True)
    
    # Save split datasets
    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for example in new_train_data:
            f.write(json.dumps(example) + "\n")
    
    with open(output_dir / "validation.jsonl", "w", encoding="utf-8") as f:
        for example in validation_data:
            f.write(json.dumps(example) + "\n")
    
    with open(output_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for example in final_test_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"Datasets saved to {output_dir}")

if __name__ == "__main__":
    split_gsm8k_dataset()
