#!/usr/bin/env python3
"""
Test script to verify utility functions work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import load_gsm8k_data, load_r1_zero_prompt, format_r1_zero_prompt

# Import reward function from drgrpo_grader
sys.path.append(str(Path(__file__).parent.parent))
from drgrpo_grader import r1_zero_reward_fn


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    try:
        data = load_gsm8k_data("../../data/gsm8k_split/validation.jsonl")
        print(f"✓ Successfully loaded {len(data)} examples")
        
        # Check first example structure
        first_example = data[0]
        assert "question" in first_example
        assert "answer" in first_example
        print("✓ Example structure is correct")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_prompt_loading():
    """Test prompt template loading."""
    print("\nTesting prompt template loading...")
    
    try:
        prompt_template = load_r1_zero_prompt()
        print("✓ Successfully loaded r1_zero prompt template")
        print(f"Template length: {len(prompt_template)} characters")
        print(f"Template preview: {prompt_template[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Prompt loading failed: {e}")
        return False


def test_prompt_formatting():
    """Test prompt formatting functionality."""
    print("\nTesting prompt formatting...")
    
    try:
        question = "What is 2 + 2?"
        formatted_prompt = format_r1_zero_prompt(question)
        print("✓ Successfully formatted prompt")
        print(f"Formatted prompt length: {len(formatted_prompt)} characters")
        print(f"Formatted prompt preview: {formatted_prompt[:200]}...")
        return True
    except Exception as e:
        print(f"✗ Prompt formatting failed: {e}")
        return False


def test_reward_function():
    """Test reward function functionality."""
    print("\nTesting reward function...")
    
    try:
        # Test correct format and answer
        correct_response = "<think>I need to add 2 and 2. 2 + 2 = 4.</think><answer>4</answer>"
        rewards = r1_zero_reward_fn(correct_response, "4")
        print(f"✓ Correct format & answer rewards: {rewards}")
        
        # Test correct format but wrong answer
        wrong_answer_response = "<think>I need to add 2 and 2. 2 + 2 = 5.</think><answer>5</answer>"
        rewards = r1_zero_reward_fn(wrong_answer_response, "4")
        print(f"✓ Correct format, wrong answer rewards: {rewards}")
        
        # Test wrong format
        wrong_format_response = "The answer is 4"
        rewards = r1_zero_reward_fn(wrong_format_response, "4")
        print(f"✓ Wrong format rewards: {rewards}")
        
        return True
    except Exception as e:
        print(f"✗ Reward function failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running utility function tests...\n")
    
    tests = [
        test_data_loading,
        test_prompt_loading,
        test_prompt_formatting,
        test_reward_function,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Ready to run evaluation.")
    else:
        print("✗ Some tests failed. Please fix issues before running evaluation.")
    
    return passed == total


if __name__ == "__main__":
    main()
