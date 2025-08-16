#!/usr/bin/env python3
"""
测试数学答案评估函数的正确性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "cs336"))

from alignment.shared.drgrpo_grader import r1_zero_reward_fn, grade, grade_answer_mathd, grade_answer_sympy

def test_grader():
    """测试评估函数的各种情况"""
    
    print("=== 测试数学答案评估函数 ===\n")
    
    # 测试用例
    test_cases = [
        {
            "name": "简单数字匹配",
            "response": "</think> <answer>42</answer>",
            "ground_truth": "42",
            "expected": True
        },
        {
            "name": "带boxed的答案",
            "response": "</think> <answer>\\boxed{42}</answer>",
            "ground_truth": "\\boxed{42}",
            "expected": True
        },
        {
            "name": "分数答案",
            "response": "</think> <answer>\\frac{1}{2}</answer>",
            "ground_truth": "\\frac{1}{2}",
            "expected": True
        },
        {
            "name": "小数答案",
            "response": "</think> <answer>3.14</answer>",
            "ground_truth": "3.14",
            "expected": True
        },
        {
            "name": "格式错误",
            "response": "答案是42",
            "ground_truth": "42",
            "expected": False
        },
        {
            "name": "答案错误",
            "response": "</think> <answer>43</answer>",
            "ground_truth": "42",
            "expected": False
        },
        {
            "name": "GSM8K风格答案",
            "response": "</think> <answer>Therefore, the answer is 42.</answer>",
            "ground_truth": "42",
            "expected": True
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"测试 {i+1}: {case['name']}")
        print(f"Response: {case['response']}")
        print(f"Ground Truth: {case['ground_truth']}")
        
        # 测试r1_zero_reward_fn
        result = r1_zero_reward_fn(case['response'], case['ground_truth'])
        print(f"r1_zero_reward_fn结果: {result}")
        
        # 测试grade函数
        if "</think> <answer>" in case['response'] and "</answer>" in case['response']:
            model_answer = case['response'].split("<answer>")[-1].replace("</answer>", "")
            grade_result = grade(model_answer, case['ground_truth'])
            print(f"grade函数结果: {grade_result}")
            
            # 测试具体的评估函数
            mathd_result = grade_answer_mathd(model_answer, case['ground_truth'])
            sympy_result = grade_answer_sympy(model_answer, case['ground_truth'])
            print(f"mathd评估: {mathd_result}, sympy评估: {sympy_result}")
        
        print(f"期望结果: {case['expected']}")
        print("-" * 50)
    
    # 测试一些实际的数学问题
    print("\n=== 测试实际数学问题 ===")
    
    math_cases = [
        ("2 + 2", "4"),
        ("\\frac{1}{2} + \\frac{1}{2}", "1"),
        ("\\sqrt{16}", "4"),
        ("3^2", "9"),
        ("\\frac{3}{4} \\times \\frac{4}{3}", "1"),
    ]
    
    for question, answer in math_cases:
        response = f"</think> <answer>{answer}</answer>"
        result = r1_zero_reward_fn(response, answer)
        print(f"问题: {question} = {answer}")
        print(f"评估结果: {result}")
        print("-" * 30)

if __name__ == "__main__":
    test_grader()
