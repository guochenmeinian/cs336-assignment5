# GSM8K Math Baseline Evaluation

This directory contains scripts to evaluate Qwen 2.5 Math 1.5B zero-shot performance on GSM8K dataset.

## Files

- `split_dataset.py`: Splits the original GSM8K dataset into train/validation/test sets
- `evaluate_gsm8k.py`: Main evaluation script that runs the model on validation data
- `analyze_results.py`: Analyzes evaluation results and provides detailed analysis
- `utils.py`: Utility functions for data loading, prompt formatting, and evaluation
- `README.md`: This file

## Usage

### 1. Split the Dataset

First, split the GSM8K dataset into train/validation/test sets:

```bash
python split_dataset.py
```

This will create a `gsm8k_split` directory with:
- `train.jsonl`: 6792 examples
- `validation.jsonl`: 1000 examples  
- `test.jsonl`: 1000 examples

### 2. Run Evaluation

Run the evaluation on the validation set:

```bash
python evaluate_gsm8k.py
```

This will:
- Load Qwen 2.5 Math 1.5B model
- Format prompts using r1_zero template
- Generate responses for all validation examples
- Compute format and answer rewards
- Save results to `gsm8k_baseline_results.jsonl`

### 3. Analyze Results

Analyze the evaluation results:

```bash
python analyze_results.py
```

This provides detailed analysis including:
- Categorization of results into three groups
- Examples of each category
- Analysis of whether issues are with model output or parser
- Overall performance metrics

## Dataset Structure

The GSM8K dataset contains math word problems with:
- `question`: The math problem text
- `answer`: The solution with step-by-step reasoning and final answer after `####`

## Prompt Format

The r1_zero prompt template requires responses in the format:
```
<think>reasoning process here</think><answer>answer here</answer>
```

## Reward Function

The evaluation uses `r1_zero_reward_fn` which provides:
- `format_reward`: 1.0 if response follows required format, 0.0 otherwise
- `answer_reward`: 1.0 if mathematical answer is correct, 0.0 otherwise
- `reward`: Overall reward (1.0 only if both format and answer are correct)

## Requirements

- vllm
- transformers
- torch
- Access to Qwen 2.5 Math 1.5B model
