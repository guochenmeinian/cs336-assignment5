# Math Baseline Evaluation

This module provides evaluation and analysis tools for math baseline models on the GSM8K dataset.

## Structure

```
math_baseline/
├── __init__.py          # Module exports
├── utils.py             # Core utility functions
├── eval.py          # Evaluation logic
├── analyze.py           # Results analysis
└── README.md           # This file
```

## Usage

### 1. Run Eval

```bash
python -m alignment.math_baseline.eval
```

This will:
- Load the Qwen 2.5 Math 1.5B model
- Evaluate it on GSM8K validation set
- Analyze the results
- Save results to `results/gsm8k_baseline_results.jsonl`

### 2. Analysis

```bash
python -m alignment.math_baseline.analyze
```

**Performance Metrics**

The analysis categorizes results into:
1. **Correct format & answer**: Both format and mathematical reasoning are correct
2. **Correct format, wrong answer**: Format is correct but math is wrong
3. **Wrong format**: Model failed to follow the required format

**Analysis output**: 
   - Format accuracy: 20.10%
   - Answer accuracy: 0.00%
   - Overall accuracy (both format and answer): 0.00%
