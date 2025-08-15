# Math Baseline Evaluation

This module provides evaluation and analysis tools for math baseline models on the GSM8K dataset.

## Structure

```
math_baseline/
├── __init__.py          # Module exports
├── utils.py             # Core utility functions
├── evaluate.py          # Evaluation logic
├── analyze.py           # Results analysis
├── run_baseline.py      # Main execution script
└── README.md           # This file
```

## Usage

### 1. Run Complete Pipeline

```bash
cd cs336
python -m alignment.math_baseline.run_baseline
```

This will:
- Load the Qwen 2.5 Math 1.5B model
- Evaluate it on GSM8K validation set
- Analyze the results
- Save results to `results/gsm8k_baseline_results.jsonl`

### 2. Run Individual Components

#### Evaluation Only
```python
from alignment.math_baseline.evaluate import evaluate_baseline

evaluate_baseline(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    validation_data_path="data/gsm8k/validation.jsonl",
    output_path="results/gsm8k_baseline_results.jsonl",
    model_cache_dir="/root/autodl-tmp/models"
)
```

#### Analysis Only
```python
from alignment.math_baseline.analyze import load_results, analyze_results, print_analysis

results = load_results("results/gsm8k_baseline_results.jsonl")
analysis = analyze_results(results)
print_analysis(analysis)
```

### 3. Custom Configuration

You can modify the default parameters in `evaluate.py`:
- Model name and parameters
- Data paths
- Output paths
- Sampling parameters

## Dependencies

- `vllm`: For model loading and inference
- `torch`: For tensor operations
- `transformers`: For tokenizer operations

## Data Requirements

The script expects:
- `data/gsm8k/validation.jsonl`: GSM8K validation dataset
- `alignment/prompts/r1_zero.prompt`: R1-zero prompt template

## Output

The evaluation produces:
1. **Results file**: `results/gsm8k_baseline_results.jsonl`
   - Contains all examples with prompts, responses, and rewards
   
2. **Console output**: 
   - Evaluation progress
   - Detailed analysis with examples
   - Performance metrics

## Performance Metrics

The analysis categorizes results into:
1. **Correct format & answer**: Both format and mathematical reasoning are correct
2. **Correct format, wrong answer**: Format is correct but math is wrong
3. **Wrong format**: Model failed to follow the required format

## Troubleshooting

### Common Issues

1. **Model loading fails**: Check if the model is available and GPU memory is sufficient
2. **Import errors**: Ensure you're running from the `cs336` directory
3. **Data not found**: Verify data paths are correct relative to the `cs336` directory

### Debug Mode

Enable debug logging by modifying the logging level in `run_baseline.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```
