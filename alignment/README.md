# Alignment Methods

This directory contains implementations of various alignment methods for the CS336 assignment.

## Architecture

The code is organized by **method** rather than by **function**, which makes it easier to:
- Understand the complete workflow of each method
- Compare different methods
- Maintain and extend individual implementations

```
alignment/
├── shared/                    # Shared utilities across methods
│   ├── data_utils.py         # Data loading and formatting
│   └── evaluation_utils.py   # Common evaluation functions
├── math_baseline/            # Baseline model evaluation
│   ├── evaluate.py           # Evaluation logic
│   ├── analyze.py            # Results analysis
│   └── run_baseline.py       # Main execution script
├── math_sft/                 # Supervised Fine-Tuning (future)
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation logic
│   └── analyze.py            # Results analysis
├── math_grpo/                # GRPO training (future)
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation logic
│   └── analyze.py            # Results analysis
└── prompts/                  # Prompt templates
    └── r1_zero.prompt        # R1-zero prompt template
```

## Current Status

### ✅ Implemented
- **math_baseline**: Complete evaluation and analysis pipeline
- **shared**: Common utilities for data and evaluation

### 🔄 Planned
- **math_sft**: Supervised fine-tuning implementation
- **math_grpo**: GRPO training implementation

## Usage

### 1. Math Baseline Evaluation

```bash
cd cs336
python -m alignment.math_baseline.run_baseline
```

### 2. Using Shared Utilities

```python
from alignment.shared.data_utils import load_gsm8k_data, format_r1_zero_prompt
from alignment.shared.evaluation_utils import evaluate_vllm, compute_metrics

# Load data
data = load_gsm8k_data("data/gsm8k/validation.jsonl")

# Format prompts
prompts = [format_r1_zero_prompt(ex["question"]) for ex in data]

# Evaluate (assuming you have a model and reward function)
results = evaluate_vllm(model, reward_fn, prompts, ground_truths, sampling_params)

# Compute metrics
metrics = compute_metrics(results)
```

## Adding New Methods

To add a new alignment method (e.g., `math_ppo`):

1. Create a new directory: `alignment/math_ppo/`
2. Implement the required modules:
   - `train.py` (if training is involved)
   - `evaluate.py` (evaluation logic)
   - `analyze.py` (results analysis)
   - `run_ppo.py` (main execution script)
3. Use shared utilities from `alignment/shared/` to avoid code duplication
4. Add the method to this README

## Dependencies

- `vllm`: Model loading and inference
- `torch`: Deep learning framework
- `transformers`: Hugging Face transformers library
- `numpy`: Numerical computing
- `pandas`: Data manipulation (optional)

## Data Requirements

All methods expect:
- GSM8K dataset in `data/gsm8k/`
- Prompt templates in `alignment/prompts/`
- Model cache directory (configurable, default: `/root/autodl-tmp/models`)

## Output Structure

Results are saved to:
- `results/gsm8k_{method}_results.jsonl`: Raw evaluation results
- Console output: Progress logs and analysis summaries

## Contributing

When adding new features:
1. Follow the existing code structure
2. Use shared utilities when possible
3. Add appropriate logging and error handling
4. Update documentation
5. Test thoroughly before committing
