

python -m alignment.math_sft.run_sft --dataset_size 3400 --epochs 1

python -m alignment.shared.math_evaluate models/sft_qwen_math_15b_full_lr2e-05_bs2_steps500/policy

python -m alignment.shared.math_analyze evaluations/sft_qwen_math_15b_full_lr2e-05_bs2_steps500_evaluation.jsonl

