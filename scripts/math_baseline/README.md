# Math Baseline: a/b/c Mapping

- (a) Baseline zero-shot evaluation
  - Code: `scripts/math_baseline/evaluate_math_qwen.py`
  - Reused utils: `scripts/math_baseline/utils.py`
    - `load_gsm8k_data(path)` → 读取 `/data/a5-alignment/gsm8k/test.jsonl`
    - `format_r1_zero_prompt(question)` / `load_r1_zero_prompt()` → r1_zero 提示格式化
    - `evaluate_vllm(vllm_model, reward_fn, prompts, ground_truths, eval_sampling_params, output_path)` → 统一生成/打分/落盘
  - Output (by default):
    - Per-example: `results/math_baseline/qwen_gsm8k_baseline.jsonl`
    - Metrics: `results/math_baseline/qwen_gsm8k_baseline_metrics.json`

- (b) 类别计数 + 示例 + 简要评论
  - Code: `scripts/math_baseline/analyze_results.py`
  - Input: 上一步 `.jsonl`
  - Output: `*_analysis.json`，并在控制台打印三类：
    1) format=1 & answer=1（正确格式且答案正确）
    2) format=1 & answer=0（正确格式但答案错误）
    3) format=0（格式不符合，答案计为0）
  - 用于撰写：数量统计、≥10例时的原因判断与示例摘录

- (c) 1–2 句整体指标结论
  - Source: `results/math_baseline/qwen_gsm8k_baseline_metrics.json`
  - 常用字段：
    - `overall_accuracy`（整体正确率）
    - `format_accuracy`（格式遵循率）
    - 计数：`correct_format_and_answer` / `correct_format_wrong_answer` / `wrong_format`
