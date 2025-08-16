# alignment/math_sft/run_sft.py
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, logging, random
from pathlib import Path
from typing import List, Dict

# 让 alignment/ 与 tests/ 作为可导入源
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .trainer import SFTTrainer, SFTConfig
from .logger  import log_generations

# Import shared utilities
from ..shared.config import (
    RESULTS_DIR, GSM8K_CONVERTED_TRAIN_PATH, GSM8K_CONVERTED_VALIDATION_PATH,
    WANDB_PROJECT, WANDB_NAME, WANDB_TAGS, WANDB_ENABLED,
    SFT_DEFAULT_CONFIG, SFT_LOG_EVERY, SFT_EVAL_BATCH_SIZE, SFT_MAX_NEW_TOKENS
)
from ..shared.math_data_utils import format_r1_zero_prompt
from ..shared.wandb_config import log_evaluation_batch

# SFT-specific configuration
from .config import SFTConfig

# reward 函数（优先用你的 shared 版本）
try:
    from ..shared.drgrpo_grader import r1_zero_reward_fn as reward_fn
except Exception:
    import re
    _BOX = re.compile(r"\\boxed\{([^}]*)\}")
    def _final(x:str):
        m = list(_BOX.finditer(x or ""))
        if m: return m[-1].group(1).strip()
        nums = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", x or "")
        return nums[-1].strip() if nums else None
    def reward_fn(resp:str, gt:str)->dict:
        p=_final(resp); g=_final(gt)
        ok = 1.0 if (p is not None and g is not None and p==g) else 0.0
        return {"reward": ok, "format_reward": 1.0, "answer_reward": ok}

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("math_sft")

def load_jsonl(path: Path)->List[Dict[str,str]]:
    rows=[]
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def convert_gsm8k_to_sft_format(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert GSM8K format (question/answer) to SFT format (prompt/response)."""
    converted_rows = []
    for row in rows:
        question = row["question"]
        answer = row["answer"]
        
        # Format prompt using r1_zero template
        prompt = format_r1_zero_prompt(question)
        
        converted_rows.append({
            "prompt": prompt,
            "response": answer
        })
    return converted_rows

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / "sft_qwen_math_15b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 配置 ===
    cfg = SFTConfig(
        # Use SFT-specific configuration with shared wandb settings
        wandb_enabled=WANDB_ENABLED,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME,
        wandb_tags=WANDB_TAGS,
        # Memory optimization
        use_flash_attention=True,
        torch_dtype="bfloat16",
        # Training parameters (can be customized)
        batch_size=2,
        grad_accum=8,
        max_steps=2000,
        lr=2e-5,
    )

    # === 数据 ===
    # 使用转换后的GSM8K数据（<think>格式）
    train_rows_raw = load_jsonl(GSM8K_CONVERTED_TRAIN_PATH)
    val_rows_raw = load_jsonl(GSM8K_CONVERTED_VALIDATION_PATH)
    
    # 转换为SFT格式（prompt/response）
    train_rows = convert_gsm8k_to_sft_format(train_rows_raw)
    val_rows = convert_gsm8k_to_sft_format(val_rows_raw)

    # === 训练器 ===
    trainer = SFTTrainer(cfg)

    # === 训练过程 + in-the-loop 生成日志 ===
    def on_step_log(step:int, scalars:Dict[str,float]):
        logger.info(f"[train] step={step} " + " ".join(f"{k}={v:.4f}" for k,v in scalars.items()))
        # 抽样验证集做 generation 日志
        sample_k = 64
        if len(val_rows) > sample_k:
            idx = random.sample(range(len(val_rows)), sample_k)
            sub = [val_rows[i] for i in idx]
        else:
            sub = val_rows
        prompts = [r["prompt"] for r in sub[:8]]  # 打印太多影响日志清晰，这里取 8 条生成
        gts     = [r["response"] for r in sub[:8]]
        lg = log_generations(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            prompts=prompts,
            ground_truths=gts,
            reward_fn=reward_fn,
            max_new_tokens=SFT_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            batch_size=SFT_EVAL_BATCH_SIZE,
            device=cfg.device,
            return_examples=True,  # Changed to True to get examples for wandb
        )
        s = lg["summary"]
        logger.info(
            f"[gen] step={step} "
            f"R={s['reward_mean']:.3f} fmt={s['format_reward_mean']:.3f} ans={s['answer_reward_mean']:.3f} "
            f"H={s['avg_token_entropy_mean']:.3f} "
            f"len={s['resp_len_mean']:.1f}/{s['resp_len_mean_correct']:.1f}/{s['resp_len_mean_incorrect']:.1f}"
        )
        
        # Log generation metrics and examples to wandb together
        eval_metrics = {
            "reward_mean": s["reward_mean"],
            "format_reward_mean": s["format_reward_mean"],
            "answer_reward_mean": s["answer_reward_mean"],
            "avg_token_entropy_mean": s["avg_token_entropy_mean"],
            "resp_len_mean": s["resp_len_mean"],
            "resp_len_mean_correct": s["resp_len_mean_correct"],
            "resp_len_mean_incorrect": s["resp_len_mean_incorrect"],
        }
        
        # Log everything together at the same step
        log_evaluation_batch(
            step=step, 
            eval_metrics=eval_metrics, 
            generation_examples=lg["examples"] if lg["examples"] else None,
            algorithm="sft"
        )

    trainer.train_one_like(train_rows, log_every=SFT_LOG_EVERY, on_step_log=on_step_log)

    # === 保存最终权重 ===
    trainer.save(str(out_dir / "policy"))
    logger.info(f"Saved to {out_dir / 'policy'}")
    return True

if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
