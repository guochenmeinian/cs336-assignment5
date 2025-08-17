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

from .trainer import SFTTrainer
from .logger  import log_generations

# Import shared utilities
from ..shared.config import (
    EVALUATIONS_DIR, MODEL_DIR, MATH_TRAIN_PATH, MATH_VALIDATION_PATH,
    SFT_LOG_EVERY, SFT_EVAL_BATCH_SIZE, SFT_MAX_NEW_TOKENS
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

def convert_math_to_sft_format(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert MATH format (question/answer) to SFT format (prompt/response)."""
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

def main(train_path=None, validation_path=None, config_overrides=None, dataset_size=None):
    """运行SFT训练
    
    Args:
        train_path: 训练数据路径，默认使用MATH_TRAIN_PATH
        validation_path: 验证数据路径，默认使用MATH_VALIDATION_PATH
        config_overrides: 配置覆盖字典
        dataset_size: 训练数据集大小，None表示使用完整数据集
    """
    # 使用默认路径或自定义路径
    train_data_path = train_path or MATH_TRAIN_PATH
    val_data_path = validation_path or MATH_VALIDATION_PATH
    
    # === 配置 ===
    cfg = SFTConfig(
        # 训练参数
        lr=2e-5,
        batch_size=2,
        grad_accum=8,
        max_steps=2000,
        max_grad_norm=1.0,
        bf16=True,
        amp=False,
        use_flash_attention=True,
        torch_dtype="bfloat16",
        seed=42,
        device="cuda:0",
        # 实验特定的wandb配置
        wandb_enabled=True,
        wandb_name="sft_qwen_math_15b",
        wandb_tags=["sft", "qwen", "math", "15b"],
    )
    
    # 应用配置覆盖
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    
    # === 数据 ===
    # 使用指定路径或默认MATH数据集（<think>格式）
    train_rows_raw = load_jsonl(train_data_path)
    val_rows_raw = load_jsonl(val_data_path)
    
    # 如果指定了数据集大小，创建子集
    if dataset_size and dataset_size < len(train_rows_raw):
        import random
        random.seed(42)  # 固定随机种子
        indices = random.sample(range(len(train_rows_raw)), dataset_size)
        train_rows_raw = [train_rows_raw[i] for i in indices]
        print(f"使用数据集子集: {dataset_size} 个样本")
    
    # 转换为SFT格式（prompt/response）
    train_rows = convert_math_to_sft_format(train_rows_raw)
    val_rows = convert_math_to_sft_format(val_rows_raw)
    
    # 根据数据集大小调整max_steps
    if dataset_size:
        # 每个样本最多用一次，避免过拟合
        samples_per_step = cfg.batch_size * cfg.grad_accum
        cfg.max_steps = max(1, dataset_size // samples_per_step)
        print(f"根据数据集大小调整max_steps: {cfg.max_steps} (每个样本用1次)")
    
    print(f"训练数据: {len(train_rows)} 条")
    print(f"验证数据: {len(val_rows)} 条")
    print(f"训练数据路径: {train_data_path}")
    print(f"验证数据路径: {val_data_path}")
    print(f"每步处理样本数: {cfg.batch_size} × {cfg.grad_accum} = {cfg.batch_size * cfg.grad_accum}")
    print(f"总训练样本数: {cfg.max_steps} × {cfg.batch_size * cfg.grad_accum} = {cfg.max_steps * cfg.batch_size * cfg.grad_accum}")
    
    # 创建输出目录 - 包含关键参数信息
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 构建包含参数的目录名
    base_name = "sft_qwen_math_15b"
    size_suffix = f"size{len(train_rows)}" if dataset_size else "full"
    param_suffix = f"lr{cfg.lr:.0e}_bs{cfg.batch_size}_steps{cfg.max_steps}"
    if cfg.grad_accum != 8:  # 如果不是默认值，也加上
        param_suffix += f"_ga{cfg.grad_accum}"
    
    out_dir = MODEL_DIR / f"{base_name}_{size_suffix}_{param_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 更新wandb名称，包含参数信息
    cfg.wandb_name = f"{base_name}_{size_suffix}_{param_suffix}"
    
    print(f"输出目录: {out_dir}")
    print(f"Wandb实验名: {cfg.wandb_name}")

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
        
        # 提取纯答案用于reward计算
        pure_answers = []
        for r in sub[:8]:
            answer = r["response"]
            if "<answer>" in answer and "</answer>" in answer:
                pure_answer = answer.split("<answer>")[-1].replace("</answer>", "").strip()
            else:
                pure_answer = answer  # fallback
            pure_answers.append(pure_answer)
        
        lg = log_generations(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            prompts=prompts,
            ground_truths=pure_answers,  # 使用纯答案
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
