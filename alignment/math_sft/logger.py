# alignment/math_sft/logger.py
from __future__ import annotations
import os, sys
from typing import Sequence, Callable, Optional, Dict, Any
import numpy as np
import torch

# --- 让 tests/ 成为可导入的包源（alignment 与 tests 并列时生效） ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 来自 tests/adapters.py 的函数 ---
from tests.adapters import (  # type: ignore
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_get_response_log_probs as get_response_log_probs,
)

@torch.no_grad()
def log_generations(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    reward_fn: Callable[[str, str], dict],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    device: Optional[str] = None,
    return_examples: bool = False,
) -> Dict[str, Any]:
    """
    产出作业要求的全部日志：
    1) prompt  
    2) 生成 response  
    3) ground-truth
    4) 奖励(总/format/answer)  
    5) 平均 token 熵  
    6) 长度统计(总均值/正确均值/错误均值)
    """
    assert len(prompts) == len(ground_truths)
    if device is None:
        device = str(next(model.parameters()).device)

    was_training = model.training
    model.eval()

    # === 生成 responses ===
    responses = []
    for s in range(0, len(prompts), batch_size):
        e = min(s + batch_size, len(prompts))
        enc = tokenizer(prompts[s:e], padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        full = tokenizer.batch_decode(gen, skip_special_tokens=True)
        src  = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)
        # 截出纯 response 段（若前缀不完全匹配则保留全文）
        responses.extend([f[len(p):] if f.startswith(p) else f for p, f in zip(src, full)])

    # === token 熵（仅对 response tokens 求平均） ===
    pack = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = pack["input_ids"].to(device)
    labels    = pack["labels"].to(device)
    rmask     = pack["response_mask"].to(device)  # (B, T) bool

    out = get_response_log_probs(
        model=model, input_ids=input_ids, labels=labels, return_token_entropy=True
    )
    token_entropy = out["token_entropy"]                            # (B, T)
    mask = rmask.to(dtype=token_entropy.dtype)
    resp_token_counts = mask.sum(dim=-1).clamp_min(1.0)             # (B,)
    avg_token_entropy = ((token_entropy * mask).sum(dim=-1) / resp_token_counts).cpu().tolist()

    # === 奖励与长度 ===
    examples = []
    correct_flags = []
    resp_lens = []
    for i, (p, r, gt) in enumerate(zip(prompts, responses, ground_truths)):
        scores = reward_fn(r, gt)
        fmt   = float(scores.get("format_reward", 0.0))
        ans   = float(scores.get("answer_reward", 0.0))
        # 计算总reward：格式正确且答案正确时为1.0，否则为0.0
        total = 1.0 if (fmt >= 0.5 and ans >= 0.5) else 0.0
        L = len(tokenizer.encode(r, add_special_tokens=False))
        examples.append({
            "prompt": p,
            "response": r,
            "ground_truth": gt,
            "reward": total,
            "format_reward": fmt,
            "answer_reward": ans,
            "avg_token_entropy": float(avg_token_entropy[i]),
            "response_len_tokens": int(L),
        })
        correct_flags.append(ans >= 0.5)   # 以 answer_reward>=0.5 当作正确
        resp_lens.append(L)

    # === 聚合统计 ===
    resp_lens = np.array(resp_lens, dtype=float)
    correct_flags = np.array(correct_flags, dtype=bool)
    def _mean(x): return float(x.mean()) if x.size > 0 else float("nan")

    summary = {
        "n": len(examples),
        "reward_mean": _mean(np.array([e["reward"] for e in examples], dtype=float)),
        "format_reward_mean": _mean(np.array([e["format_reward"] for e in examples], dtype=float)),
        "answer_reward_mean": _mean(np.array([e["answer_reward"] for e in examples], dtype=float)),
        "avg_token_entropy_mean": _mean(np.array([e["avg_token_entropy"] for e in examples], dtype=float)),
        "resp_len_mean": _mean(resp_lens),
        "resp_len_mean_correct": _mean(resp_lens[correct_flags]) if correct_flags.any() else float("nan"),
        "resp_len_mean_incorrect": _mean(resp_lens[~correct_flags]) if (~correct_flags).any() else float("nan"),
    }

    if was_training:
        model.train()
    return {"summary": summary, "examples": examples if return_examples else None}
