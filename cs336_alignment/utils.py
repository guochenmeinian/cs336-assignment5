from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


@torch.inference_mode()
def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    output_jsonl_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Minimal in-the-loop generation logger.

    Per example logs: prompt, response, ground_truth, rewards, token_entropy_mean, response_length_tokens.
    Returns aggregate metrics and optionally writes JSONL records if output_jsonl_path is provided.
    """
    assert len(prompts) == len(ground_truths)

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")

    examples: List[Dict[str, Any]] = []

    for prompt, gt in zip(prompts, ground_truths):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.size(1)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-6) if do_sample else 1.0,
            "top_p": top_p,
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        generated = model.generate(input_ids=input_ids, **gen_kwargs)
        full_ids = generated[0].tolist()
        response_ids = full_ids[prompt_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Reward
        rewards = reward_fn(response, gt) or {}
        format_reward = float(rewards.get("format_reward", 0.0))
        answer_reward = float(rewards.get("answer_reward", 0.0))
        total_reward = float(rewards.get("reward", 0.0))

        # Entropy over response span (positions predicting response tokens)
        inputs = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits = model(inputs).logits  # (1, L, V)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)[0]  # (L,)
        if token_entropy.size(0) <= 1:
            token_entropy_mean = 0.0
        else:
            start = max(prompt_len - 1, 0)
            end = max(token_entropy.size(0) - 1, start)
            token_entropy_mean = (
                token_entropy[start:end].mean().item() if end > start else 0.0
            )

        ex = {
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "rewards": {
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "reward": total_reward,
            },
            "token_entropy_mean": float(token_entropy_mean),
            "response_length_tokens": int(len(response_ids)),
        }
        examples.append(ex)

    # Aggregates
    entropies = [e["token_entropy_mean"] for e in examples]
    lengths = [e["response_length_tokens"] for e in examples]
    correct_mask = [e["rewards"]["answer_reward"] >= 0.5 for e in examples]
    lengths_correct = [l for l, ok in zip(lengths, correct_mask) if ok]
    lengths_incorrect = [l for l, ok in zip(lengths, correct_mask) if not ok]

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    metrics = {
        "avg_token_entropy": _avg(entropies),
        "avg_response_length": _avg(lengths),
        "avg_response_length_correct": _avg(lengths_correct),
        "avg_response_length_incorrect": _avg(lengths_incorrect),
        "accuracy": _avg([1.0 if ok else 0.0 for ok in correct_mask]),
        "total_examples": len(examples),
    }

    if output_jsonl_path is not None:
        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(
        "Logged %d examples | acc=%.3f len=%.1f ent=%.3f",
        metrics["total_examples"],
        metrics["accuracy"],
        metrics["avg_response_length"],
        metrics["avg_token_entropy"],
    )

    return {"examples": examples, "metrics": metrics}



