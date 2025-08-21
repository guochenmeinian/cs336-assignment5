from __future__ import annotations

import os
import token
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

import torch.nn.functional as F


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    
    # Tokenize each prompt and output string
    prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_tokens = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    
    # Calculate lengths for each prompt+output combination
    prompt_and_output_lens = [len(pt) + len(ot) for pt, ot in zip(prompt_tokens, output_tokens)]
    max_len = max(prompt_and_output_lens)
    
    input_ids = []
    labels = []
    response_mask = []
    
    for pt, ot in zip(prompt_tokens, output_tokens):
        # Concatenate prompt and output tokens
        ids = pt + ot
        # Pad to max_len
        pad_len = max_len - len(ids)
        ids_padded = ids + [tokenizer.pad_token_id] * pad_len
        # Shifted labels (next token prediction)
        labels_padded = ids_padded[1:] + [tokenizer.pad_token_id]

        # align mask and labels：len(pt)-1 set as True
        mask_for_labels = (
            [False] * max(len(pt) - 1, 0) +
            [True]  * len(ot) +
            [False] * (pad_len + 1)
        )
    
        # Remove last token for input_ids, labels, mask
        input_ids.append(torch.tensor(ids_padded[:-1]))
        labels.append(torch.tensor(labels_padded[:-1]))
        response_mask.append(torch.tensor(mask_for_labels[:-1], dtype=torch.bool))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "response_mask": torch.stack(response_mask)
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    # Compute raw rewards for each response against its ground truth
    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size == len(repeated_ground_truths)

    raw_rewards_list: list[float] = []
    format_rewards_list: list[float] = []
    answer_rewards_list: list[float] = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, gt)
        raw_rewards_list.append(float(scores["reward"]))
        # Optional metadata
        format_rewards_list.append(float(scores.get("format_reward", scores["reward"])))
        answer_rewards_list.append(float(scores.get("answer_reward", scores["reward"])))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # Group-normalize within each group of size `group_size`
    assert rollout_batch_size % group_size == 0
    num_groups = rollout_batch_size // group_size

    normalized = torch.empty_like(raw_rewards)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]
        group_mean = group_rewards.mean()
        if normalize_by_std:
            group_std = group_rewards.std(unbiased=True)
            denom = group_std + advantage_eps
            normalized[start:end] = (group_rewards - group_mean) / denom
        else:
            normalized[start:end] = (group_rewards - group_mean)

    metadata: dict[str, float] = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=True).item()),
        "format_reward_mean": float(torch.tensor(format_rewards_list).mean().item()),
        "answer_reward_mean": float(torch.tensor(answer_rewards_list).mean().item()),
    }

    return normalized, raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # Convert logits to log-probabilities (log-softmax for numerical stability)
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, sequence_length, vocab_size)
    probs = torch.exp(log_probs)               # (batch_size, sequence_length, vocab_size)
    # Entropy: -sum(p * log(p)) over vocab dimension
    entropy = -(probs * log_probs).sum(dim=-1) # (batch_size, sequence_length)
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # Forward pass to get logits
    outputs = model(input_ids)
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, sequence_length, vocab_size)
    
    # Get log probs of the labels (next token prediction)
    log_probs_of_labels = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Shape: (batch_size, sequence_length)

    # Compute token entropy if requested
    token_entropy = None
    if return_token_entropy:
        token_entropy = run_compute_entropy(logits)

    # Return the results in a dictionary
    result = {"log_probs": log_probs_of_labels}
    if return_token_entropy:
        result["token_entropy"] = token_entropy

    return result
    

def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    # Broadcast rewards/advantages across sequence dimension and negate for minimization
    return -(raw_rewards_or_advantages * policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    # Expand advantages to match sequence length for broadcasting
    advantages_expanded = advantages.expand(-1, policy_log_probs.shape[1])
    
    # Ratio between current and old policy
    ratio = torch.exp(policy_log_probs - old_log_probs)
    # Unclipped and clipped objectives
    unclipped = ratio * advantages_expanded
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = clipped_ratio * advantages_expanded
    # PPO-style clipped objective: take the minimum, negate to form a loss we minimize
    per_token_loss = -torch.minimum(unclipped, clipped)

    # Metadata (clip fraction)
    was_clipped = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
    metadata = {"clip_fraction": was_clipped.float().mean()}
    return per_token_loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = run_compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards, policy_log_probs=policy_log_probs
        )
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = run_compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages, policy_log_probs=policy_log_probs
        )
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return run_compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    mask_float = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_float
    if dim is None:
        denom = mask_float.sum().clamp_min(1.0)
        return masked.sum() / denom
    else:
        denom = mask_float.sum(dim=dim).clamp_min(1.0)
        return masked.sum(dim=dim) / denom


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    # Negative log-likelihood over response tokens
    masked_loss = -policy_log_probs * response_mask

    numerator = masked_loss.sum()
    batch_size = policy_log_probs.size(0)
    denom = gradient_accumulation_steps * batch_size * normalize_constant

    loss = numerator / denom

    loss.backward()

    # 可回传一些元信息便于监控
    meta = {
        "masked_sum": numerator.detach(),
        "batch_size": torch.tensor(batch_size, dtype=policy_log_probs.dtype),
        "ga_steps": torch.tensor(gradient_accumulation_steps, dtype=policy_log_probs.dtype),
        "normalize_constant": torch.tensor(normalize_constant, dtype=policy_log_probs.dtype),
        "num_response_tokens": response_mask.sum().to(policy_log_probs.dtype),
    }
    return loss.detach(), meta

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    # Compute per-token loss according to selected policy gradient objective
    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards if raw_rewards is not None else None,
        advantages=advantages if advantages is not None else None,
        old_log_probs=old_log_probs if old_log_probs is not None else None,
        cliprange=cliprange if cliprange is not None else None,
    )

    # Reduce over response tokens by summing (Dr.GRPO style) and scale for accumulation
    loss = run_masked_normalize(
        tensor=per_token_loss,
        mask=response_mask.to(dtype=per_token_loss.dtype),
        dim=None,
        normalize_constant=1.0,
    )
    loss = loss / float(gradient_accumulation_steps)
    loss.backward()
    return loss.detach(), metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    # Apply mask to tensor (set masked elements to 0)
    masked_tensor = tensor * mask
    
    # Sum over the specified dimension(s)
    if dim is not None:
        # Sum over the specified dimension
        summed = masked_tensor.sum(dim=dim)
    else:
        # Sum over all dimensions
        summed = masked_tensor.sum()
    
    # Normalize by the constant
    normalized = summed / normalize_constant
    
    return normalized


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    import json
    import random
    from torch.utils.data import Dataset as TorchDataset

    with open(dataset_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if shuffle:
        random.shuffle(records)

    # Build a single long token stream from concatenated prompt+response pairs
    eos_id = tokenizer.eos_token_id
    all_tokens: list[int] = []
    for rec in records:
        prompt: str = rec["prompt"]
        response: str = rec["response"]
        text = prompt + response
        ids = tokenizer.encode(text, add_special_tokens=True)
        if eos_id is not None:
            ids = ids + [eos_id]
        all_tokens.extend(ids)

    # Create packed (input_ids, labels) of fixed seq_length
    examples: list[dict[str, Tensor]] = []
    total_len = len(all_tokens)
    # We need labels to be next-token shifted targets, so step by seq_length
    for start in range(0, max(total_len - 1, 0), seq_length):
        end = start + seq_length
        if end + 1 > total_len:
            break
        input_ids_chunk = all_tokens[start:end]
        labels_chunk = all_tokens[start + 1 : end + 1]
        examples.append(
            {
                "input_ids": torch.tensor(input_ids_chunk, dtype=torch.long),
                "labels": torch.tensor(labels_chunk, dtype=torch.long),
            }
        )

    class PackedSFTDataset(TorchDataset):
        def __init__(self, data: list[dict[str, Tensor]]):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            return self.data[idx]

    return PackedSFTDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    class BatchIterator:
        def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool):
            self.dataset = dataset
            self.batch_size = batch_size
            indices = list(range(len(dataset)))
            if shuffle:
                import random

                random.shuffle(indices)
            self.indices = indices
            # Pre-slice into batches of indices
            self._batches: list[list[int]] = [
                self.indices[i : i + batch_size]
                for i in range(0, len(self.indices), batch_size)
            ]

        def __len__(self) -> int:
            import math

            return math.ceil(len(self.indices) / self.batch_size)

        def __iter__(self):
            for batch_idxs in self._batches:
                batch_examples = [self.dataset[i] for i in batch_idxs]
                input_ids = torch.stack([ex["input_ids"] for ex in batch_examples])
                labels = torch.stack([ex["labels"] for ex in batch_examples])
                yield {"input_ids": input_ids, "labels": labels}

    return BatchIterator(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    import re

    # Look for the first standalone letter A-D (case-insensitive)
    match = re.search(r"\b([A-D])\b", model_output, flags=re.IGNORECASE)
    if not match:
        return None
    letter = match.group(1).upper()
    # Ensure the letter is one of the provided options
    options = mmlu_example.get("options", [])
    if letter in ["A", "B", "C", "D"] and 0 <= ord(letter) - ord("A") < len(options):
        return letter
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    import re

    numbers = re.findall(r"-?\d+", model_output)
    return numbers[-1] if numbers else None


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    import torch.nn.functional as F

    def compute_response_logprob(model: torch.nn.Module, prompt: str, response: str) -> torch.Tensor:
        # Tokenize without adding special tokens to control exact offsets
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
        outputs = model(input_ids)
        logits = outputs.logits  # (1, L, V)
        log_probs = F.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:]
        token_log_probs = torch.gather(log_probs[:, :-1, :], dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # Positions whose next-token target lies within the response span
        prompt_len = len(prompt_ids)
        total_len = input_ids.size(1)
        # t indices contributing to response tokens: from prompt_len-1 to total_len-2
        start_t = max(prompt_len - 1, 0)
        end_t = total_len - 1  # exclusive in slicing
        if end_t <= start_t:
            return torch.tensor(0.0)
        resp_token_log_probs = token_log_probs[:, start_t:end_t]
        return resp_token_log_probs.sum()

    # Compute model and reference log-probabilities for chosen and rejected
    logp_chosen = compute_response_logprob(lm, prompt, response_chosen)
    logp_rejected = compute_response_logprob(lm, prompt, response_rejected)
    logp_chosen_ref = compute_response_logprob(lm_ref, prompt, response_chosen)
    logp_rejected_ref = compute_response_logprob(lm_ref, prompt, response_rejected)

    delta_model = logp_chosen - logp_rejected
    delta_ref = logp_chosen_ref - logp_rejected_ref

    loss = -F.logsigmoid(beta * (delta_model - delta_ref))
    return loss
