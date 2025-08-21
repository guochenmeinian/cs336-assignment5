#!/usr/bin/env python3
"""
GRPO trainer for math alignment experiments.
"""

import os
import sys
import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from adapters and shared modules
from tests.adapters import (
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_get_response_log_probs as get_response_log_probs,
    run_compute_group_normalized_rewards as compute_group_normalized_rewards,
    run_grpo_microbatch_train_step as grpo_microbatch_train_step,
    run_masked_mean as masked_mean,
)
from .config import GRPOConfig
from ..shared.wandb_config import (
    WandbConfig, init_wandb, log_training_batch, log_evaluation_batch, 
    log_model_config, finish_wandb
)
from ..shared.config import WANDB_PROJECT, WANDB_ENABLED
from ..shared.math_data_utils import load_math_data, format_r1_zero_prompt
from ..shared.math_evaluation_utils import evaluate_vllm


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_pad(tokenizer):
    """Ensure tokenizer has pad token."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"


def compute_gradient_norm(model) -> float:
    """Compute the gradient norm of the model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def compute_token_entropy(logits: torch.Tensor, response_mask: torch.Tensor) -> float:
    """Compute the average token entropy over response tokens."""
    from tests.adapters import run_compute_entropy
    
    entropy = run_compute_entropy(logits)  # (batch_size, sequence_length)
    # Average over response tokens only
    avg_entropy = masked_mean(entropy, response_mask, dim=None)
    return avg_entropy.item()


class GRPOTrainer:
    """GRPO trainer for math alignment."""
    
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.step = 0
        
        set_seed(cfg.seed)
        
        # Initialize tokenizer and model for training
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        ensure_pad(self.tokenizer)
        
        # Load model
        dtype = torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32
        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
        self.model = self.model.to(cfg.device)
        
        # Enable FlashAttention-2 after moving to GPU
        if cfg.use_flash_attention:
            self.model.config.attn_implementation = "flash_attention_2"
            
        self.model.gradient_checkpointing_enable()
        self.model.train()
        
        # Initialize vLLM model for generation
        self.vllm_model = LLM(
            model=cfg.model_id,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )
        
        # Initialize optimizer
        self.optimizer = cfg.get_optimizer(self.model)
        
        # Initialize wandb
        if cfg.wandb_enabled:
            wandb_config = WandbConfig(
                project=cfg.wandb_project or WANDB_PROJECT,
                name=cfg.wandb_name or "grpo_qwen_math_15b",
                tags=cfg.wandb_tags or ["grpo", "qwen", "math", "15b"],
                algorithm="grpo",
                dataset="math",
                model_name="qwen2.5-math-1.5b",
            )
            init_wandb(wandb_config)
            
            # Log model configuration
            config_dict = {
                "model_id": cfg.model_id,
                "n_grpo_steps": cfg.n_grpo_steps,
                "learning_rate": cfg.learning_rate,
                "advantage_eps": cfg.advantage_eps,
                "rollout_batch_size": cfg.rollout_batch_size,
                "group_size": cfg.group_size,
                "sampling_temperature": cfg.sampling_temperature,
                "sampling_max_tokens": cfg.sampling_max_tokens,
                "sampling_min_tokens": cfg.sampling_min_tokens,
                "epochs_per_rollout_batch": cfg.epochs_per_rollout_batch,
                "train_batch_size": cfg.train_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "loss_type": cfg.loss_type,
                "use_std_normalization": cfg.use_std_normalization,
                "max_grad_norm": cfg.max_grad_norm,
            }
            log_model_config(config_dict, algorithm="grpo")
    
    def sample_rollouts(self, questions: List[str], answers: List[str], reward_fn: Callable) -> Tuple[List[str], List[str], List[float]]:
        """Sample rollouts for the current batch of questions."""
        self.model.eval()
        
        # Format prompts using r1_zero template
        prompts = [format_r1_zero_prompt(q) for q in questions]
        
        # Repeat each prompt group_size times for group-based sampling
        repeated_prompts = []
        repeated_ground_truths = []
        for prompt, answer in zip(prompts, answers):
            repeated_prompts.extend([prompt] * self.cfg.group_size)
            repeated_ground_truths.extend([answer] * self.cfg.group_size)
        
        # Generate responses using vLLM
        sampling_params = SamplingParams(
            temperature=self.cfg.sampling_temperature,
            max_tokens=self.cfg.sampling_max_tokens,
            min_tokens=self.cfg.sampling_min_tokens,
            top_p=self.cfg.sampling_top_p,
            stop=self.cfg.sampling_stop,
            n=1,  # Generate one response per prompt
            seed=self.cfg.seed,
        )
        
        outputs = self.vllm_model.generate(repeated_prompts, sampling_params)
        
        # Extract responses
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        # Compute raw rewards
        raw_rewards = []
        for response, gt in zip(responses, repeated_ground_truths):
            reward_result = reward_fn(response, gt)
            raw_rewards.append(reward_result["reward"])
        
        self.model.train()
        return responses, repeated_ground_truths, raw_rewards
    
    def compute_advantages(self, raw_rewards: List[float]) -> torch.Tensor:
        """Compute group-normalized advantages."""
        # Convert to tensor
        raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
        
        # Group normalize manually since we already have raw rewards
        rollout_batch_size = len(raw_rewards)
        num_groups = rollout_batch_size // self.cfg.group_size
        
        advantages = torch.empty_like(raw_rewards_tensor)
        for g in range(num_groups):
            start = g * self.cfg.group_size
            end = start + self.cfg.group_size
            group_rewards = raw_rewards_tensor[start:end]
            group_mean = group_rewards.mean()
            if self.cfg.use_std_normalization:
                group_std = group_rewards.std(unbiased=True)
                denom = group_std + self.cfg.advantage_eps
                advantages[start:end] = (group_rewards - group_mean) / denom
            else:
                advantages[start:end] = (group_rewards - group_mean)
        
        return advantages
    
    def train_on_rollout_batch(self, responses: List[str], advantages: torch.Tensor, old_log_probs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Train on a rollout batch with proper microbatching.

        Splits the rollout batch of size `rollout_batch_size` into
        `n_microbatches_per_rollout_batch` microbatches, each of size
        `micro_train_batch_size`. Scales the loss by the number of microbatches
        to emulate large-batch training with gradient accumulation.
        """
        assert len(responses) == self.cfg.rollout_batch_size
        assert advantages.numel() == self.cfg.rollout_batch_size

        micro_size = self.cfg.micro_train_batch_size
        num_micro = self.cfg.n_microbatches_per_rollout_batch

        total_loss = 0.0
        total_entropy = 0.0
        last_metadata: Dict[str, torch.Tensor] = {}

        for m in range(num_micro):
            start = m * micro_size
            end = start + micro_size
            micro_responses = responses[start:end]
            micro_adv = advantages[start:end]

            # Tokenize responses (empty prompts; responses are full texts)
            pack = tokenize_prompt_and_output([], micro_responses, self.tokenizer)
            input_ids = pack["input_ids"].to(self.cfg.device)
            labels = pack["labels"].to(self.cfg.device)
            response_mask = pack["response_mask"].to(self.cfg.device)

            # Current policy log-probs (and entropy)
            out = get_response_log_probs(self.model, input_ids, labels, return_token_entropy=True)
            policy_log_probs = out["log_probs"]
            token_entropy = out.get("token_entropy")

            # Old log-probs (off-policy GRPO-Clip)
            micro_old_log_probs = None
            if old_log_probs is not None:
                micro_old_log_probs = old_log_probs[start:end].to(policy_log_probs.device)

            # One microbatch backward
            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=num_micro,
                loss_type=self.cfg.loss_type,
                raw_rewards=None,
                advantages=micro_adv.unsqueeze(-1),
                old_log_probs=micro_old_log_probs,
                cliprange=0.2 if self.cfg.loss_type == "grpo_clip" else None,
            )

            # Metrics accumulation
            mb_entropy = (
                masked_mean(token_entropy, response_mask, dim=None).item()
                if token_entropy is not None
                else compute_token_entropy(self.model(input_ids).logits, response_mask)
            )

            total_loss += float(loss.item())
            total_entropy += mb_entropy
            last_metadata = metadata

        # Optimizer step after all microbatches
        clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Aggregate metrics
        metrics = {
            "loss": total_loss / max(num_micro, 1),
            "token_entropy": total_entropy / max(num_micro, 1),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
        }
        if self.cfg.loss_type == "grpo_clip" and "clip_fraction" in last_metadata:
            metrics["clip_fraction"] = last_metadata["clip_fraction"].item()
        return metrics
    
    def evaluate_policy(self, validation_data: List[Dict], reward_fn: Callable) -> Dict[str, float]:
        """Evaluate the current policy on validation data."""
        self.model.eval()
        
        # Prepare validation data
        questions = [item["question"] for item in validation_data]
        answers = [item["answer"] for item in validation_data]
        
        # Format prompts
        prompts = [format_r1_zero_prompt(q) for q in questions]
        
        # Evaluation sampling parameters
        eval_sampling_params = SamplingParams(
            temperature=self.cfg.sampling_temperature,
            max_tokens=self.cfg.sampling_max_tokens,
            min_tokens=self.cfg.sampling_min_tokens,
            top_p=self.cfg.sampling_top_p,
            stop=self.cfg.sampling_stop,
            n=1,
            seed=self.cfg.seed,
        )
        
        # Evaluate
        results = evaluate_vllm(
            vllm_model=self.vllm_model,
            prompts=prompts,
            eval_sampling_params=eval_sampling_params,
            ground_truths=answers,
            reward_fn=reward_fn,
        )
        
        # Compute metrics
        rewards = [r["reward"] for r in results]
        format_rewards = [r["format_reward"] for r in results]
        answer_rewards = [r["answer_reward"] for r in results]
        
        metrics = {
            "val_reward_mean": np.mean(rewards),
            "val_reward_std": np.std(rewards),
            "val_format_reward_mean": np.mean(format_rewards),
            "val_answer_reward_mean": np.mean(answer_rewards),
            "val_accuracy": np.mean([r > 0 for r in rewards]),
        }
        
        self.model.train()
        return metrics, results
    
    def train(self, train_data: List[Dict], validation_data: List[Dict], reward_fn: Callable):
        """Main training loop for GRPO."""
        print(f"Starting GRPO training with {self.cfg.n_grpo_steps} steps")
        print(f"Rollout batch size: {self.cfg.rollout_batch_size}, Group size: {self.cfg.group_size}")
        print(f"Train batch size: {self.cfg.train_batch_size}, Micro batch size: {self.cfg.micro_train_batch_size}")
        
        # Training loop
        for step in range(self.cfg.n_grpo_steps):
            self.step = step
            
            # Sample rollouts
            batch_indices = random.sample(range(len(train_data)), self.cfg.n_prompts_per_rollout_batch)
            batch_data = [train_data[i] for i in batch_indices]
            
            questions = [item["question"] for item in batch_data]
            answers = [item["answer"] for item in batch_data]
            
            print(f"Step {step + 1}/{self.cfg.n_grpo_steps}: Sampling rollouts...")
            responses, repeated_ground_truths, raw_rewards = self.sample_rollouts(questions, answers, reward_fn)
            
            # Compute advantages
            advantages = self.compute_advantages(raw_rewards)
            
            # Store old log probs if using GRPO-Clip and off-policy
            old_log_probs = None
            if self.cfg.loss_type == "grpo_clip" and self.cfg.epochs_per_rollout_batch > 1:
                # Compute old log probs once and reuse for each epoch
                pack = tokenize_prompt_and_output([], responses, self.tokenizer)
                input_ids = pack["input_ids"].to(self.cfg.device)
                labels = pack["labels"].to(self.cfg.device)
                
                with torch.no_grad():
                    out = get_response_log_probs(self.model, input_ids, labels, return_token_entropy=False)
                    old_log_probs = out["log_probs"].detach()
            
            # Train on rollout batch for multiple epochs if specified
            for epoch in range(self.cfg.epochs_per_rollout_batch):
                metrics = self.train_on_rollout_batch(responses, advantages, old_log_probs)
                
                # Log training metrics
                if self.cfg.wandb_enabled and step % self.cfg.log_every == 0:
                    grad_norm = compute_gradient_norm(self.model)
                    log_training_batch(
                        step=step,
                        loss=metrics["loss"],
                        lr=self.optimizer.param_groups[0]["lr"],
                        grad_norm=grad_norm,
                        algorithm="grpo",
                        **metrics
                    )
            
            # Evaluate periodically
            if (step + 1) % self.cfg.eval_every == 0:
                print(f"Step {step + 1}: Evaluating policy...")
                val_metrics, val_results = self.evaluate_policy(validation_data, reward_fn)
                
                if self.cfg.wandb_enabled:
                    log_evaluation_batch(
                        step=step,
                        algorithm="grpo",
                        **val_metrics
                    )
                
                print(f"Validation - Reward: {val_metrics['val_reward_mean']:.4f}, Accuracy: {val_metrics['val_accuracy']:.4f}")
            
            # Log step progress
            if step % 10 == 0:
                print(f"Step {step + 1}/{self.cfg.n_grpo_steps} completed")
                print(f"  Loss: {metrics['loss']:.6f}")
                print(f"  Token Entropy: {metrics['token_entropy']:.4f}")
                print(f"  Advantage Mean: {metrics['advantage_mean']:.4f}")
                if "clip_fraction" in metrics:
                    print(f"  Clip Fraction: {metrics['clip_fraction']:.4f}")
        
        print("GRPO training completed!")
    
    def save(self, out_dir: str):
        """Save the trained model."""
        os.makedirs(out_dir, exist_ok=True)
        
        self.model.save_pretrained(out_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(out_dir)
        
        if self.cfg.wandb_enabled:
            finish_wandb()
        
        print(f"GRPO model saved to {out_dir}")
