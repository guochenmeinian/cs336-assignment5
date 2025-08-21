#!/usr/bin/env python3
"""
Expert Iteration trainer for math alignment experiments.
"""

import os
import sys
import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from SFT trainer and adapters
from .config import EIConfig
from tests.adapters import (
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_get_response_log_probs as get_response_log_probs,
    run_sft_microbatch_train_step as sft_microbatch_train_step,
)
# Import wandb utilities from shared
from ..shared.wandb_config import (
    WandbConfig, init_wandb, log_training_batch, log_evaluation_batch, 
    log_ei_iteration, log_model_config, finish_wandb
)
from ..shared.config import WANDB_PROJECT, WANDB_ENABLED


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_pad(tokenizer):
    """Ensure tokenizer has pad token."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"


def compute_entropy(log_probs: torch.Tensor) -> float:
    """Compute entropy from log probabilities."""
    probs = torch.exp(log_probs)
    # Add small epsilon to avoid log(0)
    probs = probs + 1e-8
    probs = probs / probs.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy.mean().item()


class EITrainer:
    """Expert Iteration trainer."""
    
    def __init__(self, cfg: EIConfig):
        self.cfg = cfg
        self.ei_step = 0
        self._last_nll = None
        
        set_seed(cfg.seed)
        
        # Initialize tokenizer and model for training
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        ensure_pad(self.tokenizer)
        
        # Load model first, then enable FlashAttention-2 after moving to GPU
        dtype = torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32
        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        
        # Load model without FlashAttention first
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
        
        # Move to GPU first
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
            gpu_memory_utilization=0.5,
        )
        
        # Initialize optimizer
        self.optim = AdamW(self.model.parameters(), lr=cfg.lr)
        
        # Initialize wandb
        if cfg.wandb_enabled:
            wandb_config = WandbConfig(
                project=cfg.wandb_project or WANDB_PROJECT,
                name=cfg.wandb_name or "ei_qwen_math_15b",
                tags=cfg.wandb_tags or ["ei", "qwen", "math", "15b"],
                algorithm="ei",
                dataset="math",
                model_name="qwen2.5-math-1.5b",
            )
            init_wandb(wandb_config)
            
            # Log model configuration
            config_dict = {
                "model_id": cfg.model_id,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum,
                "max_steps": cfg.max_steps,
                "n_ei_steps": cfg.n_ei_steps,
                "G": cfg.G,
                "sampling_temperature": cfg.sampling_temperature,
                "sampling_max_tokens": cfg.sampling_max_tokens,
                "sampling_min_tokens": cfg.sampling_min_tokens,
                "sampling_top_p": cfg.sampling_top_p,
                "sampling_stop": cfg.sampling_stop,
                "ei_batch_sizes": cfg.ei_batch_sizes,
                "use_flash_attention": cfg.use_flash_attention,
            }
            log_model_config(config_dict, algorithm="ei")
    
    def sample_responses(self, questions: List[str], answers: List[str], reward_fn: Callable, ei_batch_size: int) -> List[Dict]:
        """Sample G responses for each question using vLLM and filter by reward."""
        self.model.eval()
        filtered_data = []
        total_samples = 0
        correct_samples = 0
        
        # Use vLLM for generation
        sampling_params = SamplingParams(
            temperature=self.cfg.sampling_temperature,
            max_tokens=self.cfg.sampling_max_tokens,
            min_tokens=self.cfg.sampling_min_tokens,
            top_p=self.cfg.sampling_top_p,
            stop=self.cfg.sampling_stop,
            n=self.cfg.G,
            seed=self.cfg.seed,
        )
        
        # Process questions in batches
        for i in range(0, len(questions), ei_batch_size):
            batch_questions = questions[i:i + ei_batch_size]
            batch_answers = answers[i:i + ei_batch_size]
            
            # Generate G responses for each question in the batch
            all_prompts = []
            for question in batch_questions:
                all_prompts.extend([question] * self.cfg.G)
            
            # Generate responses using vLLM
            outputs = self.vllm_model.generate(all_prompts, sampling_params)
            
            # Process outputs
            for j, (question, answer) in enumerate(zip(batch_questions, batch_answers)):
                question_outputs = outputs[j * self.cfg.G:(j + 1) * self.cfg.G]
                
                for output in question_outputs:
                    response = output.outputs[0].text.strip()
                    total_samples += 1
                    
                    # Compute reward with correct parameter order: (response, ground_truth)
                    reward_result = reward_fn(response, answer)  # Fixed: (response, ground_truth)
                    reward = reward_result["reward"]  # Extract the reward value
                    
                    if reward > 0:  # Keep correct responses
                        correct_samples += 1
                        filtered_data.append({
                            "prompt": question,
                            "response": response,
                            "reward": reward,
                            "format_reward": reward_result.get("format_reward", 0.0),
                            "answer_reward": reward_result.get("answer_reward", 0.0)
                        })
        
        # Log sampling statistics to wandb
        if self.cfg.wandb_enabled:
            log_ei_iteration(
                ei_step=self.ei_step,
                total_samples=total_samples,
                correct_samples=correct_samples,
                filtered_size=len(filtered_data),
                algorithm="ei"
            )
        
        self.model.train()
        return filtered_data
    
    def train_one_ei_iteration(self, questions: List[str], answers: List[str], reward_fn: Callable, ei_batch_size: int):
        """Run one complete Expert Iteration step."""
        print(f"Starting EI iteration {self.ei_step + 1}/{self.cfg.n_ei_steps} with batch size {ei_batch_size}")
        
        # Step 1: Sample and filter responses
        filtered_data = self.sample_responses(questions, answers, reward_fn, ei_batch_size)
        print(f"Generated {len(filtered_data)} correct samples")
        
        if len(filtered_data) == 0:
            print("Warning: No correct samples generated. Skipping this iteration.")
            return False
        
        # Step 2: Train on filtered data
        self._train_on_filtered_data(filtered_data)
        
        self.ei_step += 1
        return True
    
    def _train_on_filtered_data(self, filtered_data: List[Dict]):
        """Train the model on filtered data."""
        step = 0
        total_loss = 0.0
        total_nll = 0.0
        total_entropy = 0.0
        
        while step < self.cfg.max_steps:
            # Sample batch
            batch = random.sample(filtered_data, min(self.cfg.batch_size, len(filtered_data)))
            prompts = [item["prompt"] for item in batch]
            responses = [item["response"] for item in batch]
            
            # Tokenize and train
            pack = tokenize_prompt_and_output(prompts, responses, self.tokenizer)
            input_ids = pack["input_ids"].to(self.cfg.device)
            labels = pack["labels"].to(self.cfg.device)
            rmask = pack["response_mask"].to(self.cfg.device)
            
            out = get_response_log_probs(self.model, input_ids, labels, return_token_entropy=False)
            logp = out["log_probs"]
            
            # Compute entropy for logging
            entropy = compute_entropy(logp)
            total_entropy += entropy
            
            loss, meta = sft_microbatch_train_step(
                policy_log_probs=logp,
                response_mask=rmask,
                gradient_accumulation_steps=self.cfg.grad_accum,
                normalize_constant=1.0,
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            nll = (meta["masked_sum"] / meta["num_response_tokens"].clamp_min(1)).item()
            total_nll += nll
            
            # Log to wandb every few steps
            if self.cfg.wandb_enabled and step % 10 == 0:
                grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
                log_training_batch(
                    step=step,
                    loss=loss.item(),
                    nll=nll,
                    lr=self.optim.param_groups[0]["lr"],
                    grad_norm=grad_norm,
                    algorithm="ei"
                )
            
            # Gradient accumulation
            if (step + 1) % self.cfg.grad_accum == 0:
                clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
            
            step += 1
        
        # Log final training statistics for this EI iteration
        if self.cfg.wandb_enabled:
            avg_loss = total_loss / max(step, 1)
            avg_nll = total_nll / max(step, 1)
            avg_entropy = total_entropy / max(step, 1)
            log_ei_iteration(
                ei_step=self.ei_step,
                final_avg_loss=avg_loss,
                final_avg_nll=avg_nll,
                final_avg_entropy=avg_entropy,
                total_steps=step,
                algorithm="ei"
            )
        
        # Store final NLL for this iteration
        self._last_nll = total_nll / max(step, 1)
    
    def run_expert_iteration(self, questions: List[str], answers: List[str], reward_fn: Callable):
        """Run one Expert Iteration experiment with specified batch size."""
        print(f"Starting Expert Iteration experiment with batch_size={self.cfg.experiment_batch_size}, {self.cfg.n_ei_steps} iterations")
        
        # Run EI iterations for this experiment
        for ei_step in range(self.cfg.n_ei_steps):
            success = self.train_one_ei_iteration(questions, answers, reward_fn, self.cfg.experiment_batch_size)
            if not success:
                print(f"Experiment failed at iteration {ei_step + 1}")
                break
        
        print("Expert Iteration experiment completed!")
    
    def save(self, out_dir: str):
        """Save the trained model."""
        os.makedirs(out_dir, exist_ok=True)
        
        self.model.save_pretrained(out_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(out_dir)
        
        if self.cfg.wandb_enabled:
            finish_wandb()
        
        print(f"EI model saved to {out_dir}")
