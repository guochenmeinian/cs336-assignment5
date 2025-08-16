#!/usr/bin/env python3
"""
Unified Wandb configuration and utilities for all math alignment algorithms.
Provides consistent logging format across SFT, EI, GRPO, etc.
"""

import os
import wandb
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import torch
import numpy as np

# Logging frequency configuration
@dataclass
class LoggingConfig:
    """Configuration for logging frequencies and settings."""
    # Training metrics logging frequency
    train_log_every: int = 10  # Log training metrics every N steps
    
    # Evaluation and generation logging frequency  
    eval_log_every: int = 50   # Log evaluation metrics every N steps
    
    # EI iteration logging (per iteration)
    ei_log_per_iteration: bool = True
    
    # Wandb specific settings
    wandb_log_media: bool = True      # Log tables, images, etc.
    wandb_log_config: bool = True     # Log model config to wandb
    wandb_log_examples: bool = True   # Log generation examples as tables
    
    # Logging verbosity
    log_training_loss: bool = True
    log_training_nll: bool = True
    log_training_lr: bool = True
    log_training_grad_norm: bool = True
    log_eval_rewards: bool = True
    log_eval_accuracy: bool = True
    log_eval_lengths: bool = True

# Default logging configuration
DEFAULT_LOGGING_CONFIG = LoggingConfig()


@dataclass
class WandbConfig:
    """Unified wandb configuration for all math algorithms."""
    
    # Basic wandb settings
    project: str = "cs336-math-alignment"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    enabled: bool = True
    save_code: bool = True
    
    # Algorithm-specific settings
    algorithm: str = "sft"  # "sft", "ei", "grpo", etc.
    dataset: str = "gsm8k"  # "gsm8k", "math", etc.
    model_name: str = "qwen2.5-math-1.5b"
    
    def __post_init__(self):
        # Set default entity from environment if not specified
        if self.entity is None:
            self.entity = os.environ.get("WANDB_ENTITY")
        
        # Set default project from environment if not specified
        if self.project is None:
            self.project = os.environ.get("WANDB_PROJECT", "cs336-math-alignment")
        
        # Set default name if not specified
        if self.name is None:
            self.name = f"{self.algorithm}-{self.dataset}-{self.model_name}"
        
        # Set default tags if not specified
        if self.tags is None:
            self.tags = [self.algorithm, self.dataset, self.model_name, "math"]


def init_wandb(config: WandbConfig, **kwargs) -> Optional[wandb.run]:
    """Initialize wandb run with unified configuration."""
    if not config.enabled:
        return None
    
    try:
        run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.name,
            tags=config.tags,
            notes=config.notes,
            save_code=config.save_code,
            config={
                "algorithm": config.algorithm,
                "dataset": config.dataset,
                "model_name": config.model_name,
            },
            **kwargs
        )
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, 
                algorithm: str = "unknown", **kwargs):
    """Log metrics to wandb with unified format."""
    if wandb.run is None:
        return
    
    # Add algorithm prefix if not already present
    formatted_metrics = {}
    for key, value in metrics.items():
        if not key.startswith(f"{algorithm}/"):
            formatted_key = f"{algorithm}/{key}"
        else:
            formatted_key = key
        formatted_metrics[formatted_key] = value
    
    wandb.log(formatted_metrics, step=step, **kwargs)


def log_training_step(step: int, loss: float, nll: float, lr: float, 
                      grad_norm: float, algorithm: str = "unknown", **kwargs):
    """Log training step metrics with unified format."""
    log_metrics({
        "train/step": step,
        "train/loss": loss,
        "train/nll": nll,
        "train/learning_rate": lr,
        "train/gradient_norm": grad_norm,
    }, step=step, algorithm=algorithm, **kwargs)


def log_evaluation_metrics(metrics: Dict[str, float], step: int, 
                          algorithm: str = "unknown", **kwargs):
    """Log evaluation metrics with unified format."""
    formatted_metrics = {}
    for key, value in metrics.items():
        formatted_metrics[f"eval/{key}"] = value
    
    log_metrics(formatted_metrics, step=step, algorithm=algorithm, **kwargs)


def log_generation_examples(examples: List[Dict], step: int, 
                           algorithm: str = "unknown", **kwargs):
    """Log generation examples to wandb as a table with unified format."""
    if wandb.run is None:
        return
    
    # Create a table for generation examples
    table_data = []
    for i, example in enumerate(examples):
        table_data.append([
            i,
            example["prompt"][:200] + "..." if len(example["prompt"]) > 200 else example["prompt"],
            example["response"][:200] + "..." if len(example["response"]) > 200 else example["response"],
            example.get("ground_truth", "")[:200] + "..." if len(example.get("ground_truth", "")) > 200 else example.get("ground_truth", ""),
            example.get("reward", 0.0),
            example.get("format_reward", 0.0),
            example.get("answer_reward", 0.0),
            example.get("avg_token_entropy", 0.0),
            example.get("response_len_tokens", 0),
        ])
    
    table = wandb.Table(
        columns=["Index", "Prompt", "Response", "Ground Truth", "Reward", "Format Reward", "Answer Reward", "Token Entropy", "Response Length"],
        data=table_data
    )
    
    wandb.log({f"{algorithm}/generation_examples": table}, step=step, **kwargs)


def log_ei_iteration(ei_step: int, total_samples: int, correct_samples: int, 
                     filtered_size: int, algorithm: str = "ei",
                     logging_config: Optional[LoggingConfig] = None, **kwargs):
    """Log Expert Iteration statistics with unified format."""
    if logging_config is None:
        logging_config = DEFAULT_LOGGING_CONFIG
    
    # Check if we should log EI iterations
    if not logging_config.ei_log_per_iteration:
        return
    
    metrics = {
        f"{algorithm}/iteration_{ei_step}/sampling_total": total_samples,
        f"{algorithm}/iteration_{ei_step}/sampling_correct": correct_samples,
        f"{algorithm}/iteration_{ei_step}/sampling_accuracy": correct_samples / max(total_samples, 1),
        f"{algorithm}/iteration_{ei_step}/filtered_dataset_size": filtered_size,
    }
    
    log_metrics(metrics, algorithm=algorithm, **kwargs)


def log_model_config(config_dict: Dict[str, Any], algorithm: str = "unknown",
                    logging_config: Optional[LoggingConfig] = None, **kwargs):
    """Log model and training configuration with unified format."""
    if logging_config is None:
        logging_config = DEFAULT_LOGGING_CONFIG
    
    if not logging_config.wandb_log_config or wandb.run is None:
        return
    
    # Add algorithm prefix to config keys
    formatted_config = {}
    for key, value in config_dict.items():
        formatted_config[f"{algorithm}/config/{key}"] = value
    
    wandb.config.update(formatted_config, allow_val_change=True)


def finish_wandb():
    """Finish wandb run if available."""
    if wandb.run is not None:
        wandb.finish()


# Convenience functions for specific algorithms
def log_sft_metrics(metrics: Dict[str, Any], step: Optional[int] = None, **kwargs):
    """Log SFT-specific metrics."""
    log_metrics(metrics, step=step, algorithm="sft", **kwargs)


def log_ei_metrics(metrics: Dict[str, Any], step: Optional[int] = None, **kwargs):
    """Log EI-specific metrics."""
    log_metrics(metrics, step=step, algorithm="ei", **kwargs)


def log_grpo_metrics(metrics: Dict[str, Any], step: Optional[int] = None, **kwargs):
    """Log GRPO-specific metrics."""
    log_metrics(metrics, step=step, algorithm="grpo", **kwargs)


def log_unified_step(step: int, metrics: Dict[str, Any], algorithm: str = "unknown", **kwargs):
    """
    Unified logging function that logs all metrics together at regular intervals.
    This ensures consistent timing and better wandb visualization.
    """
    if wandb.run is None:
        return
    
    # Add step and algorithm prefix to all metrics
    formatted_metrics = {}
    for key, value in metrics.items():
        if not key.startswith(f"{algorithm}/"):
            formatted_key = f"{algorithm}/{key}"
        else:
            formatted_key = key
        formatted_metrics[formatted_key] = value
    
    # Add step information
    formatted_metrics[f"{algorithm}/step"] = step
    
    wandb.log(formatted_metrics, step=step, **kwargs)


def log_training_batch(step: int, loss: float, nll: float, lr: float, 
                       grad_norm: float, algorithm: str = "unknown", 
                       logging_config: Optional[LoggingConfig] = None, **kwargs):
    """Log training batch metrics with configurable frequency."""
    if logging_config is None:
        logging_config = DEFAULT_LOGGING_CONFIG
    
    # Check if we should log at this step
    if step % logging_config.train_log_every != 0:
        return
    
    # Build metrics dict based on config
    metrics = {}
    if logging_config.log_training_loss:
        metrics["train/loss"] = loss
    if logging_config.log_training_nll:
        metrics["train/nll"] = nll
    if logging_config.log_training_lr:
        metrics["train/learning_rate"] = lr
    if logging_config.log_training_grad_norm:
        metrics["train/gradient_norm"] = grad_norm
    
    if metrics:  # Only log if we have metrics to log
        log_metrics(metrics, step=step, algorithm=algorithm, **kwargs)


def log_evaluation_batch(step: int, eval_metrics: Dict[str, float], 
                        generation_examples: Optional[List[Dict]] = None,
                        algorithm: str = "unknown", 
                        logging_config: Optional[LoggingConfig] = None, **kwargs):
    """
    Log evaluation metrics and optionally generation examples together.
    This ensures they appear at the same step in wandb.
    """
    if logging_config is None:
        logging_config = DEFAULT_LOGGING_CONFIG
    
    # Check if we should log at this step
    if step % logging_config.eval_log_every != 0:
        return
    
    if wandb.run is None:
        return
    
    # Filter eval metrics based on config
    formatted_eval_metrics = {}
    for key, value in eval_metrics.items():
        if key.startswith("reward") and logging_config.log_eval_rewards:
            formatted_eval_metrics[f"{algorithm}/eval/{key}"] = value
        elif key.startswith("accuracy") and logging_config.log_eval_accuracy:
            formatted_eval_metrics[f"{algorithm}/eval/{key}"] = value
        elif key.startswith("length") and logging_config.log_eval_lengths:
            formatted_eval_metrics[f"{algorithm}/eval/{key}"] = value
        else:
            # Log other metrics by default
            formatted_eval_metrics[f"{algorithm}/eval/{key}"] = value
    
    # Log generation examples if enabled and provided
    if logging_config.wandb_log_examples and generation_examples and logging_config.wandb_log_media:
        table = _create_generation_table(generation_examples)
        formatted_eval_metrics[f"{algorithm}/generation_examples"] = table
    
    # Add step information
    formatted_eval_metrics[f"{algorithm}/step"] = step
    
    if formatted_eval_metrics:  # Only log if we have metrics to log
        wandb.log(formatted_eval_metrics, step=step, **kwargs)


def _create_generation_table(examples: List[Dict]) -> wandb.Table:
    """Create wandb table for generation examples."""
    table_data = []
    for i, example in enumerate(examples):
        table_data.append([
            i,
            example["prompt"][:200] + "..." if len(example["prompt"]) > 200 else example["prompt"],
            example["response"][:200] + "..." if len(example["response"]) > 200 else example["response"],
            example.get("ground_truth", "")[:200] + "..." if len(example.get("ground_truth", "")) > 200 else example.get("ground_truth", ""),
            example.get("reward", 0.0),
            example.get("format_reward", 0.0),
            example.get("answer_reward", 0.0),
            example.get("avg_token_entropy", 0.0),
            example.get("response_len_tokens", 0),
        ])
    
    return wandb.Table(
        columns=["Index", "Prompt", "Response", "Ground Truth", "Reward", "Format Reward", "Answer Reward", "Token Entropy", "Response Length"],
        data=table_data
    )
