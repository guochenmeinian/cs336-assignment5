# alignment/math_sft/trainer.py
from __future__ import annotations
import os, sys, random, os
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 复用 tests/adapters.py ---
from tests.adapters import (  # type: ignore
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_get_response_log_probs as get_response_log_probs,
    run_sft_microbatch_train_step as sft_microbatch_train_step,
)

# Import shared utilities
from ..shared.wandb_config import (
    WandbConfig, init_wandb, log_training_batch, log_evaluation_batch, 
    log_model_config, finish_wandb
)
from ..shared.config import (
    WANDB_PROJECT, WANDB_NAME, WANDB_TAGS, WANDB_ENABLED,
    MODEL_CONFIGS, SFT_DEFAULT_CONFIG
)

# 可选：按你给的 shared/log.py 使用配置
try:
    from ..shared.log import GenLogConfig  # type: ignore
except Exception:
    from dataclasses import dataclass
    @dataclass
    class GenLogConfig:
        every_steps: int = 500
        max_new_tokens: int = 128
        batch_size: int = 8
        do_sample: bool = False
        temperature: float = 0.0
        top_p: float = 1.0
        sample_k: int = 64
        device: Optional[torch.device] = None
        to_wandb: bool = False

def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_pad(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

@dataclass
class SFTConfig:
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    lr: float = 2e-5
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 2000
    max_grad_norm: float = 1.0
    bf16: bool = True
    amp: bool = False
    device: str = "cuda:0"
    seed: int = 42
    
    # Wandb configuration
    wandb_enabled: bool = True
    wandb_project: str = "cs336-sft"
    wandb_name: Optional[str] = None
    wandb_tags: Optional[list] = None

class SFTTrainer:
    """
    纯训练器：不关心数据读写和评估，提供：
      - train_one_like(rows, on_step_log=...)
      - step_loggable_metrics()
      - save(out_dir)
    """
    def __init__(self, cfg: SFTConfig):
        self.cfg = cfg
        self.step = 0
        self._last_nll = 0.0
        
        # Get model configuration from shared config
        model_key = getattr(cfg, 'model_id', 'qwen_math_15b')
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
        
        model_config = MODEL_CONFIGS[model_key]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
        # Fix padding for Flash Attention compatibility
        self.tokenizer.padding_side = 'left'
        
        # Initialize model with proper dtype and move to GPU first
        if cfg.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif cfg.torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        model_kwargs = {"torch_dtype": dtype}
        
        # Load model on CPU first
        self.model = AutoModelForCausalLM.from_pretrained(model_config["path"], **model_kwargs)
        
        # Move to GPU first, then enable Flash Attention
        self.model = self.model.to(cfg.device)
        
        # Enable Flash Attention after moving to GPU
        if cfg.use_flash_attention:
            self.model.config.attn_implementation = "flash_attention_2"
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Initialize optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        
        # Initialize gradient scaler for mixed precision
        self._scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        
        # Initialize wandb if enabled
        if cfg.wandb_enabled:
            wandb_config = WandbConfig(
                project=cfg.wandb_project or WANDB_PROJECT,
                name=cfg.wandb_name or WANDB_NAME,
                tags=cfg.wandb_tags or WANDB_TAGS,
                algorithm="sft",
                dataset="gsm8k",
                model_name="qwen2.5-math-1.5b",
                enabled=cfg.wandb_enabled
            )
            init_wandb(wandb_config)
            
            # Log model configuration
            config_dict = {
                "model_id": cfg.model_id,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum,
                "max_steps": cfg.max_steps,
                "use_flash_attention": cfg.use_flash_attention,
                "torch_dtype": cfg.torch_dtype,
                "device": cfg.device,
            }
            log_model_config(config_dict, algorithm="sft")

    def iterate_batches(self, rows:List[Dict[str,str]], shuffle:bool, seed:int):
        idx = list(range(len(rows)))
        if shuffle:
            rng = random.Random(seed); rng.shuffle(idx)
        bs = self.cfg.batch_size
        for s in range(0, len(idx), bs):
            part = [rows[i] for i in idx[s:s+bs]]
            yield [r["prompt"] for r in part], [r["response"] for r in part]

    def train_one_like(
        self,
        rows: List[Dict[str,str]],
        *,
        log_every: int,
        on_step_log: Optional[Callable[[int, Dict[str,float]], None]] = None,
    ):
        """跑到 max_steps 为止"""
        # Clear GPU cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        while self.step < self.cfg.max_steps:
            for prompts, gts in self.iterate_batches(rows, shuffle=True, seed=self.cfg.seed + self.step):
                if self.step >= self.cfg.max_steps:
                    break

                pack = tokenize_prompt_and_output(prompts, gts, self.tokenizer)
                input_ids = pack["input_ids"].to(self.cfg.device)
                labels    = pack["labels"].to(self.cfg.device)
                rmask     = pack["response_mask"].to(self.cfg.device)

                # Forward pass with FlashAttention-2
                with torch.cuda.amp.autocast(enabled=self.cfg.amp, dtype=torch.float16):
                    out = get_response_log_probs(self.model, input_ids, labels, return_token_entropy=False)
                    logp = out["log_probs"]
                    loss, meta = sft_microbatch_train_step(
                        policy_log_probs=logp,
                        response_mask=rmask,
                        gradient_accumulation_steps=self.cfg.grad_accum,
                        normalize_constant=1.0,
                    )

                # Optimized gradient accumulation: update every grad_accum steps
                if (self.step + 1) % self.cfg.grad_accum == 0:
                    # Clip gradients for stability
                    clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    # Take optimizer step
                    self.optim.step()
                    # Zero gradients for next accumulation cycle
                    self.optim.zero_grad(set_to_none=True)

                # 记录 NLL 方便外层 log
                nll = (meta["masked_sum"] / meta["num_response_tokens"].clamp_min(1)).item()
                self._last_nll = float(nll)
                
                # Log to wandb every few steps (not every step to avoid spam)
                if self.cfg.wandb_enabled and (self.step % 10 == 0):
                    grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
                    log_training_batch(
                        step=self.step,
                        loss=loss.item(),
                        nll=self._last_nll,
                        lr=self.optim.param_groups[0]["lr"],
                        grad_norm=grad_norm,
                        algorithm="sft"
                    )

                if on_step_log and (self.step % log_every == 0):
                    on_step_log(self.step, {"train/nll": self._last_nll})

                self.step += 1
                if self.step >= self.cfg.max_steps:
                    break

    def step_loggable_metrics(self) -> Dict[str, float]:
        return {"train/nll": self._last_nll if self._last_nll is not None else float("nan")}

    def save(self, out_dir: str):
        """Save the trained model and tokenizer."""
        os.makedirs(out_dir, exist_ok=True)
        
        # Save model weights with FlashAttention-2 support
        self.model.save_pretrained(
            save_directory=out_dir,
            safe_serialization=True,  # Use safetensors for better compatibility
            max_shard_size="2GB"      # Split large models into smaller shards
        )
        
        # Save tokenizer (even if not modified) for self-contained model
        self.tokenizer.save_pretrained(save_directory=out_dir)
        
        # Save training configuration
        import json
        model_config = MODEL_CONFIGS[self.cfg.model_id]
        config_dict = {
            "model_config": model_config,  # 使用shared config中的模型配置
            "training_config": {
                "lr": self.cfg.lr,
                "batch_size": self.cfg.batch_size,
                "grad_accum": self.cfg.grad_accum,
                "max_steps": self.cfg.max_steps,
                "max_grad_norm": self.cfg.max_grad_norm,
                "bf16": self.cfg.bf16,
                "use_flash_attention": self.cfg.use_flash_attention,
                "torch_dtype": self.cfg.torch_dtype,
            },
            "final_step": self.step,
            "final_nll": self._last_nll,
        }
        
        with open(os.path.join(out_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Finish wandb run if enabled
        if self.cfg.wandb_enabled:
            finish_wandb()
            
        print(f"Model saved to {out_dir}")
        print(f"Training config saved to {out_dir}/training_config.json")
