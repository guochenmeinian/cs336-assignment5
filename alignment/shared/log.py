from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GenLogConfig:
    every_steps: int = 500            # 每多少 step 触发一次
    max_new_tokens: int = 128
    batch_size: int = 8
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    sample_k: int = 64                # 每次日志抽样多少条验证样本
    device: Optional[torch.device] = None
    to_wandb: bool = False            # 可选：上传到 W&B

