# download_models.py
from huggingface_hub import snapshot_download
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / "models"

# Create models directory if it doesn't exist
models_dir.mkdir(exist_ok=True)

# Qwen 2.5 Math 1.5B
snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_dir=str(models_dir / "Qwen2.5-Math-1.5B"),
    local_dir_use_symlinks=False,
    resume_download=True,
)

# Llama 3.1 8B Base
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B",
    local_dir=str(models_dir / "Llama-3.1-8B"),
    local_dir_use_symlinks=False,
    resume_download=True,
)

# Llama 3.3 70B Instruct
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
    local_dir=str(models_dir / "Llama-3.3-70B-Instruct"),
    local_dir_use_symlinks=False,
    resume_download=True,
)
