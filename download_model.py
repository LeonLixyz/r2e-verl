import os
os.environ["HF_HOME"] = "/workspace"

from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-32B",
    local_dir="/workspace/relace-verl/models/qwen3-32b",
    resume_download=True
)
