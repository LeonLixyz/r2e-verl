from huggingface_hub import snapshot_download
from tqdm import tqdm

# Automatically uses tqdm
snapshot_download(repo_id="relace/qwen-rl-ckpt20-training", repo_type="model")
