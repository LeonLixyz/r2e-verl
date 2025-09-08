### Installation

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
 ~/miniconda3/bin/conda init bash
source ~/.bashrc
```

```
conda create -n verl python==3.10
conda activate verl
cd relace-verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install aiodns
apt-get update && apt-get install -y tmux
```