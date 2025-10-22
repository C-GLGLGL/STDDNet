# STDDNet
Official PyTorch implementation of "STDDNet".

# Getting Start
## Installation
**1. Envs: python=3.10.13 and CUDA=11.8**
```python
conda create -n stddnet python=3.10.13
conda activate stddnet
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.3 timm==0.4.12 einops==0.7.0 packaging==23.2 tqdm
pip install -U scikit-learn
conda install -c conda-forge opencv
```
**2. Requirements (if necessary)**

Please refer to [github-url]: https://github.com/hustvl/Vim to download vim_requirements.txt
```python
pip install -r your_path/vim_requirements.txt
```
**3. Install Causal_conv1d and Mamba_ssm**

A. Following the setup of [github-url]: https://github.com/hustvl/Vim
```python
  pip install -e causal_conv1d>=1.1.0
  pip install -e mamba-1p1p1
```
B. Download and Install **causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl** and **mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl**
```python
Downloads:
https://github.com/Dao-AILab/causal-conv1d/releases?page=2
https://github.com/state-spaces/mamba/releases?page=3
Install:
pip install your_path/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install your_path/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
**Then,**

a. **replace the *mamba_simple.py* at /your_home_path/anaconda3/envs/stddnet/lib/python3.10/site-packages/mamba_ssm/modules/**
b. **replace the *selective_scan_interface.py* at /your_home_path/anaconda3/envs/stddnet/lib/python3.10/site-packages/mamba_ssm/ops/**

**4. Training and Testing**
```python
//Train:
python /your_project_path/STDDNet/scripts/my_train.py
//Test:
python /your_project_path/STDDNet/scripts/my_test.py
```
