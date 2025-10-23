# STDDNet
Official PyTorch implementation of "STDDNet: Harnessing Mamba for Video Polyp Segmentation via Spatial-aligned Temporal Modeling and Discriminative Dynamic Representation Learning".

# Getting Start
## Installation
**1. Envs: python=3.10.13 and CUDA=11.8**
```
conda create -n stddnet python=3.10.13
conda activate stddnet
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.3 timm==0.4.12 einops==0.7.0 packaging==23.2 tqdm
pip install -U scikit-learn
conda install -c conda-forge opencv
```
**2. Requirements (if necessary)**

Please refer to [Vim](https://github.com/hustvl/Vim) to download vim_requirements.txt
```
pip install -r your_path/vim_requirements.txt
```
**3. Install Causal_conv1d and Mamba_ssm (Two options)**

A. Following the setup of [Vim](https://github.com/hustvl/Vim)
```
  pip install -e causal_conv1d>=1.1.0
  pip install -e mamba-1p1p1
```
B. Download and Install **causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl** and **mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl**
```
Downloads:
https://github.com/Dao-AILab/causal-conv1d/releases?page=2
https://github.com/state-spaces/mamba/releases?page=3

Installation:
pip install your_path/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install your_path/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
**Then,**

a. **replace the compiled file *mamba_simple.py* (at /your_home_path/anaconda3/envs/stddnet/lib/python3.10/site-packages/mamba_ssm/modules/)**

b. **replace the compiled file *selective_scan_interface.py* (at /your_home_path/anaconda3/envs/stddnet/lib/python3.10/site-packages/mamba_ssm/ops/)**

## Training and Testing
```python
//Train:
python /your_project_path/scripts/my_train.py
//Test:
python /your_project_path/scripts/my_test.py
```
# Acknowledgement
Our work builds upon the excellent foundational research of [PNS+](https://github.com/GewelsJI/VPS) and [Vim](https://github.com/hustvl/Vim). We thank the authors for their awesome works and publicly available codes.

# Citation
if you find our work useful, please cite:
```
To be updated..
```
