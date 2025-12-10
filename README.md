# ECP-Mamba
ECP-Mamba: An Efficient Multi-scale Self-supervised Contrastive Learning Method 
with State Space Model for PolSAR Image Classification

## UnZip 
Unzip ECP-Mamba.zip first

## Installation
We use cuda 11.8 for our codes. Use the following list of commands to install required libraries. 

```commandline
conda create -n your_env_name python=3.10.13
conda activate your_env_name
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install causal-conv1d==1.1.1 
pip install mamba-ssm==1.1.3.post1
pip install wandb
pip install matplotlib
pip install h5py
pip install rgb2hex
pip install colormap
pip install scipy
pip install timm
pip install einops
pip install pandas
pip install seaborn
pip install numpy==1.26.0
```

## Pre-training
You should use the codes folder for pretraining. 

```python
python pretrain.py
```


## Fine-tuning 
You should use the codes folder for pretraining. 
```python
python finetune.py
```

## Contact

If you have any question, please email `kzz794466014@stu.xjtu.edu.cn`

## Citation

If you find this repository useful, please consider giving a star and citation:
```
@article{kuang2025ecp,
  title={Ecp-mamba: An efficient multi-scale self-supervised contrastive learning method with state space model for polsar image classification},
  author={Kuang, Zuzheng and Bi, Haixia and Li, Fan and Xu, Chen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```

## Abstract

Recently, polarimetric synthetic aperture radar (PolSAR) image classification has been greatly promoted by deep neural networks. However, current deep learning (DL)-based PolSAR image classification methods are caught in the dilemma of obtaining high accuracy with sparse labels while maintaining high computational efficiency. To solve this issue, we present ECP-Mamba, an efficient framework integrating multiscale self-supervised contrastive learning (CL) with a state space model (SSM) backbone. Specifically, we design a cross-scale predictive pretext task, which learns representations via aligning local and global polarimetric features, effectively mitigating the annotation scarcity issue. To enhance computational efficiency, we introduce Mamba architecture to PolSAR image classification for the first time. A spiral scanning strategy tailored for pixelwise classification task is proposed within this framework, prioritizing causally relevant features near the central pixel. Additionally, a lightweight cross Mamba module is proposed to facilitate complementary multiscale feature interaction. Extensive experiments on four benchmark datasets demonstrate the effectiveness of ECP-Mamba in balancing high accuracy with computational efficiency. On the Flevoland 1989 dataset, ECP-Mamba achieves state-of-the-art performance with an overall accuracy (OA) of 99.70%, an average accuracy (AA) of 99.64%, and a Kappa coefficient (Kappa) of 0.9962. 
