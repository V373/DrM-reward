# DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization
<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/abs/2310.19668">[Paper]</a>&emsp;<a href="https://guoweixu.com/drm/">[Project Website]</a>
</p>

This repository is the official PyTorch implementation of **DrM**. **DrM**, a visual reinforcement learning algorithm, minimizes the dormant ratio to guide exploration-exploitation trade-offs and achieves remarkable significant sample efficiency and asymptotic performance in the hardest locomotion and manipulation tasks.
<p align="center">
  <br><img src='images/title.gif' width="500"/><br>
</p>

# 🛠️ Installation Instructions

The supported training environment is `drm-cu128`: Python 3.9, PyTorch
2.8.0 with CUDA 12.8, and NumPy 1.26.4. It requires an NVIDIA driver that
supports CUDA 12.8. On Ubuntu, install the system libraries used by
MuJoCo's off-screen EGL renderer first:

```bash
sudo apt update
sudo apt install libosmesa6-dev libegl1 libgl1 libglfw3
```

Create the training environment, then install the packages bundled with this
repository in editable mode. Run these commands from the repository root:

```bash
conda env create -f conda_env.yml
conda activate drm-cu128
git lfs install
git lfs pull
pip install -e ./metaworld
pip install -e ./rrl-dependencies
pip install -e ./rrl-dependencies/mj_envs
pip install -e ./rrl-dependencies/mjrl
```

The PBRS task configurations require the tracked checkpoints and H5 assets in
`shaped_reward/assets/`; Git LFS downloads them with `git lfs pull`. Sparse
and dense reward runs do not load these assets.

`free-mujoco-py` compiles its extension when it is first imported. For GPU
off-screen rendering, verify that the generated module is the Linux GPU
extension:

```bash
python -c "import mujoco_py; print(mujoco_py.cymj)"
```

If this step cannot initialize EGL, first confirm that the host NVIDIA driver
is visible (for example with `nvidia-smi`). Do not install Conda Mesa/EGL
packages into `drm-cu128`; they can take precedence over the host NVIDIA EGL
implementation and prevent MuJoCo from rendering on the GPU.

### Plotting environment

Evaluation-success plots use a separate, minimal environment so Matplotlib
does not alter the training stack:

```bash
conda env create -f conda_plot_env.yml
conda run -n drm-plot python scripts/plot_metaworld_eval_success.py --downsample 5
```

The plotting script reads selected `exp_local/**/eval.csv` files and writes
figures under `images/metaworld_eval_success/`.

## 💻 Code Usage
If you would like to run DrM on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), please use train_dmc.py to train DrM policies on different configs.

```bash
python train_dmc.py task=dog_walk agent=drm
```

If you would like to run DrM on [MetaWorld](https://meta-world.github.io/), please use train_mw.py to train DrM policies on different configs.

```bash
python train_mw.py task=coffee-push agent=drm_mw
python train_mw.py task=disassemble agent=drm_mw
```

If you would like to run DrM on Adroit, please use train_adroit.py to train DrM policies on different configs.

```bash
python train_adroit.py task=pen agent=drm_adroit
```

## 📝 Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@inproceedings{
drm,
title={DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization},
author={Guowei Xu, Ruijie Zheng, Yongyuan Liang, Xiyao Wang, Zhecheng Yuan, Tianying Ji, Yu Luo, Xiaoyu Liu, Jiaxin Yuan, Pu Hua, Shuzhen Li, Yanjie Ze, Hal Daumé III, Furong Huang, Huazhe Xu.},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=MSe8YFbhUE}
}
```

## 🙏 Acknowledgement
DrM is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the [DrQv2](https://github.com/facebookresearch/drqv2) codebase. Our implementation builds on top of their repository.
