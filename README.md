# AMR-ECL

Official code for **AMR-ECL: Adaptive Memory Replay for Plasticity-Aware Edge Continual Learning**.

## Introduction
AMR-ECL is an edge continual learning framework that combines:

- **T-DWEM**: dynamic sparse update
- **E-DAQSM**: differentiable affine quantization
- **MF-MRM**: codebook-based feature replay

The goal is to improve continual learning performance under limited memory and computing resources.

## Environment
Recommended:

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- tensorboard
- pytorchcv

Install:
```bash
pip install torch torchvision numpy matplotlib tensorboard pytorchcv
```

## Dataset
This code is mainly organized for the **CORe50** continual learning setting.

Make sure the dataset path and loader are correctly configured in your project.

## Main Files
- `tdm_crumb_chameleon.py`: main training script
- `tdm_crumb_mobilenet.py`: MobileNet-based replay backbone
- `params2_tmd.cfg`: training configuration

## Train
Example:
```bash
python tdm_crumb_chameleon.py --scenario ni --run 1
```

Optional arguments:
```bash
--replay_method er
--memory_blocks 128
--block_size 32
--buffer_weight 0.5
```

Example:
```bash
python tdm_crumb_chameleon.py --scenario ni --run 1 --replay_method er --memory_blocks 128 --block_size 32
```

## Logs
Training logs and results will be saved to:

- `results/`
- `logs/`

TensorBoard:
```bash
tensorboard --logdir logs
```

## Citation
```bibtex
@article{zhang2026amrecl,
  title={Adaptive Memory Replay for Plasticity-Aware Edge Continual Learning},
  author={Peng Zhang and Jing Yang and Yong Yao and Xiaoli Ruan and Yuling Chen and Xu Wang and Yanping Chen},
  year={2026}
}
```
