# Adaptive Re-calibration Learning for Balanced Multimodal Intention Recognition

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116409)

> **Adaptive Re-calibration Learning for Balanced Multimodal Intention Recognition**  
> Qu Yang, Xiyang Li, Fu Lin, Mang Ye [[Link]](https://neurips.cc/virtual/2025/loc/san-diego/poster/116409)

<div align="center">
  <img src="image/arl.png" alt="ARL Framework" width="800"/>
</div>

## Overview

This repository provides the implementation of Adaptive Re-calibration Learning (ARL), a novel approach for balanced multimodal intention recognition. 

## Quick Start

```bash
# Create and activate conda environment
conda create --name arl python=3.6
conda activate arl 

# Install PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Clone the repository
git clone git@github.com:yan9qu/NeurIPS25-ARL.git
cd NeurIPS25-ARL

# Install dependencies
pip install -r requirements.txt

# Run the code
python run.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{yang2025adaptive,
  title={Adaptive Re-calibration Learning for Balanced Multimodal Intention Recognition},
  author={Yang, Qu and Li, Xiyang and Lin, Fu and Ye, Mang},
  booktitle={Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements

This codebase is built upon [MIntRec](https://github.com/thuiar/MIntRec). We thank the authors for their excellent work and open-source contribution.

## License

This project is licensed under the MIT License.

