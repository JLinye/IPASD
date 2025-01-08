# Intra-class Progressive and Adaptive Self-Distillation (IPASD)

## Overview

This repository contains the code for the method proposed in the paper "Intra-class Progressive and Adaptive Self-Distillation" (IPASD). The framework aims to improve self-distillation by progressively distilling knowledge within the same class through intra-class compactness and adaptive mechanisms. It compares student features with teacher prototype features to enhance the performance of student models. The approach uses confidence-based adjustment and class probability distribution integration to fine-tune the learning process across multiple stages.

## Installation

Install the required dependencies:

```
pip install -r requirements.txt
```

### Training

To train a model with IPASD, use the following command:

```
python train.py 
```

## Notice

Our code will be released soon. Please stay tuned for further updates.

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{LinJYPRCV2024,
  title={Self-Distillation via Intra-Class Compactness},
  author={Lin, Jiaye and Li, Lin and Yu, Baosheng and Ou, Weihua and Gou, Jianping},
  booktitle={The 7th Chinese Conference on Pattern Recognition and Computer Vision },
  pages={1--13},
  year={2024},
  organization={Springer}
}
```

