# Physics Inspired Hybrid Attention for SAR Target Recognition

## 0. Table of Contents

* [Introduction](#1-introduction)
* [Features](#2-features) 
* [Contributions](#3-contributions) 
* [Getting Started](#4-getting-started)
* [Contributors](#5-contributors)

## 1. Introduction

This is the official implementation for paper "Physics Inspired Hybrid Attention for SAR Target Recognition". 

DOI: 10.1016/j.isprsjprs.2023.12.004

Full paper access: [Arxiv](https://arxiv.org/abs/2309.15697) [ResearchGate](https://www.researchgate.net/publication/376832791_Physics_inspired_hybrid_attention_for_SAR_target_recognition)

```
@article{huang2024physics,
  title={Physics inspired hybrid attention for SAR target recognition},
  author={Huang, Zhongling and Wu, Chong and Yao, Xiwen and Zhao, Zhicheng and Huang, Xiankai and Han, Junwei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={207},
  pages={164--174},
  year={2024},
  publisher={Elsevier}
}
```

## 2. Features

<div align=center>
<img src="https://github.com/XAI4SAR/PIHA/blob/main/img/network.png">
</div>

## 3. Contributions
-   A novel physics-inspired hybrid attention (PIHA) mechanism is proposed for SAR target recognition, in which the semantic prior of physical information is adaptively incorporated with the attention mechanism. It is flexible for different types of physical information and can be incorporated into various deep architectures to enhance performance.
    
-   We propose the once-for-all (OFA) evaluation protocol for MSTAR dataset to thoroughly assess the algorithm, demonstrating the robustness and generalization capabilities more effectively.

-   The physical information of SAR targets used in this study together with the source code are open to public, which ensures the reproducibility of our work and facilitates a fair comparison of the results with other methodologies.


## 4. Getting Started

### 4.1 Data Preparation

The experimented dataset is MSTAR. We extracted the Attributed Scattering Centers (ASC) for SAR targets and processed the targets into K parts with local semantics. The ASC data can be downloaded at:

https://drive.google.com/file/d/1OqdgOodVVAJclnjSH06B4tvVn9F1C4Ns/view?usp=sharing

### 4.2 Training

To train a PIHA based model, run the following command

```
python submitit_pretrain.py \
    --datatxt_train ${TRAIN_LIST_PATH} \
    --datatxt_OFA1 ${OFA1_LIST_PATH} \
    --datatxt_OFA2 ${OFA2_LIST_PATH} \
    --datatxt_OFA3 ${OFA3_LIST_PATH} \
    --datatxt_val ${VAL_LIST_PATH} \
    --train_num 5 \
    --num_epochs 1000 \
    --patience 200 \
    --batch_size 32 \
    --device '0' \
    --arch 'Densenet121_PIHA' \
    --cate_num 10 \
    --part_num 4 \
    --attention_setting True \
    --save_path ${SAVE_PATH} \
    --pretrain None \
```
-   Here the datatxt_train, datatxt_OFA1, datatxt_OFA2, datatxt_OFA3, datatxt_val are the path of data list which are provided in the above link. 
-   Train_num is the number of training process to ensure the stability of result. 
-   Patience is the parameter of earlystop strategy to stop training when accuracy of validation set does not improve. 
-   Arch is the type of backbone which can be selected during Densenet121_PIHA, Aconvnet_PIHA and MSNet_PIHA.
-   Part_num is the numbers of clusters in data preparation and part_num of our data is 4.
-   Attention_setting decide whether to use our PIHA.

## 5. Contributors

In this repository, the applied backbones are based on [DenseNet121](https://github.com/liuzhuang13/DenseNet), [MS-CVNets](https://github.com/Crush0416/MS-CVNets-a-novel-complex-valued-neural-networks-for-SAR-ATR) and [A_ConvNet](https://github.com/fudanxu/MSTAR-AConvNet). We deeply appreciate the authors for releasing their codes.

Main contributors: [@nwpuwwc](https://github.com/orgs/XAI4SAR/people/nwpuwwc) [@Alien9427](https://github.com/orgs/XAI4SAR/people/Alien9427)

Maintenance: [@Guozi2002](https://github.com/orgs/XAI4SAR/people/Guozi2002)

If you have any questions, please contact huangzhongling@nwpu.edu.cn

