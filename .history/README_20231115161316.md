# PIHA: Physics Inspired Hybrid Attention

## 0. Table of Contents

* [Introduction](#1-introduction)
    * [Features](#11-features) 
    * [Contributions](#12-contributions) 
* [Getting Started](#2-getting-started)
* [Contributors](#3-contributors)
* [Citation](#4-citation)
* [References](#5-References)

## 1. Introduction

This project is for paper "Physics Inspired Hybrid Attention for SAR Target Recognition".

### 1.1 Features

<div align=center>
<img src="https://github.com/XAI4SAR/PIHA/blob/main/img/network.png">
</div>

### 1.2 Contributions
-   On the basis of knowledge-guided model architecture design, a novel physics-inspired hybrid attention (PIHA) mechanism is proposed for SAR target recognition, in which the semantic prior of physical information is adaptively incorporated with the attention mechanism. It is flexible for different types of physical information and can be incorporated into various deep architectures to enhance performance.
    
-   We propose the once-for-all (OFA) evaluation protocol to thoroughly assess the algorithm, demonstrating the robustness and generalization capabilities more effectively.

-   This study delves into the comprehensive examination of the effects of data-driven and physics-driven attentions, offering valuable insights that can serve as a source of inspiration for design concepts.

-   The physical information of SAR targets used in this study together with the source code are open to public, which ensures the reproducibility of our work and facilitates a fair comparison of the results with other methodologies.
## 2. Getting Started
### 2.1 Data Preparation
The dataset we adopt is MSTAR. We extract the ASC centers for target and adopt the k-means algorithm to divide the targets into four parts. The data can be downloaded at following link:
https://drive.google.com/file/d/1OqdgOodVVAJclnjSH06B4tvVn9F1C4Ns/view?usp=sharing

### 2.2 Training
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
## 3. Contributors
In this repository, our work are based on DenseNet121, CV-MSNET and Aconvnet. We carry out our experiment on MSTAR dataset. Thanks for all the above works' contribution.


## 4. Citation

If you find this repository useful for your publications, please consider citing our paper.

