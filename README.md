# Unofficial re-implementation of DAM

## introduction

This repository contains reproduction code for DAM method ["Dual Attention Guided Gaze Target Detection in theWild" (CVPR2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Dual_Attention_Guided_Gaze_Target_Detection_in_the_Wild_CVPR_2021_paper.pdf)

## Overview

The whole repository can be divided into three parts:

1. a pre-processing package containing eye extraction and depth estimation.

2. Pre-trained code for the 3D gaze estimation module in the DAM method.

3. Main code for training and evaluating the DAM method.

## Prerequisites
- Python>=3.8.0
- Pytorch>=1.7.1
- face-alignment>=1.3.5
- opencv3>= 3.1.0


## Generate the pre-trained model (optional)

1.Download the [Gaze360 dataset](http://gaze360.csail.mit.edu/download.php) to datasets directory.

2.Obtain the annotation files

```bash
cd gazeestimation/tools
python datasplit.py
```

3.Run the training procedure of 3D gazeestimation

```bash
cd gazeestimation
python main.py
```

4.Generate the pre-trained model weight from the saved checkpoint

```bash
cd gazeestimation/tools
python savepretrained.py
```


## Instruction

1.Clone our repo and make directory "datasets".

```bash
git clone https://github.com/nkuhzx/DAM-unoffical
cd DAM-unoffical
mkdir datasets
```

2.Download the GazeFollow dataset and VideoAttentionTarget dataset [refer to ejcgt's repository](https://github.com/ejcgt/attention-target-detection)
to datasets directory.

3.Generate the annotation files and preprocess files

```bash
cd preprocess
python data_process.py
```

4.Download the pre-trained [model weight](https://drive.google.com/file/d/14tCl7yUltHvWs2aC-ZOkjnVFzB9M3lEB/view?usp=sharing) to directory "dammethod/modelpara" or follow the [Train the pre-trained model](#_19) to obtain the pre-trained model weight

5.Run the training procedure on GazeFollow dataset.

```bash
cd dammethod
python main.py --is_train
```

6.After 5, Run the training procedure on VideoAttentionTarget dataset, you need to specify the "PATH" to
model weight trained on the GazeFollow dataset .
```bash
cd dammethod
python main_videoatt.py ----init_model PATH --is_train
```

## Evaluation
### Evaluation on GazeFollow dataset
1.Set up the "dammethod/config/gazefollow_cfg.yaml" as follow, "PATH" is the path to model weights
```yaml
resume: True
resume_add: "PATH"
```
2.Run the evaluation procedure.
```bash
cd dammethod
python main.py --is_test
```
### Evaluation on VideoAttentionTarget dataset
1.Set up the "dammethod/config/gazefollow_cfg.yaml" as follow, "PATH" is the path to model weights
```yaml
resume: True
resume_add: "PATH"
```
2.Run the evaluation procedure.
```bash
cd dammethod
python main_videoatt.py --is_test
```
### Evaluation without training
We also provide the model weights for evaluation
1. [model weight for GazeFollow](https://drive.google.com/file/d/1CUejYe6JA7xOyv_ZZM_c-gB-O2iQu49q/view?usp=sharing)
2. [model_weight for VideoAttentionTarget]()

