# Unofficial re-implementation of DAM

# introduction

This repository contains reproduction code for DAM method ["Dual Attention Guided Gaze Target Detection in theWild" (CVPR2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Dual_Attention_Guided_Gaze_Target_Detection_in_the_Wild_CVPR_2021_paper.pdf)

# Overview

The whole repository can be divided into three parts:

1. a pre-processing package containing eye extraction and depth estimation.

2. Pre-trained code for the 3D gaze estimation module in the DAM method.

3. Main code for training and evaluating the DAM method.

# Instruction

1.Clone our repo and make directory "datasets".

```bash
git clone https://github.com/nkuhzx/DAM-unoffical
cd DAM-unoffical
mkdir datasets
```

1. Download the GazeFollow dataset and VideoAttentionTarget dataset [refer to ejcgt's repository](https://github.com/ejcgt/attention-target-detection)
to datasets directory.

2. Annotation process