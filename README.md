# 使用双模态信息的视觉语言模型置信度后验校准方法vi
vi是visual informed缩写，是一种结合视觉模态信息的后验校准方法，可提高模型的可靠性。
vi方法中使用的温度缩放系数为$\frac{\exp(-\text{dist}(当前文本特征-文本特征中心))}{\exp(-\text{dist}(当前图像特征-图像特征中心))}$

## Setup

**1. Installation** 

For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md).

**2. Data preparation**

Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

<!-- **3. Checkpoints**

For CLIP models, our reported results are based on [checkpoints](clip/clip.py) provided OpenAI. For our main results in Table 2, the checkpoint is available [here](https://arxiv.org/abs/2109.01134).
 -->


## Quick Start

Please refer to ``./run`` for more info about our scripts. 

**1. Tuning & Evaluation** 

```bash
GPU_ID=1 # replace it with your GPU ID
bash run/classification/zeroshot.sh ${GPU_ID} # zero-shot CLIP
bash run/classification/fewshot.sh ${GPU_ID} # fine-tuned CLIP
```


**2. 如果你只想复现DAC+vi的结果** 

```bash
GPU_ID=1
bash run/calibration/test_visual_informed.sh ${GPU_ID}
```

The results will be logged in ``output/base2new/logs_base2new.csv``. 
<!-- Furthermore, we provide a guideline in [RUN.md](docs/RUN.md) for detailed instructions about our repo. -->

## Acknowledgements

My code is inspired by [CoOp](https://github.com/KaiyangZhou/CoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [DAC](https://github.com/ml-stat-Sustech/CLIP_Calibration). I thank the authors for releasing their code.
