<h3 align="center"><a href="" style="color:#9C276A">
Uni-Sign: Toward Unified Sign Language Understanding at Scale</a></h3>
<h5 align="center"> 
If our project helps you, please give us a star🌟 on GitHub, that would motivate us a lot!
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.15187-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2501.15187) 
[![CSL-Dataset](https://img.shields.io/badge/HuggingFace🤗-%20CSL%20News%20RGB%20Format-blue.svg)](https://huggingface.co/datasets/ZechengLi19/CSL-News)
[![CSL-Dataset](https://img.shields.io/badge/BaiDu☁-%20CSL%20News%20RGB%20Format-green.svg)](https://pan.baidu.com/s/17W6kIreNMHYtD4y2llKmDg?pwd=ncvo)
[![CSL-Dataset](https://img.shields.io/badge/HuggingFace🤗-%20CSL%20News%20Pose%20Format-yellow.svg)](https://huggingface.co/datasets/ZechengLi19/CSL-News_pose)
[![Model-Checkpoints](https://img.shields.io/badge/HuggingFace🤗-%20Model%20Checkpoints-purple.svg)](https://huggingface.co/ZechengLi19/Uni-Sign) 


[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-ms-asl&#41;]&#40;https://paperswithcode.com/sota/sign-language-recognition-on-ms-asl?p=uni-sign-toward-unified-sign-language&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl100&#41;]&#40;https://paperswithcode.com/sota/sign-language-recognition-on-wlasl100?p=uni-sign-toward-unified-sign-language&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl-2000&#41;]&#40;https://paperswithcode.com/sota/sign-language-recognition-on-wlasl-2000?p=uni-sign-toward-unified-sign-language&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-csl-daily&#41;]&#40;https://paperswithcode.com/sota/sign-language-recognition-on-csl-daily?p=uni-sign-toward-unified-sign-language&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-csl&#41;]&#40;https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-csl?p=uni-sign-toward-unified-sign-language&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-2&#41;]&#40;https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-2?p=uni-sign-toward-unified-sign-language&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-3&#41;]&#40;https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-3?p=uni-sign-toward-unified-sign-language&#41;)
</h5>

## 🎥 系统演示 / Demo

**ReliSign** 是一个多模态手语翻译系统（基于 Uni-Sign），支持实时摄像头手语识别、骨架可视化、语音播报、模型归因分析、智能纠错学习与用户历史管理。

> **ReliSign** is a multimodal sign language translation system built on top of Uni-Sign.

![实时演示 Demo](assets/demo.gif)

### 界面截图 / Screenshots

| | |
|---|---|
| ![截图 1](assets/screenshot_1.png) | ![截图 2](assets/screenshot_2.png) |
| ![截图 3](assets/screenshot_3.png) | ![截图 4](assets/screenshot_4.png) |

📹 完整演示视频：[assets/demo.mp4](./assets/demo.mp4)（点击在线播放，或下载后观看）

![Uni-Sign](docs/design/framework.png)

## 💥 News
[2025/1/25] This [paper](https://openreview.net/pdf?id=0Xt7uT04cQ) is accepted by `ICLR 2025` 🎉🎉!

[2025/2/24] Release CSL-News dataset and code implementation.

[2025/6/5] Release lighter pose extraction and online inference implementation. Check it out [here](./demo/README.md) 🎊🎊. 

[2025/12/20] Release pose-only checkpoints and inference code for How2Sign and OpenASL. We hope this will be useful to researchers interested in ASL. Check it out [here](https://huggingface.co/ZechengLi19/Uni-Sign) 👏👏.


## 🛠️ Installation
We suggest to create a new conda environment. 
```bash
# create environment
conda create --name Uni-Sign python=3.9
conda activate Uni-Sign
# install other relevant dependencies
pip install -r requirements.txt
# (Optional) If you want to evaluate BLEURT scores on the How2Sign or OpenASL datasets
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ../
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```

## 📖 Preparation
Please follow the instructions provided in [DATASET.md](./docs/datasets/DATASET.md) for data preparation.

## 💻 Model Checkpoints
The Uni-Sign checkpoints are available for download [Here](https://huggingface.co/ZechengLi19/Uni-Sign/tree/main).

## 🔨 Training & Evaluation
All scripts must be executed within the Uni-Sign directory.

For the original upstream stage scripts and the older generic training entry, see:

- `legacy/script/train_stage1.sh`
- `legacy/script/train_stage2.sh`
- `legacy/script/train_stage3.sh`
- `legacy/script/eval_stage3.sh`

For this local CE-CSL research project, the current active workflow is documented in:

- `docs/01_项目总览/01_项目状态与主线.md`
- `docs/02_训练与实验/01_训练与评估主线.md`
- `docs/02_训练与实验/02_双线实验操作手册.md`

Historical Round6 phase notes were archived under:

- `docs/archive/round6_stage_notes_2026-04-13/`

The current main training entry for this local project is:

```text
local_sign_tools/finetune_cecsl_rgb_support.py
```

The current main evaluation entry is:

```text
script/eval_cecsl_samples.py
```

## 🎨 Inference Results
If you'd like to skip the environment setup and model weight configuration, you can find sample pose-only sign language translation results (stage 2) [Here](./out) for quick comparison.

## 👨‍💻 Todo
- [x] Release CSL-News dataset
- [x] Release Uni-Sign implementation 
- [x] Release pose extraction and online inference implementation

## 📮 Contact
If you have any questions, please feel free to contact Zecheng Li (lizecheng19@gmail.com). Thank you.

## 👍 Acknowledgement
The codebase of Uni-Sign is adapted from [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP), while the implementations of the pose/temporal encoders are derived from [CoSign](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf). We sincerely appreciate the authors of CoSign for personally sharing their code 🙏. \
We are also grateful for the following projects our Uni-Sign arise from:
* 🤟[SSVP-SLT](https://github.com/facebookresearch/ssvp_slt): a excellent sign language translation framework! 
* 🏃️[MMPose](https://github.com/open-mmlab/mmpose): an open-source toolbox for pose estimation.
* 🤠[FUNASR](https://github.com/modelscope/FunASR): a high-performance speech-to-text toolkit.


## 📑 Citation
If you find Uni-Sign useful for your research and applications, please cite using this BibTeX:
```
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```
