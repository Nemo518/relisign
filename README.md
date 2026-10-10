<h1 align="center">ReliSign</h1>
<h3 align="center">可靠性感知多模态手语翻译系统</h3>
<h5 align="center">Reliability-aware Multimodal Sign Language Translation</h5>
<h5 align="center">如果这个项目对你有帮助，欢迎点一个 star 🌟</h5>

---

## 🧭 项目简介 / What is ReliSign

**ReliSign** 是一套**可靠性感知的多模态手语翻译系统**：围绕「可靠性感知融合」提出三项算法创新，并落地为一套可实时演示、可解释、可边缘部署的完整系统。**原创代码 10,000+ 行**，覆盖算法、训练、后端服务、前端界面与工程部署全链路。

> **ReliSign** is a reliability-aware multimodal sign language translation system with three algorithmic innovations and a full-stack real-time demo.

## ✨ 核心贡献 / Core Contributions

| 创新点 | 说明 | 代码位置 |
|---|---|---|
| **不确定性感知融合门控**<br>Uncertainty-Aware Fusion Gate | 由「手部置信度 + 选择密度 + 手部存在率」组成的 3D 元数据驱动 `Conv1D→GELU→Conv1D+Sigmoid` 门控，动态调节 RGB 贡献权重，解决遮挡 / 光照 / 运动模糊下盲目融合的噪声问题 | `models.py` |
| **PGF 姿态引导融合**<br>Pose-Guided Fusion | 以归一化手部关键点坐标为几何先验，驱动 `DeformableAttention2D` 在 RGB 特征图上采样相关区域，显式对齐姿态与 RGB | `models.py` · `deformable_attention_2d.py` |
| **置信度引导手部帧选择**<br>Confidence-Guided Hand Frame Selection | 4 种策略（confidence_topk / confidence_weighted / uniform / random）从长视频中选取高质量手部帧 | `datasets.py` |

**系统工程**：FastAPI 后端（16 端点 + XAI 可解释性）+ Vue3 前端 + 实时摄像头演示 + 模型归因分析 + 智能纠错学习 + 用户历史管理 + 边缘部署（Docker / ONNX）。

## 🎥 系统演示 / Demo

![实时演示 Demo](assets/demo.gif)

### 界面截图 / Screenshots

| | |
|---|---|
| ![截图 1](assets/screenshot_1.png) | ![截图 2](assets/screenshot_2.png) |
| ![截图 3](assets/screenshot_3.png) | ![截图 4](assets/screenshot_4.png) |

📹 完整演示视频：[assets/demo.mp4](./assets/demo.mp4)（点击在线播放，或下载后观看）

## 🏗️ 整体框架 / Framework

![ReliSign 框架](docs/design/relisign_framework.png)

## 🧩 系统功能 / Features

| 功能 | 说明 |
|---|---|
| 🎥 实时演示 | 摄像头手语识别、骨架可视化、语音播报 |
| 🖼️ 模型展示 | 高精度样本播放与在线推理 |
| 🔍 模型归因 | 批量样本的参考 / 预测 / 相似度对比（XAI 可解释性） |
| 📚 智能学习 | 识别纠错记忆 |
| 👤 用户中心 | 登录注册、历史记录 |

## 🚀 快速开始 / Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端（默认端口 8001）
python view/web_app/app.py

# 3. 浏览器访问
# http://localhost:8001
```

> 前端源码位于 `view/web-frontend/`（Vue3），修改后需重新构建：
> ```bash
> cd view/web-frontend && npm run build && cp -r dist/* ../web_app/vue-dist/
> ```

## ⚙️ 技术栈 / Tech Stack

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + WebSocket（16 端点） |
| 前端 | Vue3 |
| 深度学习 | PyTorch · ONNX Runtime (GPU / TensorRT) |
| 姿态估计 | PoseTracker / rtmlib（Wholebody） |
| 文本模型 | mT5 |
| 语音 | edge-tts + pygame（语音播报） |
| 部署 | Docker · ONNX 边缘部署 |

## 🧠 系统工程 / Engineering

- **显存自动管理**：切换页面时自动释放 / 恢复 GPU 模型，姿态估计器按显存余量智能选择 CPU / GPU，避免 OOM。
- **异步姿态管线**：独立线程提取姿态 + 异步发送，非阻塞架构，零延迟积压。
- **可解释性**：模型归因分析 + XAI 服务。

## 📂 项目结构 / Structure

```
view/
├── web_app/              ★ 现行主应用（FastAPI + WebSocket）
│   ├── app.py                后端服务主程序
│   └── vue-dist/             前端构建产物
├── web-frontend/         Vue3 前端源码
├── archive/              已归档的旧 Streamlit 系统
├── inference_adapter.py  在线推理适配器
├── xai_service.py        XAI 可解释性分析
├── report_service.py     归因报告服务
├── data_service.py       数据处理服务
└── bootstrap.py          路径初始化
```

## 🙏 致谢 / Acknowledgements

本系统的基础骨架与数据集继承自 [Uni-Sign: Toward Unified Sign Language Understanding at Scale](https://arxiv.org/abs/2501.15187)（ICLR 2025）——**ST-GCN 姿态编码器、mT5 解码器、CSL-News 数据集**；Uni-Sign 又受益于 [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP) 与 [CoSign](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf)。若使用了 Uni-Sign 基础部分，请引用：

```
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```

---

🤖 Built with the assistance of [Claude Code](https://claude.com/claude-code).
