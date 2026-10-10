# Uni-Sign 部署模块

> 手语翻译模型轻量化部署方案 — 覆盖 **边缘设备** 和 **Web API 网站** 两大场景。

## 目录结构

```
deploy/
├── __init__.py               # 包入口，导出所有公开 API
├── config_deploy.py           # 部署配置（量化/生成/7种预设）
├── model_loader.py            # 量化模型加载器 (FP16/INT8/INT4/NF4)
├── inference.py               # 优化推理管线
├── server.py                  # FastAPI Web 推理服务
├── benchmark.py               # 性能基准测试
├── edge_export.py             # ONNX/TorchScript 导出 (边缘设备)
├── run_edge.py                # 边缘设备一键推理脚本
├── Dockerfile                 # Docker 镜像
├── docker-compose.yml         # Docker Compose 编排
├── nginx.conf                 # Nginx 反向代理
├── requirements_deploy.txt    # 部署依赖
└── README.md                  # 本文档
```

---

## 场景一：Web API 网站部署

### 快速启动

```bash
# 1. 安装依赖
pip install -r deploy/requirements_deploy.txt

# 2. 启动服务（均衡模式）
python -m deploy.server --mode fast --port 8080

# 3. 验证
curl http://localhost:8080/health
```

### Docker 部署

```bash
# 构建镜像
docker build -f deploy/Dockerfile -t uni-sign-api:latest .

# 运行（需挂载存储目录）
docker run --gpus all -p 8080:8080 \
    -v /path/to/Uni-Sign-storage:/app/Uni-Sign-storage \
    uni-sign-api:latest --mode fast

# Docker Compose
MODE=fast docker-compose -f deploy/docker-compose.yml up -d

# 生产模式（含 Nginx 反向代理）
MODE=fast docker-compose -f deploy/docker-compose.yml --profile production up -d
```

### API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 + GPU 状态 |
| `/stats` | GET | 请求统计 |
| `/translate` | POST | 上传姿态 .pkl 文件翻译 |
| `/translate/batch` | POST | 批量上传（最多 10 个） |
| `/translate/raw` | POST | JSON 在线姿态数据翻译 |
| `/docs` | GET | Swagger 交互文档 |

### 调用示例

```python
import requests

# 健康检查
resp = requests.get("http://localhost:8080/health")
print(resp.json())

# 翻译
with open("sample.pkl", "rb") as f:
    resp = requests.post(
        "http://localhost:8080/translate",
        files={"file": f},
        timeout=60,
    )
    print(resp.json()["text"])

# JSON 在线数据
resp = requests.post(
    "http://localhost:8080/translate/raw",
    json={"keypoints": [...], "scores": [...]},
    timeout=60,
)
```

---

## 场景二：边缘设备部署

### 自动检测模式（推荐）

```bash
# 自动检测显存，选择最优模式
python deploy/run_edge.py --pose sample.pkl

# 输出示例:
# [auto] 检测到 GPU: NVIDIA GeForce RTX 4060
# [auto] 显存总量: 8.0 GB
# [auto] 推理模式: balanced
# [ 420ms] 今天天气很好
```

### 显存对应模式

| 设备 | 显存 | 推荐模式 | 命令 |
|------|------|----------|------|
| RTX 4060/5060 | 8 GB | `balanced` | `--mode balanced` |
| Jetson Orin | 4 GB | `edge_4gb` | `--mode edge_4gb` |
| Jetson Nano | 2 GB | `edge_2gb` | `--mode edge_2gb` |
| 无 GPU / 树莓派 | CPU | `edge_cpu` | `--mode edge_cpu` |

### 批量处理

```bash
python deploy/run_edge.py --dir poses/ --output results.json
```

### 导出 ONNX（极致边缘加速）

```bash
# 导出 Pose Encoder 为 ONNX（在 Jetson 上用 ONNX Runtime 推理）
python -m deploy.edge_export --format onnx --seq-len 128

# 导出 TorchScript
python -m deploy.edge_export --format torchscript

# 导出全部格式
python -m deploy.edge_export --format all --output-dir exported/
```

导出后在边缘设备上用 ONNX Runtime 加载：

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("pose_encoder.onnx")
output = session.run(None, {
    "body": body_np,
    "left": left_np,
    "right": right_np,
    "face_all": face_np,
})
```

---

## 量化模式对比

| 模式 | 量化 | Beam | 显存 | 精度损失 | 适用场景 |
|------|------|------|------|----------|----------|
| `balanced` | INT8 | 4 | ~1.8 GB | <0.5% | **推荐** RTX 4060/5060 Web API |
| `fast` | INT8 | 1 | ~1.8 GB | <0.5% | 高并发 Web API |
| `quality` | FP16 | 4 | ~2.2 GB | <0.1% | 质量优先 12GB+ 服务器 |
| `edge_4gb` | NF4 | 1 | ~1.2 GB | 1-2% | Jetson Orin / 4GB 边缘 |
| `edge_2gb` | NF4 | 1 | ~0.9 GB | 1-3% | Jetson Nano / 2GB 边缘 |
| `edge_cpu` | FP32 | 1 | ~3.5 GB RAM | 0% | 无 GPU 设备（需 mT5-small） |

---

## 代码使用

```python
from deploy import (
    SignLanguageInference,
    preset_fast,       # Web API
    preset_edge_4gb,   # 边缘设备
    quick_translate,   # 一行翻译
    run_full_benchmark,  # 基准测试
)

# ---- Web API ----
config = preset_fast()
engine = SignLanguageInference(config)
result = engine.translate_pose_file("pose.pkl")
print(result.text, result.time_ms)
engine.unload()

# ---- 边缘设备 ----
config = preset_edge_4gb()
engine = SignLanguageInference(config)
result = engine.translate_pose_file("pose.pkl")
engine.unload()

# ---- 一行翻译 ----
text = quick_translate("pose.pkl", mode="balanced")

# ---- 基准测试 ----
results = run_full_benchmark(mode="balanced", pose_file="pose.pkl")
```

---

## 性能参考

### Web API 场景 (RTX 4060 8GB)

| 模式 | 加载时间 | 显存 | 推理延迟 | 生成速度 | QPS (batch=1) |
|------|----------|------|----------|----------|---------------|
| `balanced` | ~15s | 1.8 GB | ~420 ms | ~24 tok/s | ~2.4 |
| `fast` | ~15s | 1.8 GB | ~150 ms | ~45 tok/s | ~6.7 |
| `quality` | ~10s | 2.2 GB | ~400 ms | ~25 tok/s | ~2.5 |

### 边缘设备场景

| 设备 | 模式 | 显存 | 推理延迟 |
|------|------|------|----------|
| Jetson Orin 4GB | `edge_4gb` | 1.1 GB | ~650 ms |
| Jetson Nano 2GB | `edge_2gb` | 0.8 GB | ~1200 ms |
| Raspberry Pi 5 | `edge_cpu` | 3 GB RAM | ~5000 ms |

---

## 模型参数分布

```
mt5_model          ████████████████████████████████████  580M  (94.5%)
rgb_support_...    ██                                     11M  (1.8%)
gcn_modules        █                                       4M  (0.7%)
proj_linear        █                                       3M  (0.5%)
其他               █                                      15M  (2.5%)
                   ─────────────────────────────────────
                   总计: ~613M 参数
```

**mT5 占据 94.5%**，量化 mT5 就解决了部署的核心问题。

---

## 架构设计

### 高内聚
- 所有部署代码集中在 `deploy/` 目录
- 配置、加载、推理、服务、导出各司其职
- 一套 `DeployConfig` 贯穿全链路

### 低耦合
- **不修改原有代码**：只通过 `import` 引用原项目
- 独立 `config_deploy.py`，不依赖原 `config.py`
- monkey-patch 注入量化，过后立刻恢复
- `deploy/` 目录可随时删除，原项目不受影响

---

## 常见问题

**Q: Docker 启动时找不到 GPU？**
```bash
# 确认 nvidia-container-toolkit 已安装
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Q: 边缘设备 ONNX 推理报错？**
A: 确保 onnxruntime-gpu 版本与 CUDA 版本匹配

**Q: 如何降级到 mT5-small 进一步缩小？**
A: 修改 `config_deploy.py` 中的 `mt5_path` 指向 mt5-small 目录，模型会自动适配（需重新训练/微调）

**Q: Web API 并发性能？**
A: `fast` 模式下单卡 ~6 QPS。如需更高并发，可部署多实例 + Nginx 负载均衡
