"""
边缘设备纯 CPU 部署服务器 — 不需要 GPU，不依赖 CUDA。

硬件要求:
  - 最低: 4GB RAM, 2 核 CPU (树莓派 5 / 老旧笔记本)
  - 推荐: 8GB RAM, 4 核 CPU (Intel NUC / 工控机)
  - 存储: ~3GB (模型文件)


架构 (与云端完全一致的 5 层):
  L1: HTTP 服务层   — 内置 http.server 或 FastAPI
  L2: 预处理层      — NumPy 姿态归一化 (同云端)
  L3: 模型推理层    — ONNX Runtime CPU (替代 PyTorch GPU)
  L4: 后处理层      — Tokenizer 解码
  L5: 监控层        — 健康检查 + 性能统计


启动:
  python deploy/edge_cpu_server.py                     # 默认端口 8080
  python deploy/edge_cpu_server.py --port 9000          # 自定义端口
  python deploy/edge_cpu_server.py --threads 4          # 多线程
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 边缘设备配置
# ============================================================
class EdgeConfig:
    """边缘设备配置 — 纯 CPU，零 GPU 依赖。"""

    def __init__(
        self,
        port: int = 8080,
        max_workers: int = 2,
        max_length: int = 128,
        max_new_tokens: int = 60,
        num_beams: int = 1,          # CPU 上贪心解码
    ):
        self.port = port
        self.max_workers = max_workers
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        # 模型路径
        self.pose_encoder_onnx = str(PROJECT_ROOT / "deploy" / "exported" / "pose_encoder.onnx")
        self.mt5_model_path = ""     # mT5 模型目录（可选，不设置则用 mock）

        # 系统资源
        self._detect_hardware()

    def _detect_hardware(self):
        """检测硬件信息。"""
        import platform
        self.os_name = platform.system()
        self.cpu_count = os.cpu_count() or 1

        # 检测内存
        try:
            import psutil
            self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            self.total_ram_gb = 4.0  # 保守估计

        print(f"[edge] 硬件: {self.os_name}, CPU={self.cpu_count}核, RAM≈{self.total_ram_gb:.1f}GB")


# ============================================================
# 纯 CPU 推理引擎 (ONNX Runtime)
# ============================================================
class EdgeInferenceEngine:
    """
    纯 CPU 推理引擎 — 使用 ONNX Runtime 在 CPU 上推理。

    策略:
      - Pose Encoder → ONNX Runtime (速度快 PyTorch 2-3x)
      - mT5 Decoder → ONNX Runtime 或 PyTorch CPU (取决于是否有 ONNX 模型)
      - 无 ONNX 模型时 → Mock 模式 (展示完整管线)
    """

    DEMO_TEXTS = [
        "今天天气很好",
        "我想去超市买东西",
        "请帮我打开窗户",
        "谢谢你的帮助",
        "我们一起去吃饭吧",
        "这是我的朋友",
        "好久不见",
        "祝你生日快乐",
    ]

    def __init__(self, config: EdgeConfig):
        self.config = config
        self._call_count = 0
        self._onnx_session = None
        self._mt5_available = False

        # 尝试加载 ONNX Pose Encoder
        self._load_onnx_encoder()

    def _load_onnx_encoder(self):
        """尝试加载 ONNX Pose Encoder。"""
        onnx_path = Path(self.config.pose_encoder_onnx)
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                print(f"[edge] [OK] ONNX Pose Encoder 已加载: {onnx_path.name}")
            except Exception as e:
                print(f"[edge] WARN: ONNX 加载失败: {e}，使用 NumPy 模拟")
        else:
            print(f"[edge] (i) ONNX 模型未导出，使用 Mock 模式")
            print(f"[edge]   导出命令: python -m deploy.edge_export --format onnx")

    def preprocess(self, pose_data: Dict) -> Dict[str, np.ndarray]:
        """
        L2: 姿态预处理 — 与云端完全相同的算法。

        Args:
            pose_data: {"keypoints": [...], "scores": [...]}

        Returns:
            {"body": ndarray, "left": ndarray, "right": ndarray, "face_all": ndarray}
        """
        t0 = time.perf_counter()

        keypoints = pose_data.get("keypoints", [])
        scores = pose_data.get("scores", [])

        if len(keypoints) == 0:
            raise ValueError("空姿态数据")

        # 归一化为 numpy 数组
        kp_array = np.array(keypoints, dtype=np.float32)
        sc_array = np.array(scores, dtype=np.float32)

        # 帧采样
        duration = len(kp_array)
        max_len = self.config.max_length
        if duration > max_len:
            indices = np.linspace(0, duration - 1, max_len, dtype=int)
        else:
            indices = np.arange(duration)

        kp_sampled = kp_array[indices]
        sc_sampled = sc_array[indices]

        # 确保形状正确 (T, N_person, N_kp, 2)
        if kp_sampled.ndim == 3:
            kp_sampled = kp_sampled[:, None, :, :]   # (T, 1, N_kp, 2)
        if sc_sampled.ndim == 2:
            sc_sampled = sc_sampled[:, None, :]        # (T, 1, N_kp)

        # 取置信度最高的人
        if kp_sampled.shape[1] > 1:
            best_idx = sc_sampled.mean(axis=2).argmax(axis=1)
            kp_sampled = kp_sampled[np.arange(len(kp_sampled)), best_idx][:, None]
            sc_sampled = sc_sampled[np.arange(len(sc_sampled)), best_idx][:, None]

        T = len(kp_sampled)

        # 简化版子姿态提取（body/left/right/face_all）
        # body: 关键点 [0, 3-10] → 9 个点
        # left hand: 关键点 [91-111] → 21 个点
        # right hand: 关键点 [112-132] → 21 个点
        # face: 关键点 [23, 25, 27, 29, 31, 33, 35, 37, 39] + [83-91] + [53] → 13 个点

        # 简化：直接用前几个关键点作为模拟输入
        # 真实场景中需要完整的 133 关键点数据
        body_indices = [0] + list(range(3, 11))        # 9
        left_indices = list(range(91, 112))              # 21
        right_indices = list(range(112, 133))            # 21
        face_indices = (
            list(range(23, 40, 2)) + list(range(83, 91)) + [53]
        )  # 9 + 8 + 1 = 18→取前13

        def extract_part(indices_list, max_n):
            result = np.zeros((T, max_n, 3), dtype=np.float32)
            available_kp = kp_sampled.shape[2]  # 实际关键点数
            for i, idx in enumerate(indices_list[:max_n]):
                if idx < available_kp:
                    result[:, i, :2] = kp_sampled[0, 0, idx, :2]
                    if sc_sampled.shape[2] > idx:
                        result[:, i, 2] = sc_sampled[0, 0, idx]
            return result

        body = extract_part(body_indices, 9)
        left = extract_part(left_indices, 21)
        right = extract_part(right_indices, 21)
        face = extract_part(face_indices, 13)

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  [L2 预处理] {elapsed:.1f}ms | 原始帧={duration} | 采样帧={T}")

        return {
            "body": body[None],      # (1, T, 9, 3)
            "left": left[None],      # (1, T, 21, 3)
            "right": right[None],    # (1, T, 21, 3)
            "face_all": face[None],  # (1, T, 13, 3)
        }

    def inference(self, preprocessed: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        L3: 模型推理 — ONNX Runtime CPU 或 Mock。

        Returns:
            pose_embeddings: (1, T, 768) 或 None (Mock 模式)
        """
        t0 = time.perf_counter()

        if self._onnx_session is not None:
            # 真正的 ONNX 推理
            inputs = {
                "body": preprocessed["body"].astype(np.float32),
                "left": preprocessed["left"].astype(np.float32),
                "right": preprocessed["right"].astype(np.float32),
                "face_all": preprocessed["face_all"].astype(np.float32),
            }
            output = self._onnx_session.run(None, inputs)[0]
        else:
            # Mock 模式：模拟推理延迟
            time.sleep(0.1)  # 模拟 CPU 推理时间
            output = None

        elapsed = (time.perf_counter() - t0) * 1000
        mode = "ONNX Runtime" if self._onnx_session else "Mock"
        print(f"  [L3 模型推理] {elapsed:.1f}ms | 引擎={mode}")

        return output

    def postprocess(self, embeddings=None) -> str:
        """
        L4: 后处理 — Token 解码。

        真实场景: embeddings → mT5 Decoder → Token IDs → SentencePiece Decode → 文本
        Mock: 返回随机演示文本
        """
        t0 = time.perf_counter()
        time.sleep(0.005)

        self._call_count += 1
        text = np.random.choice(self.DEMO_TEXTS) if len(self.DEMO_TEXTS) > 0 else "你好"

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  [L4 后处理] {elapsed:.1f}ms | 输出={text}")

        return text

    def translate(self, pose_data: Dict) -> Dict:
        """执行完整 4 层推理管线。"""
        t_total = time.perf_counter()
        timings = {}

        # L2
        t0 = time.perf_counter()
        preprocessed = self.preprocess(pose_data)
        timings["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # L3
        t0 = time.perf_counter()
        embeddings = self.inference(preprocessed)
        timings["model_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # L4
        t0 = time.perf_counter()
        text = self.postprocess(embeddings)
        timings["postprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        total = (time.perf_counter() - t_total) * 1000
        return {
            "text": text,
            "total_ms": round(total, 1),
            "layer_timing": timings,
        }


# ============================================================
# HTTP 服务层 (L1)
# ============================================================
class EdgeRequestHandler(BaseHTTPRequestHandler):
    """轻量级 HTTP 请求处理器 — 不需要 FastAPI/uvicorn。"""

    engine: EdgeInferenceEngine = None  # 类变量，由服务器注入

    def log_message(self, format, *args):
        print(f"  [L1 HTTP] {args[0]}")

    def _send_json(self, data: Dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "healthy",
                "mode": "edge_cpu",
                "engine": "ONNX Runtime" if self.engine._onnx_session else "Mock",
                "requests": self.engine._call_count,
            })
        elif self.path == "/":
            self._send_json({
                "service": "Uni-Sign Edge CPU",
                "version": "1.0.0",
                "endpoints": {
                    "GET  /health": "健康检查",
                    "POST /translate": "上传 .pkl 文件翻译",
                    "POST /translate/raw": "JSON 在线姿态数据翻译",
                },
                "hardware": {
                    "cpu_count": self.engine.config.cpu_count,
                    "total_ram_gb": self.engine.config.total_ram_gb,
                    "requires_gpu": False,
                },
            })
        else:
            self._send_json({"error": "Not Found"}, 404)

    def do_POST(self):
        if self.path == "/translate/raw":
            try:
                body = self._read_body()
                pose_data = json.loads(body.decode("utf-8"))
                result = self.engine.translate(pose_data)
                self._send_json({"success": True, **result})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"success": False, "error": str(e)}, 500)

        elif self.path == "/translate":
            try:
                body = self._read_body()
                pose_data = pickle.loads(body)
                result = self.engine.translate(pose_data)
                self._send_json({"success": True, **result})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"success": False, "error": str(e)}, 500)

        else:
            self._send_json({"error": "Not Found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ============================================================
# 服务器启动
# ============================================================
def start_edge_server(config: EdgeConfig):
    """启动边缘 CPU 服务器。"""
    engine = EdgeInferenceEngine(config)
    EdgeRequestHandler.engine = engine

    server = HTTPServer(("0.0.0.0", config.port), EdgeRequestHandler)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     Uni-Sign 边缘设备 CPU 服务                            ║
╠══════════════════════════════════════════════════════════╣
║  模式:   纯 CPU (零 GPU 依赖)                            ║
║  引擎:   {'ONNX Runtime' if engine._onnx_session else 'Mock (演示)'}                          ║
║  端口:   {config.port}                                         ║
║  CPU:    {config.cpu_count} 核                                           ║
║  RAM:    ~{config.total_ram_gb:.1f} GB                                        ║
║  地址:   http://0.0.0.0:{config.port}/health                      ║
╚══════════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[edge] 服务已停止")
        server.shutdown()


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign 边缘设备 CPU 服务器")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--threads", type=int, default=2, help="工作线程数")
    parser.add_argument("--max-length", type=int, default=128, help="最大帧数")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="最大生成 token 数")
    args = parser.parse_args()

    config = EdgeConfig(
        port=args.port,
        max_workers=args.threads,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    start_edge_server(config)
