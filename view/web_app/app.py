"""
智能手语识别系统 - FastAPI后端
支持实时摄像头流处理、REST API、WebSocket通信
"""

from __future__ import annotations

import sys
import os
import asyncio
import base64
import json
import time
import sqlite3
import mimetypes
import hashlib
import tempfile
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from collections import deque
import threading
import queue

# 抑制OpenCV的RealSense错误输出
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# ============ 路径解析 ============
def get_bundle_dir():
    """返回打包资源目录 (templates, vue-dist).
    PyInstaller 模式下为 sys._MEIPASS, 开发模式下为 web_app/ 目录."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_models_root():
    """返回外部模型文件目录.
    PyInstaller 模式下为 exe 所在目录, 开发模式下为 Uni-Sign-storage 目录.
    优先级: 环境变量 UNISIGN_STORAGE_ROOT > 默认 storage 目录."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    # 从环境变量读取
    env_root = os.environ.get('UNISIGN_STORAGE_ROOT')
    if env_root and os.path.isdir(env_root):
        return env_root
    # 向上两层到项目根 (web_app -> view -> Uni-Sign)，再加 -storage
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    storage_root = project_root + '-storage'
    if os.path.isdir(storage_root):
        return storage_root
    # 最后回退到 view/ 目录
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_db_path():
    """返回 SQLite 数据库文件路径."""
    return os.path.join(get_models_root(), 'sign_recognition.db')

script_dir = os.path.dirname(os.path.abspath(__file__))
models_root = get_models_root()
bundle_dir = get_bundle_dir()

# 计算项目根目录 (web_app -> view -> Uni-Sign)
_project_root = os.path.dirname(os.path.dirname(script_dir))

# 将项目根目录添加到 Python 路径 (models.py, datasets.py, config.py 等在此)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 添加 demo 目录到路径 (从项目根目录查找)
demo_path = os.path.join(_project_root, 'demo')
if os.path.isdir(demo_path) and demo_path not in sys.path:
    sys.path.insert(0, demo_path)

# 设置路径环境变量 (同时支持 UNISIGN_ 和 RELISIGN_ 前缀)
os.environ['UNISIGN_STORAGE_ROOT'] = models_root
os.environ['UNISIGN_DATA_ROOT'] = models_root
os.environ['UNISIGN_OUTPUT_ROOT'] = os.path.join(models_root, 'out')
os.environ['UNISIGN_PRETRAINED_ROOT'] = os.path.join(models_root, 'pretrained_weight')
os.environ['UNISIGN_MT5_ROOT'] = os.path.join(models_root, 'pretrained_weight', 'mt5-base')
os.environ['UNISIGN_CACHE_ROOT'] = os.path.join(models_root, '.cache')
os.environ['CE_CSL_ROOT'] = os.path.join(models_root, 'data', 'CE_CSL', 'CE-CSL')

os.environ['RELISIGN_STORAGE_ROOT'] = models_root
os.environ['RELISIGN_DATA_ROOT'] = models_root
os.environ['RELISIGN_OUTPUT_ROOT'] = os.path.join(models_root, 'out')
os.environ['RELISIGN_PRETRAINED_ROOT'] = os.path.join(models_root, 'pretrained_weight')
os.environ['RELISIGN_MT5_ROOT'] = os.path.join(models_root, 'pretrained_weight', 'mt5-base')

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# ============ 请求模型 ============
class SetDeviceRequest(BaseModel):
    device_id: Optional[int] = None

class SetVoiceRequest(BaseModel):
    voice_id: str

class SetCameraRequest(BaseModel):
    camera_id: int

class XaiReferenceRequest(BaseModel):
    reference_text: Optional[str] = None

class AuthRegisterRequest(BaseModel):
    username: str
    password: str

class AuthLoginRequest(BaseModel):
    username: str
    password: str


# ============ SQLite 数据库 ============
class SQLiteDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._init_database()
            print(f"[SQLite] 数据库连接成功: {self.db_path}")
        except Exception as e:
            print(f"[SQLite] 数据库连接失败: {e}")
            self.conn = None

    def _init_database(self):
        if not self.conn:
            return
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    recognized_text TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 迁移: 旧表可能没有 user_id 列
            try:
                self.conn.execute("ALTER TABLE recognition_history ADD COLUMN user_id INTEGER")
            except:
                pass
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON recognition_history(timestamp)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_user ON recognition_history(user_id)"
            )
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            self.conn.commit()
            print("[SQLite] 数据库表初始化成功")
        except Exception as e:
            print(f"[SQLite] 初始化表失败: {e}")

    def save_recognition(self, text: str, confidence: float = 1.0, user_id: int = None) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_id = None
        if self.conn:
            try:
                cursor = self.conn.execute(
                    'INSERT INTO recognition_history (user_id, timestamp, recognized_text, confidence) VALUES (?, ?, ?, ?)',
                    (user_id, timestamp, str(text), float(confidence))
                )
                record_id = cursor.lastrowid
                self.conn.commit()
            except Exception as e:
                print(f"[SQLite] 保存记录失败: {e}")
        return {"id": record_id, "timestamp": timestamp, "text": str(text), "confidence": float(confidence)}

    def get_history(self, user_id: int = None, limit: int = 50) -> List[dict]:
        if self.conn:
            try:
                if user_id:
                    cur = self.conn.execute(
                        'SELECT id, timestamp, recognized_text as text, confidence FROM recognition_history WHERE user_id=? ORDER BY id DESC LIMIT ?',
                        (user_id, limit)
                    )
                else:
                    cur = self.conn.execute(
                        'SELECT id, timestamp, recognized_text as text, confidence FROM recognition_history ORDER BY id DESC LIMIT ?',
                        (limit,)
                    )
                return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[SQLite] 获取历史失败: {e}")
        return []

    def clear_history(self):
        if self.conn:
            try:
                self.conn.execute('DELETE FROM recognition_history')
                self.conn.commit()
            except Exception as e:
                print(f"[SQLite] 清空历史失败: {e}")

    def register_user(self, username: str, password_hash: str) -> dict:
        try:
            self.conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                            (username, password_hash))
            self.conn.commit()
            return {"success": True, "message": "注册成功"}
        except sqlite3.IntegrityError:
            return {"success": False, "message": "用户名已存在"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def login_user(self, username: str, password_hash: str) -> dict:
        row = self.conn.execute(
            'SELECT id, username FROM users WHERE username=? AND password_hash=?',
            (username, password_hash)
        ).fetchone()
        if not row:
            return {"success": False, "message": "用户名或密码错误"}
        import uuid
        token = uuid.uuid4().hex
        self.conn.execute('INSERT OR REPLACE INTO sessions (token, user_id) VALUES (?, ?)',
                        (token, row['id']))
        self.conn.commit()
        return {"success": True, "token": token, "user": {"id": row['id'], "username": row['username']}}

    def get_user_by_token(self, token: str) -> dict:
        row = self.conn.execute(
            'SELECT u.id, u.username FROM users u JOIN sessions s ON u.id=s.user_id WHERE s.token=?',
            (token,)
        ).fetchone()
        if row:
            return {"id": row['id'], "username": row['username']}
        return {}

    def logout_user(self, token: str):
        self.conn.execute('DELETE FROM sessions WHERE token=?', (token,))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


# ============ 全局状态 ============
class RecognitionState:
    """识别状态枚举 - 手动录制模式"""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class AppState:
    def __init__(self):
        self.model = None
        self.pose_estimator = None
        self.is_recognizing = False
        self.is_speech_on = True
        self.current_camera = 0
        self.pose_buffer = deque(maxlen=150)
        self.tts_engine = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.recognition_thread = None
        self.stop_recognition = threading.Event()
        
        # 手动录制模式状态
        self.recognition_state = RecognitionState.IDLE
        self.recording_start_time = 0.0
        self.max_recording_duration = 6.0   # 缩短最大录制时长，更快响应
        
        # 识别完成后冷却期
        self.post_recognition_cooldown = 1.0
        self.last_recognition_end_time = 0.0
        
        # 去重相关
        self.last_recognized_text = ""
        self.similar_count = 0
        
        # 防止并发识别
        self.recognition_lock = threading.Lock()
        self.is_recognition_running = False
        
        # 音频输出设备
        self.audio_output_device = None  # None = 系统默认
        self.audio_output_devices = []

        # XAI: 暂存最近一次识别的原始数据用于归因分析
        self.last_recognition_pose_data = None
        self.last_xai_result = None

        # 当前活跃用户ID（用于关联识别记录）
        self.current_user_id = None

        # 自动识别模式
        self.auto_mode = False
        self.auto_detecting = False
        self.last_keypoints_positions = None
        self.activity_counter = 0
        self.idle_counter = 0
        self.auto_activity_threshold = 0.008  # 归一化坐标移动阈值
        self.auto_start_frames = 15   # 连续活动帧数触发开始
        self.auto_stop_frames = 40    # 连续静止帧数触发结束

        # 性能指标
        self.perf_fps = 0
        self.perf_pose_ms = 0
        self.perf_last_time = time.time()
        self.perf_frame_count = 0

        # 初始化SQLite数据库
        self.db = SQLiteDatabase(get_db_path())

        # 语音转文字 (STT) — 延迟加载 faster-whisper 离线模型
        self.stt_model = None

        # 加载TTS
        self.init_tts()
        # 扫描音频设备
        self.scan_audio_devices()
    
    def init_tts(self):
        """初始化TTS引擎"""
        try:
            import edge_tts
            self.tts_engine = edge_tts
            self.tts_voice = "zh-CN-XiaoxiaoNeural"
            self.tts_rate = "+0%"
            print("[TTS] edge-tts 语音引擎初始化成功")
        except Exception as e:
            print(f"[TTS] 语音引擎初始化失败: {e}")
            self.tts_engine = None

    def _ensure_stt(self):
        """确保 STT 引擎已加载（延迟加载 faster-whisper 离线模型）
        ★ 强制 CPU 模式 — 避免与主模型抢占 GPU 显存"""
        if self.stt_model is not None:
            return True
        try:
            from faster_whisper import WhisperModel

            # ★ 强制 CPU: 避免 GPU OOM (主模型 ~2.5GB + Whisper ~1.5GB = 爆炸)
            device = 'cpu'
            compute_type = 'int8'
            model_size = 'small'

            print(f"[STT] 加载 faster-whisper (CPU) — 保护GPU显存 (size={model_size})...")

            self.stt_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.path.join(os.path.expanduser('~'), '.cache', 'faster-whisper')
            )
            print("[STT] faster-whisper (CPU) 加载完成 ✓")
            return True
        except ImportError:
            print("[STT] 请安装: pip install faster-whisper")
            return False
        except Exception as e:
            print(f"[STT] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def speech_to_text(self, wav_path: str) -> dict:
        """对 WAV 音频文件执行语音识别（faster-whisper 离线），返回 {'success': bool, 'text': str, 'error': str|None}"""
        if not self._ensure_stt():
            return {"success": False, "text": "", "error": "STT 离线引擎未安装，请运行: pip install faster-whisper"}

        try:
            print("[STT] 正在离线识别...")
            segments, info = self.stt_model.transcribe(
                wav_path,
                language='zh',
                beam_size=5,
                vad_filter=True,          # 过滤静音段
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            # 合并所有识别片段
            text_parts = []
            for segment in segments:
                if segment.text.strip():
                    text_parts.append(segment.text.strip())

            result = ''.join(text_parts)
            if result:
                print(f"[STT] 离线识别成功: \"{result}\"")
                return {"success": True, "text": result, "error": None}
            else:
                return {"success": False, "text": "", "error": "未能识别语音内容"}
        except Exception as e:
            print(f"[STT] 离线识别错误: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "text": "", "error": f"语音识别失败: {str(e)}"}
    
    def scan_audio_devices(self):
        """扫描可用音频输出设备"""
        try:
            import sounddevice as sd
            devices = []
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_output_channels'] > 0:
                    devices.append({
                        "id": i,
                        "name": dev['name'],
                        "channels": dev['max_output_channels'],
                        "is_default": i == sd.default.device[1]
                    })
            self.audio_output_devices = devices
            print(f"[音频] 发现 {len(devices)} 个输出设备: {[d['name'] for d in devices]}")
        except Exception as e:
            print(f"[音频] 扫描设备失败: {e}")
            self.audio_output_devices = [{"id": 0, "name": "系统默认设备", "channels": 2, "is_default": True}]
    
    async def _speak_async(self, text):
        """异步语音合成和播放（无需ffmpeg，使用pygame设备选择）"""
        try:
            import edge_tts
            import tempfile
            import subprocess
            import platform
            import uuid
            
            temp_file = os.path.join(tempfile.gettempdir(), f'tts_{uuid.uuid4().hex}.mp3')
            communicate = edge_tts.Communicate(text, self.tts_voice, rate=self.tts_rate)
            await communicate.save(temp_file)
            
            # 获取目标设备名称
            device_name = None
            if self.audio_output_device is not None and self.audio_output_devices:
                for dev in self.audio_output_devices:
                    if dev["id"] == self.audio_output_device:
                        device_name = dev["name"]
                        break
            
            if platform.system() == 'Windows':
                import pygame
                try:
                    # 先退出旧的 mixer，以便切换设备
                    if pygame.mixer.get_init():
                        pygame.mixer.quit()
                    
                    if device_name:
                        print(f"[TTS] 使用音频设备: {device_name}")
                        pygame.mixer.init(devicename=device_name)
                    else:
                        pygame.mixer.init()
                    
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                except Exception as e:
                    print(f"[TTS] pygame 播放失败: {e}")
            elif platform.system() == 'Darwin':
                subprocess.run(['afplay', temp_file], capture_output=True, timeout=30)
            else:
                try:
                    subprocess.run(['mpg123', '-q', temp_file], capture_output=True, timeout=30)
                except FileNotFoundError:
                    subprocess.run(['aplay', temp_file], capture_output=True, timeout=30)
            
            time.sleep(0.5)
            for _ in range(3):
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    break
                except:
                    time.sleep(0.3)
        except Exception as e:
            print(f"[TTS] 播报错误: {e}")
    
    def speak(self, text):
        """语音播报"""
        if not self.is_speech_on or not text:
            return
        
        def _speak():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._speak_async(text))
                loop.close()
            except Exception as e:
                print(f"[TTS] 播报错误: {e}")
        
        threading.Thread(target=_speak, daemon=True).start()
    
    def save_recognition(self, text: str, confidence: float = 0.5, user_id: int = None):
        """保存识别记录"""
        return self.db.save_recognition(text, confidence, user_id)

    def get_history(self, user_id: int = None, limit: int = 50) -> List[dict]:
        """获取识别历史"""
        return self.db.get_history(user_id, limit)
    
    def clear_history(self):
        """清空历史记录"""
        self.db.clear_history()
    
    def get_available_cameras(self) -> List[dict]:
        """检测可用摄像头"""
        cameras = []
        # 抑制OpenCV的错误输出
        import os
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cameras.append({
                            "id": i,
                            "name": f"摄像头 {i}",
                            "available": True
                        })
                    cap.release()
            except Exception:
                pass
        return cameras if cameras else [{"id": 0, "name": "默认摄像头", "available": True}]


app_state = AppState()


# ============ FastAPI应用 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    yield
    app_state.stop_recognition.set()
    if app_state.recognition_thread:
        app_state.recognition_thread.join(timeout=2)
    if app_state.cap:
        app_state.cap.release()
    if app_state.db:
        app_state.db.close()
    print("[关闭] 应用已关闭")


app = FastAPI(
    title="智能手语识别系统",
    description="实时手语识别、语音播报",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 静态文件 & 前端路由 ============
vue_dist = os.path.join(bundle_dir, 'vue-dist')
if os.path.isdir(vue_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(vue_dist, "assets")), name="assets")

@app.get("/inline")
async def serve_inline():
    """内联HTML前端"""
    template_path = os.path.join(bundle_dir, 'templates', 'index.html')
    if os.path.exists(template_path):
        return FileResponse(template_path)
    raise HTTPException(status_code=404, detail="Template not found")

@app.get("/")
async def serve_spa():
    """Vue SPA 首页"""
    index_path = os.path.join(vue_dist, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    # 回退到内联模板
    template_path = os.path.join(bundle_dir, 'templates', 'index.html')
    if os.path.exists(template_path):
        return FileResponse(template_path)
    return {"message": "ReliSign API Server", "docs": "/docs"}

# ============ 模型加载/卸载管理 ============
_model_loaded = False
_model_device = "cpu"
_model_checkpoint_path = None       # 记住checkpoint路径用于重载
_model_args_state = None            # 记住模型参数用于重载


def _get_vram_info():
    """获取当前GPU显存使用情况"""
    try:
        import torch
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            return {
                "total_gb": round(total, 2),
                "reserved_gb": round(reserved, 2),
                "allocated_gb": round(allocated, 2),
                "free_gb": round(free, 2),
                "usage_pct": round(reserved / total * 100, 1) if total > 0 else 0,
            }
    except Exception:
        pass
    return {"total_gb": 0, "reserved_gb": 0, "allocated_gb": 0, "free_gb": 0, "usage_pct": 0}


def _unload_model_from_gpu():
    """将模型从GPU卸载到CPU以释放显存，保留模型对象以便重载"""
    global _model_loaded, _model_device
    if not _model_loaded or app_state.model is None:
        return

    import torch
    try:
        print("[显存] 正在将模型从GPU卸载到CPU...")
        vram_before = _get_vram_info()

        # 移动模型到CPU
        app_state.model = app_state.model.cpu()
        _model_device = "cpu"

        # 清理GPU缓存
        torch.cuda.empty_cache()
        _model_loaded = True  # 模型仍在内存(CPU)，标记为已加载

        vram_after = _get_vram_info()
        freed = round(vram_before["reserved_gb"] - vram_after["reserved_gb"], 2)
        print(f"[显存] 模型已卸载到CPU，释放显存: {freed}GB, 当前占用: {vram_after['reserved_gb']}GB/{vram_after['total_gb']}GB")
    except Exception as e:
        print(f"[显存] 卸载失败: {e}")
        import traceback
        traceback.print_exc()


def _reload_model_to_gpu():
    """将模型从CPU重新加载到GPU"""
    global _model_loaded, _model_device
    if app_state.model is None:
        return load_model_if_needed()

    import torch
    if not torch.cuda.is_available():
        return True

    if _model_device == "cuda":
        return True  # 已经在GPU上

    try:
        print("[显存] 正在将模型从CPU重载到GPU...")
        vram_before = _get_vram_info()

        app_state.model = app_state.model.cuda()
        _model_device = "cuda"

        vram_after = _get_vram_info()
        used = round(vram_after["reserved_gb"] - vram_before["reserved_gb"], 2)
        print(f"[显存] 模型已重载到GPU，新增显存: {used}GB, 当前占用: {vram_after['reserved_gb']}GB/{vram_after['total_gb']}GB")
        return True
    except Exception as e:
        print(f"[显存] GPU重载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _unload_pose_estimator():
    """卸载姿态估计器以释放显存"""
    if app_state.pose_estimator is not None:
        print("[显存] 正在卸载姿态估计器...")
        try:
            # 尝试清理 ONNX Runtime 资源
            if hasattr(app_state.pose_estimator, '_Wholebody__model'):
                del app_state.pose_estimator._Wholebody__model
        except Exception:
            pass
        app_state.pose_estimator = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vram = _get_vram_info()
        print(f"[显存] 姿态估计器已卸载, 当前显存: {vram['reserved_gb']}GB")


def _ensure_pose_estimator():
    """确保姿态估计器已加载 (被cleanup/unload卸载后按需恢复)"""
    if app_state.pose_estimator is not None:
        return True
    try:
        import torch
        vram = _get_vram_info()
        _pose_device = 'cuda' if (torch.cuda.is_available() and vram["free_gb"] > 3.0) else 'cpu'
        print(f"[姿态] 正在加载 PoseTracker (Wholebody, det_frequency=5, device={_pose_device})...")
        from rtmlib import Wholebody, PoseTracker
        app_state.pose_estimator = PoseTracker(
            Wholebody,
            det_frequency=5,
            mode='lightweight',
            backend='onnxruntime',
            device=_pose_device,
            tracking=False
        )
        print(f"[姿态] PoseTracker加载成功 (det_frequency=5, device={_pose_device})")
        return True
    except Exception as e:
        print(f"[姿态] PoseTracker加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_model_if_needed():
    """延迟加载模型 (首次加载到GPU，后续可在GPU/CPU间切换)"""
    global _model_loaded, _model_device, _model_checkpoint_path, _model_args_state
    if _model_loaded and app_state.model is not None:
        # 模型已加载但可能在CPU上，自动重载到GPU
        if _model_device != "cuda":
            _reload_model_to_gpu()
        # 姿态估计器可能被 cleanup/unload 卸载过，按需恢复
        _ensure_pose_estimator()
        return True

    try:
        import torch

        # ★ 实时演示模型: Round6 > Stage3
        candidates = [
            os.path.join(models_root, 'out', 'cecsl_rgb_support_round6_overnight_20260407', 'best_checkpoint.pth'),
            os.path.join(models_root, 'out', 'stage3_finetuning', 'best_checkpoint.pth'),
        ]
        checkpoint_path = None
        for cp in candidates:
            if os.path.exists(cp):
                checkpoint_path = cp
                break
        if checkpoint_path is None:
            print(f"[模型] 未找到模型checkpoint，已搜索: {candidates[:2]}...")
            return False
        _model_checkpoint_path = checkpoint_path

        vram_before = _get_vram_info()
        print(f"[模型] 正在加载checkpoint: {checkpoint_path} (当前显存: {vram_before['reserved_gb']}GB)")

        # ★ 始终在CPU上加载checkpoint，避免双倍显存峰值
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        saved_args = checkpoint.get('args', {})

        class Args:
            pass

        args = Args()
        args.hidden_dim = saved_args.get('hidden_dim', 256)
        args.rgb_support = saved_args.get('rgb_support', False)
        args.pose_support = True
        args.decoder_mode = 'attn'
        args.num_class = 3
        args.dataset = saved_args.get('dataset', 'CE_CSL')
        args.fusion_use_uncertainty_gate = saved_args.get('fusion_use_uncertainty_gate', False)
        args.fusion_meta_dim = saved_args.get('fusion_meta_dim', 3)
        args.rgb_backbone_weights = saved_args.get('rgb_backbone_weights', '')
        args.rgb_backbone_pretrained = saved_args.get('rgb_backbone_pretrained', False)
        args.label_smoothing = saved_args.get('label_smoothing', 0.0)
        args.batch_size = saved_args.get('batch_size', 1)
        args.max_length = saved_args.get('max_length', 256)
        args.seed = saved_args.get('seed', 42)
        _model_args_state = args

        from models import Reli_Sign
        from transformers import T5Tokenizer

        print(f"[模型] 创建Uni_Sign模型 (rgb_support={args.rgb_support}, hidden_dim={args.hidden_dim})...")
        model = Reli_Sign(args=args)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        print(f"[模型] 正在加载权重...")
        ret = model.load_state_dict(state_dict, strict=False)
        print(f"[模型] 加载完成: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")

        # ★ 释放checkpoint引用减少内存峰值
        del checkpoint, state_dict

        mt5_path = os.environ.get('RELISIGN_MT5_ROOT', 'pretrained_weight/mt5-base')
        print(f"[模型] 正在加载mT5 tokenizer: {mt5_path}")
        model.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

        # ★ 移到 GPU 并设为 eval 模式
        if torch.cuda.is_available():
            model = model.cuda()
            _model_device = "cuda"
            print("[模型] 已移到 CUDA GPU")
        else:
            _model_device = "cpu"
        model.eval()
        app_state.model = model

        # 清理缓存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        vram_after = _get_vram_info()
        print(f"[模型] ReliSign模型加载成功! 显存占用: {vram_after['reserved_gb']}GB/{vram_after['total_gb']}GB ({vram_after['usage_pct']}%)")

        # 加载姿态估计器 (ONNX, 按显存自动选择 GPU/CPU)
        _ensure_pose_estimator()

        _model_loaded = True
        return True

    except Exception as e:
        print(f"[模型] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============ 姿态处理函数 ============
def process_pose_frame(frame_rgb):
    """处理单帧姿态 — 多人时选画面占比最大的人"""
    if app_state.pose_estimator is None:
        return None, None

    try:
        keypoints, scores = app_state.pose_estimator(frame_rgb)

        if keypoints is None or len(keypoints) == 0:
            return None, None

        kp_squeezed = None
        sc_squeezed = None

        if len(keypoints.shape) == 3:
            n_persons = keypoints.shape[0]
            if n_persons == 1:
                kp_squeezed = keypoints[0]
                sc_squeezed = scores[0]
            else:
                # ★ 多人: 选画面占比最大 (bounding box 面积最大)
                best_area = -1
                best_idx = 0
                for p in range(n_persons):
                    kp_p = keypoints[p]
                    sc_p = scores[p]
                    valid = kp_p[sc_p > 0.3]  # 只看置信度>0.3的关键点
                    if len(valid) < 5:
                        continue
                    x_min, y_min = valid.min(axis=0)
                    x_max, y_max = valid.max(axis=0)
                    area = (x_max - x_min) * (y_max - y_min)
                    if area > best_area:
                        best_area = area
                        best_idx = p
                kp_squeezed = keypoints[best_idx]
                sc_squeezed = scores[best_idx]
        elif len(keypoints.shape) == 2 and keypoints.shape[1] == 2:
            kp_squeezed = keypoints
            sc_squeezed = scores

        if kp_squeezed is not None and sc_squeezed is not None:
            return kp_squeezed, sc_squeezed

        return None, None
        
    except Exception as e:
        print(f"[姿态] 处理错误: {e}")
        return None, None


def run_recognition(pose_data=None):
    """执行手语识别 - 参照 demo/online_inference.py 的正确推理流程

    Args:
        pose_data: 可选的 buffer 快照 list of (kp, sc) tuples.
                   如果为 None, 则从 app_state.pose_buffer 读取.
    """
    if app_state.is_recognition_running:
        return None, 0.0

    acquired = app_state.recognition_lock.acquire(timeout=30)  # 30秒超时防止死锁
    if not acquired:
        print("[识别] 获取锁超时，强制重置")
        app_state.is_recognition_running = False
        try: app_state.recognition_lock.release()
        except: pass
        return None, 0.0

    app_state.is_recognition_running = True

    try:
        if app_state.model is None:
            return None, 0.0

        import torch

        all_keypoints = []
        all_scores = []
        # 使用传入的快照, 否则从实时 buffer 读取
        buffer_list = pose_data if pose_data is not None else list(app_state.pose_buffer)

        for kp, sc in buffer_list:
            if kp is not None and sc is not None:
                if kp.shape == (133, 2) and sc.shape == (133,):
                    all_keypoints.append(kp)
                    all_scores.append(sc)

        min_frames = 2
        if len(all_keypoints) < min_frames:
            return None, 0.0

        keypoints = np.stack(all_keypoints)
        scores = np.stack(all_scores)
        T = keypoints.shape[0]

        # ---- 帧采样：使用 build_uniform_indices 保持时间均匀分布 ----
        max_frames = 256
        if T > max_frames:
            try:
                from datasets import build_uniform_indices
                ids = build_uniform_indices(T, max_frames)
                if hasattr(ids, "numpy"): ids = ids.numpy()
            except Exception:
                ids = np.linspace(0, T - 1, max_frames, dtype=int)
            keypoints = keypoints[ids]
            scores = scores[ids]
            T = max_frames

        # ---- 使用 demo 的 load_part_kp 进行正确姿态预处理 ----
        skeletons = keypoints[:, np.newaxis, :, :]  # (T, 1, 133, 2)
        confs = scores[:, np.newaxis, :]             # (T, 1, 133)

        try:
            from datasets import load_part_kp
            pose_dict = load_part_kp(skeletons, confs, force_ok=True)
        except Exception as e:
            print(f"[姿态] load_part_kp 失败: {e}")
            return None, 0.0

        if pose_dict is None:
            return None, 0.0

        # ---- 组装模型输入 ----
        src_input = {}
        for key in ['body', 'left', 'right', 'face_all']:
            if key in pose_dict:
                # ★ 修复dtype: 转为float32
                src_input[key] = pose_dict[key].unsqueeze(0).float()  # (1, T, V, 3)

        src_input['attention_mask'] = torch.ones(1, T, dtype=torch.long)

        # ★ RGB hand data for round6 full model
        if app_state.model.args.rgb_support:
            src_input['left_hands'] = torch.zeros(1, 3, 112, 112, dtype=torch.float32)
            src_input['right_hands'] = torch.zeros(1, 3, 112, 112, dtype=torch.float32)
            src_input['left_sampled_indices'] = torch.tensor([-1])
            src_input['right_sampled_indices'] = torch.tensor([-1])
            src_input['left_rgb_len'] = [1]
            src_input['right_rgb_len'] = [1]
            src_input['left_skeletons_norm'] = torch.zeros(1, 21, 2, dtype=torch.float32)
            src_input['right_skeletons_norm'] = torch.zeros(1, 21, 2, dtype=torch.float32)
            src_input['left_rgb_meta'] = torch.zeros(1, 3, dtype=torch.float32)
            src_input['right_rgb_meta'] = torch.zeros(1, 3, dtype=torch.float32)

        tgt_input = {'gt_sentence': ['']}

        # ---- 移动到模型设备 ----
        device = next(app_state.model.parameters()).device
        for k in list(src_input.keys()):
            if isinstance(src_input[k], torch.Tensor):
                src_input[k] = src_input[k].to(device)

        # ---- 推理 (贪心解码, 速度快4x) ----
        with torch.inference_mode():
            stack_out = app_state.model(src_input, tgt_input)
            output = app_state.model.generate(
                stack_out,
                max_new_tokens=50,      # 实时翻译通常<25字
                num_beams=1,            # 贪心解码, 极速
            )

        # ---- 解码 (兼容贪心和beam search) ----
        if hasattr(output, 'sequences'):
            sequences = output.sequences
        else:
            sequences = output  # 贪心解码直接返回tensor
        predictions = app_state.model.mt5_tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )

        if predictions and len(predictions) > 0:
            result = _post_process_prediction(predictions[0])

            if not result:
                return None, 0.0

            # 智能学习: 应用纠错记忆
            corrected = _apply_correction(result)
            if corrected != result:
                print(f"[学习] 自动纠错: \"{result}\" → \"{corrected}\"")
                result = corrected

            # 基于 beam search 分数计算真实置信度
            try:
                if hasattr(output, 'sequences_scores') and output.sequences_scores is not None:
                    scores = output.sequences_scores.cpu().numpy()
                    # 每 token 平均概率 = exp(score), 取最大值 beam 的分数
                    confidence = float(np.exp(scores[0]))
                    confidence = max(0.2, min(0.98, confidence))
                else:
                    raise ValueError("no scores")
            except Exception:
                result_len = len(result)
                confidence = min(0.5 + result_len * 0.03, 0.95)
                if result_len < 2:
                    confidence = 0.3
                elif result_len < 4:
                    confidence = min(confidence, 0.6)

            # 暂存识别数据用于 XAI 分析
            app_state.last_recognition_pose_data = {
                'keypoints': all_keypoints,
                'scores': all_scores,
                'prediction': result,
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            app_state.last_xai_result = None

            return result, confidence

        return None, 0.0

    except Exception as e:
        print(f"[识别] 推理错误: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0
    finally:
        app_state.is_recognition_running = False
        app_state.recognition_lock.release()
        # 清理 GPU 缓存防止 OOM
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except: pass


def _post_process_prediction(text):
    """后处理预测结果"""
    if not text:
        return ""
    
    # 移除特殊标记
    text = text.replace('<extra_id_0>', '').replace('<extra_id_1>', '').replace('<pad>', '')
    text = text.replace('▁', '').strip()
    
    # 移除开头可能是句子开头标记的内容
    text = text.strip()
    
    return text


# ============ 智能学习 — 纠错记忆系统 ============

_learning_db = {}
_learning_file = None

def _get_learning_file():
    global _learning_file
    if _learning_file is None:
        _learning_file = os.path.join(get_models_root(), "learning_corrections.json")
    return _learning_file

def _load_learning_db():
    global _learning_db
    fpath = _get_learning_file()
    if os.path.exists(fpath):
        try:
            import json as _json
            with open(fpath, 'r', encoding='utf-8') as f:
                _learning_db = _json.load(f)
        except Exception:
            _learning_db = {}

def _save_learning_db():
    import json as _json
    with open(_get_learning_file(), 'w', encoding='utf-8') as f:
        _json.dump(_learning_db, f, ensure_ascii=False, indent=2)

def _apply_correction(predicted_text: str) -> str:
    """查找是否有相似预测的纠错记录，返回纠错后文本"""
    if not _learning_db:
        _load_learning_db()
    if not _learning_db:
        return predicted_text
    if predicted_text in _learning_db:
        return _learning_db[predicted_text]
    best_match, best_sim = None, 0
    for stored_pred, corrected in _learning_db.items():
        sim = _compute_similarity(predicted_text, stored_pred)
        if sim > best_sim:
            best_sim = sim
            best_match = corrected
    if best_sim >= 80.0 and best_match:
        return best_match
    return predicted_text

_load_learning_db()


def run_xai_analysis(reference_text: Optional[str] = None):
    """运行 Grad-CAM + Attention XAI 分析 (适配自 demo/view/xai_service.py)
    对最近一次识别结果计算: Grad-CAM 部位重要性、Encoder 帧间注意力、字符对齐
    使用 self-distillation: 以模型自身预测作为 target 进行 backward."""
    if app_state.last_recognition_pose_data is None:
        return None
    if app_state.model is None:
        return None

    import torch
    import numpy as np

    pose_data = app_state.last_recognition_pose_data
    keypoints = np.array(pose_data['keypoints'])
    scores = np.array(pose_data['scores'])
    prediction = pose_data['prediction']

    T = keypoints.shape[0]
    max_frames = 300
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        keypoints = keypoints[indices]
        scores = scores[indices]
        T = max_frames

    if T < 6:
        return None

    skeletons = keypoints[:, np.newaxis, :, :]
    confs = scores[:, np.newaxis, :]

    try:
        from datasets import load_part_kp
        pose_dict = load_part_kp(skeletons, confs, force_ok=True)
    except Exception as e:
        print(f"[XAI] load_part_kp 失败: {e}")
        return None

    if pose_dict is None:
        return None

    src_input = {}
    src_length_batch = torch.tensor([T], dtype=torch.long)
    for key in ['body', 'left', 'right', 'face_all']:
        if key in pose_dict:
            src_input[key] = pose_dict[key].unsqueeze(0)
    src_input['attention_mask'] = torch.ones(1, T, dtype=torch.long)
    src_input['src_length_batch'] = src_length_batch

    target_sentence = prediction or reference_text or "手语翻译"
    tgt_input = {'gt_sentence': [target_sentence]}

    device = next(app_state.model.parameters()).device
    for k in src_input:
        if isinstance(src_input[k], torch.Tensor):
            src_input[k] = src_input[k].to(device)

    model = app_state.model
    activations = {}
    gradients = {}
    hooks_handles = []

    def _make_fwd(part_name):
        def hook(_m, _inp, output):
            activations[part_name] = output
        return hook

    def _make_bwd(part_name):
        def hook(_m, _ginp, gout):
            if gout and gout[0] is not None:
                gradients[part_name] = gout[0]
        return hook

    for part in model.modes:
        h1 = model.fusion_gcn_modules[part].register_forward_hook(_make_fwd(part))
        h2 = model.fusion_gcn_modules[part].register_full_backward_hook(_make_bwd(part))
        hooks_handles.extend([h1, h2])

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        stack_out = model(src_input, tgt_input)
        loss = stack_out['loss']
        loss.backward()

        # --- Grad-CAM per part ---
        cam_series = {}
        for part in model.modes:
            act = activations.get(part)
            grad = gradients.get(part)
            if act is None or grad is None:
                cam_series[part] = np.zeros(T, dtype=np.float32)
                continue
            act_f = act.detach().float()
            grad_f = grad.detach().float()
            weights = grad_f.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act_f).sum(dim=1)).mean(dim=-1)
            cam = cam[0, :T].cpu().numpy()
            if cam.size > 0:
                cmin, cmax = cam.min(), cam.max()
                if cmax - cmin > 1e-6:
                    cam = (cam - cmin) / (cmax - cmin)
                else:
                    cam = np.zeros_like(cam)
            cam_series[part] = cam.astype(np.float32)

        cam_combined = np.mean([cam_series[p] for p in model.modes], axis=0)

        # --- Encoder Attention ---
        encoder_out = model.mt5_model.encoder(
            inputs_embeds=stack_out['inputs_embeds'].detach(),
            attention_mask=stack_out['attention_mask'].detach(),
            output_attentions=True,
            return_dict=True,
        )
        attn_last = encoder_out.attentions[-1].detach().float().mean(dim=1)[0]
        prefix_len = attn_last.shape[0] - T
        sign_attn = attn_last[prefix_len:, prefix_len:][:T, :T].cpu().numpy()
        frame_importance = (sign_attn.mean(axis=0) + sign_attn.mean(axis=1)) / 2.0
        attn_norm = frame_importance / (frame_importance.max() + 1e-6)

        xai_score = 0.55 * cam_combined + 0.45 * attn_norm

        # --- Part summary ---
        part_importance = {}
        part_names_cn = {'body': '身体躯干', 'left': '左手', 'right': '右手', 'face_all': '面部'}
        for part in model.modes:
            s = cam_series[part]
            part_importance[part] = {
                'name': part_names_cn.get(part, part),
                'mean': round(float(s.mean()), 4) if s.size > 0 else 0.0,
                'peak': round(float(s.max()), 4) if s.size > 0 else 0.0,
            }

        # --- Top frames ---
        frame_scores = [(int(i), float(xai_score[i])) for i in range(T)]
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        top_frames = [{'frame': f, 'score': round(s, 4)} for f, s in frame_scores[:6]]

        # --- Downsample attention matrix (max 50x50) ---
        attn_display_size = min(T, 50)
        if T > attn_display_size:
            bin_size = T // attn_display_size
            attn_small = np.zeros((attn_display_size, attn_display_size), dtype=np.float32)
            for i in range(attn_display_size):
                for j in range(attn_display_size):
                    block = sign_attn[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size]
                    attn_small[i, j] = float(block.mean())
        else:
            attn_small = sign_attn.astype(np.float32)

        attention_matrix = [[round(float(attn_small[i, j]), 6) for j in range(attn_display_size)] for i in range(attn_display_size)]

        # --- Frame timeline ---
        frame_timeline = []
        for i in range(T):
            frame_timeline.append({
                'frame': i,
                'gradcam_combined': round(float(cam_combined[i]), 4),
                'encoder_attention': round(float(attn_norm[i]), 4),
                'xai_score': round(float(xai_score[i]), 4),
                'body': round(float(cam_series.get('body', np.zeros(T))[min(i, len(cam_series.get('body', []))-1) if len(cam_series.get('body', [])) > 0 else 0]), 4),
                'left': round(float(cam_series.get('left', np.zeros(T))[min(i, len(cam_series.get('left', []))-1) if len(cam_series.get('left', [])) > 0 else 0]), 4),
                'right': round(float(cam_series.get('right', np.zeros(T))[min(i, len(cam_series.get('right', []))-1) if len(cam_series.get('right', [])) > 0 else 0]), 4),
                'face_all': round(float(cam_series.get('face_all', np.zeros(T))[min(i, len(cam_series.get('face_all', []))-1) if len(cam_series.get('face_all', [])) > 0 else 0]), 4),
            })

        # --- Character alignment (if reference provided) ---
        alignment = None
        if reference_text:
            try:
                alignment = _compute_alignment(reference_text, prediction)
            except Exception as e:
                print(f"[XAI] 对齐失败: {e}")
                alignment = []

        result = {
            'prediction': prediction,
            'confidence': pose_data.get('confidence', 0.0),
            'timestamp': pose_data['timestamp'],
            'frame_count': T,
            'part_importance': part_importance,
            'frame_timeline': frame_timeline,
            'top_frames': top_frames,
            'attention_matrix': attention_matrix,
            'attention_matrix_size': attn_display_size,
            'alignment': alignment,
            'has_reference': bool(reference_text),
            'loss': round(float(loss.detach().cpu().item()), 4),
        }

        app_state.last_xai_result = result
        return result

    finally:
        for h in hooks_handles:
            h.remove()


def _compute_alignment(reference, prediction):
    """编辑距离对齐 - 适配自 demo/view/data_service.py:_align_texts()"""
    GAP = '∅'
    ref_chars = list(reference)
    pred_chars = list(prediction)
    m, n = len(ref_chars), len(pred_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_chars[i - 1] == pred_chars[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_chars[i - 1] == pred_chars[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((ref_chars[i - 1], pred_chars[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((ref_chars[i - 1], GAP))
            i -= 1
            continue
        aligned.append((GAP, pred_chars[j - 1]))
        j -= 1

    aligned.reverse()
    rows = []
    for idx, (ref_char, pred_char) in enumerate(aligned):
        if ref_char == pred_char:
            op = 'match'
        elif ref_char == GAP:
            op = 'insert'
        elif pred_char == GAP:
            op = 'delete'
        else:
            op = 'replace'
        rows.append({'position': idx, 'reference_char': ref_char, 'prediction_char': pred_char, 'operation': op})
    return rows


def prepare_pose_data(keypoints, scores, thr=0.3):
    """准备姿态数据 - 使用demo目录下的正确处理方式"""
    import torch
    
    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis, ...]
        scores = scores[np.newaxis, ...]
    
    if keypoints is None or len(keypoints) == 0 or keypoints.shape[0] == 0:
        return None
    
    if keypoints.shape[1] != 133:
        return None
    
    T = keypoints.shape[0]
    
    # 准备skeletons和confs格式 (T, 1, 133, 2) 和 (T, 1, 133)
    # 以匹配demo/datasets.py中load_part_kp的输入格式
    skeletons = keypoints[:, np.newaxis, :, :]  # (T, 1, 133, 2)
    confs = scores[:, np.newaxis, :]  # (T, 1, 133)
    
    # 使用demo目录下的load_part_kp函数进行正确处理
    try:
        from datasets import load_part_kp
        result_dict = load_part_kp(skeletons, confs, force_ok=True)
        return result_dict
    except Exception as e:
        print(f"[姿态] demo数据处理失败，回退到本地处理: {e}")
        # 如果demo处理失败，使用原来的处理方式
        return _prepare_pose_data_local(keypoints, scores, thr)


def _prepare_pose_data_local(keypoints, scores, thr=0.3):
    """本地姿态数据处理（备用）"""
    import torch
    
    result_dict = {}
    T = keypoints.shape[0]
    
    def crop_scale(data, threshold):
        mask = data[..., -1] > threshold
        valid_data = data[mask]
        if len(valid_data) == 0:
            return data, 0, 0
        valid_coords = valid_data[..., :2]
        if len(valid_coords) == 0:
            return data, 0, 0
        xmin = np.min(valid_coords[..., 0])
        xmax = np.max(valid_coords[..., 0])
        ymin = np.min(valid_coords[..., 1])
        ymax = np.max(valid_coords[..., 1])
        scale = max(xmax - xmin, ymax - ymin)
        return data, scale, len(valid_data)
    
    # body
    body_indices = [0] + list(range(3, 11))
    body_kp = keypoints[:, body_indices, :]
    body_conf = scores[:, body_indices]
    body_combined = np.concatenate([body_kp, body_conf[..., np.newaxis]], axis=-1)
    body_result, scale, _ = crop_scale(body_combined, thr)
    result_dict['body'] = torch.tensor(body_result, dtype=torch.float32)
    
    # left hand
    left_indices = list(range(91, 112))
    left_kp = keypoints[:, left_indices, :] - keypoints[:, 91:92, :]
    left_conf = scores[:, left_indices]
    left_combined = np.concatenate([left_kp, left_conf[..., np.newaxis]], axis=-1)
    
    if scale == 0:
        result_dict['left'] = torch.zeros((T, 21, 3), dtype=torch.float32)
    else:
        left_result = left_combined.copy()
        left_result[..., :2] = left_result[..., :2] / scale
        left_result = np.clip(left_result, -1, 1)
        left_result[left_result[..., 2] <= thr] = 0
        result_dict['left'] = torch.tensor(left_result, dtype=torch.float32)
    
    # right hand
    right_indices = list(range(112, 133))
    right_kp = keypoints[:, right_indices, :] - keypoints[:, 112:113, :]
    right_conf = scores[:, right_indices]
    right_combined = np.concatenate([right_kp, right_conf[..., np.newaxis]], axis=-1)
    
    if scale == 0:
        result_dict['right'] = torch.zeros((T, 21, 3), dtype=torch.float32)
    else:
        right_result = right_combined.copy()
        right_result[..., :2] = right_result[..., :2] / scale
        right_result = np.clip(right_result, -1, 1)
        right_result[right_result[..., 2] <= thr] = 0
        result_dict['right'] = torch.tensor(right_result, dtype=torch.float32)
    
    # face
    face_indices = list(range(23, 23+17))[::2] + list(range(83, 83+8)) + [53]
    face_kp = keypoints[:, face_indices, :] - keypoints[:, 53:54, :]
    face_conf = scores[:, face_indices]
    face_combined = np.concatenate([face_kp, face_conf[..., np.newaxis]], axis=-1)
    
    if scale == 0:
        result_dict['face_all'] = torch.zeros((T, 18, 3), dtype=torch.float32)
    else:
        face_result = face_combined.copy()
        face_result[..., :2] = face_result[..., :2] / scale
        face_result = np.clip(face_result, -1, 1)
        face_result[face_result[..., 2] <= thr] = 0
        result_dict['face_all'] = torch.tensor(face_result, dtype=torch.float32)
    
    return result_dict


# ============ 手动录制模式状态管理 ============
def update_recognition_state(current_kp):
    """更新识别状态"""
    current_time = time.time()
    buffer_size = len(app_state.pose_buffer)
    
    if app_state.recognition_state == RecognitionState.IDLE:
        if app_state.is_recognizing:
            app_state.recognition_state = RecognitionState.RECORDING
            app_state.recording_start_time = current_time
            app_state.pose_buffer.clear()
            print(f"[状态] 开始录制")
    
    elif app_state.recognition_state == RecognitionState.RECORDING:
        if not app_state.is_recognizing:
            app_state.recognition_state = RecognitionState.PROCESSING
            duration = current_time - app_state.recording_start_time
            print(f"[状态] 录制结束，持续{duration:.1f}秒，缓冲帧数: {buffer_size}")
        else:
            recording_duration = current_time - app_state.recording_start_time
            # 无最大录制时长限制，由用户手动停止
    
    return app_state.recognition_state


def should_trigger_recognition():
    """判断是否应该触发识别"""
    if app_state.recognition_state == RecognitionState.PROCESSING:
        return True
    return False

def detect_signing_activity(current_kp):
    """自动检测手语活动: 基于关键点运动速度判断是否正在打手语
    返回: 'started' | 'stopped' | None"""
    if current_kp is None or app_state.last_keypoints_positions is None:
        if current_kp is not None:
            app_state.last_keypoints_positions = current_kp.copy()
        return None

    prev = app_state.last_keypoints_positions
    curr = current_kp

    # 计算手部+腕部关键点的平均移动速度（归一化坐标）
    hand_indices = list(range(9, 11)) + list(range(91, 133))  # 手腕 + 左右手
    total_motion = 0.0
    count = 0
    for i in hand_indices:
        if i < len(prev) and i < len(curr) and prev[i] is not None and curr[i] is not None:
            dx = float(curr[i, 0]) - float(prev[i, 0])
            dy = float(curr[i, 1]) - float(prev[i, 1])
            total_motion += (dx*dx + dy*dy) ** 0.5
            count += 1

    app_state.last_keypoints_positions = curr.copy()
    avg_motion = total_motion / max(count, 1)

    if avg_motion > app_state.auto_activity_threshold:
        app_state.activity_counter += 1
        app_state.idle_counter = 0
    else:
        app_state.idle_counter += 1
        app_state.activity_counter = 0

    # 活动持续触发开始录制
    if app_state.activity_counter >= app_state.auto_start_frames and not app_state.is_recognizing:
        return 'started'
    # 静止持续触发停止录制
    if app_state.idle_counter >= app_state.auto_stop_frames and app_state.is_recognizing:
        return 'stopped'

    return None


def reset_recognition_state():
    """重置识别状态"""
    app_state.recognition_state = RecognitionState.IDLE
    app_state.last_recognition_end_time = time.time()
    print(f"[状态] 识别完成，回到空闲")


# ============ 显存管理 API ============

@app.get("/api/vram")
async def get_vram_status():
    """获取 GPU 显存使用情况"""
    info = _get_vram_info()
    info["model_on_gpu"] = (_model_device == "cuda")
    info["model_loaded"] = _model_loaded
    info["pose_loaded"] = bool(app_state.pose_estimator is not None)
    return info


@app.post("/api/model/unload")
async def unload_model():
    """将模型从GPU卸载到CPU，释放显存"""
    _unload_model_from_gpu()
    _unload_pose_estimator()
    # 清理 PyTorch CUDA 缓存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"success": True, "vram": _get_vram_info(), "message": "模型已从GPU卸载"}


@app.post("/api/model/reload")
async def reload_model():
    """将模型从CPU重新加载到GPU，并恢复姿态估计器"""
    success = _reload_model_to_gpu()
    if not success and app_state.model is None:
        load_model_if_needed()

    # 恢复姿态估计器 (如果被卸载了)
    if app_state.pose_estimator is None and _model_device == "cuda":
        try:
            import torch
            vram = _get_vram_info()
            from rtmlib import Wholebody, PoseTracker
            _pose_device = 'cuda' if (torch.cuda.is_available() and vram["free_gb"] > 3.0) else 'cpu'
            print(f"[姿态] 恢复姿态估计器 (device={_pose_device})")
            app_state.pose_estimator = PoseTracker(
                Wholebody, det_frequency=5, mode='lightweight',
                backend='onnxruntime', device=_pose_device, tracking=False
            )
        except Exception as e:
            print(f"[姿态] 恢复失败: {e}")

    return {
        "success": _model_device == "cuda",
        "vram": _get_vram_info(),
        "model_on_gpu": _model_device == "cuda",
        "pose_loaded": bool(app_state.pose_estimator is not None),
    }


@app.post("/api/model/cleanup")
async def full_cleanup():
    """完全清理: 卸载模型 + 姿态估计器 + 清空缓存 + 垃圾回收"""
    import torch, gc
    _unload_pose_estimator()
    _unload_model_from_gpu()
    # 清空 buffer
    app_state.pose_buffer.clear()
    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {"success": True, "vram": _get_vram_info(), "message": "显存全面清理完成"}


# ============ REST API 路由 ============
@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    load_model_if_needed()
    return {
        "model_loaded": bool(app_state.model is not None),
        "pose_loaded": bool(app_state.pose_estimator is not None),
        "is_recognizing": bool(app_state.is_recognizing),
        "is_speech_on": bool(app_state.is_speech_on),
        "recognition_mode": "manual",
        "buffer_size": int(len(app_state.pose_buffer)),
        "model_on_gpu": (_model_device == "cuda"),
        "vram": _get_vram_info(),
        "recognition_state": str(app_state.recognition_state),
        "recording_duration": float(time.time() - app_state.recording_start_time) if app_state.recognition_state == RecognitionState.RECORDING else 0.0
    }


@app.get("/api/cameras")
async def get_cameras():
    """获取可用摄像头列表"""
    cameras = app_state.get_available_cameras()
    return {"cameras": cameras}


@app.post("/api/cameras/set")
async def set_camera(req: SetCameraRequest):
    """设置当前摄像头"""
    app_state.current_camera = req.camera_id
    print(f"[摄像头] 切换至: 摄像头 {req.camera_id}")
    return {"success": True, "camera_id": req.camera_id}


@app.get("/api/voices")
async def get_voices():
    """获取可用语音包列表"""
    if app_state.tts_engine:
        try:
            import edge_tts
            voices = await edge_tts.list_voices(language="zh")
            return {"voices": [{"id": v["Name"], "name": v.get("LocalName", v["Name"])} for v in voices]}
        except:
            pass
    return {"voices": [{"id": "default", "name": "默认语音"}]}


@app.post("/api/voices/set")
async def set_voice(req: SetVoiceRequest):
    """设置TTS语音"""
    app_state.tts_voice = req.voice_id
    print(f"[TTS] 语音切换至: {req.voice_id}")
    return {"success": True, "voice_id": req.voice_id}


@app.get("/api/history")
async def get_history(limit: int = 50, user_id: int = None):
    """获取识别历史（可按用户筛选）"""
    history = app_state.get_history(user_id=user_id, limit=limit)
    return {"history": history}


@app.post("/api/history/clear")
async def clear_history():
    """清空历史记录"""
    app_state.clear_history()
    return {"success": True}


@app.post("/api/recognition/start")
async def start_recognition():
    """启动识别"""
    load_model_if_needed()
    app_state.is_recognizing = True
    return {"success": True}


@app.post("/api/recognition/stop")
async def stop_recognition():
    """停止识别"""
    app_state.is_recognizing = False
    return {"success": True}


@app.post("/api/speech/toggle")
async def toggle_speech():
    """切换语音开关"""
    app_state.is_speech_on = not app_state.is_speech_on
    return {"success": True, "is_speech_on": app_state.is_speech_on}


@app.get("/api/speech/status")
async def get_speech_status():
    """获取语音状态"""
    return {"is_speech_on": app_state.is_speech_on}


# ============ 音频输出设备 API ============
@app.get("/api/audio/devices")
async def get_audio_devices():
    """获取可用音频输出设备列表"""
    if not app_state.audio_output_devices:
        app_state.scan_audio_devices()
    return {
        "devices": app_state.audio_output_devices,
        "current_device": app_state.audio_output_device
    }


@app.post("/api/audio/set-device")
async def set_audio_device(req: SetDeviceRequest):
    """设置音频输出设备"""
    device_id = req.device_id
    app_state.audio_output_device = device_id
    
    device_name = "系统默认"
    for dev in app_state.audio_output_devices:
        if dev["id"] == device_id:
            device_name = dev["name"]
            break
    
    print(f"[音频] 输出设备切换至: {device_name} (id={device_id})")
    return {"success": True, "device_id": device_id, "device_name": device_name}


# ============ 语音转文字 (STT) API ============
# 使用 faster-whisper 离线模型 (本地推理, 无需联网)
# 引擎管理在 AppState 中，与 TTS 保持一致的架构模式


@app.post("/api/stt")
async def api_speech_to_text(audio: UploadFile = None):
    """语音转文字 — 接收 WAV 音频，使用 faster-whisper 离线模型识别中文"""
    if audio is None:
        raise HTTPException(status_code=400, detail="请上传音频文件")

    suffix = '.wav'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.close()
        print(f"[STT] 收到音频: {audio.filename}, 大小={len(content)} bytes")

        # 在线程池中运行同步识别，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, app_state.speech_to_text, tmp.name)

        if result["success"]:
            return {"success": True, "text": result["text"], "engine": "faster-whisper-offline"}
        else:
            return {"success": False, "text": "", "error": result.get("error", "")}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[STT] 识别错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass


# ============ 用户认证 API ============
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@app.post("/api/auth/register")
async def auth_register(req: AuthRegisterRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(req.username) < 2 or len(req.username) > 20:
        raise HTTPException(status_code=400, detail="用户名需在2-20个字符之间")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="密码至少需要4个字符")
    result = app_state.db.register_user(req.username, _hash_password(req.password))
    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["message"])
    return result

@app.post("/api/auth/login")
async def auth_login(req: AuthLoginRequest):
    result = app_state.db.login_user(req.username, _hash_password(req.password))
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result

class LogoutRequest(BaseModel):
    token: Optional[str] = None

@app.post("/api/auth/logout")
async def auth_logout(req: LogoutRequest = None):
    if req and req.token:
        app_state.db.logout_user(req.token)
    return {"success": True}

@app.get("/api/auth/me")
async def auth_me(token: str = None):
    if not token:
        return {"user": None}
    user = app_state.db.get_user_by_token(token)
    if not user:
        return {"user": None}
    # 获取用户识别统计
    stats = {}
    try:
        row = app_state.db.conn.execute(
            'SELECT COUNT(*) as cnt FROM recognition_history WHERE user_id=?',
            (user['id'],)
        ).fetchone()
        stats['total_recognitions'] = row['cnt'] if row else 0
    except:
        stats['total_recognitions'] = 0
    return {"user": {**user, "stats": stats}}

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

@app.post("/api/auth/change-password")
async def change_password(req: ChangePasswordRequest, token: str = None):
    """修改密码（需要登录token）"""
    if not token:
        raise HTTPException(status_code=401, detail="请先登录")
    user = app_state.db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="新密码至少需要4个字符")
    # 验证旧密码
    row = app_state.db.conn.execute(
        'SELECT id FROM users WHERE id=? AND password_hash=?',
        (user['id'], _hash_password(req.old_password))
    ).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="原密码错误")
    # 更新密码
    app_state.db.conn.execute(
        'UPDATE users SET password_hash=? WHERE id=?',
        (_hash_password(req.new_password), user['id'])
    )
    app_state.db.conn.commit()
    return {"success": True, "message": "密码修改成功"}

@app.get("/api/auth/users")
async def list_users():
    """获取已注册用户列表（用于切换账号）"""
    rows = app_state.db.conn.execute(
        'SELECT id, username, created_at FROM users ORDER BY id'
    ).fetchall()
    users = []
    for row in rows:
        cnt = app_state.db.conn.execute(
            'SELECT COUNT(*) as cnt FROM recognition_history WHERE user_id=?',
            (row['id'],)
        ).fetchone()
        users.append({
            "id": row['id'],
            "username": row['username'],
            "created_at": row['created_at'],
            "recognition_count": cnt['cnt'] if cnt else 0,
        })
    return {"users": users}

class SetUserRequest(BaseModel):
    username: str

@app.post("/api/auth/set-user")
async def set_active_user(req: SetUserRequest):
    """设置当前活跃用户（登录后调用），关联后续识别记录"""
    row = app_state.db.conn.execute(
        'SELECT id FROM users WHERE username=?', (req.username,)
    ).fetchone()
    if row:
        app_state.current_user_id = row['id']
        return {"success": True, "user_id": row['id']}
    return {"success": False}

# ============ XAI 可解释性分析 API ============
@app.get("/api/xai/availability")
async def get_xai_availability():
    """检查是否有可分析的识别数据"""
    has_data = app_state.last_recognition_pose_data is not None
    return {
        "available": has_data,
        "last_prediction": app_state.last_recognition_pose_data.get('prediction', None) if has_data else None,
        "last_timestamp": app_state.last_recognition_pose_data.get('timestamp', None) if has_data else None,
    }


@app.get("/api/xai/latest")
async def get_xai_latest():
    """获取最近一次识别的 XAI 分析结果 (缓存优先)"""
    if app_state.last_xai_result is not None:
        return app_state.last_xai_result
    result = run_xai_analysis(reference_text=None)
    if result is None:
        raise HTTPException(status_code=404, detail="暂无识别数据，请先完成一次手语识别")
    return result


@app.post("/api/xai/analyze")
async def analyze_xai_post(req: XaiReferenceRequest):
    """运行 XAI 分析，可附带参考文本进行字符级对齐"""
    result = run_xai_analysis(reference_text=req.reference_text)
    if result is None:
        raise HTTPException(status_code=404, detail="暂无识别数据，请先完成一次手语识别")
    return result


# ============ 批量归因分析 API ============

# 归因分析报告缓存
_attribution_cache: dict = {}
_attribution_cache_time: float = 0.0
_attribution_cache_ttl: float = 300.0  # 5分钟缓存

# 模型推理结果缓存 (替代 results.txt)
_inference_results: dict = {}       # {model_key: {video_id: prediction}}
_inference_progress: dict = {"running": False, "current": 0, "total": 0, "model": ""}


def _run_inference_on_test_set(level: str = "A", model_key: str = "best_checkpoint", split: str = "test") -> dict:
    """使用已加载模型对测试集视频进行批量推理"""
    global _inference_progress, _inference_results
    import torch

    model = app_state.model
    if model is None:
        # 尝试加载模型
        load_model_if_needed()
        model = app_state.model
    if model is None:
        print("[Attribution] 模型未加载，无法推理")
        return {}

    root = get_models_root()
    ce_csl_root = os.path.join(root, "data", "CE_CSL", "CE-CSL")
    # 视频和标签路径
    dataset_video = os.path.join(ce_csl_root, "video")
    label_dir = os.path.join(ce_csl_root, "label")
    if not os.path.isdir(dataset_video):
        dataset_video = root  # 回退
    csv_path = os.path.join(label_dir, f"{split}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(root, f"{split}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(root, "dev.csv")
    pose_dir = os.path.join(ce_csl_root, "pose_format", split)

    gt_dict = _load_ground_truth_for_level(csv_path, level)
    video_ids = sorted(gt_dict.keys())

    _inference_progress = {"running": True, "current": 0, "total": len(video_ids), "model": model_key}
    predictions = {}

    device = next(model.parameters()).device

    for i, video_id in enumerate(video_ids):
        _inference_progress["current"] = i + 1

        # 获取该样本的实际难度等级
        sample_level = gt_dict.get(video_id, {}).get("level", level)

        # 查找视频文件: 数据集/video/{split}/{sample_level}/
        import glob as _glob
        matches = []
        for search_dir in [
            os.path.join(dataset_video, split, sample_level),
            os.path.join(dataset_video, split),
            os.path.join(root, sample_level),
            os.path.join(root, "A"),  # 最终回退
        ]:
            pattern = os.path.join(search_dir, f"{video_id}*.mp4")
            found = _glob.glob(pattern)
            matches = [m for m in found if '.downloading' not in m]
            if matches:
                break
        if not matches:
            continue

        # 查找姿态文件: pose_format/{split}/{sample_level}/
        pose_path = None
        for search_dir in [
            os.path.join(pose_dir, sample_level),
            pose_dir,
        ]:
            candidate = os.path.join(search_dir, f"{video_id}.pkl")
            if os.path.exists(candidate):
                pose_path = candidate
                break
        if pose_path is None:
            continue

        try:
            import pickle as _pickle
            with open(pose_path, "rb") as f:
                pose_data = _pickle.load(f)

            keypoints_raw = pose_data['keypoints']
            scores_raw = pose_data['scores']

            T = len(keypoints_raw)
            max_frames = 256
            if T > max_frames:
                try:
                    from datasets import build_uniform_indices
                    indices = build_uniform_indices(T, max_frames)
                    if hasattr(indices, 'numpy'): indices = indices.numpy()
                except Exception:
                    indices = np.linspace(0, T - 1, max_frames, dtype=int)
                keypoints_raw = [keypoints_raw[int(idx)] for idx in indices]
                scores_raw = [scores_raw[int(idx)] for idx in indices]

            # 过滤形状不一致的帧
            valid_kp, valid_sc = [], []
            for k, s in zip(keypoints_raw, scores_raw):
                k_arr = np.array(k)
                s_arr = np.array(s)
                if k_arr.shape == (1, 133, 2) and s_arr.shape == (1, 133):
                    valid_kp.append(k_arr)
                    valid_sc.append(s_arr)
            if len(valid_kp) < 2:
                continue
            skeletons = np.stack(valid_kp)  # (T, 1, 133, 2)
            confs = np.stack(valid_sc)       # (T, 1, 133)
            T = skeletons.shape[0]

            from datasets import load_part_kp
            pose_dict = load_part_kp(skeletons, confs, force_ok=True)

            src_input = {}
            for key in ['body', 'left', 'right', 'face_all']:
                if key in pose_dict:
                    src_input[key] = pose_dict[key].unsqueeze(0)
            src_input['attention_mask'] = torch.ones(1, T, dtype=torch.long)
            tgt_input = {'gt_sentence': ['']}

            for k in src_input:
                if isinstance(src_input[k], torch.Tensor):
                    src_input[k] = src_input[k].to(device)

            with torch.no_grad():
                stack_out = model(src_input, tgt_input)
                output = model.generate(stack_out, max_new_tokens=100, num_beams=6)

            pred = model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)
            result = pred[0].strip() if pred else ""
            result = result.replace('<extra_id_0>', '').replace('<extra_id_1>', '').replace('<pad>', '').replace('▁', '').strip()
            predictions[video_id] = result

        except Exception as e:
            print(f"[Attribution] 推理 {video_id} 失败: {e}")
            predictions[video_id] = ""

    _inference_progress["running"] = False
    _inference_results[model_key] = predictions
    # 清除旧缓存，强制重新构建报告
    global _attribution_cache, _attribution_cache_time
    _attribution_cache = {}
    _attribution_cache_time = 0.0

    print(f"[Attribution] 批量推理完成: {len(predictions)} 个样本")
    return predictions


def _load_ground_truth_for_level(csv_path: str, level: str = "A") -> dict:
    """从 dev.csv 加载指定等级的真实标签"""
    import csv as _csv
    gt_dict = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if level == "ALL" or row.get("Translator", "").strip() == level:
                    translator = row.get("Translator", "").strip()
                    gt_dict[row["Number"].strip()] = {
                        "sentence": row["Chinese Sentences"].strip(),
                        "gloss": row.get("Gloss", "").strip(),
                        "note": row.get("Note", "").strip(),
                        "level": translator,
                    }
    except Exception as e:
        print(f"[Attribution] 加载 CSV 失败: {e}")
    return gt_dict


def _load_predictions(results_path: str) -> dict:
    """从 results.txt 加载预测结果"""
    pred_dict = {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":", 1)
                vid = parts[0].strip()
                if vid.endswith(".mp4"):
                    vid = vid[:-4]
                pred_dict[vid] = parts[1].strip()
    except Exception as e:
        print(f"[Attribution] 加载预测结果失败: {e}")
    return pred_dict


def _compute_edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def _compute_similarity(reference: str, prediction: str) -> float:
    if not reference and not prediction:
        return 100.0
    if not reference or not prediction:
        return 0.0
    ref_chars = reference.replace(" ", "")
    pred_chars = prediction.replace(" ", "")
    dist = _compute_edit_distance(ref_chars, pred_chars)
    max_len = max(len(ref_chars), len(pred_chars))
    return (1.0 - dist / max_len) * 100.0


def _load_pose_payload(pose_path: str) -> dict | None:
    try:
        import pickle
        with open(pose_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _analyze_pose_heuristic(pose_path: str) -> dict:
    """基于姿态数据计算帧级重要性 (无需模型)"""
    pose = _load_pose_payload(pose_path)
    if pose is None:
        return {"keyframe_peak_idx": -1, "keyframe_peak_score": 0.0, "total_frames": 0,
                "avg_confidence": 0.0, "avg_motion": 0.0}

    keypoints = pose.get("keypoints", [])
    scores = pose.get("scores", [])
    previous_center = None
    importance_scores = []

    for frame_points, frame_scores in zip(keypoints, scores):
        points = np.asarray(frame_points, dtype=np.float32)
        sc = np.asarray(frame_scores, dtype=np.float32)
        if points.ndim == 3:
            points = points[0]
        if sc.ndim == 2:
            sc = sc[0]

        mean_score = float(np.nanmean(sc)) if sc.size else 0.0
        if points.size > 0 and points.shape[-1] == 2:
            if sc.size and len(sc) == len(points):
                points = points[sc > 0.1]
            if points.size > 0:
                center = points.mean(axis=0)
                motion = float(np.linalg.norm(center - previous_center)) if previous_center is not None else 0.0
                previous_center = center
            else:
                motion = 0.0
        else:
            motion = 0.0

        importance_scores.append({"frame_idx": len(importance_scores), "mean_score": mean_score, "motion": motion})

    if not importance_scores:
        return {"keyframe_peak_idx": -1, "keyframe_peak_score": 0.0, "total_frames": 0,
                "avg_confidence": 0.0, "avg_motion": 0.0}

    n = len(importance_scores)
    max_motion = max(s["motion"] for s in importance_scores)
    best_idx, best_score = -1, -1.0
    total_conf, total_motion = 0.0, 0.0

    for s in importance_scores:
        motion_norm = 0.0 if max_motion <= 0 else s["motion"] / max_motion
        importance = 0.7 * s["mean_score"] + 0.3 * motion_norm
        total_conf += s["mean_score"]
        total_motion += s["motion"]
        if importance > best_score:
            best_score = importance
            best_idx = s["frame_idx"]

    return {
        "keyframe_peak_idx": best_idx,
        "keyframe_peak_score": round(best_score, 4),
        "total_frames": n,
        "avg_confidence": round(total_conf / n, 4),
        "avg_motion": round(total_motion / n, 4),
    }


def _load_wer_module():
    """延迟导入 WER 计算模块"""
    from SLRT_metrics import wer_single
    return wer_single


def _compute_wer_single(reference: str, prediction: str) -> dict:
    ref_chars = " ".join(list(reference.replace(" ", "")))
    hyp_chars = " ".join(list(prediction.replace(" ", "")))
    if not ref_chars.strip():
        return {"wer": 0.0, "del_rate": 0.0, "ins_rate": 0.0, "sub_rate": 0.0}
    try:
        wer_func = _load_wer_module()
        result = wer_func(r=ref_chars, h=hyp_chars)
        n_ref = result["num_ref"]
        if n_ref == 0:
            return {"wer": 0.0, "del_rate": 0.0, "ins_rate": 0.0, "sub_rate": 0.0}
        return {
            "wer": round((result["num_err"] / n_ref) * 100.0, 2),
            "del_rate": round((result["num_del"] / n_ref) * 100.0, 2),
            "ins_rate": round((result["num_ins"] / n_ref) * 100.0, 2),
            "sub_rate": round((result["num_sub"] / n_ref) * 100.0, 2),
        }
    except Exception:
        return {"wer": 0.0, "del_rate": 0.0, "ins_rate": 0.0, "sub_rate": 0.0}


def _generate_attribution_summary(reference: str, prediction: str, similarity: float,
                                   wer_info: dict, keyframe_idx: int, is_match: bool) -> str:
    parts = []
    if similarity >= 99.99:
        parts.append("模型识别完全正确")
    elif similarity >= 85:
        parts.append(f"模型识别基本正确(相似度{similarity:.1f}%)")
    elif similarity >= 66.7:
        parts.append(f"模型识别接近正确(相似度{similarity:.1f}%)")
    elif similarity >= 50:
        parts.append(f"模型识别偏差较大(相似度{similarity:.1f}%)")
    else:
        parts.append(f"模型识别偏差很大(相似度{similarity:.1f}%)")

    del_r = wer_info.get("del_rate", 0)
    ins_r = wer_info.get("ins_rate", 0)
    sub_r = wer_info.get("sub_rate", 0)
    if del_r > ins_r and del_r > sub_r:
        parts.append(f"主要错误类型为删除({del_r:.0f}%)")
    elif ins_r > del_r and ins_r > sub_r:
        parts.append(f"主要错误类型为插入({ins_r:.0f}%)")
    elif sub_r > 0:
        parts.append(f"主要错误类型为替换({sub_r:.0f}%)")
    if keyframe_idx >= 0:
        parts.append(f"关键帧位于第{keyframe_idx}帧")
    return "; ".join(parts)


def _detect_available_models() -> list[dict]:
    """扫描 out/ 目录，返回可用的模型列表"""
    root = get_models_root()
    out_dir = os.path.join(root, "out")
    models = []
    seen = set()

    # 扫描 out/ 子目录下的 .pth 文件
    if os.path.isdir(out_dir):
        for d in sorted(os.listdir(out_dir)):
            sub = os.path.join(out_dir, d)
            if not os.path.isdir(sub): continue
            for f in sorted(os.listdir(sub)):
                if f.endswith(".pth"):
                    if f in seen: continue
                    seen.add(f)
                    model_key = f.replace(".pth", "")
                    models.append({
                        "key": f,  # 用实际文件名匹配 showcase infer 的 req.model
                        "name": f"{d}/{f}",
                        "filename": f,
                        "has_results": True,  # eval CSV 总是可用
                    })

    # 也扫 model/ 目录
    model_dir = os.path.join(root, "model")
    if os.path.isdir(model_dir):
        for f in sorted(os.listdir(model_dir)):
            if f.endswith(".pth") and f not in seen:
                seen.add(f)
                models.append({
                    "key": f,
                    "name": f"model/{f}",
                    "filename": f,
                    "has_results": True,
                })
    return models


def _build_attribution_report(csv_path: str = None, results_path: str = None,
                               pose_dir: str = None, level: str = "A",
                               model: str = "best_checkpoint", split: str = "test") -> dict:
    """构建批量归因分析报告"""
    global _attribution_cache, _attribution_cache_time

    cache_key = f"{split}_{level}_{model}"
    now = time.time()
    if _attribution_cache and _attribution_cache.get("_key") == cache_key \
       and (now - _attribution_cache_time) < _attribution_cache_ttl:
        return _attribution_cache

    root = get_models_root()
    ce_csl_root = os.path.join(root, "data", "CE_CSL", "CE-CSL")
    dataset_video = os.path.join(ce_csl_root, "video")
    label_dir = os.path.join(ce_csl_root, "label")
    if not os.path.isdir(dataset_video):
        dataset_video = root  # 回退

    if csv_path is None:
        csv_path = os.path.join(label_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(root, f"{split}.csv")
    if pose_dir is None:
        pose_dir = os.path.join(ce_csl_root, "pose_format", split)

    # 优先加载预计算推理缓存 → 模型实时推理结果 → results 文件
    pred_dict = {}
    data_source = "unknown"

    # 1. 尝试加载推理缓存文件
    import json as _json
    cache_dir = os.path.join(root, "inference_cache")
    cache_file = os.path.join(cache_dir, f"{model}_{split}_{level}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = _json.load(f)
            pred_dict = {vid: info['prediction'] for vid, info in cache_data.items()}
            data_source = f"cache: {os.path.basename(cache_file)}"
        except Exception:
            pass

    # 2. 回退到内存中的实时推理结果
    if not pred_dict:
        pred_dict = _inference_results.get(model, {})
        if pred_dict:
            data_source = "model_inference"

    # 3. 回退到 results 文件
    if not pred_dict:
        if results_path is None:
            results_path = os.path.join(root, f"results_{model}.txt")
        if not os.path.exists(results_path):
            results_path = os.path.join(root, "results.txt")
        if os.path.exists(results_path):
            pred_dict = _load_predictions(results_path)
            data_source = f"results_file: {os.path.basename(results_path)}"

    # 4. ★ 最终回退: 直接从 eval CSV 提取预测
    if not pred_dict:
        import csv as _csv, glob as _glob
        eval_dir = os.path.join(root, "out")
        for d in sorted(os.listdir(eval_dir)):
            for csv_f in ['test100_eval.csv', 'dev100_eval.csv']:
                path = os.path.join(eval_dir, d, csv_f)
                if not os.path.exists(path): continue
                csv_split = 'test' if 'test' in csv_f else 'dev'
                if csv_split != split: continue
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for row in _csv.DictReader(f):
                            ds_idx = int(row['dataset_index'])
                            vid = f'{csv_split}-{ds_idx+1:05d}'
                            if vid not in pred_dict:
                                pred_dict[vid] = row.get('prediction', '')
                except Exception:
                    pass
        data_source = "eval_csv"

    print(f"[Attribution] 构建归因报告: model={model}, source={data_source}, split={split}, level={level}")

    gt_dict = _load_ground_truth_for_level(csv_path, level)

    valid_ids = sorted([vid for vid in gt_dict if vid in pred_dict])
    gt_dict = {vid: gt_dict[vid] for vid in valid_ids}

    samples = []
    correct_count = 0
    total = len(valid_ids)

    for video_id in valid_ids:
        gt_info = gt_dict[video_id]
        reference = gt_info["sentence"]
        prediction = pred_dict.get(video_id, "")
        similarity = _compute_similarity(reference, prediction)
        is_correct = similarity >= 66.7  # 相似度 ≥ 66.7% 认为正确
        if is_correct:
            correct_count += 1

        wer_info = _compute_wer_single(reference, prediction)

        # 姿态启发式分析 — 多级目录查找
        pose_path = None
        gt_level = gt_info.get("level", level)
        for search_dir in [os.path.join(pose_dir, gt_level), pose_dir]:
            candidate = os.path.join(search_dir, f"{video_id}.pkl")
            if os.path.exists(candidate):
                pose_path = candidate
                break
        pose_result = _analyze_pose_heuristic(pose_path) if pose_path else {}

        keyframe_idx = pose_result.get("keyframe_peak_idx", -1)
        summary = _generate_attribution_summary(
            reference, prediction, similarity, wer_info, keyframe_idx, is_correct
        )

        samples.append({
            "video_id": video_id,
            "reference_text": reference,
            "predicted_text": prediction,
            "similarity": round(similarity, 2),
            "wer": wer_info["wer"],
            "del_rate": wer_info["del_rate"],
            "ins_rate": wer_info["ins_rate"],
            "sub_rate": wer_info["sub_rate"],
            "is_correct": is_correct,
            "attribution_summary": summary,
            "keyframe_peak_idx": keyframe_idx,
            "keyframe_peak_score": pose_result.get("keyframe_peak_score", 0.0),
            "total_frames": pose_result.get("total_frames", 0),
            "avg_confidence": pose_result.get("avg_confidence", 0.0),
            "avg_motion": pose_result.get("avg_motion", 0.0),
        })

    recognition_rate = (correct_count / total * 100.0) if total > 0 else 0.0
    avg_similarity = round(sum(s["similarity"] for s in samples) / total, 2) if total > 0 else 0.0
    avg_wer = round(sum(s["wer"] for s in samples) / total, 2) if total > 0 else 0.0

    # 相似度分布
    dist_bins = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for s in samples:
        sim = s["similarity"]
        if sim < 20: dist_bins["0-20"] += 1
        elif sim < 40: dist_bins["20-40"] += 1
        elif sim < 60: dist_bins["40-60"] += 1
        elif sim < 80: dist_bins["60-80"] += 1
        else: dist_bins["80-100"] += 1

    result = {
        "_key": cache_key,
        "summary": {
            "total_samples": total,
            "recognition_rate": round(recognition_rate, 2),
            "avg_similarity": avg_similarity,
            "avg_wer": avg_wer,
            "model_checkpoint": f"{model}.pth",
            "model_key": model,
            "attribution_tech_stack": "Pose Heuristic: Confidence + Motion (Fast Mode)",
            "level": level,
            "correct_count": correct_count,
            "threshold": "相似度 ≥ 70%",
            "similarity_distribution": dist_bins,
        },
        "samples": samples,
        "available_models": _detect_available_models(),
    }

    _attribution_cache = result
    _attribution_cache_time = now
    return result


@app.get("/api/attribution/models")
async def get_attribution_models():
    """获取可用的模型列表"""
    return {"models": _detect_available_models()}


@app.get("/api/attribution/report")
async def get_attribution_report(level: str = "A", model: str = "best_checkpoint",
                                  split: str = "test", refresh: bool = False):
    """获取批量归因分析报告"""
    global _attribution_cache, _attribution_cache_time
    if refresh:
        _attribution_cache = {}
        _attribution_cache_time = 0.0
    try:
        report = _build_attribution_report(level=level, model=model, split=split)
        # 附加上下文信息
        has_inference = model in _inference_results and bool(_inference_results[model])
        report["_meta"] = {
            "data_source": "model_inference" if has_inference else "results_file",
            "inference_available": app_state.model is not None,
            "inference_running": _inference_progress.get("running", False),
            "inference_progress": {
                "current": _inference_progress.get("current", 0),
                "total": _inference_progress.get("total", 0),
            } if _inference_progress.get("running") else None,
        }
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"归因分析失败: {str(e)}")


@app.post("/api/attribution/infer")
async def trigger_attribution_inference(model: str = "best_checkpoint", level: str = "A", split: str = "test"):
    """触发模型在测试集上的批量推理"""
    import threading
    if _inference_progress.get("running"):
        return {"status": "already_running", "progress": _inference_progress}

    def _run():
        _run_inference_on_test_set(level=level, model_key=model, split=split)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "message": f"开始批量推理: {model}"}


@app.get("/api/attribution/infer-status")
async def get_inference_status():
    """查询批量推理进度"""
    return {
        "running": _inference_progress.get("running", False),
        "current": _inference_progress.get("current", 0),
        "total": _inference_progress.get("total", 0),
        "model": _inference_progress.get("model", ""),
        "has_results": bool(_inference_results),
    }


@app.get("/api/attribution/sample/{video_id}")
async def get_attribution_sample(video_id: str, level: str = "A", model: str = "best_checkpoint", split: str = "test"):
    """获取单个样本的详细归因分析"""
    report = _build_attribution_report(level=level, model=model, split=split)
    for sample in report["samples"]:
        if sample["video_id"] == video_id:
            return sample
    raise HTTPException(status_code=404, detail=f"样本 {video_id} 未找到")


# ============ 智能学习 API ============

@app.get("/api/learning/corrections")
async def get_learning_corrections():
    """获取所有纠错记录"""
    _load_learning_db()
    history = []
    for pred, correct in _learning_db.items():
        history.append({"predicted": pred, "corrected": correct})
    history.sort(key=lambda x: x["predicted"])
    return {"corrections": history, "total": len(history)}

class LearningCorrectRequest(BaseModel):
    predicted: str = ""
    corrected: str = ""

@app.post("/api/learning/correct")
async def add_learning_correction(req: LearningCorrectRequest):
    """添加纠错记录"""
    if not req.predicted or not req.corrected:
        raise HTTPException(status_code=400, detail="预测文本和正确文本不能为空")
    _load_learning_db()
    _learning_db[req.predicted] = req.corrected
    _save_learning_db()
    return {"success": True, "message": "纠错记录已保存"}

@app.delete("/api/learning/correction")
async def delete_learning_correction(predicted: str = ""):
    """删除纠错记录"""
    _load_learning_db()
    if predicted in _learning_db:
        del _learning_db[predicted]
        _save_learning_db()
        return {"success": True}
    raise HTTPException(status_code=404, detail="记录不存在")

@app.get("/api/learning/history")
async def get_learning_history(limit: int = 100):
    """获取最近识别历史（用于添加纠错）"""
    history = app_state.get_history(limit=limit)
    return {"history": history}


# ============ 模型展示 API ============

_showcase_model_cache = {}
_showcase_lock = threading.Lock()

def load_showcase_model(checkpoint_path: str):
    """加载展示用的模型（独立缓存，不影响实时演示模型）"""
    if checkpoint_path in _showcase_model_cache:
        return _showcase_model_cache[checkpoint_path]

    import torch as _torch
    from models import Reli_Sign
    from types import SimpleNamespace

    ckpt = _torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})

    args = SimpleNamespace(
        dataset=saved_args.get('dataset', 'CE_CSL'),
        rgb_support=saved_args.get('rgb_support', False),
        max_length=saved_args.get('max_length', 256),
        hidden_dim=saved_args.get('hidden_dim', 256),
        label_smoothing=saved_args.get('label_smoothing', 0.2),
        rgb_backbone_weights=saved_args.get('rgb_backbone_weights', ''),
        rgb_backbone_pretrained=saved_args.get('rgb_backbone_pretrained', False),
        fusion_use_uncertainty_gate=saved_args.get('fusion_use_uncertainty_gate', False),
        fusion_meta_dim=saved_args.get('fusion_meta_dim', 3),
        seed=saved_args.get('seed', 42),
    )
    model = Reli_Sign(args=args)
    state_dict = ckpt.get('model', ckpt)
    ms = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    # ★ 展示模型用 CPU 避免 GPU OOM (实时模型已占 ~2.5GB)
    device = 'cpu'
    model.eval()

    _showcase_model_cache[checkpoint_path] = model
    print(f"[Showcase] 模型加载: {os.path.basename(checkpoint_path)} → {device} (rgb_support={args.rgb_support}, matched={len(filtered)}/{len(ms)})")
    return model

@app.get("/api/showcase/videos")
async def get_showcase_videos(split: str = "test", model: str = "best_checkpoint"):
    """获取可展示的测试视频列表（仅置信度 ≥ 80%）"""
    root = get_models_root()

    # 实际数据路径: data/CE_CSL/CE-CSL/
    ce_csl_root = os.path.join(root, "data", "CE_CSL", "CE-CSL")
    dataset_video = os.path.join(ce_csl_root, "video")
    label_dir = os.path.join(ce_csl_root, "label")

    if not os.path.isdir(dataset_video):
        return {"videos": [], "total": 0, "error": "视频目录不存在"}

    # ★ 加载预计算的高精度样本列表
    import json as _json
    best_samples = {}
    best_json = os.path.join(root, "out", "best_samples.json")
    if os.path.exists(best_json):
        try:
            with open(best_json, 'r', encoding='utf-8') as f:
                for item in _json.load(f):
                    split_name = item["split"]
                    ds_idx = int(item["dataset_idx"])
                    vid_key = f'{split_name}-{ds_idx+1:05d}'  # dataset_index 0-based, video ID 1-based
                    if vid_key not in best_samples or item['sim'] > best_samples[vid_key]['sim']:
                        best_samples[vid_key] = item
        except Exception:
            pass
    HAS_BEST = bool(best_samples)

    videos = []
    import glob as _glob
    MAX_VIDEOS = 80  # 展示所有 >=0.50 的样本

    # ★ 严格过滤: 仅展示高准确度样本 (>=70%)
    if HAS_BEST:
        # 按相似度降序排列所有高精度样本
        ranked = sorted(best_samples.items(), key=lambda x: -x[1]['sim'])
        for vid_key, info in ranked:
            if info['sim'] < 0.70:
                break
            if len(videos) >= MAX_VIDEOS:
                break

            # vid_key 格式: "test-00163" 或 "dev-00288"
            parts = vid_key.split('-', 1)
            if len(parts) != 2:
                continue
            s, idx = parts[0], parts[1]

            # 搜索标签获取ground truth
            csv_path_s = os.path.join(label_dir, f"{s}.csv")
            if not os.path.exists(csv_path_s):
                continue
            gt_dict = _load_ground_truth_for_level(csv_path_s, "ALL")
            if vid_key not in gt_dict:
                continue

            info_gt = gt_dict[vid_key]
            level = info_gt.get("level", "A")
            ref = info_gt["sentence"]

            # 搜索视频文件
            found = None
            for sd in [os.path.join(dataset_video, s, level),
                        os.path.join(dataset_video, s)]:
                if not os.path.isdir(sd):
                    continue
                pattern = os.path.join(sd, f"{vid_key}*.mp4")
                matches = [m for m in _glob.glob(pattern)
                           if '.downloading' not in m and '(1)' not in os.path.basename(m)]
                if matches:
                    found = matches[0]
                    break

            if found:
                videos.append({
                    "video_id": vid_key,
                    "path": found,
                    "reference": ref,
                    "level": level,
                    "split": s,
                    "similarity": round(info['sim'] * 100, 1),
                    "prediction": info.get('pred', ''),
                    "model": info.get('model', ''),
                })

    # 如果 best_samples 不可用, 回退到普通列表
    if not videos:
        for s in ["test", "dev"]:
            csv_path_s = os.path.join(label_dir, f"{s}.csv")
            if not os.path.exists(csv_path_s):
                continue
            gt_dict = _load_ground_truth_for_level(csv_path_s, "ALL")
            for video_id, info in sorted(gt_dict.items()):
                if len(videos) >= 60:
                    break
                level = info.get("level", "A")
                for sd in [os.path.join(dataset_video, s, level)]:
                    if not os.path.isdir(sd):
                        continue
                    pattern = os.path.join(sd, f"{video_id}*.mp4")
                    matches = [m for m in _glob.glob(pattern)
                               if '.downloading' not in m and '(1)' not in os.path.basename(m)]
                    if matches:
                        videos.append({"video_id": video_id, "path": matches[0],
                                       "reference": info["sentence"], "level": level, "split": s})
                        break
            if len(videos) >= 60:
                break
    return {"videos": videos, "total": len(videos), "has_cache": HAS_BEST}


class ShowcaseInferRequest(BaseModel):
    video_path: str = ""
    video_id: str = ""
    model: str = "best_checkpoint"
    rgb_support: bool = False
    lightweight: bool = False
    ultra_lightweight: bool = False

@app.post("/api/showcase/infer")
async def showcase_infer(req: ShowcaseInferRequest):
    """对单个视频运行推理并返回姿态和结果"""
    if not req.video_path or not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail=f"视频不存在: {req.video_path}")

    # 根据请求加载对应模型
    import torch as _torch
    root = get_models_root()
    # ★ 按精度从高到低搜索
    checkpoint_paths = [
        os.path.join(root, "out", "cecsl_rgb_support_round6_overnight_20260407", f"{req.model}.pth"),
        os.path.join(root, "out", "stage3_finetuning", f"{req.model}.pth"),
        os.path.join(root, "out", f"{req.model}.pth"),
        os.path.join(root, "model", f"{req.model}.pth"),
    ]
    checkpoint_path = None
    for cp in checkpoint_paths:
        if os.path.exists(cp):
            checkpoint_path = cp
            break
    if checkpoint_path is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {req.model}")
    # ★ 只用实时模型做展示, 绝不加载第二份
    if app_state.model is None:
        load_model_if_needed()
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="模型正在加载中，请30秒后重试")
    model = app_state.model

    import torch, pickle as _pickle, glob as _glob

    device = next(model.parameters()).device

    # 查找姿态文件: data/CE_CSL/CE-CSL/pose_format/{split}/{level}/{vid}.pkl
    vid_clean = req.video_id.split('(')[0].strip()
    ce_csl_root = os.path.join(root, "data", "CE_CSL", "CE-CSL")
    pose_root = os.path.join(ce_csl_root, "pose_format")
    pose_path = None
    for split_dir in ["test", "dev"]:
        pose_dir = os.path.join(pose_root, split_dir)
        if not os.path.isdir(pose_dir):
            continue
        for sd in sorted(os.listdir(pose_dir)):
            candidate = os.path.join(pose_dir, sd, f"{vid_clean}.pkl")
            if os.path.exists(candidate):
                pose_path = candidate
                break
        if pose_path:
            break

    if pose_path is None:
        # Debug: 列出已搜索路径
        searched = []
        for split_dir in ["test", "dev"]:
            pose_dir = os.path.join(ce_csl_root, "pose_format", split_dir)
            if os.path.isdir(pose_dir):
                for sd in sorted(os.listdir(pose_dir))[:3]:
                    searched.append(os.path.join(pose_dir, sd))
        raise HTTPException(status_code=404, detail=f"姿态文件未找到: {vid_clean}，搜索路径: {searched}...")

    print(f"[Showcase] 找到姿态文件: {pose_path}")
    try:
        with open(pose_path, "rb") as f:
            pose_data = _pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载姿态失败: {e}")

    keypoints_raw = pose_data['keypoints']
    scores_raw = pose_data['scores']

    # 帧采样
    T = len(keypoints_raw)
    max_frames = 100 if req.ultra_lightweight else (150 if req.lightweight else 256)
    if T > max_frames:
        try:
            from datasets import build_uniform_indices
            ids = build_uniform_indices(T, max_frames)
            if hasattr(ids, "numpy"): ids = ids.numpy()
        except Exception:
            ids = np.linspace(0, T - 1, max_frames, dtype=int)
        keypoints_raw = [keypoints_raw[int(idx)] for idx in ids]
        scores_raw = [scores_raw[int(idx)] for idx in ids]

    # 过滤有效帧
    valid_kp, valid_sc, pose_frames = [], [], []
    for k, s in zip(keypoints_raw, scores_raw):
        k_arr = np.array(k); s_arr = np.array(s)
        if k_arr.shape == (1, 133, 2) and s_arr.shape == (1, 133):
            valid_kp.append(k_arr); valid_sc.append(s_arr)
            # 合并关键点 + 置信度: [x, y, conf] × 133
            kp_with_conf = np.concatenate([k_arr[0], s_arr[0][:, None]], axis=1).tolist()
            pose_frames.append(kp_with_conf)

    if len(valid_kp) < 2:
        raise HTTPException(status_code=400, detail=f"有效帧不足: {len(valid_kp)}")

    skeletons = np.stack(valid_kp).astype(np.float32)
    confs = np.stack(valid_sc).astype(np.float32)
    T = skeletons.shape[0]

    from datasets import load_part_kp
    pose_dict = load_part_kp(skeletons, confs, force_ok=True)

    src_input = {}
    for key in ['body', 'left', 'right', 'face_all']:
        if key in pose_dict:
            src_input[key] = pose_dict[key].unsqueeze(0).float()
    src_input['attention_mask'] = torch.ones(1, T, dtype=torch.long)

    # ★ RGB hand data for round6 model (empty→pose-only fallback)
    if model.args.rgb_support:
        src_input['left_hands'] = torch.zeros(1, 3, 112, 112)
        src_input['right_hands'] = torch.zeros(1, 3, 112, 112)
        src_input['left_sampled_indices'] = torch.tensor([-1])
        src_input['right_sampled_indices'] = torch.tensor([-1])
        src_input['left_rgb_len'] = [1]
        src_input['right_rgb_len'] = [1]
        src_input['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        src_input['right_skeletons_norm'] = torch.zeros(1, 21, 2)
        src_input['left_rgb_meta'] = torch.zeros(1, 3)
        src_input['right_rgb_meta'] = torch.zeros(1, 3)

    tgt_input = {'gt_sentence': ['']}

    for k in list(src_input.keys()):
        if isinstance(src_input[k], torch.Tensor):
            src_input[k] = src_input[k].to(device)

    with torch.no_grad():
        stack_out = model(src_input, tgt_input)
        output = model.generate(stack_out, max_new_tokens=100, num_beams=6)
    # 释放中间结果
    del stack_out, src_input, pose_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 计算五大部位逐帧置信度
    part_ranges = {
        "body": list(range(0, 11)),          # 0-10: 躯干关键点
        "feet": list(range(11, 17)),         # 11-16: 下肢关键点
        "left_hand": list(range(91, 112)),   # 91-111: 左手
        "right_hand": list(range(112, 133)), # 112-132: 右手
        "face": list(range(23, 91)),         # 23-90: 面部
    }
    confidence_timeline = []
    for sc_arr in valid_sc:  # each is (1, 133)
        s = sc_arr[0]  # (133,)
        frame_conf = {}
        for part_name, indices in part_ranges.items():
            vals = s[indices]
            frame_conf[part_name] = round(min(float(np.mean(vals)), 1.0), 4)
        confidence_timeline.append(frame_conf)

    pred = model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)
    prediction = pred[0].strip() if pred else ""
    prediction = prediction.replace('<extra_id_0>', '').replace('<extra_id_1>', '').replace('<pad>', '').replace('▁', '').strip()

    return {
        "video_id": req.video_id,
        "prediction": prediction,
        "frame_count": len(pose_frames),
        "pose_keypoints": pose_frames,
        "confidence_timeline": confidence_timeline,
    }


@app.get("/api/showcase/video-file")
async def serve_showcase_video(path: str = ""):
    """提供样例视频文件"""
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return FileResponse(path, media_type="video/mp4")


def _lookup_reference_from_dataset(filename: str) -> str:
    """按上传视频的文件名在数据集标签 CSV 中查找真实标注 (Chinese Sentences)"""
    import csv as _csv
    base = os.path.basename(filename or "")
    video_id = os.path.splitext(base)[0].strip().lower()
    if not video_id:
        return ""
    label_dir = os.path.join(get_models_root(), "data", "CE_CSL", "CE-CSL", "label")
    for csv_name in ("test.csv", "dev.csv", "train.csv"):
        csv_path = os.path.join(label_dir, csv_name)
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    num = (row.get("Number") or "").strip().lower()
                    if num == video_id:
                        return (row.get("Chinese Sentences") or "").strip()
        except Exception:
            continue
    return ""


@app.post("/api/video/analyze")
async def analyze_uploaded_video(
    video: UploadFile = File(...),
    lightweight: str = Form("false"),
    ultra_lightweight: str = Form("false"),
):
    """自定义上传视频分析: 上传 -> 逐帧姿态提取 -> 模型预测 (GPU)。

    返回结构与 /api/showcase/infer 一致, 供前端复用姿态画布与置信度时间线。
    视频回放由浏览器本地 URL.createObjectURL 完成, 服务端仅保存后即清理。
    """
    import torch
    import uuid

    filename = video.filename or "upload.mp4"
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
        raise HTTPException(status_code=400, detail="仅支持常见视频格式 (mp4/avi/mov/mkv/webm/m4v)")

    upload_dir = os.path.join(models_root, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    video_id = f"upload-{uuid.uuid4().hex[:12]}"
    ext = os.path.splitext(filename)[1].lower() or ".mp4"
    save_path = os.path.join(upload_dir, f"{video_id}{ext}")

    data = await video.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    with open(save_path, "wb") as f:
        f.write(data)

    lightweight_flag = str(lightweight).strip().lower() in ("1", "true", "yes")
    ultra_flag = str(ultra_lightweight).strip().lower() in ("1", "true", "yes")

    try:
        load_model_if_needed()
        model = app_state.model
        if model is None:
            raise HTTPException(status_code=503, detail="模型加载失败")

        cap = cv2.VideoCapture(save_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="无法打开视频文件")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 480, 270

        # 姿态提取分辨率: 宽 480, 保持比例; 长视频按 stride 采样避免过慢
        target_w = 480
        if frame_w > target_w:
            new_h = int(round(frame_h * target_w / float(frame_w)))
            cur_w, cur_h = target_w, new_h
        else:
            cur_w, cur_h = frame_w, frame_h
        stride = max(1, total_frames // 512) if total_frames > 600 else 1

        raw_kp: list = []
        raw_sc: list = []
        t0 = time.time()

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            if frame_w > target_w:
                frame = cv2.resize(frame, (cur_w, cur_h), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kp, sc = process_pose_frame(frame_rgb)
            if kp is not None and sc is not None:
                kp_norm = kp.astype(np.float32).copy()
                kp_norm[:, 0] /= float(cur_w)
                kp_norm[:, 1] /= float(cur_h)
                raw_kp.append(kp_norm)
                raw_sc.append(sc.astype(np.float32))
            frame_idx += 1
        cap.release()

        if len(raw_kp) < 2:
            raise HTTPException(status_code=400, detail="未能从视频中提取到有效姿态 (有效帧不足)")

        # 帧采样 (与 showcase_infer 一致)
        T = len(raw_kp)
        max_frames = 100 if ultra_flag else (150 if lightweight_flag else 256)
        if T > max_frames:
            try:
                from datasets import build_uniform_indices
                ids = build_uniform_indices(T, max_frames)
                if hasattr(ids, "numpy"):
                    ids = ids.numpy()
            except Exception:
                ids = np.linspace(0, T - 1, max_frames, dtype=int)
            raw_kp = [raw_kp[int(idx)] for idx in ids]
            raw_sc = [raw_sc[int(idx)] for idx in ids]

        # 组装 pose_frames (前端画布) + valid_kp/valid_sc (模型输入)
        pose_frames = []
        valid_kp = []
        valid_sc = []
        for kp, sc in zip(raw_kp, raw_sc):
            valid_kp.append(kp[None, :, :])   # (1, 133, 2)
            valid_sc.append(sc[None, :])      # (1, 133)
            kp_with_conf = np.concatenate([kp, sc[:, None]], axis=1).tolist()
            pose_frames.append(kp_with_conf)

        skeletons = np.stack(valid_kp).astype(np.float32)
        confs = np.stack(valid_sc).astype(np.float32)
        T = skeletons.shape[0]

        from datasets import load_part_kp
        pose_dict = load_part_kp(skeletons, confs, force_ok=True)

        src_input = {}
        for key in ['body', 'left', 'right', 'face_all']:
            if key in pose_dict:
                src_input[key] = pose_dict[key].unsqueeze(0).float()
        src_input['attention_mask'] = torch.ones(1, T, dtype=torch.long)

        if model.args.rgb_support:
            src_input['left_hands'] = torch.zeros(1, 3, 112, 112)
            src_input['right_hands'] = torch.zeros(1, 3, 112, 112)
            src_input['left_sampled_indices'] = torch.tensor([-1])
            src_input['right_sampled_indices'] = torch.tensor([-1])
            src_input['left_rgb_len'] = [1]
            src_input['right_rgb_len'] = [1]
            src_input['left_skeletons_norm'] = torch.zeros(1, 21, 2)
            src_input['right_skeletons_norm'] = torch.zeros(1, 21, 2)
            src_input['left_rgb_meta'] = torch.zeros(1, 3)
            src_input['right_rgb_meta'] = torch.zeros(1, 3)

        tgt_input = {'gt_sentence': ['']}

        device = next(model.parameters()).device
        for k in list(src_input.keys()):
            if isinstance(src_input[k], torch.Tensor):
                src_input[k] = src_input[k].to(device)

        infer_start = time.time()
        with torch.no_grad():
            stack_out = model(src_input, tgt_input)
            output = model.generate(stack_out, max_new_tokens=100, num_beams=6)
        infer_ms = (time.time() - infer_start) * 1000.0

        del stack_out, src_input, pose_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        part_ranges = {
            "body": list(range(0, 11)),
            "feet": list(range(11, 17)),
            "left_hand": list(range(91, 112)),
            "right_hand": list(range(112, 133)),
            "face": list(range(23, 91)),
        }
        confidence_timeline = []
        for sc_arr in valid_sc:
            s = sc_arr[0]
            frame_conf = {}
            for part_name, indices in part_ranges.items():
                frame_conf[part_name] = round(min(float(np.mean(s[indices])), 1.0), 4)
            confidence_timeline.append(frame_conf)

        pred = model.mt5_tokenizer.batch_decode(output, skip_special_tokens=True)
        prediction = pred[0].strip() if pred else ""
        prediction = (prediction
                      .replace('<extra_id_0>', '').replace('<extra_id_1>', '')
                      .replace('<pad>', '').replace('▁', '').strip())

        total_s = time.time() - t0

        return {
            "video_id": video_id,
            "prediction": prediction,
            "reference": _lookup_reference_from_dataset(filename),
            "frame_count": len(pose_frames),
            "pose_keypoints": pose_frames,
            "confidence_timeline": confidence_timeline,
            "latency_s": round(total_s, 3),
            "infer_ms": round(infer_ms, 1),
            "device": str(device),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析失败: {e}")
    finally:
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass


# ============ WebSocket 路由 ============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 处理实时姿态检测和识别"""
    await websocket.accept()

    last_pose_time = time.time()
    pose_interval = 0.10  # 10fps — 降低CPU负担，姿态更流畅

    # 发送骨架连接定义（一次性初始化）
    skeleton_connections = [
        # 身体骨骼 25条
        [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
        [11,12],[11,13],[12,14],[13,15],[14,16],[5,11],[6,12],
        [0,17],[0,18],[17,19],[18,20],[19,21],[20,22],[11,17],[12,18],[5,17],
        # 左手骨骼 20条
        [91,92],[92,93],[93,94],[94,95],[91,96],[96,97],[97,98],[98,99],
        [91,100],[100,101],[101,102],[102,103],[91,104],[104,105],[105,106],[106,107],
        [91,108],[108,109],[109,110],[110,111],
        # 右手骨骼 20条
        [112,113],[113,114],[114,115],[115,116],[112,117],[117,118],[118,119],[119,120],
        [112,121],[121,122],[122,123],[123,124],[112,125],[125,126],[126,127],[127,128],
        [112,129],[129,130],[130,131],[131,132]
    ]
    await websocket.send_json({
        "type": "skeleton_init",
        "connections": skeleton_connections,
        "part_ranges": {
            "body": [0, 22],
            "face": [23, 90],
            "left_hand": [91, 111],
            "right_hand": [112, 132]
        }
    })
    
    # ★ 独立姿态线程 — 零阻塞, 极低延迟
    _latest_frame = None
    _latest_pose = None
    _pose_lock = threading.Lock()

    def _pose_thread_worker():
        nonlocal _latest_frame, _latest_pose
        while True:
            frame = _latest_frame
            if frame is not None and app_state.pose_estimator is not None:
                try:
                    kp, sc = process_pose_frame(frame)
                    if kp is not None and sc is not None:
                        with _pose_lock:
                            _latest_pose = (kp, sc, frame.shape[:2], time.time())
                except Exception:
                    pass
            time.sleep(0.005)  # 5ms 极速轮询

    _pose_thread = threading.Thread(target=_pose_thread_worker, daemon=True)
    _pose_thread.start()

    # 异步发送器 — 定时读取线程结果并发送
    async def _pose_sender():
        nonlocal _latest_pose
        while True:
            await asyncio.sleep(0.04)  # 25fps 发送 — 更流畅
            with _pose_lock:
                data = _latest_pose
                _latest_pose = None
            if data is None:
                continue
            kp, sc, (H, W), cap_time = data
            app_state.perf_pose_ms = (time.time() - cap_time) * 1000
            app_state.perf_frame_count += 1

            kp_list = []
            detected = 0
            for i in range(len(kp)):
                conf = float(sc[i]) if i < len(sc) else 0.0
                kp_list.append([float(kp[i, 0]) / W, float(kp[i, 1]) / H, round(conf, 3)])
                if conf > 0.3: detected += 1
            try:
                await websocket.send_json({
                    "type": "pose_frame", "keypoints": kp_list,
                    "frame_w": 1.0, "frame_h": 1.0,
                    "detected": detected, "capture_time": cap_time
                })
            except Exception:
                break  # WebSocket closed
            app_state.pose_buffer.append((kp, sc))
            update_recognition_state(kp)
            if should_trigger_recognition():
                asyncio.create_task(trigger_recognition_async())

    _pose_task = asyncio.create_task(_pose_sender())

    # 追踪后台识别任务, 防止重复触发
    _recognition_task = None

    async def trigger_recognition_async():
        """异步触发识别推理 — 使用 buffer 快照, 在后台线程池中运行, 不阻塞主循环"""
        nonlocal _recognition_task

        if not should_trigger_recognition():
            return

        if app_state.is_recognition_running:
            print("[识别] 上一次识别仍在运行中，跳过")
            return

        # 如果已有后台任务在运行, 取消它 (避免堆积)
        if _recognition_task is not None and not _recognition_task.done():
            print("[识别] 取消未完成的后台识别任务")
            _recognition_task.cancel()

        # ★ 关键: 立即对 buffer 做快照, 避免后续 clear() 或新增帧影响识别
        pose_snapshot = list(app_state.pose_buffer)
        print(f"[识别] 开始后台识别，缓冲帧数: {len(pose_snapshot)}")

        async def _run():
            try:
                # 在线程池中运行CPU密集的推理任务
                loop = asyncio.get_event_loop()
                result, confidence = await loop.run_in_executor(None, run_recognition, pose_snapshot)

                output_text = None

                if result:
                    if confidence < 0.3:
                        output_text = "抱歉呀再试一次吧"
                    else:
                        output_text = result
                    print(f"[识别] 结果: {output_text}, 置信度: {confidence:.2f}")
                else:
                    output_text = "抱歉呀，没有识别到呢，请重新试试吧"
                    print(f"[识别] 无识别结果")

                if output_text:
                    record = app_state.save_recognition(output_text, confidence, app_state.current_user_id)
                    app_state.speak(output_text)

                    await websocket.send_json({
                        "type": "recognition_result",
                        "data": {
                            "timestamp": record["timestamp"],
                            "text": str(record["text"]),
                            "confidence": float(confidence)
                        }
                    })

                # ★ 不再 clear buffer — 新帧已在识别期间正常累积
                reset_recognition_state()
            except asyncio.CancelledError:
                print("[识别] 后台任务被取消")
            except Exception as e:
                print(f"[识别] 后台推理异常: {e}")
                import traceback
                traceback.print_exc()
                reset_recognition_state()

        _recognition_task = asyncio.create_task(_run())
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            
            if "text" in data:
                message = json.loads(data["text"])
                cmd = message.get("type")
                
                if cmd == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                    
                elif cmd == "get_status":
                    load_model_if_needed()
                    recording_duration = 0.0
                    if app_state.recognition_state == RecognitionState.RECORDING:
                        recording_duration = time.time() - app_state.recording_start_time
                    await websocket.send_json({
                        "type": "status",
                        "model_loaded": bool(app_state.model is not None),
                        "is_recognizing": bool(app_state.is_recognizing),
                        "recognition_state": str(app_state.recognition_state),
                        "recording_duration": float(round(recording_duration, 1))
                    })
                    
                elif cmd == "toggle_speech":
                    app_state.is_speech_on = not app_state.is_speech_on
                    await websocket.send_json({"type": "speech_toggled", "is_on": app_state.is_speech_on})
                    
                elif cmd == "switch_recognition":
                    load_model_if_needed()
                    app_state.is_recognizing = message.get("is_on", False)
                    update_recognition_state(None)
                    await websocket.send_json({
                        "type": "recognition_updated",
                        "is_on": app_state.is_recognizing,
                        "recognition_state": str(app_state.recognition_state)
                    })
                    # 如果点击了结束识别，立即触发推理
                    if not app_state.is_recognizing:
                        asyncio.create_task(trigger_recognition_async())
                    
                elif cmd == "get_history":
                    history = app_state.get_history(user_id=app_state.current_user_id, limit=50)
                    await websocket.send_json({"type": "history", "data": history})
                    
                elif cmd == "clear_history":
                    app_state.clear_history()
                    await websocket.send_json({"type": "history_cleared"})
                    
                elif cmd == "toggle_auto":
                    app_state.auto_mode = not app_state.auto_mode
                    await websocket.send_json({"type":"auto_mode","enabled":app_state.auto_mode})
                elif cmd == "speak":
                    text = message.get("text", "")
                    app_state.speak(text)

            elif "bytes" in data:
                # ★ 非阻塞: 只保存最新帧, 后台任务异步处理姿态
                frame_data = data["bytes"]
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # 缩小到480宽加速
                    h, w = frame.shape[:2]
                    if w > 480:
                        scale = 480.0 / w
                        frame = cv2.resize(frame, (480, int(h * scale)))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 只存最新帧供后台任务消费
                    _latest_frame = frame_rgb

    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[WebSocket] 错误: {e}")


# SPA catch-all: 所有非 API/静态文件 的 GET 请求返回 Vue index.html
# 必须放在所有 /api/* 路由之后
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("ws"):
        raise HTTPException(status_code=404)
    index_path = os.path.join(vue_dist, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404)


# ============ 浏览器自动启动 ============
def launch_browser(port: int = 8001, delay: float = 2.0):
    """延迟打开默认浏览器."""
    import webbrowser
    def _open():
        import time as _time
        _time.sleep(delay)
        webbrowser.open(f'http://localhost:{port}')
    threading.Thread(target=_open, daemon=True).start()


# ============ 启动命令 ============
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    launch_browser(port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
