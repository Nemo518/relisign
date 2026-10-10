# 智能手语识别系统 - Web版

基于 FastAPI + WebSocket 的实时手语识别系统，支持语音播报、全屏弹幕显示。

## 功能特性

### 核心功能
- 实时摄像头视频流处理
- 基于 ReliSign 模型的手语识别
- WebSocket 实时通信
- TTS 语音播报
- 识别结果弹幕展示

### 界面功能
- 现代简约风格 UI
- 摄像头控制（开启/关闭/切换）
- 识别控制（启动/停止/间隔调节）
- 语音播报开关
- 语音包切换
- 全屏模式（摄像头占满屏幕）
- 识别结果实时显示
- 历史记录（带时间戳）

### 快捷键
- `F` - 切换全屏
- `C` - 切换摄像头
- `R` - 切换识别
- `ESC` - 退出全屏

## 安装依赖

```bash
cd web_app
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

然后在浏览器访问: http://localhost:8000

## 项目结构

```
web_app/
├── app.py                 # FastAPI 后端服务
├── requirements.txt       # 依赖列表
├── templates/
│   └── index.html         # 前端页面
├── static/                # 静态资源目录
└── sign_recognition.db   # SQLite 数据库（自动生成）
```

## 技术栈

- **后端**: FastAPI + uvicorn
- **实时通信**: WebSocket
- **数据库**: SQLite
- **前端**: HTML5 + CSS3 + JavaScript (原生)
- **语音合成**: pyttsx3 / Windows SAPI
- **姿态估计**: rtmlib
- **手语识别**: ReliSign

## API 接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | / | 主页面 |
| GET | /api/status | 系统状态 |
| GET | /api/cameras | 可用摄像头列表 |
| GET | /api/voices | 可用语音包列表 |
| GET | /api/history | 识别历史记录 |
| POST | /api/history/clear | 清空历史记录 |
| POST | /api/camera/switch | 切换摄像头 |
| POST | /api/recognition/start | 启动识别 |
| POST | /api/recognition/stop | 停止识别 |
| POST | /api/speech/toggle | 切换语音开关 |
| POST | /api/speech/voice | 设置语音包 |
| WS | /ws | WebSocket 端点 |

## 配置说明

在 `app.py` 中可以修改以下配置：

- `recognition_interval`: 识别间隔（默认 3 秒）
- `max_buffer_size`: 姿态数据缓冲区大小（默认 150 帧）
- `camera_id`: 默认摄像头 ID（默认 0）
