# 智能手语识别系统 - Vue3前端 + FastAPI后端 + MySQL

## 项目结构

```
hand_swap/
├── web-frontend/          # Vue3前端项目
│   ├── src/
│   │   ├── api/           # API接口
│   │   ├── App.vue        # 主组件
│   │   ├── main.js        # 入口文件
│   │   └── style.css      # 全局样式
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── web_app/               # FastAPI后端
│   ├── app.py            # 主应用
│   └── requirements.txt
└── demo/                  # 模型和推理代码
```

## 技术栈

### 前端
- **框架**: Vue 3 + Vite
- **HTTP**: Axios
- **通信**: WebSocket

### 后端
- **框架**: FastAPI
- **数据库**: MySQL
- **模型推理**: PyTorch + Transformers
- **姿态估计**: rtmlib

## 环境变量

后端MySQL连接配置（可选，默认使用SQLite模拟模式）:
```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_DATABASE=sign_recognition
```

## 启动方式

### 1. 启动后端
```bash
cd web_app
pip install -r requirements.txt
python app.py
```

### 2. 启动前端
```bash
cd web-frontend
npm install
npm run dev
```

### 3. 访问
打开浏览器访问 `http://localhost:5173`

## MySQL数据库初始化

如果使用MySQL，需要先创建数据库:
```sql
CREATE DATABASE sign_recognition CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

程序会自动创建 `recognition_history` 表。

## API接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/status | 获取系统状态 |
| GET | /api/cameras | 获取摄像头列表 |
| GET | /api/history | 获取识别历史 |
| POST | /api/history/clear | 清空历史记录 |
| POST | /api/speech/toggle | 切换语音开关 |
| WS | /ws | WebSocket实时通信 |

## 功能特性

- 实时摄像头姿态检测
- 手语识别与翻译
- 语音播报（edge-tts）
- 识别历史记录
- 全屏模式
- 响应式设计
