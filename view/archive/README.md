# Uni-Sign 答辩展示视图

`view/` 是独立于训练与评估主链路的演示模块目录。

## 目标

1. 展示代码和主项目逻辑分开
2. 最小改动复用现有 `out/`、`data/`、`demo/online_inference.py`
3. 支持国奖答辩风格的可视化展示
4. 接入真实 attention / Grad-CAM / 混淆分析接口

## 目录结构

```text
view/
├── app.py
├── data_service.py
├── inference_adapter.py
├── xai_service.py
├── requirements.txt
└── README.md
```

## 启动

```bash
C:\Users\Nemo518\.conda\envs\torch13\python.exe -m pip install -r view/requirements.txt
C:\Users\Nemo518\.conda\envs\torch13\python.exe -m uvicorn view.api_server:app --host 127.0.0.1 --port 8000
C:\Users\Nemo518\.conda\envs\torch13\python.exe -m streamlit run view/app.py
```

也可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File view/start_stack.ps1
```

## 当前能力

- 实时演示
  - 样例视频
  - 本地上传
  - 前端经由 FastAPI 调用真实 `demo/online_inference.py`
  - ECharts 指标看板增强展示层次

- 真实可解释 AI
  - MT5 编码器真实 attention
  - GCN 末层 Grad-CAM 式时序热力图
  - 字符级混淆分析矩阵
  - 关键帧截图画廊

- 模型对比
  - 训练曲线
  - 多模型指标对比
  - 错误切片与典型案例

- 架构展示
  - 独立 view 层
  - 推理 / 分析 / 数据层边界清晰

- 答辩增强
  - jQuery 驱动讲解词轮播
  - 关键页签自动切换
  - 评委演示 PDF 导出

## 说明

- `view/app.py` 内嵌了 jQuery 与 ECharts 的前端片段，用于答辩模式轮播与高观感图表。
- 如果现场网络受限，Plotly 图表仍可用；ECharts / jQuery 依赖 CDN，建议赛前用一次联网环境缓存。

## 后续建议

1. 给 XAI 结果增加导出报告与答辩截图打包
2. 增加 WebSocket 实时推理流，替代轮询式请求
3. 决赛阶段再升级 React + WebSocket 前端
