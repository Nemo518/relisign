<template>
  <div class="arch-page">
    <!-- 背景层 — 8层叠加 -->
    <div class="arch-bg">
      <!-- 主光球 -->
      <div class="arch-bg-orb arch-orb-a"></div>
      <div class="arch-bg-orb arch-orb-b"></div>
      <div class="arch-bg-orb arch-orb-c"></div>
      <!-- 辅助光斑 -->
      <div class="arch-bg-spots">
        <div class="arch-spot arch-spot-1"></div>
        <div class="arch-spot arch-spot-2"></div>
        <div class="arch-spot arch-spot-3"></div>
      </div>
      <!-- 光束 -->
      <div class="arch-bg-beams">
        <div class="arch-beam"></div>
        <div class="arch-beam"></div>
        <div class="arch-beam"></div>
        <div class="arch-beam"></div>
      </div>
      <!-- 六边形科技网格 -->
      <div class="arch-bg-hex"></div>
      <!-- 粒子 -->
      <div class="arch-bg-particles">
        <div class="arch-particle"></div><div class="arch-particle"></div>
        <div class="arch-particle"></div><div class="arch-particle"></div>
        <div class="arch-particle"></div><div class="arch-particle"></div>
        <div class="arch-particle"></div><div class="arch-particle"></div>
        <div class="arch-particle"></div><div class="arch-particle"></div>
      </div>
      <!-- 扫描线 -->
      <div class="arch-bg-scanlines"></div>
      <!-- 边缘晕影 -->
      <div class="arch-bg-vignette"></div>
      <!-- 底部渐变 -->
      <div class="arch-bg-bottom"></div>
    </div>

    <div class="arch-container">
      <!-- Hero -->
      <div class="arch-hero">
        <div class="arch-hero-badge">SYSTEM ARCHITECTURE</div>
        <h1 class="arch-hero-title">技术架构</h1>
        <p class="arch-hero-sub">ST-GCN 时空图卷积 &nbsp;+&nbsp; mT5 多语言 Transformer &nbsp;—&nbsp; 133 关键点全身建模 &nbsp;·&nbsp; 端到端延迟 &lt; 500ms</p>
        <div class="arch-hero-stats">
          <div class="hero-stat"><span class="hs-val">133</span><span class="hs-lbl">全身关键点</span></div>
          <div class="hero-stat"><span class="hs-val">&lt;500</span><span class="hs-lbl">毫秒延迟</span></div>
          <div class="hero-stat"><span class="hs-val">30</span><span class="hs-lbl">FPS 帧率</span></div>
          <div class="hero-stat"><span class="hs-val">4</span><span class="hs-lbl">并行通道</span></div>
        </div>
      </div>

      <!-- 01 核心模块 — 4卡片 2x2 -->
      <div class="arch-section">
        <div class="arch-section-label">CORE MODULES</div>
        <div class="arch-section-header">
          <span class="as-num">01</span>
          <h2 class="as-title">核心模块</h2>
          <p class="as-desc">四大核心引擎协同工作，覆盖从姿态感知到文本生成的完整推理链路</p>
        </div>
        <div class="core-grid">
          <div class="core-card" v-for="(card, idx) in coreCards" :key="card.title" :style="{'--ci': idx}">
            <div class="core-card-accent"></div>
            <div class="core-card-body">
              <h3>{{ card.title }}</h3>
              <p>{{ card.desc }}</p>
              <div class="core-tags"><span v-for="t in card.tags" :key="t">{{ t }}</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- 02 推理流水线 -->
      <div class="arch-section">
        <div class="arch-section-label">INFERENCE PIPELINE</div>
        <div class="arch-section-header">
          <span class="as-num">02</span>
          <h2 class="as-title">推理流水线</h2>
          <p class="as-desc">端到端全链路流程，从摄像头采集到中文文本输出，每一步均经过深度优化</p>
        </div>
        <div class="pipeline-full">
          <div class="pf-step" v-for="(step, i) in pipelineSteps" :key="i">
            <div class="pf-num-ring">{{ i + 1 }}</div>
            <div class="pf-label" v-html="step"></div>
            <div class="pf-arrow" v-if="i < pipelineSteps.length - 1">→</div>
          </div>
        </div>
      </div>

      <!-- 03+04 并排: 性能指标 + 模型架构 -->
      <div class="arch-row-2">
        <!-- 03 性能指标 -->
        <div class="arch-section arch-section-half">
          <div class="arch-section-label">PERFORMANCE</div>
          <div class="arch-section-header">
            <span class="as-num">03</span>
            <h2 class="as-title">性能指标</h2>
          </div>
          <div class="metrics-big">
            <div class="mb-card" v-for="m in metrics" :key="m.label">
              <div class="mb-value">{{ m.value }}</div>
              <div class="mb-unit">{{ m.unit }}</div>
              <div class="mb-desc">{{ m.desc }}</div>
            </div>
          </div>
        </div>

        <!-- 04 模型架构 -->
        <div class="arch-section arch-section-half">
          <div class="arch-section-label">MODEL DESIGN</div>
          <div class="arch-section-header">
            <span class="as-num">04</span>
            <h2 class="as-title">模型架构</h2>
          </div>
          <div class="model-diagram">
            <div class="md-block md-encoder">
              <div class="md-block-label">ENCODER</div>
              <div class="md-block-title">ST-GCN 姿态编码器</div>
              <div class="md-steps">
                <span>133 关键点</span><span>四部位拆分</span><span>空间图卷积</span><span>时序建模</span><span>768d 特征</span>
              </div>
            </div>
            <div class="md-connector">
              <div class="md-conn-line"></div>
              <div class="md-conn-label">CROSS-ATTENTION</div>
            </div>
            <div class="md-block md-decoder">
              <div class="md-block-label">DECODER</div>
              <div class="md-block-title">mT5 文本解码器</div>
              <div class="md-steps">
                <span>前缀提示</span><span>12层编码器</span><span>12层解码器</span><span>Beam Search</span><span>中文输出</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 05 技术栈全景 -->
      <div class="arch-section">
        <div class="arch-section-label">TECH STACK</div>
        <div class="arch-section-header">
          <span class="as-num">05</span>
          <h2 class="as-title">技术栈全景</h2>
          <p class="as-desc">深度学习 · 姿态估计 · 前后端工程 · 可解释性 AI — 四大技术领域全栈覆盖</p>
        </div>
        <div class="tech-grid">
          <div class="tech-group" v-for="g in techStack" :key="g.title">
            <div class="tech-group-head">
              
              <h4>{{ g.title }}</h4>
            </div>
            <ul>
              <li v-for="item in g.items" :key="item">{{ item }}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const coreCards = [
  { icon:'', title:'姿态编码器', desc:'采用 ST-GCN 时空图卷积网络，对 133 个全身关键点进行空间-时间双维度建模。支持身体躯干、左手、右手、面部四个独立部位并行编码，提取高维姿态特征向量。', tags:['ST-GCN','时空注意力','133关键点','四部位并行'] },
  { icon:'', title:'文本解码器', desc:'以多语言 mT5-Base 作为序列解码器，通过交叉注意力机制将姿态特征映射为自然语言文本。支持 Beam Search 解码策略，兼顾翻译质量与推理速度，可无缝切换中英文输出。', tags:['mT5-Base','Beam Search','交叉注意力','中英双语'] },
  { icon:'', title:'多模态融合', desc:'基于可变形注意力的姿态-RGB 特征对齐模块，配合不确定性门控自适应融合机制，在 RGB 模态可用时有效提升手部细节翻译精度，显著降低手指级别误识率。', tags:['Deformable Attn','Uncertainty Gate','PGF 模块','自适应融合'] },
  { icon:'', title:'实时推理引擎', desc:'前端 WebSocket 双向通信 + ONNX Runtime 加速姿态提取 + PyTorch 推理流水线，端到端延迟 &lt;500ms。支持 CPU/GPU 自适应切换，兼容 Windows/macOS/Linux 多平台部署。', tags:['ONNX 加速','PyTorch','WebSocket','跨平台'] },
]

const pipelineSteps = ['摄像头<br>实时采集','133 点<br>全身姿态','ST-GCN<br>时空编码','mT5<br>文本解码','后处理<br>输出']
const stepIcons = ['','','','','']

const metrics = [
  { value:'133', unit:'关键点', desc:'COCO-WholeBody 全身姿态估计，覆盖身体/双手/面部' },
  { value:'<500', unit:'毫秒', desc:'端到端全链路推理延迟，实时流畅交互' },
  { value:'30', unit:'FPS', desc:'RTMLib + ONNX 加速，姿态提取帧率稳定' },
  { value:'4', unit:'通道', desc:'身体 / 左手 / 右手 / 面部四通道并行编码' },
]

const techStack = [
  { icon:'', title:'深度学习框架', items:['PyTorch 2.x — 模型训练与推理','ST-GCN — 时空图卷积网络','mT5-Base — 多语言 Transformer','Deformable Attention 2D — 多模态融合','EfficientNet-B0 — RGB 特征提取','Label Smoothing — 正则化策略'] },
  { icon:'', title:'姿态估计', items:['RTMLib Wholebody — 133 点全身姿态','ONNX Runtime — 推理加速引擎','OpenCV — 视频帧采集与预处理','COCO-WholeBody — 关键点标注标准','关键点归一化与阈值滤波'] },
  { icon:'', title:'前后端技术', items:['FastAPI + Uvicorn — 异步 Web 服务','WebSocket — 双向实时通信','Vue 3 + Vite — 响应式前端框架','SQLite — 轻量级数据持久化','edge-tts — 语音合成播报','PyInstaller — 桌面应用打包'] },
  { icon:'', title:'可解释性 AI', items:['Grad-CAM — 关键部位空间定位','Self-Attention — 帧间依赖分析','编辑距离对齐 — 字符级误差诊断','Self-Distillation — 归因信号生成','混淆矩阵 — 错误模式分析'] },
]
</script>
