<template>
  <!-- embedded mode: inline page -->
  <div v-if="embedded" class="xai-embedded">
    <div class="page-container">
      <div class="page-hero">
        <h1 class="page-title">模型归因分析</h1>
        <p class="page-subtitle">Grad-CAM 关键部位定位 · Encoder Attention 依赖捕获 · 字符级预测对齐</p>
        <button class="ref-btn" @click="loadAnalysis" style="margin-top:12px">🔄 刷新分析</button>
      </div>
      <div class="xai-embedded-content">
        <div class="xai-tabs">
          <button v-for="tab in tabs" :key="tab.key" class="xai-tab" :class="{ active: activeTab===tab.key }" @click="activeTab=tab.key">{{ tab.label }}</button>
        </div>
        <div class="xai-content" v-if="!isLoading && xaiData">
          <!-- same tab content as drawer mode -->
          <section v-if="activeTab==='overview'" class="xai-section">
            <div class="xai-section-title">身体部位 Grad-CAM 重要性</div>
            <div class="bar-chart-wrap"><svg :viewBox="`0 0 320 ${partBars.length*36+20}`" class="bar-chart-svg"><g v-for="(item,idx) in partBars" :key="item.key"><text x="70" :y="idx*36+22" text-anchor="end" class="bar-label">{{ item.name }}</text><rect x="76" :y="idx*36+10" :width="Math.max(item.peak*220,2)" height="14" rx="7" :fill="item.color" opacity="0.85"/><text x="76" :y="idx*36+37" class="bar-val">峰值 {{ (item.peak*100).toFixed(0) }}%</text><text x="302" :y="idx*36+22" text-anchor="end" class="bar-val-sm">均值 {{ (item.mean*100).toFixed(0) }}%</text></g></svg></div>
            <div class="xai-section-title" style="margin-top:16px">Top 关键帧</div>
            <div class="top-frames-grid"><div v-for="f in xaiData.top_frames" :key="f.frame" class="top-frame-chip"><span class="frame-idx">#{{ f.frame }}</span><span class="frame-score">{{ (f.score*100).toFixed(0) }}%</span></div></div>
            <div class="xai-meta">帧数 {{ xaiData.frame_count }} · 损失 {{ xaiData.loss }}</div>
          </section>
          <section v-if="activeTab==='timeline'" class="xai-section">
            <div class="xai-section-title">帧级重要性时间线</div>
            <div class="xai-legend"><span v-for="l in lineLegend" :key="l.key" class="legend-item" @click="toggleLine(l.key)"><span class="legend-swatch" :class="{off:hiddenLines[l.key]}" :style="{background:l.color}"></span><span class="legend-label" :class="{off:hiddenLines[l.key]}">{{ l.label }}</span></span></div>
            <div class="line-chart-wrap"><svg :viewBox="`0 0 ${lineChartW} ${lineChartH+30}`" class="line-chart-svg"><line v-for="n in 5" :key="n" :x1="40" :y1="(n/5)*lineChartH+4" :x2="lineChartW" :y2="(n/5)*lineChartH+4" stroke="#374151" stroke-width="0.5"/><g v-for="l in lineLegend" :key="l.key"><polyline v-if="!hiddenLines[l.key]" :points="linePoints(l.key)" :stroke="l.color" stroke-width="1.8" fill="none" stroke-linejoin="round" :stroke-dasharray="l.dashed?'6,3':'none'"/></g><line x1="40" y1="4" x2="40" :y2="lineChartH+4" stroke="#6b7280" stroke-width="1"/><line x1="40" :y1="lineChartH+4" :x2="lineChartW" :y2="lineChartH+4" stroke="#6b7280" stroke-width="1"/><text v-for="n in xLabels" :key="n.frame" :x="n.x" :y="lineChartH+22" text-anchor="middle" class="axis-label">{{ n.frame }}</text></svg></div>
          </section>
          <section v-if="activeTab==='attention'" class="xai-section">
            <div class="xai-section-title">Encoder 帧间注意力依赖矩阵</div>
            <div class="heatmap-wrap"><canvas ref="heatmapEl" class="heatmap-canvas"></canvas></div>
            <div class="xai-meta">尺寸 {{ xaiData.attention_matrix_size }}×{{ xaiData.attention_matrix_size }} · 越亮越相关</div>
          </section>
          <section v-if="activeTab==='alignment'" class="xai-section">
            <div class="xai-section-title">字符级预测对齐</div>
            <div class="ref-input-row"><input v-model="referenceText" type="text" class="ref-input" placeholder="输入正确文本进行对比..." @keyup.enter="runAlignment"/><button class="ref-btn" @click="runAlignment" :disabled="alignLoading">{{ alignLoading?'分析中...':'对比' }}</button></div>
            <div class="prediction-display"><span class="pred-label">预测结果：</span><span class="pred-val">{{ xaiData.prediction||'无' }}</span></div>
            <div v-if="alignmentData&&alignmentData.length>0" class="align-grid">
              <div class="align-legend-row"><span class="align-legend"><span class="align-chip match">匹配</span></span><span class="align-legend"><span class="align-chip replace">替换</span></span><span class="align-legend"><span class="align-chip delete">删除</span></span><span class="align-legend"><span class="align-chip insert">插入</span></span></div>
              <div class="align-tokens"><span v-for="(item,idx) in alignmentData" :key="idx" class="align-token" :class="item.operation"><span class="token-ref">{{ item.reference_char==='∅'?'—':item.reference_char }}</span><span class="token-arrow">→</span><span class="token-pred">{{ item.prediction_char==='∅'?'—':item.prediction_char }}</span></span></div>
            </div>
            <div v-else-if="alignmentData&&alignmentData.length===0" class="align-empty">参考文本与预测文本完全相同</div>
            <div v-else class="align-empty">输入参考文本后点击"对比"进行字符级对齐分析</div>
          </section>
        </div>
        <div class="xai-loading" v-if="isLoading"><div class="spinner"></div><div class="msg">正在计算归因分析...</div></div>
        <div class="xai-empty" v-if="!isLoading&&!xaiData"><div class="msg">暂无分析数据</div><div class="hint">请先在「实时演示」中完成一次手语识别</div></div>
      </div>
    </div>
  </div>

  <!-- drawer mode: slide-out panel -->
  <template v-else>
  <div class="xai-overlay" :class="{ open: visible }" @click="$emit('close')"></div>
  <div class="xai-drawer" :class="{ open: visible }">
    <div class="drawer-header">
      <span class="title">模型归因分析</span>
      <button class="close-btn" @click="loadAnalysis">🔄</button>
      <button class="close-btn" @click="$emit('close')">✕</button>
    </div>
    <div class="xai-tabs">
      <button v-for="tab in tabs" :key="tab.key" class="xai-tab" :class="{ active: activeTab===tab.key }" @click="activeTab=tab.key">{{ tab.label }}</button>
    </div>

    <!-- content -->
    <div class="xai-content" v-if="!isLoading && xaiData">
      <!-- ====== 概览标签页 ====== -->
      <section v-if="activeTab === 'overview'" class="xai-section">
        <div class="xai-section-title">身体部位 Grad-CAM 重要性</div>
        <div class="bar-chart-wrap">
          <svg :viewBox="`0 0 320 ${partBars.length * 36 + 20}`" class="bar-chart-svg">
            <g v-for="(item, idx) in partBars" :key="item.key">
              <text x="70" :y="idx * 36 + 22" text-anchor="end" class="bar-label">{{ item.name }}</text>
              <rect x="76" :y="idx * 36 + 10" :width="Math.max(item.peak * 220, 2)" height="14" rx="7" :fill="item.color" opacity="0.85" />
              <text x="76" :y="idx * 36 + 37" class="bar-val">峰值 {{ (item.peak * 100).toFixed(0) }}%</text>
              <text x="302" :y="idx * 36 + 22" text-anchor="end" class="bar-val-sm">均值 {{ (item.mean * 100).toFixed(0) }}%</text>
            </g>
          </svg>
        </div>
        <div class="xai-section-title" style="margin-top: 16px;">Top 关键帧</div>
        <div class="top-frames-grid">
          <div v-for="f in xaiData.top_frames" :key="f.frame" class="top-frame-chip">
            <span class="frame-idx">#{{ f.frame }}</span>
            <span class="frame-score">{{ (f.score * 100).toFixed(0) }}%</span>
          </div>
        </div>
        <div class="xai-meta">帧数 {{ xaiData.frame_count }} · 损失 {{ xaiData.loss }}</div>
      </section>

      <!-- ====== 时间线标签页 ====== -->
      <section v-if="activeTab === 'timeline'" class="xai-section">
        <div class="xai-section-title">帧级重要性时间线</div>
        <div class="xai-legend">
          <span v-for="l in lineLegend" :key="l.key" class="legend-item" @click="toggleLine(l.key)">
            <span class="legend-swatch" :class="{ off: hiddenLines[l.key] }" :style="{ background: l.color }"></span>
            <span class="legend-label" :class="{ off: hiddenLines[l.key] }">{{ l.label }}</span>
          </span>
        </div>
        <div class="line-chart-wrap">
          <svg :viewBox="`0 0 ${lineChartW} ${lineChartH + 30}`" class="line-chart-svg">
            <!-- grid lines -->
            <line v-for="n in 5" :key="n" :x1="40" :y1="(n / 5) * lineChartH + 4"
                  :x2="lineChartW" :y2="(n / 5) * lineChartH + 4" stroke="#374151" stroke-width="0.5" />
            <!-- lines -->
            <g v-for="l in lineLegend" :key="l.key">
              <polyline v-if="!hiddenLines[l.key]"
                :points="linePoints(l.key)"
                :stroke="l.color" stroke-width="1.8" fill="none" stroke-linejoin="round"
                :stroke-dasharray="l.dashed ? '6,3' : 'none'" />
            </g>
            <!-- axes -->
            <line x1="40" y1="4" x2="40" :y2="lineChartH + 4" stroke="#6b7280" stroke-width="1" />
            <line x1="40" :y1="lineChartH + 4" :x2="lineChartW" :y2="lineChartH + 4" stroke="#6b7280" stroke-width="1" />
            <!-- x labels -->
            <text v-for="n in xLabels" :key="n.frame" :x="n.x" :y="lineChartH + 22" text-anchor="middle" class="axis-label">{{ n.frame }}</text>
          </svg>
        </div>
      </section>

      <!-- ====== 注意力标签页 ====== -->
      <section v-if="activeTab === 'attention'" class="xai-section">
        <div class="xai-section-title">Encoder 帧间注意力依赖矩阵</div>
        <div class="heatmap-wrap">
          <canvas ref="heatmapEl" class="heatmap-canvas"></canvas>
        </div>
        <div class="xai-meta">尺寸 {{ xaiData.attention_matrix_size }}×{{ xaiData.attention_matrix_size }} · 越亮越相关</div>
      </section>

      <!-- ====== 对齐标签页 ====== -->
      <section v-if="activeTab === 'alignment'" class="xai-section">
        <div class="xai-section-title">字符级预测对齐</div>
        <div class="ref-input-row">
          <input
            v-model="referenceText"
            type="text"
            class="ref-input"
            placeholder="输入正确文本进行对比..."
            @keyup.enter="runAlignment"
          />
          <button class="ref-btn" @click="runAlignment" :disabled="alignLoading">
            {{ alignLoading ? '分析中...' : '对比' }}
          </button>
        </div>
        <div class="prediction-display">
          <span class="pred-label">预测结果：</span>
          <span class="pred-val">{{ xaiData.prediction || '无' }}</span>
        </div>
        <div v-if="alignmentData && alignmentData.length > 0" class="align-grid">
          <div class="align-legend-row">
            <span class="align-legend"><span class="align-chip match">匹配</span></span>
            <span class="align-legend"><span class="align-chip replace">替换</span></span>
            <span class="align-legend"><span class="align-chip delete">删除</span></span>
            <span class="align-legend"><span class="align-chip insert">插入</span></span>
          </div>
          <div class="align-tokens">
            <span
              v-for="(item, idx) in alignmentData"
              :key="idx"
              class="align-token"
              :class="item.operation"
            >
              <span class="token-ref">{{ item.reference_char === '∅' ? '—' : item.reference_char }}</span>
              <span class="token-arrow">→</span>
              <span class="token-pred">{{ item.prediction_char === '∅' ? '—' : item.prediction_char }}</span>
            </span>
          </div>
        </div>
        <div v-else-if="alignmentData && alignmentData.length === 0" class="align-empty">
          参考文本与预测文本完全相同
        </div>
        <div v-else class="align-empty">
          输入参考文本后点击"对比"进行字符级对齐分析
        </div>
      </section>
    </div>

    <!-- loading -->
    <div class="xai-loading" v-if="isLoading">
      <div class="spinner"></div>
      <div class="msg">正在计算归因分析...</div>
    </div>

    <!-- error / no data -->
    <div class="xai-empty" v-if="!isLoading && !xaiData">
      <div class="msg">暂无分析数据</div>
      <div class="hint">请先完成一次手语识别后再打开此面板</div>
    </div>
  </div>
  </template>
</template>

<script setup>
import { ref, watch, computed, nextTick } from 'vue'
import { xaiApi } from '../api/index.js'

const props = defineProps({
  visible: Boolean,
  prediction: String,
  embedded: { type: Boolean, default: false }
})

defineEmits(['close'])

/* tabs */
const tabs = [
  { key: 'overview', label: '概览' },
  { key: 'timeline', label: '时间线' },
  { key: 'attention', label: '注意力' },
  { key: 'alignment', label: '对齐' },
]
const activeTab = ref('overview')

/* data */
const xaiData = ref(null)
const isLoading = ref(false)
const referenceText = ref('')
const alignmentData = ref(null)
const alignLoading = ref(false)

/* heatmap */
const heatmapEl = ref(null)

/* line chart */
const hiddenLines = ref({})
const lineChartW = 600
const lineChartH = 180

const partColors = { body: '#3b82f6', left: '#f97316', right: '#22c55e', face_all: '#a855f7' }
const partNames = { body: '身体躯干', left: '左手', right: '右手', face_all: '面部' }

const lineLegend = [
  { key: 'gradcam_combined', label: 'Grad-CAM 综合', color: '#f1f5f9', dashed: false },
  { key: 'encoder_attention', label: '注意力权重', color: '#22d3ee', dashed: true },
  { key: 'body', label: '身体', color: '#3b82f6', dashed: false },
  { key: 'left', label: '左手', color: '#f97316', dashed: false },
  { key: 'right', label: '右手', color: '#22c55e', dashed: false },
  { key: 'face_all', label: '面部', color: '#a855f7', dashed: false },
]

function toggleLine(key) {
  hiddenLines.value[key] = !hiddenLines.value[key]
}

/* bar chart */
const partBars = computed(() => {
  if (!xaiData.value?.part_importance) return []
  return Object.entries(xaiData.value.part_importance).map(([key, val]) => ({
    key,
    name: val.name || partNames[key] || key,
    peak: val.peak || 0,
    mean: val.mean || 0,
    color: partColors[key] || '#888'
  }))
})

/* line chart helpers */
const xLabels = computed(() => {
  if (!xaiData.value) return []
  const n = xaiData.value.frame_count
  const labels = []
  const step = Math.max(1, Math.floor(n / 8))
  for (let i = 0; i < n; i += step) {
    labels.push({ frame: i, x: 40 + (i / Math.max(n - 1, 1)) * (lineChartW - 40) })
  }
  return labels
})

function linePoints(key) {
  if (!xaiData.value?.frame_timeline) return ''
  const tl = xaiData.value.frame_timeline
  const n = tl.length
  return tl.map((pt, i) => {
    const x = 40 + (i / Math.max(n - 1, 1)) * (lineChartW - 40)
    const y = lineChartH + 4 - (pt[key] || 0) * lineChartH
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

/* load */
async function loadAnalysis() {
  isLoading.value = true
  xaiData.value = null
  alignmentData.value = null
  try {
    const data = await xaiApi.getLatest()
    xaiData.value = data
    if (data.alignment) alignmentData.value = data.alignment
    referenceText.value = ''
  } catch (e) {
    console.error('XAI load error:', e)
  } finally {
    isLoading.value = false
  }
}

async function runAlignment() {
  if (!referenceText.value.trim()) return
  alignLoading.value = true
  try {
    const data = await xaiApi.analyze(referenceText.value.trim())
    xaiData.value = data
    alignmentData.value = data.alignment || []
  } catch (e) {
    console.error('XAI alignment error:', e)
  } finally {
    alignLoading.value = false
  }
}

/* draw heatmap */
function drawHeatmap() {
  if (!xaiData.value?.attention_matrix) return
  const canvas = heatmapEl.value
  if (!canvas) return
  const matrix = xaiData.value.attention_matrix
  const size = matrix.length
  if (size === 0) return

  const cellSize = Math.max(4, Math.floor(Math.min(400, canvas.parentElement?.clientWidth || 400) / size))
  canvas.width = size * cellSize
  canvas.height = size * cellSize
  canvas.style.width = size * cellSize + 'px'
  canvas.style.height = size * cellSize + 'px'

  const ctx = canvas.getContext('2d')
  // Find global min/max for normalization
  let gMin = Infinity, gMax = -Infinity
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = matrix[r][c]
      if (v < gMin) gMin = v
      if (v > gMax) gMax = v
    }
  }
  const range = gMax - gMin || 1

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const t = (matrix[r][c] - gMin) / range
      ctx.fillStyle = viridisColor(t)
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize)
    }
  }
}

function viridisColor(t) {
  // Simplified viridis colormap
  const r = Math.round(255 * (0.267 * Math.sin(Math.PI * (t * 2.0 - 0.5)) / 2 + 0.27 + t * 0.5))
  const g = Math.round(255 * (0.267 * Math.sin(Math.PI * (t * 2.0 - 0.0)) / 2 + 0.38 + t * 0.45))
  const b = Math.round(255 * (0.267 * Math.sin(Math.PI * (t * 2.0 + 0.5)) / 2 + 0.55 - t * 0.35))
  return `rgb(${Math.max(0, Math.min(255, r))},${Math.max(0, Math.min(255, g))},${Math.max(0, Math.min(255, b))})`
}

/* watch for panel open */
watch(() => props.visible, async (val) => {
  if (val) {
    activeTab.value = 'overview'
    await loadAnalysis()
    await nextTick()
    if (activeTab.value === 'attention') drawHeatmap()
  }
})

watch(activeTab, async (tab) => {
  if (tab === 'attention') {
    await nextTick()
    drawHeatmap()
  }
})

watch(xaiData, async () => {
  await nextTick()
  if (activeTab.value === 'attention') drawHeatmap()
})
</script>
