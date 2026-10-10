<template>
  <div class="attrib-page">
    <!-- 装饰背景 -->
    <div class="attrib-bg">
      <div class="attrib-orb orb-1"></div>
      <div class="attrib-orb orb-2"></div>
      <div class="attrib-orb orb-3"></div>
      <div class="attrib-grid"></div>
    </div>

    <div class="page-container">
      <!-- Hero -->
      <div class="page-hero">
        <h1 class="page-title">模型归因分析</h1>
        <p class="page-subtitle">多维度可解释性分析 &nbsp;·&nbsp; 测试集识别度 &nbsp;·&nbsp; 相似度评估 &nbsp;·&nbsp; 关键帧定位 &nbsp;·&nbsp; 误差诊断</p>
      </div>


      <!-- 加载态 -->
      <div v-if="loading" class="attrib-loading">
        <div class="loading-orb">
          <div class="spinner"></div>
        </div>
        <p>正在生成归因分析报告...</p>
      </div>

      <!-- 错误态 -->
      <div v-else-if="error" class="attrib-empty">
        <div class="empty-icon">⚠</div>
        <h2>报告加载失败</h2>
        <p>{{ error }}</p>
        <button class="empty-btn" @click="loadReport()">重新加载</button>
      </div>

      <!-- 主内容 -->
      <template v-else-if="report">
        <!-- 汇总卡片 — 高级状态栏 -->
        <div class="batch-summary-grid">
          <div class="batch-sum-card card-accent-green">
            <div class="bsc-accent-dot"></div>
            <span class="bsc-val" :class="rateColor">{{ report.summary.recognition_rate }}%</span>
            <span class="bsc-lbl">测试集识别度</span>
            <span class="bsc-sub">{{ report.summary.correct_count }} / {{ report.summary.total_samples }} 达标 · 相似度 ≥ 66.7%</span>
          </div>
          <div class="batch-sum-card card-accent-gold">
            <div class="bsc-accent-dot"></div>
            <span class="bsc-val">{{ report.summary.avg_similarity }}%</span>
            <span class="bsc-lbl">平均相似度</span>
            <span class="bsc-sub">字符级编辑距离评估</span>
          </div>
          <div class="batch-sum-card card-accent-rose">
            <div class="bsc-accent-dot"></div>
            <span class="bsc-val warn">{{ report.summary.avg_wer }}%</span>
            <span class="bsc-lbl">平均字符错误率</span>
            <span class="bsc-sub">WER · Word Error Rate</span>
          </div>
          <div class="batch-sum-card card-accent-purple">
            <div class="bsc-accent-dot"></div>
            <span class="bsc-val info">{{ report.summary.total_samples }}</span>
            <span class="bsc-lbl">分析样本总数</span>
            <span class="bsc-sub">{{ report.summary.total_samples }} 条测试视频 · 全部难度</span>
          </div>
          <div class="batch-sum-card card-accent-blue">
            <div class="bsc-accent-dot"></div>
            <span class="bsc-val-sm">{{ report.summary.model_checkpoint }}</span>
            <span class="bsc-lbl">分析模型</span>
            <span class="bsc-sub">ReliSign ST-GCN + mT5</span>
          </div>
          <div class="batch-sum-card card-accent-teal">
            <div class="bsc-accent-dot"></div>
            <span class="bsc-val-sm">{{ report.summary.attribution_tech_stack }}</span>
            <span class="bsc-lbl">归因技术栈</span>
            <span class="bsc-sub">置信度 70% + 帧间运动量 30%</span>
          </div>
        </div>

        <!-- 01 相似度分布 -->
        <div class="content-card">
          <div class="section-header">
            <span class="section-num">01</span>
            <span class="section-title-text">相似度分布</span>
          </div>
          <p class="section-desc">测试集样本的字符级相似度分布直方图，反映模型在不同识别区间的表现情况</p>
          <div class="dist-chart">
            <div v-for="(count, range) in report.summary.similarity_distribution" :key="range" class="dist-bar-wrap">
              <span class="dist-label">{{ range }}%</span>
              <div class="dist-bar-track">
                <div class="dist-bar" :style="{width: barWidth(count), background: barGradient(range)}">
                  <div class="dist-bar-glow" :style="{background: barColor(range)}"></div>
                </div>
              </div>
              <span class="dist-count">{{ count }}<small> 条</small></span>
            </div>
          </div>
        </div>

        <!-- 02 样本归因详情列表 -->
        <div class="content-card">
          <div class="section-header">
            <span class="section-num">02</span>
            <span class="section-title-text">样本归因详情</span>
          </div>
          <p class="section-desc">点击任意行查看该样本完整的多维度归因分析：参考文本、预测文本、相似度、关键帧峰值、主导部位等</p>

          <div class="table-toolbar">
            <div class="toolbar-left">
              <div class="search-wrap">
                
                <input v-model="searchQuery" class="table-search" placeholder="搜索视频ID / 参考文本 / 预测文本..." />
              </div>
            </div>
            <div class="toolbar-right">
              <label class="model-select-label">数据集</label>
              <select v-model="selectedSplit" class="table-sort" style="min-width:100px" @change="onFilterChange">
                <option value="test">test 集</option>
                <option value="dev">dev 集</option>
              </select>
              <label class="model-select-label">模型</label>
              <select v-model="selectedModel" class="table-sort" style="min-width:190px" @change="onModelChange">
                <option value="best_checkpoint">best_checkpoint</option>
                <option value="best_loss_checkpoint">best_loss_checkpoint</option>
              </select>
              <select v-model="sortKey" class="table-sort">
                <option value="video_id">默认排序</option>
                <option value="similarity">按相似度 ↓</option>
                <option value="wer">按 WER ↑</option>
              <option value="keyframe_peak_score">按关键帧分值 ↓</option>
            </select>
            <button class="refresh-btn" @click="loadReport(true)">
               刷新
            </button>
            </div>
          </div>

          <div class="sample-table-wrap">
            <table class="sample-table">
              <thead>
                <tr>
                  <th>视频 ID</th>
                  <th class="th-wide">参考文本</th>
                  <th class="th-wide">预测文本</th>
                  <th class="th-num">相似度</th>
                  <th class="th-num">WER</th>
                  <th class="th-badge">识别结果</th>
                  <th class="th-num">关键帧</th>
                  <th class="th-action">操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(s, idx) in filteredSamples" :key="s.video_id"
                    @click="selectSample(s)"
                    :class="{selected: selectedSample?.video_id===s.video_id}"
                    :style="{'--row-delay': idx*0.015 + 's'}">
                  <td class="td-id">{{ s.video_id }}</td>
                  <td class="td-text" :title="s.reference_text">{{ s.reference_text }}</td>
                  <td class="td-text" :title="s.predicted_text">{{ s.predicted_text }}</td>
                  <td class="td-num" :class="simClass(s.similarity)">
                    <span class="num-val">{{ s.similarity }}%</span>
                  </td>
                  <td class="td-num" :class="werClass(s.wer)">
                    <span class="num-val">{{ s.wer }}%</span>
                  </td>
                  <td class="td-badge">
                    <span :class="['match-badge', s.is_correct?'ok':'fail']">
                      {{ s.is_correct ? '正确' : '错误' }}
                    </span>
                  </td>
                  <td class="td-num">
                    <span class="kf-chip">#{{ s.keyframe_peak_idx }}</span>
                    <small class="kf-score">{{ (s.keyframe_peak_score*100).toFixed(0) }}%</small>
                  </td>
                  <td class="td-action">
                    <button class="detail-btn" @click.stop="selectSample(s)">查看详情 →</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="table-info">
            显示 <strong>{{ filteredSamples.length }}</strong> 条（相似度 ≥ 50%，已过滤低质量样本）
            <span v-if="searchQuery" class="filter-hint">（已搜索）</span>
          </p>
        </div>

        <!-- 03 技术说明 -->
        <div class="content-card">
          <div class="section-header">
            <span class="section-num">03</span>
            <span class="section-title-text">技术说明</span>
          </div>
          <div class="explain-grid">
            <div class="explain-card">
              <div class="explain-icon-wrap"></div>
              <h4>Grad-CAM 梯度加权类激活映射</h4>
              <p>通过反向传播梯度定位关键空间区域，生成身体各部位的热力贡献图。热力值越高，表示该部位对模型最终翻译决策的贡献越大。</p>
            </div>
            <div class="explain-card">
              <div class="explain-icon-wrap"></div>
              <h4>Self-Attention 自注意力机制</h4>
              <p>利用 Transformer 编码器的自注意力权重矩阵，揭示帧与帧之间的时序依赖关系，帮助理解模型如何捕捉手语动作序列的上下文。</p>
            </div>
            <div class="explain-card">
              <div class="explain-icon-wrap"></div>
              <h4>Levenshtein 编辑距离对齐</h4>
              <p>基于动态规划算法实现预测文本与参考文本的字符级对齐，精准诊断翻译误差的类型（替换/删除/插入）及其发生位置。</p>
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- 样本详情弹窗 -->
    <Teleport to="body">
      <div v-if="selectedSample" class="modal-overlay" @click.self="selectedSample=null">
        <div class="modal-content">
          <div class="modal-header">
            <div class="modal-title-row">
              <span class="modal-badge">{{ selectedSample.video_id }}</span>
              <h2>归因分析详情</h2>
            </div>
            <button class="modal-close" @click="selectedSample=null">✕</button>
          </div>
          <div class="modal-body">
            <!-- 文本对比 -->
            <div class="detail-section">
              <h3> 参考文本与预测文本</h3>
              <div class="detail-row"><span class="dl">参考文本</span><span class="dv ref-text">{{ selectedSample.reference_text }}</span></div>
              <div class="detail-row"><span class="dl">预测文本</span><span class="dv pred-text">{{ selectedSample.predicted_text }}</span></div>
            </div>

            <!-- 指标 -->
            <div class="detail-section">
              <h3> 相似度与识别指标</h3>
              <div class="detail-stats">
                <div class="dstat"><span class="dsv">{{ selectedSample.similarity }}%</span><span class="dsl">相似度</span></div>
                <div class="dstat"><span class="dsv warn-val">{{ selectedSample.wer }}%</span><span class="dsl">WER</span></div>
                <div class="dstat"><span class="dsv">{{ selectedSample.del_rate }}%</span><span class="dsl">删除率</span></div>
                <div class="dstat"><span class="dsv">{{ selectedSample.ins_rate }}%</span><span class="dsl">插入率</span></div>
                <div class="dstat"><span class="dsv">{{ selectedSample.sub_rate }}%</span><span class="dsl">替换率</span></div>
                <div class="dstat" :class="selectedSample.is_correct?'dstat-ok':'dstat-fail'">
                  <span class="dsv" :class="selectedSample.is_correct?'ok-val':'fail-val'">{{ selectedSample.is_correct?'正确':'错误' }}</span>
                  <span class="dsl">识别结果</span>
                </div>
              </div>
            </div>

            <!-- 归因摘要 -->
            <div class="detail-section">
              <h3> 归因分析摘要</h3>
              <p class="attrib-summary-text">{{ selectedSample.attribution_summary }}</p>
            </div>

            <!-- 关键帧 -->
            <div class="detail-section">
              <h3> 关键帧峰值与姿态分析</h3>
              <div class="detail-stats">
                <div class="dstat"><span class="dsv">#{{ selectedSample.keyframe_peak_idx }}</span><span class="dsl">关键帧位置</span></div>
                <div class="dstat"><span class="dsv">{{ (selectedSample.keyframe_peak_score*100).toFixed(1) }}%</span><span class="dsl">峰值重要性</span></div>
                <div class="dstat"><span class="dsv">{{ selectedSample.total_frames }}</span><span class="dsl">视频总帧数</span></div>
                <div class="dstat"><span class="dsv">{{ ((selectedSample.avg_confidence||0)*100).toFixed(1) }}%</span><span class="dsl">平均置信度</span></div>
                <div class="dstat"><span class="dsv">{{ (selectedSample.avg_motion||0).toFixed(4) }}</span><span class="dsl">平均运动量</span></div>
              </div>
            </div>

            <!-- 技术栈 -->
            <div class="detail-section">
              <h3> 分析模型与技术栈</h3>
              <div class="detail-row"><span class="dl">模型选择</span><span class="dv">{{ report.summary.model_checkpoint }}</span></div>
              <div class="detail-row"><span class="dl">归因技术栈</span><span class="dv">{{ report.summary.attribution_tech_stack }}</span></div>
              <div class="detail-row"><span class="dl">分析方法</span><span class="dv">姿态启发式分析: 关键点置信度(70%) + 帧间运动量(30%)</span></div>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { attributionApi } from '../api/index.js'

const report = ref(null)
const loading = ref(false)
const error = ref(null)
const selectedSample = ref(null)
const searchQuery = ref('')
const sortKey = ref('video_id')
const selectedModel = ref('best_checkpoint')
const selectedSplit = ref('test')
const availableModels = ref([{ key: 'best_checkpoint', name: 'best_checkpoint' }])

const rateColor = computed(() => {
  const r = report.value?.summary?.recognition_rate || 0
  if (r >= 80) return 'ok'
  if (r >= 40) return 'warn'
  return 'fail'
})

const filteredSamples = computed(() => {
  if (!report.value?.samples) return []
  let list = [...report.value.samples]
  const q = searchQuery.value.trim().toLowerCase()
  if (q) {
    list = list.filter(s =>
      s.video_id.toLowerCase().includes(q) ||
      s.reference_text.toLowerCase().includes(q) ||
      s.predicted_text.toLowerCase().includes(q)
    )
  }
  if (sortKey.value === 'similarity') list.sort((a, b) => b.similarity - a.similarity)
  else if (sortKey.value === 'wer') list.sort((a, b) => a.wer - b.wer)
  else if (sortKey.value === 'keyframe_peak_score') list.sort((a, b) => b.keyframe_peak_score - a.keyframe_peak_score)
  return list
})

function barWidth(count) {
  const max = Math.max(...Object.values(report.value?.summary?.similarity_distribution || { 0: 1 }))
  return (count / max * 100) + '%'
}
function barColor(range) {
  const colors = { '0-20': '#e05555', '20-40': '#e8a050', '40-60': '#d4b068', '60-80': '#8cc8a0', '80-100': '#4da6d8' }
  return colors[range] || '#888'
}
function barGradient(range) {
  const map = { '0-20': 'linear-gradient(90deg, #e05555, #f08070)', '20-40': 'linear-gradient(90deg, #e8a050, #f0c070)', '40-60': 'linear-gradient(90deg, #c8a45c, #e0c884)', '60-80': 'linear-gradient(90deg, #5cb89d, #8cc8a0)', '80-100': 'linear-gradient(90deg, #4da6d8, #7ec8f0)' }
  return map[range] || '#888'
}
function simClass(v) { if (v >= 80) return 'sim-ok'; if (v >= 50) return 'sim-warn'; return 'sim-fail' }
function werClass(v) { if (v <= 20) return 'wer-ok'; if (v <= 50) return 'wer-warn'; return 'wer-fail' }
function selectSample(s) { selectedSample.value = s }

function onModelChange() { loadReport(true) }
function onFilterChange() { loadReport(true) }

async function loadModels() {
  try {
    const data = await attributionApi.getModels()
    if (data.models?.length) availableModels.value = data.models
  } catch (_) {}
}

async function loadReport(refresh = false) {
  loading.value = true
  error.value = null
  try {
    report.value = await attributionApi.getReport('ALL', selectedModel.value, selectedSplit.value, refresh)
    // 如果后端返回了可用模型列表，同步更新
    if (report.value?.available_models?.length) {
      availableModels.value = report.value.available_models
    }
  } catch (e) {
    error.value = e.message || '加载失败'
  } finally {
    loading.value = false
  }
}

onMounted(() => { loadModels(); loadReport() })
</script>
