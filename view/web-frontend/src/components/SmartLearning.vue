<template>
  <div class="learn-page">
    <div class="learn-bg">
      <div class="learn-orb learn-orb-1"></div>
      <div class="learn-orb learn-orb-2"></div>
      <div class="learn-orb learn-orb-3"></div>
      <div class="learn-bg-spots">
        <div class="learn-spot learn-spot-1"></div>
        <div class="learn-spot learn-spot-2"></div>
      </div>
      <div class="learn-bg-grid"></div>
      <div class="learn-bg-particles">
        <div class="learn-particle"></div><div class="learn-particle"></div>
        <div class="learn-particle"></div><div class="learn-particle"></div>
        <div class="learn-particle"></div><div class="learn-particle"></div>
        <div class="learn-particle"></div><div class="learn-particle"></div>
      </div>
      <div class="learn-bg-vignette"></div>
    </div>

    <div class="learn-container">
      <div class="learn-hero">
        <div class="learn-hero-badge">AI LEARNING</div>
        <h1 class="learn-title">模型学习</h1>
        <p class="learn-sub">提交纠错反馈，模型自动学习并持续提升识别精度</p>
      </div>

      <!-- 统计卡片 -->
      <div class="learn-stats-row">
        <div class="learn-stat-card">
          <div class="lsc-accent"></div>
          <span class="lsc-val">{{ corrections.length }}</span>
          <span class="lsc-lbl">已学习记录</span>
        </div>
        <div class="learn-stat-card">
          <div class="lsc-accent"></div>
          <span class="lsc-val">{{ historyList.length || '—' }}</span>
          <span class="lsc-lbl">识别历史</span>
        </div>
        <div class="learn-stat-card">
          <div class="lsc-accent"></div>
          <span class="lsc-val">≥80%</span>
          <span class="lsc-lbl">模糊匹配阈值</span>
        </div>
      </div>

      <div class="learn-row-2">
        <!-- 左侧: 识别历史 + 纠错 -->
        <div class="learn-panel">
          <div class="learn-panel-head">
            <div style="flex:1;text-align:center">
              <h2>识别历史</h2>
              <span class="lph-sub">从实时演示中选择识别结果进行纠错</span>
            </div>
            <button class="lph-btn" @click="loadHistory" :disabled="loadingHistory">
              {{ loadingHistory ? '加载中' : '加载历史' }}
            </button>
          </div>

          <div class="learn-history-list" v-if="historyList.length">
            <div v-for="(h, idx) in historyList.slice(0, 15)" :key="idx"
                 class="lhi-item" :class="{active: selectedHistory?.text === h.text}"
                 @click="selectHistory(h)">
              <span class="lhi-idx">{{ idx + 1 }}</span>
              <span class="lhi-text">{{ h.text }}</span>
              <span class="lhi-time">{{ h.timestamp }}</span>
            </div>
          </div>
          <div v-else-if="!loadingHistory" class="learn-placeholder">
            <span>暂无识别历史，请先在实时演示中完成手语识别</span>
          </div>

          <div v-if="selectedHistory" class="learn-correct-card">
            <div class="lcc-header">输入正确手语文本进行纠错学习</div>
            <div class="lcc-predicted">
              <span class="lcc-pred-label">模型预测</span>
              <span class="lcc-pred-text">{{ selectedHistory.text }}</span>
            </div>
            <div class="lcc-input-row">
              <input v-model="correctText" class="lcc-input" placeholder="输入正确的手语文本..." @keyup.enter="submitCorrection" />
              <button class="lcc-submit" @click="submitCorrection" :disabled="!correctText || submitting">
                {{ submitting ? '保存中...' : '提交学习' }}
              </button>
            </div>
            <div v-if="learnMsg" class="lcc-msg" :class="learnOk ? 'ok' : 'err'">
              {{ learnOk ? '✓' : '✗' }} {{ learnMsg }}
            </div>
          </div>
        </div>

        <!-- 右侧: 已学习数据 -->
        <div class="learn-panel">
          <div class="learn-panel-head">
            <div style="flex:1;text-align:center">
              <h2>已学习纠错</h2>
              <span class="lph-sub">模型记忆库，实时识别时自动匹配并替换</span>
            </div>
            <button class="lph-btn secondary" @click="loadCorrections" :disabled="loadingCorr">
              {{ loadingCorr ? '加载中' : '刷新' }}
            </button>
          </div>

          <div class="learn-corr-grid" v-if="corrections.length">
            <div v-for="(c, idx) in corrections" :key="idx" class="lci-card">
              <div class="lci-card-num">{{ idx + 1 }}</div>
              <div class="lci-card-body">
                <div class="lci-row">
                  <span class="lci-tag pred">预测</span>
                  <span class="lci-text pred">{{ c.predicted }}</span>
                </div>
                <div class="lci-arrow-row">↓ 纠错为 ↓</div>
                <div class="lci-row">
                  <span class="lci-tag correct">正确</span>
                  <span class="lci-text correct">{{ c.corrected }}</span>
                </div>
              </div>
              <button class="lci-delete" @click="deleteCorrection(c.predicted)" title="删除此条">✕</button>
            </div>
          </div>
          <div v-else class="learn-placeholder">
            <span>暂无纠错数据，从左侧选择识别历史开始学习</span>
          </div>

          <div class="learn-how-card">
            <h3>工作原理</h3>
            <div class="lhc-steps">
              <div class="lhc-step">
                <span class="lhc-step-num">1</span>
                <span>实时识别得到预测文本</span>
              </div>
              <div class="lhc-step">
                <span class="lhc-step-num">2</span>
                <span>系统检查是否匹配已学习记录</span>
              </div>
              <div class="lhc-step">
                <span class="lhc-step-num">3</span>
                <span>相似度 ≥80% 则自动替换为正确文本</span>
              </div>
              <div class="lhc-step">
                <span class="lhc-step-num">4</span>
                <span>模型随使用越来越准确</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import api from '../api/index.js'

const historyList = ref([])
const corrections = ref([])
const selectedHistory = ref(null)
const correctText = ref('')
const loadingHistory = ref(false)
const loadingCorr = ref(false)
const submitting = ref(false)
const learnMsg = ref('')
const learnOk = ref(false)

async function loadHistory() {
  loadingHistory.value = true
  try { const r = await api.get('/learning/history?limit=100'); historyList.value = r.history || [] }
  catch (_) {}
  loadingHistory.value = false
}

async function loadCorrections() {
  loadingCorr.value = true
  try { const r = await api.get('/learning/corrections'); corrections.value = r.corrections || [] }
  catch (_) {}
  loadingCorr.value = false
}

function selectHistory(h) {
  selectedHistory.value = h; correctText.value = ''; learnMsg.value = ''
}

async function submitCorrection() {
  if (!correctText.value || !selectedHistory.value) return
  submitting.value = true; learnMsg.value = ''
  try {
    await api.post('/learning/correct', { predicted: selectedHistory.value.text, corrected: correctText.value })
    learnMsg.value = '纠错已保存！下次遇到相似手语将自动使用正确文本'
    learnOk.value = true; correctText.value = ''
    await loadCorrections()
  } catch (e) {
    learnMsg.value = e.response?.data?.detail || '保存失败'
    learnOk.value = false
  }
  submitting.value = false
}

async function deleteCorrection(predicted) {
  try { await api.delete('/learning/correction', { params: { predicted } }); await loadCorrections() }
  catch (_) {}
}

onMounted(() => { loadCorrections() })
</script>
