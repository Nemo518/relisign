<template>
  <!-- 加载画面 -->
  <div class="loading-screen" :class="{ done: !loading }">
    <div class="spinner"></div>
    <div class="msg">ReliSign 正在初始化...</div>
  </div>

  <!-- Toast -->
  <div class="toast-area">
    <div v-for="t in toasts" :key="t.id" class="toast" :class="{ leaving: t.leaving }">{{ t.message }}</div>
  </div>

  <!-- ==================== 登录/注册页 (全屏) ==================== -->
  <div class="auth-page" v-if="!loading && !user">
    <div class="auth-page-bg"></div>
    <div class="auth-page-card">
      <div class="auth-page-logo">🤟</div>
      <h1 class="auth-page-title">ReliSign</h1>
      <p class="auth-page-sub">智能手语识别系统</p>
      <div class="auth-page-tabs">
        <button class="auth-page-tab" :class="{ active: authMode==='login' }" @click="authMode='login';authError=''">登录</button>
        <button class="auth-page-tab" :class="{ active: authMode==='register' }" @click="authMode='register';authError=''">注册</button>
      </div>
      <form class="auth-page-form" @submit.prevent="handleAuth">
        <input v-model="authForm.username" type="text" placeholder="用户名" class="auth-page-input" autocomplete="username" />
        <input v-model="authForm.password" type="password" placeholder="密码" class="auth-page-input" autocomplete="current-password" />
        <p class="auth-page-error" v-if="authError">{{ authError }}</p>
        <button type="submit" class="auth-page-submit" :disabled="authLoading">
          {{ authLoading ? '请稍候...' : (authMode==='login' ? '登 录' : '注 册') }}
        </button>
      </form>
    </div>
  </div>

  <!-- ==================== 主应用壳 ==================== -->
  <div class="app-shell" v-if="!loading && user">

    <!-- 顶部导航栏 -->
    <header class="top-nav">
      <div class="nav-brand" @click="activeTab='demo'">
        <div class="nav-logo">R</div>
        <div class="nav-name">
          <span class="nav-name-main">ReliSign</span>
          <span class="nav-name-sub">智能手语识别系统</span>
        </div>
      </div>
      <nav class="nav-tabs">
        <button v-for="tab in navTabs" :key="tab.key"
          class="nav-tab" :class="{ active: activeTab === tab.key }"
          @click="activeTab = tab.key">
          <span class="nav-tab-label">{{ tab.label }}</span>
        </button>
      </nav>
      <div class="nav-actions">
        <div class="nav-status">
          <span class="status-dot" :class="wsConnected ? 'on' : 'off'"></span>
          <span class="status-text">{{ wsConnected ? '在线' : '离线' }}</span>
        </div>
        <div v-if="user" class="nav-user-menu" @click.stop>
          <button class="nav-user-btn" @click="userMenuOpen=!userMenuOpen">
            <span class="nub-avatar">{{ user.username[0]?.toUpperCase() }}</span>
            <span class="nub-name">{{ user.username }}</span>
            <span class="nub-arrow" :class="{open:userMenuOpen}">▾</span>
          </button>
          <div v-if="userMenuOpen" class="user-dropdown" @click.self="userMenuOpen=false">
            <button class="ud-item" @click="activeTab='user';userMenuOpen=false">
              用户中心
            </button>
            <button class="ud-item" @click="handleLogout();userMenuOpen=false">
              退出登录
            </button>
          </div>
        </div>
      </div>
    </header>

    <!-- ==================== 模块: 实时演示 ==================== -->
    <main class="module-demo" v-show="activeTab==='demo'">
      <div class="stage">
        <!-- 倒计时浮层 -->
        <div class="countdown-overlay" :class="{ active: countdownActive }" v-if="countdownActive">
          <div class="countdown-num" :key="countdownValue">{{ countdownValue > 0 ? countdownValue : '🎯' }}</div>
          <div class="countdown-label">{{ countdownValue > 0 ? '' : '开始!' }}</div>
        </div>
        <!-- 录制中浮层 -->
        <div class="recording-overlay" :class="{ active: isRecognizing }" v-if="isRecognizing">
          <div class="recording-pulse"></div>
          <div class="recording-time">{{ recordingDuration.toFixed(1) }}s</div>
          <div class="recording-hint">正在录制手语...</div>
        </div>
        <div class="record-border" :class="{ active: isRecognizing }"></div>
        <div class="camera-layer">
          <video ref="videoEl" autoplay playsinline muted :class="{ hidden: !cameraOn }"></video>
          <canvas ref="skeletonEl" class="skeleton-canvas" :class="{ active: skeletonActive }"></canvas>
          <div v-if="!cameraOn" class="camera-placeholder">
            <div class="ring"><span class="icon">🤟</span></div>
            <div class="label">摄像头未开启</div>
            <div class="kbd">点击 📷 或按 C 键启动</div>
          </div>
          <div v-if="cameraOn && detectedCount > 0" class="kpt-badge">
            <span class="kpt-dot" :class="poseQuality"></span>
            <span>关节 {{ detectedCount }}/133</span>
          </div>
          <!-- 性能面板 -->
          <div v-if="cameraOn" class="perf-panel">
            <div class="perf-item"><span class="perf-val">{{ perfFPS }}</span><span class="perf-lbl">FPS</span></div>
            <div class="perf-item"><span class="perf-val">{{ perfPoseMs }}</span><span class="perf-lbl">ms</span></div>
            <div class="perf-item" v-if="autoMode"><span class="perf-val" :class="{active:isRecognizing}">{{ isRecognizing?'REC':'待机' }}</span><span class="perf-lbl">自动</span></div>
          </div>
          <!-- 自动模式状态提示 -->
          <div v-if="autoMode && !isRecognizing" class="auto-hint">🤖 自动识别已开启，开始打手语将自动录制</div>
        </div>
        <div class="center-stage" :class="{ visible: !!stageResult }" v-if="stageResult">
          <div class="stage-result">{{ stageResult }}</div>
          <div class="stage-confidence" v-if="stageConfidence > 0">
            <div class="track"><div class="fill" :style="{ width: (stageConfidence*100)+'%' }"></div></div>
            <span class="pct">{{ (stageConfidence*100).toFixed(0) }}%</span>
          </div>
          <div class="stage-actions">
            <button class="stage-action-btn" @click="copyResult" title="复制">📋</button>
            <button class="stage-action-btn" @click="speakResult" title="朗读">🔊</button>
            <button class="stage-action-btn" @click="dismissResult" title="删除">✕</button>
          </div>
        </div>
        <div class="bottom-dock">
          <button class="dock-btn" :class="{ active: cameraOn }" aria-label="切换摄像头 (C)" @click="toggleCamera">📷</button>
          <span class="dock-divider"></span>
          <div class="record-btn-wrap">
            <svg v-if="isRecognizing" class="record-progress-ring" viewBox="0 0 56 56">
              <circle cx="28" cy="28" r="25" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="3"/>
              <circle cx="28" cy="28" r="25" fill="none" :stroke="progressColor" stroke-width="3" stroke-linecap="round"
                :stroke-dasharray="circumference" :stroke-dashoffset="progressOffset" transform="rotate(-90 28 28)"/>
            </svg>
            <button class="dock-btn record-btn" :class="{ recording: isRecognizing }" :disabled="!cameraOn"
              aria-label="开始/结束识别 (R)" @click="toggleRecognition">{{ isRecognizing?'⏹':'🎯' }}</button>
          </div>
          <span class="dock-divider"></span>
          <button class="dock-btn" :class="{ active: autoMode }" aria-label="自动识别模式 (A)" @click="toggleAutoMode" :disabled="!cameraOn">🤖</button>
          <span class="dock-divider"></span>
          <button class="dock-btn" :class="{ active: sttListening || sttProcessing }" aria-label="语音输入 (V)"
            @click="toggleSTT" :disabled="sttProcessing"
            :title="sttSupported ? '语音转文字 — 点击开始/停止 (V)' : '浏览器不支持录音，请使用 Chrome/Edge'">
            {{ sttProcessing ? '⏳' : (sttListening ? '🎙️◉' : '🎙️') }}
          </button>
          <span class="dock-divider"></span>
          <button class="dock-btn" :class="{ active: speechOn }" aria-label="语音播报" @click="toggleSpeech">{{ speechOn?'🔊':'🔇' }}</button>
          <button class="dock-btn" :class="{ active: fullscreen }" aria-label="全屏模式 (F)" @click="toggleFullscreen">⛶</button>
          <button class="dock-btn settings-btn" :class="{ active: settingsOpen }" aria-label="设置" @click="settingsOpen=!settingsOpen">
            ⚙ <span class="settings-btn-text">设置</span>
          </button>
          <button class="dock-btn" :class="{ active: historyOpen }" aria-label="识别历史 (H)" @click="historyOpen=!historyOpen">📋</button>
        </div>
        <div class="history-overlay" :class="{ open: historyOpen }" @click="historyOpen=false"></div>
        <div class="history-drawer" :class="{ open: historyOpen }">
          <div class="drawer-header"><span class="title">识别历史</span><button class="close-btn" @click="historyOpen=false">✕</button></div>
          <div class="drawer-list" v-if="historyList.length>0">
            <div v-for="(item,i) in historyList" :key="i" class="drawer-item" @click="recallHistory(item)">
              <div class="time">{{ item.timestamp }}</div><div class="text">{{ item.text }}</div>
            </div>
          </div>
          <div v-else class="drawer-empty">暂无识别记录</div>
          <div v-if="confidenceHistory.length>1" class="confidence-history" style="padding:0 20px 14px">
            <div class="ch-title">置信度趋势</div>
            <div class="ch-bar-row">
              <div v-for="(h,i) in confidenceHistory" :key="i" class="ch-bar" :style="{height:(h.conf*100*0.8)+'px'}" :title="h.text+': '+(h.conf*100).toFixed(0)+'%'"></div>
            </div>
          </div>
          <div class="drawer-footer" v-if="historyList.length>0">
            <button class="dock-btn" style="width:100%;border-radius:24px;font-size:15px;gap:8px;height:44px" @click="clearHistory">🗑 清空记录</button>
          </div>
        </div>
        <div class="settings-overlay" :class="{ open: settingsOpen }" @click="settingsOpen=false"></div>
        <div class="settings-drawer" :class="{ open: settingsOpen }">
          <div class="drawer-header"><span class="title">设置</span><button class="close-btn" @click="settingsOpen=false">✕</button></div>
          <div class="settings-body">
            <div class="setting-group"><label class="setting-label">摄像头源</label>
              <select v-model="selectedCameraId" class="setting-select"><option v-for="cam in cameraDevices" :key="cam.deviceId" :value="cam.deviceId">{{ cam.label }}</option></select></div>
            <div class="setting-group"><label class="setting-label">语音播报音色</label>
              <select v-model="selectedVoiceId" class="setting-select" @change="onVoiceChange"><option v-for="v in ttsVoices" :key="v.id" :value="v.id">{{ v.name }}</option></select></div>
            <div class="setting-group"><label class="setting-label">音频输出设备</label>
              <select v-model="selectedAudioDeviceId" class="setting-select" @change="onAudioDeviceChange"><option v-for="dev in audioDevices" :key="dev.id" :value="dev.id">{{ dev.name }}</option></select></div>
          </div>
        </div>
      </div>
    </main>

    <!-- ==================== 模块: 模型归因 ==================== -->
    <main class="module-page" v-show="activeTab==='attribution'">
      <AttributionPage />
    </main>

    <!-- ==================== 模块: 模型展示 ==================== -->
    <main class="module-page" v-show="activeTab==='showcase'">
      <ModelShowcase />
    </main>

    <!-- ==================== 模块: 智能学习 ==================== -->
    <main class="module-page" v-show="activeTab==='learning'">
      <SmartLearning />
    </main>

    <!-- ==================== 模块: 技术架构 ==================== -->
    <main class="module-page" v-show="activeTab==='architecture'">
      <ArchitecturePage />
    </main>

    <!-- ==================== 模块: 应用场景 ==================== -->
    <main class="module-page" v-show="activeTab==='scenarios'">
      <ScenariosPage />
    </main>

    <!-- ==================== 模块: 用户中心 ==================== -->
    <main class="module-page" v-show="activeTab==='user'">
      <UserPage :user="user" :onLogout="handleLogout" />
    </main>

    <!-- 功能导览浮层 -->
    <div class="onboarding-overlay" :class="{ visible: showOnboarding }" v-if="showOnboarding">
      <div class="onboarding-card">
        <div class="onboarding-icon">{{ onboardingSteps[onboardingStep].icon }}</div>
        <div class="onboarding-title">{{ onboardingSteps[onboardingStep].title }}</div>
        <div class="onboarding-desc">{{ onboardingSteps[onboardingStep].desc }}</div>
        <div class="onboarding-dots">
          <span v-for="(s,i) in onboardingSteps" :key="i" class="onboarding-dot" :class="{ active: i===onboardingStep }"></span>
        </div>
        <div class="onboarding-btns">
          <button class="onboarding-skip" @click="dismissOnboarding">跳过</button>
          <button class="onboarding-next" @click="nextOnboarding">{{ onboardingStep<onboardingSteps.length-1?'下一步':'开始使用' }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { createWebSocket, historyApi, speechApi, cameraApi, statusApi, audioApi, xaiApi, vramApi } from './api/index.js'
import XaiPanel from './components/XaiPanel.vue'
import AttributionPage from './components/AttributionPage.vue'
import ArchitecturePage from './components/ArchitecturePage.vue'
import ScenariosPage from './components/ScenariosPage.vue'
import ModelShowcase from './components/ModelShowcase.vue'
import SmartLearning from './components/SmartLearning.vue'
import UserPage from './components/UserPage.vue'
import { renderSkeleton, clearCanvas } from './utils/SkeletonRenderer.js'

/* ======== 导航 ======== */
const activeTab = ref('demo')
const userMenuOpen = ref(false)
const navTabs = [
  { key: 'demo', label: '实时演示' },
  { key: 'attribution', label: '模型归因' },
  { key: 'showcase', label: '模型展示' },
  { key: 'learning', label: '模型学习' },
  { key: 'architecture', label: '技术架构' },
  { key: 'scenarios', label: '应用场景' },
  { key: 'user', label: '用户中心' },
]

/* ======== 认证 ======== */
const authMode = ref('login')
const authForm = ref({ username: '', password: '' })
const authError = ref('')
const authLoading = ref(false)
const user = ref(null)

async function handleAuth() {
  authError.value = ''
  if (!authForm.value.username || !authForm.value.password) { authError.value = '请填写完整信息'; return }
  authLoading.value = true
  try {
    const endpoint = authMode.value === 'login' ? '/api/auth/login' : '/api/auth/register'
    const r = await fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(authForm.value)
    })
    const data = await r.json()
    if (!r.ok) { authError.value = data.detail || '操作失败'; return }
    if (authMode.value === 'login') {
      user.value = data.user
      localStorage.setItem('unising_user', JSON.stringify(data.user))
      localStorage.setItem('unising_token', data.token)
      // 绑定后续识别记录到当前用户
      fetch('/api/auth/set-user', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: data.user.username }) }).catch(() => {})
      await onLoginSuccess()
    } else {
      authError.value = ''
      toast('注册成功，请登录')
      authMode.value = 'login'
      authForm.value.password = ''
    }
  } catch (e) { authError.value = '网络错误' }
  finally { authLoading.value = false }
}

function handleLogout() {
  user.value = null
  localStorage.removeItem('unising_user')
  localStorage.removeItem('unising_token')
  stopCamera()
  if (wsRef.value) wsRef.value.close()
  wsConnected.value = false
  toast('已退出登录')
}

/* ======== 状态 ======== */
const loading = ref(true)
const cameraOn = ref(false)
const isRecognizing = ref(false)
const speechOn = ref(true)
const fullscreen = ref(false)
const historyOpen = ref(false)
const settingsOpen = ref(false)
const wsConnected = ref(false)
const modelReady = ref(false)
const recordingDuration = ref(0)
const recordingFrameCount = ref(0)
const maxRecordDuration = 6  // 最长录制6秒
const countdownActive = ref(false)
const countdownValue = ref(0)
let countdownTimer = null
let recordingDurationTimer = null

// 自动模式 + 性能监控
const autoMode = ref(false)

// STT 语音转文字
const sttListening = ref(false)
const sttProcessing = ref(false)
const sttSupported = ref(false)
let sttMediaRecorder = null
let sttStream = null  // 保存麦克风流引用以便停止时释放

const perfFPS = ref(0)
const perfPoseMs = ref(0)
const perfBuffer = ref(0)
const autoActivity = ref('idle') // 'idle'|'active'|'recording'
const confidenceHistory = ref([])
const showOnboarding = ref(false)
const onboardingStep = ref(0)
const circumference = 2 * Math.PI * 25
const progressOffset = ref(circumference)
const progressColor = ref('#e0574d')
const stageResult = ref('')
const stageConfidence = ref(0)
const historyList = ref([])
const videoEl = ref(null)
const skeletonEl = ref(null)
const skeletonCtx = ref(null)
const skeletonActive = ref(false)
const lastPoseData = ref(null)
const detectedCount = ref(0)
const poseQuality = ref('idle')
const skeletonConnections = ref([])
const skeletonPartRanges = ref({})
const mediaStream = ref(null)
const frameTimer = ref(null)
const wsRef = ref(null)
const xaiPanelOpen = ref(false)
const xaiAvailable = ref(false)
const cameraDevices = ref([])
const selectedCameraId = ref('')
const ttsVoices = ref([])
const selectedVoiceId = ref('zh-CN-XiaoxiaoNeural')
const audioDevices = ref([])
const selectedAudioDeviceId = ref('default')

/* ======== Toast ======== */
const toasts = ref([])
let tid = 0
function toast(msg) {
  const id = ++tid; toasts.value.push({ id, message: msg, leaving: false })
  setTimeout(() => { const t=toasts.value.find(x=>x.id===id); if(t)t.leaving=true; setTimeout(()=>{toasts.value=toasts.value.filter(x=>x.id!==id)},300) }, 2500)
}

/* ======== 初始化 ======== */
async function onLoginSuccess() {
  loading.value = true
  initWS()
  await enumerateCameras()
  await fetchVoices()
  await fetchAudioDevices()
  await fetchStatus()
  if (!localStorage.getItem('unising_onboarded')) setTimeout(() => { showOnboarding.value = true }, 2500)
  setTimeout(() => { loading.value = false }, 1000)
}

onMounted(async () => {
  // 检测浏览器是否支持录音 (MediaRecorder API)
  sttSupported.value = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder)
  if (sttSupported.value) console.log('[STT] 浏览器支持录音 (MediaRecorder API)')

  const saved = localStorage.getItem('unising_user')
  if (saved) {
    try {
      user.value = JSON.parse(saved)
      await onLoginSuccess()
    } catch (_) { user.value = null }
  }
  loading.value = false
  document.addEventListener('keydown', onKey)
  document.addEventListener('click', onDocClick)
})
onUnmounted(() => { stopFrameSending(); stopCamera(); if(wsRef.value)wsRef.value.close(); if(stageResultTimer)clearTimeout(stageResultTimer); if(sttStream){try{sttStream.getTracks().forEach(t=>t.stop())}catch(_){}}; if(sttMediaRecorder){try{sttMediaRecorder.disconnect()}catch(_){}}; if(sttAudioCtx){try{sttAudioCtx.close()}catch(_){}}; document.removeEventListener('keydown',onKey); document.removeEventListener('click',onDocClick) })

function onDocClick() { userMenuOpen.value = false }

/* ======== WebSocket ======== */
function initWS() {
  wsRef.value = createWebSocket((data) => {
    wsConnected.value = true
    switch(data.type) {
      case 'status':
        modelReady.value = !!data.model_loaded
        if (data.recording_duration !== undefined) { recordingDuration.value = data.recording_duration; updateProgress() }
        break
      case 'recognition_result':
        onRecognitionResult(data.data)
        updateProgress()
        break
      case 'buffer_update':
        if (data.recording_duration !== undefined) { recordingDuration.value = data.recording_duration; updateProgress() }
        if (data.perf) { perfFPS.value = data.perf.fps; perfPoseMs.value = data.perf.pose_ms; perfBuffer.value = data.perf.buffer }
        break
      case 'auto_mode': autoMode.value = data.enabled; toast(data.enabled ? '自动识别已开启' : '自动识别已关闭'); break
      case 'auto_detect':
        if (data.action === 'started') { startRecording(); autoActivity.value = 'recording' }
        else if (data.action === 'stopped') { autoActivity.value = 'active'; stopRecording() }
        break
      case 'speech_toggled': speechOn.value = data.is_on; break
      case 'recognition_updated': isRecognizing.value = data.is_on; break
      case 'history': historyList.value = data.data || []; break
      case 'history_cleared': historyList.value = []; toast('历史已清空'); break
      case 'skeleton_init': skeletonConnections.value = data.connections || []; skeletonPartRanges.value = data.part_ranges || {}; break
      case 'pose_frame':
        lastPoseData.value = data
        detectedCount.value = data.detected || 0
        poseQuality.value = data.detected > 100 ? 'good' : data.detected > 50 ? 'ok' : 'poor'
        drawPoseOnCanvas(data)
        break
      case 'pong': break
    }
  }, () => { wsConnected.value = false; setTimeout(initWS, 3000) })
}
function send(msg) { if(wsRef.value?.readyState===WebSocket.OPEN)wsRef.value.send(JSON.stringify(msg)) }

/* ======== 弹幕自动消失计时器 ======== */
let stageResultTimer = null

function resetStageResultTimer() {
  if (stageResultTimer) clearTimeout(stageResultTimer)
  stageResultTimer = setTimeout(() => {
    stageResult.value = ''
    stageConfidence.value = 0
  }, 30000)  // 30秒后自动消失
}

/* ======== 识别结果 ======== */
function onResult(data) {
  stageResult.value=data.text||''; stageConfidence.value=data.confidence||0; xaiAvailable.value=true; recordingDuration.value=0
  resetStageResultTimer()  // 新结果出现, 启动/重置 30s 倒计时
  historyList.value.unshift({timestamp:data.timestamp||new Date().toLocaleTimeString(),text:data.text||'',confidence:data.confidence||0})
  if(historyList.value.length>50)historyList.value.pop()
  // 置信度历史
  confidenceHistory.value.push({time:new Date().toLocaleTimeString(),text:data.text||'',conf:data.confidence||0})
  if(confidenceHistory.value.length>20)confidenceHistory.value.shift()
}
function recallHistory(item) { stageResult.value=item.text; stageConfidence.value=item.confidence||0; resetStageResultTimer(); historyOpen.value=false }

/* ======== 导览 ======== */
const onboardingSteps = [
  {title:'开启摄像头',desc:'点击 📷 或按 C 键启动摄像头，将上半身和手部对准镜头',icon:'📷'},
  {title:'手语识别',desc:'点击 🎯 或按 R 键开始录制，骨架实时显示检测到的动作',icon:'🎯'},
  {title:'AI 分析',desc:'识别完成后切换到「模型归因」查看 AI 分析详情',icon:'🔬'},
]
function dismissOnboarding() { showOnboarding.value=false; localStorage.setItem('unising_onboarded','1') }
function nextOnboarding() { if(onboardingStep.value<onboardingSteps.length-1)onboardingStep.value++; else dismissOnboarding() }

/* ======== 结果操作 ======== */
async function copyResult() { try{await navigator.clipboard.writeText(stageResult.value);toast('已复制')}catch(_){toast('复制失败')} }
function speakResult() { if(stageResult.value){send({type:'speak',text:stageResult.value});toast('朗读中...')} }
function dismissResult() {
  if (stageResultTimer) clearTimeout(stageResultTimer)
  stageResult.value = ''
  stageConfidence.value = 0
}

/* ======== 进度环 ======== */
function updateProgress() {
  const ratio=Math.min(recordingDuration.value/maxRecordDuration,1)
  progressOffset.value=circumference*(1-ratio)
  progressColor.value=ratio>0.8?'#e0574d':ratio>0.5?'#e0a45c':'#5cb89d'
}
// 后端结果回来时清理状态
function onRecognitionResult(data) {
  stopRecording()
  onResult(data)
}

/* ======== 摄像头 ======== */
async function toggleCamera() { cameraOn.value?stopCamera():await startCamera() }
async function startCamera() {
  try{
    const vc={width:{min:1280,ideal:1920},height:{min:720,ideal:1080},frameRate:{ideal:30}}
    if(selectedCameraId.value)vc.deviceId={exact:selectedCameraId.value}; else vc.facingMode='user'
    mediaStream.value=await navigator.mediaDevices.getUserMedia({video:vc,audio:false})
    if(videoEl.value)videoEl.value.srcObject=mediaStream.value
    cameraOn.value=true; startFrameSending(); startSkeletonLoop(); refreshCameraLabels(); toast('摄像头已开启')
  }catch(e){toast('摄像头启动失败: '+e.message)}
}
function stopCamera() {
  mediaStream.value?.getTracks().forEach(t=>t.stop()); mediaStream.value=null
  if(videoEl.value)videoEl.value.srcObject=null
  cameraOn.value=false; stopRecording()
  detectedCount.value=0; poseQuality.value='idle'; lastPoseData.value=null; stopSkeletonLoop(); stopFrameSending()
}
function startFrameSending() {
  const c=document.createElement('canvas'),ctx=c.getContext('2d')
  frameTimer.value=setInterval(()=>{
    if(!cameraOn.value||wsRef.value?.readyState!==WebSocket.OPEN)return
    const v=videoEl.value; if(!v?.videoWidth)return
    c.width=v.videoWidth;c.height=v.videoHeight;ctx.drawImage(v,0,0)
    c.toBlob(b=>{if(b&&wsRef.value?.readyState===WebSocket.OPEN)b.arrayBuffer().then(buf=>wsRef.value.send(buf))},'image/jpeg',0.70)
  },80)  // ~12fps — 骨架更流畅
}
function stopFrameSending() { if(frameTimer.value){clearInterval(frameTimer.value);frameTimer.value=null} }

/* ======== 骨架 ======== */
let skeletonAnimTime=0, skeletonRafId=null, latestPoseFrame=null
function startSkeletonLoop() {
  if(skeletonRafId)return
  function loop(ts){skeletonRafId=requestAnimationFrame(loop);skeletonAnimTime=ts*0.001;if(!latestPoseFrame)return;ensureCanvasSize();const ctx=skeletonCtx.value;if(!ctx)return;clearCanvas(ctx);renderSkeleton(ctx,latestPoseFrame.keypoints,latestPoseFrame.frame_w,latestPoseFrame.frame_h,{recording:skeletonActive.value,time:skeletonAnimTime,showLabels:true})}
  skeletonRafId=requestAnimationFrame(loop)
}
function stopSkeletonLoop() { if(skeletonRafId){cancelAnimationFrame(skeletonRafId);skeletonRafId=null};latestPoseFrame=null;if(skeletonCtx.value)clearCanvas(skeletonCtx.value) }
function ensureCanvasSize() { const c=skeletonEl.value,v=videoEl.value;if(!c||!v||!v.videoWidth)return;if(c.width!==v.videoWidth||c.height!==v.videoHeight){c.width=v.videoWidth;c.height=v.videoHeight;skeletonCtx.value=c.getContext('2d')} }
function drawPoseOnCanvas(data) { latestPoseFrame=data }

/* ======== 识别/语音/全屏 ======== */
function toggleRecognition() {
  if(!cameraOn.value){toast('请先开启摄像头');return}
  if(isRecognizing.value){
    // 手动停止录制
    stopRecording()
    return
  }
  // ★ 开始倒计时
  startCountdown()
}

function startCountdown() {
  if(countdownActive.value) return
  countdownActive.value = true
  countdownValue.value = 3
  const countdownInterval = setInterval(() => {
    countdownValue.value--
    if(countdownValue.value <= 0) {
      clearInterval(countdownInterval)
      // 倒计时结束，开始录制
      countdownActive.value = false
      startRecording()
    }
  }, 800)  // 0.8秒一步
  countdownTimer = countdownInterval
}

function startRecording() {
  isRecognizing.value = true
  skeletonActive.value = true
  recordingDuration.value = 0
  send({type:'switch_recognition',is_on:true})
  // 计时器
  recordingDurationTimer = setInterval(() => {
    recordingDuration.value += 0.1
    if(recordingDuration.value >= maxRecordDuration) {
      stopRecording()
    }
  }, 100)
}

function stopRecording() {
  if(countdownTimer){clearInterval(countdownTimer);countdownTimer=null}
  if(recordingDurationTimer){clearInterval(recordingDurationTimer);recordingDurationTimer=null}
  countdownActive.value = false
  isRecognizing.value = false
  skeletonActive.value = false
  send({type:'switch_recognition',is_on:false})
  recordingDuration.value = 0
}
function toggleAutoMode() { if(!cameraOn.value){toast('请先开启摄像头');return};send({type:'toggle_auto'}) }

/* ======== 语音转文字 (STT) ======== */
let sttAudioCtx = null
let sttPcmBuffer = []  // Float32 采样数据

function _encodeWav(samples, sampleRate) {
  // 将 Float32 PCM 编码为 16-bit WAV
  const numChannels = 1, bitsPerSample = 16
  const byteRate = sampleRate * numChannels * bitsPerSample / 8
  const blockAlign = numChannels * bitsPerSample / 8
  const dataLen = samples.length * blockAlign
  const buf = new ArrayBuffer(44 + dataLen)
  const view = new DataView(buf)
  // RIFF header
  const _s = (p, v) => { for (let i=0;i<v.length;i++) view.setUint8(p+i, v.charCodeAt(i)) }
  _s(0, 'RIFF'); view.setUint32(4, 36+dataLen, true); _s(8, 'WAVE')
  _s(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, numChannels, true); view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true); view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)
  _s(36, 'data'); view.setUint32(40, dataLen, true)
  // PCM data
  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
    offset += 2
  }
  return new Blob([buf], { type: 'audio/wav' })
}

async function toggleSTT() {
  if (!sttSupported.value) { toast('浏览器不支持录音，请使用 Chrome 或 Edge'); return }
  if (sttProcessing.value) { toast('正在识别中，请稍候...'); return }

  if (sttListening.value) {
    // 停止录制 → 编码 WAV → 发送识别
    sttProcessing.value = true
    sttListening.value = false

    try {
      // 断开音频图并释放麦克风
      if (sttStream) {
        sttStream.getTracks().forEach(t => t.stop())
        sttStream = null
      }
      if (sttMediaRecorder) {
        sttMediaRecorder.disconnect()
        sttMediaRecorder = null
      }
      if (sttAudioCtx) {
        await sttAudioCtx.close()
        sttAudioCtx = null
      }

      if (sttPcmBuffer.length < 16000 * 0.5) {  // 少于0.5秒
        toast('录音太短，请重新录制')
        sttPcmBuffer = []
        sttProcessing.value = false
        return
      }

      toast('正在识别语音...')
      const wavBlob = _encodeWav(sttPcmBuffer, 16000)
      sttPcmBuffer = []

      const formData = new FormData()
      formData.append('audio', wavBlob, 'recording.wav')

      const r = await fetch('/api/stt', { method: 'POST', body: formData })
      if (!r.ok) {
        const err = await r.json()
        throw new Error(err.detail || '识别失败')
      }
      const data = await r.json()
      if (data.text) {
        stageResult.value = '🎙️ ' + data.text
        stageConfidence.value = 0.9
        resetStageResultTimer()
        historyList.value.unshift({
          timestamp: new Date().toLocaleTimeString(),
          text: '[语音] ' + data.text,
          confidence: 0.9
        })
        if (historyList.value.length > 50) historyList.value.pop()
        toast('语音识别完成')
      } else {
        toast('未识别到语音内容')
      }
    } catch (e) {
      console.error('[STT] 错误:', e)
      toast('语音识别失败: ' + (e.message || '未知错误'))
    } finally {
      sttProcessing.value = false
    }
    return
  }

  // 开始录制 — 使用 AudioContext 采集原始 PCM
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    sttStream = stream  // 保存流引用以便停止时释放
    sttAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
    const source = sttAudioCtx.createMediaStreamSource(stream)
    // 使用 ScriptProcessorNode 捕获 PCM（简单兼容）
    const processor = sttAudioCtx.createScriptProcessor(4096, 1, 1)
    sttPcmBuffer = []

    processor.onaudioprocess = (e) => {
      if (!sttListening.value) return
      const input = e.inputBuffer.getChannelData(0)
      // 复制一份数据（避免被覆盖）
      const copy = new Float32Array(input.length)
      copy.set(input)
      for (let i = 0; i < copy.length; i++) sttPcmBuffer.push(copy[i])
    }

    source.connect(processor)
    processor.connect(sttAudioCtx.destination)  // 必须连接 destination 才能触发
    sttMediaRecorder = processor  // 保存引用用于 stop

    sttListening.value = true
    toast('正在录音...请说话')
  } catch (e) {
    console.error('[STT] 麦克风错误:', e)
    toast('无法访问麦克风: ' + (e.message || '权限被拒绝'))
    if (sttStream) { try { sttStream.getTracks().forEach(t => t.stop()) } catch (_) {} sttStream = null }
    if (sttAudioCtx) { try { sttAudioCtx.close() } catch (_) {} sttAudioCtx = null }
    sttListening.value = false
  }
}

async function toggleSpeech() { speechOn.value=!speechOn.value;try{await speechApi.toggle()}catch(_){send({type:'toggle_speech'})} }
function toggleFullscreen() { fullscreen.value=!fullscreen.value; if(fullscreen.value)document.documentElement.requestFullscreen?.().catch(()=>{}); else document.exitFullscreen?.().catch(()=>{}) }
async function clearHistory() { if(!confirm('确定清空所有识别历史？'))return; try{await historyApi.clear()}catch(_){send({type:'clear_history'})} }

/* ======== 设置 ======== */
async function enumerateCameras() { let d=await navigator.mediaDevices.enumerateDevices();let cams=d.filter(x=>x.kind==='videoinput');cameraDevices.value=cams.map((d,i)=>({deviceId:d.deviceId,label:d.label||`摄像头 ${i+1}`}));if(cameraDevices.value.length>0&&!selectedCameraId.value)selectedCameraId.value=cameraDevices.value[0].deviceId }
async function refreshCameraLabels() { let d=await navigator.mediaDevices.enumerateDevices();let cams=d.filter(x=>x.kind==='videoinput');if(cams.length>0)cameraDevices.value=cams.map((d,i)=>({deviceId:d.deviceId,label:d.label||`摄像头 ${i+1}`})) }
async function fetchVoices() { try{const r=await fetch('/api/voices').then(r=>r.json());ttsVoices.value=r.voices||[]}catch(_){ttsVoices.value=[{id:'zh-CN-XiaoxiaoNeural',name:'默认语音'}]} }
async function fetchAudioDevices() { try{const r=await audioApi.getDevices();audioDevices.value=(r.devices||[]).map(d=>({id:d.id!=null?String(d.id):'default',name:d.is_default?d.name+' (默认)':d.name}))}catch(_){audioDevices.value=[{id:'default',name:'系统默认设备'}]} }
async function onVoiceChange() { try{await fetch('/api/voices/set',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voice_id:selectedVoiceId.value})});toast('语音已切换')}catch(_){toast('切换失败')} }
async function onAudioDeviceChange() { try{await audioApi.setDevice(selectedAudioDeviceId.value==='default'?null:Number(selectedAudioDeviceId.value));toast('音频设备已切换')}catch(_){toast('切换失败')} }
async function openXaiPanel() { try{const a=await xaiApi.getAvailability();xaiAvailable.value=a.available}catch(_){xaiAvailable.value=false};xaiPanelOpen.value=!xaiPanelOpen.value }
async function fetchStatus() { try{const r=await statusApi.get();modelReady.value=!!r.model_loaded}catch(_){} }

/* ======== 键盘 ======== */
function onKey(e) {
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return
  const k=e.key.toLowerCase()
  if(k==='a'&&!e.ctrlKey&&!e.metaKey)toggleAutoMode()
  else if(k==='c'&&!e.ctrlKey&&!e.metaKey)toggleCamera()
  else if(k==='r'&&!e.ctrlKey&&!e.metaKey)toggleRecognition()
  else if(k==='f'&&!e.ctrlKey&&!e.metaKey)toggleFullscreen()
  else if(k==='v'&&!e.ctrlKey&&!e.metaKey)toggleSTT()
  else if(k==='x'&&!e.ctrlKey&&!e.metaKey)openXaiPanel()
  else if(k==='h'&&!e.ctrlKey&&!e.metaKey)historyOpen.value=!historyOpen.value
  else if(k==='escape'){ if(historyOpen.value)historyOpen.value=false; else if(settingsOpen.value)settingsOpen.value=false; else if(fullscreen.value)toggleFullscreen() }
  else if(k>='1'&&k<='4'&&!e.ctrlKey) activeTab.value=navTabs[parseInt(k)-1]?.key||'demo'
}

/* ======== 显存自动管理: 切换标签时自动卸载/重载模型 ======== */
let _vramMgrTimer = null
watch(activeTab, (newTab, oldTab) => {
  // 离开实时演示 → 自动卸载模型释放显存
  if (oldTab === 'demo' && newTab !== 'demo') {
    console.log('[VRAM] 离开实时演示，自动清理GPU显存...')
    stopCamera()
    // 延迟卸载，确保摄像头已完全停止
    clearTimeout(_vramMgrTimer)
    _vramMgrTimer = setTimeout(() => {
      vramApi.fullCleanup().then(r => {
        console.log('[VRAM] 清理完成:', r?.vram)
        toast('GPU显存已释放 (节约显存)')
      }).catch(() => {})
    }, 1500)
  }
  // 进入实时演示 → 确保模型在GPU上
  if (newTab === 'demo' && oldTab !== 'demo') {
    console.log('[VRAM] 进入实时演示，重载模型到GPU...')
    clearTimeout(_vramMgrTimer)
    vramApi.reloadModel().then(r => {
      console.log('[VRAM] 模型已就绪:', r?.vram)
    }).catch(() => {})
  }
})
</script>
