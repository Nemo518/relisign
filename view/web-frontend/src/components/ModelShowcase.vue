<template>
  <div class="showcase-page">
    <div class="showcase-bg">
      <div class="showcase-orb sc-orb-1"></div>
      <div class="showcase-orb sc-orb-2"></div>
      <div class="showcase-bg-spots">
        <div class="showcase-spot showcase-spot-1"></div>
        <div class="showcase-spot showcase-spot-2"></div>
      </div>
      <div class="showcase-grid"></div>
    </div>

    <div class="showcase-container">
      <div class="showcase-hero">
        <h1 class="showcase-title">模型展示 <span class="sc-ver-badge">v3</span></h1>
        <div class="showcase-topbar">
          <select v-model="selectedVideoId" class="sc-select" @change="onVideoChange">
            <option value="">-- 选择样例视频 --</option>
            <option v-for="v in videoList" :key="v.video_id" :value="v.video_id">
              {{ v.video_id }} [{{ v.split }}/{{ v.level }}] {{ v.reference.substring(0, 15) }}...
            </option>
          </select>
          <div class="sc-mode-group">
            <button :class="['sc-mode-btn', {active: mode==='lightweight'}]" @click="mode='lightweight'">轻量</button>
            <button :class="['sc-mode-btn', {active: mode==='normal'}]" @click="mode='normal'">标准</button>
            <button :class="['sc-mode-btn', {active: mode==='ultra'}]" @click="mode='ultra'">超量</button>
          </div>
          <button class="sc-infer-btn" @click="startDemo" :disabled="!selectedVideoId || inferLoading">
            {{ inferLoading ? '推理中...' : '开始演示' }}
          </button>
          <button class="sc-ctrl-btn" @click="toggleAll" :disabled="!selectedVideoId">
            {{ allRunning ? '全部暂停' : '全部播放' }}
          </button>
          <div class="sc-upload-group">
            <label class="sc-upload-btn">
              <input type="file" accept="video/*" class="sc-file-input" @change="onFileChange" />
              选择视频文件
            </label>
            <button class="sc-infer-btn" @click="analyzeUpload" :disabled="!uploadFile || inferLoading">
              {{ inferLoading ? '分析中...' : '分析上传视频' }}
            </button>
            <span v-if="uploadFile" class="sc-file-name" :title="uploadFile.name">已选: {{ uploadFile.name }}</span>
          </div>
        </div>
      </div>

      <!-- 左右等大面板 -->
      <div class="showcase-main">
        <!-- 左侧: 视频播放器 + 置信度面板 -->
        <div class="showcase-pane">
          <div class="sc-pane-label">视频播放</div>
          <div class="sc-video-wrap">
            <video ref="videoEl" :src="videoSrc" muted playsinline class="sc-video"
                   @loadedmetadata="onVideoLoaded"
                   @timeupdate="onVideoTimeUpdate"
                   @seeked="onVideoSeeked"
                   @ended="onVideoEnded"></video>
            <div v-if="!videoSrc" class="sc-video-placeholder">
              <span>选择样例视频或上传自定义视频</span>
            </div>
          </div>
          <!-- 五大部位置信度 -->
          <div class="sc-confidence" v-if="currentConfidence">
            <div class="sc-conf-title">部位置信度</div>
            <div class="sc-conf-bars">
              <div class="sc-conf-item" v-for="p in confParts" :key="p.key">
                <span class="sc-conf-label">{{ p.label }}</span>
                <div class="sc-conf-track">
                  <div class="sc-conf-fill" :style="{width: Math.min(currentConfidence[p.key]*100, 100)+'%', background: p.color}"></div>
                </div>
                <span class="sc-conf-val">{{ Math.min((currentConfidence[p.key]*100), 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 右侧: 姿态可视化 -->
        <div class="showcase-pane">
          <div class="sc-pane-label">姿态提取可视化</div>
          <div class="sc-canvas-wrap">
            <canvas ref="poseCanvas" class="sc-canvas"></canvas>
            <div v-if="!poseFrames.length" class="sc-canvas-placeholder">
              <span>选择样例视频或上传视频后点击「分析」</span>
              <span>右侧将展示关键点姿态提取结果</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 底部控制 + 结果 -->
      <div class="showcase-bottom">
        <div class="sc-bottom-left">
          <div class="sc-toggle-row">
            <span class="sc-toggle-label">模型推理</span>
            <button :class="['sc-toggle', {active: inferenceEnabled}]" @click="inferenceEnabled=!inferenceEnabled">
              {{ inferenceEnabled ? 'ON' : 'OFF' }}
            </button>
          </div>
          <div class="sc-toggle-row">
            <span class="sc-toggle-label">RGB-Support</span>
            <button :class="['sc-toggle', {active: rgbSupport}]" @click="rgbSupport=!rgbSupport">
              {{ rgbSupport ? 'ON' : 'OFF' }}
            </button>
          </div>
        </div>
        <div class="sc-bottom-right" v-if="result">
          <span class="sc-br-item" v-if="groundTruth">真实: <strong class="ref">{{ groundTruth }}</strong></span>
          <span class="sc-br-item">预测: <strong class="pred">{{ result.prediction }}</strong></span>
          <span class="sc-br-item" v-if="groundTruth">相似度: <strong :class="simClass">{{ similarity }}%</strong></span>
          <span class="sc-br-item">帧数: <strong>{{ result.frame_count }}</strong></span>
          <span class="sc-br-item" v-if="result.latency_s != null">耗时: <strong>{{ result.latency_s }}s</strong></span>
          <span class="sc-br-item" v-if="result.device">设备: <strong class="pred">{{ result.device }}</strong></span>
        </div>
        <div class="sc-bottom-right" v-else>
          <span class="sc-br-hint">选择样例视频或上传自定义视频进行分析</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import api from '../api/index.js'

const videoList = ref([])
const selectedVideoId = ref('')
const videoSrc = ref('')
const videoEl = ref(null)
const videoDuration = ref(0)

const inferenceEnabled = ref(true)
const rgbSupport = ref(false)
const mode = ref('normal')
const inferLoading = ref(false)
const result = ref(null)

const allRunning = ref(false)

const uploadFile = ref(null)
const uploadMode = ref(false)

const groundTruth = computed(() => {
  // 上传模式: 真实标注由后端按文件名从数据集 CSV 查回, 放在 result.reference
  if (uploadMode.value) return result.value?.reference || ''
  const v = videoList.value.find(x => x.video_id === selectedVideoId.value)
  return v?.reference || ''
})

const similarity = computed(() => {
  if (!result.value?.prediction || !groundTruth.value) return 0
  const ref = groundTruth.value.replace(/\s/g, '')
  const pred = result.value.prediction.replace(/\s/g, '')
  if (!ref || !pred) return 0
  const m = ref.length, n = pred.length
  const dp = Array.from({length: m+1}, (_,i) => Array.from({length: n+1}, (_,j) => i===0?j: j===0?i: 0))
  for (let i=1; i<=m; i++) for (let j=1; j<=n; j++)
    dp[i][j] = Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(ref[i-1]===pred[j-1]?0:1))
  return ((1 - dp[m][n] / Math.max(m, n)) * 100).toFixed(1)
})

const simClass = computed(() => {
  const v = parseFloat(similarity.value)
  if (v >= 70) return 'ok'
  if (v >= 40) return 'warn'
  return 'fail'
})

// 姿态可视化
const poseCanvas = ref(null)
const poseFrames = ref([])
const confidenceTimeline = ref([])
const currentFrame = ref(1)
const currentConfidence = ref(null)

const confParts = [
  { key: 'body', label: '身体', color: '#d4b068' },
  { key: 'feet', label: '足部', color: '#e8a87c' },
  { key: 'left_hand', label: '左手', color: '#8cc8a0' },
  { key: 'right_hand', label: '右手', color: '#4da6d8' },
  { key: 'face', label: '面部', color: '#b8a0d0' },
]

function drawFrame() {
  const canvas = poseCanvas.value
  if (!canvas || !poseFrames.value.length) return
  const ctx = canvas.getContext('2d')
  const w = canvas.parentElement?.clientWidth || 800
  const h = canvas.parentElement?.clientHeight || 600
  canvas.width = w; canvas.height = h

  // 背景: 深色渐变 + 微网格
  const bgGrad = ctx.createRadialGradient(w/2, h/3, 0, w/2, h/2, Math.max(w, h)*0.7)
  bgGrad.addColorStop(0, '#14121a'); bgGrad.addColorStop(1, '#06050a')
  ctx.fillStyle = bgGrad; ctx.fillRect(0, 0, w, h)
  ctx.strokeStyle = 'rgba(255,255,255,0.02)'; ctx.lineWidth = 0.5
  for (let gx = 0; gx < w; gx += 40) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke() }
  for (let gy = 0; gy < h; gy += 40) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke() }

  const kp = poseFrames.value[Math.min(currentFrame.value - 1, poseFrames.value.length - 1)]
  if (!kp) return
  const scaleX = w * 0.82, scaleY = h * 0.82
  const offsetX = w * 0.09, offsetY = h * 0.08

  // 部位定义
  const parts = {
    body:    { indices: [...Array(17).keys()], color: '#f0c060', glow: 'rgba(240,192,96,0.6)' },
    face:    { indices: [...Array(68).keys()].map(i => i+23).filter(i => i<91), color: '#c0a0e0', glow: 'rgba(192,160,224,0.5)' },
    lhand:   { indices: [...Array(21).keys()].map(i => i+91), color: '#f09060', glow: 'rgba(240,144,96,0.5)' },
    rhand:   { indices: [...Array(21).keys()].map(i => i+112), color: '#60d0a0', glow: 'rgba(96,208,160,0.5)' },
  }

  // 骨架连接
  const bodyLines = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],[11,12],[11,13],[12,14],[13,15],[14,16],[5,11],[6,12]]
  const faceLines = [[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],
    [39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49],[49,50],[50,51],
    [53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,60],[60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,68],
    [23,39],[24,40],[25,41],[53,23],[53,39],[53,54],[69,70],[70,71],[71,72],[72,73],[73,74],[74,75],[75,76],[76,77],[77,78],[78,79],[79,80],
    [80,81],[81,82],[82,83],[83,84],[84,85],[85,86],[86,87],[87,88],[88,89],[89,90],[90,69]]

  const getXY = (i) => ({ x: kp[i][0]*scaleX+offsetX, y: kp[i][1]*scaleY+offsetY, ok: kp[i][2] >= 0.1 })

  // 画发光背景线
  const drawGlowLines = (lines, color, glowColor, lw) => {
    if (!lines.length) return
    ctx.save()
    ctx.shadowColor = glowColor; ctx.shadowBlur = 1.5
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.globalAlpha = 0.25
    ctx.beginPath()
    for (const [a, b] of lines) {
      const pa = getXY(a), pb = getXY(b)
      if (!pa.ok || !pb.ok) continue
      ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y)
    }
    ctx.stroke(); ctx.restore()
  }

  // 画主骨架线
  const drawMainLines = (lines, color, lw) => {
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.globalAlpha = 0.9
    ctx.beginPath()
    for (const [a, b] of lines) {
      const pa = getXY(a), pb = getXY(b)
      if (!pa.ok || !pb.ok) continue
      ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y)
    }
    ctx.stroke(); ctx.globalAlpha = 1
  }

  drawGlowLines(bodyLines, parts.body.color, parts.body.glow, 1.2)
  drawMainLines(bodyLines, parts.body.color, 0.8)
  drawGlowLines(faceLines, parts.face.color, parts.face.glow, 0.8)
  drawMainLines(faceLines, parts.face.color, 0.5)

  // 手部连线
  const handLines = (base) => {
    const lines = []
    for (let i=0; i<4; i++) for (let j=0; j<4; j++) {
      lines.push([base+i*4+j, base+i*4+j+1])
      if (i<3) lines.push([base+i*4+j, base+(i+1)*4+j])
    }
    return lines
  }
  drawGlowLines(handLines(91), parts.lhand.color, parts.lhand.glow, 0.8)
  drawMainLines(handLines(91), parts.lhand.color, 0.5)
  drawGlowLines(handLines(112), parts.rhand.color, parts.rhand.glow, 0.8)
  drawMainLines(handLines(112), parts.rhand.color, 0.5)

  // 画关键点 (大小随置信度)
  for (const [name, part] of Object.entries(parts)) {
    for (const i of part.indices) {
      if (i >= kp.length) continue
      const conf = kp[i][2]
      if (conf < 0.1) continue
      const { x, y } = getXY(i)
      const r = 0.3 + conf * 1.0   // 极微型关键点: 0.3~1.3px
      // 外发光
      ctx.beginPath(); ctx.arc(x, y, r+0.4, 0, Math.PI*2)
      ctx.fillStyle = part.glow; ctx.globalAlpha = 0.12; ctx.fill()
      // 内点
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2)
      ctx.fillStyle = part.color; ctx.globalAlpha = 0.75; ctx.fill()
    }
  }
  ctx.globalAlpha = 1
}

// 视频播放时同步驱动姿态帧
function syncPoseToVideo() {
  if (!videoEl.value || !poseFrames.value.length) return
  const ratio = videoEl.value.currentTime / (videoEl.value.duration || 1)
  const idx = Math.min(Math.floor(ratio * poseFrames.value.length), poseFrames.value.length - 1)
  if (idx >= 0 && idx !== currentFrame.value - 1) {
    currentFrame.value = idx + 1
    drawFrame()
    // 同步置信度
    if (confidenceTimeline.value.length > idx) {
      currentConfidence.value = confidenceTimeline.value[idx]
    }
  }
}
function onVideoTimeUpdate() { if (allRunning.value) syncPoseToVideo() }
function onVideoSeeked() { syncPoseToVideo() }
function onVideoEnded() { pauseAll() }

function playAll() {
  if (!videoEl.value) return
  videoEl.value.playbackRate = 1.0
  videoEl.value.play()
  allRunning.value = true
  syncPoseToVideo()
}
function pauseAll() {
  if (videoEl.value) videoEl.value.pause()
  allRunning.value = false
}
function toggleAll() {
  if (allRunning.value) pauseAll()
  else playAll()
}

const showcaseModel = ref('best_checkpoint')

async function loadVideoList() {
  try { const r = await api.get(`/showcase/videos?split=test&model=${showcaseModel.value}`); videoList.value = r.videos || [] }
  catch (_) {}
}

function onVideoChange() {
  const v = videoList.value.find(x => x.video_id === selectedVideoId.value)
  videoSrc.value = v ? `/api/showcase/video-file?path=${encodeURIComponent(v.path)}` : ''
  uploadMode.value = false
  result.value = null; poseFrames.value = []; stopPoseAnimation(); allRunning.value = false
}
function onVideoLoaded() { if (videoEl.value) videoDuration.value = videoEl.value.duration }

function onFileChange(e) {
  const f = e.target.files && e.target.files[0]
  if (!f) return
  uploadFile.value = f
  uploadMode.value = true
  selectedVideoId.value = ''
  videoSrc.value = URL.createObjectURL(f)
  result.value = null; poseFrames.value = []; confidenceTimeline.value = []
  currentConfidence.value = null; allRunning.value = false
  e.target.value = ''  // 允许再次选择同一文件
}

async function startDemo() {
  if (!selectedVideoId.value || inferLoading.value) return
  inferLoading.value = true; result.value = null; poseFrames.value = []
  const v = videoList.value.find(x => x.video_id === selectedVideoId.value)
  if (!v) { inferLoading.value = false; return }
  try {
    const r = await api.post('/showcase/infer', {
      video_path: v.path, video_id: v.video_id, model: showcaseModel.value,
      rgb_support: rgbSupport.value,
      lightweight: mode.value === 'lightweight',
      ultra_lightweight: mode.value === 'ultra',
    })
    result.value = r
    if (r.pose_keypoints) {
      poseFrames.value = r.pose_keypoints
      confidenceTimeline.value = r.confidence_timeline || []
      currentFrame.value = 1
      currentConfidence.value = null
      await nextTick()
      drawFrame()
      playAll()
    }
  } catch (e) { console.error(e) }
  finally { inferLoading.value = false }
}

async function analyzeUpload() {
  if (!uploadFile.value || inferLoading.value) return
  inferLoading.value = true; result.value = null; poseFrames.value = []
  try {
    const fd = new FormData()
    fd.append('video', uploadFile.value)
    fd.append('lightweight', String(mode.value === 'lightweight'))
    fd.append('ultra_lightweight', String(mode.value === 'ultra'))
    const r = await api.post('/video/analyze', fd, { timeout: 300000 })
    result.value = r
    if (r.pose_keypoints) {
      poseFrames.value = r.pose_keypoints
      confidenceTimeline.value = r.confidence_timeline || []
      currentFrame.value = 1
      currentConfidence.value = null
      await nextTick()
      drawFrame()
      playAll()
    }
  } catch (e) {
    console.error(e)
    result.value = { prediction: '分析失败', frame_count: 0 }
  } finally { inferLoading.value = false }
}

function stopPoseAnimation() { /* 无动画循环，占位以避免引用错误 */ }

onMounted(() => { loadVideoList() })
onUnmounted(() => { stopPoseAnimation() })
</script>

<style scoped>
.sc-ver-badge { font-size: 13px; font-weight: 700; color: #1a1820; background: var(--gold);
  padding: 2px 10px; border-radius: var(--radius-full); vertical-align: middle; margin-left: 6px; }
.sc-upload-group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.sc-file-input { position: absolute; width: 1px; height: 1px; opacity: 0; overflow: hidden; clip: rect(0 0 0 0); }
.sc-upload-btn { position: relative; display: inline-flex; align-items: center; gap: 6px; padding: 14px 30px;
  border-radius: var(--radius-full); border: 1px solid var(--gold); background: var(--gold); color: #1a1820;
  font-size: 17px; font-weight: 800; cursor: pointer; font-family: inherit; white-space: nowrap;
  transition: all 0.2s; box-shadow: 0 4px 18px rgba(200,164,92,0.35); }
.sc-upload-btn:hover { background: #dbb86a; box-shadow: 0 6px 24px rgba(200,164,92,0.5); }
.sc-file-name { font-size: 13px; color: var(--text-muted); max-width: 220px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
</style>
