/**
 * COCO-133 增强骨架渲染引擎 v3
 * 零延迟实时渲染 — 粗线条、常亮发光、高对比度
 */

// 与 ModelShowcase 完全一致的色彩配置
const PART_STYLE = {
  body:      { hex: '#f0c060', rgb: [240,192,96],   glow: 'rgba(240,192,96,0.55)',    w: 1.2, jr: 1.3 },
  left_hand: { hex: '#f09060', rgb: [240,144,96],   glow: 'rgba(240,144,96,0.50)',    w: 0.8, jr: 1.0 },
  right_hand:{ hex: '#60d0a0', rgb: [96,208,160],   glow: 'rgba(96,208,160,0.50)',    w: 0.8, jr: 1.0 },
  face:      { hex: '#c0a0e0', rgb: [192,160,224],  glow: 'rgba(192,160,224,0.40)',   w: 0.6, jr: 0.8 },
}

const PART_RANGES = {
  body: [0, 22], face: [23, 90], left_hand: [91, 111], right_hand: [112, 132],
}

const SKELETON_CONNS = [
  [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
  [11,12],[11,13],[12,14],[13,15],[14,16],[5,11],[6,12],
  [0,17],[0,18],[17,19],[18,20],[19,21],[20,22],[11,17],[12,18],[5,17],
  [91,92],[92,93],[93,94],[94,95],[91,96],[96,97],[97,98],[98,99],
  [91,100],[100,101],[101,102],[102,103],[91,104],[104,105],[105,106],[106,107],
  [91,108],[108,109],[109,110],[110,111],
  [112,113],[113,114],[114,115],[115,116],[112,117],[117,118],[118,119],[119,120],
  [112,121],[121,122],[122,123],[123,124],[112,125],[125,126],[126,127],[127,128],
  [112,129],[129,130],[130,131],[131,132],
]

const KEY_LABELS = {
  0:'头', 5:'左肩', 6:'右肩', 7:'左肘', 8:'右肘',
  9:'左腕', 10:'右腕', 11:'左髋', 12:'右髋', 91:'左手', 112:'右手',
}

function getStyle(i) {
  for (const [p, [s, e]] of Object.entries(PART_RANGES))
    if (i >= s && i <= e) return PART_STYLE[p]
  return PART_STYLE.body
}

function isKeyJoint(i) { return [0,5,6,7,8,9,10,11,12,91,112].includes(i) }

// 残影存储
let trails = []
const MAX_TRAIL = 5

export function renderSkeleton(ctx, kps, fw, fh, opts = {}) {
  const { recording = false, time = 0, showLabels = true } = opts
  const cw = ctx.canvas.width
  const ch = ctx.canvas.height
  const sx = cw / fw
  const sy = ch / fh

  if (!kps || kps.length === 0) return
  const n = Math.min(kps.length, 133)

  // 坐标映射
  const px = new Float32Array(n), py = new Float32Array(n), cf = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    px[i] = (kps[i][0] || 0) * fw * sx
    py[i] = (kps[i][1] || 0) * fh * sy
    cf[i] = kps[i][2] || 0
  }

  const breathe = 1 + Math.sin(time * 2.5) * 0.06

  ctx.save()

  // ── 运动残影 ──
  for (let t = 0; t < trails.length; t++) {
    const tr = trails[t]
    const alpha = (t + 1) / (MAX_TRAIL * 3)
    for (const [a, b] of SKELETON_CONNS) {
      if (a >= n || b >= n) continue
      if ((tr[a]||0) < 0.1 || (tr[b]||0) < 0.1) continue
      ctx.beginPath()
      ctx.moveTo(tr[`px${a}`], tr[`py${a}`])
      ctx.lineTo(tr[`px${b}`], tr[`py${b}`])
      ctx.strokeStyle = `rgba(212,176,104,${(alpha*0.35).toFixed(3)})`
      ctx.lineWidth = 0.8 * alpha
      ctx.lineCap = 'round'
      ctx.stroke()
    }
  }

  // 更新残影
  const nt = { time }
  for (let i = 0; i < n; i++) { nt[i] = cf[i]; nt[`px${i}`] = px[i]; nt[`py${i}`] = py[i] }
  trails.push(nt)
  while (trails.length > MAX_TRAIL) trails.shift()

  // ── 骨骼连线（常亮发光） ──
  for (const [a, b] of SKELETON_CONNS) {
    if (a >= n || b >= n) continue
    const ca = cf[a], cb = cf[b]
    if (ca < 0.08 || cb < 0.08) continue
    const alpha = Math.min(ca, cb)
    const st = getStyle(a)
    const w = st.w * alpha * (recording ? breathe : 1)

    ctx.beginPath()
    ctx.moveTo(px[a], py[a])
    ctx.lineTo(px[b], py[b])

    // 发光层（对齐 ModelShowcase: shadowBlur + 半透明）
    ctx.shadowColor = st.glow
    ctx.shadowBlur = 1.5
    ctx.strokeStyle = st.hex
    ctx.globalAlpha = 0.35 * alpha
    ctx.lineWidth = w + 0.5
    ctx.lineCap = 'round'
    ctx.stroke()
    ctx.shadowBlur = 0
    ctx.globalAlpha = 1

    // 主体线
    ctx.strokeStyle = `rgba(${st.rgb.join(',')},${(alpha*0.85).toFixed(2)})`
    ctx.lineWidth = w
    ctx.stroke()
  }

  // ── 关键点（ModelShowcase风格 + 稍大可见） ──
  for (let i = 0; i < n; i++) {
    if (cf[i] < 0.10) continue
    const st = getStyle(i)
    const conf = cf[i]
    const r = 0.8 + conf * 2.2  // 0.8~3.0px, 粗犷清晰

    ctx.beginPath()
    ctx.arc(px[i], py[i], r + 0.4, 0, Math.PI*2)
    ctx.fillStyle = st.glow
    ctx.globalAlpha = 0.12
    ctx.fill()

    ctx.beginPath()
    ctx.arc(px[i], py[i], r, 0, Math.PI*2)
    ctx.fillStyle = st.hex
    ctx.globalAlpha = 0.75
    ctx.fill()
    ctx.globalAlpha = 1
  }

  // ── 部位标签 ──
  if (showLabels) {
    ctx.shadowBlur = 0
    for (const [is, label] of Object.entries(KEY_LABELS)) {
      const i = parseInt(is)
      if (i >= n || cf[i] < 0.4) continue
      const st = getStyle(i)
      const x = px[i] + 16, y = py[i] - 6
      const fs = Math.max(11, cw / 60)
      ctx.font = `bold ${fs}px "Microsoft YaHei","PingFang SC",sans-serif`
      // 阴影描边
      ctx.fillStyle = 'rgba(0,0,0,0.7)'
      ctx.fillText(label, x+1, y+1)
      ctx.fillText(label, x-1, y-1)
      ctx.fillText(label, x+1, y-1)
      ctx.fillText(label, x-1, y+1)
      // 主体
      ctx.fillStyle = st.hex
      ctx.fillText(label, x, y)
    }
  }

  ctx.restore()
}

export function clearCanvas(ctx) {
  if (ctx) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
  trails = []
}

export { PART_STYLE, PART_RANGES, SKELETON_CONNS }
