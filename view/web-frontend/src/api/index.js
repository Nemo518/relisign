import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// 拦截器
api.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// 识别历史相关
export const historyApi = {
  getList: (limit = 50) => api.get(`/history?limit=${limit}`),
  clear: () => api.post('/history/clear')
}

// 语音相关
export const speechApi = {
  toggle: () => api.post('/speech/toggle'),
  getStatus: () => api.get('/speech/status'),
  getVoices: () => api.get('/speech/voices')
}

// 摄像头相关
export const cameraApi = {
  getList: () => api.get('/cameras'),
  setCamera: (cameraId) => api.post('/cameras/set', { camera_id: cameraId })
}

// 音频输出设备相关
export const audioApi = {
  getDevices: () => api.get('/audio/devices'),
  setDevice: (deviceId) => api.post('/audio/set-device', { device_id: deviceId })
}

// 状态相关
export const statusApi = {
  get: () => api.get('/status')
}

// 用户认证与账户管理
export const authApi = {
  register: (username, password) => api.post('/auth/register', { username, password }),
  login: (username, password) => api.post('/auth/login', { username, password }),
  logout: (token) => api.post('/auth/logout', { token }),
  getMe: (token) => api.get(`/auth/me?token=${token || ''}`),
  setUser: (username) => api.post('/auth/set-user', { username }),
  changePassword: (oldPassword, newPassword, token) =>
    api.post(`/auth/change-password?token=${token || ''}`, { old_password: oldPassword, new_password: newPassword }),
  listUsers: () => api.get('/auth/users'),
}

// XAI 可解释性分析 (实时单样本)
export const xaiApi = {
  getAvailability: () => api.get('/xai/availability'),
  getLatest: () => api.get('/xai/latest'),
  analyze: (referenceText) => api.post('/xai/analyze', { reference_text: referenceText || null })
}

// 批量归因分析 (数据集测试集)
export const attributionApi = {
  getModels: () => api.get('/attribution/models'),
  getReport: (level = 'A', model = 'best_checkpoint', split = 'test', refresh = false) =>
    api.get(`/attribution/report?level=${level}&model=${model}&split=${split}&refresh=${refresh}`),
  getSample: (videoId, level = 'A', model = 'best_checkpoint', split = 'test') =>
    api.get(`/attribution/sample/${videoId}?level=${level}&model=${model}&split=${split}`),
  triggerInference: (model = 'best_checkpoint', level = 'A', split = 'test') =>
    api.post(`/attribution/infer?model=${model}&level=${level}&split=${split}`),
  getInferStatus: () => api.get('/attribution/infer-status'),
}

// WebSocket连接
export function createWebSocket(onMessage, onError) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)
  
  ws.binaryType = 'arraybuffer'
  
  ws.onopen = () => {
    console.log('WebSocket连接已建立')
    // 启动心跳
    setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30000)
  }
  
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const data = JSON.parse(event.data)
      onMessage && onMessage(data)
    }
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket错误:', error)
    onError && onError(error)
  }
  
  return ws
}

// 显存管理
export const vramApi = {
  getStatus: () => api.get('/vram'),
  unloadModel: () => api.post('/model/unload'),
  reloadModel: () => api.post('/model/reload'),
  fullCleanup: () => api.post('/model/cleanup'),
}

export default api
