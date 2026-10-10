<template>
  <div class="user-page">
    <!-- 背景层 — 8层叠加 (紫色主题) -->
    <div class="user-bg">
      <div class="user-bg-orb user-orb-a"></div>
      <div class="user-bg-orb user-orb-b"></div>
      <div class="user-bg-orb user-orb-c"></div>
      <div class="user-bg-spots">
        <div class="user-spot user-spot-1"></div>
        <div class="user-spot user-spot-2"></div>
        <div class="user-spot user-spot-3"></div>
      </div>
      <div class="user-bg-beams">
        <div class="user-beam"></div>
        <div class="user-beam"></div>
        <div class="user-beam"></div>
        <div class="user-beam"></div>
      </div>
      <div class="user-bg-hex"></div>
      <div class="user-bg-particles">
        <div class="user-particle"></div><div class="user-particle"></div>
        <div class="user-particle"></div><div class="user-particle"></div>
        <div class="user-particle"></div><div class="user-particle"></div>
        <div class="user-particle"></div><div class="user-particle"></div>
        <div class="user-particle"></div><div class="user-particle"></div>
      </div>
      <div class="user-bg-scanlines"></div>
      <div class="user-bg-vignette"></div>
      <div class="user-bg-bottom"></div>
    </div>

    <div class="user-container">
      <!-- Hero -->
      <div class="user-hero">
        <div class="user-hero-badge">USER CENTER</div>
        <h1 class="user-hero-title">用户中心</h1>
        <p class="user-hero-sub">管理账户信息 &nbsp;·&nbsp; 查看识别统计 &nbsp;·&nbsp; 切换账号 &nbsp;·&nbsp; 安全设置</p>
      </div>

      <div class="user-row-2">
        <!-- 左侧: 个人信息卡片 -->
        <div class="user-section user-section-main">
          <div class="user-profile-card">
            <div class="upc-avatar">{{ userInitial }}</div>
            <div class="upc-info">
              <h2>{{ user?.username || '未登录' }}</h2>
              <span class="upc-role">普通用户</span>
              <span class="upc-date">注册时间: {{ user?.created_at || '—' }}</span>
            </div>
            <div class="upc-stats">
              <div class="upc-stat">
                <span class="upcs-val">{{ stats.total_recognitions || 0 }}</span>
                <span class="upcs-lbl">总识别次数</span>
              </div>
              <div class="upc-stat">
                <span class="upcs-val">{{ stats.today_recognitions || 0 }}</span>
                <span class="upcs-lbl">今日识别</span>
              </div>
            </div>
          </div>

          <!-- 快捷操作 -->
          <div class="user-actions-card">
            <h3>快捷操作</h3>
            <div class="ua-grid">
              <button class="ua-btn" @click="switchSection='password'">
                
                <span class="ua-label">修改密码</span>
              </button>
              <button class="ua-btn" @click="switchSection='accounts'">
                
                <span class="ua-label">切换账号</span>
              </button>
              <button class="ua-btn" @click="handleLogout">
                
                <span class="ua-label">退出登录</span>
              </button>
            </div>
          </div>
        </div>

        <!-- 右侧: 功能面板 -->
        <div class="user-section user-section-side">
          <!-- 修改密码 -->
          <div v-if="switchSection==='password'" class="user-panel">
            <div class="user-panel-head">
              
              <h3>修改密码</h3>
            </div>
            <div class="user-panel-body">
              <div class="upb-field">
                <label>原密码</label>
                <input v-model="pwdForm.oldPassword" type="password" placeholder="输入原密码" />
              </div>
              <div class="upb-field">
                <label>新密码</label>
                <input v-model="pwdForm.newPassword" type="password" placeholder="至少4个字符" />
              </div>
              <div class="upb-field">
                <label>确认新密码</label>
                <input v-model="pwdForm.confirmPassword" type="password" placeholder="再次输入新密码" />
              </div>
              <p v-if="pwdMsg" class="upb-msg" :class="pwdOk?'ok':'err'">{{ pwdMsg }}</p>
              <button class="upb-submit" @click="changePassword" :disabled="pwdLoading">
                {{ pwdLoading ? '修改中...' : '确认修改' }}
              </button>
            </div>
          </div>

          <!-- 切换账号 -->
          <div v-else-if="switchSection==='accounts'" class="user-panel">
            <div class="user-panel-head">
              
              <h3>切换账号</h3>
            </div>
            <div class="user-panel-body">
              <p class="upb-hint">选择一个已注册的账号进行切换，或注册新账号</p>
              <div class="account-list">
                <div
                  v-for="acct in accountList"
                  :key="acct.id"
                  class="account-item"
                  :class="{ active: acct.username === user?.username }"
                  @click="switchAccount(acct)"
                >
                  <div class="acct-avatar">{{ acct.username[0]?.toUpperCase() }}</div>
                  <div class="acct-info">
                    <span class="acct-name">{{ acct.username }}</span>
                    <span class="acct-meta">{{ acct.recognition_count }} 次识别</span>
                  </div>
                  <span v-if="acct.username === user?.username" class="acct-badge">当前</span>
                  <span v-else class="acct-arrow">→</span>
                </div>
                <div v-if="accountList.length <= 1" class="account-empty">
                  <p>暂无其他账号</p>
                  <span>请先退出当前账号，注册新账号后再切换</span>
                </div>
              </div>
              <p v-if="switchMsg" class="upb-msg" :class="switchOk?'ok':'err'">{{ switchMsg }}</p>
            </div>
          </div>

          <!-- 默认欢迎面板 -->
          <div v-else class="user-panel">
            <div class="user-panel-head">
              
              <h3>欢迎回来</h3>
            </div>
            <div class="user-panel-body">
              <p class="upb-welcome">
                你好，<strong>{{ user?.username || '用户' }}</strong>！<br><br>
                在这里你可以管理账户信息、查看识别统计、修改密码或切换到其他账号。<br><br>
                选择一个快捷操作开始吧。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { authApi } from '../api/index.js'

const props = defineProps({
  user: Object,
  onLogout: Function,
})

const userInitial = computed(() => (props.user?.username || 'U')[0]?.toUpperCase())
const switchSection = ref('welcome')

const stats = ref({ total_recognitions: 0, today_recognitions: 0 })
const accountList = ref([])

// 修改密码
const pwdForm = ref({ oldPassword: '', newPassword: '', confirmPassword: '' })
const pwdMsg = ref('')
const pwdOk = ref(false)
const pwdLoading = ref(false)

// 切换账号
const switchMsg = ref('')
const switchOk = ref(false)

function handleLogout() {
  if (props.onLogout) props.onLogout()
}

async function loadStats() {
  try {
    const token = localStorage.getItem('unising_token')
    if (token) {
      const data = await authApi.getMe(token)
      if (data.user?.stats) stats.value = data.user.stats
    }
  } catch (_) {}
}

async function loadAccounts() {
  try {
    const data = await authApi.listUsers()
    accountList.value = data.users || []
  } catch (_) {}
}

async function changePassword() {
  pwdMsg.value = ''
  if (!pwdForm.value.oldPassword) { pwdMsg.value = '请输入原密码'; pwdOk.value = false; return }
  if (!pwdForm.value.newPassword || pwdForm.value.newPassword.length < 4) {
    pwdMsg.value = '新密码至少需要4个字符'; pwdOk.value = false; return
  }
  if (pwdForm.value.newPassword !== pwdForm.value.confirmPassword) {
    pwdMsg.value = '两次输入的新密码不一致'; pwdOk.value = false; return
  }
  pwdLoading.value = true
  try {
    const token = localStorage.getItem('unising_token')
    await authApi.changePassword(pwdForm.value.oldPassword, pwdForm.value.newPassword, token)
    pwdMsg.value = '密码修改成功'
    pwdOk.value = true
    pwdForm.value = { oldPassword: '', newPassword: '', confirmPassword: '' }
  } catch (e) {
    pwdMsg.value = e.response?.data?.detail || e.message || '修改失败'
    pwdOk.value = false
  } finally {
    pwdLoading.value = false
  }
}

async function switchAccount(acct) {
  if (acct.username === props.user?.username) return
  switchMsg.value = ''
  // 切换账号需要重新登录，这里引导用户回到登录页
  handleLogout()
  // 预填用户名
  localStorage.setItem('unising_last_username', acct.username)
  switchMsg.value = `正在切换到 ${acct.username}...`
  switchOk.value = true
}

onMounted(() => {
  loadStats()
  loadAccounts()
})
</script>
