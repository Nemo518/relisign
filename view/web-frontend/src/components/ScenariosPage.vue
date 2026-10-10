<template>
  <div class="scen-page">
    <!-- 专属背景 — 8层叠加 -->
    <div class="scen-bg">
      <div class="scen-bg-orb scen-orb-a"></div>
      <div class="scen-bg-orb scen-orb-b"></div>
      <div class="scen-bg-orb scen-orb-c"></div>
      <div class="scen-bg-spots">
        <div class="scen-spot scen-spot-1"></div>
        <div class="scen-spot scen-spot-2"></div>
        <div class="scen-spot scen-spot-3"></div>
      </div>
      <div class="scen-bg-beams">
        <div class="scen-beam"></div>
        <div class="scen-beam"></div>
        <div class="scen-beam"></div>
        <div class="scen-beam"></div>
      </div>
      <div class="scen-bg-hex"></div>
      <div class="scen-bg-particles">
        <div class="scen-particle"></div><div class="scen-particle"></div>
        <div class="scen-particle"></div><div class="scen-particle"></div>
        <div class="scen-particle"></div><div class="scen-particle"></div>
        <div class="scen-particle"></div><div class="scen-particle"></div>
        <div class="scen-particle"></div><div class="scen-particle"></div>
      </div>
      <div class="scen-bg-scanlines"></div>
      <div class="scen-bg-vignette"></div>
      <div class="scen-bg-bottom"></div>
    </div>

    <div class="scen-container">
      <!-- Hero -->
      <div class="scen-hero">
        <div class="scen-hero-badge">APPLICATION SCENARIOS</div>
        <h1 class="scen-hero-title">应用场景</h1>
        <p class="scen-hero-sub">赋能听障人士无障碍沟通 &nbsp;·&nbsp; 医疗 · 教育 · 政务 · 终端 · 媒体 · 交互 &nbsp;—&nbsp; 六大领域深度落地</p>
      </div>

      <!-- 01 落地场景 — 3×2 网格 -->
      <div class="scen-section">
        <div class="scen-section-label">DEPLOYMENT</div>
        <div class="scen-section-head">
          <span class="ss-num">01</span>
          <h2 class="ss-title">落地场景</h2>
          <p class="ss-desc">六大核心领域全面覆盖，每个场景针对手语翻译的差异化需求深度定制解决方案</p>
        </div>
        <div class="scen-grid">
          <div class="scen-card" v-for="(s, idx) in scenarios" :key="s.title" :style="{'--si': idx}">
            <div class="scen-card-accent"></div>
            <div class="scen-card-head">
              <div>
                <h3>{{ s.title }}</h3>
                <span class="scen-card-domain">场景 {{ idx + 1 }} / 6</span>
              </div>
            </div>
            <p>{{ s.desc }}</p>
            <ul class="scen-points">
              <li v-for="pt in s.points" :key="pt">{{ pt }}</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 02+03 并排: 社会价值 + 核心优势 -->
      <div class="scen-row-2">
        <!-- 02 社会价值 -->
        <div class="scen-section scen-section-half">
          <div class="scen-section-label">SOCIAL IMPACT</div>
          <div class="scen-section-head">
            <span class="ss-num">02</span>
            <h2 class="ss-title">社会价值</h2>
          </div>
          <p class="ss-desc">全球约 4.66 亿听障人士 — 手语翻译技术将从根本上改变这一群体的沟通方式</p>
          <div class="impact-grid">
            <div class="impact-card" v-for="imp in impacts" :key="imp.label">
              <div class="impact-value">{{ imp.value }}<span>{{ imp.suffix }}</span></div>
              <div class="impact-label">{{ imp.label }}</div>
              <div class="impact-sub">{{ imp.sub }}</div>
            </div>
          </div>
        </div>

        <!-- 03 核心竞争优势 -->
        <div class="scen-section scen-section-half">
          <div class="scen-section-label">COMPETITIVE EDGE</div>
          <div class="scen-section-head">
            <span class="ss-num">03</span>
            <h2 class="ss-title">核心竞争优势</h2>
          </div>
          <div class="adv-grid">
            <div class="adv-card" v-for="a in advantages" :key="a.title">
              <div class="adv-body">
                <h4>{{ a.title }}</h4>
                <p>{{ a.desc }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const scenarios = [
  { icon:'', title:'医疗问诊', desc:'帮助听障患者在就医时与医生无障碍沟通，准确表达症状与需求。支持急诊分诊、常规问诊、用药指导等场景，大幅提升诊疗效率与准确性。', points:['急诊快速分诊沟通','病史采集与症状描述','用药指导与康复说明'] },
  { icon:'', title:'特殊教育', desc:'辅助手语教学与课堂评估，为听障学生提供实时翻译支持。帮助教师准确理解学生表达，促进融合教育发展，让课堂真正无碍。', points:['课堂实时双向翻译','手语学习智能评估','远程教学无障碍支持'] },
  { icon:'', title:'政务服务', desc:'在政务大厅、银行网点、法院立案窗口等公共服务场所部署手语翻译终端，保障听障人士平等获取公共服务的合法权益。', points:['政务窗口即时翻译','银行柜台业务办理','司法程序沟通保障'] },
  { icon:'', title:'智能终端', desc:'集成至智能手机、平板电脑、智能屏等消费电子设备，打造随身手语翻译助手。用户只需打开应用即可随时随地与健听人士顺畅交流。', points:['手机 App 实时翻译','智能音箱语音播报','可穿戴设备便携方案'] },
  { icon:'', title:'媒体传播', desc:'为新闻播报、在线课程、短视频等视频内容自动生成手语翻译字幕，扩大信息传播的受众覆盖面，推动信息无障碍建设。', points:['新闻直播双语字幕','在线教育内容适配','短视频平台无障碍'] },
  { icon:'', title:'人机交互', desc:'作为新一代自然人机交互方式，支持手语控制智能家居设备、AR/VR 虚拟环境操作与服务机器人指令输入，拓展交互边界。', points:['智能家居手势控制','AR/VR 虚拟交互','服务机器人指令输入'] },
]

const impacts = [
  { value:'4.66', suffix:'亿', label:'全球听障人口', sub:'世界卫生组织统计数据' },
  { value:'70', suffix:'+', label:'世界各国手语', sub:'每种手语独立语法体系' },
  { value:'<500', suffix:'ms', label:'实时翻译延迟', sub:'端到端全链路推理时间' },
  { value:'24/7', suffix:'', label:'全天候服务', sub:'不受时间地点限制' },
]

const advantages = [
  { icon:'', title:'实时响应', desc:'端到端延迟 &lt;500ms，支持流式摄像头输入，无需等待完整视频录制即可实时输出翻译结果' },
  { icon:'', title:'全身建模', desc:'133 个关键点覆盖身体躯干、双手和面部，比传统仅建模手部的方法捕捉更丰富的手语表达信息' },
  { icon:'', title:'可解释性', desc:'内置 Grad-CAM 和 Self-Attention 分析，用户可直观理解模型决策依据，增强系统信任与可靠性' },
  { icon:'', title:'轻量部署', desc:'支持纯 CPU 推理 + PyInstaller 打包为独立桌面应用，无需 GPU 即可在普通电脑上流畅运行' },
  { icon:'', title:'多语言支持', desc:'基于多语言 mT5 模型架构，可无缝扩展中文、英文及其他语言的手语翻译，覆盖更广泛用户群体' },
  { icon:'', title:'隐私安全', desc:'所有推理计算在本地完成，原始视频数据不上传云端，充分保护用户隐私，适合医疗政务等敏感场景' },
]
</script>
