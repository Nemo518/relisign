import { GraduationCap, Stethoscope, Home, Cpu, Gauge, Wifi, MapPin } from "lucide-react";
import { cn } from "@/lib/utils";

const scenes = [
  {
    icon: GraduationCap,
    title: "教育场景",
    subtitle: "课堂实时手语翻译",
    items: [
      { text: "课堂实时手语翻译，助听障学生同步理解教学内容", tag: "多模态" },
      { text: "AI 手语教学辅助，提供标准动作参考与实时反馈", tag: "实时推理" },
      { text: "远程教育无障碍字幕生成，覆盖录播与直播场景", tag: "归因分析" },
    ],
    techNote:
      "核心技术挑战：教室多光源环境下姿态估计鲁棒性，多人场景下的目标人物锁定。融合 RGB 与 Pose 双流信息实现抗干扰识别。",
    color: "from-sky-400 to-blue-600",
    glow: "shadow-sky-400/20",
  },
  {
    icon: Stethoscope,
    title: "医疗场景",
    subtitle: "医患无障碍沟通桥梁",
    items: [
      { text: "门诊无障碍沟通，医患实时手语翻译桥梁", tag: "低延迟" },
      { text: "住院护理需求表达，覆盖日常沟通与紧急呼叫", tag: "高可用" },
      { text: "急诊预检分诊引导，缩短听障患者就医等待时间", tag: "实时" },
    ],
    techNote:
      "核心技术挑战：医疗术语的专业翻译准确性，急诊场景下的极低延迟要求（<200ms）。轻量化模型 + ONNX Runtime 加速是关键。",
    color: "from-emerald-400 to-teal-600",
    glow: "shadow-emerald-400/20",
  },
  {
    icon: Home,
    title: "智能家居",
    subtitle: "多设备协同手语识别",
    items: [
      { text: "多设备协同手语识别，电视/音箱/门禁联动", tag: "边缘部署" },
      { text: "边缘端轻量部署，保护用户隐私数据不出户", tag: "隐私保护" },
      { text: "跨终端统一体验，手机/平板/智慧屏多端适配", tag: "跨平台" },
    ],
    techNote:
      "核心技术挑战：边缘设备算力受限（<2GB RAM），模型压缩与量化是关键。三层解耦架构使展示层可独立适配各终端，不依赖云端。",
    color: "from-violet-400 to-purple-600",
    glow: "shadow-violet-400/20",
  },
];

const technicalHighlights = [
  {
    icon: <Cpu size={20} />,
    title: "多模态融合架构",
    desc: "RGB 视觉 + Pose 骨骼双流并行编码，自适应融合模块实现互补增强，消融实验验证融合有效性。",
  },
  {
    icon: <Gauge size={20} />,
    title: "三级性能模式",
    desc: "标准模式 ~500ms / 轻量模式 ~200ms / 边缘模式 ~100ms，在准确率与延迟之间灵活平衡。",
  },
  {
    icon: <Wifi size={20} />,
    title: "端-边-云协同",
    desc: "展示层独立于推理引擎，同一套 API 可接入 GPU 服务器、ONNX Runtime 或边缘 NPU，架构不设限。",
  },
];

export function ScenesPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">应用场景与落地方案</h2>
        <span className="tech-badge rounded-md bg-amber-400/10 border border-amber-400/20 px-2.5 py-1 text-[10px] text-amber-400">
          APPLICATION
        </span>
      </div>

      {/* Main Scene Cards */}
      <div className="grid grid-cols-3 gap-5">
        {scenes.map((scene) => (
          <div
            key={scene.title}
            className="glass-card group rounded-3xl p-6 transition-all duration-300 hover:-translate-y-1 hover:border-cyan-400/30"
          >
            <div
              className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br ${scene.color} text-white shadow-lg ${scene.glow}`}
            >
              <scene.icon size={24} />
            </div>
            <h3 className="text-lg font-bold text-slate-100">{scene.title}</h3>
            <p className="mb-3 text-sm text-cyan-400 font-medium">
              {scene.subtitle}
            </p>
            <ul className="mb-4 space-y-2.5">
              {scene.items.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-sm leading-relaxed">
                  <span className="mt-0.5 shrink-0 rounded bg-slate-700/60 px-1.5 py-0.5 text-[10px] font-semibold text-slate-400 group-hover:text-slate-300 transition-colors">
                    {item.tag}
                  </span>
                  <span className="text-slate-400">{item.text}</span>
                </li>
              ))}
            </ul>
            <div className="rounded-xl bg-amber-400/5 border border-amber-400/15 p-3 text-xs leading-relaxed text-amber-300/80">
              <span className="font-semibold text-amber-400">技术难点：</span>
              {scene.techNote}
            </div>
          </div>
        ))}
      </div>

      {/* Technical Enablers */}
      <div className="glass-card rounded-3xl p-6">
        <h3 className="mb-4 text-base font-bold text-slate-200 flex items-center gap-2">
          <Cpu size={18} className="text-cyan-400" />
          技术支撑：为什么这些场景可行
        </h3>
        <div className="grid grid-cols-3 gap-5">
          {technicalHighlights.map((item) => (
            <div key={item.title} className="flex gap-3">
              <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-teal-500 text-white shadow-md shadow-cyan-400/20">
                {item.icon}
              </div>
              <div>
                <h4 className="text-sm font-bold text-slate-200">{item.title}</h4>
                <p className="mt-1 text-xs leading-relaxed text-slate-400">
                  {item.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Deployment Versions */}
      <div className="glass-card rounded-3xl p-5">
        <h3 className="mb-3 text-base font-bold text-slate-200 flex items-center gap-2">
          <MapPin size={18} className="text-teal-400" />
          部署版本对比
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50 text-slate-400">
                <th className="py-2.5 text-left text-xs font-semibold">版本</th>
                <th className="py-2.5 text-left text-xs font-semibold">设备</th>
                <th className="py-2.5 text-left text-xs font-semibold">延迟</th>
                <th className="py-2.5 text-left text-xs font-semibold">吞吐</th>
                <th className="py-2.5 text-left text-xs font-semibold">适用场景</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  version: "当前版",
                  device: "PC + GPU (PyTorch)",
                  latency: "~500ms",
                  throughput: "20 FPS",
                  scenario: "答辩展示 / 论文实验",
                  active: true,
                },
                {
                  version: "增强版",
                  device: "PC + GPU + ONNX",
                  latency: "~200ms",
                  throughput: "30 FPS",
                  scenario: "教学辅助 / 在线服务",
                  active: false,
                },
                {
                  version: "决赛版",
                  device: "移动端 + 边缘端",
                  latency: "~100ms",
                  throughput: "实时流",
                  scenario: "无障碍沟通 / 智慧家居",
                  active: false,
                },
              ].map((v, i) => (
                <tr
                  key={i}
                  className={cn(
                    "border-b border-slate-700/30 transition-colors hover:bg-slate-800/20",
                    v.active && "bg-cyan-400/5",
                  )}
                >
                  <td className="py-2.5 font-semibold text-slate-200">
                    <div className="flex items-center gap-2">
                      {v.active && <div className="pulse-dot" />}
                      {v.version}
                    </div>
                  </td>
                  <td className="py-2.5 text-slate-300">{v.device}</td>
                  <td className="py-2.5 text-slate-300">{v.latency}</td>
                  <td className="py-2.5 text-slate-300">{v.throughput}</td>
                  <td className="py-2.5 text-slate-400">{v.scenario}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
