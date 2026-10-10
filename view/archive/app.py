from __future__ import annotations

from collections import deque
import html
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from .bootstrap import add_vendor_path
    from .data_service import (
        DEFAULT_CHECKPOINT,
        PROJECT_ROOT,
        VideoRecord,
        build_alignment_dataframe,
        extract_video_frames,
        build_confusion_artifacts,
        build_error_category_frame,
        build_frame_importance,
        build_latency_breakdown,
        build_metric_cards,
        build_model_radar,
        build_pose_density,
        generate_demo_prediction,
        get_video_metadata,
        list_sample_videos,
        load_cached_prediction,
        load_error_examples,
        load_metrics_summary,
        load_pose_payload,
        load_training_histories,
    )
    from .api_client import ViewAPIClient
    from .report_service import build_judge_pdf
except ImportError:  # pragma: no cover - streamlit direct script fallback
    from bootstrap import add_vendor_path
    from data_service import (
        DEFAULT_CHECKPOINT,
        PROJECT_ROOT,
        VideoRecord,
        build_alignment_dataframe,
        extract_video_frames,
        build_confusion_artifacts,
        build_error_category_frame,
        build_frame_importance,
        build_latency_breakdown,
        build_metric_cards,
        build_model_radar,
        build_pose_density,
        generate_demo_prediction,
        get_video_metadata,
        list_sample_videos,
        load_cached_prediction,
        load_error_examples,
        load_metrics_summary,
        load_pose_payload,
        load_training_histories,
    )
    from api_client import ViewAPIClient
    from report_service import build_judge_pdf

add_vendor_path()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Uni-Sign 项目展示台", page_icon="U", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    .main {
        background:
            radial-gradient(circle at 0% 0%, rgba(34, 211, 238, 0.18), transparent 32%),
            radial-gradient(circle at 100% 0%, rgba(96, 165, 250, 0.18), transparent 28%),
            linear-gradient(180deg, #f6fbff 0%, #eef5ff 48%, #f8fbff 100%);
    }
    .hero-shell {
        padding: 1.6rem 1.7rem;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(10, 52, 99, 0.96) 0%, rgba(18, 127, 167, 0.92) 58%, rgba(160, 190, 231, 0.84) 100%);
        color: #ffffff;
        box-shadow: 0 24px 52px rgba(58, 98, 149, 0.18);
        overflow: hidden;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -10% -40% auto;
        width: 380px;
        height: 380px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        filter: blur(4px);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        letter-spacing: 0.02em;
    }
    .hero-subtitle {
        max-width: 980px;
        font-size: 1rem;
        line-height: 1.75;
        color: rgba(255,255,255,0.92);
        margin: 0.35rem 0 0.9rem 0;
    }
    .hero-badges {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 0.86rem;
    }
    .metric-panel {
        border-radius: 22px;
        padding: 1.05rem 1.15rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(246,250,255,0.92));
        border: 1px solid rgba(151, 179, 214, 0.20);
        box-shadow: 0 16px 32px rgba(79, 114, 168, 0.10);
        min-height: 136px;
    }
    .metric-kicker {
        color: #64748b;
        font-size: 0.86rem;
        margin-bottom: 0.35rem;
    }
    .metric-big {
        color: #0f172a;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-note {
        color: #475569;
        font-size: 0.84rem;
        margin-top: 0.45rem;
        line-height: 1.5;
    }
    .pitch-card {
        border-radius: 22px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(243,248,255,0.94));
        border: 1px solid rgba(151, 179, 214, 0.18);
        box-shadow: 0 16px 34px rgba(79, 114, 168, 0.09);
        color: #0f172a;
    }
    .scene-card {
        border-radius: 22px;
        padding: 1.2rem 1.15rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(244,248,255,0.94));
        border: 1px solid rgba(151, 179, 214, 0.16);
        box-shadow: 0 16px 34px rgba(79, 114, 168, 0.08);
        color: #0f172a;
        min-height: 208px;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 12%, rgba(34, 211, 238, 0.12), transparent 26%),
            radial-gradient(circle at 88% 18%, rgba(59, 130, 246, 0.10), transparent 24%),
            linear-gradient(180deg, #f6fbff 0%, #eef5ff 52%, #f8fbff 100%);
    }
    .main .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2.4rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(245, 249, 255, 0.98), rgba(236, 244, 255, 0.98));
        border-right: 1px solid rgba(151, 179, 214, 0.20);
    }
    [data-testid="stSidebar"] * {
        color: #1e293b;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 16px;
        border: 1px solid rgba(122, 160, 212, 0.22);
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(241,247,255,0.96));
        color: #0f172a;
        box-shadow: 0 10px 24px rgba(79, 114, 168, 0.10);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22c1ff, #2dd4bf);
        color: #06203b;
        border: 0;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 16px !important;
        background: rgba(255,255,255,0.92) !important;
        color: #0f172a !important;
        border: 1px solid rgba(151,179,214,0.24) !important;
    }
    .command-shell {
        border-radius: 24px;
        padding: 1.2rem 1.3rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(240,247,255,0.96));
        border: 1px solid rgba(151, 179, 214, 0.20);
        box-shadow: 0 18px 44px rgba(79, 114, 168, 0.10);
    }
    .command-title {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }
    .command-meta {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 0.85rem;
    }
    .command-line {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 84px;
        background: rgba(242,247,255,0.92);
        border: 1px solid rgba(151,179,214,0.18);
        color: #0f172a;
        font-size: 1.05rem;
        line-height: 1.8;
    }
    .flow-shell, .analysis-shell {
        border-radius: 24px;
        padding: 1.05rem 1.15rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(242,247,255,0.94));
        border: 1px solid rgba(151,179,214,0.18);
        box-shadow: 0 14px 30px rgba(79, 114, 168, 0.08);
    }
    .prediction-box {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(248,252,255,0.98), rgba(241,247,255,0.98));
        border: 1px solid rgba(151,179,214,0.18);
    }
    .prediction-title {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .prediction-text {
        font-size: 1.1rem;
        color: #0f172a;
        line-height: 1.8;
        font-weight: 600;
    }
    .token-good {
        display: inline-block;
        margin: 0 0.18rem 0.28rem 0;
        padding: 0.12rem 0.44rem;
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.16);
        color: #0f766e;
        font-weight: 600;
    }
    .token-bad {
        display: inline-block;
        margin: 0 0.18rem 0.28rem 0;
        padding: 0.12rem 0.44rem;
        border-radius: 999px;
        background: rgba(248, 113, 113, 0.16);
        color: #b91c1c;
        font-weight: 600;
    }
    .token-gap {
        display: inline-block;
        margin: 0 0.18rem 0.28rem 0;
        padding: 0.12rem 0.44rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.16);
        color: #475569;
        font-weight: 600;
    }
</style>
"""


JUDGE_BRIEF = {
    "综合评委": "建议叙事顺序：问题背景 → 现场演示 → 数据支撑 → 扩展价值。",
    "技术评委": "建议叙事顺序：真实 XAI → 多模型对比 → 实时推理链路 → 架构边界。",
    "应用评委": "建议叙事顺序：使用场景 → 交互体验 → 稳定性指标 → 社会价值。",
}

SECTION_ORDER = [
    {"key": "live", "label": "实时演示", "seconds": 35, "accent": "#0B3954"},
    {"key": "xai", "label": "可解释 AI", "seconds": 55, "accent": "#087E8B"},
    {"key": "compare", "label": "模型对比", "seconds": 40, "accent": "#C08429"},
    {"key": "arch", "label": "技术架构", "seconds": 35, "accent": "#5C677D"},
    {"key": "scene", "label": "应用场景", "seconds": 30, "accent": "#3C6E71"},
]
SECTION_LOOKUP = {section["key"]: section for section in SECTION_ORDER}
QUOTE_LINES = [
    "我们展示的不是单点识别功能，而是一套可解释、可扩展、可落地的手语翻译技术方案。",
    "展示层独立、推理层服务化、分析层可追踪，这是从比赛作品走向工程系统的关键一步。",
    "评委看到的不只是结果，还能看到模型为何这样判断、在哪些帧最敏感、错误集中在哪里。",
    "最小改动保证了现场稳定，前后端分离保证了后续扩展空间。",
]
DECK_SCRIPTS = {
    "综合评委": {
        "live": ["先展示系统入口和样例视频，立刻把评委注意力拉到“能不能用”。", "点出翻译结果、延迟和吞吐三个指标，快速建立可信度。", "如果评委追问真实性，再切换到真实推理模式。"],
        "xai": ["现在不是只看结果，而是看模型在什么时间、什么部位做出判断。", "用关键帧截图说明模型关注的动作证据，用混淆矩阵解释当前主要错误类型。"],
        "compare": ["这里用对比证明我们不是偶然跑得好，而是稳定优于 baseline。", "重点讲 BLEU、ROUGE 和 WER 的综合平衡。"],
        "arch": ["这一页专门回答“为什么这个系统后续还能继续做大”。", "展示层、推理层、分析层解耦，是工程可持续性的关键。"],
        "scene": ["最后落到真实场景，说明教育、医疗、智能家居三条应用线。", "把技术价值和社会价值合到一起收尾。"],
    },
    "技术评委": {
        "live": ["现场先给出端到端链路，证明系统不是离线拼图。", "再说明 video → pose/RGB → translation 的完整推理路径。"],
        "xai": ["这一页重点讲真实 attention、Grad-CAM 和字符级 confusion 都接了真实接口。", "关键帧、部位贡献和注意力矩阵一起构成可解释闭环。"],
        "compare": ["这里建议直接讲 ablation，对比 pose-only、uncertainty gate 和完整模型。", "强调不是单一指标领先，而是多指标综合更稳。"],
        "arch": ["回答系统边界和模块职责，说明为什么这样拆可以最小改动。", "FastAPI 服务化后，React/WebSocket 只是展示层升级，不影响核心模型。"],
        "scene": ["用场景页承接边缘部署、多端适配和模型压缩。", "这是技术路线的延长线，不是另起炉灶。"],
    },
    "应用评委": {
        "live": ["先让评委看懂界面，再让评委看到结果。", "这一页的重点是低门槛、直观、稳定。"],
        "xai": ["可解释性不是炫技，而是帮助用户和评委理解系统为何可信。", "关键帧截图尤其适合应用向评委理解系统的判断依据。"],
        "compare": ["把复杂的训练实验收束成“效果更好、更稳、更能复现”。", "这页只讲最关键的对比结论，不陷入细节。"],
        "arch": ["架构页用于说明系统未来可以接更多终端和业务，不会推倒重来。", "前后端分离意味着产品化成本更低。"],
        "scene": ["教育、医疗、家居三页就是三条落地路线。", "最后一句落到无障碍沟通和社会价值。"],
    },
}


@st.cache_data(show_spinner=False)
def cached_sample_videos() -> list[VideoRecord]:
    return list_sample_videos()


@st.cache_data(show_spinner=False)
def cached_training_history() -> pd.DataFrame:
    return load_training_histories()


@st.cache_data(show_spinner=False)
def cached_metrics_summary(split: str) -> pd.DataFrame:
    return load_metrics_summary(split)


def _current_section_key() -> str:
    current = st.query_params.get("deck", SECTION_ORDER[0]["key"])
    if isinstance(current, list):
        current = current[-1] if current else SECTION_ORDER[0]["key"]
    current = str(current)
    return current if current in SECTION_LOOKUP else SECTION_ORDER[0]["key"]


def _set_section_key(section_key: str) -> None:
    st.query_params["deck"] = section_key


def _deck_script_lines(judge: str, section_key: str) -> list[str]:
    return DECK_SCRIPTS.get(judge, DECK_SCRIPTS["综合评委"]).get(section_key, [])


def _build_story_scores(cards: dict[str, float | str]) -> list[dict[str, float | str]]:
    bleu = float(cards.get("bleu4", 0.0))
    rouge = float(cards.get("rouge_l", 0.0))
    wer = float(cards.get("wer", 0.0))
    similarity = float(cards.get("average_similarity", 0.0)) * 100.0
    return [
        {"name": "技术深度", "value": round(min(100.0, 42.0 + similarity * 0.48), 1)},
        {"name": "创新表达", "value": round(min(100.0, 38.0 + rouge * 0.68), 1)},
        {"name": "实用表现", "value": round(min(100.0, 44.0 + bleu * 0.86), 1)},
        {"name": "扩展潜力", "value": round(min(100.0, 100.0 - wer * 0.55), 1)},
    ]


def render_echarts_overview(split: str) -> None:
    cards = build_metric_cards(split)
    story_scores = _build_story_scores(cards)
    summary_df = cached_metrics_summary(split).head(4)
    overview_rows = []
    for _, row in summary_df.iterrows():
        overview_rows.append(
            {
                "model": str(row["model_name"]),
                "BLEU-4": round(float(row["bleu4"]), 2),
                "ROUGE-L": round(float(row["rouge_l"]), 2),
                "WER Advantage": round(max(0.0, 100.0 - float(row["wer"])), 2),
            }
        )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=[item["value"] for item in story_scores],
                theta=[item["name"] for item in story_scores],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.18)",
                line=dict(color="#5EEAD4", width=3),
                marker=dict(color="#FBBF24", size=3),
                name="系统画像",
            )
        )
        radar_fig.update_layout(
            title="系统能力雷达",
            height=340,
            margin=dict(l=30, r=30, t=54, b=20),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            polar=dict(
                bgcolor="rgba(255,255,255,0.42)",
                radialaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.20)", linecolor="rgba(148,163,184,0.22)"),
                angularaxis=dict(gridcolor="rgba(148,163,184,0.18)", linecolor="rgba(148,163,184,0.20)"),
            ),
            showlegend=False,
        )
        st.plotly_chart(radar_fig, use_container_width=True)
    with right:
        metrics_df = pd.DataFrame(overview_rows)
        if not metrics_df.empty:
            combo_fig = go.Figure()
            combo_fig.add_trace(
                go.Bar(
                    x=metrics_df["model"],
                    y=metrics_df["BLEU-4"],
                    name="BLEU-4",
                    marker=dict(color="#38BDF8"),
                )
            )
            combo_fig.add_trace(
                go.Bar(
                    x=metrics_df["model"],
                    y=metrics_df["ROUGE-L"],
                    name="ROUGE-L",
                    marker=dict(color="#14B8A6"),
                )
            )
            combo_fig.add_trace(
                go.Scatter(
                    x=metrics_df["model"],
                    y=metrics_df["WER Advantage"],
                    name="WER Advantage",
                    mode="lines+markers",
                    line=dict(color="#FBBF24", width=3),
                    marker=dict(size=8, color="#FBBF24"),
                    yaxis="y2",
                )
            )
            combo_fig.update_layout(
                title="核心指标面板",
                height=340,
                margin=dict(l=20, r=20, t=54, b=20),
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#0f172a"),
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                xaxis=dict(gridcolor="rgba(148,163,184,0.12)", linecolor="rgba(148,163,184,0.22)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.12)", linecolor="rgba(148,163,184,0.22)", title="BLEU / ROUGE"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Advantage"),
            )
            st.plotly_chart(combo_fig, use_container_width=True)


def render_presentation_mode(controls: dict[str, object], cards: dict[str, object], summary_df: pd.DataFrame) -> str:
    current_key = _current_section_key()
    script_lines = _deck_script_lines(str(controls["judge"]), current_key)
    current_index = next(index for index, item in enumerate(SECTION_ORDER) if item["key"] == current_key)
    next_key = SECTION_ORDER[(current_index + 1) % len(SECTION_ORDER)]["key"]
    section_rows = [{"环节": item["label"], "建议时长": f"{item['seconds']} 秒"} for item in SECTION_ORDER]
    confusion_label = summary_df["label"].iloc[0] if not summary_df.empty else ""
    confusion_pairs = build_confusion_artifacts(str(confusion_label))["pair_df"] if confusion_label else pd.DataFrame()
    pdf_bytes = build_judge_pdf(
        controls=controls,
        cards=cards,
        summary_df=summary_df,
        confusion_pairs=confusion_pairs,
        quote_lines=QUOTE_LINES,
        script_lines=script_lines or QUOTE_LINES,
        section_rows=section_rows,
    )

    top_left, top_right = st.columns([1.25, 0.75], gap="large")
    with top_left:
        pills = st.columns(len(SECTION_ORDER))
        for col, section in zip(pills, SECTION_ORDER):
            button_type = "primary" if section["key"] == current_key else "secondary"
            if col.button(section["label"], key=f"deck_{section['key']}", use_container_width=True, type=button_type):
                _set_section_key(section["key"])
                st.rerun()
    with top_right:
        st.download_button(
            "导出评委演示 PDF",
            data=pdf_bytes,
            file_name=f"uni-sign-{controls['judge']}-{controls['split']}-judge-brief.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    html_block = f"""
    <div style="border-radius:24px;padding:18px 20px;background:linear-gradient(135deg,rgba(8,30,52,0.96),rgba(12,89,122,0.92));color:white;box-shadow:0 18px 40px rgba(8,30,52,0.18);">
      <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;">
        <div style="flex:1;">
          <div style="font-size:1.15rem;font-weight:800;margin-bottom:8px;">展示节奏</div>
          <div style="font-size:0.96rem;line-height:1.8;opacity:0.92;">当前页签：{html.escape(SECTION_LOOKUP[current_key]["label"])} · 评委视角：{html.escape(str(controls["judge"]))} · 展示层级：{html.escape(str(controls["stage"]))}</div>
          <div id="deck-script-line" style="margin-top:14px;padding:14px 16px;border-radius:18px;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.14);min-height:72px;font-size:1rem;line-height:1.8;"></div>
        </div>
        <div id="deck-timeline" style="width:340px;height:180px;"></div>
      </div>
      <div style="margin-top:12px;font-size:0.88rem;opacity:0.8;">自动轮播讲解词：{"开启" if controls["auto_script"] else "关闭"} ｜ 自动切换页签：{"开启" if controls["auto_switch"] else "关闭"} ｜ 间隔：{int(controls["interval_s"])} 秒</div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
      const scripts = {json.dumps(script_lines or QUOTE_LINES, ensure_ascii=False)};
      const sections = {json.dumps(SECTION_ORDER, ensure_ascii=False)};
      const currentKey = {json.dumps(current_key, ensure_ascii=False)};
      const nextKey = {json.dumps(next_key, ensure_ascii=False)};
      const autoScript = {str(bool(controls["auto_script"])).lower()};
      const autoSwitch = {str(bool(controls["auto_switch"])).lower()};
      const intervalMs = {int(controls["interval_s"]) * 1000};
      let scriptIndex = 0;

      function setScriptLine() {{
        const line = scripts[scriptIndex % scripts.length];
        $('#deck-script-line').fadeOut(120, function() {{
          $(this).html(line).fadeIn(180);
        }});
        scriptIndex += 1;
      }}

      function goSection(targetKey) {{
        const url = new URL(window.parent.location.href);
        url.searchParams.set('deck', targetKey);
        window.parent.location.href = url.toString();
      }}

      setScriptLine();
      if (autoScript && scripts.length > 1) {{
        setInterval(setScriptLine, 3600);
      }}
      if (autoSwitch) {{
        setTimeout(() => goSection(nextKey), intervalMs);
      }}

      const timeline = echarts.init(document.getElementById('deck-timeline'));
      timeline.setOption({{
        grid: {{ left: 28, right: 18, top: 28, bottom: 32 }},
        xAxis: {{
          type: 'category',
          data: sections.map(item => item.label),
          axisLabel: {{ color: 'rgba(255,255,255,0.82)', rotate: 18 }}
        }},
        yAxis: {{
          type: 'value',
          name: '秒',
          nameTextStyle: {{ color: 'rgba(255,255,255,0.75)' }},
          axisLabel: {{ color: 'rgba(255,255,255,0.78)' }},
          splitLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.12)' }} }}
        }},
        series: [{{
          type: 'bar',
          data: sections.map(item => ({{
            value: item.seconds,
            itemStyle: {{ color: item.key === currentKey ? '#FFB703' : item.accent }}
          }})),
          barWidth: 28,
          label: {{ show: true, position: 'top', color: '#fff' }}
        }}]
      }});
      window.addEventListener('resize', () => timeline.resize());
    </script>
    """
    components.html(html_block, height=240)
    return current_key


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-kicker">{title}</div>
            <div class="metric-big">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="pitch-card">
            <b>{title}</b><br/>{body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_token_html(token: str) -> str:
    if token in {"", " ", "鈭?"}:
        return "空"
    return html.escape(token)


def _build_alignment_summary(reference: str, prediction: str) -> tuple[pd.DataFrame, float]:
    alignment_df = build_alignment_dataframe(reference, prediction)
    if alignment_df.empty:
        return alignment_df, 0.0
    match_ratio = float((alignment_df["operation"] == "match").mean())
    return alignment_df, match_ratio


def render_prediction_panel(reference: str, prediction: str, similarity: float | None = None) -> None:
    alignment_df, match_ratio = _build_alignment_summary(reference, prediction)
    display_score = similarity if similarity is not None else match_ratio
    quality_label = "稳定" if display_score >= 0.75 else "可用" if display_score >= 0.5 else "待优化"

    good_tokens = []
    issue_tokens = []
    for _, row in alignment_df.iterrows():
        ref_char = str(row["reference_char"])
        pred_char = str(row["prediction_char"])
        operation = str(row["operation"])
        if operation == "match":
            good_tokens.append(f'<span class="token-good">{_safe_token_html(pred_char)}</span>')
        elif operation == "insert":
            issue_tokens.append(f'<span class="token-bad">+ {_safe_token_html(pred_char)}</span>')
        elif operation == "delete":
            issue_tokens.append(f'<span class="token-gap">缺 {_safe_token_html(ref_char)}</span>')
        else:
            issue_tokens.append(f'<span class="token-bad">{_safe_token_html(ref_char)} → {_safe_token_html(pred_char)}</span>')

    overview_left, overview_right = st.columns([1.15, 0.85], gap="large")
    with overview_left:
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-title">参考文本</div>
                <div class="prediction-text">{html.escape(reference or '暂无')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="prediction-box" style="margin-top:0.85rem;">
                <div class="prediction-title">预测文本</div>
                <div class="prediction-text">{html.escape(prediction or '暂无')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_right:
        quality_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(display_score * 100, 1),
                number={"suffix": "%"},
                title={"text": f"文本质量 · {quality_label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#22c1ff"},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(248,113,113,0.18)"},
                        {"range": [50, 75], "color": "rgba(250,204,21,0.18)"},
                        {"range": [75, 100], "color": "rgba(45,212,191,0.18)"},
                    ],
                },
            )
        )
        quality_fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=36, b=10),
            paper_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(quality_fig, use_container_width=True)

    chips_left, chips_right = st.columns([1.0, 1.0], gap="large")
    with chips_left:
        st.markdown(
            f"""
            <div class="analysis-shell">
                <div class="prediction-title">表现稳定片段</div>
                <div>{''.join(good_tokens[:18]) if good_tokens else '<span class="token-gap">暂无</span>'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with chips_right:
        st.markdown(
            f"""
            <div class="analysis-shell">
                <div class="prediction-title">需要关注片段</div>
                <div>{''.join(issue_tokens[:18]) if issue_tokens else '<span class="token-good">无明显问题</span>'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not alignment_df.empty:
        op_counts = alignment_df["operation"].value_counts().reset_index()
        op_counts.columns = ["operation", "count"]
        flow_fig = px.funnel(
            op_counts,
            x="count",
            y="operation",
            color="operation",
            color_discrete_map={
                "match": "#22c1ff",
                "replace": "#fb7185",
                "insert": "#f59e0b",
                "delete": "#94a3b8",
            },
            title="文本对齐流转",
        )
        flow_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=48, b=10),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            showlegend=False,
        )
        st.plotly_chart(flow_fig, use_container_width=True)


def speak_text(text: str, *, rate: float = 1.0, lang: str = "zh-CN") -> None:
    safe_text = json.dumps(text or "", ensure_ascii=False)
    safe_lang = json.dumps(lang)
    components.html(
        f"""
        <script>
            const text = {safe_text};
            const lang = {safe_lang};
            const rate = {float(rate):.2f};
            if (window.speechSynthesis) {{
                window.speechSynthesis.cancel();
                if (text.trim()) {{
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = lang;
                    utterance.rate = rate;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    window.speechSynthesis.speak(utterance);
                }}
            }}
        </script>
        """,
        height=0,
    )


def stop_speaking() -> None:
    components.html(
        """
        <script>
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
        </script>
        """,
        height=0,
    )


def render_speech_console(reference: str, prediction: str, *, auto_play: bool = False, trigger_auto_play: bool = False) -> None:
    st.markdown(
        """
        <div class="analysis-shell">
            <div class="prediction-title">语音播报</div>
            <div style="color:#475569;font-size:0.92rem;">支持直接播报当前预测文本，也可以切换为参考文本做现场对照。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    voice_rate = st.slider("播报语速", min_value=0.7, max_value=1.3, value=1.0, step=0.1, key="speech_rate")
    auto_voice = st.toggle("推理完成后自动播报预测文本", value=auto_play, key="speech_auto_play")
    btn_col1, btn_col2, btn_col3 = st.columns([1.0, 1.0, 0.8], gap="small")
    play_prediction = btn_col1.button("播报预测文本", use_container_width=True, key="speech_play_prediction")
    play_reference = btn_col2.button("播报参考文本", use_container_width=True, key="speech_play_reference")
    stop_voice = btn_col3.button("停止播报", use_container_width=True, key="speech_stop")

    if stop_voice:
        stop_speaking()
    elif play_prediction and prediction:
        speak_text(prediction, rate=voice_rate)
    elif play_reference and reference:
        speak_text(reference, rate=voice_rate)
    elif auto_voice and trigger_auto_play and prediction:
        speak_text(prediction, rate=voice_rate)


def render_quote_panel() -> None:
    return


def capture_webcam_clip(
    *,
    duration_s: int,
    camera_index: int,
    target_fps: int = 20,
) -> Path | None:
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    backend_flag = getattr(cv2, "CAP_DSHOW", 0)
    cap = cv2.VideoCapture(camera_index, backend_flag)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = float(target_fps if target_fps > 0 else (source_fps if source_fps > 0 else 20.0))

    capture_dir = Path(tempfile.gettempdir()) / "uni_sign_camera_captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    output_path = capture_dir / f"camera_capture_{int(time.time() * 1000)}.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    preview_slot = st.empty()
    progress_slot = st.empty()
    progress_bar = st.progress(0)
    frame_interval = 1.0 / max(fps, 1.0)
    deadline = time.perf_counter() + float(duration_s)
    next_frame_time = time.perf_counter()
    captured_any = False

    try:
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            captured_any = True
            writer.write(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state["live_camera_preview_frame"] = rgb_frame
            preview_slot.image(rgb_frame, channels="RGB", use_container_width=True, caption="摄像头实时画面")
            remaining = max(deadline - time.perf_counter(), 0.0)
            elapsed_ratio = min(1.0, 1.0 - remaining / max(float(duration_s), 1.0))
            progress_slot.markdown(f"正在采集摄像头片段：剩余 {remaining:.1f} 秒")
            progress_bar.progress(int(elapsed_ratio * 100))
            next_frame_time += frame_interval
            sleep_for = next_frame_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()
        writer.release()
        progress_bar.empty()
        progress_slot.empty()

    if not captured_any or not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        preview_slot.empty()
        return None

    return output_path


def write_camera_window_clip(frames: list, fps: float, prefix: str = "camera_window") -> Path | None:
    if not frames:
        return None
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    height, width = frames[0].shape[:2]
    capture_dir = Path(tempfile.gettempdir()) / "uni_sign_camera_captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    output_path = capture_dir / f"{prefix}_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return None
    return output_path


def run_camera_rolling_analysis(
    *,
    api_client: ViewAPIClient,
    checkpoint_path: Path,
    rgb_support: bool,
    lightweight: bool,
    ultra_lightweight: bool,
    camera_index: int,
    capture_fps: int,
    window_seconds: int,
    step_seconds: int,
    total_seconds: int,
) -> dict[str, object] | None:
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    backend_flag = getattr(cv2, "CAP_DSHOW", 0)
    cap = cv2.VideoCapture(camera_index, backend_flag)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None

    target_fps = max(int(capture_fps), 1)
    frame_interval = 1.0 / float(target_fps)
    max_buffer_frames = max(int(window_seconds * target_fps), target_fps)
    min_ready_frames = max(int(max_buffer_frames * 0.6), 8)
    frame_buffer = deque(maxlen=max_buffer_frames)
    latest_payload: dict[str, object] | None = None
    subtitle_segments = deque(maxlen=3)

    status_slot = st.empty()
    preview_slot = st.empty()
    subtitle_slot = st.empty()
    prediction_slot = st.empty()
    progress_bar = st.progress(0)

    start_time = time.perf_counter()
    next_capture_time = start_time
    next_window_time = start_time + float(window_seconds)
    window_index = 0

    try:
        while time.perf_counter() - start_time < float(total_seconds):
            current_time = time.perf_counter()
            if current_time < next_capture_time:
                time.sleep(min(next_capture_time - current_time, 0.01))
                continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            elapsed = time.perf_counter() - start_time
            frame_buffer.append(frame.copy())
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state["live_camera_preview_frame"] = rgb_frame
            preview_slot.image(rgb_frame, channels="RGB", use_container_width=True, caption="滚动窗口实时画面")
            progress_bar.progress(min(int(elapsed / max(float(total_seconds), 1.0) * 100), 100))
            status_slot.markdown(f"**滚动窗口识别中**  已运行 {elapsed:.1f}s / {total_seconds}s")
            next_capture_time += frame_interval

            if elapsed >= (next_window_time - start_time) and len(frame_buffer) >= min_ready_frames:
                window_index += 1
                clip_path = write_camera_window_clip(list(frame_buffer), target_fps, prefix=f"camera_window_{window_index}")
                result = None
                if clip_path is not None:
                    try:
                        result = api_client.predict(
                            video_path=clip_path,
                            checkpoint_path=checkpoint_path,
                            rgb_support=rgb_support,
                            lightweight=lightweight,
                            ultra_lightweight=ultra_lightweight,
                        )
                    except Exception:
                        result = None

                prediction = str(result["prediction"]) if result is not None and result.get("success") else "当前窗口未返回稳定结果"
                latency_s = float(result["latency_s"]) if result is not None and result.get("success") else 0.0
                effective_fps = float(result["effective_fps"]) if result is not None and result.get("success") else float(target_fps)
                window_end = float(elapsed)
                window_start = max(0.0, window_end - float(window_seconds))
                is_stable_prediction = bool(result is not None and result.get("success") and prediction and "未返回稳定结果" not in prediction)
                if is_stable_prediction:
                    if not subtitle_segments or subtitle_segments[-1] != prediction:
                        subtitle_segments.append(prediction)
                subtitle_text = "  |  ".join(subtitle_segments) if subtitle_segments else "等待最新识别字幕..."
                latest_payload = {
                    "success": bool(result is not None and result.get("success")),
                    "prediction": prediction,
                    "latency_s": latency_s,
                    "effective_fps": effective_fps,
                    "video_path": clip_path,
                    "subtitle_text": subtitle_text,
                    "window_label": f"窗口 {window_index} · {window_start:.1f}s - {window_end:.1f}s",
                }

                subtitle_slot.markdown(
                    f"""
                    <div class="analysis-shell">
                        <div class="prediction-title">连续字幕条</div>
                        <div class="prediction-text" style="font-size:1.28rem;line-height:1.8;">{html.escape(subtitle_text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                prediction_slot.markdown(
                    f"""
                    <div class="analysis-shell">
                        <div class="prediction-title">当前窗口结果</div>
                        <div style="color:#64748b;font-size:0.92rem;margin-bottom:0.4rem;">窗口 {window_index} · {window_start:.1f}s - {window_end:.1f}s</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:0.65rem;">{html.escape(prediction)}</div>
                        <div style="color:#475569;font-size:0.92rem;">延迟 {latency_s * 1000:.0f} ms · 有效吞吐 {effective_fps:.1f} FPS</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                next_window_time = max(
                    next_window_time + float(step_seconds),
                    time.perf_counter() + float(step_seconds),
                )
    finally:
        cap.release()
        progress_bar.empty()

    return latest_payload


def render_keyframe_gallery(video_path: Path, frame_df: pd.DataFrame, score_column: str = "xai_score") -> None:
    if frame_df.empty:
        return
    try:
        frames = extract_video_frames(video_path, frame_df["frame"].tolist())
    except Exception as exc:
        return

    if not frames:
        return

    frame_lookup = {item["frame"]: item["image"] for item in frames}
    gallery_cols = st.columns(min(3, len(frame_df)))
    for idx, (_, row) in enumerate(frame_df.iterrows()):
        col = gallery_cols[idx % len(gallery_cols)]
        image = frame_lookup.get(int(row["frame"]))
        if image is None:
            continue
        caption = f"Frame {int(row['frame'])} | {score_column}={float(row.get(score_column, 0.0)):.3f}"
        col.image(image, caption=caption, use_container_width=True)


def sidebar_controls() -> dict[str, str]:
    st.sidebar.title("项目展示台")
    stage = st.sidebar.selectbox("展示层级", ["Level 1 快速体验", "Level 2 核心功能", "Level 3 技术深度", "Level 4 创新扩展"], index=2)
    judge = st.sidebar.selectbox("评委视角", ["综合评委", "技术评委", "应用评委"], index=0)
    split = st.sidebar.selectbox("主对比数据集", ["test", "dev"], index=0)
    api_base = st.sidebar.text_input("后端 API", value=os.environ.get("UNISIGN_VIEW_API", "http://127.0.0.1:8000"))
    return {
        "stage": stage,
        "judge": judge,
        "split": split,
        "api_base": api_base,
    }


def show_hero(controls: dict[str, str]) -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-title">Uni-Sign 项目展示台</div>
            <div class="hero-subtitle">
                面向项目演示的独立展示舱，聚焦实时翻译、可解释分析与多模型对比，适合直接录制视频与现场展示。
            </div>
            <div class="hero-badges">
                <span class="hero-badge">展示层级：{controls["stage"]}</span>
                <span class="hero-badge">评委视角：{controls["judge"]}</span>
                <span class="hero-badge">主数据集：{controls["split"].upper()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_status(api_client: ViewAPIClient) -> None:
    return


def render_overview(strip: str) -> None:
    cards = build_metric_cards(strip)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("主展示模型", cards.get("model_name", "N/A"), "从现有离线汇总自动抽取，展示时可稳定复现。")
    with col2:
        render_metric_card("BLEU-4", f"{cards.get('bleu4', 0.0):.2f}", f"{strip.upper()} 集离线汇总指标。")
    with col3:
        render_metric_card("ROUGE-L", f"{cards.get('rouge_l', 0.0):.2f}", "语义一致性展示首选指标。")
    with col4:
        render_metric_card("WER", f"{cards.get('wer', 0.0):.2f}", "越低越好，适合说明错误成本。")


def select_video_source(key_prefix: str) -> tuple[Path | None, VideoRecord | None]:
    mode = st.radio("输入来源", ["样例视频", "本地上传"], horizontal=True, key=f"{key_prefix}_mode")
    sample_record = None
    selected_path = None
    if mode == "样例视频":
        videos = cached_sample_videos()
        if not videos:
            st.info("当前没有可用的样例视频，请检查外部数据仓中的 CE-CSL 视频是否已经就位。")
            return None, None
        selected_name = st.selectbox("选择样例", [record.name for record in videos], key=f"{key_prefix}_sample")
        sample_record = next(record for record in videos if record.name == selected_name)
        selected_path = sample_record.video_path
    else:
        upload = st.file_uploader("上传演示视频", type=["mp4", "avi", "mov", "mkv"], key=f"{key_prefix}_upload")
        if upload is not None:
            suffix = Path(upload.name).suffix or ".mp4"
            safe_name = Path(upload.name).stem.replace(" ", "_")
            temp_dir = Path(tempfile.gettempdir()) / "uni_sign_view_uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            selected_path = temp_dir / f"{safe_name}{suffix}"
            selected_path.write_bytes(upload.getbuffer())
    return selected_path, sample_record


def render_live_demo(split: str, api_client: ViewAPIClient) -> None:
    st.subheader("实时演示")
    left, right = st.columns([1.15, 1.0], gap="large")
    selected_path = None
    sample_record = None
    input_mode = "样例视频"
    camera_mode = "单段识别"
    realtime_profile = "标准"
    run_button = False
    capture_context = ""
    effective_real_inference = False
    with left:
        input_mode = st.radio("输入来源", ["样例视频", "本地上传", "摄像头采集"], horizontal=True, key="live_mode")
        if input_mode == "样例视频":
            videos = cached_sample_videos()
            if videos:
                selected_name = st.selectbox("选择样例", [record.name for record in videos], key="live_sample")
                sample_record = next(record for record in videos if record.name == selected_name)
                selected_path = sample_record.video_path
            else:
                st.info("当前没有可用的样例视频。")
        elif input_mode == "本地上传":
            upload = st.file_uploader("上传演示视频", type=["mp4", "avi", "mov", "mkv"], key="live_upload")
            if upload is not None:
                suffix = Path(upload.name).suffix or ".mp4"
                safe_name = Path(upload.name).stem.replace(" ", "_")
                temp_dir = Path(tempfile.gettempdir()) / "uni_sign_view_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                selected_path = temp_dir / f"{safe_name}{suffix}"
                selected_path.write_bytes(upload.getbuffer())
        else:
            camera_mode = st.radio("识别方式", ["单段识别", "滚动窗口识别"], horizontal=True, key="live_camera_mode")
            camera_index = int(st.number_input("摄像头编号", min_value=0, max_value=4, value=0, step=1, key="live_camera_index"))
            realtime_profile = st.radio("实时模式", ["标准", "轻量", "超轻量"], index=2, horizontal=True, key="live_realtime_profile")
            realtime_lightweight = realtime_profile in {"轻量", "超轻量"}
            ultra_lightweight = realtime_profile == "超轻量"
            default_camera_fps = 10 if ultra_lightweight else 12 if realtime_lightweight else 20
            capture_fps = int(st.slider("采集帧率", min_value=8, max_value=30, value=default_camera_fps, step=1, key="live_camera_fps"))
            previous_clip = st.session_state.get("live_camera_clip")
            if previous_clip and Path(previous_clip).exists():
                selected_path = Path(previous_clip)
            if camera_mode == "滚动窗口识别":
                default_total = 8 if ultra_lightweight else 12 if realtime_lightweight else 18
                default_window = 2 if realtime_lightweight else 3
                default_step = 2 if realtime_lightweight else 2
                stream_total_seconds = int(st.slider("持续识别总时长（秒）", min_value=8, max_value=60, value=default_total, step=2, key="live_stream_total_seconds"))
                window_seconds = int(st.slider("窗口长度（秒）", min_value=2, max_value=6, value=default_window, step=1, key="live_window_seconds"))
                step_seconds = int(st.slider("滑动步长（秒）", min_value=1, max_value=4, value=default_step, step=1, key="live_step_seconds"))
                capture_context = f"摄像头 {camera_index} · 持续 {stream_total_seconds} 秒 · 窗口 {window_seconds} 秒 · 步长 {step_seconds} 秒 · {capture_fps} FPS · {realtime_profile}"
            else:
                default_capture = 3 if ultra_lightweight else 4 if realtime_lightweight else 5
                capture_seconds = int(st.slider("采集时长（秒）", min_value=2, max_value=10, value=default_capture, step=1, key="live_camera_seconds"))
                capture_context = f"摄像头 {camera_index} · {capture_seconds} 秒 · {capture_fps} FPS · {realtime_profile}"
            st.markdown(
                f"""
                <div class="pitch-card">
                    <b>摄像头实时采集</b><br/>
                    {("系统会持续采集摄像头画面，并按时间窗口滚动调用模型完成手语识别。" if camera_mode == "滚动窗口识别" else "点击下方按钮后，系统会先采集一段实时视频，再立即调用模型完成手语识别。")}<br/>
                    当前设置：{capture_context}
                </div>
                """,
                unsafe_allow_html=True,
            )
        if input_mode != "摄像头采集":
            realtime_lightweight = False
            ultra_lightweight = False

        if input_mode == "摄像头采集":
            use_real_inference = True
            st.caption("摄像头模式默认直接调用真实模型推理。")
        else:
            use_real_inference = st.toggle("调用真实模型推理", value=False, key="live_real")
        rgb_default = False if input_mode == "摄像头采集" and realtime_lightweight else True
        rgb_help = "轻量/超轻量实时模式下建议关闭，可明显减轻延迟。"
        rgb_widget = st.checkbox("启用 RGB-support", value=rgb_default, help=rgb_help, disabled=ultra_lightweight, key="live_rgb")
        rgb_support = rgb_widget and not ultra_lightweight
        checkpoint_path = Path(DEFAULT_CHECKPOINT)
        with st.expander("高级设置", expanded=False):
            checkpoint_path = Path(st.text_input("推理检查点", value=str(DEFAULT_CHECKPOINT), key="live_ckpt"))
        run_button = st.button("开始实时演示", type="primary", use_container_width=True, key="live_run")

        if input_mode == "摄像头采集" and camera_mode == "单段识别" and run_button:
            with st.spinner("正在采集摄像头视频..."):
                captured_path = capture_webcam_clip(
                    duration_s=capture_seconds,
                    camera_index=camera_index,
                    target_fps=capture_fps,
                )
            if captured_path is not None:
                st.session_state["live_camera_clip"] = str(captured_path)
                selected_path = captured_path
            else:
                st.error("未能从摄像头采集到可用视频，请检查摄像头是否被其他程序占用。")
                selected_path = None

        effective_real_inference = use_real_inference

        preview_frame = st.session_state.get("live_camera_preview_frame")
        if input_mode == "摄像头采集" and preview_frame is not None:
            st.image(preview_frame, channels="RGB", use_container_width=True, caption="摄像头实时画面")
        elif selected_path is not None and selected_path.exists():
            st.video(str(selected_path))

        if sample_record is not None:
            st.markdown(
                f"""
                <div class="pitch-card">
                    <b>样例锚点</b><br/>
                    样例编号：{sample_record.sample_id}<br/>
                    参考文本：{sample_record.reference or "暂无"}<br/>
                    Gloss：{sample_record.gloss or "暂无"}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        rolling_payload = None
        if input_mode == "摄像头采集" and camera_mode == "滚动窗口识别" and run_button:
            rolling_payload = run_camera_rolling_analysis(
                api_client=api_client,
                checkpoint_path=checkpoint_path,
                rgb_support=rgb_support,
                lightweight=realtime_lightweight,
                ultra_lightweight=ultra_lightweight,
                camera_index=camera_index,
                capture_fps=capture_fps,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                total_seconds=stream_total_seconds,
            )
            if rolling_payload is not None:
                selected_path = Path(rolling_payload["video_path"]) if rolling_payload.get("video_path") else None

        if (run_button and selected_path is not None and selected_path.exists()) or rolling_payload is not None:
            cached_result = load_cached_prediction(sample_record.sample_id) if sample_record is not None else None
            result = rolling_payload
            similarity = float(cached_result["similarity"]) if cached_result is not None and "similarity" in cached_result else None
            if effective_real_inference and rolling_payload is None:
                with st.spinner("正在执行真实推理..."):
                    try:
                        result = api_client.predict(
                            video_path=selected_path,
                            checkpoint_path=checkpoint_path,
                            rgb_support=rgb_support,
                            lightweight=realtime_lightweight,
                            ultra_lightweight=ultra_lightweight,
                        )
                    except Exception:
                        result = None

            try:
                meta = get_video_metadata(selected_path)
            except Exception:
                meta = SimpleNamespace(frame_count=0, fps=0.0, width=0, height=0, duration_s=0.0)
            if rolling_payload is not None:
                prediction = str(rolling_payload["prediction"])
                mode_tag = "摄像头滚动窗口识别"
                latency_s = float(rolling_payload["latency_s"])
                effective_fps = float(rolling_payload["effective_fps"])
            elif result is not None and result.get("success"):
                prediction = result["prediction"]
                mode_tag = "摄像头实时识别" if input_mode == "摄像头采集" else "实时推理"
                latency_s = float(result["latency_s"])
                effective_fps = float(result["effective_fps"])
            elif cached_result is not None:
                prediction = str(cached_result["prediction"])
                mode_tag = f"离线复现 · {cached_result['model_name']}"
                latency_s = meta.duration_s / max(meta.fps, 1.0)
                effective_fps = meta.fps
            elif input_mode == "摄像头采集":
                prediction = "实时识别暂未返回结果"
                mode_tag = "摄像头采集 · 待重试"
                latency_s = meta.duration_s / max(meta.fps, 1.0)
                effective_fps = meta.fps
            else:
                prediction = generate_demo_prediction(sample_record.sample_id if sample_record else "upload", sample_record.reference if sample_record else None)
                mode_tag = "摄像头演示模式" if input_mode == "摄像头采集" else "演示模式"
                latency_s = meta.duration_s / max(meta.fps, 1.0)
                effective_fps = meta.fps

            reference_text = sample_record.reference if sample_record and sample_record.reference else ""
            st.markdown(
                f"""
                <div class="flow-shell">
                    <div class="prediction-title">当前链路</div>
                    <div class="prediction-text">{html.escape(mode_tag + (' · ' + capture_context if capture_context else ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("端到端延迟", f"{latency_s * 1000:.0f} ms", "整体推理耗时")
            with col2:
                render_metric_card("视频帧率", f"{meta.fps:.1f} FPS", "输入视频信息")
            with col3:
                render_metric_card("有效吞吐", f"{effective_fps:.1f} FPS", "折算后的处理速度")
            with col4:
                render_metric_card("分辨率", f"{meta.width}×{meta.height}", f"共 {meta.frame_count} 帧")

            if reference_text or prediction:
                render_prediction_panel(reference_text, prediction, similarity=similarity)
                render_speech_console(
                    reference_text,
                    prediction,
                    auto_play=True,
                    trigger_auto_play=bool(run_button),
                )

            latency_df = build_latency_breakdown(latency_s)
            latency_fig = px.bar(
                latency_df,
                x="duration_ms",
                y="stage",
                orientation="h",
                color="stage",
                title="演示链路延迟拆解",
                labels={"duration_ms": "耗时 (ms)", "stage": ""},
            )
            latency_fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            st.plotly_chart(latency_fig, use_container_width=True)

            if sample_record is not None and sample_record.pose_path is not None and sample_record.pose_path.exists():
                pose_payload = load_pose_payload(sample_record.pose_path)
                importance_df = build_frame_importance(pose_payload)
                density_df = build_pose_density(pose_payload)
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    if not importance_df.empty:
                        importance_fig = px.line(
                            importance_df,
                            x="frame",
                            y=["importance", "mean_score", "motion_norm"],
                            title="姿态代理可解释性：关键帧 / 动作波动",
                        )
                        importance_fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(importance_fig, use_container_width=True)
                with chart_col2:
                    if not density_df.empty:
                        density_fig = px.density_heatmap(
                            density_df,
                            x="x_bin",
                            y="y_bin",
                            z="density",
                            histfunc="avg",
                            title="骨架热区分布",
                            color_continuous_scale="YlGnBu",
                        )
                        density_fig.update_yaxes(autorange="reversed")
                        density_fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(density_fig, use_container_width=True)


def render_xai_lab(split: str, api_client: ViewAPIClient) -> None:
    st.subheader("真实可解释 AI 实验台")
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        selected_path, sample_record = select_video_source("xai")
        rgb_support = st.checkbox("XAI 使用 RGB-support", value=True, key="xai_rgb")
        checkpoint_path = Path(DEFAULT_CHECKPOINT)
        with st.expander("XAI 高级设置", expanded=False):
            checkpoint_path = Path(st.text_input("XAI 检查点", value=str(DEFAULT_CHECKPOINT), key="xai_ckpt"))
        selected_device = st.selectbox("XAI 设备", ["auto", "cuda", "cpu"], index=0, key="xai_device")
        summary_df = cached_metrics_summary(split)
        label_options = summary_df["label"].tolist() if "label" in summary_df.columns else []
        if label_options:
            selected_model_label = st.selectbox(
                "混淆分析模型",
                label_options,
                format_func=lambda label: label.replace("_", " "),
                key="xai_label",
            )
        else:
            selected_model_label = ""
        run_xai = st.button("执行真实 XAI 分析", type="primary", use_container_width=True, key="xai_run")

    with right:
        if run_xai and selected_path is not None and selected_path.exists():
            pose_payload = None
            if sample_record is not None and sample_record.pose_path is not None and sample_record.pose_path.exists():
                pose_payload = load_pose_payload(sample_record.pose_path)

            try:
                with st.spinner("正在计算真实 attention / Grad-CAM / confusion..."):
                    xai_result = api_client.xai(
                        video_path=selected_path,
                        checkpoint_path=checkpoint_path,
                        rgb_support=rgb_support,
                        reference_text=sample_record.reference if sample_record else None,
                        pose_path=sample_record.pose_path if sample_record and sample_record.pose_path else None,
                        device=None if selected_device == "auto" else selected_device,
                    )

                render_story_card(
                    "XAI 结果摘要",
                    f"目标文本：{xai_result['target_sentence']}<br/>生成文本：{xai_result['prediction'] or '暂无'}<br/>反向分析损失：{xai_result['loss']:.4f}",
                )

                summary_cards = st.columns(3)
                with summary_cards[0]:
                    render_metric_card("关键帧峰值", f"{float(xai_result['top_frames_df']['xai_score'].max()):.3f}", "注意力与 Grad-CAM 融合后最强响应。")
                with summary_cards[1]:
                    dominant_part = str(xai_result["part_summary_df"].iloc[0]["part"]) if not xai_result["part_summary_df"].empty else "N/A"
                    render_metric_card("主导部位", dominant_part, "峰值贡献最大的身体区域。")
                with summary_cards[2]:
                    mismatch_count = int((xai_result["alignment_df"]["operation"] != "match").sum()) if not xai_result["alignment_df"].empty else 0
                    render_metric_card("文本差异位", str(mismatch_count), "参考文本与预测文本的差异位置数。")

                timeline_df = xai_result["xai_timeline_df"]
                overview_fig = px.line(
                    timeline_df,
                    x="frame",
                    y=["xai_score", "encoder_attention", "gradcam_combined", "body", "left", "right", "face_all"],
                    title="真实时序解释：Attention 与 Grad-CAM 叠合",
                )
                overview_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(overview_fig, use_container_width=True)

                st.markdown("**最关键的 6 帧**")
                render_keyframe_gallery(selected_path, xai_result["top_frames_df"], score_column="xai_score")

                part_summary_df = xai_result["part_summary_df"]
                if not part_summary_df.empty:
                    part_fig = px.bar(
                        part_summary_df,
                        x="part",
                        y=["mean_importance", "peak_importance"],
                        barmode="group",
                        title="身体部位贡献概览",
                        labels={"value": "重要性", "part": "部位"},
                    )
                    part_fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(part_fig, use_container_width=True)

                matrix_df = xai_result["attention_matrix_df"]
                if not matrix_df.empty:
                    pivot = matrix_df.pivot(index="source_frame", columns="target_frame", values="attention").sort_index(ascending=True)
                    matrix_fig = px.imshow(
                        pivot.values,
                        color_continuous_scale="Blues",
                        title="MT5 编码器真实注意力矩阵",
                        labels={"x": "目标帧", "y": "源帧", "color": "Attention"},
                    )
                    matrix_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(matrix_fig, use_container_width=True)

                rgb_cross_df = xai_result["rgb_cross_df"]
                if not rgb_cross_df.empty:
                    rgb_fig = px.density_heatmap(
                        rgb_cross_df,
                        x="patch_x",
                        y="patch_y",
                        z="attention",
                        facet_col="branch",
                        histfunc="avg",
                        nbinsx=4,
                        nbinsy=4,
                        color_continuous_scale="Sunsetdark",
                        title="Deformable RGB Cross-Attention 热区",
                    )
                    rgb_fig.update_yaxes(autorange="reversed")
                    rgb_fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(rgb_fig, use_container_width=True)

                alignment_df = xai_result["alignment_df"]
                if not alignment_df.empty:
                    st.markdown("**预测差异对齐表**")
                    st.dataframe(alignment_df, use_container_width=True, hide_index=True)
            except Exception:
                pass

        confusion = build_confusion_artifacts(selected_model_label) if selected_model_label else {
            "pair_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count", "operation"]),
            "matrix_df": pd.DataFrame(columns=["reference_char", "prediction_char", "count"]),
            "summary_df": pd.DataFrame(columns=["operation", "count"]),
        }
        summary_col1, summary_col2 = st.columns([0.7, 1.3], gap="large")
        with summary_col1:
            st.markdown("**混淆操作分布**")
            if not confusion["summary_df"].empty:
                op_fig = px.pie(confusion["summary_df"], names="operation", values="count", hole=0.46, title="插入 / 删除 / 替换")
                op_fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(op_fig, use_container_width=True)
        with summary_col2:
            st.markdown("**高频混淆对**")
            if not confusion["pair_df"].empty:
                st.dataframe(confusion["pair_df"], use_container_width=True, hide_index=True)

        if not confusion["matrix_df"].empty:
            heatmap = confusion["matrix_df"].pivot(index="reference_char", columns="prediction_char", values="count").fillna(0)
            confusion_fig = px.imshow(
                heatmap.values,
                x=list(heatmap.columns),
                y=list(heatmap.index),
                text_auto=True,
                color_continuous_scale="YlOrRd",
                title="字符级替换混淆矩阵",
                labels={"x": "预测字符", "y": "参考字符", "color": "频次"},
            )
            confusion_fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(confusion_fig, use_container_width=True)


def render_model_compare(split: str) -> None:
    st.subheader("模型对比与错误切片")
    summary_df = cached_metrics_summary(split)
    history_df = cached_training_history()
    if summary_df.empty:
        return
    top_left, top_right = st.columns([1.0, 1.0], gap="large")
    with top_left:
        metrics_bar = summary_df.melt(
            id_vars=["model_name", "label"],
            value_vars=["bleu4", "rouge_l", "wer"],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            metrics_bar,
            x="model_name",
            y="value",
            color="metric",
            barmode="group",
            title=f"{split.upper()} 集核心指标对比",
            labels={"model_name": "模型", "value": "数值", "metric": "指标"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with top_right:
        radar_df = build_model_radar(summary_df)
        radar_fig = px.line_polar(radar_df, r="value", theta="metric", color="model_name", line_close=True, title="多维能力雷达图")
        radar_fig.update_traces(fill="toself")
        radar_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(radar_fig, use_container_width=True)

    if not history_df.empty:
        selected_history = st.multiselect(
            "训练曲线实验",
            sorted(history_df["experiment"].unique()),
            default=[name for name in sorted(history_df["experiment"].unique()) if "round7" in name or "ablation" in name][:4],
        )
        history_view = history_df[history_df["experiment"].isin(selected_history)] if selected_history else history_df
        if not history_view.empty:
            curve_fig = px.line(
                history_view,
                x="epoch",
                y="selection_average_similarity",
                color="experiment",
                title="训练中平均相似度变化",
                markers=True,
            )
            curve_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(curve_fig, use_container_width=True)

    selected_label = st.selectbox("错误切片模型", summary_df["label"].tolist(), key="compare_label")
    error_df = build_error_category_frame(selected_label)
    examples = load_error_examples(selected_label)
    bottom_left, bottom_right = st.columns([0.9, 1.1], gap="large")
    with bottom_left:
        if not error_df.empty:
            error_fig = px.bar(
                error_df.sort_values("avg_similarity", ascending=False),
                x="category",
                y="avg_similarity",
                color="count",
                title="句型层面的误差切片",
            )
            error_fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(error_fig, use_container_width=True)
    with bottom_right:
        st.markdown("**最佳案例 Top-5**")
        st.dataframe(pd.DataFrame(examples.get("best_examples", [])), use_container_width=True, hide_index=True)
        st.markdown("**最难案例 Top-5**")
        st.dataframe(pd.DataFrame(examples.get("worst_examples", [])), use_container_width=True, hide_index=True)


def render_architecture() -> None:
    st.subheader("技术架构与展示逻辑")
    sankey = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=22,
                    thickness=18,
                    line=dict(color="rgba(8,30,52,0.2)", width=0.5),
                    label=["演示视图 view/", "在线推理", "XAI 分析", "离线指标", "主模型 Uni-Sign", "out/ 结果库", "data/ 视频与 pose"],
                    color=["#0B3954", "#087E8B", "#C08429", "#5C677D", "#0F4C5C", "#BFD7EA", "#D9E2EC"],
                ),
                link=dict(
                    source=[0, 0, 0, 1, 2, 3, 4, 4],
                    target=[1, 2, 3, 4, 4, 5, 5, 6],
                    value=[4, 4, 3, 4, 4, 3, 5, 5],
                ),
            )
        ]
    )
    sankey.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.plotly_chart(sankey, use_container_width=True)
    with right:
        roadmap_df = pd.DataFrame(
            [
                {"阶段": "当前展示版", "落地点": "view/ 独立展示模块", "价值": "最小改动、快速可讲"},
                {"阶段": "下一阶段", "落地点": "FastAPI 封装真实推理 / XAI", "价值": "前后端边界更清晰"},
                {"阶段": "增强阶段", "落地点": "React / WebSocket / 多端联动", "价值": "更强现场观感"},
            ]
        )
        st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class="pitch-card">
                <b>核心讲述逻辑</b><br/>
                我们没有把演示代码侵入训练主链路，而是把“展示层、分析层、推理层”做了软连接式解耦。
                这样既保证现场稳定，也给后续工程化留下了完整接口。
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scenes() -> None:
    st.subheader("应用场景与未来扩展")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown(
            """
            <div class="scene-card">
                <b>教育场景</b><br/><br/>
                课堂实时翻译、手语教学辅助、AI 教练模式。展示时可以演示“识别结果 + 关键帧解释 + 纠错建议”的闭环。
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="scene-card">
                <b>医疗场景</b><br/><br/>
                门诊问询、住院沟通、急诊指引。重点突出无障碍沟通的社会价值和真实需求强度。
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="scene-card">
                <b>智能家居</b><br/><br/>
                多设备联动、边缘部署、跨终端适配。适合引出模型压缩、量化和端侧部署的后续工作。
            </div>
            """,
            unsafe_allow_html=True,
        )

    deploy_df = pd.DataFrame(
        [
            {"版本": "当前版", "方案": "Streamlit 独立展示台", "优势": "快、稳、最小改动"},
            {"版本": "增强版", "方案": "Streamlit + FastAPI", "优势": "接口清晰，可服务化"},
            {"版本": "决赛版", "方案": "React + FastAPI + WebSocket", "优势": "视觉更强，支持多设备演示"},
        ]
    )
    st.dataframe(deploy_df, use_container_width=True, hide_index=True)


def render_stage_console(controls: dict[str, object], cards: dict[str, object], summary_df: pd.DataFrame) -> str:
    current_key = _current_section_key()
    script_lines = _deck_script_lines(str(controls["judge"]), current_key)
    section_rows = [{"环节": item["label"], "建议时长": item["seconds"]} for item in SECTION_ORDER]
    confusion_label = summary_df["label"].iloc[0] if not summary_df.empty else ""
    confusion_pairs = build_confusion_artifacts(str(confusion_label))["pair_df"] if confusion_label else pd.DataFrame()
    pdf_bytes = build_judge_pdf(
        controls=controls,
        cards=cards,
        summary_df=summary_df,
        confusion_pairs=confusion_pairs,
        quote_lines=QUOTE_LINES,
        script_lines=script_lines or QUOTE_LINES,
        section_rows=section_rows,
    )

    top_left, top_right = st.columns([1.25, 0.75], gap="large")
    with top_left:
        pills = st.columns(len(SECTION_ORDER))
        for col, section in zip(pills, SECTION_ORDER):
            button_type = "primary" if section["key"] == current_key else "secondary"
            if col.button(section["label"], key=f"safe_deck_{section['key']}", use_container_width=True, type=button_type):
                _set_section_key(section["key"])
                st.rerun()
    with top_right:
        st.download_button(
            "导出演示 PDF",
            data=pdf_bytes,
            file_name=f"uni-sign-{controls['judge']}-{controls['split']}-judge-brief.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    left, right = st.columns([1.7, 0.9], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="command-shell">
                <div class="command-title">演示节奏</div>
                <div class="command-meta">{html.escape(SECTION_LOOKUP[current_key]["label"])} ｜ {html.escape(str(controls["judge"]))} ｜ {html.escape(str(controls["stage"]))}</div>
                <div class="command-line">{html.escape((script_lines or QUOTE_LINES)[0])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        timeline_fig = go.Figure(
            data=[
                go.Bar(
                    x=[item["label"] for item in SECTION_ORDER],
                    y=[item["seconds"] for item in SECTION_ORDER],
                    marker=dict(color=[("#F97316" if item["key"] == current_key else item["accent"]) for item in SECTION_ORDER]),
                    text=[item["seconds"] for item in SECTION_ORDER],
                    textposition="outside",
                )
            ]
        )
        timeline_fig.update_layout(
            height=230,
            margin=dict(l=10, r=10, t=18, b=10),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            xaxis=dict(linecolor="rgba(148,163,184,0.22)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.12)", linecolor="rgba(148,163,184,0.22)", title="秒"),
            showlegend=False,
        )
        st.plotly_chart(timeline_fig, use_container_width=True)
    return current_key


def render_presentation_mode(controls: dict[str, object], cards: dict[str, object], summary_df: pd.DataFrame) -> str:
    return render_stage_console(controls, cards, summary_df)


def main() -> None:
    controls = sidebar_controls()
    api_client = ViewAPIClient(controls["api_base"])
    cards = build_metric_cards(controls["split"])
    summary_df = cached_metrics_summary(controls["split"])
    show_hero(controls)
    render_overview(controls["split"])
    render_echarts_overview(controls["split"])
    current_section = render_stage_console(controls, cards, summary_df)

    if current_section == "live":
        render_live_demo(controls["split"], api_client)
    elif current_section == "xai":
        render_xai_lab(controls["split"], api_client)
    elif current_section == "compare":
        render_model_compare(controls["split"])
    elif current_section == "arch":
        render_architecture()
    else:
        render_scenes()


if __name__ == "__main__":
    main()
