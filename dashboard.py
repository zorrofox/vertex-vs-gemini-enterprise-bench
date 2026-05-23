import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROXY_BASE = os.environ.get("PROXY_BASE", "http://127.0.0.1:8000")
PROXY_URL = f"{PROXY_BASE}/v1/chat/completions"
RESULTS_FILE = "benchmark_results.json"
URL_PATTERN = re.compile(r"https?://[^\s)\]]+")


@dataclass(frozen=True)
class ModelStyle:
    label: str
    desc: str
    bar_color: str
    line_color: str
    header_gradient: str


MODEL_STYLES: dict[str, ModelStyle] = {
    "gemini-standard": ModelStyle(
        "gemini-standard", "标准 2.5 Flash（无搜索）", "#4b5563", "#9ca3af",
        "linear-gradient(90deg, #4b5563 0%, #1f2937 100%)",
    ),
    "vertex-grounded-search": ModelStyle(
        "vertex-grounded-search", "Vertex AI Grounding（内置搜索）", "#0284c7", "#38bdf8",
        "linear-gradient(90deg, #0284c7 0%, #0369a1 100%)",
    ),
    "discovery-standard": ModelStyle(
        "discovery-standard", "Discovery 标准（不启用搜索）", "#ec4899", "#f472b6",
        "linear-gradient(90deg, #db2777 0%, #9d174d 100%)",
    ),
    "discovery-grounded-search": ModelStyle(
        "discovery-grounded-search", "Discovery Grounded（企业管道）", "#7c3aed", "#c084fc",
        "linear-gradient(90deg, #7c3aed 0%, #6d28d9 100%)",
    ),
}
MODELS = list(MODEL_STYLES)
BAR_COLORS = {m: s.bar_color for m, s in MODEL_STYLES.items()}
LINE_COLORS = {m: s.line_color for m, s in MODEL_STYLES.items()}

CUSTOM_CSS = """
<style>
.stApp { background: linear-gradient(135deg, #0e1117 0%, #161a25 100%); }
.glass-card {
    background: rgba(255,255,255,0.03); border-radius: 12px; padding: 20px;
    border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
    backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); margin-bottom: 20px;
}
.gradient-text {
    background: linear-gradient(90deg, #a855f7 0%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 2.8rem; margin-bottom: 5px;
}
.model-header {
    padding: 8px 15px; border-radius: 8px; font-weight: bold; color: #fff;
    margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.1);
}
.metric-number {
    font-family: 'Courier New', monospace; font-size: 1.5rem; font-weight: bold; color: #38bdf8;
}
.metric-box {
    background: rgba(255,255,255,0.02); padding: 10px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.05); margin-top: 10px;
}
</style>
"""


def model_header_html(model: str) -> str:
    s = MODEL_STYLES[model]
    return (
        f"<div class='model-header' style='background:{s.header_gradient};'>{s.label}<br>"
        f"<span style='font-size:0.8rem;font-weight:normal;color:#cbd5e1;'>{s.desc}</span></div>"
    )


def metric_card_html(ttft: float, total: float, cps: float, cites: int, demo: bool) -> str:
    suffix = " (模拟)" if demo else ""
    return f"""<div class='metric-box'>
    ⏱️ 首包耗时 (TTFT): <span class='metric-number'>{ttft:.2f}s</span>{suffix}<br>
    ⏱️ 总响应耗时: <span class='metric-number'>{total:.2f}s</span>{suffix}<br>
    🚀 平均吞吐量: <span class='metric-number'>{cps:.1f} 字/s</span>{suffix}<br>
    🔗 网页引源数量: <span class='metric-number'>{cites}</span>{suffix}
    </div>"""


@st.cache_data(ttl=5)
def load_results() -> dict[str, Any] | None:
    if not os.path.exists(RESULTS_FILE):
        return None
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_cached_response(prompt: str, model: str) -> dict[str, Any] | None:
    data = load_results()
    if not data:
        return None
    p = prompt.strip()
    for q in data.get("results", []):
        qq = q["query"].strip()
        if p == qq or p in qq or qq in p:
            m = q["models"].get(model, {})
            if m.get("response"):
                return m
    return None


DEMO_FALLBACK = {
    "gemini-standard": (
        "我没有配备实时搜索工具，无法提供该话题的最新进展。请查阅旁边配备 Grounding 的通道。",
        0.5, 1.2, 0,
    ),
    "vertex-grounded-search": (
        "正在通过 **Vertex AI Google Search Grounding** 检索最新信息...\n\n"
        "Sources:\n1. [Vertex AI Grounding Docs](https://cloud.google.com/vertex-ai/docs/generative-ai/grounding/overview)\n"
        "2. [Google Search](https://www.google.com)",
        1.5, 3.5, 2,
    ),
    "discovery-standard": (
        "通过 **Discovery Engine 企业管道** 直接生成（不启用搜索），用于孤立对比 Discovery 链路的固定延迟开销。",
        1.1, 2.1, 0,
    ),
    "discovery-grounded-search": (
        "已连接 **Discovery Engine 企业检索生成管道**，整合 Google Search 数据。\n\n"
        "Sources:\n1. [Discovery Engine Docs](https://cloud.google.com/generative-ai-app-builder/docs)\n"
        "2. [GCP App Builder](https://cloud.google.com/generative-ai-app-builder)\n"
        "3. [UEFA News](https://www.uefa.com)",
        1.2, 2.8, 3,
    ),
}


async def stream_model(model: str, prompt: str, text_ph, metric_ph, demo: bool) -> None:
    if demo:
        cached = find_cached_response(prompt, model)
        if cached:
            text, ttft, total, cites = (
                cached["response"],
                cached.get("ttft", 1.0),
                cached.get("total_latency", 3.0),
                cached.get("citations_count", 0),
            )
        else:
            text, ttft, total, cites = DEMO_FALLBACK[model]
            text = f"您输入了：「{prompt}」\n\n{text}"

        await asyncio.sleep(ttft)
        step = max(1, len(text) // 80)
        delay = max(0.005, min((total - ttft) / max(1, len(text) // step), 0.04))
        buf = ""
        for i in range(0, len(text), step):
            buf = text[: i + step]
            text_ph.markdown(buf + "▌")
            await asyncio.sleep(delay)
        text_ph.markdown(text)
        cps = len(text) / (total - ttft) if total > ttft else len(text)
        metric_ph.markdown(metric_card_html(ttft, total, cps, cites, demo=True), unsafe_allow_html=True)
        return

    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    start = time.perf_counter()
    ttft: float | None = None
    full = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", PROXY_URL, json=payload) as r:
                if r.status_code != 200:
                    err = (await r.aread()).decode(errors="replace")
                    text_ph.error(f"请求失败 ({r.status_code}): {err}")
                    return
                async for raw in r.aiter_lines():
                    line = raw.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content", "")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        full += content
                        text_ph.markdown(full + "▌")
    except Exception as e:
        text_ph.error(f"接口异常: {e}")
        return

    total = time.perf_counter() - start
    text_ph.markdown(full or "_（无内容返回）_")
    ttft_val = ttft if ttft is not None else total
    gen_time = max(total - ttft_val, 1e-6)
    cps = len(full) / gen_time
    cites = len(URL_PATTERN.findall(full))
    metric_ph.markdown(metric_card_html(ttft_val, total, cps, cites, demo=False), unsafe_allow_html=True)


def render_sidebar() -> bool:
    st.sidebar.markdown(
        "<h2 style='text-align:center;color:#a855f7;'>⚡ 评测控制中心</h2>", unsafe_allow_html=True
    )
    demo = st.sidebar.checkbox(
        "🎮 竞技场演示模式 (Demo Mode)",
        value=False,
        help="开启后优先从 benchmark_results.json 加载已保存结果；关闭后实时请求本地网关。",
    )
    try:
        res = httpx.get(f"{PROXY_BASE}/health", timeout=2.0)
        if res.status_code == 200:
            d = res.json()
            st.sidebar.success(
                f"🟢 代理运行中\n\n项目ID: `{d.get('gcp_project_id')}`\n\n"
                f"项目编号: `{d.get('gcp_project_number')}`"
            )
        else:
            st.sidebar.error("🔴 代理服务异常")
    except Exception:
        st.sidebar.error("🔴 未能连接代理\n\n请启动: `python app.py`")

    st.sidebar.markdown("---\n### ⚙️ 基准评测")
    st.sidebar.info("点击下方按钮执行全量 benchmark（约 1-2 分钟）。")
    if st.sidebar.button("🚀 启动全量自动化基准评测"):
        run_benchmark_subprocess()
    return demo


def run_benchmark_subprocess() -> None:
    log_path = "benchmark_run.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "benchmark.py"], stdout=log_file, stderr=subprocess.STDOUT
        )
    bar = st.sidebar.progress(0)
    status = st.sidebar.empty()
    start = time.time()
    while proc.poll() is None:
        elapsed = time.time() - start
        bar.progress(min(int(elapsed / 120 * 95), 95))
        status.info(f"运行中... {int(elapsed)}s（日志: {log_path}）")
        time.sleep(1.0)
    bar.progress(100)
    if proc.returncode == 0:
        st.sidebar.success("🎉 基准评测完成，数据已更新。")
        load_results.clear()
        st.rerun()
    else:
        with open(log_path) as f:
            tail = f.read()[-2000:]
        st.sidebar.error(f"评测失败 (exit {proc.returncode})。日志末尾：\n```\n{tail}\n```")


def render_arena(demo: bool) -> None:
    st.markdown(
        "<div class='glass-card'><h4>💡 自定义实时提问与并排对比</h4>"
        "输入需要搜索印证的最新问题，系统将<b>并发流式</b>请求四个后端。"
        "可开启 Demo Mode 加载预存结果，或关闭后实时请求本地网关。</div>",
        unsafe_allow_html=True,
    )
    options = ["-- 手动输入自定义 Prompt --"]
    if data := load_results():
        options += [q["query"] for q in data.get("results", [])]
    selected = st.selectbox("🎯 快速选择经典基准问题：", options=options)
    if selected != options[0]:
        prompt = selected
    else:
        prompt = st.text_input("输入自定义 Prompt：", placeholder="最近科技界有什么重大新闻？")
    if not prompt:
        return

    cols = st.columns(len(MODELS))
    placeholders = []
    for col, model in zip(cols, MODELS):
        with col:
            st.markdown(model_header_html(model), unsafe_allow_html=True)
            placeholders.append((st.empty(), st.empty()))

    async def run_all():
        await asyncio.gather(
            *(
                stream_model(m, prompt, tp, mp, demo)
                for m, (tp, mp) in zip(MODELS, placeholders)
            )
        )

    asyncio.run(run_all())


def results_to_df(data: dict[str, Any]) -> pd.DataFrame:
    records = []
    for q in data["results"]:
        for m, res in q["models"].items():
            if "error" in res:
                continue
            scores = res.get("scores", {})
            records.append({
                "id": q["id"],
                "query": q["query"][:12] + "...",
                "full_query": q["query"],
                "model": m,
                "ttft": res.get("ttft", 0.0),
                "total_latency": res.get("total_latency", 0.0),
                "chars_per_sec": res.get("chars_per_sec", res.get("throughput", 0.0)),
                "citations_count": res.get("citations_count", 0),
                "accuracy": scores.get("accuracy", 1),
                "recency": scores.get("recency", 1),
                "citations": scores.get("citations", 1),
                "rationale": scores.get("rationale", ""),
                "response": res.get("response", ""),
            })
    return pd.DataFrame(records)


def avg_bar(df: pd.DataFrame, y: str, title: str, ylabel: str) -> go.Figure:
    fig = px.bar(df, x="model", y=y, color="model", title=title, color_discrete_map=BAR_COLORS)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=ylabel,
                      height=240, margin=dict(t=30, b=10, l=10, r=10))
    return fig


def render_analytics() -> None:
    data = load_results()
    if not data:
        st.markdown(
            "<div style='text-align:center;padding:40px;' class='glass-card'>"
            "<h3>📊 暂无基准测试数据</h3>"
            "<p style='color:#94a3b8;'>请通过左侧 <b>🚀 启动全量自动化基准评测</b> 生成数据。</p></div>",
            unsafe_allow_html=True,
        )
        return

    ts = data["metadata"].get("timestamp", "")
    fmt_ts = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "未知"
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;margin-bottom:15px;'>"
        f"<span style='color:#94a3b8;'>📊 最近运行：<b>{fmt_ts}</b></span>"
        f"<span style='color:#94a3b8;'>裁判模型：<code>{data['metadata'].get('judge_model')}</code></span></div>",
        unsafe_allow_html=True,
    )

    df = results_to_df(data)
    if df.empty:
        st.warning("结果文件中无有效记录。")
        return
    avg_df = df.groupby("model").mean(numeric_only=True).reset_index()

    st.markdown("### 🏆 核心效能对比均值")
    c1, c2, c3, c4 = st.columns(4)
    c1.plotly_chart(avg_bar(avg_df, "accuracy", "🧠 事实准确度均值 (1-5)", "分值"), width="stretch")
    c2.plotly_chart(avg_bar(avg_df, "recency", "⏱️ 信息时效性均值 (1-5)", "分值"), width="stretch")
    c3.plotly_chart(avg_bar(avg_df, "ttft", "⚡ 首包延迟 TTFT 均值 (秒↓)", "秒"), width="stretch")
    c4.plotly_chart(avg_bar(avg_df, "citations_count", "🔗 网页引源均值 (个)", "个"), width="stretch")

    st.markdown("---\n### ⏱️ 时延与首包波动深度剖析")
    l1, l2 = st.columns(2)
    fig_box = px.box(df, x="model", y="ttft", color="model", points="all",
                     title="📦 首包时延 TTFT 波动分布", color_discrete_map=BAR_COLORS)
    fig_box.update_layout(xaxis_title=None, yaxis_title="首包时间 (秒)", showlegend=False, height=340)
    l1.plotly_chart(fig_box, width="stretch")
    fig_line = px.line(df, x="id", y="total_latency", color="model", markers=True,
                       title="📈 逐题总响应时延趋势", color_discrete_map=LINE_COLORS)
    fig_line.update_layout(xaxis_title="问题 ID", yaxis_title="总响应时间 (秒)", height=340)
    l2.plotly_chart(fig_line, width="stretch")

    st.markdown("---\n### 🎯 质量裁判综合多维雷达图")
    cats = ["事实准确度", "信息时效性", "引用完备性"]
    fig_radar = go.Figure()
    for m in MODELS:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        vals = [sub["accuracy"].mean(), sub["recency"].mean(), sub["citations"].mean()]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            name=m, line=dict(color=LINE_COLORS[m]),
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                            showlegend=True, height=380, margin=dict(t=30, b=30))
    st.plotly_chart(fig_radar, width="stretch")

    st.markdown("---\n### 📝 问题级别裁判评语与生成明细")
    qid = st.selectbox(
        "选择问题查看详情：", options=df["id"].unique(),
        format_func=lambda x: f"问题 {x}: {df[df['id'] == x]['full_query'].iloc[0][:35]}...",
    )
    q_rec = df[df["id"] == qid]
    st.markdown(f"**❓ 评测问题原始文本：**\n> {q_rec['full_query'].iloc[0]}")
    tabs = st.tabs(MODELS)
    for tab, m in zip(tabs, MODELS):
        with tab:
            mr = q_rec[q_rec["model"] == m]
            if mr.empty:
                st.warning("该模型在此题目上无成功响应记录。")
                continue
            r = mr.iloc[0]
            st.markdown(
                f"<div style='display:flex;gap:15px;margin-bottom:12px;'>"
                f"<span>🧠 准确度: <b>{r['accuracy']}</b>/5</span>"
                f"<span>⏱️ 时效性: <b>{r['recency']}</b>/5</span>"
                f"<span>🔗 引用分: <b>{r['citations']}</b>/5</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("**生成的原始文本 (含引源)：**")
            st.code(r["response"], language="markdown")
            st.info(f"**⚖️ 裁判打分说明：**\n\n{r['rationale']}")


def main() -> None:
    st.set_page_config(
        page_title="GCP Discovery vs Vertex AI Benchmark Arena",
        page_icon="⚡", layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    demo = render_sidebar()
    st.markdown("<p class='gradient-text'>GCP Discovery vs. Vertex AI Arena</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:1.1rem;color:#94a3b8;margin-top:-10px;margin-bottom:25px;'>"
        "针对 GCP Discovery Engine 与 Vertex AI 进行性能与事实准确性（LLM-as-a-Judge）量化对比。</p>",
        unsafe_allow_html=True,
    )
    tab_arena, tab_analytics = st.tabs(["🔥 实时并排对比竞技场", "📊 批量基准评测可视化报告"])
    with tab_arena:
        render_arena(demo)
    with tab_analytics:
        render_analytics()


if __name__ == "__main__":
    main()
