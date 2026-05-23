"""从 benchmark_results.json 渲染对比图表为静态 PNG，用于文档嵌入。"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, ".")
from dashboard import BAR_COLORS, LINE_COLORS, MODELS, results_to_df  # noqa: E402

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATE = "plotly_dark"
LABEL = {
    "gemini-standard": "Vertex (无搜索)",
    "vertex-grounded-search": "Vertex + Search",
    "discovery-standard": "GE Assist (无搜索)",
    "discovery-grounded-search": "GE Assist + Search",
}


def save(fig: go.Figure, name: str, w: int = 900, h: int = 450) -> None:
    fig.update_layout(template=TEMPLATE, font=dict(size=13))
    path = OUT / f"{name}.png"
    fig.write_image(str(path), width=w, height=h, scale=2)
    print(f"  → {path}")


def main() -> None:
    with open("benchmark_results.json", encoding="utf-8") as f:
        data = json.load(f)
    df = results_to_df(data)
    df["label"] = df["model"].map(LABEL)
    avg = df.groupby(["model", "label"]).mean(numeric_only=True).reset_index()
    order = [LABEL[m] for m in MODELS]

    # 1. 质量综合分
    avg["overall"] = (avg["accuracy"] + avg["recency"] + avg["citations"]) / 3
    fig = px.bar(
        avg, x="label", y="overall", color="model", text_auto=".2f",
        title="LLM-as-Judge 综合质量分（准确度+时效性+引用，满分 5）",
        color_discrete_map=BAR_COLORS, category_orders={"label": order},
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="综合分", yaxis_range=[0, 5])
    save(fig, "01-quality-overall")

    # 2. 三维质量分组对比
    qdf = avg.melt(
        id_vars=["label", "model"], value_vars=["accuracy", "recency", "citations"],
        var_name="metric", value_name="score",
    )
    qdf["metric"] = qdf["metric"].map({"accuracy": "准确度", "recency": "时效性", "citations": "引用完备性"})
    fig = px.bar(
        qdf, x="metric", y="score", color="model", barmode="group", text_auto=".2f",
        title="三维质量分项对比（1-5 分）",
        color_discrete_map=BAR_COLORS,
    )
    fig.for_each_trace(lambda t: t.update(name=LABEL.get(t.name, t.name)))
    fig.update_layout(xaxis_title=None, yaxis_title="分值", yaxis_range=[0, 5.2])
    save(fig, "02-quality-breakdown")

    # 3. 延迟对比（TTFT + 总耗时）
    ldf = avg.melt(
        id_vars=["label", "model"], value_vars=["ttft", "total_latency"],
        var_name="metric", value_name="seconds",
    )
    ldf["metric"] = ldf["metric"].map({"ttft": "首包延迟 TTFT", "total_latency": "总响应耗时"})
    fig = px.bar(
        ldf, x="label", y="seconds", color="metric", barmode="group", text_auto=".2f",
        title="延迟对比（秒，越低越好）",
        color_discrete_sequence=["#38bdf8", "#a855f7"], category_orders={"label": order},
    )
    fig.update_layout(xaxis_title=None, yaxis_title="秒")
    save(fig, "03-latency")

    # 4. TTFT 箱线分布
    fig = px.box(
        df, x="label", y="ttft", color="model", points="all",
        title="首包延迟 TTFT 波动分布（8 题）",
        color_discrete_map=BAR_COLORS, category_orders={"label": order},
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="TTFT (秒)")
    save(fig, "04-ttft-box")

    # 5. 逐题总耗时趋势
    fig = px.line(
        df, x="id", y="total_latency", color="model", markers=True,
        title="逐题总响应耗时趋势",
        color_discrete_map=LINE_COLORS,
    )
    fig.for_each_trace(lambda t: t.update(name=LABEL.get(t.name, t.name)))
    fig.update_layout(xaxis_title="问题 ID", yaxis_title="总耗时 (秒)")
    save(fig, "05-latency-per-question")

    # 6. 雷达图
    cats = ["准确度", "时效性", "引用完备性"]
    fig = go.Figure()
    for m in MODELS:
        sub = df[df["model"] == m]
        vals = [sub["accuracy"].mean(), sub["recency"].mean(), sub["citations"].mean()]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            name=LABEL[m], line=dict(color=LINE_COLORS[m]),
        ))
    fig.update_layout(
        title="质量裁判多维雷达图",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    )
    save(fig, "06-radar", w=700, h=550)

    print(f"\n✓ 生成 6 张图表 → {OUT}/")


if __name__ == "__main__":
    main()
