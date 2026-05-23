import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from typing import Any

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from discovery_api.auth import token_cache
from discovery_api.config import settings
from discovery_api.schemas import MODEL_ROUTES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("discovery-benchmark")

PROXY_BASE = f"http://127.0.0.1:{settings.proxy_port}"
PROXY_URL = f"{PROXY_BASE}/v1/chat/completions"
RESULTS_FILE = "benchmark_results.json"
MODELS_TO_TEST = [m for m in MODEL_ROUTES if m != "gemini-2.5-flash"]
URL_PATTERN = re.compile(r"https?://[^\s)\]]+")


class JudgeResult(BaseModel):
    accuracy: int = Field(description="准确度（1-5）：是否符合真实事实，是否存在编造、时间错乱。")
    recency: int = Field(description="时效性（1-5）：是否捕获到最新动态。无最新内容或表示不了解给低分。")
    citations: int = Field(description="引用完备性（1-5）：尾部是否提供带序号标题的真实跳转链接。无 URL 必须给 1 分。")
    rationale: str = Field(description="详细打分说明与各模型优缺点对比。")


class MultiModelEvaluation(BaseModel):
    gemini_standard: JudgeResult
    vertex_grounded_search: JudgeResult
    discovery_standard: JudgeResult
    discovery_grounded_search: JudgeResult


BENCHMARK_QUESTIONS = [
    {
        "id": 1,
        "query": "2025年欧洲冠军联赛（UEFA Champions League）决赛谁夺冠了？比分是多少？",
        "reference": "2024-2025赛季欧冠决赛于2025年5月31日在慕尼黑安联球场举行，请模型搜索最新真实战况。",
    },
    {
        "id": 2,
        "query": "英伟达（NVIDIA, NVDA）最新一个季度的营收（Revenue）是多少？相比去年同期增长了多少？",
        "reference": "查找英伟达在2025/2026年发布的最新财报数据，评估财务指标的真实性。",
    },
    {
        "id": 3,
        "query": "下一次世界范围内的日全食（Total Solar Eclipse）将在什么时候发生？在哪些国家或地区可以观测到？",
        "reference": "下一次日全食预计发生在2026年8月12日，主要掠过冰岛、西班牙、格陵兰岛等。",
    },
    {
        "id": 4,
        "query": "请问当前的英国首相是谁？他是属于哪个党派的？",
        "reference": "Keir Starmer（基尔·斯塔默，工党，自2024年7月起执政，需核实是否有最新更替）。",
    },
    {
        "id": 5,
        "query": "请提供今天（最新）比特币（Bitcoin, BTC）的价格，并简述过去一周的主要价格波动原因。",
        "reference": "需要获取当下最新的实盘价格或最近一两天的最新行情。",
    },
    {
        "id": 6,
        "query": "最近一届奥斯卡金像奖（Academy Awards, 2026年或2025年）的最佳影片（Best Picture）是哪部？导演是谁？",
        "reference": "查找最新的奥斯卡最佳影片得主，模型必须展示最新的真实结果。",
    },
    {
        "id": 7,
        "query": "OpenAI 公司在最近几周有什么最重大的高管变动或核心产品模型的发布消息？",
        "reference": "需要获取最新的科技新闻，验证搜索接地能否捕捉最新业界动态。",
    },
    {
        "id": 8,
        "query": "美国宇航局（NASA）的阿耳忒弥斯2号（Artemis II）载人绕月飞行任务，目前最新的发射计划时间是什么时候？",
        "reference": "Artemis II 原计划不早于2025年9月发射，核实是否有进一步推迟。",
    },
]


def _error_result(message: str, latency: float = 0.0) -> dict[str, Any]:
    return {
        "response": f"【报错】{message}",
        "ttft": 0.0,
        "total_latency": latency,
        "chars_per_sec": 0.0,
        "citations_count": 0,
        "error": message,
    }


def _penalty_scores(reason: str) -> dict[str, Any]:
    base = {"accuracy": 1, "recency": 1, "citations": 1, "rationale": reason}
    return {m.replace("-", "_"): dict(base) for m in MODELS_TO_TEST}


async def run_single_test(client: httpx.AsyncClient, model: str, query: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": True,
        "temperature": 1.0,
    }
    start = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []

    try:
        async with client.stream("POST", PROXY_URL, json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")
                return _error_result(f"HTTP {r.status_code}: {body}")

            async for raw in r.aiter_lines():
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    return _error_result(data["error"].get("message", "Unknown downstream error"))
                choices = data.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    chunks.append(content)
    except Exception as e:
        logger.error("Exception calling proxy for model %s: %s", model, e)
        return _error_result(str(e))

    total = time.perf_counter() - start
    full = "".join(chunks)
    if not full:
        return _error_result("流式接口未返回任何文本", latency=total)

    return {
        "response": full,
        "ttft": ttft if ttft is not None else total,
        "total_latency": total,
        "chars_per_sec": len(full) / total if total > 0 else 0.0,
        "citations_count": len(URL_PATTERN.findall(full)),
    }


def get_judge_client() -> genai.Client:
    creds = token_cache.get_credentials()
    return genai.Client(
        vertexai=True,
        project=settings.gcp_project_id,
        location=settings.vertex_location,
        credentials=creds,
    )


async def _judge_call(contents: str, config: types.GenerateContentConfig) -> str:
    """带 3 次指数退避的裁判模型调用。"""
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            client = get_judge_client()
            async with asyncio.timeout(60.0):
                res = await client.aio.models.generate_content(
                    model=settings.gemini_model, contents=contents, config=config
                )
                return res.text
        except Exception as e:
            last_err = e
            logger.warning("Judge attempt %d/3 failed: %s", attempt, e)
            if attempt < 3:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"All judge attempts failed: {last_err}")


async def generate_ground_truth(query: str, ref: str) -> str:
    prompt = f"""你是一位极其专业、冷酷客观、手握实时网络接地工具的最高事实核查法官。
为了树立绝对公正的"物理世界黄金标准"（Ground Truth），请你务必使用 google_search 实时网页工具，彻底核实该问题在真实世界中的现状。

检索时必须使用包含具体年份（2025/2026）和明确实体的精确关键词，例如：
- 欧冠：'2025 UEFA Champions League final winner score'
- 英伟达：'NVIDIA latest quarter revenue 2025 YoY'
- 首相/比特币/奥斯卡：包含当前年份和具体实体名称

必须识别并过滤以下伪装成真实引用的高级幻觉：
- 维基百科用户沙盒/草稿空间（wikipedia.org/wiki/User:...）
- 游戏同人/体育模拟社区（Fandom、FM 模拟数据等）
- 对未来日期的未证实预测
若该事实尚未发生，请明确指出"尚未发生"并写明目前最接近的真实情况。

[需要核实的问题]
{query}

[人工设定的参考线索]
{ref}

[任务]
输出一份详尽的《物理世界客观事实黄金标准（Fact Sheet）》，包含：
1. 真实的物理世界事实
2. 指明哪些是常见虚构/模拟内容（如有）
3. 准确的参考来源 URL
"""
    try:
        return await _judge_call(
            prompt,
            types.GenerateContentConfig(temperature=0.1, tools=[{"google_search": {}}]),
        )
    except Exception as e:
        logger.error("Fact sheet generation failed: %s", e)
        return f"无法在线抓取（{e}）。请完全信任人工预设的参考线索：{ref}"


async def evaluate_responses(query: str, ref: str, model_responses: dict[str, Any]) -> dict[str, Any]:
    logger.info("Step 1: Fetching live Ground Truth via Google Search tool...")
    ground_truth = await generate_ground_truth(query, ref)
    logger.info("Ground Truth Fact Sheet generated (%d chars)", len(ground_truth))

    logger.info("Step 2: Structured grading against Fact Sheet...")
    sections = []
    for name, res in model_responses.items():
        ans = res.get("response", "（请求失败或无回复）")
        sections.append(f"\n### 模型: {name}\n--- 回答开始 ---\n{ans}\n--- 回答结束 ---\n")

    prompt = f"""你是一位极其专业、冷酷客观的高级 AI 质量评测专家与事实核查最高法官。
请对照下方【Fact Sheet】与【各模型输出】，对每个模型在三个维度打分（1-5）。

[测试问题]
{query}

[物理世界客观事实黄金标准（Fact Sheet）]
{ground_truth}

[各模型输出]
{"".join(sections)}

[打分细则]
1. accuracy：信息是否与 Fact Sheet 符合。若模型提供了正确的最新事实且引用可交叉验证，给高分甚至满分；不能因 Fact Sheet 单点缺失而误判。完全捏造或过时给 1-2 分。
2. recency：是否包含当下最新进展。无搜索能力只给老旧历史或诚实致歉给 1-2 分。
3. citations：尾部是否带真实网页 URL 的 Sources 栏。无任何 URL 必须给 1 分；有真实 URL 给 4-5 分。

严格按 JSON Schema 输出，不要包裹 markdown 代码块。
"""
    try:
        text = await _judge_call(
            prompt,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MultiModelEvaluation,
                temperature=0.1,
            ),
        )
        return json.loads(text)
    except Exception as e:
        logger.error("Grading failed, falling back to penalty scores: %s", e)
        return _penalty_scores(f"裁判评分失败: {e}")


async def main() -> None:
    logger.info("Initializing Benchmark Pipeline...")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            res = await client.get(f"{PROXY_BASE}/health")
            res.raise_for_status()
            logger.info("Connected to local Proxy Server.")
        except Exception:
            logger.error("Cannot reach Proxy at %s. Start it with: python app.py", PROXY_BASE)
            sys.exit(1)

        try:
            get_judge_client()
            logger.info("Judge client initialized on project: %s", settings.gcp_project_id)
        except Exception as e:
            logger.error("Failed to initialize judge client: %s", e)
            sys.exit(1)

        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "judge_model": settings.gemini_model,
                "project_id": settings.gcp_project_id,
            },
            "results": [],
        }

        for item in BENCHMARK_QUESTIONS:
            qid, query, ref = item["id"], item["query"], item["reference"]
            logger.info("\n%s\n[Q%s] %s\n%s", "=" * 40, qid, query, "=" * 40)

            model_responses: dict[str, Any] = {}
            for model in MODELS_TO_TEST:
                logger.info("Testing model: %s", model)
                model_responses[model] = await run_single_test(client, model, query)
                await asyncio.sleep(2.0)

            scores = await evaluate_responses(query, ref, model_responses)
            for model in MODELS_TO_TEST:
                key = model.replace("-", "_")
                model_responses[model]["scores"] = scores.get(
                    key, {"accuracy": 1, "recency": 1, "citations": 1, "rationale": "未找到匹配得分"}
                )

            results["results"].append(
                {"id": qid, "query": query, "reference": ref, "models": model_responses}
            )
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Benchmark completed. Results saved to: %s", RESULTS_FILE)


if __name__ == "__main__":
    asyncio.run(main())
