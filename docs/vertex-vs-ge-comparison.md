# Vertex AI Gemini vs. Gemini Enterprise App API 对比报告

> 评测日期：2026-05-23
> 评测工具：本仓库 `benchmark.py` + `dashboard.py`
> 裁判模型：`gemini-2.5-flash`（双阶段 LLM-as-a-Judge）

---

## 1. 对比对象

| 维度 | **Vertex AI 直调** | **Gemini Enterprise (GE) App** |
|------|---|---|
| 产品定位 | 开发者裸 API，按量付费 | 企业员工 AI 助手，按席位订阅 |
| API 端点 | `aiplatform.googleapis.com/v1/.../models/{model}:streamGenerateContent` | `discoveryengine.googleapis.com/v1alpha/.../engines/{app}/assistants/{assistant}:streamAssist` |
| Google Search 接入方式 | `tools: [{google_search: {}}]` | `toolsSpec: {webGroundingSpec: {}}`（Assistant 需开启 `webGroundingType: GOOGLE_SEARCH`） |
| 模型版本指定 | `gemini-2.5-flash` | `generationSpec.modelId`（如 `gemini-2.5-flash/answer_gen/v1`，留空用 App 默认） |
| 认证 | ADC / Service Account | ADC + `X-Goog-User-Project` header（席位需分配到目标项目） |

---

## 2. 实测性能（8 题 × 4 通道，全量 Benchmark）

### 2.1 均值汇总

| 通道 | TTFT | 总耗时 | 吞吐 | 引源数 | 准确度 | 时效性 | 引用分 | **综合** |
|------|------|--------|------|--------|--------|--------|--------|---------|
| Vertex（无搜索） | 6.52s | 8.08s | 51 字/s | 0 | 1.50 | 1.62 | 1.00 | 1.38 |
| **Vertex + Google Search** | **6.30s** | **7.53s** | **477 字/s** | 9.8 | 4.00 | 4.62 | 4.75 | **4.46** |
| GE Assist（无搜索） | 14.65s | 19.09s | 36 字/s | 0 | 2.12 | 2.25 | 1.00 | 1.79 |
| **GE Assist + Google Search** | 8.73s | 14.00s | 268 字/s | **10.2** | **4.25** | 4.62 | 4.75 | **4.54** |

![综合质量分](images/01-quality-overall.png)

![延迟对比](images/03-latency.png)

### 2.2 两条 Grounded 路径直接 PK

| 维度 | Vertex + Search | GE + Search | 差异 |
|------|----------------|-------------|------|
| 综合质量分 | 4.46 | **4.54** | GE +1.8%（统计噪声范围内） |
| TTFT | **6.30s** | 8.73s | GE 慢 **38%** |
| 总耗时 | **7.53s** | 14.00s | GE 慢 **86%** |
| 吞吐 | **477 字/s** | 268 字/s | GE 慢 **44%** |
| 引源数 | 9.8 | 10.2 | 持平 |

**结论**：两条路径回答质量无显著差异（信息源同为 Google Search）；GE 的 Assistant 编排层（plannerSteps + persona 注入）引入约 2.4s TTFT 和 6.5s 端到端固定开销。

![三维质量分项](images/02-quality-breakdown.png)

![质量雷达图](images/06-radar.png)

### 2.3 逐题明细（准确/时效/引用，括号为 TTFT）

| 题目 | Vertex+Search | GE+Search | 备注 |
|------|---------------|-----------|------|
| Q1 2025 欧冠决赛 | 5/5/5 (3.3s) | 5/5/5 (9.0s) | 平 |
| Q2 NVDA 季度营收 | 2/5/5 (3.9s) | **5/5/5** (8.3s) | GE 胜（Vertex 准确度仅 2） |
| Q3 下次日全食 | **5/5/5** (3.9s) | 3/5/5 (9.4s) | Vertex 胜 |
| Q4 英国首相 | 5/5/5 (3.1s) | 5/5/5 (9.4s) | 平 |
| Q5 BTC 实时价格 | 5/5/5 (9.9s) | 5/5/5 (8.7s) | 平 |
| Q6 奥斯卡最佳影片 | 5/5/4 (5.8s) | 5/5/4 (9.1s) | 平 |
| Q7 OpenAI 近期动态 | 1/2/4 (15.6s) | 1/2/4 (6.7s) | 双方均未答准 |
| Q8 Artemis II 计划 | 4/5/5 (5.0s) | **5/5/5** (9.1s) | GE 微胜 |

![TTFT 波动分布](images/04-ttft-box.png)

![逐题总耗时趋势](images/05-latency-per-question.png)

---

## 3. 多模态输出能力

`AssistantContent.data` 是 union 类型，GE `:streamAssist` 在单一端点内自动编排多模态工具：

| 输出类型 | Vertex `generateContent` | GE `:streamAssist` | 实测验证 |
|---------|--------------------------|--------------------|---------| 
| `text` | ✅ | ✅ | — |
| `inlineData` (Blob) | ✅ | ✅ | — |
| `file` (mimeType + fileId) | ❌ | ✅ | 返回 `image/png` + fileId |
| `executableCode` | 需显式 `tools:[{code_execution:{}}]` | ✅ 自动触发 | `{language: PYTHON}` |
| `codeExecutionResult` | 同上 | ✅ 含失败自动重试 | `OUTCOME_OK` |
| 图片生成 | 需切换到 Imagen API / `-image` 模型 | ✅ `toolsSpec:{imageGenerationSpec:{}}` | 调度 Nano Banana |
| `thought` 思考过程 | 需 thinking 模型 | ✅ | `thought: true` |
| `diagnosticInfo.plannerSteps` | ❌ | ✅ | Agent 推理轨迹 |
| 多模态输入（文件上传） | `fileData` URI | `AddContextFile` 上传到 session | — |

---

## 4. 定价与配额

### 4.1 Vertex AI（按量付费）

| 计费项 | 价格 |
|--------|------|
| Gemini 2.5 Flash input | $0.30 / 1M tokens |
| Gemini 2.5 Flash output | $2.50 / 1M tokens |
| Grounding with Google Search (Gemini 2.x) | **$35 / 1,000 prompts**（每天前 1,500 次免费；单 prompt 内多次搜索算 1 次） |
| Grounding with Google Search (Gemini 3.x，2026-01-05 起) | **$14 / 1,000 search queries**（每月前 5,000 次免费；按内部搜索次数计） |
| 检索注入的 input tokens | **不收费** |
| Rate limit | 30,000 RPM/model/region；Flash Tier 3 = 10M TPM（按 30 天滚动消费分级） |

**典型时事问答单价**（实测 ~30 in / ~2100 out tokens）：$0.035 grounding + $0.005 tokens ≈ **$0.040/次**

### 4.2 Gemini Enterprise（席位订阅 + Pooled Quota）

| 版本 | 月费/席 | Assistant 查询日配额/席¹ | 月配额/席 | **配额内单价** |
|------|---------|--------------------------|-----------|---------------|
| Frontline Starter | — | 20 | 600 | — |
| Frontline | $12.50 | 40 | 1,200 | $0.01042 |
| Business | $21 | 120 | 3,600 | **$0.00583** |
| Standard | $30 | 160 | 4,800 | **$0.00625** |
| Plus | $50 | 200 | 6,000 | $0.00833 |

¹ Assistant 查询配额**已包含** Google Search / Web Grounding 用量，无需另付搜索费。

**配额机制**：
- **Pooled**：项目总配额 = 每席配额 × 席位数，全员共享
- **防滥用 Rate Limit**：池化配额的 10× 上限（非池化）
- **API Rate Limit**：`:streamAssist` 600 QPM/project/region（实测）
- **席位调整**：订阅期内**不可缩减**

**超额定价 (Overage)**：
| 项目 | 超额单价 |
|------|---------|
| Assistant 查询 | **$0.10 / 次** |
| 存储与索引 | $5 / GiB / 月 |
| 视频生成 | $0.40 / 秒 |
| Deep Research | $4.00 / 次 |

---

## 5. 性价比模型

### 5.1 三段式盈亏区间（1 个 Standard $30 席位 vs Vertex $0.040/次）

| 月调用量 Q | GE 成本 | Vertex 成本 | 胜方 |
|-----------|---------|-------------|------|
| Q < **750** | $30（席位闲置） | $0.040 × Q | **Vertex**（席位费摊不平） |
| 750 ≤ Q ≤ 4,800 | $30 | $30 ~ $192 | **GE**（最高便宜 6.4×） |
| 4,800 < Q ≤ 7,500 | $30 + (Q−4800)×$0.10 | $192 ~ $300 | **GE**（优势收窄） |
| Q > **7,500** | overage 占主导 | $0.040 × Q | **Vertex**（GE overage 是 Vertex 的 2.5×） |

### 5.2 多席位场景

按峰值预购足够席位时，GE **边际成本恒为 $0.00583~$0.00625**，比 Vertex 便宜 **6.4~6.9×**。代价是席位订阅期内不可缩减、流量低谷期沉没、突发流量只能吃 $0.10 overage。

---

## 6. 选型决策矩阵

| 场景 | 推荐 | 理由 |
|------|------|------|
| 企业内部固定员工，人均 30~160 次/天 | **GE Standard/Business** | 配额覆盖，单价 $0.006，便宜 6× |
| 程序化 API / 对外产品 / 流量波动大 | **Vertex** | 纯弹性、无席位沉没、无 overage 惩罚、600 QPM 限制不适用 |
| 低频（< 25 次/天/席） | **Vertex** | GE 席位费摊不平 |
| 延迟敏感（TTFT < 5s 硬要求） | **Vertex** | GE 实测 TTFT 慢 38%、端到端慢 86% |
| 需要图片/视频/代码执行/Deep Research 一站式 | **GE Plus** | 单端点自动编排；Vertex 需分别调 Imagen/Veo/CodeExecution 单独计费 |
| 需要 Agent 推理轨迹、session 记忆、企业合规（VPC-SC/CMEK） | **GE** | 原生支持 |
| 需要全网实时信息 + 极致成本（< 1500 次/天） | **Vertex** | Grounding 免费额度内仅付 token ≈ $0.005/次 |

---

## 7. 一句话总结

**两条路径接同一个 Google Search，回答质量无差异。** Vertex 是「快、弹性、按量」的开发者 API；GE 是「慢 1 倍、席位包月、多模态一站式」的企业员工助手。配额内 GE 便宜 6×，配额外 GE 贵 2.5×——选哪个取决于你是在**做产品**（选 Vertex）还是在**给员工配工具**（选 GE）。

---

## 附录 A：本仓库复现方法

```bash
# 1. 配置 .env（参考 .env.example）
GCP_PROJECT_ID=<your-project>
AGENT_SEARCH_ENGINE=<your-agentspace-app-id>   # 用 python scripts/list_engines.py 查询
AGENT_SEARCH_ASSISTANT=default_assistant
GEMINI_MODEL=gemini-2.5-flash

# 2. 启动代理
python app.py

# 3. 运行全量 benchmark（8 题 × 4 通道 + Judge 评分）
python benchmark.py

# 4. 查看可视化报告
streamlit run dashboard.py

# 5. 重新渲染本文档图表
python scripts/render_charts.py
```

## 附录 B：参考来源

- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Grounding with Google Search](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-google-search)
- [Gemini Enterprise StreamAssist Guide](https://docs.cloud.google.com/gemini/enterprise/docs/get-answers-from-streamassist)
- [AssistAnswer REST Reference](https://docs.cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/projects.locations.collections.engines.sessions.assistAnswers)
- [Gemini Enterprise Licenses & Pooled Quotas](https://docs.cloud.google.com/gemini/enterprise/docs/licenses)
