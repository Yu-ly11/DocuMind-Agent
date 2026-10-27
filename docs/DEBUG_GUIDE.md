# DocuMind-Agent: VS Code 运行与单步调试指南

面向本项目在 VS Code 中的**日常运行**与**断点调试**。所有命令均在项目根目录 `DocuMind-Agent/` 下执行。

---

## 一、一次性准备（3 步，只需做一次）

1. **选择解释器**：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `.\venv\Scripts\python.exe`
   （仓库的 `.vscode/settings.json` 已把默认解释器指向它，若仍提示找不到模块，就是这步没生效。）
2. **准备密钥**：确认根目录存在 `.env`；没有则 `Copy-Item .env.example .env` 后填入 `OPENAI_API_KEY`。
3. **打开终端**：`Ctrl+`` 打开集成终端，应看到提示符前带 `(.venv)`。

---

## 二、三种启动方式

### 方式 A：终端直接跑（最快，日常用这个）

```powershell
# 1) 交互式工作台（审计报告 + Copilot 问答）
streamlit run app.py
# 浏览器打开 http://localhost:8501

# 2) FastAPI 微服务
uvicorn src.api:app --host 127.0.0.1 --port 8000
# 文档 http://127.0.0.1:8000/docs
```

终端里 `Ctrl+C` 停止服务。

### 方式 B：F5 断点调试（仓库已内置 `.vscode/launch.json`）

`Ctrl+Shift+D` 打开「运行和调试」→ 顶部下拉框选择配置 → `F5`：

| 配置 | 作用 |
| --- | --- |
| ① Streamlit 工作台 (app.py) | 调试 UI 与两条业务链路，断点会真实命中 |
| ② FastAPI 微服务 (src/api.py) | 调试接口与 SSE 事件流 |
| ③ 证据链 Δ=0 测试 | 单步看 `get_chunk_location` 如何反查物理行号 |
| ④ PDF 解析 / 扫描件熔断测试 | 看表格还原、页眉剔除与 OCR 降级 |
| ⑤ API 端点测试 | 调试上传 → 解析 → 报告链路 |
| ⑥ 检索消融实验 | 单步看 BM25 / 向量 / RRF / 精排四路融合 |
| ⑦ 算法手撕自测（免密钥） | 不联网验证 RRF、父子块、Critic 路由 |
| ⑧ 调试当前 Python 文件 | 临时脚本用 |

> 若提示 `Configured debug type 'debugpy' is not supported`，把 `launch.json` 里的 `"type": "debugpy"` 全部换成 `"type": "python"`（旧版 Python 扩展）。

### 方式 C：只跑测试验证环境

```powershell
python training/practice_exam.py     # 免密钥，纯算法断言
python tests/test_evidence_anchor.py # 证据坐标 + Fail-Closed（需 Embedding 密钥）
python tests/test_pdf_parser.py
python tests/test_api.py
python tests/eval_benchmark.py       # 检索消融，输出 Hit@k / NDCG / MRR
```

---

## 三、断点打在何处（核心观测点）

打开 `src/graph_agent.py`，按行号在左侧单击打红点：

| 节点 | 推荐断点 | 观察目标 |
| --- | --- | --- |
| Node 1 Task Planner | `task_planner_node`（约 L77） | `state["plan_tasks"]` 如何被拆成多个审查议题 |
| Node 2 Hybrid Retriever | `retrieve(...)` 之后（约 L135） | 召回条数、`score`、父块是否正确回填 |
| Node 3 Penalty Calculator | `calc_penalty_risk(...)`（约 L194） | 参数抽取结果与本地算子出具的 `penalty_cert` |
| Node 4 Clause Evaluator | `get_chunk_location(...)` 之后（约 L262） | 物理行号/页码如何写入 `issue["evidence"]` |
| Node 5 Critic | `is_grounded = ...`（约 L311） | 证据是否真的在原文中出现；何时触发重试 |
| 条件路由 | `critic_decision`（约 L545） | 走向 `hybrid_retriever`（循环）还是 `report_aggregator` |
| Node 6 Report Aggregator | `report_aggregator_node`（约 L404） | 安全闸门如何把未闭环项抬成"至少中风险" |

行号会随代码改动漂移，以函数名为准（`Ctrl+F` 搜函数名更快）。

---

## 四、调试快捷键

- `F10` 单步跳过 · `F11` 单步进入 · `Shift+F11` 跳出 · `F5` 继续到下一个断点
- `Ctrl+Shift+F5` 重启调试 · `Shift+F5` 停止

## 五、Debug Console 常用表达式

停在断点时，在底部「调试控制台」直接输入：

```python
state["plan_tasks"]                         # 议题拆解结果
state["retrieved_clauses"][0]["score"]      # 当前 Top1 检索得分
state["retrieved_clauses"][0]["content"]    # 命中的父块内容

loc = self.retriever.get_chunk_location("乙方在签署本协议前已经独立研发")
loc["line_start"], loc["verification_status"]   # 行号是否与原文一致、是否 VERIFIED
```

> 上面的行号/文本只是示例，请替换成你当前样本合同里的真实片段。

---

## 六、调试 Streamlit 的注意事项

1. **断点会冻结页面**：命中断点时浏览器会"转圈"，这是正常的，`F5` 继续即可。
2. **改完代码先重启**：Streamlit 会热重载，但调试会话不会自动重启；改逻辑后按 `Ctrl+Shift+F5`。
3. **不要在 `st.cache` 边界内打太多断点**：高频 rerun 会反复命中，容易误以为死循环。

---

## 七、常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `ModuleNotFoundError: streamlit` / `langgraph` | 解释器不是 `.venv`，回到「一次性准备」第 1 步 |
| `未检测到有效的大模型 API 密钥` | `.env` 缺失或键名写错。项目各模块在导入时用 `python-dotenv` 自动加载**根目录**的 `.env`，所以文件必须放在项目根目录 |
| `Address already in use` | 端口被占：`streamlit run app.py --server.port 8502` 或 uvicorn `--port 8001` |
| 首次上传扫描件很慢 | 本地 RapidOCR 首次加载模型 + 逐页识别，之后走磁盘缓存会秒级完成 |
| 控制台中文乱码 | 已在 `settings.json` 设置 `PYTHONUTF8=1`；仍乱码则 `chcp 65001` |
