# DocuMind-Agent: 工业级逆向单步调试实战指南

本指南供你在 VS Code 中通过 **单步断点调试（Step-by-Step Debugging）**，亲眼透视整个 LangGraph 5 节点状态机在面对真实 5,067 字合同时的内部数据流转。

---

## 一、断点打在何处？（核心观测点位）

打开项目中的 [`src/graph_agent.py`](../src/graph_agent.py)，在行号左侧单击鼠标左键，打上 **红点（断点）**：

| 节点 / 阶段 | 推荐断点位置 | 核心观察目标与变量 |
| :--- | :--- | :--- |
| **Node 1: Task Planner** | `task_planner_node` 方法入口与返回行 | 观察大模型如何将模糊指令拆解为 `state['plan_tasks']` 结构体列表 |
| **Node 2: Hybrid Retriever** | `hits = self.retriever.retrieve(...)` 之后 | 观察子块是如何高精捕捞、父块是如何完整回溯，以及检索置信度得分 `score` |
| **Node 3: Penalty Calculator** | `calc_penalty_risk(...)` 执行处 | 观察参数抽取结果与本地 Python 算子出具的 `penalty_cert` 确定性凭据 |
| **Node 4: Clause Evaluator** | `self.retriever.get_chunk_location(...)` 之后 | 观察**真实物理行号**是如何硬编码绑定到 `issue_dict['evidence']` 中的 |
| **Node 5: Critic 反思检验** | `critic_node` 的 `is_grounded` 判断处 | 观察可溯源性检验逻辑是否真实在合同全文中检索匹配摘录，观察是否触发重试 |
| **Router 条件路由** | `critic_decision` 函数入口 | 观察当前状态决定走向 `hybrid_retriever`（循环重试）还是 `report_aggregator` |

---

## 二、如何启动 VS Code 调试？

1. 在 VS Code 中打开 [`src/graph_agent.py`](../src/graph_agent.py)；
2. 按键盘快捷键 **`F5`**（或点击左侧导航栏的“运行和调试”小虫子图标 ➡️ 点击“运行和调试”）；
3. 选择 **`Python 调试程序: 当前文件`**；
4. 程序启动后，会在第一个断点处自动暂停（高亮显示黄色箭头）。

---

## 三、调试时按键操作技巧

- **F10（单步跳过）**：执行当前行，停在下一行代码；
- **F11（单步进入）**：如果当前行是一个函数调用（如调用了 `retriever.retrieve`），按 F11 会钻进函数内部查看执行细节；
- **Shift + F11（单步跳出）**：退出当前子函数，返回上一层；
- **F5（继续执行）**：一直运行直到下一个断点。

---

## 四、调试控制台（Debug Console）实时查验技巧

当程序停在断点处时，点击 VS Code 底部的 **调试控制台（Debug Console）** 标签页，你可以直接在输入框敲入 Python 表达式查看实时变量：

```python
# 1. 查看任务规划列表
state["plan_tasks"]

# 2. 查看当前检索命中的条款及得分
state["retrieved_clauses"][0]["score"]
state["retrieved_clauses"][0]["content"]

# 3. 验证物理行号是否真实存在于原文中
loc = self.retriever.get_chunk_location("乙方在签署本协议前已经独立研发")
print(loc)  # 应输出 {'line_start': 89, 'line_end': 89, 'page': 1, 'status': 'located', 'verification_status': <VerificationStatus.VERIFIED: 'verified'>}
```
