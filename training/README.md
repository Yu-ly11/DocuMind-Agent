# DocuMind-Agent 算法手撕练习与核心逻辑验证集 (Training & Benchmarks)

本目录为开发者闭卷练习与核心底层算法的独立单测验证集，包含生产代码中抽取的关键算子：

## 文件清单
1. **`practice_exam.py`**: 三大核心工业级算法手撕练习（RRF 倒数排名融合、Small-to-Big 拓扑去重、Critic 事实核查防死循环状态机），内置完整断言自检套件。
   - 验证运行：`python training/practice_exam.py`
2. **`practice_langgraph.py`**: LangGraph 节点编排与多跳循环攻坚练习。
3. **`practice_core.py`**: 底层检索引擎与向量库接入原理解析脚本。
4. **`test_env.py`**: 本地环境模型连接与依赖就绪度自检探针。
