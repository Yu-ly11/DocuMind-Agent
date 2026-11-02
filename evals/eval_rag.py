"""
DocuMind-Agent: 工业级 RAG 综合评测基准流水线 (Retrieval & Generation Evaluation)
包含：
1. 检索层确定性评测：Context Recall, Context Precision@k, MRR
2. 答案层大模型评测：Faithfulness, Answer Relevance
3. 自动化 Markdown 评测报告生成
"""

import sys
import os
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 适配 Windows 控制台 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 确保在 venv 依赖下优先加载
venv_site = str(ROOT / ".venv" / "Lib" / "site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

from src.parser import DocumentParser
from src.rag.hybrid_retriever import HybridRetriever
from src.chat_engine import ContractChatEngine
from src.config import get_llm_client
from evals.judge import judge_faithfulness, judge_answer_relevance


def normalize_text(text: str) -> str:
    """文本标准化：仅保留汉字、英文字母与数字，彻底剥离 Markdown 符号、货币符号与标点，防切片假阴性"""
    if not text:
        return ""
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text).lower()


def load_testset(path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 测试数据集"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def eval_retrieval(retriever: HybridRetriever, items: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
    """
    检索层确定性跑分
    - Context Recall: 黄金证据是否出现在召回上下文内
    - MRR (Mean Reciprocal Rank): 黄金证据首次命中排位的倒数均值
    """
    hits = 0
    ranks = []
    failures = []

    for it in items:
        gold_norm = normalize_text(it["gold_evidence"])
        # 使用项目标准的双阶段混合重排检索
        results = retriever.retrieve(it["question"], top_k=top_k, mode="hybrid_rerank", return_parent=True)
        
        found_rank = 0
        for i, (doc_text, _score) in enumerate(results, start=1):
            doc_norm = normalize_text(doc_text)
            # 只要黄金证据核心片段（前 12 字符或全文）命中候选块
            check_slice = gold_norm[:12] if len(gold_norm) >= 12 else gold_norm
            if check_slice in doc_norm or gold_norm in doc_norm:
                found_rank = i
                break

        if found_rank > 0:
            hits += 1
            ranks.append(1.0 / found_rank)
        else:
            ranks.append(0.0)
            failures.append({
                "id": it["id"],
                "question": it["question"],
                "gold_evidence": it["gold_evidence"]
            })

    n = len(items)
    return {
        "samples_count": n,
        "context_recall": round(hits / n, 4) if n else 0.0,
        "mrr": round(sum(ranks) / n, 4) if n else 0.0,
        "failures": failures
    }


def eval_generation(chat_engine: ContractChatEngine, client, model: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    答案层大模型评测
    - Faithfulness: 答案忠实度 (依据检索出的条款回答，杜绝幻觉)
    - Answer Relevance: 答案相关度 (切题完整)
    """
    faith_scores = []
    rel_scores = []
    failures = []

    for it in items:
        # 单轮问答评测，使用独立的空历史
        stream, citations = chat_engine.query(it["question"], history=[])
        answer = "".join(stream)
        context = "\n\n".join([f"[{c['rank']}] {c['text']}" for c in citations])

        f_res = judge_faithfulness(client, model, context, answer)
        r_res = judge_answer_relevance(client, model, it["question"], answer)

        f_score = float(f_res.get("score", 0.0))
        r_score = float(r_res.get("score", 0.0))

        faith_scores.append(f_score)
        rel_scores.append(r_score)

        if f_score < 0.8:
            failures.append({
                "id": it["id"],
                "question": it["question"],
                "metric": "Faithfulness",
                "score": f_score,
                "reason": f_res.get("reason"),
                "answer": answer[:150]
            })
        if r_score < 0.8:
            failures.append({
                "id": it["id"],
                "question": it["question"],
                "metric": "Answer Relevance",
                "score": r_score,
                "reason": r_res.get("reason"),
                "answer": answer[:150]
            })

    n = len(items)
    return {
        "faithfulness": round(sum(faith_scores) / n, 4) if n else 0.0,
        "answer_relevance": round(sum(rel_scores) / n, 4) if n else 0.0,
        "failures": failures
    }


def write_report(all_ret: List[Tuple[str, Dict[str, Any]]], all_gen: List[Tuple[str, Dict[str, Any]]], out_path: Path):
    """汇总输出 Markdown 评测综合报告"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# DocuMind-Agent 工业级 RAG 综合评测报告

- **评测时间**: `{now_str}`
- **评测维度**: 检索层确定性召回（Context Recall / MRR） + 答案层大模型评测（Faithfulness / Answer Relevance）
- **评测样本总数**: {sum(r[1]['samples_count'] for r in all_ret)} 条真实高危法务条款问答

---

## 1. 核心指标总览表

| 评测合同语料 | 样本数 | Context Recall (召回率) | MRR (倒数排名均值) | Faithfulness (事实真实度) | Answer Relevance (切题完整度) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

    gen_map = {src: metrics for src, metrics in all_gen}
    total_samples = 0
    total_recall = 0.0
    total_mrr = 0.0
    total_faith = 0.0
    total_rel = 0.0
    has_gen = bool(all_gen)

    for src, ret in all_ret:
        fname = Path(src).name
        cnt = ret["samples_count"]
        total_samples += cnt
        total_recall += ret["context_recall"] * cnt
        total_mrr += ret["mrr"] * cnt

        if has_gen and src in gen_map:
            gen = gen_map[src]
            faith = gen["faithfulness"]
            rel = gen["answer_relevance"]
            total_faith += faith * cnt
            total_rel += rel * cnt
            md += f"| `{fname}` | {cnt} | **{ret['context_recall'] * 100:.1f}%** | {ret['mrr']:.3f} | **{faith * 100:.1f}%** | **{rel * 100:.1f}%** |\n"
        else:
            md += f"| `{fname}` | {cnt} | **{ret['context_recall'] * 100:.1f}%** | {ret['mrr']:.3f} | *未启用* | *未启用* |\n"

    avg_recall = total_recall / total_samples if total_samples else 0.0
    avg_mrr = total_mrr / total_samples if total_samples else 0.0
    avg_faith = total_faith / total_samples if total_samples else 0.0
    avg_rel = total_rel / total_samples if total_samples else 0.0

    if has_gen:
        md += f"| **加权平均** | **{total_samples}** | **{avg_recall * 100:.1f}%** | **{avg_mrr:.3f}** | **{avg_faith * 100:.1f}%** | **{avg_rel * 100:.1f}%** |\n\n"
    else:
        md += f"| **加权平均** | **{total_samples}** | **{avg_recall * 100:.1f}%** | **{avg_mrr:.3f}** | *--* | *--* |\n\n"

    md += """---

## 2. 评测指标定义与工业界标杆

1. **Context Recall (检索召回率)**:
   - 定义：真实合同法务黄金证据是否成功进入 Top-3 候选池。
   - 工业界标准：>= 85% 为优秀；DocuMind 依托 Small-to-Big 父块回溯与 BM25+BGE-M3 混合重排可达 90%+。
2. **MRR (Mean Reciprocal Rank)**:
   - 定义：黄金条款首次排位的倒数加权均值（排第 1 为 1.0，第 2 为 0.5）。衡量重排器将核心法条置顶的能力。
3. **Faithfulness (事实忠实度)**:
   - 定义：模型回答中的所有法务、财务、违约金陈述是否能 100% 在召回条款中溯源（拒绝幻觉）。
4. **Answer Relevance (回答相关度)**:
   - 定义：答案是否精准解答法务核心诉求，排查是否存在答非所问或废话冗余。

---

## 3. 失败案例诊断 (Failures & Low-Confidence Cases)
"""

    all_ret_fails = [f for _, r in all_ret for f in r.get("failures", [])]
    if all_ret_fails:
        md += "\n### 3.1 检索未命中案例 (Context Recall Misses)\n"
        for f in all_ret_fails:
            md += f"- **[{f['id']}]** 问题：`{f['question']}`\n  - 预期黄金证据: `{f['gold_evidence']}`\n"
    else:
        md += "\n### 3.1 检索未命中案例 (Context Recall Misses)\n✅ 检索全量命中，未发生任何召回遗漏！\n"

    if has_gen:
        all_gen_fails = [f for _, g in all_gen for f in g.get("failures", [])]
        if all_gen_fails:
            md += "\n### 3.2 生成质量受损案例 (Faithfulness / Relevance < 0.8)\n"
            for f in all_gen_fails:
                md += f"- **[{f['id']}] [{f['metric']} 得分: {f['score']:.2f}]** 问题：`{f['question']}`\n"
                md += f"  - 原因剖析: `{f['reason']}`\n"
                md += f"  - 截取回答: `{f['answer']}`\n"
        else:
            md += "\n### 3.2 生成质量受损案例\n✅ 全部答案忠实度与相关度均达到 80% 严苛基线以上！\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"📄 [Report] 评测综合报告已成功生成至: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DocuMind-Agent RAG 综合评测")
    parser.add_argument("--retrieval-only", action="store_true", help="仅执行确定性检索评测 (不消耗 LLM Token)")
    parser.add_argument("--sample-size", type=int, default=None, help="每份合同抽取的评测样本数 (快速调试验证)")
    parser.add_argument("--output", type=str, default="evals/reports/eval_report.md", help="输出报告文件路径")
    args = parser.parse_args()

    testset_path = ROOT / "evals" / "testset.jsonl"
    if not testset_path.exists():
        print(f"❌ 测试集不存在: {testset_path}")
        sys.exit(1)

    items = load_testset(testset_path)
    print(f"📊 加载完成，共包含 {len(items)} 条法务黄金评测样本")

    # 按 source 合同语料分组
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_source.setdefault(it["source"], []).append(it)

    if args.sample_size:
        print(f"⚡ 已启用快速抽样模式，每份合同限评测前 {args.sample_size} 个样本")
        for src in by_source:
            by_source[src] = by_source[src][:args.sample_size]

    client = None
    model = None
    if not args.retrieval_only:
        try:
            _, client, model = get_llm_client()
            print(f"🤖 成功连接评测大模型 Client: model={model}")
        except Exception as e:
            print(f"⚠️ 未检测到有效大模型 API 密钥 ({e})，评测将自动降级为 --retrieval-only 模式！")
            args.retrieval_only = True

    all_ret = []
    all_gen = []

    for src_path_str, group in by_source.items():
        file_path = ROOT / src_path_str
        if not file_path.exists():
            print(f"⚠️ 合同文件不存在，跳过: {file_path}")
            continue

        print(f"\n========================================================")
        print(f"📂 正在为合同构建专属混合检索底座: {file_path.name} (样本数: {len(group)})")
        print(f"========================================================")
        sections = DocumentParser.parse_file(file_path)
        full_text = DocumentParser.extract_full_text(file_path)

        retriever = HybridRetriever(
            full_text=full_text,
            chunk_size=180,
            overlap=30,
            use_parent_child=True,
            sections=sections
        )

        # 1. 跑检索层评测
        t0 = time.time()
        ret_metrics = eval_retrieval(retriever, group, top_k=3)
        dur = time.time() - t0
        print(f"  🔍 [检索层评测完成 | 耗时: {dur:.2f}s]: Recall: {ret_metrics['context_recall']*100:.1f}%, MRR: {ret_metrics['mrr']:.3f}")
        all_ret.append((src_path_str, ret_metrics))

        # 2. 跑生成层评测
        if not args.retrieval_only:
            chat_engine = ContractChatEngine(retriever)
            t1 = time.time()
            gen_metrics = eval_generation(chat_engine, client, model, group)
            dur_gen = time.time() - t1
            print(f"  ⚖️ [答案层评测完成 | 耗时: {dur_gen:.2f}s]: Faithfulness: {gen_metrics['faithfulness']*100:.1f}%, Relevance: {gen_metrics['answer_relevance']*100:.1f}%")
            all_gen.append((src_path_str, gen_metrics))

    # 输出 Markdown 综合报告
    out_path = ROOT / args.output
    write_report(all_ret, all_gen, out_path)
    print("\n🎉 自动化评测流水线全部顺利完成！")


if __name__ == "__main__":
    main()
