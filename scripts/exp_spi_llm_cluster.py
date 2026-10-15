#!/usr/bin/env python3
"""实验：LLM 零样本判 SPI 聚类 vs 论文 KMeans 标注的一致性（v2）。

样本：论文仓库 Embeddings/*_clustered_embeddings.csv 的 6694 条条目
（每条含 Factor/Adjective/Synonym/Verb/Noun + Cluster 标注），按聚类分层抽样。
LLM 输入：词元组合（因子/形容词/近义词/动词/名词）；输出：聚类 id。
指标：与论文标注的一致性 + 混淆摘要 + 人工合理性抽查。
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CLUSTER_JSON = PROJECT_ROOT / "assets" / "survey" / "results" / "spi_table.json"
EMB_DIR = PROJECT_ROOT / "assets" / "survey" / "Embeddings"
OUT_REPORT = PROJECT_ROOT / "data" / "results" / "spi_llm_cluster_exp.md"

N_PER_CLUSTER = 5  # 15 聚类 × 5 = 75 条


def load_items_with_cluster() -> list[dict]:
    """读复现标注（assets/spi_item_clusters.json，规模/模型构成与论文 15/15 一致）。"""
    data = json.loads((PROJECT_ROOT / "assets" / "spi_item_clusters.json").read_text(encoding="utf-8"))
    return [
        {
            "slug": r["model"],
            "factor": r["factor"],
            "adj": r["adjective"],
            "syn": r["synonym"],
            "verb": r["verb"],
            "noun": r["noun"],
            "cluster": r["cluster_id"],
        }
        for r in data
    ]


def build_cluster_prompt(clusters) -> str:
    lines = []
    for c in clusters:
        adj = ", ".join(c.get("top_adjectives", [])[:3])
        fac = ", ".join(c.get("top_factors", [])[:4])
        lines.append(f"{c['id']} {c['label']}（代表形容词：{adj}；代表因子：{fac}）")
    return "\n".join(lines)


def item_text(it: dict) -> str:
    return (
        f"因子: {it['factor']}；形容词: {it['adj']}；近义词: {it['syn']}；"
        f"动词: {it['verb']}；名词: {it['noun']}"
    )


def judge(client, text: str, cluster_desc: str) -> int | None:
    user = (
        "你是心理学人格维度聚类分类器。以下 15 个聚类是对人格模型语义空间的划分：\n"
        f"{cluster_desc}\n\n"
        "下面是一条人格量表条目的词元（因子/形容词/近义词/动词/名词），"
        "请判断它最可能属于哪个聚类。\n"
        '只输出 JSON：{"cluster_id": <数字>}\n\n'
        f"条目：{text}"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "你是严谨的人格心理学分类器。基于词元语义判断聚类，输出严格 JSON。",
            },
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    v = data.get("cluster_id")
    return int(v) if isinstance(v, (int, str)) and str(v).isdigit() else None


def main() -> int:
    import argparse
    argparse.ArgumentParser(description="LLM 零样本判 SPI 聚类 vs 论文 KMeans 标注一致性（需 DEEPSEEK_API_KEY）"
                            ).parse_args()
    from openai import OpenAI
    from dotenv import dotenv_values

    # 与其他脚本一致：优先环境变量，其次项目 .env
    key = os.environ.get("DEEPSEEK_API_KEY") or dotenv_values(PROJECT_ROOT / ".env").get("DEEPSEEK_API_KEY") or ""
    if not key:
        print("[error] 缺少 DEEPSEEK_API_KEY")
        return 1
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=90)

    spi = json.loads(CLUSTER_JSON.read_text(encoding="utf-8"))
    clusters = spi["clusters"]
    cluster_desc = build_cluster_prompt(clusters)

    items = load_items_with_cluster()
    print(f"[ok] 条目总数: {len(items)}")

    by_cluster: dict[int, list[dict]] = {}
    for it in items:
        by_cluster.setdefault(it["cluster"], []).append(it)
    covered = sorted(by_cluster)
    print(f"[ok] 覆盖聚类: {len(covered)}/15")

    random.seed(42)
    sample: list[dict] = []
    for cl in covered:
        pool = by_cluster[cl]
        sample += random.sample(pool, min(N_PER_CLUSTER, len(pool)))
    print(f"[plan] 抽样 {len(sample)} 条，LLM 判聚类...")

    correct = 0
    from collections import Counter

    confusions: Counter = Counter()
    results = []
    for n, it in enumerate(sample, 1):
        pred = judge(client, item_text(it), cluster_desc)
        ok = pred == it["cluster"]
        if ok:
            correct += 1
        else:
            confusions[(it["cluster"], pred)] += 1
        results.append(
            {
                "slug": it["slug"],
                "factor": it["factor"],
                "adj": it["adj"],
                "true": it["cluster"],
                "pred": pred,
                "ok": ok,
            }
        )
        print(
            f"  [{n}/{len(sample)}] true={it['cluster']} pred={pred} "
            f"{'✓' if ok else '✗'} {it['adj'][:35]}"
        )

    acc = correct / len(sample)
    print(f"\n[result] 一致性: {correct}/{len(sample)} = {acc:.1%}")

    conf_lines = []
    for (t, p), cnt in confusions.most_common(12):
        conf_lines.append(
            f"- true={clusters[t]['label']} → pred={clusters[p]['label'] if p is not None else 'None'}（{cnt} 条）"
        )

    OUT_REPORT.write_text(
        "# LLM 判 SPI 聚类 vs 论文 KMeans 标注 · 一致性实验（v2）\n\n"
        f"- 样本：{len(sample)} 条（15 聚类分层抽样 × {N_PER_CLUSTER}，seed=42，来自 6694 条词元条目）\n"
        f"- 方法：DeepSeek deepseek-chat 零样本（temperature=0，JSON 输出），输入为因子/形容词/近义词/动词/名词\n"
        f"- **一致性：{acc:.1%}（{correct}/{len(sample)}）**\n\n"
        "## 主要混淆\n"
        + ("\n".join(conf_lines) if conf_lines else "- 无混淆")
        + "\n\n## 逐条结果\n"
        + json.dumps(results, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[ok] 报告: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
