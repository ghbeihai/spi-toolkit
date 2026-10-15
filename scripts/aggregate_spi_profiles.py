#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPI 被测人画像汇总：从逐篇 spi-quant profile 结果汇总出 cluster_summary。

输入结构（spi-quant profile 产出）：
  {"spi": {"cluster_hits": {"C1": 0.95, ...}, "cluster_names": {"C1": "情感温暖", ...}, "top_factors": [...]}}

输出结构：
  {"cluster_summary": [{"cluster": "C1", "label": "情感温暖", "n_first": 5, "n_top5": 12,
                        "avg_relevance": 5.6, "avg_hit_norm": 0.85}, ...]}

口径（v2）：
- 每篇按 LLM 原始契合度（top_clusters[].score，1-7）降序只取前 --top-k（默认 5）个计 n_top5；
- avg_relevance = 命中篇的原始契合度均值（1-7，不再是逐篇 max 归一后的值）；
- avg_hit_norm = 逐篇归一命中（cluster_hits，最大=1.0）均值，保留旧口径供对照；
- failed=True 的篇跳过并计数。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="SPI 被测人画像汇总")
    ap.add_argument("profiles_dir", help="spi-quant profile 产出的逐篇 JSON 目录")
    ap.add_argument("--out", default="spi_cluster_all.json", help="输出文件名")
    ap.add_argument("--top-k", type=int, default=5, help="每篇计入 n_top5 的聚类数（默认 5）")
    args = ap.parse_args()

    profiles_dir = Path(args.profiles_dir)
    if not profiles_dir.exists():
        print(f"错误：目录不存在 {profiles_dir}", file=sys.stderr)
        sys.exit(1)

    cluster_stats = defaultdict(lambda: {"n_first": 0, "n_top5": 0, "total_relevance": 0.0,
                                         "total_norm": 0.0, "label": ""})
    n_docs = 0
    n_failed = 0

    files = list(profiles_dir.glob("*.json"))
    if not files:
        print(f"错误：目录下无 JSON 文件 {profiles_dir}", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 跳过 {f.name}: {e}", file=sys.stderr)
            continue

        spi = data.get("spi", {}) or {}
        if spi.get("failed"):
            n_failed += 1
            continue
        cluster_hits = spi.get("cluster_hits", {}) or {}
        cluster_names = spi.get("cluster_names", {}) or {}
        tops = spi.get("top_clusters") or []
        if tops:  # 原始契合度（1-7）
            ranked = [(c["cluster"], float(c.get("score", 0)), c.get("label", "")) for c in tops]
        else:     # 旧 profile / local 模式：只有归一命中
            ranked = [(cid, float(w) * 7.0, "") for cid, w in cluster_hits.items()]
        ranked.sort(key=lambda x: -x[1])
        ranked = ranked[: args.top_k]
        if not ranked:
            continue
        n_docs += 1

        for i, (cid, rel, label) in enumerate(ranked):
            st = cluster_stats[cid]
            st["n_top5"] += 1
            st["total_relevance"] += rel
            st["total_norm"] += float(cluster_hits.get(cid, rel / 7.0))
            if i == 0:
                st["n_first"] += 1
            if not st["label"]:
                st["label"] = cluster_names.get(cid) or label

    cluster_summary = []
    for cid, stats in cluster_stats.items():
        n_top5 = stats["n_top5"]
        cluster_summary.append({
            "cluster": cid,
            "label": stats["label"] or cid,
            "n_first": stats["n_first"],
            "n_top5": n_top5,
            "avg_relevance": round(stats["total_relevance"] / max(n_top5, 1), 3),
            "avg_hit_norm": round(stats["total_norm"] / max(n_top5, 1), 3),
        })

    cluster_summary.sort(key=lambda x: (-x["n_first"], -x["n_top5"]))
    if n_failed:
        print(f"[warn] 跳过 {n_failed} 篇 SPI 判聚类失败的 profile", file=sys.stderr)
    if not n_docs:
        print("错误：没有可汇总的有效 profile", file=sys.stderr)
        sys.exit(1)

    result = {
        "n_docs": n_docs,
        "n_failed": n_failed,
        "top_k": args.top_k,
        "cluster_summary": cluster_summary,
        "note": "由 aggregate_spi_profiles.py 从逐篇 profile 汇总",
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总完成：{n_docs} 篇 → {len(cluster_summary)} 个聚类 → {out_path}")
    for cs in cluster_summary[:5]:
        print(f"  {cs['label']}: first={cs['n_first']}, top5={cs['n_top5']}, rel={cs['avg_relevance']}")


if __name__ == "__main__":
    main()
