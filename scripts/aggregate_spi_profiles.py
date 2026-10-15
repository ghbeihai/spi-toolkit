#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPI 被测人画像汇总：从逐篇 spi-quant profile 结果汇总出 cluster_summary。

输入结构（spi-quant profile 产出）：
  {"spi": {"cluster_hits": {"C1": 0.95, ...}, "cluster_names": {"C1": "情感温暖", ...}, "top_factors": [...]}}

输出结构：
  {"cluster_summary": [{"cluster": "C1", "label": "情感温暖", "n_first": 5, "n_top5": 12, "avg_relevance": 0.85}, ...]}
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
    args = ap.parse_args()

    profiles_dir = Path(args.profiles_dir)
    if not profiles_dir.exists():
        print(f"错误：目录不存在 {profiles_dir}", file=sys.stderr)
        sys.exit(1)

    cluster_stats = defaultdict(lambda: {"n_first": 0, "n_top5": 0, "total_relevance": 0.0, "label": ""})
    n_docs = 0

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

        spi = data.get("spi", {})
        cluster_hits = spi.get("cluster_hits", {})
        cluster_names = spi.get("cluster_names", {})

        if not cluster_hits:
            continue
        n_docs += 1

        for i, (cid, weight) in enumerate(cluster_hits.items()):
            cluster_stats[cid]["n_top5"] += 1
            cluster_stats[cid]["total_relevance"] += weight
            if i == 0:
                cluster_stats[cid]["n_first"] += 1
            if not cluster_stats[cid]["label"] and cid in cluster_names:
                cluster_stats[cid]["label"] = cluster_names[cid]

    cluster_summary = []
    for cid, stats in cluster_stats.items():
        n_top5 = stats["n_top5"]
        cluster_summary.append({
            "cluster": cid,
            "label": stats["label"] or cid,
            "n_first": stats["n_first"],
            "n_top5": n_top5,
            "avg_relevance": round(stats["total_relevance"] / max(n_top5, 1), 3),
        })

    cluster_summary.sort(key=lambda x: -x["n_first"])

    result = {
        "n_docs": n_docs,
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
