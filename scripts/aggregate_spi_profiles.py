#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPI 被测人画像汇总：从逐篇 spi-quant profile 结果汇总出 cluster_summary。

输入：spi-quant profile 产出的逐篇 JSON 目录
输出：spi_cluster_all.json（build_persona_generic.py 可读的 cluster_summary 结构）

用法：
  python3 aggregate_spi_profiles.py <profiles_dir> --out spi_cluster_all.json
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

    # 按 cluster → {n_first, n_top5, total_relevance} 汇总
    cluster_stats = defaultdict(lambda: {"n_first": 0, "n_top5": 0, "total_relevance": 0.0, "n_docs": 0})

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
        if not spi:
            continue

        cluster_hits = spi.get("cluster_hits", [])
        cluster_names = spi.get("cluster_names", [])
        top_factors = spi.get("top_factors", [])

        cluster_stats["_n_docs"]["n_docs"] += 1

        # 统计每篇的 top1 命中
        if cluster_hits:
            for i, hit in enumerate(cluster_hits):
                cluster_id = hit.get("cluster", f"C{i}") if isinstance(hit, dict) else str(hit)
                label = hit.get("label", "") if isinstance(hit, dict) else ""
                relevance = hit.get("relevance", 0) if isinstance(hit, dict) else 0

                cluster_stats[cluster_id]["n_top5"] += 1
                cluster_stats[cluster_id]["total_relevance"] += relevance
                if i == 0:
                    cluster_stats[cluster_id]["n_first"] += 1
                if label and not cluster_stats[cluster_id].get("label"):
                    cluster_stats[cluster_id]["label"] = label

    n_docs = cluster_stats["_n_docs"]["n_docs"]

    # 生成 cluster_summary
    cluster_summary = []
    for cluster_id, stats in cluster_stats.items():
        if cluster_id == "_n_docs":
            continue
        n_top5 = stats["n_top5"]
        cluster_summary.append({
            "cluster": cluster_id,
            "label": stats.get("label", cluster_id),
            "n_first": stats["n_first"],
            "n_top5": n_top5,
            "avg_relevance": round(stats["total_relevance"] / max(n_top5, 1), 3),
        })

    # 按 n_first 降序
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
        print(f"  {cs['label']}: first={cs['n_first']}, top5={cs['n_top5']}")


if __name__ == "__main__":
    main()
