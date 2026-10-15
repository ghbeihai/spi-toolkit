#!/usr/bin/env python3
"""统计 29 因子在 6694 条聚类标注中的真实聚类分布。"""
import argparse
import json
from collections import Counter
from pathlib import Path

argparse.ArgumentParser(description="统计 29 因子在 6694 条聚类标注中的真实聚类分布").parse_args()
_ASSET = Path(__file__).resolve().parent.parent / "assets" / "spi_item_clusters.json"  # 不依赖 cwd
items = json.load(open(_ASSET, encoding="utf-8"))
F29 = {
    "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness",
    "Depression", "Hypochondriasis", "Hysteria", "Psychopathic_Deviate",
    "Masculinity_Femininity", "Paranoia", "Psychasthenia", "Schizophrenia",
    "Hypomania", "Social_Introversion",
    "Behavior", "Cognition", "Environment", "Interpersonal", "Affective",
    "Machiavellianism", "Narcissism", "Psychopathy", "Sadism",
    "Competing", "Collaborating", "Compromising", "Avoiding", "Accommodating",
}

factor_cluster = {}
for it in items:
    fac = it.get("factor")
    cid = it.get("cluster_id")
    if fac in F29 and cid is not None:
        factor_cluster.setdefault(fac, Counter()).update([int(cid)])

print("=== 29 因子 → 聚类（6694 条复现标注，取最多 4 个） ===")
covered = set()
for fac in sorted(F29):
    cc = factor_cluster.get(fac)
    if not cc:
        print(f"{fac:<24} 未入库")
        continue
    top = cc.most_common(4)
    for c, n in top:
        covered.add(c)
    print(f"{fac:<24} " + ", ".join(f"C{c}×{n}" for c, n in top))

print()
print("覆盖聚类数:", len(covered))
print("未覆盖:", sorted(set(range(15)) - covered))
