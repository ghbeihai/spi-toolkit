#!/usr/bin/env python3
"""跨人画像比较工具：多个 32 因子画像 → 比较矩阵（热力图数据 + 偏差数据）。

用法：
  python scripts/compare_profiles.py \
    --baseline data/results/zhang_profiles/zhang_report_all.json \
    --profiles <person_a.json> <person_b.json> ... \
    --out data/results/compare/profiles_matrix.json

说明：
- 输入须为同尺子版本（scale_version 1.0.0 / 32 因子）的画像 JSON；
- baseline 可为聚合报告（含 aggregate 字段）或单篇画像（含 scores 字段）；
- profiles 同理；输出含：矩阵、偏差（vs baseline）、跨人区分度（σ）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spi_quant.factor_chains import load_factor_chains, all_factor_chains

FACTOR_ORDER = [c.factor for c in all_factor_chains(load_factor_chains())]
MODEL_OF = {}
for c in all_factor_chains(load_factor_chains()):
    MODEL_OF[c.factor] = c.model_id


def extract_scores(path: Path) -> dict:
    """从画像 JSON 提取 {factor: score}；兼容聚合报告(aggregate)与单篇(scores)。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in data:
        return {f: v["mean"] for f, v in data["aggregate"].items()}
    if "scores" in data:
        return {f: v["score"] for f, v in data["scores"].items() if not v.get("failed")}
    raise ValueError(f"无法识别画像结构: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="基准画像 JSON（如某作者聚合画像）")
    ap.add_argument("--profiles", nargs="+", required=True, help="待比较画像 JSON（≥1 个）")
    ap.add_argument("--names", nargs="*", help="展示名（与 profiles 一一对应，缺省用文件名）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "results" / "compare" / "profiles_matrix.json"))
    args = ap.parse_args()

    baseline = extract_scores(Path(args.baseline))
    names = args.names or [Path(p).stem for p in args.profiles]
    if len(names) != len(args.profiles):
        print("[warn] --names 数量与 --profiles 不一致，用文件名")
        names = [Path(p).stem for p in args.profiles]

    prof_scores = []
    for p in args.profiles:
        s = extract_scores(Path(p))
        missing = [f for f in FACTOR_ORDER if f not in s]
        if missing:
            print(f"[error] {p} 缺因子: {missing}（请确保同尺子版本 32 因子）")
            return 1
        prof_scores.append(s)

    persons = [("基准", baseline)] + list(zip(names, prof_scores))

    # 矩阵
    matrix = []
    deviations = []
    discrim = []
    for fac in FACTOR_ORDER:
        row = {"factor": fac, "model": MODEL_OF[fac], "values": {}}
        for name, s in persons:
            row["values"][name] = round(s[fac], 2)
        matrix.append(row)
        # 偏差 vs 基准
        d = {name: round(s[fac] - baseline[fac], 2) for name, s in persons[1:]}
        deviations.append({"factor": fac, "model": MODEL_OF[fac], "dev": d})
        # 跨人区分度（含基准在内的 σ）
        vals = [s[fac] for _, s in persons]
        discrim.append({"factor": fac, "model": MODEL_OF[fac],
                        "sigma": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0})

    out = {
        "scale_version": "1.0.0",
        "persons": [n for n, _ in persons],
        "factors": FACTOR_ORDER,
        "model_of": MODEL_OF,
        "matrix": matrix,
        "deviations": deviations,
        "discrimination": sorted(discrim, key=lambda x: -x["sigma"]),
        "baseline_label": "基准",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ok] 比较矩阵已生成: {out_path}")
    print(f"人物: {', '.join(n for n, _ in persons)}")
    top = sorted(discrim, key=lambda x: -x["sigma"])[:5]
    print("区分度最高因子: " + ", ".join(f"{t['factor']} σ={t['sigma']}" for t in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
