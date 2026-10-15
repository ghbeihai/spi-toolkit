"""四项验证：schema 合法率、重测稳定性、判别效度、SPI 一致性。

产出：results/eval_report.md + results/eval_report.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from .factor_chains import load_factor_chains
from .scorer_llm import FactorScorer, FactorScore, save_profile
from .spi_mapper import SpiMapper

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def schema_legal_rate(results: Dict[str, FactorScore]) -> float:
    """JSON 结构化评分成功率：raw_scores 非空且分数在 1-7 内的因子占比。"""
    ok = 0
    total = 0
    for s in results.values():
        total += 1
        if s.raw_scores and all(1 <= x <= 7 for x in s.raw_scores):
            ok += 1
    return ok / total if total else 0.0


def rerun_stability(scorer: FactorScorer, chains, text: str, runs: int = 3) -> dict:
    """同一文本重复 runs 次画像，评估因子分数的波动。"""
    profiles: List[Dict[str, float]] = []
    for i in range(runs):
        res = scorer.score_text(chains, text, progress=False)
        profiles.append({k: v.score for k, v in res.items()})
    # 以首轮为基准，平均绝对差
    base = profiles[0]
    mad = []
    for other in profiles[1:]:
        diffs = [abs(base[k] - other[k]) for k in base if k in other]
        mad.append(statistics.mean(diffs) if diffs else 0.0)
    # 因子级波动（各因子跨 runs 的标准差均值）
    factor_sd = []
    for k in base:
        vals = [p[k] for p in profiles if k in p]
        factor_sd.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return {
        "runs": runs,
        "mean_abs_diff_vs_first": round(statistics.mean(mad), 3) if mad else None,
        "mean_factor_sd": round(statistics.mean(factor_sd), 3),
        "max_factor_sd": round(max(factor_sd), 3),
    }


def discriminative_validity(
    scorer: FactorScorer,
    chains,
    manifest: Path,
    samples_dir: Path,
) -> dict:
    """判别效度：对每个样本断言其预期高因子分数应 >= 阈值。"""
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    checks = []
    for item in manifest_data["samples"]:
        path = samples_dir / item["file"]
        text = path.read_text(encoding="utf-8").strip()
        res = scorer.score_text(chains, text, progress=False)
        for expect in item["expect_high"]:
            score = res[expect].score
            checks.append({
                "sample": item["file"],
                "factor": expect,
                "score": round(score, 2),
                "passed": score >= 4.5,
            })
    passed = sum(1 for c in checks if c["passed"])
    return {
        "total_checks": len(checks),
        "passed": passed,
        "pass_rate": round(passed / len(checks), 3) if checks else 0.0,
        "checks": checks,
    }


def spi_consistency(
    scorer: FactorScorer,
    chains,
    mapper: SpiMapper,
    manifest: Path,
    samples_dir: Path,
    top_factors: int = 3,
) -> dict:
    """SPI 一致性：LLM 高分因子的聚类 vs 本地 SPI 映射 top 因子聚类重合度。"""
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for item in manifest_data["samples"]:
        text = (samples_dir / item["file"]).read_text(encoding="utf-8").strip()
        res = scorer.score_text(chains, text, progress=False)
        llm_top = sorted(res.values(), key=lambda s: -s.score)[:top_factors]
        llm_factors = {(s.model_id, s.factor) for s in llm_top}
        spi = mapper.map_text(text, top_k=top_factors)
        spi_factors = {(tf["model"], tf["factor"]) for tf in spi["top_factors"]}
        overlap = len(llm_factors & spi_factors)
        rows.append({
            "sample": item["file"],
            "overlap": overlap,
            "max_possible": top_factors,
            "llm_top": [f"{s.factor}({s.score:.1f})" for s in llm_top],
            "spi_top": [f"{tf['factor']}({tf['similarity']:.2f})" for tf in spi["top_factors"]],
        })
    avg = statistics.mean(r["overlap"] / r["max_possible"] for r in rows)
    return {"avg_overlap_ratio": round(avg, 3), "rows": rows}


def run_full_eval(scorer: FactorScorer, chains, mapper, samples_dir: Path, out_dir: Path) -> dict:
    """跑完整四项验证并写报告。"""
    samples_dir = samples_dir or PROJECT_ROOT / "data" / "samples"
    manifest = samples_dir / "samples_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1+2：用第一个样本做 schema 与稳定性
    sample_files = [p for p in sorted(samples_dir.glob("*.txt"))]
    if not sample_files:
        raise FileNotFoundError("无样本文本")
    first_text = sample_files[0].read_text(encoding="utf-8").strip()

    schema = {"sample": sample_files[0].name}
    stability = {}
    print("[eval] 1/4 schema 合法率 + 2/4 重测稳定性（需多次调用 LLM，耗时较长）")
    runs = 3
    results_all: List[Dict[str, FactorScore]] = []
    for i in range(runs):
        results_all.append(scorer.score_text(chains, first_text, progress=(i == 0)))
    schema["legal_rate"] = round(schema_legal_rate(results_all[0]), 3)
    # 用已有 runs 计算稳定性
    base = results_all[0]
    mad, f_sd = [], []
    for other in results_all[1:]:
        diffs = [abs(base[k].score - other[k].score) for k in base if k in other]
        mad.append(statistics.mean(diffs) if diffs else 0)
    for k in base:
        vals = [p[k].score for p in results_all if k in p]
        f_sd.append(statistics.stdev(vals) if len(vals) > 1 else 0)
    stability = {
        "runs": runs,
        "mean_abs_diff_vs_first": round(statistics.mean(mad), 3) if mad else None,
        "mean_factor_sd": round(statistics.mean(f_sd), 3),
        "max_factor_sd": round(max(f_sd), 3),
    }

    print("[eval] 3/4 判别效度")
    disc = discriminative_validity(scorer, chains, manifest, samples_dir)

    print("[eval] 4/4 SPI 一致性（本地嵌入）")
    spi = spi_consistency(scorer, chains, mapper, manifest, samples_dir)

    report = {
        "schema_legal_rate": schema,
        "rerun_stability": stability,
        "discriminative_validity": disc,
        "spi_consistency": spi,
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# SPI 五件套人格量化模型 · 评估报告",
        "",
        f"## 1. Schema 合法率（样本: {schema['sample']}）",
        f"- 结构化评分成功率：**{schema['legal_rate']:.1%}**",
        "",
        "## 2. 重测稳定性",
        f"- 与首轮平均绝对差：{stability.get('mean_abs_diff_vs_first')}",
        f"- 因子分数跨轮标准差均值：{stability['mean_factor_sd']}，最大：{stability['max_factor_sd']}",
        "",
        "## 3. 判别效度",
        f"- 断言通过率：**{disc['pass_rate']:.1%}**（{disc['passed']}/{disc['total_checks']}）",
        "",
        "| 样本 | 因子 | 分数 | 通过(≥4.5) |",
        "|---|---|---|---|",
    ]
    for c in disc["checks"]:
        md.append(f"| {c['sample']} | {c['factor']} | {c['score']} | {'✅' if c['passed'] else '❌'} |")
    md += [
        "",
        "## 4. SPI 一致性（LLM 高分因子 vs 本地嵌入命中）",
        f"- 平均重合率：**{spi['avg_overlap_ratio']:.1%}**",
        "",
        "| 样本 | 重合 | LLM top | SPI top |",
        "|---|---|---|---|",
    ]
    for r in spi["rows"]:
        md.append(f"| {r['sample']} | {r['overlap']}/{r['max_possible']} | {'、'.join(r['llm_top'])} | {'、'.join(r['spi_top'])} |")
    md.append("")
    (out_dir / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] 评估报告: {out_dir / 'eval_report.md'}")
    return report
