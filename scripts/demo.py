#!/usr/bin/env python3
"""端到端 demo：文本 → 五件套画像 + SPI 聚类命中 + persona 描述。

用法：
  python scripts/demo.py [sample_file] [--out data/results/demo.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spi_quant.factor_chains import load_factor_chains, all_factor_chains
from spi_quant.scorer_llm import FactorScorer
from spi_quant.spi_mapper import SpiMapper, SpiMapperLLM
from spi_quant.persona import build_profile_summary, build_persona_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sample = sys.argv[1] if len(sys.argv) > 1 else "data/samples/sample_high_extra_dominance.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else str(PROJECT_ROOT / "data" / "results" / "demo.json")

    text = (PROJECT_ROOT / sample).read_text(encoding="utf-8").strip()
    chains = load_factor_chains()
    scorer = FactorScorer()

    if scorer.simulated:
        print("[warn] 未检测到 DEEPSEEK_API_KEY，使用模拟模式（演示流程，非真实测量）\n")

    print(f"样本: {sample}（{len(text)} 字）\n")

    # 1) LLM 零样本画像
    results = scorer.score_text(chains, text)
    print(build_profile_summary(results))

    # 2) SPI 聚类命中（默认 LLM 判聚类；无 key 时降级本地嵌入）
    spi = None
    try:
        mapper = SpiMapperLLM()
        mapper.prepare(None)
        spi = mapper.map_text(text)
        print("\n### SPI 聚类命中（LLM 判聚类）")
        for cid, w in list(spi["cluster_hits"].items())[:5]:
            print(f"- {cid} {spi['cluster_names'][cid]}（{w:.2f}）")
    except RuntimeError:
        try:
            mapper = SpiMapper()
            mapper.prepare(chains)
            spi = mapper.map_text(text)
            print("\n### SPI 聚类命中（本地嵌入，降级）")
            for cid, w in list(spi["cluster_hits"].items())[:5]:
                print(f"- {cid} {spi['cluster_names'][cid]}（{w:.2f}）")
        except RuntimeError as e:
            print(f"\n[skip] SPI 映射不可用: {e}")

    # 3) persona
    persona = build_persona_prompt(
        results, (spi or {}).get("cluster_hits"), (spi or {}).get("cluster_names")
    )

    payload = {
        "sample": sample,
        "text": text,
        "simulated": scorer.simulated,
        "scores": {k: v.to_dict() for k, v in results.items()},
        "spi": spi,
        "persona_prompt": persona,
    }
    out_path = PROJECT_ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 40)
    print(persona)
    print(f"\n[ok] 完整结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
