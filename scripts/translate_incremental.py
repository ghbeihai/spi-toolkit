#!/usr/bin/env python3
"""增量翻译：只为缺失的因子生成中文链，已有双语条目原样保留（尺子固化）。

用法：python scripts/translate_incremental.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spi_quant.factor_chains import load_factor_chains, all_factor_chains
from spi_quant.scorer_llm import FactorScorer

BILINGUAL_PATH = PROJECT_ROOT / "assets" / "bilingual_factor_chains.json"


def main() -> int:
    import argparse
    argparse.ArgumentParser(description="增量翻译因子链（需 DEEPSEEK_API_KEY，结果写入双语表）").parse_args()
    chains = load_factor_chains()
    scorer = FactorScorer(rounds=1)
    if scorer.simulated:
        print("[error] 需要 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1

    bilingual = {}
    if BILINGUAL_PATH.exists():
        bilingual = json.loads(BILINGUAL_PATH.read_text(encoding="utf-8"))

    missing = [c for c in all_factor_chains(chains) if f"{c.model_id}:{c.factor}" not in bilingual]
    if not missing:
        print(f"[ok] 无缺失条目（当前 {len(bilingual)} 条全部就绪）")
        return 0

    print(f"[plan] 仅增量翻译 {len(missing)} 条（已有 {len(bilingual)} 条不动）")
    from openai import OpenAI

    client = OpenAI(api_key=scorer.api_key, base_url=scorer.base_url, timeout=60)
    for i, chain in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] 翻译 {chain.model_id}:{chain.factor} ...", flush=True)
        prompt = (
            "把以下人格因子的五层词汇编码逐条翻译为简体中文，保持结构。\n"
            f"因子名称: {chain.factor}\n"
            f"{chain.lexical_signature}\n\n"
            "输出严格 JSON: {\"factor_zh\": \"中文因子名\", \"adjectives_zh\": [\"...\"], "
            "\"synonyms_zh\": [...], \"verbs_zh\": [...], \"nouns_zh\": [...]}"
        )
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=scorer.model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "你是人格心理学术语翻译专家，只输出 JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                bilingual[f"{chain.model_id}:{chain.factor}"] = data
                break
            except Exception as e:  # noqa: BLE001
                print(f"    retry: {e}")
                time.sleep(2)

    BILINGUAL_PATH.write_text(
        json.dumps(bilingual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] 双语资产更新：共 {len(bilingual)} 条（新增 {len(missing)}）→ {BILINGUAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
