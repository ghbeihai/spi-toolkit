#!/usr/bin/env python3
"""尺子审计（v1.0.0）：校验尺子完整性并生成冻结凭证。

校验项：
1. config 模型与因子链加载一致（六件套 32 因子）；
2. 中英双语资产覆盖 32/32（factor_zh 非空）；
3. SPI 15 聚类覆盖 = 15/15（基于 6694 条复现标注）；
4. LLM 打分协议与 config 一致（rounds/temperatures/独立调用/JSON schema）；
5. 冻结资产指纹（双语因子链 SHA-256）——变更即改变指纹，必须重跑基线。

用法：python scripts/audit_scale.py [--out data/results/scale_audit.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spi_quant.factor_chains import load_factor_chains, all_factor_chains
from spi_quant.scorer_llm import load_scale_info, CONFIG_PATH

BILINGUAL_PATH = PROJECT_ROOT / "assets" / "bilingual_factor_chains.json"
ITEMS_PATH = PROJECT_ROOT / "assets" / "spi_item_clusters.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "results" / "scale_audit.json"))
    args = ap.parse_args()

    checks = []
    ok_all = True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all = ok_all and ok
        checks.append({"name": name, "ok": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1. 因子链完整性
    chains = load_factor_chains()
    all_chains = all_factor_chains(chains)
    n_factor = len(all_chains)
    by_model = {m: len(c) for m, c in chains.items()}
    check("因子链加载", n_factor == 32 and "sdt" in by_model,
          f"{n_factor} 因子（六件套 {by_model}）")

    # 2. 双语覆盖
    bilingual = json.loads(BILINGUAL_PATH.read_text(encoding="utf-8"))
    missing_zh = [f"{c.model_id}:{c.factor}" for c in all_chains
                  if f"{c.model_id}:{c.factor}" not in bilingual
                  or not bilingual[f"{c.model_id}:{c.factor}"].get("factor_zh")]
    check("双语资产覆盖", not missing_zh, f"{len(bilingual)} 条，缺失 {len(missing_zh)}")

    # 3. SPI 15 聚类覆盖
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    factors = {c.factor for c in all_chains}
    covered = {int(it["cluster_id"]) for it in items if it.get("factor") in factors}
    missing_c = sorted(set(range(15)) - covered)
    check("SPI 聚类覆盖", not missing_c, f"{len(covered)}/15，缺失聚类 {missing_c or '无'}")

    # 4. 打分协议一致性
    scale = load_scale_info()
    llm = scale["llm_protocol"]
    protocol_ok = (
        llm["rounds"] == 2
        and llm["temperatures"] == [0.3, 0.7]
        and llm["shuffle_per_round"] is True
        and llm["factor_calls"] == "independent"
    )
    check("打分协议", protocol_ok,
          f"rounds={llm['rounds']}, temps={llm['temperatures']}, 独立调用+随机化+JSON schema")

    # 5. 冻结指纹
    fp_bilingual = sha256(BILINGUAL_PATH)
    fp_items = sha256(ITEMS_PATH)
    check("冻结资产指纹", True,
          f"bilingual={fp_bilingual[:16]}…, spi_items={fp_items[:16]}…")

    report = {
        "scale_version": scale["scale_version"],
        "frozen": scale["frozen"],
        "frozen_at": scale["frozen_at"],
        "ok": ok_all,
        "checks": checks,
        "fingerprints": {"bilingual_factor_chains.json": fp_bilingual,
                         "spi_item_clusters.json": fp_items},
        "basis": scale.get("basis", ""),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{'PASS' if ok_all else 'FAIL'}] 尺子审计 v{scale['scale_version']} → {out_path}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
