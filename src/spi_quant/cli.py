"""命令行入口。

用法：
  spi-quant profile --file <text.txt> [--out results/profile.json] [--rounds 2] [--no-shuffle] [--spi-mode llm|local]
  spi-quant spi     --file <text.txt> [--mode llm|local]   # 仅 SPI 聚类命中
  spi-quant eval    [--samples-dir data/samples] [--out results]   # 四项验证
  spi-quant translate                              # 生成中英双语因子链资产（需 API key）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .factor_chains import load_factor_chains, all_factor_chains
from .scorer_llm import FactorScorer, save_profile, load_scale_info
from .spi_mapper import SpiMapper, SpiMapperLLM
from .persona import build_profile_summary, build_persona_prompt
from .validate import run_full_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_mapper(mode: str):
    """按模式构建 SPI 映射器。mode: llm（默认，推荐）| local（离线嵌入）。"""
    if mode == "llm":
        return SpiMapperLLM()
    if mode == "local":
        return SpiMapper()
    raise ValueError(f"未知 SPI 模式: {mode}（可选 llm/local）")


def cmd_profile(args) -> int:
    text = Path(args.file).read_text(encoding="utf-8").strip()
    if len(text) < 30:
        print("[warn] 文本过短（<30 字），画像置信度会很低", file=sys.stderr)

    chains = load_factor_chains()
    scorer = FactorScorer(rounds=args.rounds)
    if scorer.simulated:
        print("[warn] 未检测到 DEEPSEEK_API_KEY，使用模拟模式（结果仅供流程验证）", file=sys.stderr)

    print(f"[profile] 分析文本 {len(text)} 字，共 {len(all_factor_chains(chains))} 个因子 × {args.rounds} 轮")
    results = scorer.score_text(chains, text, shuffle=not args.no_shuffle)

    # SPI 聚类映射（默认 LLM 判聚类）
    mapper = build_mapper(args.spi_mode)
    try:
        mapper.prepare(chains)
        spi = mapper.map_text(text)
    except RuntimeError as e:
        print(f"[warn] SPI 映射跳过: {e}", file=sys.stderr)
        spi = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "simulated": scorer.simulated,
        "scale": load_scale_info(),
        "scores": {k: v.to_dict() for k, v in results.items()},
        "spi": spi,
        "persona_prompt": build_persona_prompt(
            results, (spi or {}).get("cluster_hits"), (spi or {}).get("cluster_names")
        ),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(build_profile_summary(results, (spi or {}).get("top_factors")))
    print(f"\n[ok] 画像已保存: {out_path}")
    return 0


def cmd_spi(args) -> int:
    text = Path(args.file).read_text(encoding="utf-8").strip()
    mapper = build_mapper(args.mode)
    try:
        mapper.prepare(None)
        result = mapper.map_text(text, top_k=args.top_k)
    except RuntimeError as e:
        print(f"[error] SPI 映射失败: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args) -> int:
    samples_dir = Path(args.samples_dir) if args.samples_dir else (PROJECT_ROOT / "data" / "samples")
    out_dir = Path(args.out) if args.out else (PROJECT_ROOT / "data" / "results")
    chains = load_factor_chains()
    scorer = FactorScorer(rounds=args.rounds)
    if scorer.simulated:
        print("[warn] 未检测到 DEEPSEEK_API_KEY，评估将使用模拟模式（无判别意义）", file=sys.stderr)
    mapper = SpiMapper()
    try:
        mapper.prepare(chains)
    except RuntimeError as e:
        print(f"[warn] SPI 一致性检查跳过: {e}", file=sys.stderr)
        mapper = None
    run_full_eval(scorer, chains, mapper, samples_dir, out_dir)
    return 0


def cmd_translate(args) -> int:
    """生成中英双语因子链资产（DeepSeek 翻译）。"""
    from .factor_chains import load_factor_chains, all_factor_chains

    chains = load_factor_chains()
    scorer = FactorScorer(rounds=1)
    if scorer.simulated:
        print("[error] 翻译需要 DEEPSEEK_API_KEY（环境变量或 .env），当前未配置", file=sys.stderr)
        return 1

    import time
    from openai import OpenAI

    client = OpenAI(api_key=scorer.api_key, base_url=scorer.base_url, timeout=60)
    bilingual: dict = {}
    total = len(all_factor_chains(chains))
    for i, chain in enumerate(all_factor_chains(chains), 1):
        print(f"  [{i}/{total}] 翻译 {chain.factor} ...", flush=True)
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
            except Exception as e:
                print(f"    retry: {e}")
                time.sleep(2)

    out = PROJECT_ROOT / "assets" / "bilingual_factor_chains.json"
    out.write_text(json.dumps(bilingual, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 中英双语因子链已生成: {out}（{len(bilingual)} 条）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="spi-quant", description="SPI 五件套人格量化模型")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="对一段文本输出完整人格画像")
    p.add_argument("--file", required=True)
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "results" / "profile.json"))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--spi-mode", choices=["llm", "local"], default="llm",
                   help="SPI 聚类映射模式：llm=DeepSeek 判聚类（推荐，77.3% 一致性）；local=本地嵌入（离线）")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("spi", help="仅 SPI 聚类命中（默认 LLM 判聚类）")
    p.add_argument("--file", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", choices=["llm", "local"], default="llm",
                   help="llm=DeepSeek 判聚类（推荐）；local=本地嵌入（离线）")
    p.set_defaults(func=cmd_spi)

    p = sub.add_parser("eval", help="四项验证并输出评估报告")
    p.add_argument("--samples-dir")
    p.add_argument("--out")
    p.add_argument("--rounds", type=int, default=2)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("translate", help="生成中英双语因子链资产")
    p.set_defaults(func=cmd_translate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
