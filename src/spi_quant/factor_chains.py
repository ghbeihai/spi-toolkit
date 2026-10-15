"""五件套因子链加载器。

解析 atlas 仓库中 OCEAN/MMPI/SCM/DT4/TKI 的因子链 CSV（5 列 schema：
Factor, Adjective, Synonym, Verb, Noun；TKI 额外带 Category 列），
按模型聚合成结构化的因子定义，供打分器与 SPI 映射使用。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# 项目根目录（src/spi_quant/ 的上一级再上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
ATLAS_ROOT = PROJECT_ROOT / "assets" / "survey"


@dataclass
class FactorChain:
    """单个因子的链式定义。"""
    model_id: str
    factor: str
    adjectives: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    verbs: List[str] = field(default_factory=list)
    nouns: List[str] = field(default_factory=list)
    zh: Optional[dict] = None  # 中英双语资产（translate 生成）

    @property
    def lexical_signature(self) -> str:
        """五列合并成一段词汇签名，用于 prompt 与本地嵌入。"""
        parts = [
            f"特质形容词: {'、'.join(self.adjectives)}",
            f"同义词: {'、'.join(self.synonyms)}",
            f"行为动词: {'、'.join(self.verbs)}",
            f"相关名词/情境: {'、'.join(self.nouns)}",
        ]
        return "; ".join(parts)

    @property
    def lexical_signature_zh(self) -> str:
        """中文五列签名（存在双语资产时优先用于 LLM prompt）。"""
        if self.zh:
            parts = [
                f"特质形容词: {'、'.join(self.zh.get('adjectives_zh', self.adjectives))}",
                f"同义词: {'、'.join(self.zh.get('synonyms_zh', self.synonyms))}",
                f"行为动词: {'、'.join(self.zh.get('verbs_zh', self.verbs))}",
                f"相关名词/情境: {'、'.join(self.zh.get('nouns_zh', self.nouns))}",
            ]
            return "; ".join(parts)
        return self.lexical_signature


def load_models_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_factor_chains(
    atlas_root: Path = ATLAS_ROOT,
    model_ids: Optional[List[str]] = None,
) -> Dict[str, List[FactorChain]]:
    """加载五件套因子链。

    Returns:
        {model_id: [FactorChain, ...]}，每个因子聚合其全部条目行。
    """
    config = load_models_config()
    models = config["models"]
    if model_ids:
        models = [m for m in models if m["id"] in model_ids]

    result: Dict[str, List[FactorChain]] = {}
    for model in models:
        csv_path = atlas_root / model["file"]
        if not csv_path.exists():
            raise FileNotFoundError(f"因子链文件缺失: {csv_path}")

        # 聚合：factor -> FactorChain
        chains: Dict[str, FactorChain] = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                factor = row.get("Factor", "").strip()
                if not factor:
                    continue
                chain = chains.setdefault(
                    factor,
                    FactorChain(model_id=model["id"], factor=factor),
                )
                for col, target in (
                    ("Adjective", "adjectives"),
                    ("Synonym", "synonyms"),
                    ("Verb", "verbs"),
                    ("Noun", "nouns"),
                ):
                    val = row.get(col, "").strip()
                    if val:
                        # 同义词/动词可能以分号或引号包裹
                        for item in _split_items(val):
                            getattr(chain, target).append(item)

        # 去重、保序
        for chain in chains.values():
            for target in ("adjectives", "synonyms", "verbs", "nouns"):
                seen: List[str] = []
                for item in getattr(chain, target):
                    if item not in seen:
                        seen.append(item)
                setattr(chain, target, seen)

        result[model["id"]] = list(chains.values())

    # 挂载中英双语资产（若已生成）
    bilingual_path = PROJECT_ROOT / "assets" / "bilingual_factor_chains.json"
    if bilingual_path.exists():
        import json
        try:
            bilingual = json.loads(bilingual_path.read_text(encoding="utf-8"))
            for mid, items in result.items():
                for chain in items:
                    key = f"{mid}:{chain.factor}"
                    if key in bilingual:
                        chain.zh = bilingual[key]
        except Exception:
            pass

    return result


def _split_items(raw: str) -> List[str]:
    """拆分单元格内多个条目（逗号/分号分隔），去掉引号与空白。"""
    items: List[str] = []
    for part in raw.replace(";", ",").split(","):
        item = part.strip().strip("\"'").strip()
        if item:
            items.append(item)
    return items


def all_factor_chains(chains_by_model: Dict[str, List[FactorChain]]) -> List[FactorChain]:
    """扁平化为单因子列表（顺序固定，保证评分稳定性）。"""
    flat: List[FactorChain] = []
    for model_id in ("ocean", "mmpi", "scm", "dt4", "tki", "sdt"):
        flat.extend(chains_by_model.get(model_id, []))
    return flat


if __name__ == "__main__":
    chains = load_factor_chains()
    total = 0
    for mid, items in chains.items():
        print(f"{mid}: {len(items)} 个因子")
        for c in items:
            print(f"  - {c.factor}（形容词 {len(c.adjectives)}、同义词 {len(c.synonyms)}、动词 {len(c.verbs)}、名词 {len(c.nouns)}）")
        total += len(items)
    print(f"\n合计 {total} 个因子")
