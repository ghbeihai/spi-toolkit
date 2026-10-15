"""SPI 聚类映射：文本 → 15 个语义聚类命中画像。

两种模式：
- local（默认，向后兼容）：完全本地、无 API。
  1. 从 atlas 仓库的 *_clustered_embeddings.csv 提取 (model_id, factor) → Cluster 映射；
  2. 用本地多语言 sentence-transformers 对 29 个因子链的词汇签名做预计算嵌入（缓存）；
  3. 输入文本嵌入 → 与全部因子链余弦相似度 → Top-K 命中因子；
  4. 因子命中权重按聚类聚合 → 15 维命中向量。
- llm：DeepSeek 零样本判聚类（2026-10-10 实验一致性 77.3%，远超本地嵌入 33.3%）。
  用论文 spi_table.json 的 15 聚类定义（label+代表形容词+代表因子），LLM 判文本命中聚类。

注：仓库主实验使用 OpenAI 1536 维嵌入；本地 384/768 维多语言模型精度低（33.3% 天花板），
本地路径定位为"离线探索参考"；推荐使用 LLM 判聚类模式。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS_ROOT = PROJECT_ROOT / "assets" / "survey"
CACHE_PATH = PROJECT_ROOT / "assets" / "local_factor_embeddings.json"
SPI_TABLE_PATH = ATLAS_ROOT / "results" / "spi_table.json"

SPI_CLUSTER_NAMES = {
    "C0": "Warmth and Prosociality 温暖与亲社会",
    "C1": "Dominance and Persistence 支配与坚持",
    "C2": "Clinical Psychopathology 临床精神病理",
    "C3": "Depression and Anhedonia 抑郁与快感缺失",
    "C4": "Activation and Approach 激活与趋近",
    "C5": "Imagination and Openness 想象与开放性",
    "C6": "Withdrawal and Avoidance 退缩与回避",
    "C7": "Self-Direction and Autonomy 自我定向与自主",
    "C8": "Reactive Aggression 反应性攻击",
    "C9": "Shame and Self-Punishment 羞耻与自我惩罚",
    "C10": "Dark Manipulation 黑暗操纵",
    "C11": "Grandiose Narcissism 夸大自恋",
    "C12": "Anxiety and Fear 焦虑与恐惧",
    "C13": "Analytical Cognition 分析性认知",
    "C14": "Applied Life Domains 应用生活领域",
}


def load_factor_cluster_map(atlas_root: Path = ATLAS_ROOT) -> Dict[str, str]:
    """从 *_clustered_embeddings.csv 读取 (model_id:factor) -> Cluster 映射。"""
    mapping: Dict[str, str] = {}
    for csv_path in sorted(atlas_root.glob("Embeddings/*_clustered_embeddings.csv")):
        model_id = csv_path.name.replace("_clustered_embeddings.csv", "")
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                factor = (row.get("Factor") or "").strip()
                cluster = (row.get("Cluster") or "").strip()
                # CSV 中 Cluster 为数字；统一为论文命名 C0-C14
                if cluster.isdigit():
                    cluster = f"C{cluster}"
                if factor and cluster:
                    mapping[f"{model_id}:{factor}"] = cluster
    return mapping


class SpiMapper:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        # B 路线：384 维 MiniLM → 768 维 mpnet（多语言），提升语义区分度
        self.model_name = model_name
        self._encoder = None
        self.factor_embeddings: List[Tuple[str, str, List[float]]] = []  # (model_id, factor, emb)
        self.cluster_map: Dict[str, str] = {}

    # ---------- 懒加载编码器 ----------
    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "本地嵌入需要 sentence-transformers，请执行: uv pip install -e '.[spi-local]'"
                ) from e
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    # ---------- 因子链嵌入（缓存优先） ----------
    def prepare(self, chains_by_model: Dict[str, List], force_rebuild: bool = False) -> None:
        """预计算 29 个因子链词汇签名的嵌入并缓存。"""
        self.cluster_map = load_factor_cluster_map()

        if CACHE_PATH.exists() and not force_rebuild:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            self.factor_embeddings = [(m, f, e) for m, f, e in cache]
            print(f"[spi] 因子链嵌入缓存命中: {len(self.factor_embeddings)} 条")
            return

        encoder = self._get_encoder()
        items: List[Tuple[str, str, str]] = []
        for model_id, chains in chains_by_model.items():
            for chain in chains:
                items.append((model_id, chain.factor, chain.lexical_signature))

        texts = [sig for _, _, sig in items]
        vecs = encoder.encode(texts, normalize_embeddings=True).tolist()
        self.factor_embeddings = [(m, f, v) for (m, f, _), v in zip(items, vecs)]

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(self.factor_embeddings, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[spi] 因子链嵌入已计算并缓存: {len(self.factor_embeddings)} 条")

    # ---------- 映射 ----------
    def map_text(
        self, text: str, top_k: int = 5, cluster_names: Optional[Dict[str, str]] = None
    ) -> dict:
        """输入文本 → {top_factors, cluster_hits}。"""
        names = cluster_names or SPI_CLUSTER_NAMES
        encoder = self._get_encoder()
        text_vec = encoder.encode([text], normalize_embeddings=True)[0]

        # 余弦相似度（向量已归一化）
        scores = []
        for model_id, factor, emb in self.factor_embeddings:
            sim = float(sum(a * b for a, b in zip(text_vec, emb)))
            scores.append((sim, model_id, factor))

        scores.sort(reverse=True)
        top = scores[:top_k]

        # 聚类聚合
        cluster_weights: Dict[str, float] = {}
        for sim, model_id, factor in scores:
            key = f"{model_id}:{factor}"
            cluster = self.cluster_map.get(key)
            if cluster:
                cluster_weights[cluster] = cluster_weights.get(cluster, 0.0) + sim
        # 归一化到 0-1（相对）
        max_w = max(cluster_weights.values(), default=1.0)
        cluster_hits = {
            cid: round(w / max_w, 3)
            for cid, w in sorted(cluster_weights.items(), key=lambda kv: -kv[1])
        }

        top_factors = [
            {
                "model": model_id,
                "factor": factor,
                "similarity": round(sim, 4),
                "cluster": self.cluster_map.get(f"{model_id}:{factor}", "?"),
            }
            for sim, model_id, factor in top
        ]

        return {
            "text_len": len(text),
            "top_factors": top_factors,
            "cluster_hits": cluster_hits,
            "cluster_names": {cid: names.get(cid, cid) for cid in cluster_hits},
        }


class SpiMapperLLM:
    """LLM 零样本判聚类模式（推荐）。

    用论文 spi_table.json 的 15 聚类定义（label + top adjectives + top factors），
    让 DeepSeek 判断输入文本命中哪些语义聚类。2026-10-10 实验一致性 77.3%。
    返回结构与 SpiMapper.map_text 兼容（cluster_hits / cluster_names / top_factors）。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        timeout: int = 90,
    ):
        try:
            from dotenv import dotenv_values
        except ImportError:  # pragma: no cover
            dotenv_values = None
        if dotenv_values is not None:
            env = dotenv_values(PROJECT_ROOT / ".env")
            api_key = env.get("DEEPSEEK_API_KEY") or ""
        else:
            api_key = ""
        if not api_key:
            raise RuntimeError(
                "LLM 判聚类模式需要 DEEPSEEK_API_KEY（项目 .env 中配置）"
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.clusters = self._load_clusters()
        self.cluster_map: Dict[str, str] = {}

    def _load_clusters(self) -> List[dict]:
        spi = json.loads(SPI_TABLE_PATH.read_text(encoding="utf-8"))
        return spi["clusters"]  # [{id, label, top_adjectives, top_factors, ...}]

    def prepare(self, chains_by_model=None, force_rebuild: bool = False) -> None:
        """接口兼容：LLM 模式无需嵌入，仅加载 15 聚类定义。"""
        print(f"[spi] LLM 判聚类模式（{self.model}），已加载 {len(self.clusters)} 个聚类定义")

    # ---------- 映射 ----------
    def map_text(
        self, text: str, top_k: int = 5, cluster_names: Optional[Dict[str, str]] = None
    ) -> dict:
        """输入文本 → {top_factors, top_clusters, cluster_hits, cluster_names}。

        LLM 输出 JSON：{"hits": [{"cluster_id": N, "relevance": 1-7, "reason": "..."}], "top": [...]}
        relevance 1-7 表示文本与该聚类的契合度（>=4 才列出，最多 5 个）。
        """
        names = cluster_names or SPI_CLUSTER_NAMES
        desc = self._cluster_desc()
        user = (
            "你是心理学人格维度聚类分类器。以下 15 个聚类是对人格模型语义空间的划分：\n"
            f"{desc}\n\n"
            "下面是一段文字（文章、发言、讲稿或量表条目），"
            "请判断这段文字在多大程度上属于各个聚类。\n"
            "只列出契合度明确（relevance >= 4）的聚类，最多 5 个。\n"
            '输出严格 JSON：{"hits": [{"cluster_id": <数字>, "relevance": <1-7>, '
            '"reason": "<一句话依据>"}]}\n\n'
            f"文字：{text[:6000]}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的人格心理学聚类分类器。基于文本语义判断，输出严格 JSON。",
                },
                {"role": "user", "content": user},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        hits = data.get("hits", []) or []

        label_by_id = {c["id"]: c["label"] for c in self.clusters}
        top_clusters: List[dict] = []
        cluster_hits: Dict[str, float] = {}
        for h in hits:
            cid = h.get("cluster_id")
            if cid is None:
                continue
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if cid not in label_by_id:
                continue
            ckey = f"C{cid}"
            rel = float(h.get("relevance", 4))
            cluster_hits[ckey] = round(min(rel, 7.0) / 7.0, 3)
            top_clusters.append(
                {
                    "cluster": ckey,
                    "label": label_by_id[cid],
                    "score": rel,
                    "reason": str(h.get("reason", "")),
                }
            )
        # 按契合度降序，取 top_k
        top_clusters = sorted(top_clusters, key=lambda x: -x["score"])[:top_k]
        # 归一化命中（与 local 版口径一致：最大命中 = 1.0）
        maxv = max(cluster_hits.values(), default=1.0)
        cluster_hits = {
            k: round(v / maxv, 3) for k, v in sorted(cluster_hits.items(), key=lambda kv: -kv[1])
        }

        # 兼容字段：top_factors（映射为聚类命中，供 profile summary 展示）
        top_factors = [
            {
                "model": "SPI",
                "factor": names.get(c["cluster"], c["label"]),
                "similarity": round(c["score"] / 7.0, 4),
                "cluster": c["cluster"],
            }
            for c in top_clusters
        ]

        return {
            "text_len": len(text),
            "mode": "llm",
            "model": self.model,
            "top_factors": top_factors,
            "top_clusters": top_clusters,
            "cluster_hits": cluster_hits,
            "cluster_names": {k: names.get(k, k) for k in cluster_hits},
        }

    def _cluster_desc(self) -> str:
        lines = []
        for c in self.clusters:
            adj = ", ".join(c.get("top_adjectives", [])[:3])
            fac = ", ".join(c.get("top_factors", [])[:4])
            lines.append(f"{c['id']} {c['label']}（代表形容词：{adj}；代表因子：{fac}）")
        return "\n".join(lines)
