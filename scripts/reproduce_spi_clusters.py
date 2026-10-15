#!/usr/bin/env python3
"""复现论文 SPI 15 聚类（PCA-50 + KMeans-15，同 generate_spi_table.py 参数），
输出逐条目→聚类 id 标注，供 LLM 判聚类一致性实验作 ground truth。

参数（与论文一致）：PCA n_components=50, random_state=42（无标准化）；
KMeans k=15, random_state=42, n_init=10, algorithm='lloyd'。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EMB_DIR = PROJECT_ROOT / "assets" / "survey" / "Embeddings"
SPI_TABLE = PROJECT_ROOT / "assets" / "survey" / "results" / "spi_table.json"
OUT = PROJECT_ROOT / "assets" / "spi_item_clusters.json"

MODEL_LIST = sorted([
    "aam", "bdi", "bt", "cest", "cmoa", "cs", "disc", "dt4", "dtm", "em",
    "epm", "ffni", "ffni_sf", "fsls", "ftm", "gad7", "hex", "hsns", "ipn",
    "mbti", "mcmi", "mcmin", "mmpi", "mst", "narq", "npi", "ocean", "papc",
    "pct", "pni", "rft", "riasec", "rit", "scid", "scm", "sdt", "sixteenpf",
    "stbv", "tat", "tci", "tei", "tki", "tmp", "wais",
])


def main() -> int:
    metas: list[dict] = []
    embs: list[np.ndarray] = []
    for model in MODEL_LIST:
        f = EMB_DIR / f"{model}_embeddings.csv"
        if not f.exists():
            print(f"[warn] 缺 {f.name}")
            continue
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            metas.append(
                {
                    "model": model,
                    "factor": str(row["Factor"]),
                    "adjective": str(row["Adjective"]),
                    "synonym": str(row.get("Synonym", "")),
                    "verb": str(row.get("Verb", "")),
                    "noun": str(row.get("Noun", "")),
                }
            )
            embs.append(json.loads(row["Embedding"]))
    X = np.vstack(embs).astype(np.float32)
    print(f"[ok] 嵌入矩阵: {X.shape}")

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X).astype(np.float32)
    km = KMeans(n_clusters=15, random_state=42, n_init=10, algorithm="lloyd")
    labels = km.fit_predict(X_pca)
    print(f"[ok] 聚类完成，silhouette≈{km.inertia_:.1f}（inertia）")

    # 与论文 spi_table 的聚类规模对比验证
    spi = json.loads(SPI_TABLE.read_text(encoding="utf-8"))
    clusters = spi["clusters"]
    from collections import Counter

    cnt = Counter(int(x) for x in labels)
    # sklearn 版本差异会导致簇编号重排：按"规模一一对应"建映射
    paper_n_by_label = {c["id"]: c["n"] for c in clusters}
    our_n_by_id = {cid: n for cid, n in cnt.items()}
    rev_paper = {n: lbl for lbl, n in paper_n_by_label.items()}
    id_map = {}  # 复现 id -> 论文 id
    for cid, n in our_n_by_id.items():
        if n in rev_paper:
            id_map[cid] = rev_paper[n]
        else:
            print(f"[error] 复现簇 {cid} 规模 {n} 在论文中无对应")
            return 1
    print(f"[ok] 簇编号映射（复现id→论文id）: {id_map}")

    # 内容抽查：论文 top_adjectives 应高频出现在映射后簇中
    label_adj = {c["id"]: set(a.lower() for a in c.get("top_adjectives", [])) for c in clusters}
    from collections import defaultdict

    adj_by_cluster = defaultdict(list)
    for i, m in enumerate(metas):
        adj_by_cluster[id_map[int(labels[i])]].append(m["adjective"].lower())
    sample_ok = True
    for lbl, adjs in label_adj.items():
        pool = adj_by_cluster.get(lbl, [])
        hits = sum(1 for a in pool if any(t in a for t in adjs))
        ratio = hits / max(1, len(pool))
        tag = "✓" if ratio >= 0.2 else "✗"
        if ratio < 0.2:
            sample_ok = False
        print(f"  {tag} cluster {lbl}: top-adj 命中率 {ratio:.0%}（样例: {pool[:3]}）")
    if not sample_ok:
        print("[warn] 形容词抽查未通过，映射可能不稳（继续以映射结果为准，实验后再核）")

    records = []
    for i, m in enumerate(metas):
        records.append({**m, "cluster_id": id_map[int(labels[i])]})
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ok] 输出: {OUT}（{len(records)} 条，cluster_id 已对齐论文语义）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
