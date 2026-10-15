"""画像 → persona 描述生成（确定性模板，无需 LLM）。

将五件套分数 + SPI 聚类命中组织为：
1. 人类可读的人格画像摘要（中文）；
2. 供下一期"思维/写作模仿"使用的人设指令文本（persona prompt）。
"""
from __future__ import annotations

from typing import Dict, List

_MODEL_ZH = {
    "ocean": "大五人格(OCEAN)",
    "mmpi": "临床人格(MMPI)",
    "scm": "社会认知模型(SCM)",
    "dt4": "暗黑四联(DT4)",
    "tki": "冲突处理风格(TKI)",
    "sdt": "自我决定论(SDT)",
}


def _level(score: float) -> str:
    if score >= 6:
        return "显著偏高"
    if score >= 4.5:
        return "偏高"
    if score >= 3.5:
        return "中等"
    if score >= 2.5:
        return "偏低"
    return "显著偏低"


def build_profile_summary(scores: Dict[str, object], top_factors: List[dict] | None = None) -> str:
    """生成人类可读画像摘要。scores: {factor: FactorScore}"""
    lines: List[str] = []
    by_model: Dict[str, List] = {}
    for s in scores.values():
        by_model.setdefault(s.model_id, []).append(s)

    for model_id in ("ocean", "mmpi", "scm", "dt4", "tki"):
        items = by_model.get(model_id, [])
        if not items:
            continue
        lines.append(f"### {_MODEL_ZH.get(model_id, model_id)}")
        for s in sorted(items, key=lambda x: -x.score):
            lines.append(f"- {s.factor}：{s.score:.1f}/7（{_level(s.score)}）")

    if top_factors:
        lines.append("### SPI 语义聚类命中")
        for tf in top_factors[:5]:
            lines.append(f"- {tf['factor']}（{tf['model']}）→ {tf.get('cluster', '?')}")

    return "\n".join(lines)


def build_persona_prompt(
    scores: Dict[str, object],
    cluster_hits: dict | None = None,
    cluster_names: dict | None = None,
) -> str:
    """生成可直接注入 LLM 的人设指令（下一期模仿环节使用）。"""
    traits: List[str] = []
    for s in scores.values():
        traits.append(f"- {s.factor}: {s.score:.1f}/7（{_level(s.score)}）")

    cluster_lines: List[str] = []
    if cluster_hits:
        for cid, w in list(cluster_hits.items())[:5]:
            name = (cluster_names or {}).get(cid, cid)
            cluster_lines.append(f"- {cid} {name}（命中强度 {w:.2f}）")

    return (
        "请你以以下人格设定来思考和写作（模仿对象的画像）：\n"
        "【人格特质分数（1-7）】\n" + "\n".join(traits) +
        ("\n【语义倾向聚类】\n" + "\n".join(cluster_lines) if cluster_lines else "") +
        "\n\n在分析问题时，优先采用与上述特质一致的推理倾向（例如高尽责性者注重条理与计划，"
        "高神经质者更关注风险与不确定性，高支配性者倾向直接给出结论并掌控节奏）；"
        "写作时保持与该画像一致的语气、关注点与价值取向。"
    )
