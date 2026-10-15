"""LLM 零样本人格打分器（DeepSeek，OpenAI 兼容接口）。

方案要点（对齐项目设计）：
- 单因子独立调用：每个因子一次请求，避免跨因子污染与顺序效应；
- 结构化输出：JSON {score: 1-7, evidence: [...], confidence: 0-1}；
- 双轮平均：temperature 0.3 / 0.7 两轮取均值，压低评分抖动；
- 顺序随机化：每轮打乱因子顺序；
- 中文适配：优先使用中英双语因子链资产（见 scripts/translate_factor_chains.py）；
- 无 API key 时进入模拟模式（确定性启发式，结果带 simulated 标记）。
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"

# 尺子协议常量（v1.0.0 冻结，2026-10-10）：变更必须走 config 并重跑基线
SCALE_SCHEMA_VERSION = "1.0.0"


def load_scale_info() -> dict:
    """从 config/models.yaml 读取尺子元信息（版本、冻结状态、llm 协议），作为单一来源。"""
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project = cfg.get("project", {})
    scale = project.get("scale", {})
    llm = cfg.get("llm", {})
    return {
        "scale_version": scale.get("version", SCALE_SCHEMA_VERSION),
        "frozen": scale.get("frozen", False),
        "frozen_at": scale.get("frozen_at", ""),
        "change_rule": scale.get("change_rule", ""),
        "llm_protocol": {
            "provider": llm.get("provider", "deepseek"),
            "model": llm.get("model", "deepseek-chat"),
            "rounds": llm.get("rounds", 2),
            "temperatures": llm.get("temperatures", [0.3, 0.7]),
            "shuffle_per_round": True,
            "json_schema": "{score:1-7, evidence:[...], confidence:0-1}",
            "factor_calls": "independent",
        },
    }

_SYSTEM_PROMPT = """你是一位严谨的人格测量编码器（psychometric encoder）。

你会收到两部分输入：
1. 【人格因子定义】——某个人格因子的五层词汇编码：特质形容词、同义词、行为动词、相关名词/情境；
2. 【待分析文本】——一段自然语言文本。

你的任务：仅根据这段文本的实际内容，判断【文本作者】在多大程度上符合该人格因子的定义，
并输出一个严格 JSON 对象（不要输出任何其他文字或解释）：

{"score": 1到7的整数, "evidence": ["原文片段1", ...最多3条], "confidence": 0到1之间的小数}

评分标尺：
- 7 = 文本强烈且一致地体现该因子（存在多处明确证据）
- 5~6 = 明显体现
- 3~4 = 部分或含糊体现
- 1~2 = 几乎不体现，或体现的是相反特征

规则：
- 评分必须基于文本本身，不得臆测作者未表达的信息；
- evidence 必须逐字引用原文片段（保留原文语言，不做改写）；
- confidence 表示你对评分的把握程度（0=完全没把握，1=非常有把握）。"""


@dataclass
class FactorScore:
    model_id: str
    factor: str
    score: float  # 双轮平均后的分数
    raw_scores: List[int] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    simulated: bool = False

    def to_dict(self) -> dict:
        return {
            "model": self.model_id,
            "factor": self.factor,
            "score": round(self.score, 2),
            "raw_scores": self.raw_scores,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 2),
            "simulated": self.simulated,
        }


class FactorScorer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        rounds: int = 2,
        temperatures: Optional[List[float]] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or self._resolve_key()
        self.base_url = base_url
        self.model = model
        self.rounds = rounds
        self.temperatures = temperatures or [0.3, 0.7]
        self.timeout = timeout
        self.client = None
        self.simulated = not self.api_key
        if not self.simulated:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)

    @staticmethod
    def _resolve_key() -> Optional[str]:
        """优先环境变量，其次项目 .env。"""
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            return key
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            try:
                from dotenv import dotenv_values
                return dotenv_values(env_path).get("DEEPSEEK_API_KEY") or None
            except Exception:
                pass
        return None

    # ---------- 核心调用 ----------
    def _call_once(self, factor_definition: str, factor_name: str, text: str, temperature: float) -> dict:
        user = (
            f"【人格因子定义】\n因子名称：{factor_name}\n{factor_definition}\n\n"
            f"【待分析文本】\n{text}\n\n请按系统要求输出 JSON。"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        return json.loads(content)

    def score_factor(self, factor_definition: str, factor_name: str, text: str) -> FactorScore:
        """对一个因子做双轮评分（每轮独立调用，取平均）。"""
        if self.simulated:
            return self._simulate(factor_name, factor_definition, text)

        raw_scores: List[int] = []
        all_evidence: List[str] = []
        confidences: List[float] = []

        for t in self.temperatures[: self.rounds]:
            for attempt in range(2):  # 解析失败重试一次
                try:
                    data = self._call_once(factor_definition, factor_name, text, t)
                    score = int(data["score"])
                    if not (1 <= score <= 7):
                        raise ValueError(f"score 越界: {score}")
                    raw_scores.append(score)
                    ev = data.get("evidence", [])
                    all_evidence.extend(str(e) for e in ev if str(e).strip())
                    confidences.append(float(data.get("confidence", 0.5)))
                    break
                except Exception as e:
                    if attempt == 1:
                        print(f"  [warn] {factor_name} 第 {len(raw_scores)+1} 轮失败: {e}")
                        time.sleep(1)
                    else:
                        time.sleep(2)

        if not raw_scores:  # 全部失败
            return FactorScore(model_id="", factor=factor_name, score=0.0, simulated=True)

        # evidence 去重、保序、截断
        seen: List[str] = []
        for e in all_evidence:
            if e not in seen and len(seen) < 3:
                seen.append(e)

        return FactorScore(
            model_id="",
            factor=factor_name,
            score=sum(raw_scores) / len(raw_scores),
            raw_scores=raw_scores,
            evidence=seen,
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )

    def score_text(
        self,
        factor_chains: Dict[str, List],
        text: str,
        model_ids: Optional[List[str]] = None,
        shuffle: bool = True,
        progress: bool = True,
    ) -> Dict[str, FactorScore]:
        """对一段文本跑完整五件套画像。

        Args:
            factor_chains: {model_id: [FactorChain, ...]}
            text: 待分析文本
            model_ids: 限定模型（默认全部五件套）
            shuffle: 每轮打乱因子顺序
            progress: 打印进度
        Returns:
            {factor_name: FactorScore}
        """
        flat: List = []
        for mid in model_ids or ["ocean", "mmpi", "scm", "dt4", "tki", "sdt"]:
            flat.extend(factor_chains.get(mid, []))

        # 每轮独立乱序
        rounds_chains = []
        for _ in range(self.rounds):
            order = list(flat)
            if shuffle:
                random.shuffle(order)
            rounds_chains.append(order)

        # 为节省 API 调用：相同因子仅调用 rounds 次；这里按因子维度聚合，逐轮调用
        results: Dict[str, FactorScore] = {}
        total_calls = len(flat) * self.rounds
        call_idx = 0

        # 打乱后的顺序遍历（每轮不同），同一因子的多轮结果聚合
        per_factor: Dict[str, List] = {}
        for order in rounds_chains:
            for chain in order:
                per_factor.setdefault(chain.factor, []).append(chain)

        for factor_name, chain_list in per_factor.items():
            chain = chain_list[0]
            call_idx += self.rounds
            if progress:
                print(f"  [{call_idx}/{total_calls}] {factor_name} ...", flush=True)
            # 双轮评分（每轮一次独立调用）
            score = self._score_rounds(chain, text)
            score.model_id = chain.model_id
            results[factor_name] = score

        return results

    def _score_rounds(self, chain, text: str) -> FactorScore:
        """对同一因子执行 rounds 次独立评分（不同温度）。中文因子链优先。"""
        if self.simulated:
            return self._simulate(chain.factor, chain.lexical_signature, text)

        definition = chain.lexical_signature_zh  # 中文资产存在时自动优先
        raw_scores: List[int] = []
        all_evidence: List[str] = []
        confidences: List[float] = []
        for t in self.temperatures[: self.rounds]:
            for attempt in range(2):
                try:
                    data = self._call_once(definition, chain.factor, text, t)
                    score = int(data["score"])
                    if 1 <= score <= 7:
                        raw_scores.append(score)
                        ev = data.get("evidence", [])
                        all_evidence.extend(str(e) for e in ev if str(e).strip())
                        confidences.append(float(data.get("confidence", 0.5)))
                        break
                except Exception:
                    time.sleep(1.5)
        if not raw_scores:
            return FactorScore(model_id="", factor=chain.factor, score=0.0, simulated=True)
        seen: List[str] = []
        for e in all_evidence:
            if e not in seen and len(seen) < 3:
                seen.append(e)
        return FactorScore(
            model_id="",
            factor=chain.factor,
            score=sum(raw_scores) / len(raw_scores),
            raw_scores=raw_scores,
            evidence=seen,
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )

    # ---------- 模拟模式 ----------
    def _simulate(self, factor_name: str, factor_definition: str, text: str) -> FactorScore:
        """无 API key 时的确定性启发式：按词汇命中率给分，仅供流程验证。"""
        import re
        hits = 0
        total = 0
        tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
        for word in re.findall(r"[A-Za-z][A-Za-z'-]+", factor_definition):
            total += 1
            if word.lower() in tokens:
                hits += 1
        score = 1 if total == 0 else max(1, min(7, round(hits / max(total, 1) * 50)))
        evidence = [text[:60] + ("…" if len(text) > 60 else "")]
        return FactorScore(
            model_id="", factor=factor_name,
            score=float(score), raw_scores=[score],
            evidence=evidence, confidence=0.1, simulated=True,
        )


def save_profile(results: Dict[str, FactorScore], out_path: Path) -> None:
    """保存完整画像 JSON（按模型分组）。"""
    grouped: Dict[str, List[dict]] = {}
    for score in results.values():
        grouped.setdefault(score.model_id, []).append(score.to_dict())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    print(f"[ok] 画像已保存: {out_path}")
