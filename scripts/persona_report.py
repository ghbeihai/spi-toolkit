#!/usr/bin/env python3
"""人格画像解读生成器（框架 v1.0，2026-10-10 定稿）。

科学依据：
1. 绝对锚点：1-7 分语义化（≥5.5 突出 / 3.5-4.4 居中 / ≤3.5 低）；
2. 稳定性锚：跨文本 σ（<0.8 核心稳定特质，≥1.2 情境波动角色响应，特质=跨情境一致性）；
3. 「强度 × 稳定性」2×2 分类：稳定高=核心人格优势，波动高=角色响应，稳定低=稳定的低倾向；
4. 分层解读：统计→结构→情境→应用；
5. 跨人相对定位（vs 基准人群，同尺子）；
6. 方法学注记（信度/样本/口径透明）。

用法：
  python scripts/persona_report.py \
    --name <对象key> \
    --profile data/results/<obj>_profiles/<obj>_report_all.json \
    --stability data/results/<obj>_profiles/<obj>_report_all.json \
    --out data/results/<obj>_profiles/<obj>_persona_report.md
（--profile 含 aggregate(mean) 的聚合报告或单篇画像；--stability 提供 σ 的聚合报告。）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spi_quant.factor_chains import load_factor_chains, all_factor_chains

ZH_FACTOR = {
    "Extraversion": "外向性","Agreeableness": "宜人性","Conscientiousness": "尽责性","Neuroticism": "神经质",
    "Openness": "开放性","Hypochondriasis": "疑病症","Depression": "抑郁","Hysteria": "癔症",
    "Psychopathic_Deviate": "精神病态偏差","Masculinity_Femininity": "男性气质/女性气质","Paranoia": "偏执",
    "Psychasthenia": "精神衰弱","Schizophrenia": "精神分裂症","Hypomania": "轻躁狂","Social_Introversion": "社交内向",
    "Behavior": "行为","Cognition": "认知","Environment": "环境","Interpersonal": "人际","Affective": "情感",
    "Machiavellianism": "马基雅维利主义","Narcissism": "自恋","Psychopathy": "精神病态","Sadism": "施虐狂",
    "Competing": "竞争","Collaborating": "合作","Compromising": "妥协","Avoiding": "回避","Accommodating": "迁就",
    "Autonomy": "自主性","Competence": "胜任力","Relatedness": "关联性",
}
HUMAN = {
    "Extraversion": {"high": "乐于表达、社交能量足", "low": "安静内敛、独处充电", "mid": "外向程度居中"},
    "Agreeableness": {"high": "温和包容、好相处", "low": "直率有主见、不强求讨好", "mid": "温和与主见并重"},
    "Conscientiousness": {"high": "自律靠谱、做事有条理", "low": "灵活随性、不重纪律约束", "mid": "自律与灵活平衡"},
    "Neuroticism": {"high": "情绪敏感、易焦虑波动", "low": "情绪稳定、遇事不慌", "mid": "情绪大致平稳"},
    "Openness": {"high": "求知欲强、接受新事物", "low": "务实守成、偏好熟悉路径", "mid": "务实与好奇兼具"},
    "Hypochondriasis": {"high": "关注身体不适、易担心健康", "low": "身体焦虑少、心态松", "mid": "健康关注度一般"},
    "Depression": {"high": "情绪低落、兴致缺缺", "low": "无抑郁倾向、情绪底色干净", "mid": "情绪基调中性"},
    "Hysteria": {"high": "表达外放、感染力强", "low": "表达克制、不走戏剧化路线", "mid": "表达收放适度"},
    "Psychopathic_Deviate": {"high": "打破常规、不服管束", "low": "守规矩、不挑战边界", "mid": "常规与自主平衡"},
    "Masculinity_Femininity": {"high": "气质弹性大、不落性别刻板", "low": "性别气质传统分明", "mid": "性别气质较均衡"},
    "Paranoia": {"high": "多疑、防御性强", "low": "对人信任度高、不疑神疑鬼", "mid": "警觉但不过度"},
    "Psychasthenia": {"high": "易纠结、思虑过度", "low": "心思清爽、不内耗", "mid": "思虑适度"},
    "Schizophrenia": {"high": "思维跳跃、想法特立独行", "low": "思维连贯、常识感强", "mid": "思维常规"},
    "Hypomania": {"high": "精力旺盛、节奏快、话多输出", "low": "节奏平缓、不躁动", "mid": "精力适中"},
    "Social_Introversion": {"high": "回避社交、喜独处", "low": "不社恐、乐于在场互动", "mid": "社交需求居中"},
    "Behavior": {"high": "行动力强、说干就干", "low": "偏思考、行动节奏慢", "mid": "行动与思考并重"},
    "Cognition": {"high": "逻辑清晰、善于拆解分析", "low": "凭直觉感受、少理论框架", "mid": "直觉与逻辑并用"},
    "Environment": {"high": "善于调动环境资源、适应力强", "low": "对外部环境经营少", "mid": "环境适应一般"},
    "Interpersonal": {"high": "擅长连接他人、关系导向", "low": "人际介入少、独立行事", "mid": "人际适度介入"},
    "Affective": {"high": "情绪感知丰富、共情表达强", "low": "情感表达内敛", "mid": "情感表达适度"},
    "Machiavellianism": {"high": "精于算计、谋略导向", "low": "直来直去、不玩心眼", "mid": "策略性一般"},
    "Narcissism": {"high": "自我确信、重视形象与权威感", "low": "低调谦和、不端架子", "mid": "自我认同适中"},
    "Psychopathy": {"high": "冷漠冲动、缺共情与愧疚", "low": "有温度、无攻击性", "mid": "冷静且合群"},
    "Sadism": {"high": "享受贬损他人的快感", "low": "无伤害意图、与人为善", "mid": "无恶意倾向"},
    "Competing": {"high": "遇冲突力争高下、坚持己见", "low": "不争不抢、避免硬碰硬", "mid": "该争则争、该让则让"},
    "Collaborating": {"high": "寻求共赢、一起解决", "low": "少主动协作、倾向自己搞定", "mid": "协作意愿一般"},
    "Compromising": {"high": "各退一步、求折中", "low": "不轻易让步、不和稀泥", "mid": "必要时折中"},
    "Avoiding": {"high": "能躲则躲、推迟面对", "low": "直面问题、不绕弯子", "mid": "视情况直面"},
    "Accommodating": {"high": "以对方为先、好说话", "low": "不委屈自己、不迁就", "mid": "适度顾及对方"},
    "Autonomy": {"high": "自己拿主意、内驱力强", "low": "依赖外部指引、需要被推动", "mid": "自主与协同平衡"},
    "Competence": {"high": "有掌控感、做事有底气", "low": "自我效能感一般、易自我怀疑", "mid": "胜任感一般"},
    "Relatedness": {"high": "重关系温度、需要归属", "low": "对关系热度需求低、更重内容", "mid": "关系需求适中"},
}

def human_of(f: str, m: float) -> str:
    h = HUMAN.get(f, {})
    if m >= 4.5:
        return h.get("high", "")
    if m <= 3.5:
        return h.get("low", "")
    return h.get("mid", "")

MODEL_ZH = {"ocean":"OCEAN 大五","mmpi":"MMPI 临床","scm":"SCM 社会认知","dt4":"DT4 暗黑四联","tki":"TKI 冲突处理","sdt":"SDT 自我决定"}
ORDER = [c.factor for c in all_factor_chains(load_factor_chains())]
MODEL_OF = {c.factor: c.model_id for c in all_factor_chains(load_factor_chains())}


def load_means(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in d:
        return {f: v["mean"] for f, v in d["aggregate"].items()}
    if "scores" in d:
        return {f: v["score"] for f, v in d["scores"].items() if not v.get("failed")}
    raise ValueError(f"无法识别画像结构: {path}")


def load_stds(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in d:
        return {f: v["std"] for f, v in d["aggregate"].items()}
    raise ValueError(f"--stability 需要聚合报告（含 std）: {path}")


def fmt(score: float) -> str:
    return f"{score:.1f}" if score == int(score) else f"{score:.2f}".rstrip("0").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="人物名")
    ap.add_argument("--profile", required=True, help="画像 JSON（均值）")
    ap.add_argument("--stability", required=True, help="聚合报告 JSON（含 std）")
    ap.add_argument("--baseline", help="可选：基准人群聚合报告，用于跨人相对定位")
    ap.add_argument("--spi", help="可选：SPI 判聚类汇总 JSON（含 cluster_summary）")
    ap.add_argument("--html", help="可选：输出 HTML 三件套（指纹图 + 15 聚类 + 解读）")
    ap.add_argument("--out", required=True, help="输出解读 Markdown")
    args = ap.parse_args()

    means = load_means(Path(args.profile))
    stds = load_stds(Path(args.stability))
    base = load_means(Path(args.baseline)) if args.baseline else None

    # 2×2 分类（强度 × 稳定性）
    stable_high, stable_low, volatile = [], [], []
    for f in ORDER:
        m, s = means.get(f), stds.get(f)
        if m is None or s is None:
            continue
    # 强度 × 稳定性 2×2 分类（框架 v1.0）
    stable_high, stable_low, stable_mid, volatile = [], [], [], []
    for f in ORDER:
        m, s = means.get(f), stds.get(f)
        if m is None or s is None:
            continue
        if s >= 1.2:
            volatile.append((f, m, s))
        elif s < 0.8:
            if m >= 4.5:
                stable_high.append((f, m, s))
            elif m <= 3.5:
                stable_low.append((f, m, s))
            else:
                stable_mid.append((f, m, s))

    stable_high.sort(key=lambda x: -x[1]); stable_low.sort(key=lambda x: x[1]); stable_mid.sort(key=lambda x: -x[1]); volatile.sort(key=lambda x: -x[2])

    L = []
    L.append(f"# {args.name} · 人格画像解读\n")
    L.append("> 数据口径：LLM 零样本打分（六件套 32 因子 × 2 轮平均）· σ=跨文本变异 · 尺子 v1.0.0（2026-10-10 冻结）\n")

    # 核心特质（2×2，逐因子一行）
    L.append("**核心特质（强度 × 稳定性）**\n")
    L.append("| 类别 | 因子 | 分数（σ） | 说明 |")
    L.append("|---|---|---|---|")
    def rows(items, cat, sub=""):
        if not items:
            L.append(f"| {cat} | — | — | 无{sub}因子 |")
            return
        for f, m, s in items:
            L.append(f"| {cat} | {ZH_FACTOR[f]} | {fmt(m)}（σ{fmt(s)}） | {human_of(f, m)} |")
    rows(stable_high, "稳定高", "稳定突出")
    rows(stable_mid, "稳定中", "稳定居中")
    rows(stable_low, "稳定低", "稳定低倾向")
    rows(volatile, "情境波动", "情境波动")
    L.append("")

    # 相对定位
    L.append("**跨人相对定位**（vs 基准，仅同尺子可比较）\n")
    if base:
        dev = sorted([(f, means[f] - base[f]) for f in ORDER if f in means and f in base], key=lambda x: -abs(x[1]))
        sig = [(f, d) for f, d in dev if abs(d) >= 1.0]
        if sig:
            L.append("显著偏离（|Δ|≥1.0）：" + "、".join(f"{ZH_FACTOR[f]} {d:+.1f}" for f, d in sig[:8]) + "\n")
        else:
            L.append("无显著偏离（|Δ|<1.0）\n")
    else:
        L.append("（未提供基准人群，跳过）\n")

    # 方法学注记
    L.append("**方法学注记**")
    L.append("- 归纳性描述，非临床诊断；σ<0.8 核心 / σ≥1.2 情境；重测平均绝对差 0.28；跨人比较须同一尺子版本。\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] 解读已生成: {out_path}")

    if args.html:
        spi = []
        if args.spi:
            sd = json.loads(Path(args.spi).read_text(encoding="utf-8"))
            spi = sd.get("cluster_summary", [])
            n_articles = sd.get("meta", {}).get("titles")
        else:
            n_articles = None
        html = build_html(args.name, means, stds, base, spi, stable_high, stable_low, stable_mid, volatile, n_articles)
        hp = Path(args.html)
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(html, encoding="utf-8")
        print(f"[ok] HTML 三件套已生成: {hp}")
    return 0

# ==================== HTML 三件套（指纹图 + 15 聚类 + 解读）====================
CLUSTER_ZH = {
    "C0": "温暖与亲社会", "C1": "支配与坚持", "C2": "临床精神病理", "C3": "抑郁与快感缺失",
    "C4": "激活与趋近", "C5": "想象与开放性", "C6": "退缩与回避", "C7": "自我定向与自主",
    "C8": "反应性攻击", "C9": "羞耻与自我惩罚", "C10": "黑暗操纵", "C11": "夸大自恋",
    "C12": "焦虑与恐惧", "C13": "分析性认知", "C14": "应用生活领域",
}
CLUSTER_PLAIN = {
    "C0": "待人和气、愿意帮人、利他",
    "C1": "有主见、要强、坚持目标",
    "C2": "与心理异常相关的表现",
    "C3": "心情低落、提不起劲",
    "C4": "行动派、主动接近目标",
    "C5": "点子多、接受新事物、有画面感",
    "C6": "遇事缩着、能躲就躲",
    "C7": "自己拿主意、自主驱动",
    "C8": "被激怒时的攻击性反应",
    "C9": "爱自责、对自己狠",
    "C10": "算计、利用他人达成目的",
    "C11": "自信爆棚、觉得自己不一般",
    "C12": "容易担心害怕",
    "C13": "认知风格——想问题重逻辑、讲条理、系统拆解",
    "C14": "人格在具体生活/工作场景里的表现——怎么工作、怎么管钱、怎么过日子",
}
FACTOR_COLOR = [  # 分数 1-7 → 单色蓝 7 档
    (6.5, "#123f78"), (5.5, "#24589c"), (4.5, "#3f7dbf"), (3.5, "#6ea3d1"),
    (2.5, "#9dc0e2"), (1.5, "#c5d8ef"), (0, "#e8f0fa"),
]
CLUSTER_COLOR = [  # 首命中占比 0-100% → 单色蓝
    (50, "#123f78"), (30, "#24589c"), (10, "#3f7dbf"), (2, "#6ea3d1"), (0, "#9dc0e2"),
]

def _color(s: float, table) -> str:
    for thr, c in table:
        if s >= thr:
            return c
    return table[-1][1]

def _row(*cells, style=""):
    return f'<tr style="border-bottom:1px solid #e0e0e0;{style}">' + "".join(cells) + "</tr>"

def _td(txt, extra="") -> str:
    return f'<td style="padding:5px 10px;{extra}">{txt}</td>'


def _svg_esc(t: str) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _radar_svg(mid: str, means: dict) -> str:
    """单人雷达图 SVG（单色蓝）。viewBox 460x300，中心 (230,175)，半径 105。"""
    fs = [f for f in ORDER if MODEL_OF[f] == mid and f in means]
    n = len(fs)
    if not n:
        return ""
    cx, cy, r = 230, 175, 105
    angles = [-90 + i * 360 / n for i in range(n)]

    def pt(a, rad, ox=0.0, oy=0.0):
        ar = math.radians(a)
        return (cx + (rad + ox) * math.cos(ar), cy + (rad + oy) * math.sin(ar))

    def path_of(rad):
        pts = [pt(angles[i], rad) for i in range(n)]
        return "M" + " L".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts) + "Z"

    def split_name(nm):
        if "-" in nm:
            return nm.split("-")
        if len(nm) > 4:
            return [nm[:4], nm[4:]]
        return [nm]

    P = ['<svg viewBox="0 0 460 300" width="100%" xmlns="http://www.w3.org/2000/svg">']
    P.append(f'<text x="18" y="24" font-size="14" font-weight="bold" fill="#123f78">{_svg_esc(MODEL_ZH[mid])}</text>')
    for k in (0.33, 0.66, 1.0):
        P.append(f'<path d="{path_of(r * k)}" fill="none" stroke="#d5dee9" stroke-width="1.2"/>')
    for i in range(n):
        ep = pt(angles[i], r)
        P.append(f'<line x1="{cx}" y1="{cy}" x2="{ep[0]:.1f}" y2="{ep[1]:.1f}" stroke="#d5dee9" stroke-width="1.2"/>')
        lp = pt(angles[i], r + 16)
        ar = math.radians(angles[i])
        cosv, sinv = math.cos(ar), math.sin(ar)
        anchor = "start" if cosv > 0.3 else ("end" if cosv < -0.3 else "middle")
        dy = 3 if sinv > 0.4 else (-9 if sinv < -0.4 else 0)
        for j, nm in enumerate(split_name(ZH_FACTOR[fs[i]])):
            P.append(f'<text x="{lp[0]:.1f}" y="{lp[1] + j * 11 + dy:.1f}" text-anchor="{anchor}" font-size="10" fill="#444">{_svg_esc(nm)}</text>')
    vals = [means[f] for f in fs]
    d = ""
    for i in range(n):
        q = pt(angles[i], r * max(0.01, vals[i] / 7))
        d += ("M" if i == 0 else "L") + f"{q[0]:.1f},{q[1]:.1f}"
    d += "Z"
    P.append(f'<path d="{d}" fill="#24589c" fill-opacity="0.15" stroke="#24589c" stroke-width="2"/>')
    for i in range(n):
        q = pt(angles[i], r * max(0.01, vals[i] / 7))
        P.append(f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="3" fill="#24589c"/>')
    P.append('</svg>')
    return "".join(P)


def _model_detail(mid: str, means: dict) -> str:
    """每个雷达图下方的因子明细：每行「因子 分数 说明」，覆盖该模型全部因子。"""
    fs = [f for f in ORDER if MODEL_OF[f] == mid and f in means]
    rows = []
    for f in fs:
        m = means[f]
        rows.append(
            '<div style="font-size:11px;color:#333;line-height:1.7;white-space:nowrap;">'
            f'<span style="display:inline-block;width:82px;font-weight:600;color:#123f78;">{_svg_esc(ZH_FACTOR[f])}</span>'
            f'<span style="display:inline-block;width:34px;text-align:right;font-weight:600;">{m:.1f}</span>'
            f'<span style="color:#666;margin-left:8px;">{_svg_esc(human_of(f, m))}</span>'
            '</div>'
        )
    return "".join(rows)


def _grand_summary(means: dict, stable_high, stable_low, volatile) -> str:
    """六件套整体快速总结（全局最高/最低 + 稳定面 + 画像速写）。"""
    items = sorted(means.items(), key=lambda x: -x[1])
    top3 = [(ZH_FACTOR[f], v) for f, v in items[:3]]
    bot3 = [(ZH_FACTOR[f], v) for f, v in items[-3:]]
    line1 = ("全局最高：" + " · ".join(f"{n} {v:.1f}" for n, v in top3)
             + "｜全局最低：" + " · ".join(f"{n} {v:.1f}" for n, v in bot3))
    parts2 = []
    if stable_high:
        parts2.append("稳定高：" + "、".join(ZH_FACTOR[f] for f, _, _ in stable_high))
    if stable_low:
        low_names = [ZH_FACTOR[f] for f, _, _ in stable_low]
        parts2.append("稳定低：" + "、".join(low_names[:6]) + (" 等" if len(low_names) > 6 else ""))
    parts2.append(f"情境波动 {len(volatile)} 项")
    line2 = "稳定面：" + "；".join(parts2)
    if stable_high:
        core = "、".join(ZH_FACTOR[f] for f, _, _ in stable_high[:3])
        sketch = f"底色由 {core} 支撑"
    else:
        sketch = "无稳定突出特质，靠情境切换发力"
    if stable_low:
        sketch += f"；低负担面 {len(stable_low)} 项（首项 {ZH_FACTOR[stable_low[0][0]]} 等）"
    line3 = "画像速写：" + sketch
    return f"{line1}<br>{line2}<br>{line3}"

def _spi_bubble_svg(rows: list, total: int) -> str:
    """SPI 15 聚类气泡图：命中比越高气泡越大、越靠视觉中心；0 命中灰色最小。"""
    W, H, cx, cy = 780, 470, 390, 250

    def radius(pct):
        return 24 + math.sqrt(max(pct, 0.0) / 100.0) * 55

    def color_of(pct):
        if pct <= 0:
            return "#cfcfcf"
        if pct >= 50:
            return "#123f78"
        if pct >= 30:
            return "#24589c"
        if pct >= 10:
            return "#3f7dbf"
        if pct >= 2:
            return "#6ea3d1"
        return "#9dc0e2"

    def txt_color(pct):
        if pct <= 0:
            return "#777"
        return "#fff" if pct >= 10 else "#123f78"

    ranked = sorted(rows, key=lambda r: -r["n_first"])
    placed = []
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<text x="{cx}" y="20" text-anchor="middle" font-size="13" font-weight="bold" fill="#123f78">SPI 15 聚类命中分布</text>')
    out.append(f'<text x="{cx}" y="36" text-anchor="middle" font-size="10.5" fill="#888">气泡越大、越靠中心 = 首命中占比越高（0 命中为灰色）</text>')
    for r_ in ranked:
        pct = r_["n_first"] / total * 100 if total else 0
        rr = radius(pct)
        if not placed:
            x, y = cx, cy
        else:
            theta = 0.0
            while True:
                theta += 0.4
                rad = 6 + 7.0 * theta
                x = cx + rad * math.cos(theta)
                y = cy + rad * math.sin(theta)
                ok = True
                for (px, py), pr in placed:
                    if math.hypot(x - px, y - py) < rr + pr + 7:
                        ok = False
                        break
                if ok:
                    break
        placed.append(((x, y), rr))
        c = color_of(pct)
        tc = txt_color(pct)
        name = CLUSTER_ZH.get(r_["cluster"], r_["label"])
        ttip = f"{name}（{r_['cluster']}）· 首命中 {r_['n_first']} 次（{pct:.1f}%）"
        if r_.get("avg_relevance"):
            ttip += f"· 契合 {r_['avg_relevance']:.1f}"
        out.append("<g>")
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr:.1f}" fill="{c}" fill-opacity="0.85"/>')
        if rr >= 34 and pct > 0:
            out.append(f'<text x="{x:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="{tc}">{_svg_esc(name)}</text>')
            out.append(f'<text x="{x:.1f}" y="{y + 12:.1f}" text-anchor="middle" font-size="10.5" fill="{tc}">{pct:.1f}% · {r_["n_first"]}次</text>')
        elif pct > 0:
            out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" fill="{tc}">{pct:.1f}%</text>')
        else:
            out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" fill="{tc}">0</text>')
        out.append(f"<title>{_svg_esc(ttip)}</title>")
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def build_html(name: str, means: dict, stds: dict, base: dict | None,
               spi: list, stable_high, stable_low, stable_mid, volatile, n_articles: int | None) -> str:
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    H = []
    H.append('<html style="margin:0;padding:0;">')
    H.append(f'<title>{esc(name)} · 人格报告（32因子雷达 + SPI 15聚类 + 解读）</title>')
    H.append('<div style="width:100%;box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,\'PingFang SC\',\'Microsoft YaHei\',sans-serif;padding:16px 20px;">')
    H.append(f'<div style="font-size:17px;font-weight:700;color:#1a1a1a;margin-bottom:2px;">{esc(name)} · 人格报告</div>')
    n_txt = f"{n_articles} 篇全量基线" if n_articles else "画像基线"
    H.append(f'<div style="font-size:12px;color:#777;margin-bottom:14px;">{esc(n_txt)} · 第一模块：六件套 32 因子雷达；第二模块：SPI 15 聚类命中分布（气泡越大越靠中心 = 首命中占比越高）。</div>')

    # ---- 模块一：32 因子雷达（六件套 6 图，3 排 × 每排 2 个，每排底部一条统一隔离线）+ 整体快速总结 ----
    H.append('<div style="margin-bottom:4px;font-size:14px;font-weight:700;color:#1a1a1a;">六件套 32 因子雷达</div>')
    for pair in [("ocean", "mmpi"), ("scm", "dt4"), ("tki", "sdt")]:
        H.append('<div style="display:flex;flex-wrap:wrap;width:100%;border-bottom:1px solid #e0e4ea;padding-bottom:12px;margin-bottom:12px;">')
        for mid in pair:
            H.append('<div style="width:50%;box-sizing:border-box;padding:4px 8px;">')
            H.append(_radar_svg(mid, means))
            H.append(f'<div style="padding:4px 6px;">{_model_detail(mid, means)}</div>')
            H.append('</div>')
        H.append('</div>')
    H.append('<div style="margin-top:2px;padding:10px 14px;background:#f0f5fc;border:1px solid #d5e2f2;border-radius:6px;font-size:12.5px;color:#1a1a1a;line-height:1.8;">')
    H.append(f'<b style="color:#123f78;">六件套快速总结</b><br>{_grand_summary(means, stable_high, stable_low, volatile)}')
    H.append('</div>')

    # ---- ① 核心特质（强度 × 稳定性）：位于快速总结与 SPI 气泡图之间 ----
    H.append('<div style="margin:16px 0 6px;font-size:14px;font-weight:700;color:#1a1a1a;">核心特质（强度 × 稳定性）</div>')
    H.append('<table style="width:100%;border-collapse:collapse;font-size:12.5px;">')
    H.append('<tr style="background:#123f78;color:#fff;"><td style="padding:5px 10px;width:80px;">类别</td><td style="padding:5px 10px;width:120px;">因子</td><td style="padding:5px 10px;width:90px;">分数（σ）</td><td style="padding:5px 10px;">说明</td></tr>')
    cat_meta = [
        ("稳定高", stable_high, "#eaf1fb", "#123f78", "跨主题一致"),
        ("稳定中", stable_mid, "#eef4f8", "#5b6b7c", "稳定居中"),
        ("稳定低", stable_low, "#eef7f0", "#2e6b34", "同样重要"),
        ("情境波动", volatile, "#fdf3e7", "#a0661a", "随主题变化"),
    ]
    for cat, items, bg, fg, sub in cat_meta:
        if not items:
            H.append(_row(
                f'<td style="padding:6px 10px;font-weight:700;color:{fg};vertical-align:middle;">{cat}<br><span style="font-weight:400;font-size:11px;color:#888;">{sub}</span></td>',
                _td("—"), _td("—", "white-space:nowrap;"), _td(f"无{cat}因子"), style=f"background:{bg};"))
            continue
        for i, (f, m, s) in enumerate(items):
            rspan = f' rowspan="{len(items)}"' if i == 0 else ""
            if i == 0:
                first = f'<td{rspan} style="padding:6px 10px;font-weight:700;color:{fg};vertical-align:middle;">{cat}<br><span style="font-weight:400;font-size:11px;color:#888;">{sub}</span></td>'
            else:
                first = ""
            H.append(_row(first, _td(esc(ZH_FACTOR[f])), _td(f"{m:.1f}（σ{s:.2f}）", "white-space:nowrap;"), _td(esc(human_of(f, m))), style=f"background:{bg};"))
    H.append('</table>')

    # ---- 模块二：SPI 15 聚类（气泡图） ----
    rows = sorted(spi, key=lambda r: -r["n_first"])
    total = sum(r["n_first"] for r in rows) or 1
    H.append('<div style="margin:18px 0 10px;font-size:14px;font-weight:700;color:#1a1a1a;">SPI 15 聚类命中分布</div>')
    H.append(_spi_bubble_svg(rows, total))

    # ---- 解读区（② SPI 释义 · ③ 跨人 · ④ 方法学）----
    H.append('<div style="margin-top:18px;padding:14px 16px;background:#f7f9fc;border-left:4px solid #24589c;border-radius:4px;">')

    # ② SPI 15 聚类释义（紧接气泡图下方）
    H.append('<b style="color:#123f78;">SPI 15 聚类释义</b>')
    H.append('<table style="width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px;">')
    H.append('<tr style="background:#123f78;color:#fff;"><td style="padding:5px 10px;width:120px;">聚类</td><td style="padding:5px 10px;">什么意思</td><td style="padding:5px 10px;width:150px;">命中（首命中 · 契合）</td><td style="padding:5px 10px;">观察</td></tr>')
    for r in sorted(spi, key=lambda r: -r["n_first"]):
        cid = r["cluster"]
        pct = r["n_first"] / total * 100
        if r["n_first"] == 0:
            hit = "0 · —"
            obs = "未触发"
        elif pct >= 30:
            hit = f'{r["n_first"]}（{pct:.1f}%）· {r["avg_relevance"]:.1f}'
            obs = "内容主导向"
        elif pct >= 10:
            hit = f'{r["n_first"]}（{pct:.1f}%）· {r["avg_relevance"]:.1f}'
            obs = "伴随主导向"
        elif pct >= 2:
            hit = f'{r["n_first"]}（{pct:.1f}%）· {r["avg_relevance"]:.1f}'
            obs = "伴随出现"
        else:
            hit = f'{r["n_first"]}（{pct:.1f}%）· {r["avg_relevance"]:.1f}'
            obs = "偶发"
        H.append(_row(_td(f"<b>{esc(CLUSTER_ZH.get(cid, cid))}</b>"), _td(esc(CLUSTER_PLAIN.get(cid, ""))),
                      _td(hit, "white-space:nowrap;"), _td(esc(obs))))
    H.append('</table>')

    # ③ 跨人定位 ④ 方法学
    rel = f"<b>跨人相对定位</b>：{esc(name)} = 当前基准；将来任何人画像与之对比，|Δ|≥1.0 记显著偏离。"
    H.append(f'<div style="font-size:12.5px;line-height:1.8;color:#333;margin-top:12px;">{rel}<br>')
    H.append('<b style="color:#123f78;">方法学注记</b>：归纳性描述，非临床诊断；σ&lt;0.8 核心 / σ≥1.2 情境；重测平均绝对差 0.28；跨人比较须同一尺子 v1.0.0。</div>')
    H.append('</div>')
    H.append('<div style="font-size:11px;color:#999;margin-top:10px;">数据口径：六件套 32 因子 × 2 轮 LLM 零样本打分平均；SPI 聚类 = LLM 判聚类（每篇最多 5 命中且 relevance≥4），首命中占比合计 100%，契合=进入该聚类的平均契合分（1–7）。尺子 v1.0.0（2026-10-10 冻结）。</div>')
    H.append('</div></html>')
    return "\n".join(H)


if __name__ == "__main__":
    raise SystemExit(main())
