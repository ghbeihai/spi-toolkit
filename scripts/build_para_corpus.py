#!/usr/bin/env python3
"""通用段落式语料构建：\ufeff 分隔的自然段落（无编号）→ 按 >=MIN 字聚合成篇。
用法: build_para_corpus.py <ATT路径> <OUT路径> <人名> [MIN字]"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ATT = OUT = NAME = None
MIN_LEN = 700


def _parse_args():
    global ATT, OUT, NAME, MIN_LEN
    ap = argparse.ArgumentParser(description="段落式语料 → 按 >=MIN 字聚合成篇")
    ap.add_argument("att", help="原始段落文本路径")
    ap.add_argument("out", help="输出 JSON 路径")
    ap.add_argument("name", help="人名")
    ap.add_argument("min_len", nargs="?", type=int, default=700, help="每篇最少字数（默认 700）")
    a = ap.parse_args()
    ATT, OUT, NAME, MIN_LEN = Path(a.att), Path(a.out), a.name, a.min_len


def main():
    _parse_args()
    t = ATT.read_text(encoding="utf-8")
    t = t.replace("\ufeff", "\n").replace("\xa0", " ").replace("\u200b", "").replace("\u3000", " ")
    lines = [ln.strip() for ln in t.splitlines()]

    # 定位正文起点（账号明细之后）
    start = 0
    for i, ln in enumerate(lines):
        if "手写文案" in ln or "朋友圈" in ln and i > 0:
            start = i + 1
            break
    paras = [ln for ln in lines[start:] if ln]
    print(f"[{NAME}] 段落数: {len(paras)}，总字数 {sum(len(x) for x in paras):,}")

    arts, buf = [], []
    for p in paras:
        buf.append(p)
        if sum(len(x) for x in buf) >= MIN_LEN:
            arts.append({"id": f"{NAME[:2]}_{len(arts)}", "title": f"手写文案·{len(arts)+1}",
                         "text": "\n".join(buf)})
            buf = []
    if buf:
        arts.append({"id": f"{NAME[:2]}_{len(arts)}", "title": "手写文案·末段", "text": "\n".join(buf)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(arts, ensure_ascii=False, indent=1), encoding="utf-8")
    chars = sum(len(a["text"]) for a in arts)
    print(f"[{NAME}] 共 {len(arts)} 篇，总字数 {chars:,}")
    for a in arts[:4]:
        print(f"  [{a['id']}] {a['title']}: {len(a['text'])}字")
    if len(arts) > 4:
        print("  ...")
        for a in arts[-2:]:
            print(f"  [{a['id']}] {a['title']}: {len(a['text'])}字")
    lens = sorted(len(a["text"]) for a in arts)
    print("字数分布: min", lens[0], "| 中位", lens[len(lens)//2], "| max", lens[-1])

if __name__ == "__main__":
    main()
