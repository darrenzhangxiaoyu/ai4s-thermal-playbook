# -*- coding: utf-8 -*-
"""把引文核验记录汇编成参考文献表，并把 `[^N^]` 标记改写为标准数字引用。

背景
----
本仓库三份调研文档原先使用 `[^N^]` 标记引用，但**参考文献表整份丢失**：
93 个标记在正文中出现，却没有任何一处引用定义，读者无法回溯任何来源。
核验发现编号在三份文档间连续不重叠（1–63 / 64–76 / 77–107），说明原本
存在一份 **107 条的共享参考文献表**。

本脚本把逐条核验结果（``docs/verification/*.json``）汇编成：

* ``docs/参考文献.md``：完整 107 条共享参考文献表，含核验状态
* 三份文档末尾的 ``## 参考文献`` 小节（只列该文档实际引用的编号）
* 文档正文中 ``[^N^]`` → ``[N]`` 的标准数字引用改写

状态标记
--------
``已核验``    找到一手来源且页面正文逐字支持
``部分核验``  来源存在，但数字/时间/主体与文档表述有出入
``未找到``    穷尽检索无任何支持来源 → 该论点应视为未证实
``有矛盾``    权威来源与文档表述矛盾
``链接失效``  来源已知但当前无法访问

用法：
    python tools/build_bibliography.py            # 生成/更新
    python tools/build_bibliography.py --check     # 只报告，不写文件
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIF_DIR = REPO / "docs" / "verification"
DOCS = [
    REPO / "docs" / "AI4S全方位调研报告.md",
    REPO / "docs" / "流体与能源AI4S入行指南.md",
    REPO / "docs" / "热管理背景AI4S专项计划.md",
]

STATUS_LABEL = {
    "verified": "已核验",
    "partially_verified": "部分核验",
    "not_found": "未找到",
    "contradicted": "有矛盾",
    "dead_link": "链接失效",
}
STATUS_ICON = {
    "verified": "✅", "partially_verified": "⚠️", "not_found": "❌",
    "contradicted": "⛔", "dead_link": "🔗",
}
#: 正文从未出现的编号（原参考文献表的空号）
GAP_IDS = {8, 9, 14, 24, 32, 34, 43, 46, 59, 68, 81, 84, 91, 97}


def load_records() -> dict[int, dict]:
    recs: dict[int, dict] = {}
    for path in sorted(glob.glob(str(VERIF_DIR / "*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for r in data.get("references", []):
            recs[int(r["id"])] = r
    return recs


def is_gap(rec: dict) -> bool:
    return int(rec["id"]) in GAP_IDS or "未在正文" in (rec.get("claim") or "")


def format_entry(rec: dict, compact: bool = False) -> str:
    """把一条记录格式化成参考文献条目。"""
    rid, status = int(rec["id"]), rec.get("status", "not_found")
    icon = STATUS_ICON.get(status, "")
    label = STATUS_LABEL.get(status, status)

    if is_gap(rec):
        return f"**[{rid}]** —（空号：原参考文献表有此编号，但正文未引用）"

    src = rec.get("source") or {}
    title = src.get("title") or "（未找到可援引来源）"
    parts = [f"**[{rid}]** {title}"]
    tail = []
    if src.get("publisher"):
        tail.append(str(src["publisher"]))
    if src.get("date"):
        tail.append(str(src["date"]))
    line = parts[0]
    if tail:
        line += "，" + "，".join(tail)
    if src.get("url"):
        line += f" — <{src['url']}>"
    line += f"  {icon} **{label}**"

    if compact:
        return line

    out = [line]
    if src.get("type"):
        out.append(f"  - 来源类型：{src['type']}｜置信度：{rec.get('confidence', '—')}"
                   f"｜访问日期：{rec.get('accessed', '—')}")
    if rec.get("evidence_quote"):
        out.append(f"  - 逐字证据：> {rec['evidence_quote']}")
    if rec.get("notes"):
        out.append(f"  - 说明：{rec['notes']}")
    return "\n".join(out)


def build_master(recs: dict[int, dict]) -> str:
    real = [r for r in recs.values() if not is_gap(r)]
    gaps = [r for r in recs.values() if is_gap(r)]
    counts = Counter(r.get("status") for r in real)
    lines = [
        "# 参考文献（三份调研文档共享表）",
        "",
        "> 本表由 `tools/build_bibliography.py` 依据 `docs/verification/*.json` 生成。",
        "> 三份文档使用**统一编号**：`AI4S全方位调研报告` 用 1–63，"
        "`流体与能源AI4S入行指南` 用 64–76，`热管理背景AI4S专项计划` 用 77–107。",
        "",
        "## 为什么会有这份表",
        "",
        "本仓库内容最初由上一代 AI 工具产出，三份文档正文里带了 93 个 `[^N^]` 引用标记，",
        "但**参考文献表整份丢失**——每个数据点都挂着引用，却没有任何一处能回溯到来源。",
        "这对一份公开的调研报告是不可接受的。",
        "",
        "本次接手后逐条重建并核验：为每个编号回溯一手来源、用页面正文逐字取证，",
        "并对无法证实或与来源矛盾的表述给出明确裁决。核验记录（含逐字证据）",
        "存放在 `docs/verification/`。",
        "",
        "## 核验结果总览",
        "",
        "| 裁决 | 条数 | 含义 |",
        "|-|-|-|",
    ]
    meaning = {
        "verified": "找到一手来源且页面正文逐字支持",
        "partially_verified": "来源存在，但数字/时间/主体与文档表述有出入",
        "not_found": "穷尽检索无支持来源 → 视为未证实",
        "contradicted": "权威来源与文档表述矛盾",
        "dead_link": "来源已知但当前无法访问",
    }
    for k in ["verified", "partially_verified", "not_found", "contradicted", "dead_link"]:
        if counts.get(k):
            lines.append(f"| {STATUS_ICON[k]} {STATUS_LABEL[k]} | {counts[k]} | {meaning[k]} |")
    lines += [
        "",
        f"**正文实际引用的编号共 {len(real)} 个**（下表统计口径即为这 {len(real)} 条）；",
        f"另有 {len(gaps)} 个编号（{', '.join(str(r['id']) for r in sorted(gaps, key=lambda x: x['id']))}）"
        "是原参考文献表的**空号**——占用了编号但正文从未引用，已在下文逐条标出。",
        "",
        "> ⚠️ **未找到**与**有矛盾**的条目不是「待办事项」，而是结论本身需要修改或删除的信号——",
        "> 正文中相应表述已在本次接手时修正，详见 `docs/核验报告.md`。",
        "",
        "---",
        "",
        "## 条目",
        "",
    ]
    current_doc = None
    ranges = [(1, 63, "《AI4S 全方位调研报告》"),
              (64, 76, "《流体与能源 AI4S 入行指南》"),
              (77, 107, "《热管理背景 AI4S 专项计划》")]
    for lo, hi, name in ranges:
        lines.append(f"### {name}（{lo}–{hi}）")
        lines.append("")
        for i in range(lo, hi + 1):
            rec = recs.get(i)
            if rec is None:
                lines.append(f"**[{i}]** —（本次核验未覆盖此编号）")
                lines.append("")
                continue
            lines.append(format_entry(rec))
            lines.append("")
    return "\n".join(lines)


def build_doc_section(recs: dict[int, dict], ids: list[int]) -> str:
    lines = ["", "---", "", "## 参考文献", "",
             "> 编号与三份文档共享的 `docs/参考文献.md` 保持一致。",
             "> 标记：✅ 已核验 ｜ ⚠️ 部分核验（数字/口径与来源有出入）｜ "
             "❌ 未找到来源（视为未证实）｜ ⛔ 与来源矛盾。",
             "> 本次接手对引文的核验方法、结果与修正记录见 `docs/核验报告.md`。", ""]
    for i in sorted(ids):
        rec = recs.get(i)
        if rec is None:
            lines.append(f"**[{i}]** —（未核验）")
            continue
        lines.append(format_entry(rec, compact=True))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    recs = load_records()
    missing = [i for i in range(1, 108) if i not in recs]
    print(f"载入 {len(recs)} 条核验记录；未覆盖编号：{missing or '无'}")

    master = build_master(recs)
    master_path = REPO / "docs" / "参考文献.md"
    if args.check:
        print("[check] 未写入任何文件")
    else:
        master_path.write_text(master, encoding="utf-8")
        print(f"写入 {master_path}")

    # 逐份文档：改写标记 + 追加参考文献小节
    marker_re = re.compile(r"\[\^(\d+)\^\]")
    entry_re = re.compile(r"\*\*\[(\d+)\]\*\*")
    tail_re = re.compile(r"\n---\n\n## 参考文献\n")
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        # 首次运行时从 [^N^] 标记取编号；已转换过的文档则从既有参考文献小节取，
        # 保证脚本可重复运行且结果稳定（幂等）。
        ids = sorted({int(m) for m in marker_re.findall(text)})
        if not ids:
            m = tail_re.search(text)
            if m:
                ids = sorted({int(x) for x in entry_re.findall(text[m.end():])})
        if not ids:
            print(f"  {doc.name}: 未发现引用编号，跳过")
            continue
        # 去掉旧的自动生成小节，保证脚本可重复运行
        text = tail_re.split(text)[0].rstrip() + "\n"
        text = marker_re.sub(lambda mm: f"[{mm.group(1)}]", text)
        text += build_doc_section(recs, ids)
        if not args.check:
            doc.write_text(text, encoding="utf-8")
        print(f"  {doc.name}: 参考文献 {len(ids)} 条 "
              f"（{min(ids)}–{max(ids)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
