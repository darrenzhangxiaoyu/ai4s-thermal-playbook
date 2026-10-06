# -*- coding: utf-8 -*-
"""从核验记录生成 `docs/核验报告.md`。

报告不是手写的——它直接由 `docs/verification/*.json` 里逐条记录的裁决、逐字证据与
处理建议生成，避免"报告与证据不一致"。叙述性章节（背景、方法、局限）写在下方常量里。

用法：python tools/build_verification_report.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIF_DIR = REPO / "docs" / "verification"
OUT = REPO / "docs" / "核验报告.md"

DOC_OF_ID = lambda i: ("《AI4S 全方位调研报告》" if i <= 63
                       else "《流体与能源 AI4S 入行指南》" if i <= 76
                       else "《热管理背景 AI4S 专项计划》")
GAP_IDS = {8, 9, 14, 24, 32, 34, 43, 46, 59, 68, 81, 84, 91, 97}
STATUS_LABEL = {"verified": "✅ 已核验", "partially_verified": "⚠️ 部分核验",
                "not_found": "❌ 未找到", "contradicted": "⛔ 有矛盾",
                "dead_link": "🔗 链接失效"}

INTRO = """# 引文核验报告

> 生成方式：`tools/build_verification_report.py`，数据源 `docs/verification/*.json`。
> 核验时间：2026-10-06。

## 一、为什么要做这件事

本仓库的三份调研文档最初由上一代 AI 工具产出。接手检查时发现一个致命问题：

**三份文档正文里带了 93 个 `[^N^]` 引用标记，但参考文献表整份丢失**——没有任何一处引用
能回溯到来源。对一份公开的调研报告来说，"每个数据点都挂着引用、却一个都查不到"比没有
引用更糟糕：它让读者误以为结论有据可依。

进一步检查发现编号在三份文档间**连续且不重叠**（1–63 / 64–76 / 77–107），说明原本存在
一份 **107 条的共享参考文献表**。因此本次接手的第一件事就是把这张表重建出来。

## 二、方法

1. **重建编号映射**：按编号区间切分，逐条从正文中提取该编号支撑的论点原文。
2. **回溯一手来源**：优先官方发布、论文 DOI、公司公告/招股书、监管文件、产品文档；
   二手转述不作为终局依据。
3. **逐字取证**：必须实际打开来源页面并从正文摘录能支撑该论点的**逐字**片段，
   只凭搜索摘要不算核验。
4. **严格裁决**：找不到就判"未找到"，**不凑一条看似相关的引用**。时间敏感数据
   （融资额、营收、市占率、硬件规格、榜单）逐条查证。
5. **修正而非仅标注**：判为"有矛盾/未找到"的表述，在正文中实际改正或删除。

核验记录（含 URL、逐字证据、置信度）保存在 `docs/verification/`，可逐条复核。

## 三、结果总览

| 裁决 | 条数 | 占比 |
|-|-|-|
{verdict_table}

> 统计口径为**正文实际引用的 93 个编号**。另有 14 个编号（{gap_list}）是原表的空号——
> 占了编号但正文从未引用。
"""

SECTIONS = """
## 四、必须修改的问题（逐条）

以下列出所有非"已核验"的条目及其核验结论。这些结论已反映到正文修正中；
未在本次修正到位、仍需作者补充来源的条目列在第五节。

### 4.1 与来源矛盾

{contradicted}

### 4.2 穷尽检索无任何支持来源

{not_found}

### 4.3 来源存在但数字/时间/主体有出入

{partial}

### 4.4 来源已知但当前无法访问

{dead}

## 五、仍需作者补充或人工复核

以下条目本次**未能取得一手证据**，正文已相应改写为不带断言或明确标注存疑的表述，
若作者能提供原始来源，可恢复为事实陈述：

{manual}

## 六、其他可信度风险（非引文问题）

1. **主观判断被写成了事实**。例如"数学/推理方向产业化程度尚低（坐标图左上象限）"是
   作者自制的评估，却与外部引用混排。已在正文中标注为作者观察。
2. **单一编号承载多个互不相关的论点**（如 [6]、[65]）。这会让"引用"变成装饰。
   已在重构时按论点拆分记录。
3. **二手转述链**。§5 的人才供需数字（12 倍 / 0.97 / 0.15 / 73%）只能追到对脉脉报告的
   转载页，数字彼此一致、可信度较高，但正式引用应指向原始报告。
4. **图表中的文字未被检索覆盖**。技术地图（fig5/fig6）里的部分数字来自正文，
   已随正文一并修正并重新生成。

## 七、本次未做的事（明确边界）

- **没有逐条重做全部 93 个论点的独立复核**：核验是"回溯引用 + 逐字取证"，
  不是重新做一遍市场调研。数字之间的口径差异可能仍然存在。
- **没有覆盖图表审美、信息架构与可读性问题**。
- **没有把全部 ⚠️ 条目都追到原始报告**（部分原始报告需付费或需要账号）。
- **未验证的表述仍可能存在于正文**：只要它原本没有引用标记，就不在本次核验范围内。
"""


def load() -> dict[int, dict]:
    recs = {}
    for path in sorted(glob.glob(str(VERIF_DIR / "*.json"))):
        for r in json.loads(Path(path).read_text(encoding="utf-8"))["references"]:
            recs[int(r["id"])] = r
    return recs


def describe(rec: dict) -> str:
    rid = int(rec["id"])
    claim = (rec.get("claim") or "").replace("\n", " ").strip()
    if len(claim) > 150:
        claim = claim[:150] + "…"
    src = rec.get("source") or {}
    bits = [f"**[{rid}]**（{DOC_OF_ID(rid)}）{claim}"]
    if src.get("url"):
        bits.append(f"  - 实际核到的来源：{src.get('title', '—')} — <{src['url']}>")
    if rec.get("notes"):
        bits.append(f"  - 裁决说明：{rec['notes']}")
    if rec.get("evidence_quote"):
        bits.append(f"  - 逐字证据：> {rec['evidence_quote']}")
    return "\n".join(bits)


def render(items: list[dict]) -> str:
    return "\n\n".join(describe(r) for r in items) if items else "_（无）_"


def main() -> int:
    recs = load()
    real = [r for r in recs.values() if int(r["id"]) not in GAP_IDS]
    gaps = sorted(i for i in GAP_IDS if i in recs)
    counts = Counter(r.get("status") for r in real)
    total = len(real)

    rows = []
    for k in ["verified", "partially_verified", "not_found", "contradicted", "dead_link"]:
        n = counts.get(k, 0)
        if n:
            rows.append(f"| {STATUS_LABEL[k]} | {n} | {n/total*100:.1f}% |")
    rows.append(f"| **合计** | **{total}** | 100% |")

    def pick(status):
        return sorted([r for r in real if r.get("status") == status],
                      key=lambda r: int(r["id"]))

    text = INTRO.format(
        verdict_table="\n".join(rows),
        gap_list="、".join(str(g) for g in gaps),
    ) + SECTIONS.format(
        contradicted=render(pick("contradicted")),
        not_found=render(pick("not_found")),
        partial=render(pick("partially_verified")),
        dead=render(pick("dead_link")),
        manual=render(pick("contradicted") + pick("not_found") + pick("dead_link")),
    )
    OUT.write_text(text, encoding="utf-8")
    print(f"写入 {OUT}")
    print(f"  正文引用编号 {total} 条：{dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
