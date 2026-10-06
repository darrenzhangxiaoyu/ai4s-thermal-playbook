# -*- coding: utf-8 -*-
"""AI4S 调研报告图表（fig1–fig4）：统一配色风格"""
import sys
from pathlib import Path

# 共享的中文字体/路径工具；必须在 import matplotlib 之前导入（它会设置 MPLCONFIGDIR）
for _p in Path(__file__).resolve().parents:
    if (_p / 'tools' / 'cjkfont.py').exists():
        sys.path.insert(0, str(_p / 'tools'))
        break
import cjkfont  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

cjkfont.setup_cjk_font()

INK, BLUE, TEAL, AMBER, CORAL, GRAY, LIGHT = (
    '#264653', '#2E5EAA', '#2A9D8F', '#E9C46A', '#E76F51', '#8A94A6', '#F4F6F9')

# 输出到仓库内 docs/assets/，与脚本所在位置无关（原实现写死了作者机器的绝对路径）
OUT = str(cjkfont.repo_root() / 'docs' / 'assets')

def style_ax(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=11)
    ax.set_facecolor('white')

# ============ 图1：市场规模 ============
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
panels = [
    (axes[0], ['2025', '2032E'], [45.38, 262.3], '亿美元', BLUE, 'CAGR 28.9%', '全球 AI4S 市场规模'),
    (axes[1], ['2025', '2030E'], [51, 107], '亿元人民币', TEAL, 'CAGR 16%', '中国高校科研机构 AI4S 云市场'),
]
for ax, cats, vals, unit, color, cagr, title in panels:
    bars = ax.bar(cats, vals, width=0.5, color=color, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f'{v:g}', ha='center',
                fontsize=14, fontweight='bold', color=color)
    ax.plot([0, 1], vals, ls='--', color=CORAL, lw=2, zorder=4)
    ax.text(0.5, max(vals) * 0.62, cagr, ha='center', fontsize=13, fontweight='bold', color=CORAL)
    ax.set_title(title, fontsize=14, color=INK, pad=12)
    ax.set_ylabel(unit, fontsize=11, color=GRAY)
    ax.set_ylim(0, max(vals) * 1.22)
    style_ax(ax)
    ax.grid(axis='y', color=LIGHT, lw=1, zorder=0)
fig.suptitle('')
fig.text(0.5, 0.015, '数据来源：QYResearch（2026.4）；沙利文《中国高校科研机构AI4S市场追踪报告，2025》',
         ha='center', fontsize=9, color=GRAY)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(f'{OUT}/fig1_market.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# ============ 图2：里程碑时间线 ============
miles = [
    (2017, '深度势能方法提出\n（鄂维南团队）', 1, TEAL),
    (2018, '鄂维南首次提出\n“AI for Science”', -1, TEAL),
    (2020, 'AlphaFold2 破解蛋白质折叠\nDeePMD 获戈登贝尔奖', 1, BLUE),
    (2021, '北京科学智能研究院(AISI)\n全球首家 AI4S 新型研发机构', -1, BLUE),
    (2023, '科技部+自然科学基金委启动 AI4S\n专项部署；GNoME 发现 220 万种候选晶体', 1, BLUE),
    (2024, 'AlphaFold3 发布；AlphaFold 获\n诺贝尔化学奖；晶泰科技港股上市', -1, CORAL),
    (2025, '科学大模型爆发（Intern-S1 开源）\n英矽智能上市；OpenAI 组建 for Science 团队', 1, CORAL),
    (2026, '“AI制药三小龙”齐聚港股；\nAlphaProof Nexus 攻克 Erdos 难题；\nAI Scientist 通过同行评审', -1, INK),
]
fig, ax = plt.subplots(figsize=(11.6, 6.2))
ax.set_title('AI4S 发展里程碑（2017—2026）', fontsize=17, fontweight='bold', color=INK, pad=16)
ax.axhline(0, color=GRAY, lw=3, zorder=1)
shift = {2023: -0.55, 2025: 0.55, 2024: -0.5}
for year, label, side, color in miles:
    ax.scatter([year], [0], s=140, color=color, zorder=3, edgecolors='white', linewidths=1.5)
    lx = year + shift.get(year, 0)
    ax.annotate(label, xy=(year, 0), xytext=(lx, side * 0.55), ha='center',
                va='bottom' if side > 0 else 'top', fontsize=9.5, color=INK, linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=color, lw=1.4),
                arrowprops=dict(arrowstyle='-', color=color, lw=1.2,
                                connectionstyle=f'arc3,rad={0.25 if shift.get(year, 0) else 0}'), zorder=2)
for y in range(2017, 2027):
    ax.text(y, -1.45, str(y), ha='center', fontsize=12, color=INK)
ax.set_xlim(2016.2, 2026.9)
ax.set_ylim(-1.6, 1.6)
ax.axis('off')
fig.tight_layout()
fig.savefig(f'{OUT}/fig2_timeline.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# ============ 图3：领域布局气泡图 ============
fields = [
    ('生命科学/制药', 9.0, 9.4, 2400, CORAL, 'AlphaFold3 · Isomorphic Labs\n晶泰/英矽/剂泰'),
    ('材料科学', 8.0, 9.0, 2000, BLUE, 'GNoME · MatterGen · A-Lab\nMatterSim · MatChatt'),
    ('气象/气候', 7.2, 8.15, 1600, TEAL, 'GraphCast · Pangu · 伏羲\nAIFS · Aurora'),
    ('数学/推理', 3.6, 9.3, 1300, AMBER, 'AlphaProof · AlphaGeometry\nDeepSeek-Prover'),
    ('物理/流体/能源', 5.6, 7.8, 1200, INK, 'PINN · FNO 神经算子\nDeePMD 深度势能'),
    ('化学/自主实验室', 6.4, 7.0, 1050, GRAY, 'Coscientist · ChemCrow\nAI Scientist'),
    ('科学大模型/平台', 5.2, 8.6, 1750, '#B56576', 'Intern-S1 · 磐石\n玻尔平台 · SciMaster'),
]
fig, ax = plt.subplots(figsize=(11.5, 6.2))
ax.set_title('AI4S 主要领域布局（气泡大小 ≈ 赛道热度；坐标为定性评估）',
             fontsize=16, fontweight='bold', color=INK, pad=14)
for name, x, y, s_, color, note in fields:
    ax.scatter([x], [y], s=s_, color=color, alpha=0.82, zorder=3)
    ax.text(x, y + 0.18 + (s_ ** 0.5) * 0.0044, name, ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color=color, zorder=4)
    ax.annotate(note, xy=(x, y), xytext=(16, 0), textcoords='offset points',
                ha='left', va='center', fontsize=9.5, color=INK,
                bbox=dict(boxstyle='round,pad=0.4', fc=LIGHT, ec='none', alpha=0.95), zorder=4)
ax.set_xlabel('产业化 / 商业化程度 →', fontsize=12, color=INK)
ax.set_ylabel('学术影响力 / 技术成熟度 →', fontsize=12, color=INK)
ax.set_xlim(2, 12.9)
ax.set_ylim(6, 10.3)
style_ax(ax)
ax.grid(color=LIGHT, lw=1)
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_fields.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# ============ 图4：零基础学习路径 ============
stages = [
    ('第 1 阶段：筑基', '约 0—3 个月', TEAL,
     ['Python 编程与 Linux/Git', '线性代数 · 微积分 · 概率统计', '经典复现：NumPy/Pandas 小项目']),
    ('第 2 阶段：AI 核心', '约 3—6 个月', BLUE,
     ['机器学习（吴恩达/李沐）', '深度学习与 PyTorch', 'Transformer 与大模型', '    基础（HuggingFace）']),
    ('第 3 阶段：科学智能方法', '约 6—12 个月', CORAL,
     ['科学机器学习：PINN /', '    神经算子 FNO', '图神经网络：分子与晶体表征', '工具链：RDKit · ASE ·', '    DeePMD-kit · NeuralOperator']),
    ('第 4 阶段：领域深耕', '12 个月以后', INK,
     ['选定方向：生信/材料/', '    气象/数学 其一', '复现顶刊论文，参与开源社区', '参加竞赛（Kaggle/天池），', '    产出作品集']),
]
fig, ax = plt.subplots(figsize=(12, 4.7))
ax.set_title('零基础进入 AI4S 的四阶段学习路径', fontsize=17, fontweight='bold', color=INK, pad=14)
W, Hh, top = 0.235, 0.62, 0.16
for i, (title, sub, color, items) in enumerate(stages):
    x = 0.015 + i * (W + 0.022)
    ax.add_patch(plt.Rectangle((x, top), W, Hh, fc='white', ec=color, lw=2, transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((x, top + Hh - 0.135), W, 0.135, fc=color, transform=ax.transAxes))
    ax.text(x + W / 2, top + Hh - 0.052, title, ha='center', va='center', fontsize=12.5,
            fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(x + W / 2, top + Hh - 0.108, f'（{sub}）', ha='center', va='center', fontsize=10,
            color='white', transform=ax.transAxes)
    for j, it in enumerate(items):
        if it.startswith('    '):
            continue
        num = sum(1 for t in items[:j + 1] if not t.startswith('    '))
        ax.text(x + 0.012, top + Hh - 0.185 - j * 0.088, f'{num}. {it}', fontsize=9.3,
                color=INK, va='center', transform=ax.transAxes)
        if j + 1 < len(items) and items[j + 1].startswith('    '):
            ax.text(x + 0.032, top + Hh - 0.185 - (j + 1) * 0.088, items[j + 1].strip(),
                    fontsize=9.3, color=INK, va='center', transform=ax.transAxes)
    if i < 3:
        ax.annotate('', xy=(x + W + 0.024, top + Hh / 2), xytext=(x + W + 0.002, top + Hh / 2),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=2.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.tight_layout()
fig.savefig(f'{OUT}/fig4_roadmap.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print('all figures done')
