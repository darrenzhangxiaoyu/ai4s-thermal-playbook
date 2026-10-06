# -*- coding: utf-8 -*-
"""流体与能源 AI4S 方向技术地图（fig5）"""
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
import matplotlib.patches as mpatches  # noqa: E402

cjkfont.setup_cjk_font()

INK, BLUE, TEAL, AMBER, CORAL, GRAY, LIGHT = (
    '#264653', '#2E5EAA', '#2A9D8F', '#E9C46A', '#E76F51', '#8A94A6', '#F4F6F9')
# 输出到仓库内 docs/assets/，与脚本所在位置无关（原实现写死了作者机器的绝对路径）
OUT = str(cjkfont.repo_root() / 'docs' / 'assets')

fig, ax = plt.subplots(figsize=(12.5, 7.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('流体与能源 AI4S 技术地图：两条主线 × 四个应用环节',
             fontsize=17, fontweight='bold', color=INK, pad=16)

cols = ['模拟加速\n（代理模型/ROM）', '物理建模\n（闭合/本构）', '控制优化\n（RL/反问题）', '监测预测\n（预警/预报）']
rows = [
    ('流体主线（CFD/湍流）', BLUE, [
        'FNO / DeepONet 代理求解器\n扩散模型流场重建\nPOD/Krylov 降阶模型\n（STAR-CCM+ ROM 提速 6×）',
        'ML 湍流闭合（RANS/LES）\n符号回归修正 SA/SST 模型\n西北工大张伟伟团队系列工作\nJFM 2016 嵌入不变量 DNN',
        '强化学习流动控制\n翼型/机翼气动优化\n伴随+主动学习设计空间探索\nHVAC RL 控制节能（Chalmers）',
        '流场超分辨率重建\n实验数据同化\n数字孪生"机理+数据"融合\n传感布局优化',
    ]),
    ('能源主线（聚变/风/储/油气）', CORAL, [
        '聚变等离子体模拟器 TORAX\n（DeepMind 开源，JAX）\n风电场尾流快速模拟\n电池/电解槽数字孪生',
        'Landau 流体闭合 ML 代理\n等离子体输运建模\n燃烧化学机理加速\n（DeepFlame 开源平台）',
        'RL 控制托卡马克磁体\n（Nature 2022，TCV 装置）\nEAST 装置 βp 深度强化学习控制\nSPARC "AI 驾驶员"（目标 2027 净增益）',
        'HL-2A/HL-3 破裂实时预测\n（深度学习已上机运行）\n风功率预测\n设备预测性维护',
    ]),
]

x0, colw, gap = 0.145, 0.208, 0.008
rowh, rowy = 0.335, [0.52, 0.13]
for ci, c in enumerate(cols):
    x = x0 + ci * (colw + gap)
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.885), colw, 0.075,
                 boxstyle='round,pad=0.004', fc=INK, ec='none', transform=ax.transAxes))
    ax.text(x + colw / 2, 0.922, c, ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', transform=ax.transAxes, linespacing=1.4)

for (rname, rcolor, cells), ry in zip(rows, rowy):
    ax.add_patch(mpatches.FancyBboxPatch((0.008, ry), 0.125, rowh,
                 boxstyle='round,pad=0.004', fc=rcolor, ec='none', transform=ax.transAxes))
    ax.text(0.0705, ry + rowh / 2, rname, ha='center', va='center', fontsize=12.5,
            fontweight='bold', color='white', transform=ax.transAxes, rotation=90)
    for ci, cell in enumerate(cells):
        x = x0 + ci * (colw + gap)
        ax.add_patch(mpatches.FancyBboxPatch((x, ry), colw, rowh,
                     boxstyle='round,pad=0.004', fc=LIGHT, ec=rcolor, lw=1.2, transform=ax.transAxes))
        ax.text(x + colw / 2, ry + rowh / 2, cell, ha='center', va='center', fontsize=9.2,
                color=INK, transform=ax.transAxes, linespacing=1.75)

fig.text(0.5, 0.015, '注：内容为公开资料调研整理，括号内为代表性系统/数据点',
         ha='center', fontsize=9, color=GRAY)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(f'{OUT}/fig5_fluid_energy_map.png', dpi=180, bbox_inches='tight')
print('fig5 done')
