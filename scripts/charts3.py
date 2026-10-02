# -*- coding: utf-8 -*-
"""热管理 AI4S 方向技术地图（航天器热控 × 芯片液冷）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches

FONT_DIR = '/Users/zhangxiaoyu/Library/Application Support/kimi-desktop/daimon-share/daimon/runtime/python/fonts'
for f in ['NotoSansSC-Regular.ttf', 'NotoSansSC-Bold.ttf']:
    font_manager.fontManager.addfont(f'{FONT_DIR}/{f}')
plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False

INK, BLUE, TEAL, AMBER, CORAL, GRAY, LIGHT = (
    '#264653', '#2E5EAA', '#2A9D8F', '#E9C46A', '#E76F51', '#8A94A6', '#F4F6F9')
OUT = '/Users/zhangxiaoyu/Documents/02-研发/01-研发项目/50-AI4S/docs/assets'

fig, ax = plt.subplots(figsize=(12.5, 7.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('热管理 AI4S 技术地图：航天器热控 × 芯片液冷 双主线 × 四个应用环节',
             fontsize=16.5, fontweight='bold', color=INK, pad=16)

cols = ['快速设计仿真\n（代理模型/降阶）', '物理建模\n（关联式/两相/闭合）', '反问题与状态估计\n（参数辨识/场重建）', '运行优化\n（数字孪生/智能温控）']
rows = [
    ('航天器热控主线', BLUE, [
        'SMS-DL：NX TMG 批处理造数据\n+ 贝叶斯优化 DNN 代理\n（快 1000+ 倍，精度 99%+）\nNASA HEEDS + Simcenter 3D SST\n多物理 surrogate 工作流',
        '集总参数热模型（LPTM）\n图结构物理约束学习\n太阳辐射热流 / 主动热控嵌入\nPolimi MOX × Thales Alenia 路线\n（无需高保真几何）',
        '热真空试验数据同化\n热参数辨识代理模型 + 遗传算法\n（TMM 参数标定）\n9 传感器 → 14400 点温度场\nMLP 重建（预测 0.83 ms）',
        '卫星热数字孪生综述\n（Acta Astronautica 2025）\nISS 快速热分析数字孪生\n混合孪生：MLP + Modelica/FMI\n稀疏遥测 + 物理约束在线估计\n微重力两相流型生成模型\n（NASA EPSCoR）',
    ]),
    ('芯片热管理主线', CORAL, [
        'ML 预测 Nu/f 统一关联式\n（任意截面通道，Purdue）\n几何 CNN 处理非结构网格\nNeural Concept 表面温度场代理\nCAD → 仿真工作流集成',
        '微/迷你通道冷凝与两相\n压降 ML 通用关联式\n（Springer 2023）\n微通道环状两相流模型\n（Mudawar 谱系）',
        '池沸腾/流动沸腾 CHF\nML 预测（FBCE 跨重力 ANN）\n沸腾状态识别 CNN+迁移学习\n燃气轮机/半导体 PINN 反演\n对流换热系数（稀疏噪声反问题）',
        'TDP 300W→2300W→3600W\n（Vera Rubin 演进）\nRubin NVL72 100% 全液冷\n两相冷板应对 300 W/cm²\n清华 SiC 微通道 CHF 1682 W/cm²',
    ]),
]

x0, colw, gap = 0.145, 0.208, 0.008
rowh, rowy = 0.355, [0.50, 0.095]
for ci, c in enumerate(cols):
    x = x0 + ci * (colw + gap)
    ax.add_patch(mpatches.FancyBboxPatch((x, 0.885), colw, 0.075,
                 boxstyle='round,pad=0.004', fc=INK, ec='none', transform=ax.transAxes))
    ax.text(x + colw / 2, 0.922, c, ha='center', va='center', fontsize=10.5,
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
        ax.text(x + colw / 2, ry + rowh / 2, cell, ha='center', va='center', fontsize=8.8,
                color=INK, transform=ax.transAxes, linespacing=1.65)

fig.text(0.5, 0.015, '注：内容为公开资料调研整理，括号内为代表性系统/数据点',
         ha='center', fontsize=9, color=GRAY)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(f'{OUT}/fig6_thermal_map.png', dpi=180, bbox_inches='tight')
print('fig6 done')
