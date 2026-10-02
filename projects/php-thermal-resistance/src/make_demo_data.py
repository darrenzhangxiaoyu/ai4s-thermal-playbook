# -*- coding: utf-8 -*-
"""生成物理启发式 PHP 演示数据集（约 1000 样本）。

内置文献已知规律（用于阶段 1 全流程验证，不代表真实定量结论）：
- 热阻随热流密度增大而下降（幂律），热流密度是最强特征
- 充液率存在最优区间 ~50-60%，过低烧干、过高驱动力不足
- 垂直底热（90°）最优，水平（0°）在低充液率下可能无法启动
- 功率超过临界值（随充液率升高而升高）出现烧干，热阻飙升
- 窄水力直径、多弯折降低热阻
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

# 工质物性（相对热阻因子：潜热高 → 热阻低）
FLUIDS = {
    "water":   {"factor": 0.70, "hfg_kJkg": 2257},
    "ethanol": {"factor": 1.00, "hfg_kJkg": 846},
    "acetone": {"factor": 1.15, "hfg_kJkg": 501},
    "R134a":   {"factor": 1.40, "hfg_kJkg": 217},
}
fluid_names = list(FLUIDS.keys())

# 采样输入
fluid = rng.choice(fluid_names, size=N)
fill_ratio = rng.uniform(0.08, 0.95, size=N)          # 8%–95%
inclination = rng.uniform(0, 90, size=N)               # 0–90°
heat_flux = rng.uniform(2, 60, size=N)                 # W/cm²
dh = rng.uniform(1.0, 3.0, size=N)                     # mm
n_turns = rng.integers(4, 21, size=N).astype(float)
length = rng.uniform(40, 120, size=N)                  # mm 蒸发段长度

rows = []
for i in range(N):
    f = FLUIDS[fluid[i]]
    fr, th, q = fill_ratio[i], inclination[i], heat_flux[i]

    # 1) 功率效应：R ∝ q^-0.55
    r = (q / 15.0) ** -0.55
    # 2) 充液率效应：围绕全局最优值（56%）的二次惩罚
    fr_opt = 0.56
    r *= 1.0 + 2.2 * (fr - fr_opt) ** 2
    # 3) 倾角效应：90° 最优；0° 低充液率出现平滑的启动恶化（确定性，可学习）
    r *= 1.0 - 0.32 * np.sin(np.deg2rad(th))
    r *= 1.0 + 1.2 / (1.0 + np.exp((fr - 0.45) * 6)) * np.exp(-th / 8.0)
    # 4) 烧干：临界功率随充液率弱相关；平滑饱和惩罚（确定性，可学习）
    q_dry = 30.0 + 8.0 * fr
    over = max(0.0, q - q_dry)
    r *= 1.0 + 0.9 * over / (over + 8.0)
    # 5) 几何：大管径（沸腾极限宽松）、多弯折、短蒸发段有利
    r *= (dh[i] / 2.0) ** -0.8
    r *= (n_turns[i] / 10.0) ** -0.2
    r *= np.sqrt(length[i] / 60.0)
    # 6) 工质因子
    r *= f["factor"]
    # 7) 基准量级 + 对数噪声
    R = 0.5 * r * rng.lognormal(0, 0.08)
    rows.append(R)

df = pd.DataFrame({
    "fluid": fluid,
    "fill_ratio_pct": np.round(fill_ratio * 100, 1),
    "inclination_deg": np.round(inclination, 1),
    "heat_flux_wcm2": np.round(heat_flux, 2),
    "hydraulic_diameter_mm": np.round(dh, 2),
    "n_turns": n_turns.astype(int),
    "evaporator_length_mm": np.round(length, 1),
    "R_th_K_W": np.round(rows, 4),
})

out = "data/php_demo.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"saved {len(df)} rows -> {out}")
print(df.describe(include="all").to_string())
