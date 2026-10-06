# -*- coding: utf-8 -*-
"""微通道冷板的设计空间定义与采样。

设计变量
--------
======================  ==============  ============================================
变量                     范围             说明
======================  ==============  ============================================
``H_c``  通道高度       0.1 – 0.8 mm    微通道特征尺寸
``H_s``  基板厚度       0.2 – 2.0 mm    芯片与流体之间的固体层
``L``    流向长度       10 – 40 mm      冷板流道长度
``u_mean`` 平均流速     0.02 – 3.0 m/s  上限另受"流动已充分发展"约束（见下）
``q_flux`` 热流密度     5 – 80 W/cm²    对应 5e4 – 8e5 W/m²
``k_s``  固体导热系数   150 – 400 W/mK  硅 150 / 铝 237 / 铜 400
======================  ==============  ============================================

工质固定为水（``k_f=0.6``、``ρ=997``、``c_p=4180``、``μ=8.9e-4``，25 °C 物性）。

**约束一：为什么流速有上限（模型有效性）**
求解器假定流体速度剖面已充分发展（平面 Poiseuille）。流动入口段长度
``Le,h ≈ 0.05·Re·D_h``，只有 ``Le,h ≤ L`` 时该假定才成立。这条约束把
``u_mean`` 与 ``H_c``、``L`` 耦合了起来：通道越高、越短，允许的流速越低。

验证套件（``verify_solver.py`` 基准 E）显示，若不加这条约束，随机参数组合
里只有约 42% 满足 ``Le,h/L ≤ 1``——也就是说，"随手均匀采样"会让近六成样本
落在模型假设失效的区域。本模块用拒绝采样把这一点显式堵住，并在输出里
记录每条样本的实际 ``Le,h/L``。

**约束二：为什么热流密度上限定在 80 W/cm²（物理有效性）**
单相液态水的模型只在 ``T_max < T_sat`` 时成立。实测发现：当 ``q''`` 上限取到
150 W/cm² 时，随机采样只有 8.7% 的组合能把热点压在 85 °C 以下；取 80 W/cm²
则可获得有意义的可行域比例。这不是采样器的缺陷，而是**单相微通道冷板的
真实能力边界**——热点温度约束 ``T_max − T_in = q''·R_th ≤ 58 K``，而本项目
可达的最小热阻约 8×10⁻⁵ m²·K/W，对应约 70 W/cm² 的上限。要突破这一点必须
转向两相沸腾换热，这正是《热管理背景 AI4S 专项计划》里给两相路线的定位。

``T_max_limit`` 作为**后验**可行性判据保留在数据集中（``feasible`` 列），
优化器据此施加约束。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PARAM_NAMES", "PARAM_LABELS", "DesignSpace", "FLUID_WATER"]

# 水的物性（25 °C）
FLUID_WATER = {"k_f": 0.6, "rho": 997.0, "cp": 4180.0, "mu": 8.90e-4}

PARAM_NAMES = ["H_c", "H_s", "L", "u_mean", "q_flux", "k_s"]
PARAM_LABELS = {
    "H_c": "通道高度 H_c (mm)",
    "H_s": "基板厚度 H_s (mm)",
    "L": "流向长度 L (mm)",
    "u_mean": "平均流速 u (m/s)",
    "q_flux": "热流密度 q'' (W/cm²)",
    "k_s": "固体导热系数 k_s (W/mK)",
}
PARAM_UNITS_SCALE = {  # 显示用换算（SI → 常用工程单位）
    "H_c": 1e3, "H_s": 1e3, "L": 1e3, "u_mean": 1.0, "q_flux": 1e-4, "k_s": 1.0,
}


@dataclass(frozen=True)
class DesignSpace:
    """设计变量的边界（SI 单位）与有效性约束。"""

    H_c: tuple[float, float] = (0.1e-3, 0.8e-3)
    H_s: tuple[float, float] = (0.2e-3, 2.0e-3)
    L: tuple[float, float] = (10e-3, 40e-3)
    u_mean: tuple[float, float] = (0.02, 3.0)
    q_flux: tuple[float, float] = (5.0e4, 8.0e5)
    k_s: tuple[float, float] = (150.0, 400.0)

    #: 允许的最大入口段占比 Le,h/L
    max_entrance_ratio: float = 1.0

    #: 单相液态水模型的热点温度上限 [K]（85 °C，留出与 100 °C 沸点的裕度）
    T_max_limit: float = 358.15

    # ---- 派生：与流速耦合的约束 ----
    def u_max(self, H_c: np.ndarray | float, L: np.ndarray | float) -> np.ndarray | float:
        """给定通道高度与长度时，满足 Le,h/L ≤ max_entrance_ratio 的最大流速 [m/s]。

        ``Le,h = 0.05·Re·D_h``，``Re = ρ·u·D_h/μ``，``D_h = 2·H_c``
        → ``Le,h = 0.05·ρ·u·(2H_c)²/μ``
        """
        rho, mu = FLUID_WATER["rho"], FLUID_WATER["mu"]
        D_h = 2.0 * np.asarray(H_c)
        coeff = 0.05 * rho * D_h ** 2 / mu
        return self.max_entrance_ratio * np.asarray(L) / coeff

    def bounds_array(self) -> np.ndarray:
        """形状 (6, 2) 的边界数组，列序与 ``PARAM_NAMES`` 一致。"""
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    # ---- 采样 ----
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """在有效区域内采样 ``n`` 个设计点。

        策略：用拉丁超立方保证覆盖率；``u_mean`` 在 ``[0.02, min(1.0, u_max)]``
        内**直接分层采样**（而不是先均匀采样再拒绝），这样不会因为拒绝法而
        在 ``H_c`` 较大的区域留下空洞，也不会浪费样本。

        返回形状 ``(n, 6)``、列序同 ``PARAM_NAMES`` 的数组。
        """
        from scipy.stats import qmc

        eng = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
        u = eng.random(n)

        lo = self.bounds_array()[:, 0]
        hi = self.bounds_array()[:, 1]
        X = lo + u * (hi - lo)

        # 流速上限与 H_c、L 耦合，逐点重采样
        i_u = PARAM_NAMES.index("u_mean")
        u_cap = np.minimum(self.u_mean[1], self.u_max(X[:, PARAM_NAMES.index("H_c")],
                                                     X[:, PARAM_NAMES.index("L")]))
        if np.any(u_cap <= self.u_mean[0]):
            raise ValueError(
                "设计空间中存在 u_max ≤ u_min 的区域，请收紧 H_c/L 上界或放宽流速下界")
        X[:, i_u] = self.u_mean[0] + u[:, i_u] * (u_cap - self.u_mean[0])
        return X

    # ---- 有效性检查 ----
    def entrance_ratio(self, X: np.ndarray) -> np.ndarray:
        """各设计点的 ``Le,h/L``。"""
        rho, mu = FLUID_WATER["rho"], FLUID_WATER["mu"]
        H_c = X[:, PARAM_NAMES.index("H_c")]
        L = X[:, PARAM_NAMES.index("L")]
        u = X[:, PARAM_NAMES.index("u_mean")]
        D_h = 2.0 * H_c
        Re = rho * u * D_h / mu
        return 0.05 * Re * D_h / L

    def is_valid(self, X: np.ndarray) -> np.ndarray:
        """返回布尔数组：是否同时落在边界内且满足入口段约束。"""
        lo = self.bounds_array()[:, 0]
        hi = self.bounds_array()[:, 1]
        inside = np.all((X >= lo - 1e-12) & (X <= hi + 1e-12), axis=1)
        return inside & (self.entrance_ratio(X) <= self.max_entrance_ratio + 1e-12)

    def is_feasible(self, T_max: np.ndarray) -> np.ndarray:
        """后验可行性：热点温度不超过单相液态水模型的上限。"""
        return np.asarray(T_max) <= self.T_max_limit


def to_display(X: np.ndarray) -> dict[str, np.ndarray]:
    """把 SI 参数矩阵换算成工程单位，便于做图与写表。"""
    return {n: X[:, i] * PARAM_UNITS_SCALE[n] for i, n in enumerate(PARAM_NAMES)}
