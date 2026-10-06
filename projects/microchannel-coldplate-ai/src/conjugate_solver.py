# -*- coding: utf-8 -*-
"""二维稳态共轭传热的有限体积求解器（微通道冷板单元胞）。

物理模型
--------
计算域取冷板的一个重复单元（单位深度、二维）：

    y = H_s + H_c  ──────────────────────────────  绝热（相邻通道间的对称面）
                    ░░░░░ 流体通道 ░░░░░  u(y) 抛物型
    y = H_s        ──────────────────────────────  固-流耦合界面（温度、热流连续）
                    ▓▓▓▓▓ 固体基板 ▓▓▓▓▓
    y = 0          ──────────────────────────────  芯片加热面（热流 q''）
                   x = 0 ───────────────────► x = L

控制方程（稳态、常物性、不可压、忽略黏性耗散）：

    固体：  k_s ∇²T = 0
    流体：  ρ c_p u(y) ∂T/∂x = k_f ∇²T

流体速度取**充分发展的平面 Poiseuille 剖面**：

    u(η) = 6·u_m·η(1-η),   η = (y − H_s)/H_c

该假设的依据与检验：微通道的流体动力学入口段长度满足
``Le/D_h ≈ 0.05·Re·Pr``。求解器会同时返回该估计值，``verify_solver.py``
中的基准算例会打印入口段占通道长度的比例，以说明"充分发展"假设在
本项目使用的参数区间内是否成立（不成立时应当缩小参数区间或改用
全 Navier-Stokes 求解，而不是默默忽略）。

数值方法
--------
结构化网格上的单元中心有限体积离散：

- **对流项**：一阶迎风。流动方向恒为 +x（u ≥ 0），迎风格式稳定、无振荡。
- **扩散项**：界面导热系数取**调和平均**，从而在固-流界面（导热系数可差
  3 个数量级）上正确给出界面热阻。
- **线性系统**：``scipy.sparse.linalg.spsolve`` 直接求解。
- **网格**：允许固/流两侧使用不同的单元格厚度（否则 H_s/H_c 只能取有理比）。

单位一律为 SI（m、K、W、kg）。热流密度 ``q_flux`` 为 W/m²（1 W/cm² = 1e4 W/m²）。

参考
----
- Shah & London, *Laminar Flow Forced Convection in Ducts*, 1978
  （平行平板单侧定热流充分发展 Nu = 5.385）
- Patankar, *Numerical Heat Transfer and Fluid Flow*, 1980（有限体积离散）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

__all__ = ["ColdPlateParams", "ColdPlateSolution", "solve_conjugate"]


@dataclass
class ColdPlateParams:
    """冷板单元胞的几何、工况与物性参数（SI 单位）。"""

    # 几何
    H_s: float = 1.0e-3      # 固体基板厚度 [m]
    H_c: float = 0.5e-3      # 流体通道高度 [m]
    L: float = 20.0e-3       # 流向长度 [m]

    # 工况
    u_mean: float = 1.0      # 通道截面平均流速 [m/s]
    q_flux: float = 5.0e5    # 芯片加热热流密度 [W/m²]
    T_in: float = 300.0      # 入口温度 [K]
    heated_fraction: float = 1.0   # 加热区占流向长度的比例（居中），(0, 1]

    # 物性
    k_s: float = 400.0       # 固体导热系数 [W/(m·K)]（铜 ≈ 400，铝 ≈ 237，硅 ≈ 150）
    k_f: float = 0.6         # 流体导热系数 [W/(m·K)]（水 ≈ 0.6）
    rho: float = 997.0       # 流体密度 [kg/m³]
    cp: float = 4180.0       # 流体比热 [J/(kg·K)]
    mu: float = 8.90e-4      # 流体动力黏度 [Pa·s]

    # 网格
    nx: int = 200
    ny_s: int = 12
    ny_c: int = 16

    # 边界条件
    top_bc: str = "adiabatic"    # "adiabatic"（对称）| "dirichlet"
    top_T: float = 300.0         # top_bc == "dirichlet" 时的壁温
    inlet_bc: str = "inflow"     # "inflow"（入口流入通量定温）| "adiabatic"

    def __post_init__(self) -> None:
        if not 0.0 < self.heated_fraction <= 1.0:
            raise ValueError("heated_fraction 必须落在 (0, 1] 区间")
        if min(self.H_s, self.H_c, self.L, self.nx, self.ny_s, self.ny_c) <= 0:
            raise ValueError("几何尺寸与网格数必须为正")
        if self.inlet_bc not in ("inflow", "adiabatic"):
            raise ValueError("inlet_bc 只能是 'inflow' 或 'adiabatic'")
        if self.top_bc not in ("adiabatic", "dirichlet"):
            raise ValueError("top_bc 只能是 'adiabatic' 或 'dirichlet'")
        if self.top_bc == "adiabatic" and self.inlet_bc == "adiabatic":
            raise ValueError(
                "顶面绝热且入口绝热 → 热量无处可去，稳态不存在。"
                "请设置 top_bc='dirichlet' 或 inlet_bc='inflow'。"
            )

    # ---- 派生量 ----
    @property
    def D_h(self) -> float:
        """水力直径 [m]。平行平板通道 D_h = 2·H_c。"""
        return 2.0 * self.H_c

    @property
    def Re(self) -> float:
        return self.rho * self.u_mean * self.D_h / self.mu

    @property
    def Pr(self) -> float:
        return self.mu * self.cp / self.k_f

    @property
    def entrance_length(self) -> float:
        """流体动力学入口段长度估计 Le ≈ 0.05·Re·Pr·D_h [m]。"""
        return 0.05 * self.Re * self.Pr * self.D_h

    @property
    def dp_poiseuille(self) -> float:
        """平面 Poiseuille 压降 Δp = 12·μ·u_m·L / H_c² [Pa]（f·Re = 96）。"""
        return 12.0 * self.mu * self.u_mean * self.L / self.H_c ** 2

    @property
    def pumping_power_density(self) -> float:
        """单位基板面积的泵功耗 [W/m²]：Δp·(u_m·H_c)/L。"""
        return self.dp_poiseuille * self.u_mean * self.H_c / self.L


@dataclass
class ColdPlateSolution:
    """求解结果。二维场按 ``T[j, i]`` 索引（行 = y，列 = x）。"""

    T: np.ndarray                  # 温度场 [K]，形状 (ny, nx)
    x: np.ndarray                  # 单元中心 x 坐标 [m]
    y: np.ndarray                  # 单元中心 y 坐标 [m]
    params: ColdPlateParams
    T_bulk: np.ndarray             # 各列流体质流量加权平均温度 [K]
    T_wall: np.ndarray             # 界面（y = H_s）壁面温度 [K]
    Nu_local: np.ndarray           # 局部努塞尔数 Nu(x)
    energy_error: float            # 能量守恒相对误差
    extra: dict = field(default_factory=dict)

    # ---- 关键指标 ----
    @property
    def T_max_solid(self) -> float:
        """固体域最高温 [K]（热点）。"""
        ny_s = self.params.ny_s
        return float(self.T[:ny_s, :].max())

    @property
    def R_th(self) -> float:
        """热点热阻 R_th = (T_max,solid − T_in)/q'' [m²·K/W]，按加热面面积定义。"""
        return (self.T_max_solid - self.params.T_in) / self.params.q_flux

    @property
    def R_th_wall(self) -> float:
        """平均壁面过热的等效热阻 [m²·K/W]（用加热段平均壁温）。"""
        return float(self.T_wall.mean() - self.params.T_in) / self.params.q_flux

    @property
    def Nu_mean(self) -> float:
        """充分发展段平均努塞尔数（取后 50% 流向长度）。"""
        n = len(self.Nu_local)
        return float(np.nanmean(self.Nu_local[n // 2:]))

    @property
    def T_out_bulk(self) -> float:
        return float(self.T_bulk[-1])


def solve_conjugate(p: ColdPlateParams) -> ColdPlateSolution:
    """求解共轭传热问题，返回温度场与关键指标。"""
    ny = p.ny_s + p.ny_c
    nx = p.nx
    dx = p.L / nx

    # ---- 网格与物性分布（按行 j 变化）----
    dy = np.empty(ny)
    dy[:p.ny_s] = p.H_s / p.ny_s
    dy[p.ny_s:] = p.H_c / p.ny_c
    yc = np.concatenate([
        (np.arange(p.ny_s) + 0.5) * dy[0],
        p.H_s + (np.arange(p.ny_c) + 0.5) * dy[p.ny_s],
    ])
    xc = (np.arange(nx) + 0.5) * dx

    k = np.where(np.arange(ny) < p.ny_s, p.k_s, p.k_f)

    # ---- 速度剖面：平面 Poiseuille ----
    u = np.zeros(ny)
    eta = (yc[p.ny_s:] - p.H_s) / p.H_c
    u[p.ny_s:] = 6.0 * p.u_mean * eta * (1.0 - eta)

    # ---- 面导热系数（单位深度）----
    # x 方向：**仅固体保留轴向导热**。流体侧取 0，即经典的抛物化（parabolic）近似：
    #   本项目的轴向 Peclet 数 Pe = Re·Pr ≈ 10²–10⁴ ≫ 1，流体轴向导热相对对流
    #   可忽略；反过来说，若在流体里保留一个网格尺度的导热项，它会比真实值
    #   大 2–4 个数量级，并沿"入口定温"这一人为热汇把 4% 的输入功率倒流出去
    #   （该错误正是本文件验证套件基准 B 抓出来的）。固体轴向导热必须保留——
    #   它正是共轭传热里"上游预热"的物理机制。
    Dx = np.where(np.arange(ny) < p.ny_s, p.k_s, 0.0) * dy / dx
    # y 方向：界面调和平均，正确处理固-流界面的大跳跃
    Dy = np.empty(ny - 1)
    for j in range(ny - 1):
        r = (dy[j] / 2.0) / k[j] + (dy[j + 1] / 2.0) / k[j + 1]
        Dy[j] = dx / r
    # 顶面为定温边界时，界面只在半个单元格之外，面导热系数不能用单元格间的值
    Dy_top = dx / ((dy[ny - 1] / 2.0) / k[ny - 1])

    # 迎风对流通量（单位深度），恒非负
    F = p.rho * p.cp * u * dy

    # ---- 组装线性系统 ----
    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n)
    a_P = np.zeros(n)

    def idx(i: int, j: int) -> int:
        return i * ny + j

    # 加热区（沿 x 居中）
    x0 = 0.5 * (1.0 - p.heated_fraction) * p.L
    x1 = p.L - x0

    for i in range(nx):
        in_heater = (xc[i] >= x0) and (xc[i] <= x1)
        for j in range(ny):
            P = idx(i, j)

            # 入口：按迎风格式指定**流入通量**的温度，而不是固定整列温度。
            # 若把整列入口温度钉死在 T_in，紧邻热壁的那个入口单元格就成了人为
            # 热汇——固体沿轴向导过来的热被这个被钉死的单元吸收后凭空消失。
            # 验证套件基准 B 实测该错误会漏掉约 4% 的输入功率。
            # 指定流入通量则允许入口截面温度在近壁处自然升高，符合物理。
            if i == 0 and j >= p.ny_s and p.inlet_bc == "inflow" and F[j] != 0.0:
                a_P[P] += F[j]
                b[P] += F[j] * p.T_in

            # --- 对流（一阶迎风，u ≥ 0 时上游为 i-1）---
            if i > 0 and F[j] != 0.0:
                W = idx(i - 1, j)
                rows.append(P); cols.append(W); vals.append(-F[j])
                a_P[P] += F[j]

            # --- x 方向扩散 ---
            if i > 0:
                W = idx(i - 1, j)
                rows.append(P); cols.append(W); vals.append(-Dx[j])
                a_P[P] += Dx[j]
            # i == 0 且为固体列 → 绝热（不加任何项）
            if i < nx - 1:
                E = idx(i + 1, j)
                rows.append(P); cols.append(E); vals.append(-Dx[j])
                a_P[P] += Dx[j]
            # i == nx-1 → 出口零梯度 / 固体绝热（不加任何项）

            # --- y 方向扩散 ---
            if j > 0:
                S = idx(i, j - 1)
                rows.append(P); cols.append(S); vals.append(-Dy[j - 1])
                a_P[P] += Dy[j - 1]
            else:
                # 底面：加热区内为定热流，其余绝热
                if in_heater:
                    b[P] += p.q_flux * dx
                # 绝热 → 不加任何项

            if j < ny - 1:
                N = idx(i, j + 1)
                rows.append(P); cols.append(N); vals.append(-Dy[j])
                a_P[P] += Dy[j]
            else:
                # 顶面：绝热或定温
                if p.top_bc == "dirichlet":
                    b[P] += Dy_top * p.top_T
                    a_P[P] += Dy_top
                # adiabatic → 不加任何项

    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(a_P.tolist())

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    T_flat = spsolve(A, b)
    if not np.all(np.isfinite(T_flat)):
        raise RuntimeError("线性求解失败：解中出现非有限值，请检查参数与网格")
    T = T_flat.reshape(nx, ny).T          # (ny, nx)

    # ---- 后处理：流体主体温度、壁温、局部 Nu ----
    fluid = slice(p.ny_s, ny)
    cap = p.rho * p.cp * u[fluid] * dy[fluid]          # 单位深度热容流率
    if cap.sum() > 0:
        T_bulk = (T[fluid, :] * cap[:, None]).sum(axis=0) / cap.sum()
    else:
        # 静止流体（u_mean = 0，仅用于纯导热基准）：主体温度无定义，取截面平均
        T_bulk = T[fluid, :].mean(axis=0)

    # 壁面温度：定热流下由紧邻界面的流体单元格向壁面外推半个格。
    # 热流自壁面流入流体（+y 方向），故 dT/dy = −q''/k_f，壁温高于相邻流体单元格。
    dy_f = dy[p.ny_s]
    T_wall = T[p.ny_s, :] + (p.q_flux * dy_f / 2.0) / p.k_f

    with np.errstate(divide="ignore", invalid="ignore"):
        Nu_local = p.q_flux * p.D_h / (p.k_f * (T_wall - T_bulk))

    # ---- 能量守恒校核：底部加热量 = 流体焓升 + 顶面导出热量 ----
    n_heated = int(np.count_nonzero((xc >= x0) & (xc <= x1)))
    q_in = p.q_flux * dx * n_heated
    # 入口焓按流入通量的温度 T_in 计（与入口边界条件的口径一致）
    q_fluid = float((cap * (T[fluid, -1] - p.T_in)).sum())
    q_top = 0.0
    if p.top_bc == "dirichlet":
        q_top = float((Dy_top * (T[ny - 1, :] - p.top_T)).sum())
    energy_error = abs(q_in - (q_fluid + q_top)) / q_in if q_in > 0 else 0.0

    return ColdPlateSolution(
        T=T, x=xc, y=yc, params=p, T_bulk=T_bulk, T_wall=T_wall,
        Nu_local=Nu_local, energy_error=energy_error,
        extra={"u": u, "k": k, "dy": dy, "dx": dx,
               "q_in": q_in, "q_fluid": q_fluid, "q_top": q_top},
    )
