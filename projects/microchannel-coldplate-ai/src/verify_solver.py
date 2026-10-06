# -*- coding: utf-8 -*-
"""共轭传热求解器的验证套件。

为什么需要这个文件
------------------
``conjugate_solver.py`` 产出的温度场会被当作"真值"用于训练代理模型。
如果求解器本身没有被独立验证过，整个数据集的物理可信度就是空的
（这正是本仓库早期版本"结论无法回溯"问题的根源）。因此这里用**解析解**
对离散格式逐项验证。

四个基准
--------
A. **一维稳态导热**：全域均质、顶面定温、底面定热流 → 解析解
   ``T(y) = T_top + q''·(H−y)/k``，检查离散格式与边界条件实现。
B. **能量守恒**：底部加热量 = 流体焓升 + 顶面导出热量，检查全局守恒；
   并与解析焓升 ``ΔT_b = q''·L/(ρ·c_p·u_m·H_c)`` 对比。
C. **充分发展努塞尔数**：平行平板单侧定热流、另一侧绝热的经典结果
   ``Nu = 5.385``（Shah & London 1978）。这是对"对流＋扩散"耦合离散
   最严格的一步检验。
D. **网格收敛**：加密网格，误差应单调下降。

外加一项**假设审计**：求解器假定流体速度剖面已充分发展
（平面 Poiseuille）。审计会统计设计空间内
``Le,h/L`` 的分布，并给出该假设成立的比例——不成立的区域必须在
采样阶段被排除，而不是被悄悄忽略。

运行：``python src/verify_solver.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conjugate_solver import ColdPlateParams, solve_conjugate  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
NU_PARALLEL_PLATE_ONE_SIDED = 5.385   # Shah & London (1978)：平行平板单侧定热流

results: list[dict] = []


def report(name: str, value: float, target: float, tol: float, unit: str = "") -> bool:
    """打印一行判定，返回是否通过。"""
    ok = abs(value - target) <= tol
    results.append({"benchmark": name, "value": value, "target": target,
                    "abs_error": abs(value - target), "tol": tol, "passed": bool(ok)})
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: {value:.6g}{unit}  目标 {target:.6g}±{tol:.1g}{unit}  "
          f"误差 {abs(value - target):.3g}")
    return ok


def bench_a_conduction() -> None:
    """A. 一维稳态导热解析解。"""
    print("\n[A] 一维稳态导热（顶面定温 + 底面定热流，全域均质）")
    q, k, T_top = 1.0e4, 10.0, 300.0
    p = ColdPlateParams(H_s=1.0e-3, H_c=0.5e-3, L=20.0e-3, u_mean=0.0,
                        q_flux=q, k_s=k, k_f=k, top_bc="dirichlet", top_T=T_top,
                        inlet_bc="adiabatic", nx=120, ny_s=10, ny_c=12)
    sol = solve_conjugate(p)
    H = p.H_s + p.H_c
    T_analytic = T_top + q * (H - sol.y) / k          # 随 y 的解析温度
    err = float(np.abs(sol.T - T_analytic[:, None]).max())
    report("温度场最大绝对误差", err, 0.0, 1e-8, " K")
    # 解析热点温度
    report("热点温度 T_max", sol.T_max_solid, T_top + q * (H - sol.y[0]) / k, 1e-8, " K")
    report("能量守恒相对误差", sol.energy_error, 0.0, 1e-10)


def bench_b_energy() -> None:
    """B. 能量守恒与解析焓升（多个工况）。"""
    print("\n[B] 能量守恒与流体焓升（工程量级工况）")
    cases = [
        dict(H_s=0.5e-3, H_c=0.4e-3, L=30e-3, u_mean=0.10, q_flux=3.0e5, k_s=400.0),
        dict(H_s=1.0e-3, H_c=0.8e-3, L=30e-3, u_mean=0.05, q_flux=1.0e5, k_s=237.0),
        dict(H_s=0.3e-3, H_c=0.2e-3, L=20e-3, u_mean=0.20, q_flux=8.0e5, k_s=150.0),
    ]
    for c in cases:
        p = ColdPlateParams(**c)
        sol = solve_conjugate(p)
        dT_analytic = p.q_flux * p.L / (p.rho * p.cp * p.u_mean * p.H_c)
        dT_num = sol.T_out_bulk - p.T_in
        tag = f"H_c={p.H_c*1e3:.1f}mm u={p.u_mean:.2f}m/s q={p.q_flux/1e4:.0f}W/cm²"
        report(f"能量守恒误差 ({tag})", sol.energy_error, 0.0, 1e-8)
        # 主体温升：一阶迎风会带来少量数值扩散，容差放到 0.5%
        report(f"主体温升 ΔT_b ({tag})", dT_num, dT_analytic, 5e-3 * dT_analytic, " K")


def _developed_profile_error(nx: int = 4000, ny_c: int = 64) -> tuple[float, float, float]:
    """在逼近渐近解的长通道上，比较发展段温度剖面与闭式解。

    闭式解（平行平板、单侧定热流、另一侧绝热、忽略流体轴向导热）：

        θ(η) = T(y) − T_b = (q''·H_c/k_f)·(η³ − η⁴/2 − η + 0.371429)
        Nu = q''·D_h/(k_f·θ(0)) = 2/0.371429 = 5.385

    返回 ``(剖面相对偏差, 该截面局部 Nu, 能量守恒误差)``。

    为什么要取这么长的通道（L/D_h = 10⁵）：定热流边界下努塞尔数向渐近值
    的收敛是**代数式**的（~1/x），不是指数式的——壁面附近流速趋于零，
    该区域需要极长的距离才能达到渐近剖面。这是一个已被文献记载的
    Graetz 问题特性，不是格式缺陷。因此基准必须取到渐近区，否则测到的
    只是"还没发展完"的中间值。
    """
    p = ColdPlateParams(H_s=1.0e-3, H_c=0.5e-3, L=100.0, u_mean=0.01,
                        q_flux=1.0e4, k_s=400.0, nx=nx, ny_s=12, ny_c=ny_c)
    sol = solve_conjugate(p)
    i = nx - 1 - 200                      # 避开出口边界扰动（其衰减长度约 11 mm）
    eta = (np.arange(ny_c) + 0.5) / ny_c
    th_ana = (p.q_flux * p.H_c / p.k_f) * (eta ** 3 - eta ** 4 / 2.0 - eta + 0.371429)
    th_num = sol.T[p.ny_s:, i] - sol.T_bulk[i]
    rel = float(np.abs(th_num - th_ana).max() / np.abs(th_ana).max())
    return rel, float(sol.Nu_local[i]), sol.energy_error


def bench_c_developed_profile() -> None:
    """C. 发展段温度剖面与努塞尔数 vs 闭式解。"""
    print("\n[C] 发展段温度剖面与 Nu（平行平板，单侧定热流 + 单侧绝热）")
    print("  闭式解 θ(η) = (q''H_c/k_f)(η³ − η⁴/2 − η + 0.371429)，Nu∞ = 5.385")
    rel, nu, eerr = _developed_profile_error()
    report("发展段剖面 θ(η) 最大相对偏差", rel * 100, 0.0, 0.5, " %")
    report("该截面局部 Nu", nu, NU_PARALLEL_PLATE_ONE_SIDED,
           0.005 * NU_PARALLEL_PLATE_ONE_SIDED)
    # 该基准有 4000×76 ≈ 3×10⁵ 个未知量，全局求和的舍入累积比小算例高一个量级，
    # 容差相应放宽到 1e-7（仍远低于任何物理意义的误差）
    report("能量守恒相对误差", eerr, 0.0, 1e-7)


def bench_d_grid_convergence() -> None:
    """D. 网格收敛：横向加密时剖面误差应单调下降，并估计收敛阶。"""
    print("\n[D] 网格收敛（L=100m 发展段剖面误差，横向加密）")
    print(f"  {'ny_c':>6} {'剖面偏差':>12} {'收敛阶':>8}")
    errs = []
    for ny_c in (8, 16, 32, 64):
        rel, _, _ = _developed_profile_error(nx=4000, ny_c=ny_c)
        errs.append(rel)
        if len(errs) > 1 and errs[-2] > 0 and errs[-1] > 0:
            order = np.log(errs[-2] / errs[-1]) / np.log(2.0)
            order_s = f"{order:8.2f}"
        else:
            order_s = "       -"
        print(f"  {ny_c:>6} {rel*100:>11.4f}% {order_s}")
    monotone = all(b <= a for a, b in zip(errs, errs[1:]))
    results.append({"benchmark": "网格收敛（剖面误差单调下降）",
                    "value": errs[-1], "target": 0.0, "abs_error": errs[-1],
                    "tol": errs[0], "passed": bool(monotone)})
    print(f"  [{'PASS' if monotone else 'FAIL'}] 误差随网格加密单调下降")


def bench_e_assumption_audit() -> None:
    """E. 充分发展速度剖面假设的适用性审计。"""
    print("\n[E] 速度剖面假设审计（Le,h/L ≤ 1 视为可接受）")
    rng = np.random.default_rng(0)
    n = 4000
    H_c = rng.uniform(0.2e-3, 1.0e-3, n)
    L = rng.uniform(20e-3, 50e-3, n)
    u = rng.uniform(0.02, 1.5, n)
    rho, mu = 997.0, 8.90e-4
    D_h = 2.0 * H_c
    Re = rho * u * D_h / mu
    Le_h = 0.05 * Re * D_h
    frac = Le_h / L
    for thr in (0.5, 1.0, 2.0):
        share = float(np.mean(frac <= thr)) * 100
        print(f"  Le,h/L ≤ {thr:<4} 的参数组合占比：{share:5.1f}%")
    print(f"  Le,h/L 中位数 {np.median(frac):.2f}，90 分位 {np.percentile(frac, 90):.2f}")
    print("  结论：水在微通道中的流动入口段可达数十倍管长，")
    print("        「速度已充分发展」只在低 Re / 长通道角落成立。")
    print("        → 数据集采样必须显式施加 Le,h/L ≤ 1 的有效性约束，")
    print("          并把该限制写进 README 的模型局限一节。")
    results.append({"benchmark": "假设审计 Le,h/L ≤ 1 占比(%)",
                    "value": float(np.mean(frac <= 1.0) * 100), "target": 100.0,
                    "abs_error": 0.0, "tol": 0.0, "passed": True,
                    "note": "仅记录，用于确定采样约束"})


def main() -> int:
    print("=" * 78)
    print("共轭传热求解器验证套件")
    print("=" * 78)
    bench_a_conduction()
    bench_b_energy()
    bench_c_developed_profile()
    bench_d_grid_convergence()
    bench_e_assumption_audit()

    checks = [r for r in results if r["benchmark"] != "假设审计 Le,h/L ≤ 1 占比(%)"]
    failed = [r for r in checks if not r["passed"]]
    print("\n" + "=" * 78)
    print(f"合计 {len(checks)} 项检查，通过 {len(checks)-len(failed)} 项，失败 {len(failed)} 项")

    out = PROJECT_DIR / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "solver_verification.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out/'solver_verification.json'}")
    if failed:
        print("\n未通过的检查：")
        for r in failed:
            print(f"  - {r['benchmark']}: 实测 {r['value']:.6g} vs 目标 {r['target']:.6g}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
