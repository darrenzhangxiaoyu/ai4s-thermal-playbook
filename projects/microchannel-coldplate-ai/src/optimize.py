# -*- coding: utf-8 -*-
"""在代理模型上做多目标设计优化，并用真实求解器回验最优解。

优化问题
--------
    min  J(z) = w · (R_th/R_ref) + (1−w) · (P_pump/P_ref)
    s.t. T_max ≤ 358.15 K（单相液态水模型适用上限）
         Le,h/L ≤ 1（速度剖面充分发展假设，由参数化方式自动满足）

``w`` 是热阻与泵功耗之间的权重：``w→1`` 只关心散热（不惜泵功），
``w→0`` 只关心泵功。扫过 ``w ∈ [0,1]`` 即得到**帕累托前沿**——工程上真正
有意义的交付物不是"一个最优解"，而是"热阻–泵功的权衡曲线"。

为什么用变换后的坐标优化
------------------------
``u_mean`` 的上限与 ``H_c``、``L`` 耦合，直接在有约束的盒子里做全局优化容易
踩到无效区。这里优化 ``z ∈ [0,1]^6``，其中 ``z₃`` 表示"允许流速的利用比例"，
解码时自动落在有效区内——约束由参数化保证，而不是靠罚函数。

闭环验证
--------
代理模型的预测永远可能有偏差，所以最后一步是把优化得到的设计点**代回
``conjugate_solver``**重算，并报告代理预测与真实求解的偏差。这一步是整个
"代理模型加速设计"链条能否成立的关键证据。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# 共享的中文字体/路径工具；必须在 import matplotlib 之前导入
for _p in Path(__file__).resolve().parents:
    if (_p / "tools" / "cjkfont.py").exists():
        sys.path.insert(0, str(_p / "tools"))
        break
import cjkfont  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conjugate_solver import ColdPlateParams, solve_conjugate  # noqa: E402
from design_space import FLUID_WATER, PARAM_LABELS, PARAM_NAMES, DesignSpace  # noqa: E402
from train_surrogate import make_features  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
INK, BLUE, TEAL, CORAL, GRAY, AMBER = (
    "#264653", "#2E5EAA", "#2A9D8F", "#E76F51", "#8A94A6", "#C77D0A")

SPACE = DesignSpace()
I_HC, I_HS, I_L, I_U, I_Q, I_KS = range(6)
T_IN = 300.0   # 入口温度 [K]，与 conjugate_solver / make_dataset 的默认口径一致


class Surrogate:
    """解码 z ∈ [0,1]⁶ → 设计参数，并预测各项指标。"""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            pack = pickle.load(f)
        self.models = pack["models"]
        self.log_targets = set(pack["log_targets"])
        self.lo = SPACE.bounds_array()[:, 0]
        self.span = SPACE.bounds_array()[:, 1] - self.lo

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """z → 物理设计参数（保证 Le,h/L ≤ 1）。"""
        Z = np.atleast_2d(Z)
        X = self.lo + Z * self.span
        u_cap = np.minimum(SPACE.u_mean[1],
                           SPACE.u_max(X[:, I_HC], X[:, I_L]))
        X[:, I_U] = SPACE.u_mean[0] + Z[:, I_U] * (u_cap - SPACE.u_mean[0])
        return X

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        F = make_features(np.atleast_2d(X))
        out = {}
        for name, m in self.models.items():
            p = m.predict(F)
            out[name] = 10 ** p if name in self.log_targets else p
        # T_max 不直接用拟合模型，而由定义式导出：T_max = T_in + q''·R_th。
        # 依据：R_th ≡ (T_max − T_in)/q'' 是定义关系，导出式天然物理一致；
        # train_surrogate.py 的对比显示可行子集 MAPE 由 1.42% 降到 0.38%，
        # 而"直接拟合 T_max"正是早期版本在约束边界低估约 28 K、导致优化结果
        # 回验后不可行的原因。
        out["T_max"] = T_IN + np.atleast_2d(X)[:, I_Q] * out["R_th"]
        return out


def run_bo(sur: Surrogate, w: float, ref: dict, seed: int = 0,
           tmax_margin: float = 3.0,
           n_global: int = 200_000, n_local: int = 4_000) -> dict:
    """两阶段全局寻优：大规模 Sobol 采样 + 局部细化。

    为什么不用逐点调用的贝叶斯优化：代理模型对单行输入做预测时，
    sklearn/树模型每次调用的固定开销约 1 ms 量级，逐代逐个体的 DE/BO
    在这种"目标函数极便宜但调用次数极多"的场景下会被调用开销拖垮
    （实测 21 个权重跑数十分钟仍未结束）。改成**批量向量化**后，
    20 万候选点在一次 predict 里算完，质量相当而耗时降到秒级。

    阶段一用 Sobol 序列覆盖整个 6 维单位立方体；阶段二在最优点邻域
    做局部采样细化，弥补 Sobol 在 6 维下点间距仍偏大的问题。

    ``tmax_margin``：约束按 ``T_max ≤ T_max_limit − margin`` 施加。闭环验证显示
    代理模型在约束边界附近**偏乐观**（设计点代回真实求解器后热点温度会高出几 K），
    因此留安全余量，避免"优化结果经回验后不可行"。
    """
    from scipy.stats import qmc
    rng = np.random.default_rng(seed)
    tmax_cap = SPACE.T_max_limit - tmax_margin

    def best_of(Z: np.ndarray) -> tuple[np.ndarray, float]:
        X = sur.decode(Z)                     # (n, 6) 向量化解码
        p = sur.predict(X)
        j = w * (p["R_th"] / ref["R_th"]) + (1 - w) * (p["P_pump"] / ref["P_pump"])
        over = p["T_max"] - tmax_cap
        j = j + np.where(over > 0, 10.0 * (over / 30.0) ** 2, 0.0)
        k = int(np.argmin(j))
        return Z[k], float(j[k])

    Zg = qmc.Sobol(d=6, scramble=True, seed=seed).random(n_global)
    z_best, j_best = best_of(Zg)

    # 局部细化：在最优点附近逐轮收缩搜索半径
    for scale in (0.08, 0.02, 0.005):
        Zl = np.clip(z_best + rng.normal(0, scale, size=(n_local, 6)), 0.0, 1.0)
        z_try, j_try = best_of(Zl)
        if j_try < j_best:
            z_best, j_best = z_try, j_try

    X = sur.decode(z_best)[0]
    p = {k: float(v[0]) for k, v in sur.predict(X).items()}
    return {"z": z_best.tolist(), "X": X.tolist(), "pred": p, "J": j_best}


def verify_with_solver(X: np.ndarray, nx: int = 200, ny_s: int = 12, ny_c: int = 16) -> dict:
    """把优化结果代回真实求解器，返回真实指标。"""
    H_c, H_s, L, u, q, k_s = X
    p = ColdPlateParams(H_s=H_s, H_c=H_c, L=L, u_mean=u, q_flux=q, k_s=k_s,
                        nx=nx, ny_s=ny_s, ny_c=ny_c, **FLUID_WATER)
    sol = solve_conjugate(p)
    return {"R_th": sol.R_th, "dp": p.dp_poiseuille,
            "P_pump": p.pumping_power_density, "T_max": sol.T_max_solid,
            "Nu_mean": sol.Nu_mean, "Re": p.Re,
            "Le_over_L": 0.05 * p.Re * p.D_h / p.L,
            "energy_error": sol.energy_error, "solution": sol, "params": p}


def main() -> int:
    ap = argparse.ArgumentParser(description="微通道冷板多目标设计优化")
    ap.add_argument("--n-weights", type=int, default=21, help="帕累托前沿的权重采样点数")
    ap.add_argument("--top-k", type=int, default=5, help="回验的设计点数量")
    ap.add_argument("--tmax-margin", type=float, default=3.0,
                    help="T_max 约束的安全余量（K），用于抵消代理模型在边界附近的乐观偏差")
    args = ap.parse_args()

    cjkfont.setup_cjk_font()
    outdir = PROJECT_DIR / "outputs"
    sur = Surrogate(outdir / "surrogate.pkl")

    # ---- 参考设计：所有变量取区间中点（"平庸设计"）----
    Z_mid = np.full((1, 6), 0.5)
    X_mid = sur.decode(Z_mid)[0]
    ref_sol = verify_with_solver(X_mid)
    ref = {"R_th": ref_sol["R_th"], "P_pump": ref_sol["P_pump"]}
    print(f"[参考设计] 各变量取区间中点：R_th={ref['R_th']*1e4:.3f}×10⁻⁴ m²K/W，"
          f"P_pump={ref['P_pump']:.2f} W/m²，T_max={ref_sol['T_max']:.1f} K"
          f"{'（不可行）' if ref_sol['T_max'] > SPACE.T_max_limit else ''}")

    # ---- 帕累托前沿：扫权重 ----
    print(f"\n[帕累托前沿] 扫描 {args.n_weights} 个权重 w ∈ [0,1]"
          f"（T_max 约束按 {SPACE.T_max_limit - args.tmax_margin:.1f} K 施加，留 {args.tmax_margin:.0f} K 余量）")
    records = []
    for i, w in enumerate(np.linspace(0.0, 1.0, args.n_weights)):
        r = run_bo(sur, float(w), ref, seed=100 + i, tmax_margin=args.tmax_margin)
        X = np.array(r["X"])
        records.append({"w": float(w), "X": X, "pred": r["pred"]})
    feas = np.array([r["pred"]["T_max"] <= SPACE.T_max_limit for r in records])
    print(f"  代理预测可行（T_max ≤ {SPACE.T_max_limit:.1f} K）的解：{feas.sum()}/{len(records)}")
    for r in records:
        tag = "" if r["pred"]["T_max"] <= SPACE.T_max_limit else "  [违反 T_max]"
        print(f"  w={r['w']:.2f}  R_th={r['pred']['R_th']*1e4:8.3f}×10⁻⁴  "
              f"P_pump={r['pred']['P_pump']:8.2f} W/m²  T_max={r['pred']['T_max']:6.1f} K{tag}")

    # ---- 闭环验证：把可行前沿上的点代回真实求解器 ----
    feas_idx = np.where(feas)[0]
    if len(feas_idx) == 0:
        print("\n[警告] 前沿上没有可行解，跳过闭环验证")
        return 1
    # 从可行前沿中按 R_th 排序取 top-k 覆盖
    pick = feas_idx[np.linspace(0, len(feas_idx) - 1, min(args.top_k, len(feas_idx))).astype(int)]
    print(f"\n[闭环验证] 取 {len(pick)} 个前沿设计点代回共轭求解器")
    print(f"  {'w':>5} {'代理 R_th':>11} {'求解器 R_th':>12} {'偏差':>8} "
          f"{'代理 T_max':>10} {'求解器 T_max':>12} {'能量误差':>10}")
    verify_records = []
    for i in pick:
        r = records[int(i)]
        truth = verify_with_solver(np.array(r["X"]))
        err = abs(r["pred"]["R_th"] - truth["R_th"]) / truth["R_th"] * 100
        terr = abs(r["pred"]["T_max"] - truth["T_max"]) / truth["T_max"] * 100
        print(f"  {r['w']:>5.2f} {r['pred']['R_th']*1e4:>11.3f} {truth['R_th']*1e4:>12.3f} "
              f"{err:>7.2f}% {r['pred']['T_max']:>10.1f} {truth['T_max']:>12.1f} {truth['energy_error']:>10.1e}")
        verify_records.append({"w": r["w"], "X": np.asarray(r["X"]).tolist(), "pred": r["pred"],
                               "truth": {k: v for k, v in truth.items()
                                         if k not in ("solution", "params")},
                               "R_th_error_pct": err, "T_max_error_pct": terr,
                               "feasible_by_solver": bool(truth["T_max"] <= SPACE.T_max_limit)})

    ok = all(v["feasible_by_solver"] for v in verify_records)
    max_err = max(v["R_th_error_pct"] for v in verify_records)
    print(f"\n  代理预测 R_th 最大偏差 {max_err:.2f}%；"
          f"求解器回验可行性 {sum(v['feasible_by_solver'] for v in verify_records)}/{len(verify_records)}")
    if not ok:
        print("  [注意] 有设计点经真实求解器回验后违反 T_max 约束——"
              "代理模型在该区域偏乐观，正式使用前应加密该区域样本。")

    # ---- 输出表 ----
    rows = []
    for r in records:
        X = np.array(r["X"])
        rows.append([r["w"]] + X.tolist() + [r["pred"][k] for k in
                                             ("R_th", "dp", "P_pump", "T_max", "Nu_mean")])
    header = ("w," + ",".join(PARAM_NAMES)
              + ",R_th_pred,dp_pred,P_pump_pred,T_max_pred,Nu_mean_pred")
    np.savetxt(outdir / "pareto_front.csv", np.array(rows), delimiter=",",
               header=header, comments="", fmt="%.6g")

    # ---- 图：帕累托前沿 ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for r in records:
        col = TEAL if r["pred"]["T_max"] <= SPACE.T_max_limit else CORAL
        ax.scatter(r["pred"]["R_th"] * 1e4, r["pred"]["P_pump"], s=42, color=col, zorder=3)
    # 真实求解器回验点
    ax.scatter([v["truth"]["R_th"] * 1e4 for v in verify_records],
               [v["truth"]["P_pump"] for v in verify_records],
               marker="x", s=70, color=INK, zorder=4, label="真实求解器回验")
    ax.scatter([ref["R_th"] * 1e4], [ref["P_pump"]], marker="*", s=260,
               color=AMBER, zorder=5, label="参考设计（各变量取中点）")
    for v in sorted(verify_records, key=lambda r: abs(r["w"] - 0.5))[:1]:
        x = np.array(v["X"])
        ax.annotate(
            f"w={v['w']:.2f} 膝点设计\nH_c={x[0]*1e3:.2f} mm, H_s={x[1]*1e3:.2f} mm\n"
            f"L={x[2]*1e3:.0f} mm, u={x[3]:.2f} m/s\nk_s={x[5]:.0f} W/mK",
            (v["truth"]["R_th"] * 1e4, v["truth"]["P_pump"]),
            textcoords="offset points", xytext=(18, 26), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.set_yscale("log")     # 泵功耗跨 4 个数量级，线性轴会把前沿压成一条线
    ax.set_xlabel("热阻 $R_{th}$ ($10^{-4}$ m²·K/W) → 越小散热越好")
    ax.set_ylabel("泵功耗密度 $P_{pump}$ (W/m²，对数轴) → 越小越省电")
    ax.set_title("微通道冷板热阻–泵功帕累托前沿（代理模型 + 求解器回验）",
                 fontsize=12, color=INK)
    ax.scatter([], [], color=TEAL, s=42, label="可行（T_max ≤ 85 °C）")
    ax.scatter([], [], color=CORAL, s=42, label="违反 T_max 约束")
    ax.legend(frameon=False, fontsize=9)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    plt.tight_layout()
    plt.savefig(outdir / "pareto_front.png", dpi=180)
    plt.close()

    summary = {
        "reference_design": {"X": X_mid.tolist(),
                             "R_th": ref["R_th"], "P_pump": ref["P_pump"],
                             "T_max": ref_sol["T_max"]},
        "pareto": [{"w": r["w"], "X": np.asarray(r["X"]).tolist(),
                    "pred": r["pred"]} for r in records],
        "solver_verification": verify_records,
        "max_R_th_prediction_error_pct": max_err,
        "all_verified_feasible": bool(ok),
        "T_max_limit": SPACE.T_max_limit,
    }
    (outdir / "optimization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] {outdir/'pareto_front.csv'} / pareto_front.png / optimization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
