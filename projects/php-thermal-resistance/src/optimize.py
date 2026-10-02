# -*- coding: utf-8 -*-
"""阶段 3：贝叶斯优化设计闭环。

在训练好的热阻代理模型（log 尺度）上做贝叶斯优化：
给定工作点（热流密度），搜索最优设计参数（充液率/水力直径/弯折数/蒸发段长度/倾角/工质），
使预测热阻最小。

方法：sklearn GaussianProcessRegressor（Matern 核）+ EI 采集函数，
每种工质独立优化连续设计变量，最后跨工质取最优。

用法：
    python src/train.py                # 先生成 model.pkl
    python src/optimize.py             # 默认工作点 q=30 W/cm²
    python src/optimize.py --q 45      # 指定工作点
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib

warnings.filterwarnings("ignore")  # 抑制 GP 超参收敛警告
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_DIR = "/Users/zhangxiaoyu/Library/Application Support/kimi-desktop/daimon-share/daimon/runtime/python/fonts"
for _f in ["NotoSansSC-Regular.ttf", "NotoSansSC-Bold.ttf"]:
    if os.path.exists(f"{FONT_DIR}/{_f}"):
        font_manager.fontManager.addfont(f"{FONT_DIR}/{_f}")
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False
INK, BLUE, TEAL, CORAL, GRAY = "#264653", "#2E5EAA", "#2A9D8F", "#E76F51", "#8A94A6"

FLUIDS = ["water", "ethanol", "acetone", "R134a"]

# 设计变量边界（连续）
BOUNDS = np.array([
    [20.0, 80.0],    # fill_ratio_pct
    [1.0, 3.0],      # hydraulic_diameter_mm
    [4.0, 20.0],     # n_turns
    [40.0, 120.0],   # evaporator_length_mm
    [30.0, 90.0],    # inclination_deg（工程可安装范围；90=垂直底热）
])
VAR_NAMES = ["充液率(%)", "水力直径(mm)", "弯折数", "蒸发段长度(mm)", "倾角(°)"]
# 归一化到 [0,1] 便于 GP
B_LO, B_SPAN = BOUNDS[:, 0], BOUNDS[:, 1] - BOUNDS[:, 0]


def load_model(path="outputs/model.pkl"):
    with open(path, "rb") as f:
        pack = pickle.load(f)
    return pack["model"], pack["feature_names"]


def make_features(model_feats, fluid, X):
    """X: (n,5) 连续设计变量矩阵 → 与训练同构的 (n, n_feat) DataFrame。"""
    n = len(X)
    row = dict(zip(["fill_ratio_pct", "hydraulic_diameter_mm", "n_turns",
                    "evaporator_length_mm", "inclination_deg"], X.T))
    row["heat_flux_wcm2"] = np.full(n, Q_POINT)
    for fl in FLUIDS:
        row[f"fluid_{fl}"] = np.where(fluid == np.array([fl] * n), 1.0, 0.0)
    df = pd.DataFrame(row)
    return df.reindex(columns=model_feats, fill_value=0.0)


def predict_logR(model, model_feats, fluid, X):
    """X: (n,5) 原始量纲 → (n,) log R_th（向量化预测）"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fluid_col = np.repeat(fluid, len(X)) if isinstance(fluid, str) else np.asarray(fluid)
    return model.predict(make_features(model_feats, fluid_col, X))


def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu - xi) / sigma
    from scipy.stats import norm
    return (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayes_optimize(model, model_feats, fluid, n_init=8, n_iter=20, seed=0):
    """对单种工质做 EI 贝叶斯优化，返回 (best_x, best_y, history)。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    rng = np.random.default_rng(seed)

    def unnorm(z):
        return B_LO + np.asarray(z) * B_SPAN

    Xn = rng.uniform(0, 1, size=(n_init, len(BOUNDS)))
    X = unnorm(Xn)
    y = predict_logR(model, model_feats, fluid, X)

    hist = [y.min()]
    for it in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(noise_level=0.02),
            normalize_y=True, random_state=seed, n_restarts_optimizer=0)
        gp.fit(Xn, y)
        # 多起点 L-BFGS-B 最大化 EI
        from scipy.optimize import minimize
        best_acq, best_z = -np.inf, None
        for _ in range(8):
            z0 = rng.uniform(0, 1, size=len(BOUNDS))
            res = minimize(lambda z: -expected_improvement(*gp.predict(z.reshape(1, -1), return_std=True),
                                                            y.min()), z0,
                           method="L-BFGS-B", bounds=[(0, 1)] * len(BOUNDS))
            if -res.fun > best_acq:
                best_acq, best_z = -res.fun, res.x
        z_new = np.clip(best_z, 0, 1)
        Xn = np.vstack([Xn, z_new])
        X = np.vstack([X, unnorm(z_new)])
        y = np.append(y, predict_logR(model, model_feats, fluid, X[-1:]))
        hist.append(y.min())
    best_i = int(np.argmin(y))
    return X[best_i], float(y[best_i]), hist


def main():
    global Q_POINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=30.0, help="工作点热流密度 W/cm²")
    ap.add_argument("--model", default="outputs/model.pkl")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    Q_POINT = args.q
    os.makedirs(args.outdir, exist_ok=True)

    model, model_feats = load_model(args.model)
    print(f"[工作点] 热流密度 = {Q_POINT} W/cm²；目标：min R_th（贝叶斯优化，EI 采集）")

    # 每种工质独立优化
    results, hists = [], {}
    for fi, fl in enumerate(FLUIDS):
        bx, by, hist = bayes_optimize(model, model_feats, fl, seed=42 + fi)
        hists[fl] = hist
        results.append({
            "fluid": fl,
            "fill_ratio_pct": round(float(bx[0]), 1),
            "hydraulic_diameter_mm": round(float(bx[1]), 2),
            "n_turns": int(round(bx[2])),
            "evaporator_length_mm": round(float(bx[3]), 1),
            "inclination_deg": round(float(bx[4]), 1),
            "R_th_pred_K_W": round(float(np.exp(by)), 3),
        })
        print(f"  {fl:<8s} 最优 R_th = {np.exp(by):.3f} K/W  @ 充液率 {bx[0]:.0f}%, "
              f"d_h {bx[1]:.2f} mm, {bx[2]:.0f} 弯, 倾角 {bx[4]:.0f}°")

    res_df = pd.DataFrame(results).sort_values("R_th_pred_K_W")
    res_df.to_csv(f"{args.outdir}/optimal_designs.csv", index=False, encoding="utf-8-sig")
    best = res_df.iloc[0]
    print(f"\n[最优设计] 工质={best['fluid']}, R_th={best['R_th_pred_K_W']} K/W")

    # 基线对比：每个变量取区间中点的"平庸设计"
    baseline_x = B_LO + 0.5 * B_SPAN
    base_y = {fl: float(np.exp(predict_logR(model, model_feats, fl, baseline_x.reshape(1, -1))[0]))
              for fl in FLUIDS}
    base_best = min(base_y.values())
    print(f"[基线对比] 平庸设计最优工质 R_th = {base_best:.3f} K/W，"
          f"优化后 {best['R_th_pred_K_W']:.3f} K/W，改善 {(1 - best['R_th_pred_K_W'] / base_best) * 100:.1f}%")

    with open(f"{args.outdir}/optimization.json", "w", encoding="utf-8") as f:
        json.dump({"q_point_W_cm2": Q_POINT, "results": results,
                   "baseline": base_y}, f, indent=2, ensure_ascii=False)

    # ---- 图 1：收敛曲线 ----
    plt.figure(figsize=(6.5, 4.2))
    for fl, h in hists.items():
        plt.plot(np.minimum.accumulate(h), label=fl, lw=1.8)
    plt.xlabel("贝叶斯优化迭代次数")
    plt.ylabel("当前最优 log R_th")
    plt.title(f"EI 贝叶斯优化收敛（q = {Q_POINT} W/cm²）", fontsize=12, color=INK)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/bo_convergence.png", dpi=180)
    plt.close()

    # ---- 图 2：设计空间景观（充液率 × 水力直径，固定其余为最优值）----
    fl_best = best["fluid"]
    bx = np.array([best["fill_ratio_pct"], best["hydraulic_diameter_mm"], best["n_turns"],
                   best["evaporator_length_mm"], best["inclination_deg"]])
    frs = np.linspace(20, 80, 60)
    dhs = np.linspace(1.0, 3.0, 60)
    FR, DH = np.meshgrid(frs, dhs)
    Z = np.zeros_like(FR)
    for i in range(FR.shape[0]):
        xs = np.column_stack([FR[i], DH[i],
                              np.full_like(FR[i], bx[2]), np.full_like(FR[i], bx[3]),
                              np.full_like(FR[i], bx[4])])
        Z[i] = np.exp(predict_logR(model, model_feats, fl_best, xs))
    plt.figure(figsize=(7, 5))
    cs = plt.contourf(FR, DH, Z, levels=18, cmap="viridis")
    plt.colorbar(cs, label="预测热阻 R_th (K/W)")
    plt.plot(bx[0], bx[1], "*", color=CORAL, ms=16, mec="white")
    plt.annotate("BO 最优点", (bx[0], bx[1]), textcoords="offset points", xytext=(10, 8),
                 color="white", fontsize=11)
    plt.xlabel("充液率 (%)")
    plt.ylabel("水力直径 (mm)")
    plt.title(f"设计空间景观（{fl_best}，其余变量固定为最优值）", fontsize=12, color=INK)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/design_landscape.png", dpi=180)
    plt.close()

    print(f"[done] optimal_designs.csv / bo_convergence.png / design_landscape.png -> {args.outdir}/")


if __name__ == "__main__":
    main()
