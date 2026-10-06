# -*- coding: utf-8 -*-
"""微通道冷板代理模型：梯度提升（标量指标）+ POD 降阶模型（温度场）。

两条代理路线
------------
1. **标量代理**（``R_th``、``Δp``、泵功耗、``T_max``、``Nu``）：梯度提升树
   （sklearn HistGradientBoosting）+ 多层感知机（MLP）对比。树模型负责精度与
   可解释性（SHAP），MLP 负责检验"标量映射是否足够平滑、线性模型够不够"。
2. **温度场降阶模型（ROM）**：对 800 个温度场快照做 POD（本征正交分解），
   取前若干个模态，再用回归模型把设计参数映射到模态系数。这属于经典
   ROM 路线——工业软件（STAR-CCM+ 的 Krylov-ROM 等）走的就是这条路，
   相比神经算子更容易落地，也更容易解释"模态在物理上代表什么"。

训练目标一律取对数（热阻与压降都跨约 2 个数量级），与 PHP 热阻项目
（``projects/php-thermal-resistance``）的口径一致，便于横向比较。

用法：
    python src/train_surrogate.py
"""
from __future__ import annotations

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
from design_space import PARAM_LABELS, PARAM_NAMES  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]

INK, BLUE, TEAL, CORAL, GRAY, LIGHT = (
    "#264653", "#2E5EAA", "#2A9D8F", "#E76F51", "#8A94A6", "#F4F6F9")

# 训练目标（对数值）
LOG_TARGETS = ["R_th", "dp", "P_pump"]
RAW_TARGETS = ["T_max", "Nu_mean"]
ALL_TARGETS = LOG_TARGETS + RAW_TARGETS


def load_dataset(path: Path):
    d = np.load(path)
    X = d["X"]
    Y = d["Y"]
    names = [str(s) for s in d["target_names"]]
    T_fields = d["T_fields"]
    feasible = d["feasible"]
    return X, Y, names, T_fields, feasible, d


def make_features(X: np.ndarray) -> np.ndarray:
    """对数变换部分跨数量级的参数，便于树模型与神经网络学习。"""
    F = X.copy()
    for i, n in enumerate(PARAM_NAMES):
        if n in ("H_c", "H_s", "L", "u_mean", "q_flux"):
            F[:, i] = np.log10(X[:, i])
    return F


def build_models(seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "hgb": HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.06, early_stopping=True, random_state=seed),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=4000,
                         learning_rate_init=2e-3, early_stopping=True,
                         n_iter_no_change=60, random_state=seed)),
    }


def cv_evaluate(model, X, y, n_splits=5, seed=42):
    """K 折交叉验证，返回 out-of-fold 预测。"""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"r2": r2, "mape_pct": mape, "rmse": rmse}


def main() -> int:
    cjkfont.setup_cjk_font()
    data_path = PROJECT_DIR / "data" / "coldplate_dataset.npz"
    if not data_path.exists():
        raise SystemExit("数据集不存在，请先运行 python src/make_dataset.py")

    X, Y, names, T_fields, feasible, _ = load_dataset(data_path)
    F = make_features(X)
    outdir = PROJECT_DIR / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[数据] {len(X)} 个样本，{X.shape[1]} 个设计变量，"
          f"可行样本 {feasible.sum()} ({feasible.mean()*100:.1f}%)")

    results: dict = {}
    models: dict = {}
    oof_store: dict = {}

    print("\n[标量代理] 5 折交叉验证（HGB = 梯度提升树，MLP = 多层感知机）")
    print(f"  {'目标':<10}{'模型':<6}{'R²':>9}{'MAPE':>9}{'RMSE':>12}   可行子集 R²")
    for tname in ALL_TARGETS:
        j = names.index(tname)
        y_raw = Y[:, j].copy()
        y = np.log10(y_raw) if tname in LOG_TARGETS else y_raw.copy()
        for mname, proto in build_models().items():
            oof = cv_evaluate(proto, F, y)
            pred_raw = 10 ** oof if tname in LOG_TARGETS else oof
            mt = metrics(y_raw, pred_raw)
            r2_feas = metrics(y_raw[feasible], pred_raw[feasible])["r2"]
            print(f"  {tname:<10}{mname:<6}{mt['r2']:>9.4f}{mt['mape_pct']:>8.1f}%"
                  f"{mt['rmse']:>12.4g}   {r2_feas:>9.4f}")
            results[f"{tname}|{mname}"] = {**mt, "r2_feasible": r2_feas,
                                           "log_space": tname in LOG_TARGETS}
            if mname == "hgb":
                oof_store[tname] = (y_raw, pred_raw)

    # ---- 用全量数据重训 HGB 并保存（供 optimize.py 使用）----
    print("\n[拟合] 在全量数据上重训 HGB 模型并保存")
    hgb = build_models()["hgb"]
    for tname in ALL_TARGETS:
        j = names.index(tname)
        y = np.log10(Y[:, j]) if tname in LOG_TARGETS else Y[:, j].copy()
        m = build_models()["hgb"]
        m.fit(F, y)
        models[tname] = m
    with open(outdir / "surrogate.pkl", "wb") as f:
        pickle.dump({"models": models, "param_names": PARAM_NAMES,
                     "log_targets": LOG_TARGETS, "target_names": ALL_TARGETS}, f)
    print(f"  -> {outdir/'surrogate.pkl'}")

    # ---- SHAP：热阻模型的可解释性 ----
    print("\n[可解释性] 热阻模型的 SHAP 分析")
    import shap
    Xs = F[np.random.default_rng(0).choice(len(F), min(400, len(F)), replace=False)]
    try:
        explainer = shap.TreeExplainer(models["R_th"])
        sv = explainer.shap_values(Xs)
    except Exception as exc:                       # pragma: no cover
        print(f"  TreeExplainer 不可用（{exc}），回退到通用解释器")
        explainer = shap.Explainer(lambda d: models["R_th"].predict(d), Xs)
        sv = explainer(Xs).values
    if isinstance(sv, list):
        sv = sv[0]

    cn = [PARAM_LABELS[n] for n in PARAM_NAMES]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("  平均 |SHAP|（log10 R_th，越大越关键）：")
    for i in order:
        print(f"    {cn[i]:<24s} {mean_abs[i]:.4f}")
    shap_rank = [{"feature": cn[i], "mean_abs_shap": float(mean_abs[i])} for i in order]

    # ---- POD 降阶模型：温度场 ----
    print("\n[降阶模型] 温度场 POD")
    Nf, ny, nx = T_fields.shape
    flat = T_fields.reshape(Nf, -1).astype(np.float64)
    mean_field = flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(flat - mean_field, full_matrices=False)
    energy = S ** 2 / np.sum(S ** 2)
    cum = np.cumsum(energy)
    r = int(np.searchsorted(cum, 0.999) + 1)
    print(f"  快照 {Nf} 个，网格 {ny}×{nx}；取 {r} 个模态即解释 {cum[r-1]*100:.3f}% 能量")
    for k in [1, 5, 10, 20, 40]:
        if k <= len(cum):
            print(f"    前 {k:3d} 模态累计能量：{cum[k-1]*100:8.4f}%")

    coef = (flat - mean_field) @ Vt[:r].T          # (N, r) 模态系数
    from sklearn.model_selection import KFold
    from sklearn.multioutput import MultiOutputRegressor
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    recon = np.zeros_like(flat)
    for tr, va in kf.split(F):
        # HistGradientBoosting 只接受一维目标，用 MultiOutput 包装（每个模态一个回归器）
        reg = MultiOutputRegressor(build_models()["hgb"])
        reg.fit(F[tr], coef[tr])
        recon[va] = mean_field + reg.predict(F[va]) @ Vt[:r]
    pod_rmse = float(np.sqrt(np.mean((flat - recon) ** 2)))
    scale = float(np.abs(flat - mean_field).mean())
    print(f"  温度场重构 RMSE = {pod_rmse:.4f} K（温度场平均脉动 {scale:.2f} K）")
    # 最关心的是热点温度误差
    hot_true = flat.reshape(Nf, ny, nx)[:, :10, :].max(axis=(1, 2))
    hot_pred = recon.reshape(Nf, ny, nx)[:, :10, :].max(axis=(1, 2))
    hot_mape = float(np.mean(np.abs((hot_true - hot_pred) / hot_true)) * 100)
    print(f"  由重构场读出的热点温度 MAPE = {hot_mape:.3f}%")
    results["POD_field"] = {"modes": r, "energy_captured": float(cum[r-1]),
                            "rmse_K": pod_rmse, "hotspot_mape_pct": hot_mape}

    # ---- 用物理关系导出的 T_max，与直接训练 T_max 对比 ----
    # T_max = T_in + q''·R_th 是定义式（R_th ≡ (T_max − T_in)/q''）。
    # 若把 T_max 当作独立目标去拟合，模型既浪费容量、又在约束边界附近失准
    # （实测直接拟合的 T_max 在最优点回验时低估了 28 K，直接导致"优化结果不可行"）。
    # 这里用 R_th 的 out-of-fold 预测乘回 q''，再叠加 T_in，得到物理一致的 T_max。
    T_in = 300.0
    q_flux = X[:, PARAM_NAMES.index("q_flux")]
    rth_oof = oof_store["R_th"][1]
    tmax_derived = T_in + q_flux * rth_oof
    tmax_true = Y[:, names.index("T_max")]
    m_direct = metrics(tmax_true, oof_store["T_max"][1])
    m_derived = metrics(tmax_true, tmax_derived)
    print("\n[T_max 建模方式对比] 直接拟合 vs 由 R_th 导出（T_max = T_in + q''·R_th）")
    print(f"  {'方式':<14}{'R²':>9}{'MAPE':>9}{'RMSE(K)':>11}   可行子集 MAPE")
    for label, pred in [("直接拟合 T_max", oof_store["T_max"][1]),
                        ("由 R_th 导出", tmax_derived)]:
        mm = metrics(tmax_true, pred)
        feas_mape = metrics(tmax_true[feasible], pred[feasible])["mape_pct"]
        print(f"  {label:<14}{mm['r2']:>9.4f}{mm['mape_pct']:>8.2f}%{mm['rmse']:>11.3f}   {feas_mape:>8.2f}%")
    results["T_max|derived_from_Rth"] = {**m_derived,
                                         "r2_feasible": metrics(tmax_true[feasible],
                                                                tmax_derived[feasible])["r2"],
                                         "log_space": False}
    results["T_max_modeling_choice"] = {
        "chosen": "derived", "reason": "T_max ≡ T_in + q''·R_th，导出式与物理定义一致且更准"}

    # ---- 图 1：parity（热阻）----
    y_true, y_pred = oof_store["R_th"]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.scatter(y_true * 1e4, y_pred * 1e4, s=14, alpha=0.65,
               c=np.where(feasible, TEAL, CORAL),
               label=None)
    lim = [min(y_true.min(), y_pred.min()) * 1e4 * 0.9,
           max(y_true.max(), y_pred.max()) * 1e4 * 1.1]
    ax.plot(lim, lim, "--", color=GRAY, lw=1)
    ax.set_xlabel("求解器热阻 $R_{th}$ ($10^{-4}$ m²·K/W)")
    ax.set_ylabel("代理模型预测 $R_{th}$ ($10^{-4}$ m²·K/W)")
    ax.set_title(f"热阻代理 parity（5 折 CV，log 尺度 R²={results['R_th|hgb']['r2']:.4f}，"
                 f"MAPE={results['R_th|hgb']['mape_pct']:.1f}%）", fontsize=11, color=INK)
    ax.scatter([], [], c=TEAL, s=20, label="可行（T_max ≤ 85 °C）")
    ax.scatter([], [], c=CORAL, s=20, label="不可行（超出单相模型适用范围）")
    ax.legend(frameon=False, fontsize=9)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    plt.tight_layout()
    plt.savefig(outdir / "surrogate_parity_Rth.png", dpi=180)
    plt.close()

    # ---- 图 2：SHAP 排序 ----
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    idx = order[::-1]
    ax.barh([cn[i] for i in idx], [mean_abs[i] for i in idx], color=BLUE, alpha=0.85)
    ax.set_xlabel("平均 |SHAP|（log₁₀ $R_{th}$）")
    ax.set_title("热阻代理模型的特征重要性（SHAP）", fontsize=12, color=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    plt.tight_layout()
    plt.savefig(outdir / "surrogate_shap.png", dpi=180)
    plt.close()

    # ---- 图 3：POD 能量谱 + 重构示例 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    axes[0].semilogy(np.arange(1, min(60, len(S)) + 1), energy[:60], "o-",
                     color=BLUE, ms=4, lw=1.2)
    axes[0].axvline(r, color=CORAL, ls="--", lw=1.2, label=f"{r} 模态 → {cum[r-1]*100:.2f}%")
    axes[0].set_xlabel("POD 模态序号")
    axes[0].set_ylabel("能量占比")
    axes[0].set_title("温度场 POD 能量谱", fontsize=12, color=INK)
    axes[0].legend(frameon=False, fontsize=9)
    for s_ in ("top", "right"):
        axes[0].spines[s_].set_visible(False)

    i = int(np.argmax(np.abs(flat - mean_field).mean(axis=1)))
    err = (recon[i] - flat[i]).reshape(ny, nx)
    im = axes[1].imshow(err, aspect="auto", cmap="coolwarm", origin="lower")
    axes[1].set_xlabel("流向网格")
    axes[1].set_ylabel("厚度方向网格")
    axes[1].set_title(f"POD 重构误差场（样本 {i}）", fontsize=12, color=INK)
    fig.colorbar(im, ax=axes[1], label="误差 (K)")
    plt.tight_layout()
    plt.savefig(outdir / "surrogate_pod.png", dpi=180)
    plt.close()

    # ---- 落盘 ----
    meta = {
        "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "feasible_fraction": float(feasible.mean()),
        "cv_metrics": results, "shap_ranking": shap_rank,
        "pod": {"modes": r, "energy_captured": float(cum[r-1]),
                "rmse_K": pod_rmse, "hotspot_mape_pct": hot_mape},
    }
    (outdir / "surrogate_metrics.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] {outdir/'surrogate_metrics.json'} / surrogate.pkl / 3 张图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
