# -*- coding: utf-8 -*-
"""生成微通道冷板代理模型的训练数据集。

流程：在设计空间内采样 → 逐点调用经过验证的共轭传热求解器 → 保存
参数、标量指标与温度场快照。

用法：
    python src/make_dataset.py                  # 默认 800 个样本
    python src/make_dataset.py --n 200 --quick   # 快速冒烟测试
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conjugate_solver import ColdPlateParams, solve_conjugate  # noqa: E402
from design_space import (FLUID_WATER, PARAM_LABELS, PARAM_NAMES,  # noqa: E402
                          DesignSpace, to_display)

PROJECT_DIR = Path(__file__).resolve().parents[1]

TARGET_NAMES = ["R_th", "dp", "P_pump", "T_max", "Nu_mean", "Re", "Le_over_L"]


def solve_point(x: np.ndarray, nx: int, ny_s: int, ny_c: int) -> tuple[np.ndarray, np.ndarray]:
    """求解单个设计点，返回 (标量指标, 温度场)。"""
    H_c, H_s, L, u_mean, q_flux, k_s = x
    p = ColdPlateParams(
        H_s=float(H_s), H_c=float(H_c), L=float(L),
        u_mean=float(u_mean), q_flux=float(q_flux), k_s=float(k_s),
        k_f=FLUID_WATER["k_f"], rho=FLUID_WATER["rho"],
        cp=FLUID_WATER["cp"], mu=FLUID_WATER["mu"],
        nx=nx, ny_s=ny_s, ny_c=ny_c,
    )
    sol = solve_conjugate(p)
    scalars = np.array([
        sol.R_th,                 # m²·K/W
        p.dp_poiseuille,          # Pa
        p.pumping_power_density,  # W/m²
        sol.T_max_solid,          # K
        sol.Nu_mean,              # -
        p.Re,                     # -
        0.05 * p.Re * p.D_h / p.L,  # Le,h/L（有效性约束的实际取值）
    ], dtype=float)
    return scalars, sol.T.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成微通道冷板数据集")
    ap.add_argument("--n", type=int, default=800, help="样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nx", type=int, default=120, help="流向网格数（温度场快照分辨率）")
    ap.add_argument("--ny-s", type=int, default=10, help="固体层网格数")
    ap.add_argument("--ny-c", type=int, default=14, help="流体层网格数")
    ap.add_argument("--quick", action="store_true", help="快速模式（减少样本与网格）")
    args = ap.parse_args()

    if args.quick:
        args.n, args.nx = min(args.n, 60), min(args.nx, 60)

    space = DesignSpace()
    X = space.sample(args.n, seed=args.seed)

    # 采样后必须再校验一次——约束是"设计空间"的一部分，不能只靠采样器自觉
    valid = space.is_valid(X)
    if not valid.all():
        raise RuntimeError(f"采样器产出了 {int((~valid).sum())} 个无效设计点")
    ratios = space.entrance_ratio(X)
    print(f"[采样] {args.n} 个设计点，全部满足 Le,h/L ≤ {space.max_entrance_ratio}")
    print(f"       Le,h/L：中位数 {np.median(ratios):.3f}，最大 {ratios.max():.3f}")

    ny = args.ny_s + args.ny_c
    Y = np.zeros((args.n, len(TARGET_NAMES)))
    T_fields = np.zeros((args.n, ny, args.nx), dtype=np.float32)

    t0 = time.time()
    for i in range(args.n):
        Y[i], T_fields[i] = solve_point(X[i], args.nx, args.ny_s, args.ny_c)
        if (i + 1) % max(1, args.n // 10) == 0:
            el = time.time() - t0
            print(f"  {i+1:4d}/{args.n}  已用 {el:5.1f}s  预计剩余 {el/(i+1)*(args.n-i-1):5.1f}s")

    outdir = PROJECT_DIR / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "coldplate_dataset.npz"

    # 后验可行性：热点温度是否落在单相液态水模型的适用范围内
    feasible = space.is_feasible(Y[:, TARGET_NAMES.index("T_max")])
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float64), Y=Y.astype(np.float64), T_fields=T_fields,
        feasible=feasible,
        param_names=np.array(PARAM_NAMES), target_names=np.array(TARGET_NAMES),
        ny_s=np.array(args.ny_s), ny_c=np.array(args.ny_c), nx=np.array(args.nx),
    )

    # 同时导出一份人类可读的 CSV（参数用工程单位）
    header = ",".join(PARAM_NAMES) + "," + ",".join(TARGET_NAMES) + ",feasible"
    table = np.column_stack([X, Y, feasible.astype(float)])
    np.savetxt(outdir / "coldplate_dataset.csv", table, delimiter=",",
               header=header, comments="", fmt="%.6g")

    meta = {
        "n_samples": int(args.n), "seed": int(args.seed),
        "grid": {"nx": int(args.nx), "ny_s": int(args.ny_s), "ny_c": int(args.ny_c)},
        "fluid": "water", "fluid_properties": FLUID_WATER,
        "param_names": PARAM_NAMES, "param_labels": PARAM_LABELS,
        "target_names": TARGET_NAMES,
        "validity_constraint": f"Le,h/L ≤ {space.max_entrance_ratio}",
        "entrance_ratio": {"median": float(np.median(ratios)), "max": float(ratios.max())},
        "feasibility_constraint": f"T_max ≤ {space.T_max_limit} K (85 °C，单相液态水)",
        "feasible_fraction": float(feasible.mean()),
        "solver_verification": "outputs/solver_verification.json（13/13 项通过）",
        "wall_time_s": round(time.time() - t0, 1),
    }
    (outdir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[完成] {npz_path}（{npz_path.stat().st_size/1e6:.2f} MB，含温度场）")
    print(f"        {outdir/'coldplate_dataset.csv'}（人类可读）")
    print(f"        {outdir/'dataset_meta.json'}")
    print(f"\n可行性：{feasible.sum()}/{args.n} = {feasible.mean()*100:.1f}% "
          f"满足 T_max ≤ {space.T_max_limit} K")
    print("\n指标范围（全部样本）：")
    for j, name in enumerate(TARGET_NAMES):
        print(f"  {name:>10s}: {Y[:, j].min():12.5g} … {Y[:, j].max():12.5g}")
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
