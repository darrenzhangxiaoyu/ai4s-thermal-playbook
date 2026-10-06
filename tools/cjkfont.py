# -*- coding: utf-8 -*-
"""跨平台中文字体解析与仓库路径工具。

背景
----
本仓库早期版本的绘图脚本把字体目录硬编码为某 AI 桌面应用的绝对路径
（``/Users/<user>/Library/Application Support/kimi-desktop/daimon-share/...``），
输出目录同样写成绝对路径。结果是：任何人克隆这个公开仓库后都无法复现图表，
README 里"图表复现"一节对他人不成立。

本模块用一条回退链解析 CJK 字体，并把输出路径改为相对仓库根目录解析，
使脚本在没有该 AI 应用的机器上也能跑通。

字体解析顺序
------------
1. 环境变量 ``AI4S_FONT_DIR`` 指向目录中的字体文件（最高优先级，便于定向复现）
2. 仓库内 ``assets/fonts/``
3. matplotlib 已索引的系统字体（macOS 自带的 PingFang SC 等，无需注册文件）
4. 显式注册常见系统 CJK 字体文件（Linux 的 Noto CJK、Windows 的 msyh.ttc 等）

一条都找不到时打印醒目警告：脚本**不会崩溃**，数值结果依然正确，
只是图中中文会渲染成方框（豆腐块）。

用法
----
必须在 ``import matplotlib`` **之前**导入本模块（它要设置 ``MPLCONFIGDIR``）::

    import sys
    from pathlib import Path
    for _p in Path(__file__).resolve().parents:
        if (_p / "tools" / "cjkfont.py").exists():
            sys.path.insert(0, str(_p / "tools"))
            break
    import cjkfont                       # 设置 matplotlib 缓存目录
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FAMILY = cjkfont.setup_cjk_font()    # 注册并选用中文字体
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

__all__ = ["repo_root", "ensure_mpl_cache_dir", "setup_cjk_font", "PREFERRED_FAMILIES"]

# matplotlib rcParams 的字体族优先级：先无衬线黑体（图表最常用），再衬线/系统兜底
PREFERRED_FAMILIES = (
    "Noto Sans SC",         # 原脚本使用，Linux/容器常见
    "Noto Sans CJK SC",     # Noto CJK 合集
    "Source Han Sans SC",   # 思源黑体
    "PingFang SC",          # macOS 默认中文黑体
    "Hiragino Sans GB",     # macOS 备选
    "Heiti SC",
    "Songti SC",            # macOS 衬线兜底
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "Arial Unicode MS",     # 跨平台兜底（覆盖 CJK 字形）
)

# 显式注册用的系统字体文件（用不到就跳过；.ttc 兼容性差，失败会静默忽略）
_SYSTEM_FONT_FILES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

_FONT_SUFFIXES = (".ttf", ".otf", ".ttc", ".otc")


def repo_root() -> Path:
    """仓库根目录（本文件位于 ``<repo>/tools/cjkfont.py``）。"""
    return Path(__file__).resolve().parents[1]


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def ensure_mpl_cache_dir() -> str:
    """保证 matplotlib 有一个可写的缓存目录。

    默认的 ``~/.matplotlib`` 在受限/沙箱环境中常常不可写，matplotlib 会打印
    一大段警告并改用临时目录。这里提前把 ``MPLCONFIGDIR`` 指到第一个可写的
    候选目录，让输出干净且跨机器一致。用户已显式设置时不覆盖。
    """
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        return existing
    candidates = [
        Path.home() / ".matplotlib",
        Path(tempfile.gettempdir()) / "ai4s-mplcache",
    ]
    for cand in candidates:
        if _is_writable_dir(cand):
            os.environ["MPLCONFIGDIR"] = str(cand)
            return str(cand)
    return ""


# 在 matplotlib 被导入前生效
ensure_mpl_cache_dir()


def _candidate_font_files() -> list[str]:
    """按优先级收集候选字体文件路径。"""
    files: list[str] = []

    def collect_from_dir(directory: Path) -> None:
        if not directory.is_dir():
            return
        for suffix in _FONT_SUFFIXES:
            files.extend(sorted(str(p) for p in directory.glob(f"*{suffix}")))

    env_dir = os.environ.get("AI4S_FONT_DIR")
    if env_dir:
        collect_from_dir(Path(env_dir).expanduser())
    collect_from_dir(repo_root() / "assets" / "fonts")
    for path in _SYSTEM_FONT_FILES:
        if os.path.exists(path):
            files.append(path)
    return files


def setup_cjk_font(quiet: bool = False) -> str | None:
    """注册可用的 CJK 字体并写入 matplotlib rcParams。

    返回选中的字体族名；若一个都没找到则返回 ``None``（并打印警告）。
    """
    import logging

    import matplotlib
    from matplotlib import font_manager as fm

    # 系统 CJK 字体常常缺少 bold 字重（例如 macOS 的 PingFang SC），
    # matplotlib 会为每次加粗标签打印一条 findfont 警告。它只是回退到 600 字重，
    # 不影响结果，但会淹没脚本输出——这里降级为 ERROR 以保持输出干净。
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    for path in _candidate_font_files():
        try:
            fm.fontManager.addfont(path)
        except Exception:
            # 损坏/不支持的字体文件不应阻断绘图
            continue

    chosen: str | None = None
    for family in PREFERRED_FAMILIES:
        try:
            fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
        except Exception:
            continue
        chosen = family
        break

    if chosen is not None:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        if not quiet:
            print(f"[font] 中文字体：{chosen}")
    else:
        matplotlib.rcParams["axes.unicode_minus"] = False
        print(
            "[font][警告] 未找到任何中文字体，图中中文将显示为方框（豆腐块）。\n"
            "             补救方式（任选其一）：\n"
            "               1) 安装 Noto Sans CJK / 思源黑体；\n"
            "               2) 把字体文件放进 <repo>/assets/fonts/；\n"
            "               3) 设置环境变量 AI4S_FONT_DIR 指向字体所在目录。\n"
            "             注意：图表数值结果不受影响，仅标签渲染受影响。",
            file=sys.stderr,
        )
    return chosen
