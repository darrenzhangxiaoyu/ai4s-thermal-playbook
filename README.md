# AI4S 调研与实践（AI for Science × 热管理）

> 一个从"零基础进入 AI4S"到"可复现的热管理 AI 项目"的完整知识库。
> 背景：航天器及算力卫星热控系统 + 高热流密度芯片热管理，聚焦流体/能源/两相传热方向。

## 内容地图

| 路径 | 内容 | 适合谁 |
|-|-|-|
| [docs/AI4S全方位调研报告](docs/AI4S全方位调研报告.md) | AI4S 全景：六大方向、产业格局、政策市场、学习路径（附 6 张图表） | 刚入门的任何人 |
| [docs/流体与能源AI4S入行指南](docs/流体与能源AI4S入行指南.md) | 流体/能源方向特化：技术地图、雇主地图、四阶段学习路径 | 确定做流体/能源方向 |
| [docs/热管理背景AI4S专项计划](docs/热管理背景AI4S专项计划.md) | 结合航天器热控 + 芯片液冷背景：双主线技术地图、沸腾/两相 × 重力环境专题、18 个月个性化计划 | 与作者背景相近者 |
| [docs/参考文献.md](docs/参考文献.md) | **三份文档共享的 107 条参考文献表**（逐条标注核验状态） | 想核查来源的读者 |
| [docs/核验报告.md](docs/核验报告.md) | **引文核验报告**：方法、逐条裁决、已修正的问题、明确未做的事 | 想了解内容可信度边界 |
| [poster/](poster/) | A4 速查海报（可编辑设计稿）：器件 × 重力环境 × AI 切入点对照 | 打印贴工位 |
| [projects/php-thermal-resistance/](projects/php-thermal-resistance/) | 作品集项目 1：PHP 热阻预测（XGBoost + SHAP + 贝叶斯优化） | 想动手实践者 |
| [projects/microchannel-coldplate-ai/](projects/microchannel-coldplate-ai/) | 作品集项目 2：微通道冷板 AI 设计工作流（自研共轭传热求解器 → 代理模型 → 帕累托优化 → 求解器回验） | 想做"有真值"的 AI4S 项目 |

## 关于内容可信度（请先读这一段）

本仓库的调研文档最初由上一代 AI 工具产出，存在一个严重问题：**正文带了 93 个 `[^N^]`
引用标记，但参考文献表整份丢失**——每个数据点都挂着引用，却一个都回溯不到来源。

本次接手后已完成：

1. **重建并核验全部引文**：编号在文档间连续不重叠（1–63 / 64–76 / 77–107），据此还原出
   107 条共享参考文献表，逐条回溯一手来源并用页面正文逐字取证。
   结果：**已核验 44 条 · 部分核验 38 条 · 未找到来源 9 条 · 与来源矛盾 1 条 · 链接失效 1 条**。
2. **修正而非仅标注**：判为"有矛盾/未找到"的表述已在正文中实际改正或删除，
   包括创世纪使命的时间与金额、Sakana AI 融资数字、国盛证券渗透率口径、
   FuXi-S2S 的 42 天、液冷市场口径、芯片功率路线图来源等 35 处。
3. **修复可复现性问题**：绘图脚本原先硬编码了某个 AI 桌面应用的字体绝对路径，
   他人克隆后无法复现图表；现已改为跨平台字体回退链 + 仓库相对路径。

**核验记录（含 URL、逐字证据、置信度）**保存在 [docs/verification/](docs/verification/)，
可用 `python tools/build_bibliography.py` 重新生成参考文献表、
用 `python tools/build_verification_report.py` 重新生成核验报告。

关于未被核验覆盖的边界，请读 [docs/核验报告.md](docs/核验报告.md) 第七节。

## 快速开始（作品集项目 2）

```bash
cd projects/microchannel-coldplate-ai
pip install -r requirements.txt

python src/verify_solver.py     # ① 13 项解析解验证（不通过就不要往下走）
python src/make_dataset.py      # ② 800 个样本（含温度场），约 5 秒
python src/train_surrogate.py   # ③ 代理模型 + POD 降阶模型 + SHAP
python src/optimize.py          # ④ 帕累托前沿 + 真实求解器回验
```

## 快速开始（作品集项目 1）

```bash
cd projects/php-thermal-resistance
pip install -r requirements.txt
python src/make_demo_data.py   # 演示数据（合成数据，非文献数据复现）
python src/train.py            # 训练 + 5 折 CV + SHAP 分析
python src/optimize.py         # 贝叶斯优化设计闭环
```

> ⚠️ 项目 1 当前用的是**物理启发式合成数据**，其指标（CV R²=0.936、MAPE=11.0%）
> **不是**对 DLR 基准文献（3,473 数据点、RMSPE 20.2%、R²=0.941）的复现。详见该项目 README 的口径声明。

## 图表复现

```bash
python scripts/charts.py    # fig1–fig4（全景报告）
python scripts/charts2.py   # fig5（流体与能源技术地图）
python scripts/charts3.py   # fig6（热管理技术地图）
```

图表输出到 `docs/assets/`。脚本可在任意工作目录执行，路径按仓库根目录解析。

**中文字体**：脚本会按以下顺序自动查找并选用 —— 环境变量 `AI4S_FONT_DIR` 指向的目录 →
仓库内 `assets/fonts/` → 系统已索引的 CJK 字体（macOS 的 PingFang SC 等）→
常见系统字体文件（Linux 的 Noto CJK、Windows 的 msyh 等）。
一个都找不到时会打印醒目警告（图中中文显示为方框，但数值结果不受影响）。
若要精确复现某套字体，设置 `AI4S_FONT_DIR=/path/to/fonts` 即可。

## 许可证

- **代码**（scripts/、projects/、tools/ 下的脚本）：[MIT](LICENSE)
- **文档与图表内容**（docs/、blog、poster）：[CC BY 4.0](LICENSE-docs)，转载需署名

## 声明

本仓库内容为 2026 年 10 月前公开资料调研整理，仅供学习参考，不构成职业或投资建议。
调研文档中仍有个别表述未能追溯到一手来源，已在 [docs/参考文献.md](docs/参考文献.md) 中
逐条标注为"未找到"；引用前请先核对该表。演示项目中的合成数据仅用于流程验证，
定量结论以真实文献数据为准。
