# AI4S 调研与实践（AI for Science × 热管理）

> 一个从"零基础进入 AI4S"到"第一个可复现的热管理 AI 项目"的完整知识库。
> 背景：航天器及算力卫星热控系统 + 高热流密度芯片热管理，聚焦流体/能源/两相传热方向。

## 内容地图

| 路径 | 内容 | 适合谁 |
|-|-|-|
| [docs/AI4S全方位调研报告](docs/AI4S全方位调研报告.md) | AI4S 全景：六大方向、产业格局、政策市场、学习路径（附 6 张图表） | 刚入门的任何人 |
| [docs/流体与能源AI4S入行指南](docs/流体与能源AI4S入行指南.md) | 流体/能源方向特化：技术地图、雇主地图、四阶段学习路径 | 确定做流体/能源方向 |
| [docs/热管理背景AI4S专项计划](docs/热管理背景AI4S专项计划.md) | 结合航天器热控 + 芯片液冷背景：双主线技术地图、沸腾/两相 × 重力环境专题、18 个月个性化计划 | 与作者背景相近者 |
| [poster/](poster/) | A4 速查海报（可编辑设计稿）：器件 × 重力环境 × AI 切入点对照 | 打印贴工位 |
| [projects/php-thermal-resistance/](projects/php-thermal-resistance/) | 练手项目：PHP 热阻预测（XGBoost + SHAP + 贝叶斯优化），含数据源清单、复现脚本、技术博客 | 想动手实践者 |

## 快速开始（练手项目）

```bash
cd projects/php-thermal-resistance
pip install -r requirements.txt
python src/make_demo_data.py   # 演示数据
python src/train.py            # 训练 + 5 折 CV + SHAP 分析
python src/optimize.py         # 贝叶斯优化设计闭环
```

## 图表复现

```bash
python scripts/charts.py    # fig1–fig4（全景报告）
python scripts/charts2.py   # fig5（流体与能源技术地图）
python scripts/charts3.py   # fig6（热管理技术地图）
```

图表输出到 `docs/assets/`，中文字体使用 Noto Sans SC（脚本内含字体路径，按需修改）。

## 许可证

- **代码**（scripts/、projects/ 下的脚本）：[MIT](LICENSE)
- **文档与图表内容**（docs/、blog、poster）：[CC BY 4.0](LICENSE-docs)，转载需署名

## 声明

本仓库内容为 2026 年 10 月前公开资料调研整理，仅供学习参考，不构成职业或投资建议。
演示项目中的合成数据仅用于流程验证，定量结论以真实文献数据为准。
