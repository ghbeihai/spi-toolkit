# SPI 六件套人格量化模型

基于 **Personality Atlas**（[Wildertrek/survey](https://github.com/Wildertrek/survey)，AGPL-3.0）论文
Table 10 "Research (max coverage)" 套餐 + **SDT 补全**构建的人格量化模型：

**OCEAN（大五）+ MMPI-2（临床）+ SCM（社会认知模型）+ DT4（暗黑四联）+ TKI（冲突处理）+ SDT（自我决定论）= 32 个因子**，覆盖全部 15 个 SPI 语义聚类（**15/15**，2026-10-10 由 14/15 补全：SDT 的 Autonomy 覆盖 C7 自我定向与自主）。

输入一段文字 → 输出：
1. 六件套 **32 因子零样本人格分数**（1–7，含原文证据片段）；
2. **SPI 15 聚类命中画像**（LLM 判聚类，一致性 77.3%；可切本地嵌入离线模式）；
3. **persona 描述**（供下一期"思维/写作模仿"使用）。

## 尺子基线 v1.0.0（2026-10-10 冻结）

**这把尺子=同一测量协议，跨人可比较的前提。一旦冻结，不再随意变动。**

- **因子链**：六件套 32 因子，中英双语链资产 `assets/bilingual_factor_chains.json`（SHA-256 `3fd074ef…`）；
- **打分协议**：DeepSeek deepseek-chat，每因子独立调用（防跨因子污染）、每轮随机化顺序、JSON 结构化输出 `{score:1-7, evidence, confidence}`、双轮平均（temperature 0.3/0.7）；
- **SPI 聚类**：论文 15 聚类复现标注 `assets/spi_item_clusters.json`（6694 条，SHA-256 `b3b7956c…`），LLM 判聚类模式；
- **基线指标**：schema 合法率 100% · 判别效度 11/11 · 重测稳定性 0.28 · SPI 聚类一致性 77.3% · 15 聚类覆盖 15/15；
- **变更规则**：任何因子链、提示词、模型、轮次或聚类的变更 = 新版本，必须重跑全部基线验证并在此记录；审计入口 `python scripts/audit_scale.py`（输出 `data/results/scale_audit.json`，含资产指纹）。

## 首次使用：克隆上游 Personality Atlas

SPI 的因子链映射依赖 Personality Atlas 项目（1.1G，AGPL-3.0），本仓库不内嵌。
```bash
cd assets
git clone https://github.com/Wildertrek/survey.git
```

完成后 assets/survey/ 目录结构应包含 atlas/、data/、Embeddings/、results/spi_table.json。

## 快速开始

```bash
# 1. 配置 API key（DeepSeek，用于零样本打分与因子链翻译）
cp .env.example .env   # 填入 DEEPSEEK_API_KEY

# 2. 安装依赖（核心）
uv pip install -e .

# 3. 本地 SPI 映射（必需，需要 sentence-transformers + 3072 维嵌入数据）
uv pip install -e '.[spi-local]'
# 3072 维嵌入数据（~879MB）必须下载，否则 SPI 因子映射跑不了：
# huggingface-cli download Wildertrek/personality-atlas-3072 --repo-type dataset --local-dir assets/hf-3072

# 4. 生成中英双语因子链资产（提升中文打分质量，需要 key）
spi-quant translate

# 5. 跑画像（SPI 聚类默认 LLM 判聚类；--spi-mode local 可切离线嵌入）
spi-quant profile --file data/samples/sample_high_extra_dominance.txt --out data/results/profile.json

# 6. 仅 SPI 聚类命中（默认 LLM 判聚类，需要 key；--mode local 离线嵌入）
spi-quant spi --file data/samples/sample_dark_manipulation.txt

# 7. 四项验证（schema 合法率 / 重测稳定性 / 判别效度 / SPI 一致性）
spi-quant eval --out data/results
```

或直接跑端到端 demo：`python scripts/demo.py`

## 目录结构

```
SPI/
├── assets/
│   ├── survey/                    # atlas 仓库（AGPL-3.0，1.1GB）
│   ├── bilingual_factor_chains.json  # 中英双语因子链（translate 生成）
│   └── local_factor_embeddings.json  # 本地因子链嵌入缓存（可重建）
├── config/models.yaml             # 六件套因子清单（含尺子版本与协议）
├── src/spi_quant/
│   ├── factor_chains.py           # 因子链加载（32 因子）
│   ├── scorer_llm.py              # LLM 零样本打分器（DeepSeek）
│   ├── spi_mapper.py              # LLM 判聚类 + 本地嵌入 SPI 映射
│   ├── persona.py                 # 画像 → 人设描述
│   ├── validate.py                # 四项验证
│   └── cli.py                     # 命令行入口
├── data/samples/                  # 判别效度样本（6 个典型人格文本）
├── data/results/                  # 画像与评估输出
└── scripts/demo.py                # 端到端示例
```

## 打分协议（scorer_llm.py，v1.0.0 冻结）

- 单因子独立调用（32 因子 × 2 轮 = 64 次调用/文本），JSON 结构化输出
  `{"score": 1-7, "evidence": [...], "confidence": 0-1}`；
- 双轮平均（temperature 0.3/0.7）、因子顺序每轮随机化；
- 中文适配：优先使用中英双语因子链；
- 每个画像 JSON 携带 `scale` 元信息（版本/冻结状态/协议），保证跨人比较可追溯；
- **无 API key 时自动进入模拟模式**（确定性启发式，结果带 `simulated: true` 标记，仅供流程验证）。

## 当前验证结果（2026-10-10，DeepSeek deepseek-chat，真实打分）

| 验证项 | 结果 |
|---|---|
| Schema 合法率（结构化 JSON 成功率） | 100%（58/58） |
| 重测稳定性（3 轮，1-7 标尺） | 与首轮平均绝对差 0.28；因子标准差均值 0.23 |
| 判别效度（6 个典型人格样本 × 断言） | 100%（11/11），全部预期因子 ≥6.5 |
| SPI 一致性（LLM top vs 本地嵌入） | **33.3%**（mpnet-768d 英文签名） |

> 判别效度样例：高支配文本 → Extraversion 7.0 / Competing 6.5 / Psychopathy 5.0；
> 暗黑操纵文本 → Machiavellianism 7.0 / Psychopathy 7.0 / Sadism 6.0 / Agreeableness 1.0。
> 完整明细见 `data/results/eval_report.md` 与 `demo_real.json`。

### SPI 一致性实验记录（B 路线：本地嵌入升级）

| 方案 | 一致性 | 结论 |
|---|---|---|
| MiniLM-384d 英文签名（初版） | 27.8% | 基线 |
| mpnet-768d 英文签名 | 33.3% | +5.5pp，已设为默认 |
| mpnet-768d 中文签名 | 33.3% | 无增益 |

**根因**：论文的 15 聚类定义在 OpenAI 1536 维嵌入空间中，任何本地嵌入模型（384/768 维）的向量与其属不同语义空间，跨空间命中的天花板约 33%。要突破需与论文同空间（OpenAI embedding API）或放弃"命中论文聚类"、改为本地自建聚类。SPI 通道定位为探索性参考，不影响 LLM 打分主链路。

### SPI 一致性实验记录（C 路线：LLM 零样本判聚类，2026-10-10）

实验：用论文同参数复现 15 聚类（PCA-50 + KMeans-15，seed 42；簇规模与模型构成与论文 15/15 一致），分层抽样 75 条词元条目，DeepSeek 零样本判聚类。

| 方案 | 一致性 | 结论 |
|---|---|---|
| MiniLM-384d 英文签名 | 27.8% | 基线 |
| mpnet-768d 英文签名 | 33.3% | 本地嵌入天花板 |
| **DeepSeek LLM 零样本判聚类** | **77.3%** | **远超本地嵌入，无需 OpenAI embedding** |

混淆集中在语义相邻聚类（Activation↔Grandiose Narcissism、Dominance↔Dark Manipulation、Reactive Aggression↔Dark Manipulation），非系统性偏差。
**结论**：SPI 聚类映射升级方向 = LLM 判聚类（复用打分 API、零额外成本），本地嵌入路线关闭，无需采购 OpenAI embedding API。
实验报告：`data/results/spi_llm_cluster_exp.md`；复现标注：`assets/spi_item_clusters.json`（6694 条）。

## 精度与使用边界

- LLM 零样本精度上限约为相关 r≈0.29（学界已证），产物为**探索性画像**；
- 分数应结合置信度与原文证据片段解读；文本过短（<30 字）时画像置信度显著下降；
- 高因子临床模型（MMPI）为通用语言模型零样本编码，分数仅供参考；
- atlas 资产（`assets/survey/`、`assets/hf-3072/`）遵循 AGPL-3.0 许可；本项目自身代码可自由使用，商用集成 atlas 资产时请注意许可传染性。

## 全量资产与扩展（不限于六件套）

- `assets/survey/` 为论文仓库**完整克隆**：`atlas/` 下含全部 39 个模型的 `MODEL_CARD.md`、`datasets/` 含全部模型因子链 CSV、`Embeddings/` 含全部模型嵌入与聚类标签；
- `assets/hf-3072/` 为论文 HF 配套的 3072 维嵌入数据；
- 默认评分跑六件套 32 因子；要扩展任意模型，在 `config/models.yaml` 的 `models` 列表中加入 `{id, file}` 即可，打分器与 SPI 映射自动跟随（因子数上限 = 仓库全部 358 个因子）。**注意：扩展即改尺子，须按『尺子基线』变更规则重跑验证。**

## 致谢

- 论文：Raetano, Gregor & Tamang (2026). *A Survey and Computational Atlas of Personality Models*. ACM TIST 17(5), 118. DOI: 10.1145/3828667
- 数据与代码：https://github.com/Wildertrek/survey
