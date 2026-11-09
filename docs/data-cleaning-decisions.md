# Data-Cleaning Decisions — M1/M2

> Owner: A（钟诗宇，脚本由总管代笔 v1）· 2026-11-09 · 对应 `src/build_panel.py`
> 作业要求原文："missing values in your dataset should be either deleted or replaced with the mean… document your choices." ——本文即该文档。

## 样本构造

- **D1 · CEO 筛选**：`Firm_CEO_Compensation.csv` 共 6,039 行，`ceoann` 取值 `CEO` 6,038 行 / `CFO` 1 行；保留 `CEO`，删除那 1 行 CFO。
- **D2 · 一行 = CEO×firm×year**：两张表各自 gvkey-year 均无重复（已核验 0 重复键），`GVKEY+YEAR` 内连接合并后天然满足「one observation per CEO-firm-fiscal year」。合并匹配率 98.9%（5972/6039 有财务表记录，保留 left join 以记录缺口）。
- **D3 · t+1 对齐**：同 gvkey 内取下一观测年 ROA，**且仅当下一观测年恰为 year+1**（防跳年/退市缺口）。得到有效结果行 3,894（2011: 1,968；2012: 1,926）。2013 年无 t+1，仅作测试集预测对象时不用于训练标签。

## 缺失值决策（题目二选一：删除 or 均值）

| 处理 | 变量 | 理由 |
|---|---|---|
| **删除变量**（>15% 缺失） | `ret_yrs`（100%）、`NICHG`（17.8%） | 缺失过半/近两成，均值填充会制造假信息 |
| **均值填充** | 其余 14 个特征（缺失率 0–9%，见 `analysis-output/missingness.csv`） | 题目允许的口径；缺失率低，均值扭曲有限 |
| **删除行** | 仅 t+1 结果缺失的行不参与有标签训练（3,894 行保留标签） | 无 t+1 即无 y，无法监督；行本身保留在 panel.csv（y 为空） |

⚠️ 泄露提示（写给 C）：正式建模时均值应在**训练集内计算**后填充验证/测试集；panel.csv 的全样本填充仅为探索便利，`run_models.py` 须用 Pipeline(imputer+model) 重做。

## 特征清单（全部只用 t 年信息）

| 组 | 变量 |
|---|---|
| 薪酬水平/结构 | `ln_tdc1`、`salary_share`、`bonus_share`、`equity_share`（(stock+option)/tdc1） |
| 规模 | `ln_assets`、`ln_sales`、`ln_mktval` |
| 盈利/杠杆 | `roa_t`、`leverage`（=1−SEQ/ASSETS） |
| 成长 | `SALECHG`、`ASSETCHG` |
| 前期业绩 | `TRS1YR`、`TRS3YR` |
| CEO 特征 | `age` |

单位注：tdc1/ASSETS 等均为数据包原始单位（千美元）。

## 标签口径（M2 接口）

- 主定义：`roa_decline = roa_t − roa_next` 位于全样本最差 20% 分位 → `y_review=1`（正例率恰 0.200）。
- 行业调整缺位说明：数据包无 SIC/NAICS，「industry-adjusted」以全样本分位数替代（报告 §4 申明）。
- 替代定义（稳健性，C/D 在 run_models.py 内实现，不改动 panel）：最差 25% 分位；固定阈值（ROA 下降 ≥5 个百分点）。

## 已知边界

- tdc1=0 的行（若有）在 share 类特征中产生 NaN → 已随均值填充处理。
- 合并缺口（0.9% 无财务记录）经均值填充保留在样本中，`missingness.csv` 可追溯。
