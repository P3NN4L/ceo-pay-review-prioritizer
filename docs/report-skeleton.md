# Decision Report Skeleton — MAIB7010 Group Project

> **Owner: E（李展健）** · Status: v0.1 skeleton (2026-11-04)
> 规则：每节给成句英文骨架 + 页数配额 + 素材来源 + `[TODO: 负责人]` 槽位。全组往里扩，统稿只做删与顺，不重写。
> 硬约束：**正文 ≤10 页，单倍行距**——配额总计 9.5 页，留 0.5 页缓冲。

---

## Title & Header (0 p.)

**Is the CEO Pay Worth the Money? An AI-Assisted Executive-Compensation Review**

> Decision report for the Compensation Committee · MAIB7010 Group __ · Date: 2026-__

*[TODO: E] 组号、成员名单（按提交要求格式）*

---

## §1 Executive Summary (0.5 p.)

> We build a screening model that flags firm-years whose CEO pay packages merit priority human review. Using ExecuComp compensation linked to firm fundamentals for [N] CEO-firm-years (2011–2013), we predict a substantial next-year ROA decline and compare a transparent scorecard, logistic regression, and a random forest. Under a time-based split, the best model achieves `[TODO: D] precision/recall]`, versus `[TODO]` for the baseline scorecard. We recommend reviewing the top `[TODO: k]` ranked cases per year, subject to the human checks in §7.

- [ ] 数字占位：N（样本量）、precision@k、lift vs baseline —— **D 填**
- [ ] 结论一句话是否已和 §7 一致 —— **E 统稿时核**

## §2 The Decision Problem (1 p.)

> A compensation committee can manually review only a small share of CEO pay packages each year. We frame the question as a **prioritization** problem: rank firm-years by the risk that next-year performance deteriorates, so that limited review capacity is spent where it matters. The model recommends review only — it does not set pay, allege misconduct, or claim that changing pay would improve performance.

- [ ] 决策场景一段（委员会视角，非技术）—— **E**
- [ ] 三条边界（不定价/不指控/不因果）显式成段 —— **E**
- [ ] 图 1：审查决策流程（见 workflow.md 图 1 引用）—— **E**

## §3 Data & Sample Construction (1.5 p.)

> We construct one observation per CEO-firm-fiscal-year, linking `Firm_CEO_Compensation` and `Firm_Information` on GVKEY-year. Features use information available by year *t* only: pay level and mix (TDC1, salary share, equity share), firm size (ln assets), profitability (ROA), leverage, growth, and prior performance (stock returns). The outcome is defined for year *t+1*.

| 槽位 | 内容 | 负责人 |
|---|---|---|
| [TODO] | 最终样本量表：CEO 行数 → 合并后 → t+1 可配对数 | A |
| [TODO] | CEO 筛选规则（`ceoann` 口径、换帅年处理） | A |
| [TODO] | 缺失值决策：删除 vs 均值（题目二选一），每列缺失率摘要 | A/B |
| [TODO] | 完整特征表（变量名 + 定义 + t 年可得性） | B |
| [ ] | 行业调整缺位的说明（无 SIC/NAICS → 分位数替代，见 §4） | A |

## §4 Outcome Definition & Robustness (1 p.)

> Our primary review signal is a substantial decline in ROA from *t* to *t+1* (bottom-quintile of the within-sample distribution). Because the dataset lacks industry codes, "industry-adjusted" ROA is approximated by the full-sample quantile threshold; we verify robustness with two alternative definitions.

- [ ] 主定义精确口径（分位阈值、方向、极值处理）—— **B**
- [ ] 替代定义 A/B 及各定义下正例率表 —— **B**
- [ ] 稳健性结论一句（结论是否随定义改变）—— **B + C**

## §5 Models & Validation Design (1.5 p.)

> We compare (i) a transparent scorecard baseline (e.g., flag if TDC1 > 75th percentile and ROA < median), (ii) logistic regression, and (iii) a random forest. All models are trained on earlier years and evaluated on later years (time-based split), never on random splits, to mirror the committee's real use: ranking today's firms with yesterday's lessons.

| 槽位 | 内容 | 负责人 |
|---|---|---|
| [TODO] | 切分方案（train 2011(+2012) → test 2013）与理由 | C |
| [TODO] | 超参数与调参方式（validation 用法） | C |
| [TODO] | 防数据泄露自检清单 | C |
| [ ] | 逻辑回归系数方向表（可解释性卖点） | C |

## §6 Results & Evaluation for the Decision (2 p.)

> We report precision, recall, and calibration, and — most importantly for the committee — performance at a realistic review capacity: of the top-*k* flagged firm-years, how many truly deteriorate, compared with the baseline scorecard.

- [ ] 表：三模型 × 指标（precision / recall / AUC / Brier）—— **D**
- [ ] 图：lift 曲线 / precision@k 曲线（vs baseline）—— **D**
- [ ] 校准图（预测概率 vs 实际频率）—— **D**
- [ ] 一段「模型比基线值多少」的商业语言翻译 —— **D**

## §7 Case Studies & Limitations (1.5 p.)

> Four to six firm-year cases from the test set: true positives the model caught, false alarms, misses, and one surprising case. Limitations: small panel (2011–2013), no industry codes, review signal ≠ misconduct evidence.

| 槽位 | 内容 | 负责人 |
|---|---|---|
| [TODO] | 案例 1–6 卡片（公司/CEO/年份/薪酬结构/模型分数/实际结果/一句话解读） | D |
| [ ] | Limitations 成段（小样本/无行业/非因果/校准风险） | D + E |

## §8 Recommended Review Policy (1 p.)

> We recommend the committee review the top-[k] ranked firm-years each year (≈ [x]% of the sample). Before any action, staff should verify: (1) one-off events (restructuring, write-downs), (2) industry-wide shocks, (3) pay-contract details (new grants vs. vesting). The model is a prioritization aid, not a judgment.

- [ ] 推荐容量 k + 对应 precision 数字 —— **D**
- [ ] 人工核查三步清单成文 —— **E**
- [ ] 何时不应使用模型（边界重申）—— **E**

## §9 References & Appendix (≤0.5 p. + appendix)

- [ ] 数据来源（ExecuComp/Compustat 口径）、课程 PDF、方法引用 —— **E**
- [ ] 附录（不计页）：完整变量表、额外稳健性表 —— **A/B/C/D 各自给**

---

## 页数配额汇总（正文合计 9.5 页 + 0.5 缓冲）

| 节 | 页 | 主笔 |
|---|---|---|
| §1 Exec Summary | 0.5 | E |
| §2 Decision Problem | 1 | E |
| §3 Data | 1.5 | A |
| §4 Outcome | 1 | B |
| §5 Models | 1.5 | C |
| §6 Results | 2 | D |
| §7 Cases & Limits | 1.5 | D |
| §8 Policy | 1 | E+D |

> **统稿规则（E）**：每人交稿不许超自己配额；超了删自己的，不删别人的；图表一律 `.png` 放 `docs/assets/`，正文只引用。
