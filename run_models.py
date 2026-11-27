# -*- coding: utf-8 -*-
"""
run_models.py — 建模管线（读取 build_panel.py 的产物 results/panel.csv）

口径（原定方案 + 数据现实修正）：
  - 数据覆盖 2011-2013；标签 = t+1 年 ROA 大幅下滑，所以只有 2011/2012 两行的行有标签。
  - 评估用时间切分：训练 2011 → 测试 2012（唯一有真值的切法，严禁随机切分）。
  * 原计划"测试 2013"不可行：2013 行的 t+1（=2014）不在数据里，无真值。
    → 2013 的角色：用 2011+2012 训练好的模型打分，输出"优先审查清单"（交付物本身）。
    此偏离已写明（见 docs/modelling-notes.md 第 1 节）。
  - 模型：透明基线规则 + 逻辑回归 + 随机森林（NN 为可选对照，本管线砍掉）
  - 不调参：配对样本约 3.9 千行，调参=过拟合风险，全部默认参数 + 固定随机种子
  - 稳健性：三个标签定义各跑一遍，输出对比表
  - 评估：precision / recall / AUC / AP + precision@审查容量50 + 提升曲线

运行：python run_models.py（任意工作目录均可；脚本按自身目录定位 data/ 与 results/）
依赖：pip install -r requirements.txt

阅读提示：每段开头的 ★ 行说明该步骤为什么这样做，便于向使用者解释。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score,
)

# ============================================================
# CONFIG —— 产物对齐 results/panel.csv（build_panel.py 的输出）
# 路径以脚本所在目录为基准：读 results/，写 results/
# ============================================================
BASE       = Path(__file__).resolve().parent
OUT_DIR    = BASE / "results"
PANEL_CSV  = OUT_DIR / "panel.csv"
YEAR_COL   = "year"
ID_COLS    = ["gvkey", "exec_fullname", "coname"]

# 三个标签定义：主定义 + 两个替代，全部跑一遍 = 稳健性对比
LABELS = {
    "主定义_最差20%分位": "y_review",
    "替代A_最差25%分位": "y_review_25",
    "替代B_降幅超5个百分点": "y_review_5pp",
}
MAIN_LABEL_NAME = "主定义_最差20%分位"

# ★ 特征：只用 t 年信息。roa_next / roa_decline 是 t+1 真值，
#   进了特征就等于把答案喂给模型（准确率虚高的无声错误）——防泄露自检管这个。
FEATURES = [
    "总薪酬对数", "工资占比", "奖金占比", "股权占比",
    "ln_assets", "ln_sales", "ln_mktval",
    "当年ROA", "杠杆率",
    "SALECHG", "ASSETCHG", "TRS1YR", "TRS3YR", "age",
]

EVAL_TRAIN_YEAR, EVAL_TEST_YEAR = 2011, 2012   # 评估切分（唯一有真值的切法）
FINAL_TRAIN_YEARS, SCORE_YEAR   = (2011, 2012), 2013  # 最终打分：训练→给2013排序
K_REVIEW = 50        # 审查容量：委员会一年最多人工细看 50 家
RANDOM_STATE = 42

# 透明基线：t 年 ROA 越低越优先审（人类一眼能懂的规则，给模型当对照组）
BASELINE_SORT_COL, BASELINE_ASC = "当年ROA", True


def make_models():
    """★ 为什么 LR 和 RF 都要：LR 系数可直接读出'哪个薪酬特征推高风险'（委员会能看懂），
    RF 抓非线性但小样本易过拟合——两者结论一致时报告最有说服力。
    ★ 为什么标准化再跑 LR：薪酬对数 ~9、占比 ~0.1，量纲差百倍，不标准化系数不可比。"""
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    # ★ n_jobs=1 而不是 -1：并行只影响速度，不影响树；关掉它可以排除多线程带来的
    #   浮点归约顺序差异，保证"同一台机器跑两次结果逐位一致"（可复现性的最低要求）。
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=1)
    return [("logistic", lr), ("random_forest", rf)]


def precision_at_k(y_true_sorted, base_rate, k):
    """前 k 家的命中率：模型排序后前 k 名里真下滑的占比。★ 回答'审50家多抓几家'。"""
    k = min(k, len(y_true_sorted))
    hit = y_true_sorted[:k].mean() if k else float("nan")
    return k, hit


def fit_eval_one_label(df, label_col, label_name):
    """对一个标签定义：2011 训练 → 2012 评估，返回指标行 + 提升曲线 + LR 系数。"""
    labeled = df[df[label_col].notna()]
    train = labeled[labeled[YEAR_COL] == EVAL_TRAIN_YEAR]
    test  = labeled[labeled[YEAR_COL] == EVAL_TEST_YEAR]
    print(f"\n=== 标签: {label_name} ({label_col}) ===")
    print(f"[切分] 训练 {len(train)} 行(正例 {int(train[label_col].sum())}, "
          f"{train[label_col].mean():.1%}) / 测试 {len(test)} 行"
          f"(正例 {int(test[label_col].sum())}, {test[label_col].mean():.1%})")

    base_rate = test[label_col].mean()
    Xtr, ytr = train[FEATURES], train[label_col].astype(int)
    Xte, yte = test[FEATURES], test[label_col].astype(int)

    # 基线：按 t 年 ROA 升序排（ROA 越低越优先审）
    order = np.argsort(np.asarray(Xte[BASELINE_SORT_COL]))
    y_bl = np.asarray(yte)[order if BASELINE_ASC else order[::-1]]
    k_bl, hit_bl = precision_at_k(y_bl, base_rate, K_REVIEW)

    rows, lifts, coefs = [], [], None
    for name, model in make_models():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        order = np.argsort(-proba)
        y_sorted = np.asarray(yte)[order]
        k, hit = precision_at_k(y_sorted, base_rate, K_REVIEW)
        rows.append({
            "标签定义": label_name, "model": name,
            "precision": precision_score(yte, pred, zero_division=0),
            "recall":    recall_score(yte, pred, zero_division=0),
            "roc_auc":   roc_auc_score(yte, proba),
            "avg_precision": average_precision_score(yte, proba),
            f"hit@{k}": hit, "random": base_rate, "lift": hit / base_rate,
        })
        ks = np.arange(1, len(y_sorted) + 1)
        lifts.append(pd.DataFrame({
            "k": ks, "hit_rate": np.cumsum(y_sorted) / ks,
            "random": base_rate, "model": name, "标签定义": label_name}))
        if name == "logistic":
            coefs = pd.DataFrame({"feature": FEATURES,
                                  "coef_std": model.named_steps["lr"].coef_[0]})
        print(f"  {name:14s} AUC={rows[-1]['roc_auc']:.3f} AP={rows[-1]['avg_precision']:.3f} "
              f"| 前{k}家命中率 {hit:.1%}（随机 {base_rate:.1%}，提升x{hit/base_rate:.1f}）")
    print(f"  基线(按当年ROA升序) 前{k_bl}家命中率 {hit_bl:.1%}（随机 {base_rate:.1%}，"
          f"提升x{hit_bl/base_rate:.1f}）")
    return pd.DataFrame(rows), pd.concat(lifts), coefs


def save_test_predictions(df):
    """把 2012 测试集每一行的「真值 + 两模型预测分」落盘。
    ★ 为什么要有这一步：校准、
      阈值权衡、标对/标错的案例挑选，都需要逐行分数；不落盘就只能重跑脚本。
      这里只落盘原始材料，不算任何结论。"""
    label_col = LABELS[MAIN_LABEL_NAME]
    train = df[(df[YEAR_COL] == EVAL_TRAIN_YEAR) & df[label_col].notna()]
    test = df[(df[YEAR_COL] == EVAL_TEST_YEAR) & df[label_col].notna()].copy()
    out = test[ID_COLS + [YEAR_COL] + list(LABELS.values())].copy()
    for name, model in make_models():
        model.fit(train[FEATURES], train[label_col].astype(int))
        out[f"score_{name}"] = model.predict_proba(test[FEATURES])[:, 1]
    print(f"[输出] 测试集逐行预测 {len(out)} 行（含三个标签真值 + 两模型分数）→ 可直接用于校准/阈值分析")
    return out


def score_2013(df):
    """最终交付物：2011+2012 训练 → 给 2013 全部行打分 → 优先审查清单。
    ★ 注意：2013 没有真值（t+1=2014 不在数据里），所以只排序不评估——
      这正是'模型只建议审谁、不判定对错'口径的体现。"""
    label_col = LABELS[MAIN_LABEL_NAME]
    train = df[(df[YEAR_COL].isin(FINAL_TRAIN_YEARS)) & df[label_col].notna()]
    to_score = df[df[YEAR_COL] == SCORE_YEAR].copy()
    print(f"\n=== 最终打分 ({SCORE_YEAR}) ===")
    print(f"[训练] {len(train)} 行 | [打分] {len(to_score)} 行（无真值，只排序）")
    out = to_score[ID_COLS + [YEAR_COL]].copy()
    for name, model in make_models():
        model.fit(train[FEATURES], train[label_col].astype(int))
        out[f"score_{name}"] = model.predict_proba(to_score[FEATURES])[:, 1]
    # ★ 排序依据：逻辑回归（并列按 gvkey 升序，保证重跑同序）
    out = out.sort_values(["score_logistic", "gvkey"],
                          ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "审查优先级", out.index + 1)
    top = out.head(K_REVIEW)
    print(f"[交付] 审查清单已按 LR 分排序，前{K_REVIEW}家 = 建议优先审查名单")
    print(top[["审查优先级", "coname", "year", "exec_fullname", "score_logistic"]]
          .head(10).to_string(index=False))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ★ 可复现性：把运行环境写下来。sklearn/pandas 版本不同会让随机森林的
    #   建树细节和 CSV 浮点解析精度不一样，指标可能差 1-2 个百分点——这不是 bug，
    #   但引用结果时必须注明"本结果在哪个版本下产生"，否则他人重跑对不上会以为出了错。
    import sklearn, sys
    env = (f"python={sys.version.split()[0]}  sklearn={sklearn.__version__}  "
           f"pandas={pd.__version__}  numpy={np.__version__}  random_state={RANDOM_STATE}")
    print(f"[环境] {env}\n")
    with open(BASE / "RUN_ENV.txt", "w", encoding="utf-8") as f:
        f.write(env + "\n")

    df = pd.read_csv(PANEL_CSV)
    print(f"[读入] {PANEL_CSV}: {len(df)} 行 x {df.shape[1]} 列")

    # ★ 防泄露自检：特征里绝不允许出现 t+1 真值列和标签列
    forbidden = set(LABELS.values()) | {"roa_next", "roa_decline"}
    bad = forbidden & set(FEATURES)
    if bad:
        raise SystemExit(f"[自检失败] 特征里混入了真值列: {bad}")
    missing = [c for c in FEATURES + list(LABELS.values()) + [YEAR_COL] if c not in df.columns]
    if missing:
        raise SystemExit(f"[自检失败] 缺列: {missing}")

    all_metrics, all_lifts, all_coefs = [], [], []
    for name, col in LABELS.items():
        m, l, c = fit_eval_one_label(df, col, name)
        all_metrics.append(m); all_lifts.append(l)
        if c is not None:
            all_coefs.append(c.assign(标签定义=name))

    review_list = score_2013(df)
    test_pred = save_test_predictions(df)

    m = pd.concat(all_metrics, ignore_index=True)
    m.to_csv(f"{OUT_DIR}/metrics_三标签对比.csv", index=False)  # 原始输出；汇总表见 metrics_full.csv
    pd.concat(all_lifts, ignore_index=True).to_csv(f"{OUT_DIR}/lift_curve.csv", index=False)
    pd.concat(all_coefs, ignore_index=True).to_csv(f"{OUT_DIR}/lr_coefficients.csv", index=False)
    review_list.to_csv(f"{OUT_DIR}/review_list_2013.csv", index=False)
    test_pred.to_csv(f"{OUT_DIR}/test_predictions_2012.csv", index=False)
    print(f"\n[落盘] {OUT_DIR}/ 下 5 个文件：")
    print("  metrics_三标签对比.csv     ← 脚本原始输出；汇总表见 metrics_full.csv（含 Brier/标注数/基线行）")
    print("  lift_curve.csv             ← 画提升曲线（前k命中率 vs 随机线）")
    print("  lr_coefficients.csv        ← 逻辑回归系数：哪个薪酬特征推高风险")
    print("  review_list_2013.csv       ← 最终交付物：2013 优先审查清单")
    print("  test_predictions_2012.csv  ← 测试集逐行真值+分数（校准/阈值/案例用）")


if __name__ == "__main__":
    main()
