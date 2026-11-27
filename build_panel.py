# -*- coding: utf-8 -*-
"""
第一步：把两张原始表拼成一张分析用的总表。

跑法：python build_panel.py（任意工作目录均可）
产出：results/panel.csv（总表）+ results/missingness.csv（缺失情况一览）
      输入读本脚本同目录 data/，输出写同目录 results/（仓库已含一次运行产物）。

总表每一行 = 某家公司某一年 + 它的 CEO。
表里包含：当年薪酬和财务信息（用来做预测的线索）+ 明年的 ROA（要预测的答案）。
"""
import pandas as pd  # 处理表格的库
import numpy as np   # 算数字的库
from pathlib import Path  # 路径：以脚本所在目录为基准，从任何工作目录运行结果一致

# ---------- 文件路径（全部相对脚本所在目录） ----------
BASE      = Path(__file__).resolve().parent
DATA_DIR  = BASE / "data"
OUT_DIR   = BASE / "results"
CEO_FILE  = DATA_DIR / "Firm_CEO_Compensation.csv"  # 薪酬表
FIRM_FILE = DATA_DIR / "Firm_Information.csv"       # 财务表
OUT_PANEL = OUT_DIR / "panel.csv"
OUT_MISS  = OUT_DIR / "missingness.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)  # 建输出文件夹（已有就跳过）

# ============================================================
# 第 1 步：读入两张表，只留 CEO
# ============================================================
ceo  = pd.read_csv(CEO_FILE)
firm = pd.read_csv(FIRM_FILE)

# 薪酬表里有个标记列 ceoann：'CEO' 或 'CFO'。只留 CEO（6,038 行）。
ceo = ceo[ceo["ceoann"] == "CEO"]

# 财务表的列名是大写，统一改成小写，方便拼接
firm = firm.rename(columns={"GVKEY": "gvkey", "YEAR": "year"})

# ============================================================
# 第 2 步：按「公司编号 + 年份」把两张表拼起来
# ============================================================
# 拼完每行就是：这家公司这一年，CEO 拿了多少 + 公司业绩如何。
panel = ceo.merge(firm, on=["gvkey", "year"], how="left")

# ============================================================
# 第 3 步：给每行找「下一年的 ROA」（要预测的答案）
# ============================================================
# 规则：同一家公司里，按年份排序，取紧挨着的下一行。
#      如果下一行不是 year+1（说明中间断了一年），就不要——
#      不能拿 2013 年冒充 2012 年。
tmp = firm[["gvkey", "year", "ROA"]].sort_values(["gvkey", "year"]).copy()
tmp["roa_next"]   = tmp.groupby("gvkey")["ROA"].shift(-1)    # 下一年的 ROA
tmp["year_after"] = tmp.groupby("gvkey")["year"].shift(-1)   # 下一行是哪年
tmp.loc[tmp["year_after"] != tmp["year"] + 1, "roa_next"] = np.nan  # 断档就作废

panel = panel.merge(tmp[["gvkey", "year", "roa_next"]],
                    on=["gvkey", "year"], how="left")

# ============================================================
# 第 4 步：算出每个指标缺了多少（缺失值处理需要留记录）
# ============================================================
# 先挑出要用作「线索」的原始指标：
raw_cols = [
    # 薪酬
    "tdc1",           # 总薪酬（千美元）
    "salary",         # 工资
    "bonus",          # 奖金
    "stock_awards",   # 股票奖励
    "option_awards",  # 期权奖励
    # 公司
    "ASSETS",         # 总资产
    "SALES",          # 销售额
    "MKTVAL",         # 市值
    "SEQ",            # 股东权益
    "ROA",            # 当年资产收益率
    "SALECHG",        # 销售增长
    "ASSETCHG",       # 资产增长
    "TRS1YR",         # 近 1 年股票回报
    "TRS3YR",         # 近 3 年股票回报
    # CEO 个人
    "age",            # 年龄
]
missing = panel[raw_cols].isna().mean() * 100   # 每个指标缺的百分比
missing.round(2).rename("缺失百分比").to_csv(OUT_MISS)

# 不用的两个指标（缺太多，提前说明）：
#   ret_yrs（任职年限）：100% 缺，完全没数据
#   NICHG（净利润变化）：全表缺失 54.0%，且本管线未读取该列（旧稿 17.8% 系合并子集口径）
# 其他指标最多缺 9%，用平均值补。

# ============================================================
# 第 5 步：把原始指标加工成「线索」（特征）
# ============================================================
# 全部只用当年信息，不碰下一年的任何东西（防止偷看答案）。

def 取对数(列):
    """取对数；小于 1 的值（含 0 和负数）一律按 1 算，避免算错。
    这只影响极少数公司（比如市值暂时为 0 的），影响可忽略。"""
    return np.log(列.clip(lower=1))

panel = panel.copy()  # 一次性复制，消除 pandas 的碎块警告
panel["总薪酬对数"] = 取对数(panel["tdc1"])
panel["ln_assets"] = 取对数(panel["ASSETS"])
panel["ln_sales"]  = 取对数(panel["SALES"])
panel["ln_mktval"] = 取对数(panel["MKTVAL"])

# 薪酬结构：各类钱占总薪酬的比例（tdc1 为 0 时算不出来，先留空，后面补）
total = panel["tdc1"].replace(0, np.nan)
panel["工资占比"]   = panel["salary"] / total
panel["奖金占比"]   = panel["bonus"] / total
panel["股权占比"]   = (panel["stock_awards"] + panel["option_awards"]) / total

# 杠杆率：公司资产里有多少是借来的钱（资产 - 股东的钱 = 借的钱）
panel["杠杆率"] = 1 - panel["SEQ"] / panel["ASSETS"].replace(0, np.nan)
panel["当年ROA"] = panel["ROA"]

# 最终用到的 14 个线索
features = [
    "总薪酬对数", "工资占比", "奖金占比", "股权占比",   # 薪酬水平与结构
    "ln_assets", "ln_sales", "ln_mktval",              # 规模
    "当年ROA", "杠杆率",                                # 业绩与负债
    "SALECHG", "ASSETCHG",                             # 增长
    "TRS1YR", "TRS3YR",                                # 过去表现
    "age",                                             # CEO 年龄
]

# 缺失值：用这一列的平均值填上（更严谨的做法是改为只用训练集的平均值）
panel[features] = panel[features].fillna(panel[features].mean())

# ============================================================
# 第 6 步：定义「坏结果」——明年 ROA 掉得最惨的 20%
# ============================================================
# ROA 下降幅度 = 当年 ROA - 明年 ROA（越大越糟）
panel["roa_decline"] = panel["当年ROA"] - panel["roa_next"]

# 下降幅度排在最差 20%（即最大的 20%）→ 标记 1 = 值得优先审查
threshold = panel["roa_decline"].quantile(0.80)  # 80% 分位线
panel["y_review"] = (panel["roa_decline"] >= threshold).astype(int)
panel.loc[panel["roa_decline"].isna(), "y_review"] = np.nan  # 没有明年的行不标

# ---------- 两个备用定义（稳健性检验用） ----------
# 目的：证明结论不是碰巧挑了个合适的口径，换把尺子量结论也一样。
# 备用定义 1：门槛放宽到最差 25%
th25 = panel["roa_decline"].quantile(0.75)
panel["y_review_25"] = (panel["roa_decline"] >= th25).astype(int)
panel.loc[panel["roa_decline"].isna(), "y_review_25"] = np.nan
# 备用定义 2：ROA 下降超过 5 个百分点就算（固定尺子，不看排名）
panel["y_review_5pp"] = (panel["roa_decline"] >= 5).astype(int)
panel.loc[panel["roa_decline"].isna(), "y_review_5pp"] = np.nan

# ============================================================
# 第 7 步：存盘 + 打印摘要
# ============================================================
keep = (["gvkey", "year", "exec_fullname", "coname"]     # 公司与 CEO 身份
        + features                                        # 线索
        + ["roa_next", "roa_decline", "y_review",          # 答案相关
           "y_review_25", "y_review_5pp"])                  # 两个备用答案
panel[keep].to_csv(OUT_PANEL, index=False)

labeled = panel[panel["y_review"].notna()]
print("总行数              :", len(panel))
print("配上明年结果的行数  :", len(labeled), "（按年：",
      labeled["year"].value_counts().to_dict(), "）")
print("被标记为需审查的比例:", round(panel["y_review"].mean(), 3))
print("备用定义1（最差25%）标记比例:", round(panel["y_review_25"].mean(), 3))
print("备用定义2（降超5点）标记比例:", round(panel["y_review_5pp"].mean(), 3))
print("线索个数            :", len(features))
print("已生成", OUT_PANEL, "和", OUT_MISS)
