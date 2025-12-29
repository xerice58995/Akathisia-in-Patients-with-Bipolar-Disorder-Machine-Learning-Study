from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    average_precision_score,  # 對應 AUPR
    classification_report,
    confusion_matrix,
    fbeta_score,  # 對應 F-beta
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sqlalchemy import create_engine

# ==========================================
# 1. 讀取與清理資料 (Data Preparation)
# ==========================================

# 連線資料庫
db_connection = create_engine(
    "postgresql://postgres:xerice58995@localhost:5432/FDA_raw_data"
)

print("正在讀取資料...")
# 只讀取我們需要的特徵：藥名、年齡、性別、反應
query = """
SELECT drug, sex, age, receipt_date, reactions, is_akathisia
FROM raw_data
"""
df = pd.read_sql(query, db_connection)
df["receipt_date"] = pd.to_datetime(df["receipt_date"], errors="coerce")

# --- 副作用編碼 (Feature: reactions) ---
target_keywords = ["akathisia", "restlessness", "hyperactivity"]

# 篩選常見的副作用
all_reactions = []

df["reaction_list"] = df["reactions"].str.split(", ")
train_mask = df["receipt_date"].dt.year < 2025

for reactions in df.loc[train_mask, "reaction_list"]:
    for r in reactions:
        if r.lower() not in target_keywords:
            all_reactions.append(r)

reaction_counts = Counter(all_reactions)
common_reactions = [r for r, c in reaction_counts.items() if c >= 50]

# 轉成one-hot
for r in common_reactions:
    df[f"reaction_{r}"] = df["reaction_list"].apply(lambda x: int(r in x))


# --- 藥物編碼 (Feature: Drug) ---
# Brexpiprazole = 1, Aripiprazole = 0
df["drug_type"] = df["drug"].apply(lambda x: 1 if "Brexpiprazole" in x else 0)

# ---  準備訓練集 ---
# 選取最終特徵，將2025年的資料分開作為最後預測
test_year = 2025
x_train = df[df["receipt_date"].dt.year < test_year].copy()
y_train = x_train["is_akathisia"]
x_test = df[df["receipt_date"].dt.year >= test_year].copy()
y_test = x_test["is_akathisia"]
x_train = x_train.drop(
    columns=["receipt_date", "drug", "reactions", "reaction_list", "is_akathisia"]
)
x_test = x_test.drop(
    columns=["receipt_date", "drug", "reactions", "reaction_list", "is_akathisia"]
)

print(f"訓練集有效樣本數: {len(x_train)}，測試集有效樣本數: {len(x_test)}")

# ==========================================
# 2. 模型訓練 (XGBoost Training)
# ==========================================

# 切分測試集
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 計算不平衡比例
ratio = (len(y_train) - y_train.sum()) / y_train.sum()

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=ratio,
    # --- GPU 設定 ---
    device="cuda",
    tree_method="hist",
    eval_metric="logloss",
)

# 定義我們要觀察的指標
scoring = {
    "ROCAUC": "roc_auc",
    "AUPR": "average_precision",  # 對應 micro-AUPR 概念
    "F_beta_3": make_scorer(fbeta_score, beta=3),  # 自訂 F-beta score
}

print("正在進行訓練...")
# 如果5-Fold分數忽高忽低（例如：0.8, 0.6, 0.9...），代表模型很不穩定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)

print("\n--- 各折 (Fold) 詳細分數 ---")
for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"AUC={cv_results['test_ROCAUC'][i]:.4f}, "
        f"AUPR={cv_results['test_AUPR'][i]:.4f}, "
        f"F-beta={cv_results['test_F_beta_3'][i]:.4f}"
    )

print(f"5-Fold 平均 ROC-AUC:  {cv_results['test_ROCAUC'].mean():.4f}")
print(f"5-Fold 平均 AUPR:     {cv_results['test_AUPR'].mean():.4f}")
print(f"5-Fold 平均 F-beta(3):{cv_results['test_F_beta_3'].mean():.4f}")

print("使用全部訓練集預測最終模型...")
model.fit(x_train, y_train)

# 預測 2025 資料
# 硬分類0/1 (Hard Classification)，計算 Accuracy (準確率)、F1-Score、Confusion Matrix (混淆矩陣)
y_pred_2025 = model.predict(x_test)
# 軟分類 (Soft Classification)，計算 AUC-ROC、AUPR 時必須使用這個，因為這些指標看的是模型排序的能力，而不是單純的 0 或 1
y_prob_2025 = model.predict_proba(x_test)[:, 1]

# 計算指標
auc = roc_auc_score(y_test, y_prob_2025)
aupr = average_precision_score(y_test, y_prob_2025)
fbeta3 = fbeta_score(y_test, y_pred_2025, beta=3)

print("\n" + "=" * 40)
print(f"2025年最終測試結果")
print("=" * 40)
print(f"Micro-AUROC (ROC-AUC) : {auc:.4f}")
print(f"Micro-AUPR (Avg Prec) : {aupr:.4f}  <-- 針對不平衡資料請看這個")
print(f"F-beta Score (beta=3) : {fbeta3:.4f} <-- 重視 Recall 的指標")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_2025))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_2025)
print(cm)

# ==========================================
# 特徵重要性繪圖
# ==========================================
plt.figure(figsize=(10, 6))
model.get_booster().feature_names = list(x_train.columns)
xgb.plot_importance(
    model, max_num_features=20, importance_type="gain", show_values=False
)
plt.title("XGBoost Feature Importance (Gain) - Top 20")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# 設定繪圖風格
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Microsoft JhengHei",
]  # 解決中文字型問題(MacOS/Windows)
plt.rcParams["axes.unicode_minus"] = False

# 建立一個畫布，準備畫 4 張子圖 (2x2 排列)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 調整子圖間距

# =========================================
# 圖 1: 混淆矩陣熱力圖 (Confusion Matrix Heatmap)
# =========================================
# 您的結果數據
cm = np.array([[328, 181], [8, 19]])

# 定義標籤
group_names = [
    "True Neg (沒事)",
    "False Pos (誤報)",
    "False Neg (漏抓)",
    "True Pos (抓到)",
]
group_counts = [f"{value:0.0f}" for value in cm.flatten()]
# 計算百分比 (相對於總數)
group_percentages = [f"{value:.1%}" for value in cm.flatten() / np.sum(cm)]
# 組合標籤文字
labels = [
    f"{v1}\n{v2}\n({v3})"
    for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(
    cm,
    annot=labels,
    fmt="",
    cmap="Blues",
    cbar=False,
    ax=axes[0, 0],
    xticklabels=["Predicted Negative", "Predicted Positive"],
    yticklabels=["Actual Negative", "Actual Positive"],
    annot_kws={"size": 11},
)

axes[0, 0].set_title(
    "Figure 1. Confusion Matrix (Test Data 2025)\n(High False Positive Rate)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 0].set_ylabel("True Label", fontsize=12)
axes[0, 0].set_xlabel("Predicted Label", fontsize=12)


# =========================================
# 圖 2: ROC 曲線圖 (ROC Curve)
# =========================================
# 需要 y_test 和 y_prob_2025
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob_2025)
roc_auc = auc(fpr, tpr)

axes[0, 1].plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})"
)
axes[0, 1].plot(
    [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess (AUC=0.5)"
)

axes[0, 1].set_xlim([-0.02, 1.0])
axes[0, 1].set_ylim([0.0, 1.02])
axes[0, 1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
axes[0, 1].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
axes[0, 1].set_title(
    "Figure 2. ROC Curve\n(Model Discrimination Ability)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(linestyle="--")


# =========================================
# 圖 3: 精確度-召回率曲線圖 (Precision-Recall Curve)
# =========================================
# 這是不平衡資料最重要的圖
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob_2025)
pr_auc = auc(recall, precision)
baseline = y_test.sum() / len(y_test)  # 正樣本比例

axes[1, 0].plot(
    recall, precision, color="purple", lw=2, label=f"PR Curve (AUPR = {pr_auc:.4f})"
)
axes[1, 0].axhline(
    y=baseline, color="gray", linestyle="--", label=f"Baseline (Ratio={baseline:.3f})"
)

axes[1, 0].set_xlim([0.0, 1.02])
axes[1, 0].set_ylim([0.0, 1.02])
axes[1, 0].set_xlabel("Recall (Sensitivity)", fontsize=12)
axes[1, 0].set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
axes[1, 0].set_title(
    "Figure 3. Precision-Recall Curve\n(Crucial for Imbalanced Data)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 0].legend(loc="upper right")
axes[1, 0].grid(linestyle="--")

# 標示出目前 0.5 門檻點的位置 (大約)
current_recall = 19 / (19 + 8)  # 0.70
current_precision = 19 / (19 + 181)  # 0.095
axes[1, 0].plot(
    current_recall,
    current_precision,
    "ro",
    markersize=8,
    label="Current Threshold (0.5)",
)
axes[1, 0].text(
    current_recall + 0.02,
    current_precision + 0.05,
    f"Current Point\nRecall={current_recall:.2f}\nPrec={current_precision:.2f}",
    color="red",
)


# =========================================
# 圖 4: 特徵重要性 (Feature Importance)
# =========================================
# 需要 model 物件
# 抓取前 15 個最重要的特徵
importance_type = "gain"  # 或 'weight', 'cover'
top_n = 15
sorted_idx = model.feature_importances_.argsort()[::-1][:top_n]
top_features = [model.get_booster().feature_names[i] for i in sorted_idx]
top_importance = model.feature_importances_[sorted_idx]

sns.barplot(x=top_importance, y=top_features, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title(
    f"Figure 4. Top {top_n} Feature Importance (Gain)\n(What drives the risk?)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 1].set_xlabel("Importance Score (Gain)", fontsize=12)
axes[1, 1].set_ylabel("Features", fontsize=12)

# 顯示圖表
plt.tight_layout()
# plt.savefig("Akathisia_Model_Results_Summary.png", dpi=300) # 存檔用
plt.show()
