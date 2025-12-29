import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sqlalchemy import create_engine

# 1. 連線資料庫
db_connection = create_engine(
    "postgresql://postgres:xerice58995@localhost:5432/FDA_raw_data"
)

# 2. 讀取原始資料 (讀取 raw_data，不是計算後的結果)
# 我們需要原始的每一筆病人資料來重新定義 "Is_Akathisia"
print("正在讀取原始資料...")
query = "SELECT safetyreportid, drug, reactions FROM raw_data"
df_raw = pd.read_sql(query, db_connection)

# 3. 定義目標關鍵字
target_keywords = ["akathisia", "restlessness", "hyperactivity"]

# 4. 標記是否有發生目標副作用 (大小寫轉換比對)
# 只要 reactions 字串裡面包含任一關鍵字，就標記為 True
df_raw["has_akathisia"] = df_raw["reactions"].apply(
    lambda x: any(k in str(x).lower() for k in target_keywords)
)

# 5. 建立 2x2 聯列表 (Contingency Table)
# 我們要比較：Aripiprazole vs Brexpiprazole
# 去除重複：同一個病人如果同時有 Akathisia 和 Restlessness，只能算 1 人 (非常重要!)
df_unique = df_raw.drop_duplicates(subset=["safetyreportid"])

# 樞紐分析
table = pd.crosstab(df_unique["drug"], df_unique["has_akathisia"])

# 確保表格順序正確 (通常 False 在前, True 在後，或反之，這裡我們手動抓值最保險)
# 假設 table 的列是 drug，行是 has_akathisia (True/False)
print("\n--- 原始計數表格 ---")
print(table)

# 提取數值
# 注意：這裡要看您的資料庫 drug 欄位確切名稱，假設是 'Aripiprazole' 和 'Brexpiprazole' (或其他名稱)
try:
    # A: Aripiprazole 且有 Akathisia
    a = table.loc["Aripiprazole", True]
    # C: Aripiprazole 且 無 Akathisia
    c = table.loc["Aripiprazole", False]

    # B: Brexpiprazole 且有 Akathisia
    b = table.loc[
        "Brexpiprazole", True
    ]  # 這裡要注意名稱是否為 'Brexpiprazole' 還是其他
    # D: Brexpiprazole 且 無 Akathisia
    d = table.loc["Brexpiprazole", False]

    print(f"\n[數值確認]")
    print(f"A (Ari + Aka): {a}")
    print(f"C (Ari + No):  {c}")
    print(f"B (Bre + Aka): {b}")
    print(f"D (Bre + No):  {d}")

    # 6. 計算 ROR 與 信賴區間
    ror = (a * d) / (b * c)
    ln_ror = np.log(ror)
    se_ln_ror = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ror_ci_lower = np.exp(ln_ror - 1.96 * se_ln_ror)
    ror_ci_upper = np.exp(ln_ror + 1.96 * se_ln_ror)

    # 7. 計算 Chi-Square (檢定顯著性 P-value)
    # 建立觀察值陣列 [[a, c], [b, d]]
    # 注意：scipy 的 chi2 格式通常是 [[有, 無], [有, 無]] 的矩陣
    obs = np.array([[a, c], [b, d]])
    chi2, p, dof, expected = chi2_contingency(obs)

    print("-" * 30)
    print("📊 分析結果: Aripiprazole vs Brexpiprazole (Akathisia Group)")
    print("-" * 30)
    print(f"ROR (Odds Ratio) : {ror:.4f}")
    print(f"95% CI           : {ror_ci_lower:.4f} - {ror_ci_upper:.4f}")
    print(f"P-value          : {p:.4e}")  # 科學記號，例如 1.23e-05

    if p < 0.05:
        print("Result           : ★ 統計顯著 (Significant)")
    else:
        print("Result           : 不顯著 (Not Significant)")

    # 解讀
    print("-" * 30)
    if ror > 1 and ror_ci_lower > 1:
        print("結論: Aripiprazole 發生 Akathisia 的風險顯著高於 Brexpiprazole。")
    elif ror < 1 and ror_ci_upper < 1:
        print(
            "結論: Brexpiprazole 發生 Akathisia 的風險顯著高於 Aripiprazole (即 Ari 較安全)。"
        )
    else:
        print("結論: 兩者在 Akathisia 風險上無顯著差異。")

except KeyError as e:
    print(f"錯誤: 找不到藥物名稱 key，請檢查 table 的 index 名稱: {e}")
    print(table.index)
