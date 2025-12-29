import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sqlalchemy import create_engine

# 1. 連線資料庫
db_connection = create_engine(
    "postgresql://postgres:xerice58995@localhost:5432/FDA_raw_data"
)

# 2. 讀取原始資料 (讀取 raw_data，不是計算後的結果)
print("正在讀取原始資料...")

# --- 修改點 A: 在 SQL 中多抓取 'receipt_date' 欄位 ---
query = "SELECT safetyreportid, drug, reactions, receipt_date FROM raw_data"
df_raw = pd.read_sql(query, db_connection)

print(f"原始資料總筆數: {len(df_raw)}")

# --- 修改點 B: 進行日期篩選 (只留 2019 年以後) ---
# 確保日期格式正確 (處理可能的字串格式)
df_raw["receipt_date"] = pd.to_datetime(df_raw["receipt_date"], errors="coerce")

# 設定篩選年份
start_year = 2019
df_filtered = df_raw[df_raw["receipt_date"].dt.year >= start_year].copy()

print(f"篩選 {start_year} 年後資料筆數: {len(df_filtered)}")
print("-" * 30)

# 3. 定義目標關鍵字
target_keywords = ["akathisia", "restlessness", "hyperactivity"]

# 4. 標記是否有發生目標副作用 (使用篩選後的 df_filtered)
df_filtered["has_akathisia"] = df_filtered["reactions"].apply(
    lambda x: any(k in str(x).lower() for k in target_keywords)
)

# 5. 建立 2x2 聯列表
# 去除重複：同一個病人如果同時有 Akathisia 和 Restlessness，只能算 1 人
df_unique = df_filtered.drop_duplicates(subset=["safetyreportid"])

# 樞紐分析
table = pd.crosstab(df_unique["drug"], df_unique["has_akathisia"])

print("\n--- 2019年後計數表格 ---")
print(table)

# 提取數值
try:
    # 注意：請確保您的資料庫中 drug 名稱正確
    # A: Aripiprazole 且有 Akathisia
    a = table.loc["Aripiprazole", True]
    # C: Aripiprazole 且 無 Akathisia
    c = table.loc["Aripiprazole", False]

    # B: Brexpiprazole 且有 Akathisia
    b = table.loc["Brexpiprazole", True]
    # D: Brexpiprazole 且 無 Akathisia
    d = table.loc["Brexpiprazole", False]

    print(f"\n[數值確認 (2019+)]")
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

    # 7. 計算 Chi-Square
    obs = np.array([[a, c], [b, d]])
    chi2, p, dof, expected = chi2_contingency(obs)

    print("-" * 30)
    print(f"📊 分析結果 (2019-2024): Aripiprazole vs Brexpiprazole")
    print("-" * 30)
    print(f"ROR (Odds Ratio) : {ror:.4f}")
    print(f"95% CI           : {ror_ci_lower:.4f} - {ror_ci_upper:.4f}")
    print(f"P-value          : {p:.4e}")

    if p < 0.05:
        print("Result           : ★ 統計顯著 (Significant)")
    else:
        print("Result           : 不顯著 (Not Significant)")

    # 韋伯效應解讀邏輯
    print("-" * 30)
    print("【韋伯效應 (Weber Effect) 驗證】")
    if ror > 1 and ror_ci_lower > 1:
        print("結論: Aripiprazole 風險仍較高。")
    elif ror < 1 and ror_ci_upper < 1:
        print("結論: Brexpiprazole 風險仍顯著較高 (Ari 較安全)。")
    else:
        print("結論: 兩者無顯著差異。")
        print(">> 隨著時間推移，差異消失，強烈支持韋伯效應。")

except KeyError as e:
    print(f"錯誤: 找不到藥物名稱 key，請檢查 table 的 index 名稱: {e}")
    print(table.index)
except Exception as e:
    print(f"發生其他錯誤: {e}")
