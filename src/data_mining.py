import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from sqlalchemy import create_engine

# 1. 連線資料庫
db_connection = create_engine(
    "postgresql://postgres:xerice58995@localhost:5432/FDA_raw_data"
)

# 2. 讀取資料 (只讀取需要的欄位以節省記憶體)
print("正在讀取資料...")
query = "SELECT drug, reactions FROM raw_data"
df = pd.read_sql(query, db_connection)


# 3. 資料前處理：展開副作用
# 原本: "Headache, Nausea" -> 變成兩行資料
print("正在展開副作用 (Explode)... 這可能需要一點時間")
# 先把字串切成 list
df["reactions"] = df["reactions"].str.split(", ")
# 炸開 list
df_exploded = df.explode("reactions")

# 移除空值或無效的副作用
df_exploded = df_exploded[df_exploded["reactions"].str.len() > 1]

print(f"展開後總資料列數: {len(df_exploded)}")


# ---------------------計算ROR A, B, C, D----------------------------

# 1. 計算每個 (藥物, 副作用) 的出現次數 (這就是 A)
stats = df_exploded.groupby(["drug", "reactions"]).size().reset_index(name="count")

# 2. 取得總數統計
# 每個藥物的總報告數 (N_drug = A + C)
drug_totals = df_exploded.groupby("drug").size()
# 每個副作用的總報告數 (N_event = A + B)
event_totals = df_exploded.groupby("reactions").size()
# 整個資料庫的總報告數 (N_total)
grand_total = len(df_exploded)


# 3. 準備計算欄位
def calculate_metrics(row):
    # --- Part 1: 變數提取 ---
    drug = row["drug"]  # 拿到 "Aripiprazole"
    event = row["reactions"]  # 拿到 "Headache"
    a = row["count"]  # 拿到 50 (這就是 A)

    # --- Part 2: 查表找總數 ---
    # drug_totals 是一個 Series，像字典一樣。
    # drug_totals['Aripiprazole'] 會告訴你這個藥總共有多少筆報告 (A + C)
    n_drug = drug_totals[drug]

    # event_totals['Headache'] 會告訴你頭痛在整個資料庫出現幾次 (A + B)
    n_event = event_totals[event]

    # --- Part 3: 推算 B, C, D (國小數學) ---
    # C (吃這藥但沒頭痛) = 這個藥的總報告數 - 吃這藥且頭痛的人數
    c = n_drug - a

    # B (別的藥且頭痛) = 頭痛的總報告數 - 吃這藥且頭痛的人數
    b = n_event - a

    # D (別的藥且沒頭痛) = 全部資料 - (A+B+C)
    # 這裡用 grand_total (全資料庫總數) 減去 n_drug (這個藥的) 再減去 n_event (頭痛的) + a (因為重複減了一次A，要加回來)
    d = grand_total - n_drug - n_event + a

    # --- Part 4: 計算 ROR (數學公式) ---
    # 避免分母是 0 導致報錯 (除以 0 會變成無限大)
    if b == 0 or c == 0:
        ror = np.nan  # 設為空值 (NaN)
        ror_ci_lower = np.nan
        ror_ci_upper = np.nan
    else:
        # ROR 公式: (A/C) / (B/D)  => 化簡為 (A*D) / (B*C)
        ror = (a * d) / (b * c)

        # --- Part 5: 計算 95% 信賴區間 (統計學部分) ---
        # 為什麼要用 np.log (自然對數)?
        # 因為 ROR 的分佈是歪的，取 log 後會接近常態分佈，才能算標準差
        ln_ror = np.log(ror)

        # 標準誤 (Standard Error) 公式
        se_ln_ror = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

        # 算上下限：Mean ± 1.96 * SE (1.96 是 95% 信賴區間的常數)
        # 算完後用 np.exp (指數) 轉回原本的數字
        ror_ci_lower = np.exp(ln_ror - 1.96 * se_ln_ror)
        ror_ci_upper = np.exp(ln_ror + 1.96 * se_ln_ror)

    # --- Part 6: 計算 PRR ---
    # PRR 公式: (A / (A+C)) / (B / (B+D))
    # 也就是: (A / n_drug) / (B / 其他藥總數)
    denominator_prr = b / (grand_total - n_drug)

    if denominator_prr == 0:
        prr = np.nan
    else:
        prr = (a / n_drug) / denominator_prr

    # --- Part 7: 回傳結果 ---
    # 我們把算好的 8 個數字包成一個 Series 回傳
    # 這樣 Pandas 就會自動把它們變成 8 個新的欄位
    return pd.Series(
        [a, b, c, d, ror, ror_ci_lower, ror_ci_upper, prr],
        index=["A", "B", "C", "D", "ROR", "ROR_CI_Lower", "ROR_CI_Upper", "PRR"],
    )


# axis=0: 代表「直的一行一行」處理 (Column by Column)。
# axis=1: 代表 「橫的一列一列」處理 (Row by Row)。
# 因為我們要拿「同一列」裡面的 drug 和 count 來計算，所以要設 axis=1
print("正在計算 ROR 與 PRR...")
metrics = stats.apply(calculate_metrics, axis=1)
result_df = pd.concat([stats, metrics], axis=1)

# 篩選掉計算失敗的 (NaN)
result_df = result_df.dropna(subset=["ROR"])

result_df.to_excel("result_df.xlsx", index=False, engine="openpyxl")
print("計算完成！")
print("資料已儲存為 result_df.xlsx")
