import time

import pandas as pd
import requests
from sqlalchemy import create_engine, text

brexpiprazole_drug = ["BREXPIPRAZOLE"]
aripiprazole_drug = ["ARIPIPRAZOLE"]
# requests 套件會自動把空格轉成正確的 URL 編碼 (+)
brexpiprazole = 'patient.drug.activesubstance.activesubstancename:"BREXPIPRAZOLE" OR patient.drug.medicinalproduct:"REXULTI"'
aripiprazole = 'patient.drug.activesubstance.activesubstancename:"ARIPIPRAZOLE" OR patient.drug.medicinalproduct:"ABILIFY" OR patient.drug.medicinalproduct:"ARISTADA" OR patient.drug.medicinalproduct:"MAINTENA"'

EPS_KEYWORDS = {
    "akathisia",
    "restlessness",
    "psychomotor hyperactivity",  # 靜坐不能
    "dystonia",
    "muscle spasms",
    "oculogyric crisis",
    "torticollis",  # 肌張力障礙
    "parkinsonism",
    "tremor",
    "muscle rigidity",
    "bradykinesia",  # 帕金森氏症狀
    "tardive dyskinesia",
    "dyskinesia",
    "extrapyramidal disorder",
    "movement disorder",  # 其他
}


# 抓openFDA資料
def get_adverse_events(drug_query, skip, date_range=None):
    base_url = "https://api.fda.gov/drug/event.json"
    # 組合日期查詢，逐年查詢以避免超過API請求上限
    final_query = drug_query
    if date_range:
        final_query = f"({drug_query}) AND receiptdate:[{date_range}]"

    url_params = {
        "api_key": f"LaoODfngymGBaoQlnFMBD9FXDfjgkkL7TiZu7scR",
        "search": final_query,  # 根據要查的成分修改
        "limit": 1000,
        "skip": skip,
    }

    try:
        response = requests.get(base_url, params=url_params, timeout=10)
        # 針對 404 做特別處理 (openFDA 查無資料時會回傳 404，不代表程式出錯)
        if response.status_code == 404:
            print(f"[{final_query}] 查詢結束 (已無更多資料 或 查無資料)。")
            return {"results": []}  # 回傳空結果
        response.raise_for_status()
        data = response.json()

        return data

    except requests.exceptions.RequestException as e:
        print(f"請求失敗: {e}")
        return None


def check_is_eps(reaction_str):
    """檢查副作用字串中是否包含 EPS 關鍵字"""
    if not reaction_str:
        return False
    # 轉小寫比對
    return any(keyword in reaction_str.lower() for keyword in EPS_KEYWORDS)


def check_is_akathisia(reaction_str):
    """特別標記 Akathisia"""
    if not reaction_str:
        return False
    target = {"akathisia", "restlessness", "hyperactivity"}
    return any(t in reaction_str.lower() for t in target)


# 存clean data到PostgreSQL
def data_to_sql(all_data):
    if all_data.empty:
        print("無資料可供存入PostgreSQL")
        return

    db_connection = create_engine(
        "postgresql://postgres:xerice58995@localhost:5432/FDA_raw_data"
    )

    insert_sql = text("""
            INSERT INTO raw_data (
                drug,
                safetyreportid,
                safetyreportversion,
                reactions,
                active_substance,
                sex,
                age,
                age_unit,
                receipt_date,
                is_eps,
                is_akathisia,
                primarysource
            )
            VALUES (
                :drug,
                :safetyreportid,
                :safetyreportversion,
                :reactions,
                :active_substance,
                :sex,
                :age,
                :age_unit,
                :receipt_date,
                :is_eps,
                :is_akathisia,
                :primarysource
            )
            ON CONFLICT (safetyreportid, safetyreportversion)
            DO NOTHING;
        """)
    try:
        with db_connection.begin() as conn:
            conn.execute(insert_sql, all_data.to_dict(orient="records"))
        print(f"已完成資料存入 PostgreSQL")
    except Exception as e:
        print(f"資料庫寫入失敗: {e}")


def drugs_cleaning(drug_name, drug_query):
    all_data = []

    # --- 設定要抓取的年份範圍 ---
    # 這裡設定從 2004 到 2024
    years = range(2004, 2025)
    for year in years:
        date_range = f"{year}0101 TO {year}1231"
        skip = 0
        year_fetched = 0
        print(f"📅 正在處理年份: {year} ...")

        while True:
            # # 設定一個安全上限，例如測試時只抓 5000 筆，正式跑可以拿掉
            # if total_fetched >= 2000:
            #     print(f"   達到測試上限 (2000筆)，停止抓取 {drug_name}")
            #     break

            data = get_adverse_events(drug_query, skip, date_range)
            if not data:  # 如果是 None
                print(f"{drug_name} 資料抓取失敗或已無更多資料")
                break

            results = data.get("results", [])
            if not results:
                print(f"{drug_name} 資料抓取完畢")
                break

            batch_data = []

            for event in results:
                patient = event.get("patient", {})

                # --- A. 處理副作用 ---
                reaction_list = [
                    r.get("reactionmeddrapt", "") for r in patient.get("reaction", [])
                ]
                # 過濾空值
                valid_reactions = [r for r in reaction_list if r]
                reactions_str = ", ".join(valid_reactions)

                # 安全取得 activesubstance
                actives = []
                for d in patient.get("drug", []):
                    # active_sub 是一個字典，不是 List
                    active_sub = d.get("activesubstance", {})

                    # 直接從字典取值，不要跑迴圈
                    if active_sub:
                        name = active_sub.get("activesubstancename")
                        if name:
                            actives.append(name)

                # 安全取得 primarysource
                primarysource = event.get("primarysource")
                qualification = (
                    primarysource.get("qualification") if primarysource else None
                )

                clean_event = {
                    "drug": drug_name,
                    "safetyreportid": event.get("safetyreportid"),
                    "safetyreportversion": event.get("safetyreportversion"),
                    "reactions": reactions_str,
                    "active_substance": ",".join(filter(None, actives)),
                    "sex": patient.get("patientsex"),
                    "age": patient.get("patientonsetage"),
                    "age_unit": patient.get("patientonsetageunit"),
                    "receipt_date": event.get("receiptdate"),
                    "is_eps": check_is_eps(reactions_str),  # 自動判定 EPS
                    "is_akathisia": check_is_akathisia(
                        reactions_str
                    ),  # 自動判定 Akathisia
                    "primarysource": qualification,
                }

                batch_data.append(clean_event)
                all_data.append(clean_event)

            # --- 立即寫入資料庫 (邊抓邊存) ---
            if batch_data:
                df_batch = pd.DataFrame(batch_data)
                data_to_sql(df_batch)

            fetched_count = len(results)
            year_fetched += fetched_count
            skip += fetched_count

            # 顯示進度
            print(f"   [{year}] 已下載 {year_fetched} 筆... (Skip: {skip})")

            # 如果單一年份超過 24000 筆，要小心 openFDA 限制
            if skip >= 24000:
                print(
                    f"⚠️ {year} 年資料超過 24,000 筆，為避免 400 Error，強制換下一年度 (可能會有遺漏，若需完整資料需改為按月抓取)。"
                )
                break

            time.sleep(0.5)  # 禮貌性延遲

    return all_data


if __name__ == "__main__":
    data = drugs_cleaning("Aripiprazole", aripiprazole)
    if data:
        df = pd.DataFrame(data)
        df.to_excel("aripiprazole_data.xlsx", index=False, engine="openpyxl")
    print(f"{aripiprazole_drug} 資料執行結束，共取得 {len(data)} 筆資料")
    print("資料已儲存為 aripiprazole_data.xlsx")
