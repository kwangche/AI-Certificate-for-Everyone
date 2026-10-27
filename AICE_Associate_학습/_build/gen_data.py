"""AICE Associate 학습용 가상 데이터 생성기 (고정 시드로 항상 같은 데이터 생성)
사용법: 이 폴더(AICE_Associate_학습)에서  python _build/gen_data.py .
"""
import numpy as np, pandas as pd, sys, os

OUT = sys.argv[1]
rng = np.random.default_rng(2026)
sig = lambda z: 1 / (1 + np.exp(-z))

# ---------------------------------------------------------------- cust_data.csv
N = 9930
cls = rng.choice(list("ABCDEFGH"), N, p=[.08, .12, .2, .22, .16, .1, .07, .05])
sex = rng.choice(["M", "F"], N, p=[.55, .45])
age = np.clip(rng.normal(45, 13, N).round(), 18, 90).astype(int)
service = rng.choice([1, 2, 3, 4, 5], N, p=[.45, .3, .15, .07, .03])
stop = rng.choice(["Y", "N"], N, p=[.08, .92])
npay = rng.choice(["Y", "N"], N, p=[.15, .85])
cls_lvl = pd.Series(cls).map({c: i for i, c in enumerate("ABCDEFGH")}).values
A_bill = np.clip(rng.gamma(2.2, 5500, N) + cls_lvl * 1800, 0, None)
B_bill = np.clip(rng.gamma(1.5, 3000, N) * (rng.random(N) < .7), 0, None)
avg_bill = A_bill + B_bill + rng.normal(0, 600, N)
avg_bill = np.clip(avg_bill, 1000, None)
z = (-1.5 + 4.0 * (stop == "Y") + 3.0 * (npay == "Y") - 1.0 * (service - 1)
     + 0.15 * (35 - age).clip(0) + 0.25 * cls_lvl - avg_bill / 12000)
term = np.where(rng.random(N) < sig(z), "Y", "N")
cust = pd.DataFrame(dict(**{"class": cls}, sex=sex, age=age, service=service, stop=stop,
                         npay=npay, avg_bill=avg_bill.round(4), A_bill=A_bill.round(4),
                         B_bill=B_bill.round(4), termination=term))
cust.to_csv(f"{OUT}/cust_data.csv", index=False)

# cust_data1.csv : 같은 구조 + 결측치/이상치 포함(퀴즈용)
c1 = cust.copy()
for col, n in [("age", 85), ("avg_bill", 60), ("class", 40), ("A_bill", 25)]:
    c1.loc[rng.choice(N, n, replace=False), col] = np.nan
c1.loc[rng.choice(N, 12, replace=False), "avg_bill"] = rng.uniform(400000, 900000, 12).round(4)
c1.loc[rng.choice(N, 6, replace=False), "age"] = rng.choice([120, 135, 150], 6)
c1.to_csv(f"{OUT}/cust_data1.csv", index=False)

# ---------------------------------------------------------------- sc_cust_info_txn_v1.5.csv
sc = pd.DataFrame({
    "cust_class": cust["class"], "sex_type": cust["sex"], "age": cust["age"].astype(float),
    "efct_svc_count": cust["service"], "dt_stop_yn": cust["stop"], "npay_yn": cust["npay"],
    "r3m_avg_bill_amt": (cust["avg_bill"] * rng.normal(1, .03, N)).round(2),
    "r3m_A_avg_arpu_amt": (cust["A_bill"] * rng.normal(1, .03, N)).round(2),
    "r3m_B_avg_arpu_amt": (cust["B_bill"] * rng.normal(1, .03, N)).round(2),
    "termination_yn": cust["termination"],
})
for col, n in [("age", 520), ("r3m_avg_bill_amt", 310), ("r3m_A_avg_arpu_amt", 410), ("cust_class", 150), ("sex_type", 95)]:
    sc.loc[rng.choice(N, n, replace=False), col] = np.nan
idx = rng.choice(N, 45, replace=False)
sc.loc[idx, "r3m_avg_bill_amt"] = rng.uniform(250000, 1200000, 45).round(2)
sc.loc[rng.choice(N, 15, replace=False), "age"] = rng.choice([0, 1, 125, 140, 999], 15)
sc.loc[rng.choice(N, 20, replace=False), "efct_svc_count"] = rng.choice([30, 45, 60], 20)
sc.to_csv(f"{OUT}/sc_cust_info_txn_v1.5.csv", index=False)

# ---------------------------------------------------------------- cust_info.csv + service_info.csv (통신사 이탈 사례)
M = 7043
ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(M)]
gender = rng.choice(["Male", "Female"], M)
senior = (rng.random(M) < .16).astype(int)
partner = rng.choice(["Yes", "No"], M, p=[.48, .52])
dep = np.where(partner == "Yes", rng.choice(["Yes", "No"], M, p=[.5, .5]), rng.choice(["Yes", "No"], M, p=[.1, .9]))
contract = rng.choice(["Month-to-month", "One year", "Two year"], M, p=[.55, .21, .24])
ten_base = {"Month-to-month": 18, "One year": 40, "Two year": 56}
tenure = np.array([int(np.clip(rng.normal(ten_base[c], 17), 1, 72)) for c in contract])
tenure[rng.choice(M, 11, replace=False)] = 0
phone = rng.choice(["Yes", "No"], M, p=[.9, .1])
multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], M, p=[.42, .58]))
inet = rng.choice(["Fiber optic", "DSL", "No"], M, p=[.44, .34, .22])
def addon(p):
    return np.where(inet == "No", "No internet service", np.where(rng.random(M) < p, "Yes", "No"))
osec, obak, dprot, tsup, stv, smov = addon(.29), addon(.34), addon(.34), addon(.29), addon(.38), addon(.39)
paperless = rng.choice(["Yes", "No"], M, p=[.59, .41])
pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], M, p=[.34, .23, .22, .21])
monthly = (20 + 25 * (phone == "Yes") * 0.8 + 5 * (multi == "Yes") + 25 * (inet == "DSL") + 50 * (inet == "Fiber optic") - 25 * (inet == "No") * 0
           + 5 * sum((x == "Yes") for x in [osec, obak, dprot, tsup]) + 9 * sum((x == "Yes") for x in [stv, smov]) + rng.normal(0, 2.5, M))
monthly = np.clip(monthly, 18.25, 118.75).round(2)
total = (monthly * tenure * rng.normal(1, .03, M)).round(2)
z = (-1.1 + 1.5 * (contract == "Month-to-month") - 1.2 * (contract == "Two year") - 0.035 * tenure
     + 0.8 * (inet == "Fiber optic") - 0.5 * (tsup == "Yes") - 0.4 * (osec == "Yes") + 0.5 * (pay == "Electronic check")
     + 0.3 * senior + 0.25 * (paperless == "Yes") - 0.2 * (dep == "Yes") + rng.normal(0, .5, M))
churn = np.where(rng.random(M) < sig(z), "Yes", "No")
ci = pd.DataFrame(dict(customerID=ids, gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dep, tenure=tenure, Churn=churn))
total_s = pd.Series(total).astype(str)
total_s[tenure == 0] = " "             # 실제 데이터처럼 공백 문자열
si = pd.DataFrame(dict(customerID=ids, PhoneService=phone, MultipleLines=multi, InternetService=inet, OnlineSecurity=osec,
                       OnlineBackup=obak, DeviceProtection=dprot, TechSupport=tsup, StreamingTV=stv, StreamingMovies=smov,
                       Contract=contract, PaperlessBilling=paperless, PaymentMethod=pay, MonthlyCharges=monthly, TotalCharges=total_s))
# 전처리 연습거리: 결측, 중복
ci.loc[rng.choice(M, 30, replace=False), "Dependents"] = np.nan
si.loc[rng.choice(M, 25, replace=False), "MonthlyCharges"] = np.nan
si.loc[rng.choice(M, 18, replace=False), "PaymentMethod"] = np.nan
ci = pd.concat([ci, ci.sample(22, random_state=1)]).reset_index(drop=True)   # 중복행 22개
ci.to_csv(f"{OUT}/cust_info.csv", index=False)
si.sample(frac=1, random_state=3).to_csv(f"{OUT}/service_info.csv", index=False)

# ---------------------------------------------------------------- air_data_01~11.csv (서울 25개 측정소, 2025년 1~11월 시간별)
stations = [("중구", 111121, "서울 중구 덕수궁길 15"), ("종로구", 111123, "서울 종로구 종로35가길 19"), ("용산구", 111131, "서울 용산구 한남대로 136"),
            ("광진구", 111141, "서울 광진구 자양로 117"), ("성동구", 111142, "서울 성동구 뚝섬로3길 18"), ("중랑구", 111151, "서울 중랑구 용마산로 369"),
            ("동대문구", 111152, "서울 동대문구 천호대로13길 43"), ("성북구", 111161, "서울 성북구 삼양로2길 70"), ("도봉구", 111171, "서울 도봉구 시루봉로2길 34"),
            ("은평구", 111181, "서울 은평구 진흥로 215"), ("서대문구", 111191, "서울 서대문구 세검정로4길 32"), ("마포구", 111201, "서울 마포구 포은로6길 10"),
            ("강서구", 111212, "서울 강서구 강서로45다길 71"), ("구로구", 111221, "서울 구로구 가마산로 27길 45"), ("영등포구", 111231, "서울 영등포구 당산로 123"),
            ("동작구", 111241, "서울 동작구 사당로16아길 6"), ("관악구", 111251, "서울 관악구 신림동길 14"), ("강남구", 111261, "서울 강남구 학동로 426"),
            ("서초구", 111262, "서울 서초구 신반포로15길 16"), ("송파구", 111273, "서울 송파구 백제고분로 236"), ("강동구", 111274, "서울 강동구 구천면로 42길 59"),
            ("금천구", 111281, "서울 금천구 금하로21길 20"), ("강북구", 111291, "서울 강북구 삼양로139길 49"), ("양천구", 111301, "서울 양천구 중앙로52길 56"),
            ("노원구", 111311, "서울 노원구 상계로 118")]
season_pm = {1: 48, 2: 52, 3: 62, 4: 55, 5: 45, 6: 33, 7: 25, 8: 22, 9: 24, 10: 32, 11: 42}
season_o3 = {1: .015, 2: .02, 3: .028, 4: .036, 5: .042, 6: .045, 7: .035, 8: .033, 9: .03, 10: .024, 11: .016}
for m in range(1, 12):
    hours = pd.date_range(f"2025-{m:02d}-01 01:00", periods=pd.Period(f"2025-{m:02d}").days_in_month * 24, freq="h")
    H = len(hours)
    # 도시 공통 기상요인(AR 과정) + 황사 이벤트
    reg = np.zeros(H); e = rng.normal(0, 1, H)
    for t in range(1, H): reg[t] = .97 * reg[t - 1] + .25 * e[t]
    dust = np.zeros(H)
    if m in (3, 4, 5):
        for _ in range(rng.integers(1, 3)):
            s = rng.integers(0, H - 48); dust[s:s + rng.integers(18, 48)] += rng.uniform(80, 220)
    hod = hours.hour.values
    rows = []
    for name, code, addr in stations:
        loc = rng.normal(0, .08)
        traffic = 1 + .25 * np.exp(-((hod - 8) ** 2) / 6) + .3 * np.exp(-((hod - 19) ** 2) / 8)
        no2 = np.clip(.022 * traffic * np.exp(.35 * reg + loc) + rng.normal(0, .004, H), .002, None)
        co = np.clip(.35 + 8 * no2 + rng.normal(0, .05, H), .1, None)
        so2 = np.clip(.0028 + .03 * no2 + rng.normal(0, .0005, H), .001, None)
        pm10 = np.clip(season_pm[m] * np.exp(.45 * reg + loc) * (.85 + .15 * traffic) + dust * rng.uniform(.8, 1.1) + rng.normal(0, 6, H), 3, None)
        pm25 = np.clip(pm10 * (0.55 - 0.0012 * dust) + rng.normal(0, 3, H), 1, None)
        sunlight = np.clip(np.sin((hod - 6) / 14 * np.pi), 0, None)
        o3 = np.clip(season_o3[m] * (.4 + 1.2 * sunlight) - .25 * (no2 - .02) + rng.normal(0, .004, H), .001, None)
        df = pd.DataFrame({"지역": "서울 " + name, "망": "도시대기", "측정소코드": code, "측정소명": name,
                           "측정일시": np.where(hours.hour == 0, (hours - pd.Timedelta("1h")).strftime("%Y%m%d") + "24", hours.strftime("%Y%m%d%H")).astype(np.int64),
                           "SO2": so2.round(4), "CO": co.round(1), "O3": o3.round(4), "NO2": no2.round(4),
                           "PM10": pm10.round(0), "PM25": pm25.round(0), "주소": addr})
        # 결측(점검)과 이상치
        miss = rng.random(H) < .02
        df.loc[miss, ["SO2", "CO", "O3", "NO2"]] = np.nan
        df.loc[rng.random(H) < .015, "PM10"] = np.nan
        df.loc[rng.random(H) < .015, "PM25"] = np.nan
        rows.append(df)
    air = pd.concat(rows).sort_values(["측정일시", "측정소코드"]).reset_index(drop=True)
    out_idx = rng.choice(len(air), 6, replace=False)
    air.loc[out_idx, "PM10"] = rng.choice([985, 990, 999], 6)
    air.to_csv(f"{OUT}/air_data_{m:02d}.csv", index=False, encoding="cp949")
print("done")
