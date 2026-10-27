# AICE Associate 학습 노트북

번호 순서대로 열어서 위에서부터 실행하며 공부하면 됩니다.
모든 노트북은 미리 실행해 둔 상태라 **결과(표·그래프)가 저장되어 있어**, 실행 환경이 없어도 읽으며 공부할 수 있습니다.

## 학습 순서

| 단계 | 파일 | 내용 | 사용 데이터 |
|---|---|---|---|
| 1 | `01.Python Basic_01. Python 기초` | print, 변수, 연산자, 조건문, 반복문, 함수, import, 에러 읽기 | – |
| 2 | `02.Python Basic_02. 파이썬 자료형 및 활용` | 리스트, 튜플, 딕셔너리, 집합, 컴프리헨션, numpy 기초 | – |
| 3 | `03.Pandas 이해 및 활용_01. DataFrame 살펴보기` | read_csv, info, describe, value_counts, loc/iloc, 조건 필터링, 정렬 | cust_data, titanic |
| 4 | `04.Pandas 이해 및 활용_02. DataFrame 변형하기` | 컬럼 추가/삭제, map/replace/apply, groupby, pivot_table, cut/qcut | cust_data, titanic |
| 5 | `05.Pandas 이해 및 활용_03. DataFrame 병합하기` | merge(how), concat, join | cust_info, service_info, cust_data, cust_data1 |
| 6 | `06.데이터 전처리 - 데이터탐색과 결측치처리` | 결측치 탐색, dropna, fillna, SimpleImputer | sc_cust_info_txn_v1.5 |
| 7 | `07.데이터 전처리 - 이상치처리와 Feature Engineering` | IQR 이상치, 인코딩, train_test_split, 스케일링 | sc_cust_info_txn_v1.5, cust_data1 |
| 8 | `08.데이터 시각화_01. matplotlib 활용` | plot, subplots, **학습 곡선 그리기** | titanic |
| 9 | `09.데이터 시각화_02. seaborn 활용` | countplot, histplot, boxplot, jointplot, heatmap | titanic, cust_data |
| 10 | `10.머신러닝 모델 템플릿` | 분류·회귀 모델, 평가지표, GridSearchCV, 변수중요도 | titanic, cust_data |
| 11 | `11.머신러닝 모델링_퀴즈` → `_정답` | 시험 형식 13문항 (해지 예측) | cust_data1 |
| 12 | `12.딥러닝 모델 템플릿` | Sequential, Dense/Dropout, EarlyStopping, 이진/다중/회귀 | titanic, cust_data |
| 13 | `13.딥러닝 모델링_퀴즈` → `_정답` | 시험 형식 9문항 (해지 예측 딥러닝) | cust_data1 |
| 14 | `14.사례실습_통신사이탈예측_01.전처리_퀴즈` → `_정답` | 두 테이블 병합 → 정제 → EDA → 인코딩 → 저장 | cust_info, service_info |
| 15 | `15.사례실습_통신사이탈예측_02.머신러닝_딥러닝_퀴즈` → `_정답` | 여러 모델 비교, F1/Recall, 딥러닝, 임계값 조정 | churn_preprocessed (14번 결과) |
| 16 | `16.사례실습_미세먼지수치예측_01.데이터전처리` | 월별 파일 합치기, cp949, 24시 처리, 시계열 보간 | air_data_01~11 |
| 17 | `16.사례실습_미세먼지수치예측_02.모델링` | 회귀 모델(ML·DL), MAE/RMSE/R², 시간 기준 검증 | air_preprocessed (16-① 결과) |
| 마무리 | `source/` 폴더의 **샘플문항(분류·회귀)** | 90분 시간을 재고 실전처럼 풀기 | wine_quality_data, signal_data |

### 공부 방법
- **01~10, 12번**: 설명 → 예제 실행 → 맨 아래 **✏️ 연습문제** 풀기 → **✅ 정답 및 해설** 확인
- **11, 13, 14, 15번**: `_퀴즈` 파일로 먼저 풀고, 막히거나 다 풀면 `_정답` 파일과 비교
- **16번**: 각 문항 바로 아래 빈 셀에 먼저 작성해 보고, 그 다음 `[정답]` 셀 확인
- 노트북은 **위에서부터 순서대로** 실행해야 합니다. (앞 셀에서 만든 변수를 뒤에서 사용)

## 실행 환경 준비

AICE 시험 환경(Jupyter)과 비슷하게 맞추려면 아래 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

- 노트북은 csv 파일과 **같은 폴더**에서 실행해야 합니다. (`pd.read_csv('cust_data.csv')`)
- 한글 그래프 폰트는 노트북의 설정 셀이 OS(Windows/Mac/Linux)에 맞춰 자동으로 잡아 줍니다.
- `xgboost`, `lightgbm`은 선택입니다. 없으면 해당 셀이 안내 문구만 출력하고 넘어갑니다.

## 데이터 안내

| 파일 | 설명 |
|---|---|
| `titanic.csv` | 공개 타이타닉 데이터 (891명) |
| `cust_data.csv` / `cust_data1.csv` | 통신사 고객 해지 데이터. `cust_data1`은 결측치·이상치가 섞인 퀴즈용 |
| `sc_cust_info_txn_v1.5.csv` | 통신사 고객 데이터 (결측치·이상치 처리 연습용) |
| `cust_info.csv` / `service_info.csv` | 고객 기본정보 / 서비스 가입정보 (customerID로 병합, 이탈 예측) |
| `AICE Associate 실습 안내 가이드.pdf` | 시험 실습 환경 안내 (source 폴더의 원본을 복사) |
| `air_data_01~11.csv` | 서울 25개 측정소 2025년 1~11월 시간별 대기오염 측정값 (cp949 인코딩) |

> titanic 외의 데이터는 **학습용으로 만든 가상 데이터**입니다. 실제 시험 데이터와 컬럼·값은 다르지만, 시험에서 요구하는 전처리 상황(결측, 이상치, 공백 문자열, 중복, 인코딩, 24시 표기 등)을 일부러 넣어 두었습니다.
> 데이터는 `_build/gen_data.py`로 언제든 똑같이 다시 만들 수 있습니다.

## 시험 핵심 요약 (AICE Associate)

| 영역 | 자주 나오는 문항 |
|---|---|
| 데이터 분석 | 라이브러리 별칭 import, `read_csv`, seaborn `countplot`/`histplot`/`boxplot`/`jointplot`, `subplots` + `set_title` |
| 데이터 전처리 | IQR 이상치 삭제(`quantile`, `drop(...index)`), `isnull().sum()`, `dropna`, `fillna`, `get_dummies(data=, columns=)`, `train_test_split(test_size, random_state, stratify)`, 스케일러 `fit_transform` / `transform` |
| AI 모델링 | DecisionTree/RandomForest (`Classifier`/`Regressor`), `GridSearchCV(param_grid, cv, scoring, n_jobs)`, `best_estimator_`, `feature_importances_` + `sort_values`, `accuracy_score`/`mean_absolute_error`, keras `Sequential`/`Dense`/`Dropout`, `EarlyStopping(patience)`, `compile`, `fit(validation_data)`, `history.history` 학습 곡선 |

**답안 작성 형식**: 빈칸·값 문항은 `답안3 = 'subplots'`, `답안9 = 237` 처럼 지정된 변수에 저장합니다.
