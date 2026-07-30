"""
[분석 3] 모델링

지표 선택 근거
  Accuracy를 주지표로 쓰지 않음. 손상을 놓치는 것(2종 오류)이 정상 기업을 잘못
  지목하는 것(1종 오류)보다 훨씬 큰 손실이기 때문. 감사기준도 감사위험을
  '중요한 왜곡표시가 있는 재무제표에 적정의견을 주는 위험'으로만 정의함.
  → Recall과 Average Precision(PR-AUC)을 기준으로 봄.

분할 방식 두 가지를 모두 봄
  (A) 연도 분할 — 실제 운영 상황(과거로 학습해 미래를 예측)과 같음
  (B) 기업 분할 — 같은 기업이 학습·검증에 함께 들어가면 성능이 부풀 수 있음.
      특히 '당기 손상 인식 여부'가 강한 변수라 기업 단위 누수 위험이 큼

모델
  로지스틱 회귀를 주력으로 둠. 감사 근거로 쓰려면 "왜 이 기업이 위험한가"를
  계수로 설명할 수 있어야 하기 때문. 트리 모델은 비선형 탐색용 대조군.
  표본이 322건으로 작아 부스팅은 과적합 위험이 큼.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FEATS = ["영업이익률", "당기손상여부", "무형_매출", "무형_영업CF", "발생액괴리",
         "부채비율", "자산로그", "증가괴리_1y"]
LABEL = "y_2y"


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 20:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def prep(df):
    X = df[FEATS].copy()
    for c in FEATS:
        X[c] = winsorize(X[c])
    # 결측은 중앙값 대체 + 결측여부 플래그 (결측 자체가 정보일 수 있음)
    for c in FEATS:
        if X[c].isna().any():
            X[f"{c}_결측"] = X[c].isna().astype(int)
            X[c] = X[c].fillna(X[c].median())
    return X


def evaluate(name, y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "모델": name,
        "Accuracy": accuracy_score(y, yhat),
        "Recall": recall_score(y, yhat, zero_division=0),
        "Precision": precision_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat, zero_division=0),
        "AP": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
    }


def run_split(Xtr, ytr, Xte, yte, tag):
    rows = []
    base_rate = ytr.mean()

    # 기준선 1: 전부 양성으로 찍기
    rows.append(evaluate(f"[{tag}] 전부양성", yte, np.ones(len(yte))))
    # 기준선 2: 당기 손상 인식 여부만 사용
    rows.append(evaluate(f"[{tag}] 당기손상만", yte, Xte["당기손상여부"].values.astype(float)))

    lr = Pipeline([("sc", StandardScaler()),
                   ("m", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
    lr.fit(Xtr, ytr)
    rows.append(evaluate(f"[{tag}] 로지스틱", yte, lr.predict_proba(Xte)[:, 1]))

    rf = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=8,
                                class_weight="balanced", random_state=42)
    rf.fit(Xtr, ytr)
    rows.append(evaluate(f"[{tag}] 랜덤포레스트", yte, rf.predict_proba(Xte)[:, 1]))

    try:
        from xgboost import XGBClassifier
        pos = max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1e-6)
        xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=pos, eval_metric="aucpr", random_state=42)
        xgb.fit(Xtr, ytr)
        rows.append(evaluate(f"[{tag}] XGBoost", yte, xgb.predict_proba(Xte)[:, 1]))
    except ImportError:
        pass

    return pd.DataFrame(rows), lr, base_rate


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "features_v2.csv", dtype={"corp_code": str})
    d = df.dropna(subset=[LABEL]).reset_index(drop=True)
    X, y = prep(d), d[LABEL].astype(int)
    print(f"표본 {len(d)}건 / {d.corp_code.nunique()}개사 / 양성 {y.mean():.1%}")
    print(f"변수 {list(X.columns)}\n")

    # ---------- (A) 연도 분할 ----------
    tr = d["사업연도"] <= 2021
    te = ~tr
    print(f"[A] 연도 분할 — 학습 {tr.sum()}건(~2021) / 검증 {te.sum()}건(2022~)")
    print(f"    학습 양성 {y[tr].mean():.1%} / 검증 양성 {y[te].mean():.1%}")
    resA, lrA, _ = run_split(X[tr], y[tr], X[te], y[te], "연도")

    # ---------- (B) 기업 분할 ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    itr, ite = next(gss.split(X, y, groups=d["corp_code"]))
    print(f"\n[B] 기업 분할 — 학습 {len(itr)}건 / 검증 {len(ite)}건 (기업 겹침 없음)")
    print(f"    학습 양성 {y.iloc[itr].mean():.1%} / 검증 양성 {y.iloc[ite].mean():.1%}")
    resB, lrB, _ = run_split(X.iloc[itr], y.iloc[itr], X.iloc[ite], y.iloc[ite], "기업")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n" + "=" * 96)
    print("### 성능 비교")
    print("=" * 96)
    print(pd.concat([resA, resB]).to_string(index=False))

    print("\n" + "=" * 96)
    print("### 로지스틱 회귀 계수 (연도 분할 학습 기준, 표준화 후)")
    print("=" * 96)
    coef = pd.DataFrame({
        "변수": X.columns,
        "계수": lrA.named_steps["m"].coef_[0],
    })
    coef["오즈비"] = np.exp(coef["계수"])
    coef["영향력"] = coef["계수"].abs()
    print(coef.sort_values("영향력", ascending=False).drop(columns="영향력").to_string(index=False))

    print("\n" + "=" * 96)
    print("### 보조 확인 — t+1 레이블")
    print("=" * 96)
    d1 = df.dropna(subset=["y_t1"]).reset_index(drop=True)
    X1, y1 = prep(d1), d1["y_t1"].astype(int)
    tr1 = d1["사업연도"] <= 2021
    r1, _, _ = run_split(X1[tr1], y1[tr1], X1[~tr1], y1[~tr1], "t+1·연도")
    print(r1.to_string(index=False))
