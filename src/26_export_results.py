"""
[결과 내보내기] 대시보드가 읽을 분석 결과를 파일로 고정함

Streamlit이 실행 시점에 통계·모델을 다시 돌리지 않도록 결과를 미리 저장함.
배포 환경에서 재현성이 흔들리지 않게 하고 앱 로딩도 빠르게 하기 위함.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
f2 = __import__("24_features_v2")
md = __import__("25_model")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT = BASE_DIR / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def export_tests(df):
    frames = []
    for y, nm in (("y_t1", "t+1년"), ("y_t2", "t+2년"), ("y_2y", "2년 내")):
        d = df.dropna(subset=[y])
        t = f2.run_tests(d, y)
        t.insert(0, "시계", nm)
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "tests.csv", index=False, encoding="utf-8-sig")
    return out


def export_models(df):
    from sklearn.model_selection import GroupShuffleSplit

    d = df.dropna(subset=["y_2y"]).reset_index(drop=True)
    X, y = md.prep(d), d["y_2y"].astype(int)

    tr = d["사업연도"] <= 2021
    resA, lrA, _ = md.run_split(X[tr], y[tr], X[~tr], y[~tr], "연도분할")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    itr, ite = next(gss.split(X, y, groups=d["corp_code"]))
    resB, _, _ = md.run_split(X.iloc[itr], y.iloc[itr], X.iloc[ite], y.iloc[ite], "기업분할")

    d1 = df.dropna(subset=["y_t1"]).reset_index(drop=True)
    X1, y1 = md.prep(d1), d1["y_t1"].astype(int)
    tr1 = d1["사업연도"] <= 2021
    resC, _, _ = md.run_split(X1[tr1], y1[tr1], X1[~tr1], y1[~tr1], "t+1·연도분할")

    res = pd.concat([resA, resB, resC], ignore_index=True)
    res.to_csv(OUT / "models.csv", index=False, encoding="utf-8-sig")

    coef = pd.DataFrame({"변수": X.columns, "계수": lrA.named_steps["m"].coef_[0]})
    coef["오즈비"] = np.exp(coef["계수"])
    coef = coef.reindex(coef["계수"].abs().sort_values(ascending=False).index)
    coef.to_csv(OUT / "coef.csv", index=False, encoding="utf-8-sig")
    return res


def export_pipeline():
    imp = pd.read_csv(PROCESSED_DIR / "impairment.csv", dtype={"corp_code": str})
    imp2 = pd.read_csv(PROCESSED_DIR / "impairment_v2.csv", dtype={"corp_code": str})
    panel = pd.read_csv(PROCESSED_DIR / "panel.csv", dtype={"corp_code": str})
    conf = panel[(panel["상태"] == "사용가능") & (panel["신뢰도"] == "확정")]

    stats = {
        "상장사_전체": 3981,
        "업종필터_통과": 147,
        "사업보고서_보유": 125,
        "모집단_확정": int(panel["corp_code"].nunique()),
        "수집_firmyear": int(len(panel)),
        "대사성공_1차": int((imp["대사"] == "성공").sum()),
        "대사성공_보완후": int((imp2["대사"] == "성공").sum()),
        "사용가능": int((panel["상태"] == "사용가능").sum()),
        "확정표본": int(len(conf)),
        "확정기업수": int(conf["corp_code"].nunique()),
    }
    # 파서 개선 이력 (시범 10개사 50 firm-year 기준)
    stats["파서개선"] = [
        {"단계": "재무제표 본문 API", "성공률": 0.0, "설명": "대형사가 조정항목을 주석으로 빼 레이블 확보 불가"},
        {"단계": "키워드 검출 + 기말 대사", "성공률": 60.0, "설명": "표를 키워드로 고르고 기말잔액만 맞춤"},
        {"단계": "느슨한 검출 + 2점 대사", "성공률": 80.0, "설명": "기초=전기말까지 함께 대사"},
        {"단계": "표 헤더 인식 보정", "성공률": 90.0, "설명": "'(단위: 천원)' 행을 헤더로 오인해 금액이 2배가 되던 문제 해결"},
        {"단계": "정정보고서 폴백", "성공률": 96.0, "설명": "정정본에 주석이 없는 경우 원본으로 재시도"},
    ]
    (OUT / "pipeline.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "features_v2.csv", dtype={"corp_code": str})
    t = export_tests(df)
    m = export_models(df)
    s = export_pipeline()
    print(f"tests.csv  {len(t)}행")
    print(f"models.csv {len(m)}행")
    print(f"pipeline.json {list(s)[:6]} …")
    print(f"저장 → {OUT}")
