"""
[분석 1] 파생변수 설계 + 가설검정

설계 원칙: 단순 재무비율을 늘어놓는 대신, 손상이 왜 생기는지에 대한
회계적 가설을 담은 변수를 만들고 그 가설을 검정으로 확인한 뒤에만 모델에 넣음.

핵심 가설
  H1 자산은 쌓이는데 수익이 안 따라오면 손상 위험이 크다
     → 무형자산증가율 − 매출성장률 괴리
  H2 상각을 늦추면(유효상각률 하락) 비용 이연이고, 뒤에 손상으로 터진다
     → 유효상각률과 그 변화
  H3 이익의 질이 나쁘면(발생액 괴리) 손상 인식이 임박했다
     → (당기순이익 − 영업활동현금흐름) / 자산총계
  H4 무형자산 비중이 클수록 손상 시 충격이 크고 유인도 크다

레이블: t+1년 손상액/무형자산 ≥ 5% (대체 레이블: t+1년 손상액/자산총계 ≥ 0.5%)
설명변수는 t년 값만 사용해 시점 누수를 막음.

연결/별도 전환 구간은 증가율 계산에서 끊음. 무형자산이 15억→185억으로 뛴 것이
실제 증가가 아니라 재무제표 기준 변경인 경우가 22개사에서 발생하기 때문임.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

LABEL_MAIN = 0.05    # 손상액/무형자산
LABEL_ALT = 0.005    # 손상액/자산총계


def safe_div(a, b):
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def build():
    p = pd.read_csv(PROCESSED_DIR / "panel.csv", dtype={"corp_code": str})
    df = p[(p["상태"] == "사용가능") & (p["신뢰도"] == "확정")].copy()
    df = df.sort_values(["corp_code", "사업연도"]).reset_index(drop=True)

    g = df.groupby("corp_code")
    df["전기무형"] = g["무형자산"].shift(1)
    df["전기매출"] = g["매출액"].shift(1)
    df["전기영업이익"] = g["영업이익"].shift(1)
    df["전기자산"] = g["자산총계"].shift(1)
    df["전기연도"] = g["사업연도"].shift(1)

    # 직전 관측이 바로 전 해가 아니면 증가율을 만들지 않음
    contiguous = (df["사업연도"] - df["전기연도"]) == 1
    # 재무제표 기준이 바뀐 해도 증가율을 만들지 않음
    valid_growth = contiguous & (~df["기준전환"].astype(bool))

    df["무형증가율"] = np.where(valid_growth, safe_div(df["무형자산"] - df["전기무형"], df["전기무형"]), np.nan)
    df["매출성장률"] = np.where(valid_growth, safe_div(df["매출액"] - df["전기매출"], df["전기매출"]), np.nan)
    df["증가괴리"] = df["무형증가율"] - df["매출성장률"]          # H1

    avg_intan = (df["무형자산"] + df["전기무형"]) / 2
    df["유효상각률"] = np.where(valid_growth, safe_div(df["상각액"], avg_intan), np.nan)  # H2
    df["유효상각률변화"] = df.groupby("corp_code")["유효상각률"].diff().where(valid_growth)

    df["무형비중"] = safe_div(df["무형자산"], df["자산총계"])      # H4
    df["무형_매출배수"] = safe_div(df["무형자산"], df["매출액"])
    df["발생액괴리"] = safe_div(df["당기순이익"] - df["영업활동현금흐름"], df["자산총계"])  # H3
    df["영업이익률"] = safe_div(df["영업이익"], df["매출액"])
    df["영업이익률변화"] = df.groupby("corp_code")["영업이익률"].diff().where(valid_growth)
    df["부채비율"] = safe_div(df["부채총계"], df["자본총계"])
    df["당기손상비율"] = safe_div(df["손상액"], df["무형자산"])
    df["당기손상여부"] = (df["당기손상비율"] >= LABEL_MAIN).astype(int)
    df["자산로그"] = np.log(df["자산총계"].where(df["자산총계"] > 0))

    # --- 레이블: 다음 해 값 ---
    df["차기연도"] = g["사업연도"].shift(-1)
    df["차기손상액"] = g["손상액"].shift(-1)
    df["차기무형"] = g["무형자산"].shift(-1)
    df["차기자산"] = g["자산총계"].shift(-1)
    nxt = (df["차기연도"] - df["사업연도"]) == 1

    df["y_main"] = np.where(nxt, (safe_div(df["차기손상액"], df["차기무형"]) >= LABEL_MAIN).astype(float), np.nan)
    df["y_alt"] = np.where(nxt, (safe_div(df["차기손상액"], df["차기자산"]) >= LABEL_ALT).astype(float), np.nan)
    return df


FEATURES = ["증가괴리", "무형증가율", "매출성장률", "유효상각률", "유효상각률변화",
            "무형비중", "무형_매출배수", "발생액괴리", "영업이익률", "영업이익률변화",
            "부채비율", "당기손상여부", "자산로그"]


def run_tests(d, ycol):
    rows = []
    pos, neg = d[d[ycol] == 1], d[d[ycol] == 0]
    for f in FEATURES:
        a, b = pos[f].dropna(), neg[f].dropna()
        if len(a) < 10 or len(b) < 10:
            rows.append({"변수": f, "양성n": len(a), "음성n": len(b), "비고": "표본부족"})
            continue
        lev_p = stats.levene(a, b, center="median").pvalue
        equal_var = lev_p >= 0.05
        t, tp = stats.ttest_ind(a, b, equal_var=equal_var)
        u, up = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "변수": f, "양성n": len(a), "음성n": len(b),
            "양성중앙값": a.median(), "음성중앙값": b.median(),
            "등분산p": lev_p, "검정": "t" if equal_var else "Welch",
            "p값": tp, "MWU_p": up,
            "유의": "***" if min(tp, up) < 0.01 else ("**" if min(tp, up) < 0.05 else
                    ("*" if min(tp, up) < 0.1 else "")),
        })
    return pd.DataFrame(rows)


def vif_table(d):
    from itertools import combinations
    X = d[FEATURES].dropna()
    if len(X) < 30:
        return None
    corr = X.corr()
    inv = np.linalg.pinv(corr.values)
    return pd.DataFrame({"변수": corr.columns, "VIF": np.diag(inv)}).sort_values("VIF", ascending=False)


if __name__ == "__main__":
    df = build()
    df.to_csv(PROCESSED_DIR / "features.csv", index=False, encoding="utf-8-sig")

    lab = df.dropna(subset=["y_main"])
    print(f"확정 표본 {len(df)} firm-year / {df.corp_code.nunique()}개사")
    print(f"t→t+1 학습쌍 성립: {len(lab)}건 / {lab.corp_code.nunique()}개사")
    print(f"  주 레이블 양성 {int(lab['y_main'].sum())}건 ({lab['y_main'].mean():.1%})")
    print(f"  대체 레이블 양성 {int(lab['y_alt'].sum())}건 ({lab['y_alt'].mean():.1%})")
    print(f"  두 레이블 일치율 {(lab['y_main'] == lab['y_alt']).mean():.1%}")

    print("\n증가율 계산에서 끊긴 건:")
    print(f"  연도 불연속 또는 기준전환으로 무형증가율 결측: {df['무형증가율'].isna().sum()}건")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    for ycol, name in (("y_main", "주 레이블 (차기 손상/무형 ≥5%)"),
                       ("y_alt", "대체 레이블 (차기 손상/자산 ≥0.5%)")):
        d = df.dropna(subset=[ycol])
        print(f"\n{'=' * 100}\n### 가설검정 — {name}\n{'=' * 100}")
        print(run_tests(d, ycol).to_string(index=False))

    print(f"\n{'=' * 100}\n### 다중공선성 (VIF)\n{'=' * 100}")
    v = vif_table(df.dropna(subset=["y_main"]))
    if v is not None:
        print(v.to_string(index=False))

    print(f"\n저장 → {PROCESSED_DIR / 'features.csv'}")
