"""
[분석 2] 파생변수 2차 설계

1차에서 핵심 가설(무형자산증가율 − 매출성장률 괴리)이 기각됨(p=0.71).
그러나 수준 변수인 무형/매출 배수는 약하게나마 신호가 있었음(MWU p=0.081).
증가율은 1년 노이즈·기준전환·M&A에 취약한 반면 수준은 안정적이라는 해석이 가능함.

2차 설계에서 바꾼 것
  (1) 수준 변수 강화 — 무형/매출, 무형/영업현금흐름, 무형/자산
  (2) 업종 상대값 — 드라마 제작사와 음반사는 정상 상각률·정상 무형비중 자체가 다름.
      절대값으로 비교하면 업종 차이에 신호가 묻히므로 업종·연도 중앙값 대비로 봄
  (3) 다년 누적 — 1년 변화 대신 3년 누적 괴리 (콘텐츠는 투입과 회수에 시차가 있음)
  (4) 시차 확대 — 손상 인식이 지연될 수 있으므로 t+1뿐 아니라 t+2, 그리고 '2년 내'도 봄
  (5) 이상치 — 무형자산이 작은 기업에서 비율이 폭주하므로 1%/99% 윈저화
  (6) 완전공선 정리 — 증가괴리 = 무형증가율 − 매출성장률이라 셋을 함께 쓰면 특이행렬이 됨.
      1차 VIF가 1 미만으로 나온 원인이었음. 괴리만 남기고 구성요소는 뺌
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

LABEL_MAIN = 0.05


def safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def winsorize(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 20:
        return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def build():
    p = pd.read_csv(PROCESSED_DIR / "panel.csv", dtype={"corp_code": str})
    uni = pd.read_csv(PROCESSED_DIR / "universe_final.csv", dtype={"corp_code": str})
    df = p[(p["상태"] == "사용가능") & (p["신뢰도"] == "확정")].copy()
    df = df.merge(uni[["corp_code", "업종"]], on="corp_code", how="left")
    df = df.sort_values(["corp_code", "사업연도"]).reset_index(drop=True)

    g = df.groupby("corp_code")
    for k, col in (("전기무형", "무형자산"), ("전기매출", "매출액"), ("전기연도", "사업연도")):
        df[k] = g[col].shift(1)
    for k, col in (("3전무형", "무형자산"), ("3전매출", "매출액"), ("3전연도", "사업연도")):
        df[k] = g[col].shift(3)

    df["전환누적"] = g["기준전환"].transform(lambda s: s.astype(bool).cumsum())
    df["전환누적_3전"] = g["전환누적"].shift(3)

    ok1 = ((df["사업연도"] - df["전기연도"]) == 1) & (~df["기준전환"].astype(bool))
    # 3년 창 안에서 재무제표 기준이 한 번이라도 바뀌었으면 누적 증가율을 만들지 않음
    ok3 = ((df["사업연도"] - df["3전연도"]) == 3) & (df["전환누적"] == df["전환누적_3전"])

    # --- 수준 변수 ---
    df["무형_매출"] = safe_div(df["무형자산"], df["매출액"])
    df["무형_자산"] = safe_div(df["무형자산"], df["자산총계"])
    df["무형_영업CF"] = safe_div(df["무형자산"], df["영업활동현금흐름"].where(df["영업활동현금흐름"] > 0))
    df["유효상각률"] = np.where(ok1, safe_div(df["상각액"], (df["무형자산"] + df["전기무형"]) / 2), np.nan)
    df["영업이익률"] = safe_div(df["영업이익"], df["매출액"])
    df["발생액괴리"] = safe_div(df["당기순이익"] - df["영업활동현금흐름"], df["자산총계"])
    df["부채비율"] = safe_div(df["부채총계"], df["자본총계"])
    df["자산로그"] = np.log(df["자산총계"].where(df["자산총계"] > 0))
    df["당기손상여부"] = (safe_div(df["손상액"], df["무형자산"]) >= LABEL_MAIN).astype(int)

    # --- 1년 / 3년 괴리 ---
    df["증가괴리_1y"] = np.where(
        ok1, safe_div(df["무형자산"] - df["전기무형"], df["전기무형"])
        - safe_div(df["매출액"] - df["전기매출"], df["전기매출"]), np.nan)
    df["증가괴리_3y"] = np.where(
        ok3, safe_div(df["무형자산"] - df["3전무형"], df["3전무형"])
        - safe_div(df["매출액"] - df["3전매출"], df["3전매출"]), np.nan)

    # --- 업종·연도 상대값 (표본이 적은 조합은 연도 중앙값으로 대체) ---
    for col in ["무형_매출", "무형_자산", "유효상각률", "영업이익률"]:
        grp = df.groupby(["업종", "사업연도"])[col].transform("median")
        cnt = df.groupby(["업종", "사업연도"])[col].transform("count")
        yr = df.groupby("사업연도")[col].transform("median")
        base = np.where(cnt >= 5, grp, yr)
        df[f"{col}_상대"] = df[col] - base

    # --- 레이블: t+1, t+2, 2년 내 ---
    for h in (1, 2):
        df[f"연도_{h}"] = g["사업연도"].shift(-h)
        df[f"손상_{h}"] = g["손상액"].shift(-h)
        df[f"무형_{h}"] = g["무형자산"].shift(-h)
        valid = (df[f"연도_{h}"] - df["사업연도"]) == h
        df[f"y_t{h}"] = np.where(valid, (safe_div(df[f"손상_{h}"], df[f"무형_{h}"]) >= LABEL_MAIN).astype(float), np.nan)
    df["y_2y"] = np.where(df[["y_t1", "y_t2"]].notna().any(axis=1),
                          df[["y_t1", "y_t2"]].max(axis=1), np.nan)
    return df


FEATURES = ["증가괴리_1y", "증가괴리_3y", "무형_매출", "무형_자산", "무형_영업CF",
            "유효상각률", "영업이익률", "발생액괴리", "부채비율", "자산로그", "당기손상여부",
            "무형_매출_상대", "무형_자산_상대", "유효상각률_상대", "영업이익률_상대"]


def run_tests(d, ycol):
    rows = []
    for f in FEATURES:
        x = winsorize(d[f])
        a = x[d[ycol] == 1].dropna()
        b = x[d[ycol] == 0].dropna()
        if len(a) < 10 or len(b) < 10:
            rows.append({"변수": f, "양성n": len(a), "음성n": len(b), "비고": "표본부족"})
            continue
        lev = stats.levene(a, b, center="median").pvalue
        eq = lev >= 0.05
        tp = stats.ttest_ind(a, b, equal_var=eq).pvalue
        up = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        best = min(tp, up)
        rows.append({"변수": f, "양성n": len(a), "음성n": len(b),
                     "양성중앙": a.median(), "음성중앙": b.median(),
                     "검정": "t" if eq else "Welch", "p값": tp, "MWU_p": up,
                     "유의": "***" if best < 0.01 else ("**" if best < 0.05 else ("*" if best < 0.1 else ""))})
    return pd.DataFrame(rows)


def vif(d, cols):
    X = d[cols].apply(winsorize).dropna()
    if len(X) < 40:
        return None
    c = X.corr()
    if np.linalg.matrix_rank(c.values) < len(cols):
        return pd.DataFrame({"변수": cols, "VIF": ["특이행렬(완전공선)"] * len(cols)})
    inv = np.linalg.inv(c.values)
    return pd.DataFrame({"변수": cols, "VIF": np.diag(inv)}).sort_values("VIF", ascending=False)


if __name__ == "__main__":
    df = build()
    df.to_csv(PROCESSED_DIR / "features_v2.csv", index=False, encoding="utf-8-sig")

    print(f"확정 표본 {len(df)} firm-year / {df.corp_code.nunique()}개사")
    for y, nm in (("y_t1", "t+1"), ("y_t2", "t+2"), ("y_2y", "2년 내")):
        d = df.dropna(subset=[y])
        print(f"  {nm}: 학습쌍 {len(d)}건, 양성 {int(d[y].sum())}건 ({d[y].mean():.1%})")

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    for y, nm in (("y_t1", "t+1년 손상"), ("y_t2", "t+2년 손상"), ("y_2y", "2년 내 손상")):
        d = df.dropna(subset=[y])
        print(f"\n{'=' * 104}\n### {nm}\n{'=' * 104}")
        print(run_tests(d, y).to_string(index=False))

    print(f"\n{'=' * 104}\n### VIF (완전공선 정리 후)\n{'=' * 104}")
    cols = [c for c in FEATURES if c != "증가괴리_3y"]
    v = vif(df.dropna(subset=["y_t1"]), cols)
    if v is not None:
        print(v.to_string(index=False))

    print(f"\n저장 → {PROCESSED_DIR / 'features_v2.csv'}")
