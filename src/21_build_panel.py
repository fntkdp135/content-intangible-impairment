"""
[보완 3·4] 분석용 패널 구축 — 연결/별도 전환 플래그 + 적용범위 판정

3) 연결/별도 전환
   팬엔터테인먼트는 2023년까지 별도, 2024년부터 연결로 바뀌면서 무형자산이
   15억 → 185억으로 뛴다. 실제 증가가 아니라 재무제표 기준이 바뀐 것이므로,
   이 구간의 증가율을 그대로 쓰면 핵심 변수가 오염됨. 전환 연도를 표시해
   증가율 계산에서 끊을 수 있게 함.

4) 적용범위 판정
   티비씨(무형자산 비중 0.04%)·케이엔엔(0.06%)처럼 무형자산이 사실상 없는 회사는
   콘텐츠 손상을 예측할 대상 자체가 없음. 파서 실패가 아니라 모델 적용 대상이 아님.
   폐기한 5% 게이트와 혼동하지 말 것 — 그건 와이지(4.1%)·쇼박스(2.4%) 같은
   실제 엔터사를 잘라내서 폐기한 것이고, 여기 기준은 0.5%로 성격이 다름.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SCOPE_MIN_RATIO = 0.005  # 전 기간 무형자산 비중 중앙값 0.5% 미만이면 적용 대상 아님


def main():
    fin = pd.read_csv(PROCESSED_DIR / "financials.csv", dtype={"corp_code": str})
    imp = pd.read_csv(PROCESSED_DIR / "impairment_v2.csv", dtype={"corp_code": str})

    df = fin.merge(
        imp[["corp_code", "사업연도", "대사", "신뢰도", "합산표시", "손상액", "상각액"]],
        on=["corp_code", "사업연도"], how="outer")
    df = df.sort_values(["corp_code", "사업연도"]).reset_index(drop=True)

    # --- 3) 연결/별도 전환 표시 ---
    df["직전재무제표"] = df.groupby("corp_code")["재무제표"].shift(1)
    df["기준전환"] = (df["재무제표"].notna() & df["직전재무제표"].notna()
                  & (df["재무제표"] != df["직전재무제표"]))

    # --- 4) 적용범위 판정 ---
    df["무형비중"] = df["무형자산"] / df["자산총계"]
    med = df.groupby("corp_code")["무형비중"].median().rename("무형비중_중앙값")
    df = df.merge(med, on="corp_code", how="left")
    df["적용대상"] = df["무형비중_중앙값"] >= SCOPE_MIN_RATIO

    # --- 결측 사유 기록 ---
    def reason(r):
        if pd.isna(r["자산총계"]):
            return "본문없음"
        if not r["적용대상"]:
            return "적용대상아님"
        if r["대사"] != "성공":
            return "주석대사실패"
        if r["합산표시"] is True:
            return "상각손상합산"
        return "사용가능"

    df["상태"] = df.apply(reason, axis=1)
    df.to_csv(PROCESSED_DIR / "panel.csv", index=False, encoding="utf-8-sig")

    # ================= 보고 =================
    print(f"패널 {len(df):,} firm-year / {df['corp_code'].nunique()}개사\n")

    print("===== 상태별 =====")
    print(df["상태"].value_counts().to_string())

    print("\n===== 적용 대상 제외 기업 (무형자산 비중 중앙값 0.5% 미만) =====")
    out = (df[~df["적용대상"]].groupby("기업명")
           .agg(연도수=("사업연도", "size"), 무형비중중앙값=("무형비중_중앙값", "first"))
           .sort_values("무형비중중앙값"))
    print(out.assign(무형비중중앙값=lambda d: d.무형비중중앙값.map("{:.3%}".format)).to_string())

    print("\n===== 연결/별도 전환 발생 =====")
    sw = df[df["기준전환"]]
    print(f"전환 firm-year {len(sw)}건 / {sw['corp_code'].nunique()}개사")
    if len(sw):
        print(sw.groupby("기업명")["사업연도"].apply(
            lambda s: ",".join(map(str, sorted(s)))).to_string())

    use = df[df["상태"] == "사용가능"]
    print("\n===== 최종 사용 가능 표본 =====")
    print(f"firm-year {len(use):,}건 / {use['corp_code'].nunique()}개사")
    print(f"손상액 확보 {use['손상액'].notna().sum():,}건")

    nz = use[use["손상액"].fillna(0) > 0]
    print(f"손상 인식(>0) {len(nz):,}건 ({len(nz)/len(use):.1%})")

    u = use.copy()
    u["손상_무형비"] = u["손상액"] / u["무형자산"]
    u["손상_자산비"] = u["손상액"] / u["자산총계"]
    print("\n중요성 기준별 양성:")
    for lbl, cond in [("손상>0", u["손상액"] > 0),
                      ("손상/무형 ≥1%", u["손상_무형비"] >= 0.01),
                      ("손상/무형 ≥5%", u["손상_무형비"] >= 0.05),
                      ("손상/자산 ≥0.5%", u["손상_자산비"] >= 0.005),
                      ("손상/자산 ≥1%", u["손상_자산비"] >= 0.01)]:
        print(f"  {lbl:<16} {cond.sum():>4}건 ({cond.mean():.1%})")

    print("\n===== 대사 성공률 재확인 (적용 대상 기업만) =====")
    tgt = df[df["적용대상"] & df["자산총계"].notna()]
    print(f"  {(tgt['대사'] == '성공').sum()}/{len(tgt)} ({(tgt['대사'] == '성공').mean():.1%})")
    sz = tgt.dropna(subset=["자산총계"]).copy()
    sz["규모"] = pd.qcut(sz["자산총계"], 4, labels=["소형", "중소", "중대", "대형"])
    r = sz.groupby("규모", observed=True).apply(
        lambda g: pd.Series({"건수": len(g), "성공률": (g["대사"] == "성공").mean()}),
        include_groups=False)
    print(r.assign(성공률=lambda d: d.성공률.map("{:.1%}".format)).to_string())

    print(f"\n저장 → {PROCESSED_DIR / 'panel.csv'}")


if __name__ == "__main__":
    main()
