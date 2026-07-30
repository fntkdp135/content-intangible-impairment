"""
[모집단 2단계] 엔터·미디어·콘텐츠 상장사 모집단 확정

업종코드 필터는 추측이 아니라 시범 10개사의 실제 코드를 근거로 구성함.
  하이브 592 / 에스엠·와이지·JYP 59201 / 스튜디오드래곤 59114 / 쇼박스 59130
  콘텐트리중앙·에이스토리 591 / 디앤씨미디어 581 / 키다리스튜디오 631
코드 자릿수가 3~5자리로 제각각이라 접두 매칭을 씀.

상장폐지 기업을 빼지 않음: 손상을 예측하는 모델에서 사라진 기업을 제외하면
생존편향이 생겨, 정작 위험했던 사례가 표본에서 통째로 빠짐.
대신 '사업보고서를 실제 제출한 연도'를 확인해 데이터가 있는 구간만 사용함.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()

# 시범 10개사 코드로 검증된 핵심 대역
CORE_PREFIXES = {
    "591": "영상·방송프로그램 제작/배급/상영",
    "592": "오디오물·음반",
    "601": "라디오 방송",
    "602": "텔레비전 방송",
    "603": "기타 방송·음원서비스",
    "901": "창작·예술 서비스",
    "902": "창작·예술 서비스",
}
# 웹툰·웹소설 IP 및 출판. 교육출판이 섞이므로 사후 검토 대상으로 표시함
BORDERLINE_PREFIXES = {
    "581": "출판(웹툰·웹소설 IP / 교육출판 혼재)",
    "631": "포털·콘텐츠 플랫폼",
    "639": "기타 정보서비스",
}
# 게임(582x)은 5번 프로젝트와 중복되고 개발비 자산 성격이 달라 제외함
EXCLUDE_PREFIXES = ("582",)


def classify(code: str):
    code = (code or "").strip()
    if not code or code.startswith(EXCLUDE_PREFIXES):
        return None, None
    for p, label in CORE_PREFIXES.items():
        if code.startswith(p):
            return "핵심", label
    for p, label in BORDERLINE_PREFIXES.items():
        if code.startswith(p):
            return "경계", label
    return None, None


def annual_report_years(corp_code: str):
    """2016~2026년에 제출한 사업보고서로부터 보유 사업연도를 구함."""
    res = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={"crtfc_key": API_KEY, "corp_code": corp_code,
                "bgn_de": "20160101", "end_de": "20261231",
                "pblntf_detail_ty": "A001", "page_count": "100"},
        timeout=60).json()
    if res.get("status") != "000":
        return []
    years = set()
    for it in res.get("list", []):
        nm = it.get("report_nm", "")
        if "사업보고서" not in nm:
            continue
        # '사업보고서 (2023.12)' 형태에서 사업연도를 뽑음
        import re
        m = re.search(r"\((\d{4})\.\d{2}\)", nm)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def main():
    df = pd.read_csv(PROCESSED_DIR / "corp_industry.csv", dtype=str)
    df["구분"], df["업종"] = zip(*df["induty_code"].map(classify))
    cand = df[df["구분"].notna()].copy()
    print(f"업종 필터 통과: {len(cand):,}개사 (핵심 {(cand['구분']=='핵심').sum()}, 경계 {(cand['구분']=='경계').sum()})")

    print("\n사업보고서 제출 이력 조회 중…")
    rows = []
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        years = annual_report_years(r["corp_code"])
        rows.append({
            "corp_code": r["corp_code"], "기업명": r["corp_name"],
            "종목코드": r["stock_code"], "업종코드": r["induty_code"],
            "구분": r["구분"], "업종": r["업종"],
            "보고연도수": len(years),
            "최초": min(years) if years else None,
            "최종": max(years) if years else None,
            "연도목록": ",".join(map(str, years)),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(cand)}")
        time.sleep(0.05)

    uni = pd.DataFrame(rows)
    # 사업보고서가 한 건도 없으면 분석 불가이므로 제외함
    uni = uni[uni["보고연도수"] > 0].reset_index(drop=True)
    uni.to_csv(PROCESSED_DIR / "universe.csv", index=False, encoding="utf-8-sig")

    print("\n===== 모집단 요약 =====")
    print(f"최종 모집단        : {len(uni):,}개사")
    print(f"  핵심 업종        : {(uni['구분']=='핵심').sum()}개사")
    print(f"  경계 업종(검토요) : {(uni['구분']=='경계').sum()}개사")
    print(f"총 firm-year 잠재량 : {uni['보고연도수'].sum():,}건")
    print(f"기업당 평균 보고연도 : {uni['보고연도수'].mean():.1f}년")

    print("\n===== 업종별 =====")
    print(uni.groupby(["구분", "업종"]).size().to_string())

    print("\n===== 보고연도수 분포 =====")
    print(uni["보고연도수"].value_counts().sort_index().to_string())

    print("\n===== 최종 보고연도 기준 (상장폐지 추정) =====")
    print(uni["최종"].value_counts().sort_index().to_string())

    print(f"\n저장 → {PROCESSED_DIR / 'universe.csv'}")


if __name__ == "__main__":
    main()
