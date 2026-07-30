"""
[시범 수집] 손상차손 항목 확보 가능성 검증

이 프로젝트의 정답 레이블은 '무형자산손상차손'인데, DART API가 이 항목을
실제로 얼마나 반환하는지가 프로젝트 성립의 전제임. 본 수집에 앞서
소수 기업만으로 다음 두 가지를 확인함.

  (1) 무형자산 계정 자체가 잡히는가
  (2) 손상차손이 손익계산서/현금흐름표에 표시되는가 (주석에만 있으면 API로 못 잡음)

산출물: data/raw/pilot/*.json, 콘솔 요약표
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PILOT_DIR = BASE_DIR / "data" / "raw" / "pilot"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()

API_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

TARGETS = {
    "01204056": "하이브",
    "00260930": "에스엠",
    "00613318": "와이지엔터테인먼트",
    "00258689": "JYP Ent.",
    "01168684": "스튜디오드래곤",
    "00203315": "콘텐트리중앙",
    "00975290": "에이스토리",
    "00303794": "쇼박스",
    "01186404": "디앤씨미디어",
    "00140131": "키다리스튜디오",
}

YEARS = range(2018, 2025)


def fetch(corp_code: str, year: int) -> dict:
    """사업보고서(11011) 기준 연결재무제표 전체 계정을 가져옴."""
    res = requests.get(
        API_URL,
        params={
            "crtfc_key": API_KEY,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": "11011",
            "fs_div": "CFS",  # 연결. 없으면 별도로 재시도함
        },
        timeout=60,
    )
    payload = res.json()

    # 연결재무제표 미작성 회사는 별도(OFS)로 폴백함
    if payload.get("status") == "013":
        res = requests.get(
            API_URL,
            params={
                "crtfc_key": API_KEY,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": "11011",
                "fs_div": "OFS",
            },
            timeout=60,
        )
        payload = res.json()
        payload["_fs_div"] = "OFS"
    else:
        payload["_fs_div"] = "CFS"

    return payload


def main():
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    impair_account_names = {}

    for corp_code, name in TARGETS.items():
        for year in YEARS:
            payload = fetch(corp_code, year)
            status = payload.get("status")

            out = PILOT_DIR / f"{corp_code}_{year}.json"
            out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            if status != "000":
                rows.append(
                    {"기업": name, "연도": year, "재무제표": "-", "계정수": 0,
                     "무형자산": "", "손상계정": "", "상태": status}
                )
                time.sleep(0.15)
                continue

            items = payload.get("list", [])
            # 손상 관련 계정을 계정명 기준으로 탐색함 (회사마다 표기가 달라 부분일치 사용)
            impair = [
                it for it in items
                if "손상" in (it.get("account_nm") or "")
            ]
            intangible = [
                it for it in items
                if (it.get("sj_div") == "BS") and ("무형자산" in (it.get("account_nm") or ""))
            ]

            for it in impair:
                key = (it.get("sj_div"), it.get("account_nm"))
                impair_account_names[key] = impair_account_names.get(key, 0) + 1

            rows.append({
                "기업": name,
                "연도": year,
                "재무제표": payload.get("_fs_div"),
                "계정수": len(items),
                "무형자산": "O" if intangible else "X",
                "손상계정": len(impair),
                "상태": status,
            })
            time.sleep(0.15)

    df = pd.DataFrame(rows)
    print("\n===== 수집 결과 =====")
    print(df.to_string(index=False))

    print("\n===== '손상' 포함 계정명 출현 빈도 (재무제표구분, 계정명) =====")
    if impair_account_names:
        freq = pd.DataFrame(
            [{"구분": k[0], "계정명": k[1], "출현": v} for k, v in impair_account_names.items()]
        ).sort_values("출현", ascending=False)
        print(freq.to_string(index=False))
    else:
        print("!! 손상 관련 계정이 전혀 잡히지 않음 — 설계 재검토 필요")

    ok = df[df["상태"] == "000"]
    print(f"\n수집 성공 {len(ok)}/{len(df)} firm-year")
    if len(ok):
        print(f"무형자산 계정 확보율: {(ok['무형자산'] == 'O').mean():.1%}")
        print(f"손상 계정 1개 이상 확보: {(ok['손상계정'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
