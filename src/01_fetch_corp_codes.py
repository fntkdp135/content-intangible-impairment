"""
DART OpenAPI 연결 테스트 + 기업코드 마스터 수집

- DART의 모든 공시대상 회사에 대한 고유번호(corp_code) 매핑 파일을 내려받음
- 이후 모든 API 호출은 이 corp_code를 키로 사용하므로 최초 1회 실행이 필요함
- 산출물: data/raw/CORPCODE.xml, data/processed/corp_codes.csv
"""

import io
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()


def check_key():
    """인증키 유효성을 가벼운 호출로 먼저 확인함."""
    if not API_KEY or len(API_KEY) != 40:
        sys.exit(f"[중단] .env의 DART_API_KEY가 비어있거나 40자가 아님 (현재 {len(API_KEY)}자)")

    res = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={"crtfc_key": API_KEY, "bgn_de": "20260701", "end_de": "20260710", "page_count": "1"},
        timeout=30,
    )
    payload = res.json()
    status, message = payload.get("status"), payload.get("message")

    # 000=정상, 013=조회 데이터 없음(키는 유효). 그 외는 키/권한 문제로 간주함
    if status not in ("000", "013"):
        sys.exit(f"[중단] 인증키 오류 (status={status}, message={message})")

    print(f"[OK] 인증키 정상 (status={status}, message={message})")


def fetch_corp_codes():
    """전체 공시대상 회사의 고유번호 매핑을 zip으로 받아 파싱함."""
    res = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": API_KEY},
        timeout=120,
    )

    # 오류 시 zip이 아니라 JSON/XML 에러 응답이 내려오므로 방어함
    if not res.content[:2] == b"PK":
        sys.exit(f"[중단] zip 응답이 아님. 서버 응답: {res.text[:300]}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        zf.extractall(RAW_DIR)
    print(f"[OK] CORPCODE.xml 저장 → {RAW_DIR}")

    root = ET.parse(RAW_DIR / "CORPCODE.xml").getroot()
    rows = [
        {
            "corp_code": (item.findtext("corp_code") or "").strip(),
            "corp_name": (item.findtext("corp_name") or "").strip(),
            "stock_code": (item.findtext("stock_code") or "").strip(),
            "modify_date": (item.findtext("modify_date") or "").strip(),
        }
        for item in root.iter("list")
    ]
    df = pd.DataFrame(rows)

    # stock_code가 있는 곳만 상장사임 (비상장 포함 시 9만건이 넘어 분석 대상이 아님)
    listed = df[df["stock_code"].str.strip() != ""].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "corp_codes.csv"
    listed.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 전체 {len(df):,}건 중 상장사 {len(listed):,}건 → {out_path}")
    return listed


if __name__ == "__main__":
    check_key()
    listed = fetch_corp_codes()
    print("\n[샘플]")
    print(listed.head(5).to_string(index=False))
