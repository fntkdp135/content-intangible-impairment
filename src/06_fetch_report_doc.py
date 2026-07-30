"""
[A안 시범 1단계] 사업보고서 원문 확보 및 구조 파악

재무제표 본문(API)에는 손상차손이 표시되지 않는 회사가 많으므로,
사업보고서 원문의 '무형자산' 주석에서 변동내역표를 추출해야 함.
그 전에 원문이 어떤 형식으로 내려오는지부터 확인함.

절차: 공시검색(list.json)으로 사업보고서 접수번호 조회
      → 공시서류원본파일(document.xml)로 원문 zip 다운로드
"""

import io
import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "data" / "raw" / "docs"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()


def find_annual_report(corp_code: str, bsns_year: int):
    """해당 사업연도의 사업보고서 접수번호를 찾음.

    사업보고서는 결산 다음 해 3월경 제출되므로 제출연도 기준으로 검색함.
    정정보고서가 있으면 최신 것을 쓰는 게 맞으므로 접수번호 내림차순 우선.
    """
    res = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={
            "crtfc_key": API_KEY,
            "corp_code": corp_code,
            "bgn_de": f"{bsns_year + 1}0101",
            "end_de": f"{bsns_year + 1}1231",
            "pblntf_detail_ty": "A001",  # 사업보고서
            "page_count": "100",
        },
        timeout=60,
    )
    payload = res.json()
    if payload.get("status") != "000":
        return None

    cands = [
        it for it in payload.get("list", [])
        if "사업보고서" in it.get("report_nm", "")
    ]
    if not cands:
        return None
    cands.sort(key=lambda x: x["rcept_no"], reverse=True)
    return cands[0]


def download_document(rcept_no: str) -> list:
    """공시서류 원문 zip을 받아 압축 해제함."""
    res = requests.get(
        "https://opendart.fss.or.kr/api/document.xml",
        params={"crtfc_key": API_KEY, "rcept_no": rcept_no},
        timeout=300,
    )
    if res.content[:2] != b"PK":
        raise RuntimeError(f"zip 아님: {res.text[:300]}")

    out_dir = DOC_DIR / rcept_no
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        zf.extractall(out_dir)
        names = zf.namelist()
    return [out_dir / n for n in names]


if __name__ == "__main__":
    # 손상차손이 본문에 안 보였던 대표 사례로 확인함
    info = find_annual_report("01168684", 2023)
    print(f"[접수] {info['report_nm'].strip()} | rcept_no={info['rcept_no']} | {info['rcept_dt']}")

    files = download_document(info["rcept_no"])
    for f in files:
        print(f"  - {f.name}  ({f.stat().st_size:,} bytes)")
