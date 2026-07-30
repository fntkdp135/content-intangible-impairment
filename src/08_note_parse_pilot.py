"""
[A안 시범 3단계] 무형자산 주석 파싱 성공률 측정

측정 목적 3가지
  (1) 사업보고서 원문에서 무형자산 변동표를 몇 %나 추출할 수 있는가
  (2) 손상차손이 상각과 분리되어 공시되는 비율은 얼마인가
      (스튜디오드래곤처럼 '상각 및 손상'으로 합쳐 쓰면 레이블 분리가 불가능함)
  (3) 판권·콘텐츠 자산이 별도 컬럼으로 구분 공시되는 비율은 얼마인가
      (구분되면 전체 무형자산이 아닌 콘텐츠 자산만 정밀하게 분석할 수 있음)

원문은 건당 약 2MB이므로 한 번 받은 문서는 캐시해 재사용함.
"""

import io
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "data" / "raw" / "docs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()

TARGETS = {
    "01204056": "하이브", "00260930": "에스엠", "00613318": "와이지엔터테인먼트",
    "00258689": "JYP Ent.", "01168684": "스튜디오드래곤", "00203315": "콘텐트리중앙",
    "00975290": "에이스토리", "00303794": "쇼박스", "01186404": "디앤씨미디어",
    "00140131": "키다리스튜디오",
}
# 사업보고서는 결산 다음 해 3월경 제출됨. 2025 사업연도분은 2026-03에 제출 완료되어 수집 가능함
YEARS = [2021, 2022, 2023, 2024, 2025]

CONTENT_COL_PAT = re.compile(r"(판권|콘텐츠|컨텐츠|프로그램|영상|저작권|음원|음반)")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def find_annual_report(corp_code: str, bsns_year: int):
    res = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={
            "crtfc_key": API_KEY, "corp_code": corp_code,
            "bgn_de": f"{bsns_year + 1}0101", "end_de": f"{bsns_year + 1}1231",
            "pblntf_detail_ty": "A001", "page_count": "100",
        },
        timeout=60,
    )
    payload = res.json()
    if payload.get("status") != "000":
        return None
    cands = [it for it in payload.get("list", []) if "사업보고서" in it.get("report_nm", "")]
    if not cands:
        return None
    cands.sort(key=lambda x: x["rcept_no"], reverse=True)
    return cands[0]


def get_document(rcept_no: str) -> Path:
    """원문 zip을 받아 캐시함. 이미 받은 문서는 재다운로드하지 않음."""
    out_dir = DOC_DIR / rcept_no
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        return out_dir

    res = requests.get(
        "https://opendart.fss.or.kr/api/document.xml",
        params={"crtfc_key": API_KEY, "rcept_no": rcept_no},
        timeout=300,
    )
    if res.content[:2] != b"PK":
        raise RuntimeError(f"zip 아님: {res.text[:200]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        zf.extractall(out_dir)
    return out_dir


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ")).strip()


def parse_table(table_html: str):
    """표를 (헤더, 행라벨 리스트) 형태로 눌러냄."""
    rows = []
    for r in re.findall(r"<TR[\s\S]*?</TR>", table_html, flags=re.IGNORECASE):
        cells = [strip_tags(c) for c in re.findall(r"<T[DH][\s\S]*?</T[DH]>", r, flags=re.IGNORECASE)]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(cells)
    return rows


def find_intangible_tables(doc_dir: Path):
    """무형자산 변동내역표로 보이는 표들을 모음."""
    found = []
    for path in sorted(doc_dir.glob("*.xml")):
        text = read_text(path)
        for t in re.findall(r"<TABLE[\s\S]*?</TABLE>", text, flags=re.IGNORECASE):
            if "무형자산" not in t:
                continue
            if not ("기초" in t and "기말" in t):
                continue
            if not ("상각" in t or "손상" in t):
                continue
            found.append(t)
    return found


def classify(tables):
    """표에서 손상 분리 여부와 콘텐츠 자산 구분 여부를 판정함."""
    sep_impair = False   # 손상만 단독으로 표시된 행이 있는가
    combined = False     # '상각 및 손상'처럼 합산 표시인가
    content_col = False  # 판권/콘텐츠 컬럼이 있는가
    col_names = []

    for t in tables:
        rows = parse_table(t)
        if not rows:
            continue
        header = rows[0]
        if CONTENT_COL_PAT.search(" ".join(header)):
            content_col = True
            col_names = header
        for cells in rows:
            label = cells[0]
            if "손상" not in label:
                continue
            if "상각" in label:
                combined = True
            else:
                sep_impair = True
    return sep_impair, combined, content_col, col_names


def main():
    records = []
    for corp_code, name in TARGETS.items():
        for year in YEARS:
            row = {"기업": name, "사업연도": year, "접수번호": "", "표발견": 0,
                   "손상분리": "", "상각손상합산": "", "콘텐츠구분": "", "비고": ""}
            try:
                info = find_annual_report(corp_code, year)
                if not info:
                    row["비고"] = "사업보고서 없음"
                    records.append(row)
                    continue
                row["접수번호"] = info["rcept_no"]

                doc_dir = get_document(info["rcept_no"])
                tables = find_intangible_tables(doc_dir)
                row["표발견"] = len(tables)

                if tables:
                    sep, comb, content, cols = classify(tables)
                    row["손상분리"] = "O" if sep else "X"
                    row["상각손상합산"] = "O" if comb else "X"
                    row["콘텐츠구분"] = "O" if content else "X"
                    if cols:
                        row["비고"] = " / ".join(cols[:8])
            except Exception as e:  # 개별 실패가 전체를 막지 않도록 함
                row["비고"] = f"오류: {type(e).__name__} {e}"[:80]

            records.append(row)
            print(f"  {name} {year} … 표{row['표발견']}개 "
                  f"손상분리={row['손상분리']} 합산={row['상각손상합산']} 콘텐츠={row['콘텐츠구분']}")
            time.sleep(0.3)

    df = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "note_parse_pilot.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(df.drop(columns=["비고"]).to_string(index=False))

    total = len(df)
    got = df[df["표발견"] > 0]
    print("\n===== 요약 =====")
    print(f"대상 firm-year            : {total}건")
    print(f"무형자산 주석표 추출 성공  : {len(got)}건 ({len(got)/total:.1%})")
    if len(got):
        print(f"  손상 단독 행 존재       : {(got['손상분리'] == 'O').sum()}건 ({(got['손상분리'] == 'O').mean():.1%})")
        print(f"  상각·손상 합산 표시     : {(got['상각손상합산'] == 'O').sum()}건 ({(got['상각손상합산'] == 'O').mean():.1%})")
        print(f"  콘텐츠 자산 별도 컬럼   : {(got['콘텐츠구분'] == 'O').sum()}건 ({(got['콘텐츠구분'] == 'O').mean():.1%})")

    print("\n===== 기업별 컬럼 구성 (샘플) =====")
    for _, r in got.drop_duplicates(subset=["기업"]).iterrows():
        print(f"  {r['기업']:<12} {r['비고']}")


if __name__ == "__main__":
    main()
