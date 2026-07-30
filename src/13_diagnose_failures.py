"""
[진단] 남은 대사 실패 5건의 원인 규명 + 연결/별도 일관성 점검

확인 항목
  (1) 보고서 종류 — 정정보고서인가, 첨부 감사보고서가 있는가
  (2) 기준점 출처 — 연결(CFS)인가 별도(OFS)인가, 어느 계정에서 나온 금액인가
      같은 기업인데 연도별로 연결/별도가 섞이면 무형자산 증가율 변수가 망가짐
  (3) 실패 지점 — 표 자체를 못 찾은 것인가, 찾았는데 금액이 안 맞은 것인가
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FAILURES = [("하이브", 2023), ("에스엠", 2024), ("와이지엔터테인먼트", 2025),
            ("키다리스튜디오", 2023), ("키다리스튜디오", 2024)]

CODE_OF = {v: k for k, v in v2.TARGETS.items()}


def anchor_detail(corp_code, year):
    """기준점이 어느 재무제표·어느 계정에서 나왔는지까지 반환함."""
    payload = v2.fetch_fs(corp_code, year)
    if payload.get("status") != "000":
        return payload.get("_fs_div"), payload.get("status"), []
    rows = []
    for it in payload.get("list", []):
        if it.get("sj_div") != "BS":
            continue
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip()
        if aid in v2.GOODWILL_IDS or aid in v2.OTHER_INTAN_IDS or aid in v2.COMBINED_IDS \
           or ("무형자산" in nm and "상각" not in nm):
            v = v2.to_num(it.get("thstrm_amount"))
            rows.append((aid or "-", nm, v))
    return payload.get("_fs_div"), payload.get("status"), rows


def main():
    print("=" * 90)
    print("[1] 기준점 출처 — 연결/별도 일관성")
    print("=" * 90)
    prov = []
    for name, corp_code in CODE_OF.items():
        for year in v2.YEARS:
            fs_div, status, rows = anchor_detail(corp_code, year)
            prov.append({"기업": name, "연도": year, "재무제표": fs_div, "상태": status,
                         "계정수": len(rows)})
    dfp = pd.DataFrame(prov)
    pivot = dfp.pivot(index="기업", columns="연도", values="재무제표")
    print(pivot.to_string())
    mixed = [i for i, r in pivot.iterrows() if r.dropna().nunique() > 1]
    print(f"\n연결/별도가 섞인 기업: {mixed if mixed else '없음'}")

    print("\n" + "=" * 90)
    print("[2] 실패 5건 — 보고서 종류와 첨부 구성")
    print("=" * 90)
    pilot = pd.read_csv(PROCESSED_DIR / "note_parse_pilot.csv", dtype={"접수번호": str})
    for name, year in FAILURES:
        row = pilot[(pilot["기업"] == name) & (pilot["사업연도"] == year)].iloc[0]
        rno = row["접수번호"]
        info = v3.__dict__  # noqa - 사용하지 않음
        doc_dir = v2.DOC_DIR / rno
        files = sorted(p.name for p in doc_dir.glob("*.xml")) if doc_dir.exists() else []

        # 보고서명 재조회 (정정보고서 여부 확인)
        import requests
        res = requests.get(
            "https://opendart.fss.or.kr/api/list.json",
            params={"crtfc_key": v2.API_KEY, "corp_code": CODE_OF[name],
                    "bgn_de": f"{year + 1}0101", "end_de": f"{year + 2}0630",
                    "pblntf_detail_ty": "A001", "page_count": "100"},
            timeout=60).json()
        names = [f"{i['rcept_no']} {i['report_nm'].strip()}" for i in res.get("list", [])]

        print(f"\n### {name} {year}  (사용 접수번호 {rno})")
        print(f"  파일: {files}")
        print(f"  해당기간 사업보고서 공시목록:")
        for s in names:
            mark = " ←사용" if s.startswith(rno) else ""
            print(f"    {s}{mark}")

    print("\n" + "=" * 90)
    print("[3] 실패 5건 — 후보 표와 본문 금액의 괴리")
    print("=" * 90)
    anchors = v2.build_anchors()
    for name, year in FAILURES:
        row = pilot[(pilot["기업"] == name) & (pilot["사업연도"] == year)].iloc[0]
        rno = row["접수번호"]
        cur = anchors.get((name, year), set())
        prev = anchors.get((name, year - 1), set())
        print(f"\n### {name} {year}")
        print(f"  당기말 기준점 후보: {sorted(f'{a:,.0f}' for a in cur)}")
        print(f"  전기말 기준점 후보: {sorted(f'{a:,.0f}' for a in prev)}")

        doc_dir = v2.DOC_DIR / rno
        if not doc_dir.exists():
            print("  문서 없음")
            continue

        shown = 0
        for src, html in v3.iter_tables_loose(doc_dir):
            if not v3.P_INTAN_HINT.search(html):
                continue
            rows_ = v2.parse_rows(html)
            if len(rows_) < 3:
                continue
            orientation, items = v2.flatten(rows_)
            closes = v2.closing_candidates(items)
            if not closes:
                continue
            best = None
            for a in cur:
                for c in closes:
                    for u in v2.UNITS:
                        d = abs(c * u - a) / a
                        if best is None or d < best[0]:
                            best = (d, c * u, a, u)
            print(f"  - {src} 배치{orientation} 기말후보={[f'{c:,.0f}' for c in closes[:4]]}"
                  f" 최근접오차={best[0]:.4%} (단위 {best[3]:,})" if best else "")
            shown += 1
            if shown >= 6:
                break
        if shown == 0:
            print("  무형자산 관련 후보 표 없음")


if __name__ == "__main__":
    main()
