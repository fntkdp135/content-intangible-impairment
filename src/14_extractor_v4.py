"""
[A안 v4] 정정보고서 폴백 + 계정 단위 기준점 확장

v3의 남은 실패 5건 중 3건(하이브2023·에스엠2024·와이지2025)이 [기재정정]사업보고서였음.
정정보고서는 고친 부분만 담는 경우가 있어 무형자산 주석이 통째로 없을 수 있음
(하이브 2023은 후보 표가 0개였음). 가장 최근 접수번호만 쓰면 이 함정에 빠짐.

v4의 보완
  (1) 한 사업연도에 제출된 사업보고서를 모두 후보로 두고, 정정본 → 원본 순으로
      대사가 될 때까지 시도함
  (2) 영업권을 단독 기준점 후보로 추가함. 회사에 따라 영업권 표와 기타무형자산 표가
      분리 공시되어 합계로는 어느 표와도 맞지 않는 경우가 있음
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CODE_OF = {v: k for k, v in v2.TARGETS.items()}


def list_annual_reports(corp_code: str, bsns_year: int):
    """해당 사업연도의 사업보고서를 모두 반환함(정정본 포함, 최신순)."""
    res = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={"crtfc_key": v2.API_KEY, "corp_code": corp_code,
                "bgn_de": f"{bsns_year + 1}0101", "end_de": f"{bsns_year + 2}0630",
                "pblntf_detail_ty": "A001", "page_count": "100"},
        timeout=60).json()
    if res.get("status") != "000":
        return []
    cands = [i for i in res.get("list", [])
             if "사업보고서" in i.get("report_nm", "") and f"({bsns_year}.12)" in i.get("report_nm", "")]
    cands.sort(key=lambda x: x["rcept_no"], reverse=True)
    return cands


def ensure_doc(rcept_no: str):
    out_dir = v2.DOC_DIR / rcept_no
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        return out_dir
    import io
    import zipfile
    res = requests.get("https://opendart.fss.or.kr/api/document.xml",
                       params={"crtfc_key": v2.API_KEY, "rcept_no": rcept_no}, timeout=300)
    if res.content[:2] != b"PK":
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        zf.extractall(out_dir)
    return out_dir


def build_anchors_v4():
    """영업권 단독도 기준점 후보에 넣어 확장함."""
    base = v2.build_anchors()
    for corp_code, name in v2.TARGETS.items():
        for year in v2.YEARS:
            payload = v2.fetch_fs(corp_code, year)
            if payload.get("status") != "000":
                continue
            gw = {0: 0.0, 1: 0.0, 2: 0.0}
            for it in payload.get("list", []):
                if it.get("sj_div") != "BS":
                    continue
                if (it.get("account_id") or "").strip() not in v2.GOODWILL_IDS:
                    continue
                for tag, off in (("thstrm_amount", 0), ("frmtrm_amount", 1), ("bfefrmtrm_amount", 2)):
                    v = v2.to_num(it.get(tag))
                    if v:
                        gw[off] += v
            for off, val in gw.items():
                if val > 0:
                    base.setdefault((name, year - off), set()).add(round(val, 2))
    return base


if __name__ == "__main__":
    print("[1/2] 기준점 구축 중…")
    anchors = build_anchors_v4()

    print("[2/2] 정정본→원본 폴백 대사 중…")
    records = []
    for name, corp_code in CODE_OF.items():
        for year in v2.YEARS:
            cur = anchors.get((name, year), set())
            prev = anchors.get((name, year - 1), set())
            reports = list_annual_reports(corp_code, year)

            chosen, used = None, None
            for rep in reports:
                rno = rep["rcept_no"]
                if ensure_doc(rno) is None:
                    continue
                res = v3.extract(rno, cur, prev)
                if res is None:
                    continue
                chosen, used = res, rep
                if res["신뢰도"] == "확정":
                    break
                time.sleep(0.2)

            row = {"기업": name, "사업연도": year, "보고서수": len(reports)}
            if chosen is None:
                row.update({"대사": "실패", "본문무형자산": max(cur) if cur else None})
            else:
                impair = chosen["손상액"]
                if impair is None and not chosen["상각손상합산"]:
                    impair = 0.0
                row.update({
                    "대사": "성공", "신뢰도": chosen["신뢰도"],
                    "사용보고서": "정정" if "정정" in used["report_nm"] else "원본",
                    "본문무형자산": chosen["본문금액"], "주석기말": chosen["주석기말"],
                    "합산표시": chosen["상각손상합산"],
                    "손상액": impair, "상각액": chosen["상각액"],
                })
            records.append(row)
            print(f"  {name} {year} … {row['대사']}"
                  f"{'(' + row.get('사용보고서', '') + ')' if row['대사'] == '성공' else ''}")

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "note_extract_v4.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print("\n" + df.to_string(index=False))

    n, ok = len(df), df[df["대사"] == "성공"]
    print("\n===== 요약 =====")
    print(f"대상 firm-year   : {n}건")
    print(f"대사 성공        : {len(ok)}건 ({len(ok)/n:.1%})")
    if len(ok):
        print(f"  두 지점 확정   : {(ok['신뢰도'] == '확정').sum()}건")
        print(f"  원본 사용      : {(ok['사용보고서'] == '원본').sum()}건")
        print(f"  정정본 사용    : {(ok['사용보고서'] == '정정').sum()}건")
        print(f"  손상액 확보    : {ok['손상액'].notna().sum()}건")
