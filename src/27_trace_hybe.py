"""
[추적] 하이브 2025년 수치가 어디에서 나왔는지 역추적

대시보드 값과 사업보고서 주석 화면의 수치가 다르다는 지적에 대한 확인용.
어떤 재무제표(연결/별도)의 어떤 계정을 기준점으로 삼았고,
주석의 어느 표가 대사에 통과했는지를 그대로 출력함.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")
v4 = __import__("14_extractor_v4")
m18 = __import__("18_collect_main")

CC, YEAR = "01204056", 2025


def main():
    print("=" * 88)
    print("[1] 재무상태표 본문 — 무형자산 관련 계정 (API 원본)")
    print("=" * 88)
    for fs in ("CFS", "OFS"):
        import requests, os
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        key = os.getenv("DART_API_KEY", "").strip()
        r = requests.get("https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                         params={"crtfc_key": key, "corp_code": CC, "bsns_year": str(YEAR),
                                 "reprt_code": "11011", "fs_div": fs}, timeout=60).json()
        print(f"\n--- {fs} (status={r.get('status')}) ---")
        if r.get("status") != "000":
            continue
        for it in r.get("list", []):
            if it.get("sj_div") != "BS":
                continue
            nm = (it.get("account_nm") or "").strip()
            if "무형" not in nm and "영업권" not in nm:
                continue
            v = v2.to_num(it.get("thstrm_amount"))
            print(f"  {(it.get('account_id') or '-'):<48} {nm:<20} {v:>20,.0f}")

    print("\n" + "=" * 88)
    print("[2] 기준점(anchor) — 파이프라인이 대사에 쓴 후보값")
    print("=" * 88)
    anc = m18.anchors_for(CC, list(range(2015, 2026)))
    for y in (YEAR - 1, YEAR):
        print(f"  {y}: {sorted(f'{v:,.0f}' for v in anc.get(y, set()))}")

    print("\n" + "=" * 88)
    print("[3] 대사에 통과한 주석 표의 실제 내용")
    print("=" * 88)
    cur, prev = anc.get(YEAR, set()), anc.get(YEAR - 1, set())
    rep = v4.list_annual_reports(CC, YEAR)[0]
    rno = rep["rcept_no"]
    v4.ensure_doc(rno)
    doc_dir = v2.DOC_DIR / rno
    print(f"보고서 {rno} {rep['report_nm'].strip()}")

    chosen = v3.extract(rno, cur, prev)
    print(f"\n▶ 파이프라인 최종 채택: 신뢰도 {chosen['신뢰도']} | 출처 {chosen['출처']} | "
          f"단위 {chosen['단위']:,}")
    print(f"   손상액 {chosen['손상액'] or 0:,.0f} / 상각액 {chosen['상각액'] or 0:,.0f}")

    print("\n--- 대사에 통과한 표 전체 (파이프라인은 이 중 '확정'을 우선 채택함) ---")
    n = 0
    for src, html in v3.iter_tables_loose(doc_dir):
        rows = v2.parse_rows(html)
        if len(rows) < 3:
            continue
        orientation, items = v2.flatten(rows)
        rec = v3.reconcile2(items, cur, prev)
        if not rec:
            continue
        vals = v2.pull_values(items, rec["unit"])
        n += 1
        mark = "★" if (rec["신뢰도"] == chosen["신뢰도"]
                       and (vals["손상액"] or 0) == (chosen["손상액"] or 0)) else " "
        print(f"\n{mark} [{n}] {src} | 배치 {orientation} | 단위 {rec['unit']:,} | 신뢰도 {rec['신뢰도']}")
        print(f"      대사: 주석 {rec['close']:,.0f} = 본문 {rec['anchor']:,.0f}")
        print(f"      추출: 손상액 {vals['손상액'] or 0:,.0f} / 상각액 {vals['상각액'] or 0:,.0f}")
        print(f"      헤더: {' | '.join(rows[0][:10])}")
        for it in items[:16]:
            tot = it["total"]
            s = f"{tot:>20,.0f}" if tot is not None else " " * 20
            print(f"        [{it['section'] or '-':<4}] {it['label'][:34]:<34} {s}")
        if n >= 4:
            break


if __name__ == "__main__":
    main()
