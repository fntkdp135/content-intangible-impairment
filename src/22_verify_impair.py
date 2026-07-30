"""
[검증] 손상액이 무형자산 잔액을 초과하는 사례 점검

손상액/무형자산이 1을 넘는 건이 상위 10%에 달함. 기중에 자산을 거의 전액
손상하면 이론적으로 가능하지만, **손상차손누계액(잔액)을 당기 손상액으로
잘못 읽었을** 가능성도 있음. 레이블 정확성에 직결되므로 원표와 대조해 확인함.

대사에 성공한 문서는 용량 때문에 삭제했으므로 검증 대상만 다시 받음.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")
v4 = __import__("14_extractor_v4")
m20 = __import__("20_retry_failed")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TOP_N = 6


def show_table(rows, items):
    print("      [표 내용]")
    for it in items[:22]:
        sec = it["section"] or "-"
        tot = it["total"]
        tot_s = f"{tot:>22,.0f}" if tot is not None else " " * 22
        print(f"        [{sec:<4}] {it['label'][:30]:<30} {tot_s}")


def main():
    panel = pd.read_csv(PROCESSED_DIR / "panel.csv", dtype={"corp_code": str})
    uni = pd.read_csv(PROCESSED_DIR / "universe_final.csv", dtype={"corp_code": str})
    yearmap = {r["corp_code"]: [int(y) for y in str(r["연도목록"]).split(",")
                                if y.strip().isdigit() and 2015 <= int(y) <= 2025]
               for _, r in uni.iterrows()}

    use = panel[(panel["상태"] == "사용가능") & (panel["손상액"] > 0)].copy()
    use["비율"] = use["손상액"] / use["무형자산"]
    top = use.sort_values("비율", ascending=False).head(TOP_N)

    print(f"손상액/무형자산 상위 {TOP_N}건 검증\n")
    for _, r in top.iterrows():
        cc, year = r["corp_code"], int(r["사업연도"])
        print("=" * 82)
        print(f"### {r['기업명']} {year}  비율 {r['비율']:.2f}배")
        print(f"    손상액 {r['손상액']:,.0f} / 무형자산 {r['무형자산']:,.0f}")
        print("=" * 82)

        anc = m20.anchors_wide(cc, yearmap.get(cc, []))
        cur, prev = anc.get(year, set()), anc.get(year - 1, set())

        for rep in v4.list_annual_reports(cc, year):
            rno = rep["rcept_no"]
            if v4.ensure_doc(rno) is None:
                continue
            doc_dir = v2.DOC_DIR / rno
            hit = False
            for src, html in v3.iter_tables_loose(doc_dir):
                rows = v2.parse_rows(html)
                if len(rows) < 3:
                    continue
                orientation, items = v2.flatten(rows)
                rec = v3.reconcile2(items, cur, prev)
                if not rec:
                    continue
                vals = v2.pull_values(items, rec["unit"])
                if vals["손상액"] is None:
                    continue
                print(f"  출처 {src} 배치{orientation} 단위{rec['unit']:,} 신뢰도{rec['신뢰도']}")
                print(f"      추출 손상액={vals['손상액']:,.0f} 상각액="
                      f"{vals['상각액']:,.0f}" if vals['상각액'] else "")
                show_table(rows, items)
                hit = True
                break
            if hit:
                break
        print()


if __name__ == "__main__":
    main()
