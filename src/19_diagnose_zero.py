"""
[진단] 전 연도 대사 실패 기업의 공통 원인 추적

기업 단위로 0%가 나오는 것은 무작위 결측이 아니라 구조적 원인이 있다는 뜻임.
대사는 '기준점(본문 무형자산)'과 '주석 표' 두 가지가 모두 있어야 성립하므로,
어느 쪽이 없는지부터 가름.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")
v4 = __import__("14_extractor_v4")
main = __import__("18_collect_main")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ZERO = ["(주)초록뱀미디어", "(주)팬엔터테인먼트", "(주)티비씨", "(주)케이엔엔",
        "(주)와이티엔", "미스터블루(주)", "(주)더콘텐츠온"]


def main_():
    uni = pd.read_csv(PROCESSED_DIR / "universe_final.csv", dtype={"corp_code": str})
    fin = pd.read_csv(PROCESSED_DIR / "financials.csv", dtype={"corp_code": str})

    for name in ZERO:
        r = uni[uni["기업명"] == name]
        if r.empty:
            print(f"\n### {name} — 모집단에 없음")
            continue
        r = r.iloc[0]
        cc = r["corp_code"]
        years = [int(y) for y in str(r["연도목록"]).split(",")
                 if y.strip().isdigit() and 2015 <= int(y) <= 2025]

        print(f"\n{'=' * 78}\n### {name}  ({cc})  대상 {len(years)}개년\n{'=' * 78}")

        # (1) 본문 재무 상태
        f = fin[fin["corp_code"] == cc]
        if f.empty:
            print("  [본문] 재무제표 수집 자체가 없음")
        else:
            g = f[["사업연도", "재무제표", "자산총계", "무형자산", "영업권", "매출액"]].sort_values("사업연도")
            print("  [본문] 최근 4개년:")
            for _, x in g.tail(4).iterrows():
                intan = x["무형자산"]
                print(f"    {int(x['사업연도'])} {x['재무제표']} 자산={x['자산총계']:,.0f} "
                      f"무형={'없음' if pd.isna(intan) else format(intan, ',.0f')}")

        # (2) 기준점 생성 여부
        anc = main.anchors_for(cc, years)
        have = {y: len(v) for y, v in anc.items() if v}
        print(f"  [기준점] 생성된 연도 {len(have)}개 → {dict(list(sorted(have.items()))[-5:])}")

        if not have:
            print("  ▶ 원인: 본문에서 무형자산 잔액을 못 잡아 대사 자체가 시작되지 않음")
            continue

        # (3) 주석 표 상태 — 최근 성공 가능성이 가장 높은 연도로 확인
        y = max(have)
        reports = v4.list_annual_reports(cc, y)
        print(f"  [주석] {y}년 사업보고서 {len(reports)}건")
        for rep in reports[:2]:
            rno = rep["rcept_no"]
            d = v4.ensure_doc(rno)
            if d is None:
                print(f"    {rno} 원문 다운로드 실패")
                continue
            tabs = list(v3.iter_tables_loose(d))
            hinted = [t for t in tabs if v3.P_INTAN_HINT.search(t[1])]
            print(f"    {rno} {rep['report_nm'].strip()} | 후보표 {len(tabs)} / 무형관련 {len(hinted)}")

            cur, prev = anc.get(y, set()), anc.get(y - 1, set())
            best = None
            for src, html in hinted:
                rows = v2.parse_rows(html)
                if len(rows) < 3:
                    continue
                _, items = v2.flatten(rows)
                for c in v2.closing_candidates(items):
                    for u in v2.UNITS:
                        for a in cur:
                            e = abs(c * u - a) / a
                            if best is None or e < best[0]:
                                best = (e, c * u, a, u)
            if best:
                print(f"      최근접: 주석 {best[1]:,.0f} vs 본문 {best[2]:,.0f} "
                      f"(오차 {best[0]:.2%}, 단위 {best[3]:,})")
            else:
                print("      무형자산 관련 후보 표에서 기말값을 찾지 못함")


if __name__ == "__main__":
    main_()
