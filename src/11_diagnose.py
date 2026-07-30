"""
[진단] 대사 실패 원인 추적

대사에 실패한 기업-연도에 대해 후보 표를 모두 펼쳐 보고,
기말 후보값이 본문 무형자산과 어떤 배율로 어긋나는지 확인함.
배율이 깔끔한 수(1000 등)면 단위 문제, 어중간하면 표 선택 자체가 잘못된 것임.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
mod = __import__("10_note_extractor_reconciled")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CASES = [("하이브", 2023), ("하이브", 2025), ("와이지엔터테인먼트", 2023), ("키다리스튜디오", 2023)]


def main():
    anchors = mod.build_anchors()
    pilot = pd.read_csv(PROCESSED_DIR / "note_parse_pilot.csv", dtype={"접수번호": str})

    for name, year in CASES:
        row = pilot[(pilot["기업"] == name) & (pilot["사업연도"] == year)]
        if row.empty:
            continue
        rno = row.iloc[0]["접수번호"]
        anchor_set = anchors.get((name, year), set())

        print(f"\n{'#' * 78}")
        print(f"# {name} {year}  (접수 {rno})")
        print(f"# 본문 무형자산 후보: {sorted(f'{a:,.0f}' for a in anchor_set)}")
        print(f"{'#' * 78}")

        doc_dir = mod.DOC_DIR / rno
        for i, (src, html) in enumerate(mod.iter_tables(doc_dir)):
            rows = mod.parse_rows(html)
            if len(rows) < 3:
                continue
            orientation, items = mod.flatten(rows)
            cands = mod.closing_candidates(items)

            print(f"\n--- 표 #{i} ({src}) 배치={orientation} 행수={len(rows)}")
            print(f"    헤더: {' | '.join(rows[0][:9])}")
            labels = [it['label'] for it in items][:14]
            print(f"    라벨: {' / '.join(labels)}")
            print(f"    기말후보: {[f'{c:,.0f}' for c in cands[:6]]}")

            if anchor_set and cands:
                a = max(anchor_set)
                ratios = sorted({round(a / c, 4) for c in cands if c}, key=abs)[:6]
                print(f"    본문/후보 배율: {ratios}")


if __name__ == "__main__":
    main()
