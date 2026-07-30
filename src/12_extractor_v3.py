"""
[A안 v3] 검출은 느슨하게, 검증은 두 지점 대사로

v2(10번)의 한계
  · 표를 '무형자산·상각·손상' 키워드로 먼저 걸러냈는데, 정작 무형자산 변동표에
    그 단어가 없는 경우가 있음(하이브 2023은 이 때문에 표가 0개로 잡혔음)
  · 키다리스튜디오는 전기 비교표만 잡히고 당기 표를 놓침

v3의 방침
  · 검출 단계에서는 키워드를 거의 보지 않고 '기초~기말' 구조를 가진 표를 모두 후보로 둠
  · 대신 검증을 강화함. 기말이 당기말 잔액과 맞는 것에 더해,
    **기초가 전기말 잔액과도 맞는지**까지 확인함(전기이월 대사).
    두 지점이 동시에 맞는 표는 사실상 그 회사의 그 해 무형자산 변동표일 수밖에 없음.
  · 두 지점 대사가 되면 '확정', 기말만 맞으면 '보통'으로 신뢰도를 구분해 기록함.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

P_OPEN_TOK = re.compile(r"기초|전기말|기초잔액")
P_CLOSE_TOK = re.compile(r"기말|당기말|기말잔액")
P_INTAN_HINT = re.compile(r"무형|영업권|판권|콘텐츠|저작권|소프트웨어|상표권|회원권")


def opening_candidates(items):
    """기초 장부금액이 될 수 있는 값들."""
    cands, opens = [], {}
    for it in items:
        if it["total"] is None:
            continue
        if P_OPEN_TOK.search(it["label"]) and not P_CLOSE_TOK.search(it["label"]):
            sec = it["section"] or "NA"
            opens.setdefault(sec, []).append(it["total"])
            cands.append(it["total"])
    if "COST" in opens and "ACC" in opens:
        cands.append(opens["COST"][0] - abs(opens["ACC"][0]))
    return [c for c in cands if c and c > 0]


def match_any(value, anchor_set, unit):
    for a in anchor_set:
        if abs(value - a) <= v2.tolerance_for(unit, a):
            return a
    return None


def reconcile2(items, anchor_cur, anchor_prev):
    """기말(당기말)과 기초(전기말)를 함께 맞춰봄. 두 지점이 맞으면 신뢰도 '확정'."""
    closes = v2.closing_candidates(items)
    opens = opening_candidates(items)
    best = None

    for unit in v2.UNITS:
        for c in closes:
            hit_c = match_any(c * unit, anchor_cur, unit)
            if hit_c is None:
                continue
            for o in opens:
                if match_any(o * unit, anchor_prev, unit) is not None:
                    return {"unit": unit, "close": c * unit, "anchor": hit_c, "신뢰도": "확정"}
            if best is None:
                best = {"unit": unit, "close": c * unit, "anchor": hit_c, "신뢰도": "보통"}
    return best


def iter_tables_loose(doc_dir: Path):
    """키워드 대신 '기초~기말' 구조만 보고 후보를 넓게 모음."""
    main = [p for p in doc_dir.glob("*.xml") if "_" not in p.stem]
    attach = sorted(p for p in doc_dir.glob("*.xml") if "_" in p.stem)
    for path in main + attach:
        text = v2.read_text(path)
        for m in re.finditer(r"<TABLE[\s\S]*?</TABLE>", text, flags=re.IGNORECASE):
            html = m.group(0)
            if not (P_OPEN_TOK.search(html) and P_CLOSE_TOK.search(html)):
                continue
            yield path.name, html


def extract(rcept_no: str, anchor_cur, anchor_prev):
    doc_dir = v2.DOC_DIR / rcept_no
    if not doc_dir.exists() or not anchor_cur:
        return None

    fallback = None
    for src, html in iter_tables_loose(doc_dir):
        rows = v2.parse_rows(html)
        if len(rows) < 3:
            continue
        orientation, items = v2.flatten(rows)
        rec = reconcile2(items, anchor_cur, anchor_prev)
        if not rec:
            continue

        vals = v2.pull_values(items, rec["unit"])
        vals.update({
            "배치": orientation, "단위": rec["unit"], "출처": src,
            "주석기말": rec["close"], "본문금액": rec["anchor"], "신뢰도": rec["신뢰도"],
            "무형힌트": bool(P_INTAN_HINT.search(html)),
        })
        # 두 지점이 맞고 무형자산 관련 단어도 있으면 즉시 확정함
        if rec["신뢰도"] == "확정" and vals["무형힌트"]:
            return vals
        if fallback is None or (rec["신뢰도"] == "확정" and fallback["신뢰도"] != "확정"):
            fallback = vals
    return fallback


if __name__ == "__main__":
    print("[1/2] 재무상태표 기준점 구축 중…")
    anchors = v2.build_anchors()

    pilot = pd.read_csv(PROCESSED_DIR / "note_parse_pilot.csv", dtype={"접수번호": str})
    print("[2/2] 주석 2점 대사 추출 중…")

    records = []
    for _, r in pilot.iterrows():
        name, year, rno = r["기업"], int(r["사업연도"]), r["접수번호"]
        if not isinstance(rno, str) or not rno:
            continue
        cur = anchors.get((name, year), set())
        prev = anchors.get((name, year - 1), set())
        res = extract(rno, cur, prev)

        row = {"기업": name, "사업연도": year}
        if res is None:
            row.update({"대사": "실패", "본문무형자산": max(cur) if cur else None})
        else:
            impair = res["손상액"]
            if impair is None and not res["상각손상합산"]:
                impair = 0.0
            row.update({
                "대사": "성공", "신뢰도": res["신뢰도"], "본문무형자산": res["본문금액"],
                "단위": res["단위"], "배치": res["배치"], "주석기말": res["주석기말"],
                "무형힌트": res["무형힌트"], "합산표시": res["상각손상합산"],
                "손상액": impair, "상각액": res["상각액"],
            })
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "note_extract_v3.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print("\n" + df.to_string(index=False))

    n, ok = len(df), df[df["대사"] == "성공"]
    print("\n===== 요약 =====")
    print(f"대상 firm-year      : {n}건")
    print(f"대사 성공           : {len(ok)}건 ({len(ok)/n:.1%})")
    if len(ok):
        conf = ok[ok["신뢰도"] == "확정"]
        print(f"  두 지점 대사(확정) : {len(conf)}건")
        print(f"  기말만 대사(보통)  : {len(ok) - len(conf)}건")
        print(f"  무형자산 단어 확인 : {ok['무형힌트'].sum()}건")
        print(f"  손상액 확보        : {ok['손상액'].notna().sum()}건")
        print(f"  상각액 확보        : {ok['상각액'].notna().sum()}건")
