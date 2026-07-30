"""
[A안 재작성] 본문 대사(reconciliation)로 검증하는 무형자산 주석 추출기

앞선 파서(09)는 '그럴듯한 표'를 집어놓고 그것이 맞는 표인지 확인할 방법이 없었음.
그 결과 당기/전기 혼동, 연결/별도 혼동, 단위 오인, 취득원가/장부금액 혼동이 발생했고
어느 값이 맞는지 사람이 눈으로 봐야만 알 수 있었음.

해결 방식:
  재무상태표 본문의 무형자산 장부금액(API로 100% 확보됨)을 기준점으로 삼아,
  주석 표에서 계산한 기말 장부금액이 그 금액과 일치할 때만 그 표를 채택함.
  단위(원/천원/백만원)도 어느 배수에서 일치하는지로 역산함.

  → 맞지 않는 표는 통과하지 못하므로 위 네 가지 오류가 한꺼번에 걸러짐.
  → 끝까지 일치하는 표가 없으면 '추출 실패'로 명시함. 틀린 값을 성공으로 위장하지 않음.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "data" / "raw" / "docs"
FS_DIR = BASE_DIR / "data" / "raw" / "fs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()

TARGETS = {
    "01204056": "하이브", "00260930": "에스엠", "00613318": "와이지엔터테인먼트",
    "00258689": "JYP Ent.", "01168684": "스튜디오드래곤", "00203315": "콘텐트리중앙",
    "00975290": "에이스토리", "00303794": "쇼박스", "01186404": "디앤씨미디어",
    "00140131": "키다리스튜디오",
}
YEARS = [2021, 2022, 2023, 2024, 2025]

# 주석 금액은 본문과 '반올림 오차만큼'만 달라야 함. 1% 같은 느슨한 허용치를 쓰면
# 잔액이 비슷한 전년도 표가 우연히 통과함(에스엠 2022가 실제로 그렇게 오매칭됐음).
# 허용치는 표시 단위의 반올림 폭으로 잡음.
UNITS = (1, 1_000, 1_000_000)


def tolerance_for(unit: int, anchor: float) -> float:
    return max(unit * 2.0, anchor * 1e-6)

# ---- 계정 식별 -------------------------------------------------------------
GOODWILL_IDS = {"dart_GoodwillGross", "ifrs-full_Goodwill", "ifrs_Goodwill"}
OTHER_INTAN_IDS = {
    "ifrs-full_IntangibleAssetsOtherThanGoodwill",
    "ifrs_IntangibleAssetsOtherThanGoodwill",
    "dart_OtherIntangibleAssetsGross",
}
COMBINED_IDS = {"ifrs-full_IntangibleAssetsAndGoodwill", "ifrs_IntangibleAssetsAndGoodwill"}

EXCLUDE = re.compile(r"대손")
P_IMPAIR = re.compile(r"손상")
P_AMORT = re.compile(r"상각")
P_CLOSING = re.compile(r"기말|당기말|기말잔액|장부금액$")
P_SEC_COST = re.compile(r"취득원가")
P_SEC_ACC = re.compile(r"누계|상각및손상|상각 및 손상")
P_SEC_BOOK = re.compile(r"장부금액|장부가액|순장부")
P_TOTAL = re.compile(r"^\s*(합\s*계|계|총\s*계)\s*$")
P_CONTENT = re.compile(r"판권|콘텐츠|컨텐츠|프로그램|영상|저작권|음원|음반")
P_CHANGE_ANY = re.compile(r"기초|기말|취득|처분|상각|손상|대체|증감")


# ============================ 공통 유틸 ====================================
def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&cr;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def to_num(s: str):
    s = (s or "").strip()
    if s in ("", "-", "–", "—"):
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace(" ", "")
    s = re.sub(r"[^\d.\-]", "", s)
    if s in ("", "-", "."):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


# ======================= 1) 본문 무형자산 기준점 ============================
def fetch_fs(corp_code: str, year: int) -> dict:
    """사업보고서 기준 전체 재무제표를 받아 캐시함."""
    FS_DIR.mkdir(parents=True, exist_ok=True)
    cache = FS_DIR / f"{corp_code}_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    payload = {}
    for fs_div in ("CFS", "OFS"):
        res = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
            params={"crtfc_key": API_KEY, "corp_code": corp_code, "bsns_year": str(year),
                    "reprt_code": "11011", "fs_div": fs_div},
            timeout=60,
        )
        payload = res.json()
        payload["_fs_div"] = fs_div
        if payload.get("status") == "000":
            break
        time.sleep(0.2)

    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def build_anchors() -> dict:
    """(기업, 사업연도) → 무형자산 장부금액 후보들.

    보고서 1건에 당기·전기·전전기 잔액이 함께 있으므로 한 번 받아 3개년을 채움.
    영업권 포함 여부가 회사마다 다르므로 '영업권 제외'와 '영업권 포함' 두 가지를 모두 보관함.
    """
    anchors = {}
    for corp_code, name in TARGETS.items():
        for year in YEARS:
            payload = fetch_fs(corp_code, year)
            if payload.get("status") != "000":
                continue

            buckets = {0: {"other": 0.0, "gw": 0.0, "comb": 0.0, "named": 0.0},
                       1: {"other": 0.0, "gw": 0.0, "comb": 0.0, "named": 0.0},
                       2: {"other": 0.0, "gw": 0.0, "comb": 0.0, "named": 0.0}}

            for it in payload.get("list", []):
                if it.get("sj_div") != "BS":
                    continue
                aid = (it.get("account_id") or "").strip()
                nm = (it.get("account_nm") or "").strip()

                if aid in GOODWILL_IDS:
                    key = "gw"
                elif aid in OTHER_INTAN_IDS:
                    key = "other"
                elif aid in COMBINED_IDS:
                    key = "comb"
                elif "무형자산" in nm and "상각" not in nm:
                    key = "named"  # 표준계정코드 미사용 케이스
                else:
                    continue

                for tag, off in (("thstrm_amount", 0), ("frmtrm_amount", 1), ("bfefrmtrm_amount", 2)):
                    v = to_num(it.get(tag))
                    if v is not None:
                        buckets[off][key] += v

            for off, b in buckets.items():
                y = year - off
                if y not in [yy for yy in range(min(YEARS) - 2, max(YEARS) + 1)]:
                    continue
                cands = set()
                base = b["other"] or b["comb"] or b["named"]
                if base:
                    cands.add(round(base, 2))
                    if b["gw"]:
                        cands.add(round(base + b["gw"], 2))
                if b["comb"]:
                    cands.add(round(b["comb"], 2))
                if b["named"]:
                    cands.add(round(b["named"], 2))
                    if b["gw"]:
                        cands.add(round(b["named"] + b["gw"], 2))
                cands = {c for c in cands if c > 0}
                if not cands:
                    continue
                key = (name, y)
                # 최신 보고서 값을 우선하되, 없던 연도는 채움
                anchors.setdefault(key, set()).update(cands)

    return anchors


# ======================= 2) 주석 표 파싱 ====================================
def parse_rows(table_html: str):
    rows = []
    for r in re.findall(r"<TR[\s\S]*?</TR>", table_html, flags=re.IGNORECASE):
        cells = [strip_tags(c) for c in re.findall(r"<T[DH][\s\S]*?</T[DH]>", r, flags=re.IGNORECASE)]
        if any(c != "" for c in cells):
            rows.append(cells)
    return rows


def is_intangible_table(html: str) -> bool:
    if "무형자산" not in html and not P_CONTENT.search(html):
        return False
    if "기초" not in html or "기말" not in html:
        return False
    return bool(P_AMORT.search(html) or P_IMPAIR.search(html))


def flatten(rows):
    """배치와 무관하게 (섹션, 라벨, 합계값, 콘텐츠열값) 목록으로 눌러냄.

    표 첫 행이 '(단위: 천원)' 같은 안내문인 경우가 많음. 이를 헤더로 잘못 잡으면
    합계 열을 찾지 못해 모든 열을 더하게 되고, 합계 열까지 포함돼 금액이 정확히 2배가 됨
    (하이브가 5개 연도 전부 이 이유로 대사에 실패했음). 셀이 3개 이상인 첫 행을 헤더로 봄.
    """
    hdr_i = 0
    for i, r in enumerate(rows[:4]):
        if len(r) >= 3:
            hdr_i = i
            break
    header = rows[hdr_i]
    rows = rows[hdr_i:]

    row_labels = " ".join(r[0] for r in rows[1:] if r)
    orientation = "A" if len(P_CHANGE_ANY.findall(row_labels)) >= len(P_CHANGE_ANY.findall(" ".join(header))) else "B"

    items = []
    if orientation == "A":
        total_idx = next((i for i, h in enumerate(header) if P_TOTAL.match(h)), -1)
        content_idx = next((i for i, h in enumerate(header) if P_CONTENT.search(h)), -1)
        section = None
        for cells in rows[1:]:
            label = cells[0]
            nums = [to_num(c) for c in cells[1:]]
            has_num = any(v not in (None, 0) for v in nums)
            if not has_num and len(cells) <= 2:
                if P_SEC_ACC.search(label):
                    section = "ACC"
                elif P_SEC_COST.search(label):
                    section = "COST"
                elif P_SEC_BOOK.search(label):
                    section = "BOOK"
                continue
            if total_idx >= 0 and total_idx < len(cells):
                total = to_num(cells[total_idx])
            else:
                vals = [v for v in nums if v is not None]
                total = sum(vals) if vals else None
            content = to_num(cells[content_idx]) if 0 <= content_idx < len(cells) else None
            items.append({"section": section, "label": label, "total": total, "content": content})
    else:
        total_row = next((c for c in rows[1:] if c and P_TOTAL.match(c[0])), None)
        content_rows = [c for c in rows[1:] if c and P_CONTENT.search(c[0])]
        body = [total_row] if total_row else rows[1:]
        for i, h in enumerate(header):
            if i == 0 or not h:
                continue
            vals = [to_num(c[i]) for c in body if c and i < len(c)]
            vals = [v for v in vals if v is not None]
            total = sum(vals) if vals else None
            cvals = [to_num(c[i]) for c in content_rows if i < len(c)]
            cvals = [v for v in cvals if v is not None]
            items.append({"section": None, "label": h, "total": total,
                          "content": sum(cvals) if cvals else None})

    return orientation, items


def closing_candidates(items):
    """기말 장부금액이 될 수 있는 값들을 모음. 대사로 어느 것이 맞는지 판정함."""
    cands = []
    closings = {}
    for it in items:
        if it["total"] is None:
            continue
        if P_CLOSING.search(it["label"]):
            sec = it["section"] or "NA"
            closings.setdefault(sec, []).append(it["total"])
            cands.append(it["total"])

    # 취득원가 기말 - 상각·손상누계액 기말 = 장부금액 기말
    if "COST" in closings and "ACC" in closings:
        cands.append(closings["COST"][-1] - abs(closings["ACC"][-1]))
    if len(cands) >= 2:
        cands.append(sum(cands[:2]))
    return [c for c in cands if c and c > 0]


def reconcile(items, anchor_set):
    """단위를 바꿔가며 본문 금액과 맞는 조합을 찾음. 맞으면 (배수, 대사금액) 반환."""
    for unit in UNITS:
        for cand in closing_candidates(items):
            v = cand * unit
            for a in anchor_set:
                if abs(v - a) <= tolerance_for(unit, a):
                    return unit, v, a
    return None


def pull_values(items, unit):
    """대사가 통과한 표에서 손상액·상각액을 뽑음."""
    out = {"손상액": None, "상각액": None, "손상액_콘텐츠": None, "상각손상합산": False}
    for it in items:
        label = it["label"]
        if EXCLUDE.search(label) or it["total"] is None:
            continue
        is_imp, is_amo = bool(P_IMPAIR.search(label)), bool(P_AMORT.search(label))
        if is_imp and is_amo:
            out["상각손상합산"] = True
            out["상각액"] = abs(it["total"]) * unit
        elif is_imp:
            out["손상액"] = abs(it["total"]) * unit
            if it["content"] is not None:
                out["손상액_콘텐츠"] = abs(it["content"]) * unit
        elif is_amo and it["section"] != "COST":
            if out["상각액"] is None:
                out["상각액"] = abs(it["total"]) * unit
    return out


def iter_tables(doc_dir: Path):
    main = [p for p in doc_dir.glob("*.xml") if "_" not in p.stem]
    attach = sorted(p for p in doc_dir.glob("*.xml") if "_" in p.stem)
    for path in main + attach:
        text = read_text(path)
        for m in re.finditer(r"<TABLE[\s\S]*?</TABLE>", text, flags=re.IGNORECASE):
            if is_intangible_table(m.group(0)):
                yield path.name, m.group(0)


def extract(rcept_no: str, anchor_set):
    doc_dir = DOC_DIR / rcept_no
    if not doc_dir.exists() or not anchor_set:
        return None
    for src, html in iter_tables(doc_dir):
        rows = parse_rows(html)
        if len(rows) < 3:
            continue
        orientation, items = flatten(rows)
        rec = reconcile(items, anchor_set)
        if not rec:
            continue
        unit, matched, anchor = rec
        vals = pull_values(items, unit)
        vals.update({"배치": orientation, "단위": unit, "출처": src,
                     "대사금액": matched, "본문금액": anchor})
        return vals
    return None


# ============================== 실행 =======================================
if __name__ == "__main__":
    print("[1/2] 재무상태표 기준점 구축 중…")
    anchors = build_anchors()
    print(f"      기준점 확보: {len(anchors)} firm-year")

    pilot = pd.read_csv(PROCESSED_DIR / "note_parse_pilot.csv", dtype={"접수번호": str})
    print("[2/2] 주석 대사 추출 중…")

    records = []
    for _, r in pilot.iterrows():
        name, year, rno = r["기업"], int(r["사업연도"]), r["접수번호"]
        if not isinstance(rno, str) or not rno:
            continue
        anchor_set = anchors.get((name, year), set())
        res = extract(rno, anchor_set)

        row = {"기업": name, "사업연도": year}
        if res is None:
            row["대사"] = "실패"
            row["본문무형자산"] = max(anchor_set) if anchor_set else None
        else:
            # 대사가 통과한 변동표에 손상 행이 없다는 것은 그 해 손상이 없었다는 뜻으로 봄.
            # 단 상각과 합산 표시된 경우는 분리 불가이므로 결측으로 남김.
            impair = res["손상액"]
            if impair is None and not res["상각손상합산"]:
                impair = 0.0
            row.update({
                "대사": "성공", "본문무형자산": res["본문금액"],
                "단위": res["단위"], "배치": res["배치"],
                "주석기말": res["대사금액"], "합산표시": res["상각손상합산"],
                "손상액": impair, "상각액": res["상각액"],
            })
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "note_extract_reconciled.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print("\n" + df.to_string(index=False))

    n = len(df)
    ok = df[df["대사"] == "성공"]
    print("\n===== 요약 =====")
    print(f"대상 firm-year        : {n}건")
    print(f"본문 기준점 확보      : {df['본문무형자산'].notna().sum()}건")
    print(f"주석-본문 대사 성공   : {len(ok)}건 ({len(ok)/n:.1%})")
    if len(ok):
        sep = ok[~ok["합산표시"].fillna(False).astype(bool)]
        print(f"  이 중 손상 분리 가능 : {len(sep)}건")
        print(f"  손상액 수치 확보     : {ok['손상액'].notna().sum()}건")
        print(f"  상각액 수치 확보     : {ok['상각액'].notna().sum()}건")
