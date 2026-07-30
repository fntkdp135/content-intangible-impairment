"""
[A안 본작업] 무형자산 주석 변동표에서 상각액·손상액 추출

주석표는 회사·연도마다 형식이 달라 두 가지 배치를 모두 처리함.
  (A) 자산종류가 '열', 변동항목(기초·취득·상각·손상·기말)이 '행'   ← 스튜디오드래곤 형태
  (B) 자산종류가 '행', 변동항목이 '열'                            ← 반대 형태

전체 변동표를 완벽히 재구성하는 대신, 이 프로젝트에 필요한 값만 정확히 뽑는 방식을 택함.
  · 손상액(레이블의 근거)  · 상각액(유효상각률 계산용)  · 취득액  · 기말잔액

단위(원/천원/백만원)는 표 앞 문구에서 읽어 원 단위로 환산함. 재무상태표 API 값과
대조하려면 단위가 맞아야 하기 때문임.
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "data" / "raw" / "docs"

# ---- 행/열 라벨 판정용 패턴 -------------------------------------------------
# '대손상각비'가 '손상'에 걸리는 것을 막기 위해 대손 계열은 항상 먼저 배제함
EXCLUDE = re.compile(r"대손")
P_IMPAIR = re.compile(r"손상")
P_AMORT = re.compile(r"상각")
P_ACQUIRE = re.compile(r"취득|증가|신규")
P_OPENING = re.compile(r"기초|전기말|기초잔액")
P_CLOSING = re.compile(r"기말|당기말|기말잔액")
P_CHANGE_ANY = re.compile(r"기초|기말|취득|처분|상각|손상|대체|증감")

# 취득원가 블록과 상각·손상누계액 블록을 구분하기 위한 섹션 헤더
P_SEC_COST = re.compile(r"취득원가|장부금액|총장부금액")
P_SEC_ACC = re.compile(r"누계액|상각누계|손상차손누계")

P_TOTAL = re.compile(r"^\s*(합\s*계|계|소\s*계|총\s*계)\s*$")
P_CONTENT = re.compile(r"판권|콘텐츠|컨텐츠|프로그램|영상|저작권|음원|음반")

UNIT_MAP = {"백만원": 1_000_000, "천원": 1_000, "원": 1}


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
    """회계 표기를 숫자로 변환함. 괄호는 음수, '-'는 0으로 처리함."""
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


def parse_rows(table_html: str):
    rows = []
    for r in re.findall(r"<TR[\s\S]*?</TR>", table_html, flags=re.IGNORECASE):
        cells = [strip_tags(c) for c in re.findall(r"<T[DH][\s\S]*?</T[DH]>", r, flags=re.IGNORECASE)]
        if any(c != "" for c in cells):
            rows.append(cells)
    return rows


def detect_unit(preceding: str) -> int:
    """표 직전 문구에서 '(단위: 천원)' 같은 표기를 찾음. 없으면 원 단위로 봄."""
    tail = preceding[-600:]
    m = re.findall(r"단위\s*[:：]?\s*\(?\s*(백만원|천원|원)", tail)
    if m:
        return UNIT_MAP[m[-1]]
    return 1


def is_intangible_table(html: str) -> bool:
    if "무형자산" not in html and not P_CONTENT.search(html):
        return False
    if not (P_OPENING.search(html) and P_CLOSING.search(html)):
        return False
    return bool(P_AMORT.search(html) or P_IMPAIR.search(html))


def pick_total_index(header: list) -> int:
    """헤더에서 '합계' 열의 위치를 찾음. 없으면 -1."""
    for i, h in enumerate(header):
        if P_TOTAL.match(h):
            return i
    return -1


def extract_orientation_a(rows):
    """자산종류=열, 변동항목=행인 표에서 값을 뽑음."""
    header = rows[0]
    total_idx = pick_total_index(header)
    content_idx = next((i for i, h in enumerate(header) if P_CONTENT.search(h)), -1)

    out = {"상각액": None, "손상액": None, "취득액": None,
           "기말_총액": None, "손상액_콘텐츠": None, "상각손상합산": False}
    section = None

    for cells in rows[1:]:
        label = cells[0]
        if EXCLUDE.search(label):
            continue

        # 값이 없는 단독 셀은 섹션 헤더로 봄 (예: '취득원가:', '상각 및 손상차손누계액:')
        numeric = [to_num(c) for c in cells[1:]]
        has_number = any(v is not None and v != 0 for v in numeric)
        if not has_number and len(cells) <= 2:
            if P_SEC_ACC.search(label):
                section = "ACC"
            elif P_SEC_COST.search(label):
                section = "COST"
            continue

        def value_at(idx):
            if idx < 0 or idx >= len(cells):
                return None
            return to_num(cells[idx])

        # 합계열이 없으면 숫자 셀을 모두 더해 총액을 만듦
        if total_idx >= 0:
            total_val = value_at(total_idx)
        else:
            vals = [v for v in numeric if v is not None]
            total_val = sum(vals) if vals else None

        content_val = value_at(content_idx) if content_idx >= 0 else None

        is_imp = bool(P_IMPAIR.search(label))
        is_amo = bool(P_AMORT.search(label))

        if is_imp and is_amo:
            out["상각손상합산"] = True
            if out["상각액"] is None:
                out["상각액"] = abs(total_val) if total_val is not None else None
        elif is_imp:
            out["손상액"] = abs(total_val) if total_val is not None else None
            if content_val is not None:
                out["손상액_콘텐츠"] = abs(content_val)
        elif is_amo and section != "COST":
            out["상각액"] = abs(total_val) if total_val is not None else None
        elif P_ACQUIRE.search(label) and section != "ACC":
            if out["취득액"] is None:
                out["취득액"] = total_val
        elif P_CLOSING.search(label) and section != "ACC":
            out["기말_총액"] = total_val

    return out


def extract_orientation_b(rows):
    """자산종류=행, 변동항목=열인 표에서 값을 뽑음."""
    header = rows[0]
    col = {}
    for i, h in enumerate(header):
        if EXCLUDE.search(h):
            continue
        if P_IMPAIR.search(h) and P_AMORT.search(h):
            col.setdefault("상각손상", i)
        elif P_IMPAIR.search(h):
            col.setdefault("손상", i)
        elif P_AMORT.search(h):
            col.setdefault("상각", i)
        elif P_ACQUIRE.search(h):
            col.setdefault("취득", i)
        elif P_CLOSING.search(h):
            col.setdefault("기말", i)

    out = {"상각액": 0.0, "손상액": 0.0, "취득액": 0.0,
           "기말_총액": 0.0, "손상액_콘텐츠": None, "상각손상합산": "상각손상" in col}

    total_row = None
    content_rows = []
    body = rows[1:]
    for cells in body:
        if not cells:
            continue
        if P_TOTAL.match(cells[0]):
            total_row = cells
        if P_CONTENT.search(cells[0]):
            content_rows.append(cells)

    def grab(cells, key):
        i = col.get(key, -1)
        if i < 0 or i >= len(cells):
            return None
        return to_num(cells[i])

    # 합계행이 있으면 그것을, 없으면 각 자산종류 행을 합산함
    targets = [total_row] if total_row else body
    for key, out_key in (("상각", "상각액"), ("손상", "손상액"),
                         ("취득", "취득액"), ("기말", "기말_총액")):
        vals = [grab(c, key) for c in targets if c]
        vals = [abs(v) if out_key in ("상각액", "손상액") else v
                for v in vals if v is not None]
        out[out_key] = sum(vals) if vals else None

    if content_rows and "손상" in col:
        vals = [grab(c, "손상") for c in content_rows]
        vals = [abs(v) for v in vals if v is not None]
        out["손상액_콘텐츠"] = sum(vals) if vals else None

    if "상각손상" in col:
        vals = [grab(c, "상각손상") for c in targets if c]
        vals = [abs(v) for v in vals if v is not None]
        out["상각액"] = sum(vals) if vals else out["상각액"]

    return out


def extract_from_table(table_html: str, preceding: str):
    rows = parse_rows(table_html)
    if len(rows) < 3:
        return None

    header = rows[0]
    row_labels = " ".join(r[0] for r in rows[1:] if r)

    # 변동항목이 행에 있으면 A배치, 헤더에 있으면 B배치로 판정함
    score_a = len(P_CHANGE_ANY.findall(row_labels))
    score_b = len(P_CHANGE_ANY.findall(" ".join(header)))
    orientation = "A" if score_a >= score_b else "B"

    out = extract_orientation_a(rows) if orientation == "A" else extract_orientation_b(rows)
    out["배치"] = orientation
    out["단위"] = detect_unit(preceding)
    return out


def iter_candidate_tables(doc_dir: Path):
    """본문 문서를 먼저 보고, 없으면 첨부(감사보고서)를 봄.

    사업보고서 본문은 '연결재무제표 주석'이 '별도 재무제표 주석'보다 앞에 오므로
    등장 순서가 빠른 표를 연결 기준으로 간주함.
    """
    main = [p for p in doc_dir.glob("*.xml") if "_" not in p.stem]
    attach = sorted(p for p in doc_dir.glob("*.xml") if "_" in p.stem)

    for path in main + attach:
        text = read_text(path)
        for m in re.finditer(r"<TABLE[\s\S]*?</TABLE>", text, flags=re.IGNORECASE):
            html = m.group(0)
            if is_intangible_table(html):
                yield path.name, m.start(), html, text[max(0, m.start() - 800):m.start()]


def extract_for_report(rcept_no: str):
    """한 보고서에서 가장 신뢰할 만한 무형자산 변동표 1건을 골라 값을 반환함."""
    doc_dir = DOC_DIR / rcept_no
    if not doc_dir.exists():
        return None

    for src, pos, html, preceding in iter_candidate_tables(doc_dir):
        res = extract_from_table(html, preceding)
        if not res:
            continue
        # 손상액 또는 상각액 중 하나라도 잡힌 표만 채택함
        if res["손상액"] is None and res["상각액"] is None:
            continue
        res["출처파일"] = src
        res["표위치"] = pos
        return res
    return None


if __name__ == "__main__":
    pilot = pd.read_csv(BASE_DIR / "data" / "processed" / "note_parse_pilot.csv",
                        dtype={"접수번호": str})
    records = []
    for _, r in pilot.iterrows():
        if not isinstance(r["접수번호"], str) or not r["접수번호"]:
            continue
        res = extract_for_report(r["접수번호"])
        row = {"기업": r["기업"], "사업연도": r["사업연도"], "접수번호": r["접수번호"]}
        if res is None:
            row["추출"] = "실패"
        else:
            u = res["단위"]
            row.update({
                "추출": "성공",
                "배치": res["배치"],
                "단위": u,
                "합산표시": res["상각손상합산"],
                "상각액": None if res["상각액"] is None else res["상각액"] * u,
                "손상액": None if res["손상액"] is None else res["손상액"] * u,
                "취득액": None if res["취득액"] is None else res["취득액"] * u,
                "기말총액": None if res["기말_총액"] is None else res["기말_총액"] * u,
            })
        records.append(row)

    df = pd.DataFrame(records)
    out_path = BASE_DIR / "data" / "processed" / "note_extract_pilot.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print(df.to_string(index=False))
    print(f"\n추출 성공 {(df['추출'] == '성공').sum()}/{len(df)}건")
    if "손상액" in df:
        print(f"손상액 확보   : {df['손상액'].notna().sum()}건")
        print(f"상각액 확보   : {df['상각액'].notna().sum()}건")
    print(f"\n저장 → {out_path}")
