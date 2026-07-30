"""
[시범 수집 2단계] 손상차손 '값'까지 확인해 레이블 생성 가능성을 검증함

02번 스크립트는 계정명에 '손상'이 들어간 항목을 셌지만, 부분일치라
'대손상각비'(대손+상각비, 매출채권 관련)까지 섞여 들어옴. 무형자산과 무관하므로
반드시 배제해야 함.

또한 DART 보고서는 당기/전기/전전기 3개년을 함께 담고 있으므로,
7개 보고서만으로도 그보다 넓은 기간의 값을 확보할 수 있음.
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PILOT_DIR = BASE_DIR / "data" / "raw" / "pilot"

NAMES = {
    "01204056": "하이브", "00260930": "에스엠", "00613318": "와이지엔터테인먼트",
    "00258689": "JYP Ent.", "01168684": "스튜디오드래곤", "00203315": "콘텐트리중앙",
    "00975290": "에이스토리", "00303794": "쇼박스", "01186404": "디앤씨미디어",
    "00140131": "키다리스튜디오",
}

# 무형자산 손상으로 인정할 계정 (판권·콘텐츠자산도 무형자산의 일종임)
TARGET_PAT = re.compile(r"(무형자산|판권|콘텐츠).*손상차손")
# '대손상각비'는 매출채권 관련이므로 명시적으로 배제함
EXCLUDE_PAT = re.compile(r"대손")


def to_num(v):
    if v is None:
        return None
    v = str(v).replace(",", "").strip()
    if v in ("", "-"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


rows = []
for path in sorted(PILOT_DIR.glob("*.json")):
    corp_code, report_year = path.stem.split("_")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "000":
        continue

    for it in payload.get("list", []):
        nm = (it.get("account_nm") or "").strip()
        if EXCLUDE_PAT.search(nm) or not TARGET_PAT.search(nm):
            continue
        # 보고서 1건에 당기/전기/전전기 3개년이 함께 들어있음
        for tag, offset in (("thstrm_amount", 0), ("frmtrm_amount", 1), ("bfefrmtrm_amount", 2)):
            amt = to_num(it.get(tag))
            if amt is None:
                continue
            rows.append({
                "기업": NAMES[corp_code],
                "사업연도": int(report_year) - offset,
                "계정명": nm,
                "금액": amt,
                "출처보고서": report_year,
            })

df = pd.DataFrame(rows)

if df.empty:
    print("!! 무형자산 손상차손 값이 하나도 잡히지 않음 — 설계 재검토 필요")
    raise SystemExit

# 동일 기업-연도가 여러 보고서에서 중복 수집되므로 최신 보고서 값을 채택함
df = (df.sort_values("출처보고서", ascending=False)
        .drop_duplicates(subset=["기업", "사업연도", "계정명"], keep="first"))

agg = (df.groupby(["기업", "사업연도"], as_index=False)
         .agg(손상차손합계=("금액", "sum"), 계정수=("계정명", "count")))

print("===== 무형자산 손상차손 관측치 (대손상각비 배제 후) =====")
print(agg.sort_values(["기업", "사업연도"]).to_string(index=False))

nonzero = agg[agg["손상차손합계"] > 0]
print(f"\n총 관측 firm-year: {len(agg)}건")
print(f"이 중 손상차손 > 0: {len(nonzero)}건")
print(f"금액 0으로 표시된 건: {len(agg) - len(nonzero)}건")

print("\n===== 수집된 계정명 종류 =====")
print(df["계정명"].value_counts().to_string())

print("\n===== 기업별 손상 인식 연도 수 =====")
print(nonzero.groupby("기업").size().sort_values(ascending=False).to_string())
