"""
[시범 수집 4단계] 현금흐름표 전문 확인

04번에서 대형사에 손상·상각 계정이 안 보였는데, 원인이
'실제로 없음'인지 '재무제표 본문에 세부항목을 표시하지 않고 주석으로 뺌'인지
현금흐름표 전체를 그대로 출력해 확인함.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PILOT_DIR = BASE_DIR / "data" / "raw" / "pilot"

CASES = [("01168684", "스튜디오드래곤", 2023), ("01204056", "하이브", 2023)]


def to_num(v):
    v = str(v or "").replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


for corp_code, name, year in CASES:
    payload = json.loads((PILOT_DIR / f"{corp_code}_{year}.json").read_text(encoding="utf-8"))
    cf = [it for it in payload["list"] if it.get("sj_div") == "CF"]
    print(f"\n{'=' * 78}\n### {name} {year} 현금흐름표 — 총 {len(cf)}개 계정\n{'=' * 78}")
    for it in cf:
        amt = to_num(it.get("thstrm_amount"))
        amt_s = f"{amt:>22,.0f}" if amt is not None else " " * 22
        print(f"  {(it.get('account_nm') or '').strip():<40} {amt_s}")
