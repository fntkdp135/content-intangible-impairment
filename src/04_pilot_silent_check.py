"""
[시범 수집 3단계] 손상 계정이 전혀 없는 기업의 원인 진단

하이브·에스엠·와이지·콘텐트리중앙·디앤씨미디어는 현금흐름표에 손상차손이
한 번도 나타나지 않음. 두 가지 해석이 가능하며 이를 구분해야 함.

  (a) 실제로 손상 인식이 없었다            → 레이블 0으로 써도 됨
  (b) 콘텐츠 자산을 무형자산으로 분류하지 않는다 → 애초에 모집단에서 제외해야 함

재무상태표의 자산 분류와 현금흐름표의 상각 관련 계정을 함께 확인해 판별함.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PILOT_DIR = BASE_DIR / "data" / "raw" / "pilot"

SILENT = {
    "01204056": "하이브", "00260930": "에스엠", "00613318": "와이지엔터테인먼트",
    "00203315": "콘텐트리중앙", "01186404": "디앤씨미디어",
}
# 비교군: 손상차손이 정상적으로 잡히는 기업
CONTROL = {"01168684": "스튜디오드래곤"}

KEYWORDS = ("무형", "콘텐츠", "판권", "영업권", "제작", "상각", "손상", "재고")


def to_num(v):
    v = str(v or "").replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def dump(corp_code, name, year=2023):
    path = PILOT_DIR / f"{corp_code}_{year}.json"
    if not path.exists():
        print(f"\n### {name} ({year}) — 파일 없음")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "000":
        print(f"\n### {name} ({year}) — status={payload.get('status')}")
        return

    print(f"\n{'=' * 70}\n### {name} ({year}년 사업보고서)\n{'=' * 70}")
    for div_label, div in (("[재무상태표]", "BS"), ("[현금흐름표]", "CF"), ("[손익계산서]", "IS")):
        hits = [
            it for it in payload["list"]
            if it.get("sj_div") == div and any(k in (it.get("account_nm") or "") for k in KEYWORDS)
        ]
        if not hits:
            continue
        print(f"\n{div_label}")
        for it in hits:
            amt = to_num(it.get("thstrm_amount"))
            amt_s = f"{amt:>20,.0f}" if amt is not None else " " * 20
            print(f"  {(it.get('account_nm') or '').strip():<28} {amt_s}")


for code, name in {**CONTROL, **SILENT}.items():
    dump(code, name)
