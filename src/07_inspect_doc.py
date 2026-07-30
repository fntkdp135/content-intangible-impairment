"""
[A안 시범 2단계] 원문 XML 안에서 무형자산 주석 표의 위치와 형태 확인
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOC_DIR = BASE_DIR / "data" / "raw" / "docs" / "20240318000577"


def read_text(path: Path) -> str:
    """DART 원문은 대개 EUC-KR임. 실패 시 UTF-8로 폴백함."""
    raw = path.read_bytes()
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


for path in sorted(DOC_DIR.glob("*.xml")):
    text = read_text(path)
    tables = re.findall(r"<TABLE[\s\S]*?</TABLE>", text, flags=re.IGNORECASE)

    # 무형자산 변동내역표는 '기초'와 '손상' 또는 '상각'이 함께 나오는 표임
    cand = [
        t for t in tables
        if "무형자산" in t and ("기초" in t or "취득" in t) and ("상각" in t or "손상" in t)
    ]

    print(f"\n{'=' * 76}")
    print(f"{path.name}  | 길이 {len(text):,}자 | 전체 표 {len(tables)}개 | 무형자산 변동표 후보 {len(cand)}개")
    print(f"{'=' * 76}")

    if cand:
        sample = cand[0]
        # 표를 텍스트로 눌러서 미리보기
        rows = re.findall(r"<TR[\s\S]*?</TR>", sample, flags=re.IGNORECASE)
        print(f"[첫 후보 표: {len(rows)}행]")
        for r in rows[:12]:
            cells = re.findall(r"<T[DH][\s\S]*?</T[DH]>", r, flags=re.IGNORECASE)
            vals = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip() for c in cells]
            vals = [v for v in vals if v != ""]
            if vals:
                print("   | " + " | ".join(vals))
