"""
[모집단 1단계] 전 상장사 업종코드 수집

기업개황 API는 기업당 1회 호출이 필요하므로 매번 조회하지 않고
한 번 훑어 CSV로 만들어 두고 이후에는 그 파일만 읽음(앱 속도·API 한도 대응).

중간에 끊겨도 이어서 받을 수 있도록 진행분을 계속 append 함.
"""

import csv
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DART_API_KEY", "").strip()

SRC = PROCESSED_DIR / "corp_codes.csv"
OUT = PROCESSED_DIR / "corp_industry.csv"
FIELDS = ["corp_code", "corp_name", "stock_code", "induty_code", "est_dt", "acc_mt", "status"]


def load_done():
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT, dtype=str)
    return set(df["corp_code"].dropna())


def main():
    corps = pd.read_csv(SRC, dtype=str)
    done = load_done()
    todo = corps[~corps["corp_code"].isin(done)]
    print(f"전체 {len(corps):,} / 완료 {len(done):,} / 남은 {len(todo):,}")

    new_file = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()

        for i, (_, r) in enumerate(todo.iterrows(), 1):
            cc = r["corp_code"]
            try:
                res = requests.get(
                    "https://opendart.fss.or.kr/api/company.json",
                    params={"crtfc_key": API_KEY, "corp_code": cc}, timeout=30).json()
                w.writerow({
                    "corp_code": cc,
                    "corp_name": res.get("corp_name") or r["corp_name"],
                    "stock_code": (res.get("stock_code") or r["stock_code"] or "").strip(),
                    "induty_code": (res.get("induty_code") or "").strip(),
                    "est_dt": res.get("est_dt") or "",
                    "acc_mt": res.get("acc_mt") or "",
                    "status": res.get("status") or "",
                })
            except Exception as e:
                w.writerow({"corp_code": cc, "corp_name": r["corp_name"],
                            "stock_code": r["stock_code"], "induty_code": "",
                            "est_dt": "", "acc_mt": "", "status": f"ERR:{type(e).__name__}"})
            if i % 200 == 0:
                f.flush()
                print(f"  진행 {i:,}/{len(todo):,}")
            time.sleep(0.05)

    print(f"완료 → {OUT}")


if __name__ == "__main__":
    main()
