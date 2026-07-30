"""
[모집단 3단계] 무형자산 비중 2차 게이트

업종코드만으로는 같은 코드 안에 자산 구조가 전혀 다른 기업이 섞임
(포털·정보서비스 대역에 보안업체·유통업체가 함께 들어옴).
콘텐츠 무형자산 손상 모델은 무형자산이 유의미한 기업에만 적용 가능하므로,
실제 재무제표에서 무형자산 비중을 재서 판정함.

기업명으로 골라내지 않고 숫자로 판정하는 이유는, 150개사를 넘어가면
사람이 전수 확인할 수 없고 판단 근거도 남지 않기 때문임.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TOTAL_ASSET_IDS = {"ifrs-full_Assets", "ifrs_Assets"}


def latest_snapshot(corp_code: str, years: list):
    """가장 최근 보고연도의 무형자산·자산총계를 가져옴."""
    for year in sorted(years, reverse=True)[:3]:
        payload = v2.fetch_fs(corp_code, year)
        if payload.get("status") != "000":
            continue

        intan = gw = assets = 0.0
        for it in payload.get("list", []):
            if it.get("sj_div") != "BS":
                continue
            aid = (it.get("account_id") or "").strip()
            nm = (it.get("account_nm") or "").strip()
            v = v2.to_num(it.get("thstrm_amount"))
            if v is None:
                continue
            if aid in TOTAL_ASSET_IDS or nm in ("자산총계", "자산 총계"):
                assets = max(assets, v)
            elif aid in v2.GOODWILL_IDS:
                gw += v
            elif aid in v2.OTHER_INTAN_IDS or aid in v2.COMBINED_IDS:
                intan += v
            elif "무형자산" in nm and "상각" not in nm:
                intan = max(intan, v)

        if assets > 0:
            return year, intan + gw, assets
    return None, None, None


def main():
    uni = pd.read_csv(PROCESSED_DIR / "universe.csv", dtype={"corp_code": str, "종목코드": str})
    rows = []
    for i, (_, r) in enumerate(uni.iterrows(), 1):
        years = [int(y) for y in str(r["연도목록"]).split(",") if y.strip().isdigit()]
        year, intan, assets = latest_snapshot(r["corp_code"], years)
        ratio = (intan / assets) if (assets and assets > 0) else None
        rows.append({
            "기업명": r["기업명"], "corp_code": r["corp_code"], "업종코드": r["업종코드"],
            "구분": r["구분"], "업종": r["업종"], "보고연도수": r["보고연도수"],
            "기준연도": year, "무형자산": intan, "자산총계": assets, "무형비중": ratio,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(uni)}")

    df = pd.DataFrame(rows)
    df["무형비중"] = pd.to_numeric(df["무형비중"], errors="coerce")
    df["게이트"] = df["무형비중"].apply(
        lambda v: "통과" if pd.notna(v) and v >= 0.05 else ("미달" if pd.notna(v) else "데이터없음"))
    df = df.sort_values(["구분", "무형비중"], ascending=[True, False])
    df.to_csv(PROCESSED_DIR / "universe_gated.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n===== 게이트 결과 (무형자산/자산총계 5% 기준) =====")
    print(pd.crosstab(df["구분"], df["게이트"]).to_string())

    for grp in ("핵심", "경계"):
        sub = df[df["구분"] == grp]
        print(f"\n===== [{grp}] {len(sub)}개사 =====")
        show = sub[["기업명", "업종코드", "무형비중", "보고연도수", "게이트"]].head(40)
        print(show.to_string(index=False))

    passed = df[df["게이트"] == "통과"]
    print(f"\n최종 통과 {len(passed)}개사 / 잠재 firm-year {passed['보고연도수'].sum():,}건")
    print(f"저장 → {PROCESSED_DIR / 'universe_gated.csv'}")


if __name__ == "__main__":
    main()
