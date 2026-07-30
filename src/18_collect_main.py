"""
[본 수집] 모집단 78개사 × 2015~2025 사업연도

모집단 확정 근거
  · 업종코드 핵심 대역(영상·방송프로그램 591 / 음반 592 / 방송 60x / 창작예술 90x) 72개사
  · 웹툰·웹소설 IP 6개사 — 콘텐츠 자산화·상각 구조가 드라마·영화와 동일하므로 포함
  · 게임(582x)은 별도 프로젝트 주제이고 개발비 자산 성격이 달라 제외
  · 교육출판·플랫폼(네이버·카카오)은 무형자산 대부분이 M&A 영업권이라 제외
  · 상장폐지 기업도 포함함. 손상 예측 모델에서 사라진 기업을 빼면 생존편향이 생김

무형자산 비중을 모집단 필터로 쓰려던 계획은 폐기함. 실제로 재보니 이 비율은
콘텐츠 자산 집약도가 아니라 영업권 비중을 재고 있어, 와이지·쇼박스 같은 실제
엔터사를 탈락시키고 네이버·카카오·유통업체를 통과시켰음. 설명변수로만 사용함.

수집 산출물
  · financials.csv  — firm-year 재무 항목(본문 API)
  · impairment.csv  — firm-year 손상·상각액(주석 2점 대사)
중간에 끊겨도 이어받을 수 있게 firm-year 단위로 append 함.
원문 문서는 용량이 크므로 추출에 성공하면 삭제함(실패 건은 진단용으로 남김).
"""

import csv
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")
v4 = __import__("14_extractor_v4")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

YEAR_MIN, YEAR_MAX = 2015, 2025
KEEP_DOCS = False  # 성공 시 원문 삭제 (2GB 이상 누적 방지)

FIN_OUT = PROCESSED_DIR / "financials.csv"
IMP_OUT = PROCESSED_DIR / "impairment.csv"

FIN_FIELDS = ["corp_code", "기업명", "사업연도", "재무제표", "자산총계", "부채총계", "자본총계",
              "무형자산", "영업권", "재고자산", "매출액", "영업이익", "당기순이익", "영업활동현금흐름"]
IMP_FIELDS = ["corp_code", "기업명", "사업연도", "대사", "신뢰도", "사용보고서",
              "본문무형자산", "주석기말", "합산표시", "손상액", "상각액"]

# 계정 매핑 (표준계정코드 우선, 한글명 폴백)
ACC = {
    "자산총계": ({"ifrs-full_Assets", "ifrs_Assets"}, ("자산총계",)),
    "부채총계": ({"ifrs-full_Liabilities", "ifrs_Liabilities"}, ("부채총계",)),
    "자본총계": ({"ifrs-full_Equity", "ifrs_Equity"}, ("자본총계",)),
    "재고자산": ({"ifrs-full_Inventories", "ifrs_Inventories"}, ("재고자산",)),
    "매출액": ({"ifrs-full_Revenue", "ifrs_Revenue", "dart_OperatingRevenue"}, ("매출액", "영업수익")),
    "영업이익": ({"dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"}, ("영업이익",)),
    "당기순이익": ({"ifrs-full_ProfitLoss", "ifrs_ProfitLoss"}, ("당기순이익",)),
    "영업활동현금흐름": ({"ifrs-full_CashFlowsFromUsedInOperatingActivities"}, ("영업활동현금흐름", "영업활동으로 인한 현금흐름")),
}


def build_universe():
    g = pd.read_csv(PROCESSED_DIR / "universe_gated.csv", dtype={"corp_code": str})
    u = pd.read_csv(PROCESSED_DIR / "universe.csv", dtype={"corp_code": str})
    core = g[g["구분"] == "핵심"]["corp_code"]
    webtoon_names = ["탑코미디어", "키다리스튜디오", "디앤씨미디어", "미스터블루", "핑거스토리", "밀리의서재"]
    web = g[(g["구분"] == "경계") & g["기업명"].str.contains("|".join(webtoon_names))]["corp_code"]
    keep = set(core) | set(web)
    uni = u[u["corp_code"].isin(keep)].copy()
    uni.to_csv(PROCESSED_DIR / "universe_final.csv", index=False, encoding="utf-8-sig")
    return uni


def fin_items(payload):
    """재무제표 payload에서 당기/전기/전전기 주요 계정을 뽑음."""
    out = {0: {}, 1: {}, 2: {}}
    for it in payload.get("list", []):
        aid = (it.get("account_id") or "").strip()
        nm = (it.get("account_nm") or "").strip()
        sj = it.get("sj_div")

        key = None
        if sj == "BS":
            if aid in v2.GOODWILL_IDS:
                key = "영업권"
            elif aid in v2.OTHER_INTAN_IDS or aid in v2.COMBINED_IDS:
                key = "무형자산"
            elif "무형자산" in nm and "상각" not in nm:
                key = "무형자산"
        if key is None:
            for k, (ids, names) in ACC.items():
                if aid in ids or nm in names:
                    key = k
                    break
        if key is None:
            continue

        for tag, off in (("thstrm_amount", 0), ("frmtrm_amount", 1), ("bfefrmtrm_amount", 2)):
            v = v2.to_num(it.get(tag))
            if v is None:
                continue
            if key in ("무형자산", "영업권"):
                out[off][key] = out[off].get(key, 0.0) + v
            else:
                out[off].setdefault(key, v)
    return out


def anchors_for(corp_code, years):
    """기업 단위 기준점 집합. 보고서 1건이 3개년을 담으므로 겹쳐서 채움."""
    anc = {}
    for year in years:
        payload = v2.fetch_fs(corp_code, year)
        if payload.get("status") != "000":
            continue
        items = fin_items(payload)
        for off, d in items.items():
            y = year - off
            base, gw = d.get("무형자산", 0.0), d.get("영업권", 0.0)
            cands = {round(v, 2) for v in (base, base + gw, gw) if v and v > 0}
            if cands:
                anc.setdefault(y, set()).update(cands)
    return anc


def load_done(path, fields):
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype={"corp_code": str})
    return set(zip(df["corp_code"], df["사업연도"].astype(int)))


def main():
    uni = build_universe()
    print(f"모집단 {len(uni)}개사 / 잠재 firm-year {uni['보고연도수'].sum():,}건")

    fin_done = load_done(FIN_OUT, FIN_FIELDS)
    imp_done = load_done(IMP_OUT, IMP_FIELDS)
    print(f"기수집 재무 {len(fin_done):,} / 손상 {len(imp_done):,}")

    fin_new = not FIN_OUT.exists()
    imp_new = not IMP_OUT.exists()

    with FIN_OUT.open("a", newline="", encoding="utf-8-sig") as ff, \
         IMP_OUT.open("a", newline="", encoding="utf-8-sig") as fi:
        fw = csv.DictWriter(ff, fieldnames=FIN_FIELDS)
        iw = csv.DictWriter(fi, fieldnames=IMP_FIELDS)
        if fin_new:
            fw.writeheader()
        if imp_new:
            iw.writeheader()

        for n, (_, r) in enumerate(uni.iterrows(), 1):
            cc, name = r["corp_code"], r["기업명"]
            years = [int(y) for y in str(r["연도목록"]).split(",")
                     if y.strip().isdigit() and YEAR_MIN <= int(y) <= YEAR_MAX]
            if not years:
                continue

            print(f"[{n}/{len(uni)}] {name} ({len(years)}개년)", flush=True)
            anc = anchors_for(cc, years)

            # --- 재무 항목 ---
            for year in years:
                if (cc, year) in fin_done:
                    continue
                payload = v2.fetch_fs(cc, year)
                if payload.get("status") != "000":
                    continue
                d = fin_items(payload).get(0, {})
                fw.writerow({"corp_code": cc, "기업명": name, "사업연도": year,
                             "재무제표": payload.get("_fs_div"),
                             **{k: d.get(k) for k in FIN_FIELDS[4:]}})
            ff.flush()

            # --- 주석 손상·상각 ---
            for year in years:
                if (cc, year) in imp_done:
                    continue
                cur, prev = anc.get(year, set()), anc.get(year - 1, set())
                row = {"corp_code": cc, "기업명": name, "사업연도": year, "대사": "실패"}
                if cur:
                    used_dirs = []
                    for rep in v4.list_annual_reports(cc, year):
                        rno = rep["rcept_no"]
                        if v4.ensure_doc(rno) is None:
                            continue
                        used_dirs.append(v2.DOC_DIR / rno)
                        res = v3.extract(rno, cur, prev)
                        if res is None:
                            continue
                        impair = res["손상액"]
                        if impair is None and not res["상각손상합산"]:
                            impair = 0.0
                        row.update({
                            "대사": "성공", "신뢰도": res["신뢰도"],
                            "사용보고서": "정정" if "정정" in rep["report_nm"] else "원본",
                            "본문무형자산": res["본문금액"], "주석기말": res["주석기말"],
                            "합산표시": res["상각손상합산"],
                            "손상액": impair, "상각액": res["상각액"],
                        })
                        if res["신뢰도"] == "확정":
                            break
                    if not KEEP_DOCS and row["대사"] == "성공":
                        for d_ in used_dirs:
                            shutil.rmtree(d_, ignore_errors=True)
                iw.writerow(row)
                time.sleep(0.05)
            fi.flush()

    print("\n수집 완료")
    for path in (FIN_OUT, IMP_OUT):
        df = pd.read_csv(path, dtype={"corp_code": str})
        print(f"  {path.name}: {len(df):,}행")


if __name__ == "__main__":
    main()
