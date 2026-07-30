"""
[보완 1·2] 실패 firm-year 재실행 + 기준점 후보 확장

(1) 대량 수집 중 원문 다운로드가 일시적으로 실패한 건이 섞여 있음.
    와이지엔... 아니라 와이티엔 2025는 지금 코드로 오차 0.00%인데도 실패로 기록돼 있었음.
    원문이 캐시돼 있어 재실행 비용은 거의 없으므로 실패 건만 다시 돌림.

(2) 초록뱀미디어(2.83%)·미스터블루(3.42%)처럼 아깝게 안 맞는 건이 있음.
    주석 합계에 건설중인무형자산 등이 포함/제외되어 본문 계정 조합과 범위가 어긋난 것으로 보임.
    본문의 무형자산 관련 계정을 모두 모아 조합 후보를 넓힘.
    후보가 늘면 오매칭 위험이 커지지만, 기초·기말 2점 대사가 그 위험을 막아 줌.
"""

import csv
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2 = __import__("10_note_extractor_reconciled")
v3 = __import__("12_extractor_v3")
v4 = __import__("14_extractor_v4")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

IMP_IN = PROCESSED_DIR / "impairment.csv"
IMP_OUT = PROCESSED_DIR / "impairment_v2.csv"

# 무형자산 성격의 재무상태표 계정 (한글명 기준 폭넓게)
P_INTAN_ACC = re.compile(r"무형자산|영업권|개발비|판권|콘텐츠|저작권|산업재산권|상표권|회원권|소프트웨어")
P_NOT_ACC = re.compile(r"상각|손상|사용권|리스")


def intan_components(payload):
    """당기/전기/전전기별로 무형자산 관련 계정을 개별 수집함."""
    out = {0: {}, 1: {}, 2: {}}
    for it in payload.get("list", []):
        if it.get("sj_div") != "BS":
            continue
        nm = (it.get("account_nm") or "").strip()
        aid = (it.get("account_id") or "").strip()
        is_intan = (aid in v2.GOODWILL_IDS or aid in v2.OTHER_INTAN_IDS
                    or aid in v2.COMBINED_IDS
                    or (P_INTAN_ACC.search(nm) and not P_NOT_ACC.search(nm)))
        if not is_intan:
            continue
        for tag, off in (("thstrm_amount", 0), ("frmtrm_amount", 1), ("bfefrmtrm_amount", 2)):
            v = v2.to_num(it.get(tag))
            if v and v > 0:
                key = f"{aid}|{nm}"
                out[off][key] = max(out[off].get(key, 0.0), v)
    return out


def anchors_wide(corp_code, years):
    """개별 계정값 + 전체합 + (전체합 - 각 계정)을 후보로 둠."""
    anc = {}
    for year in years:
        payload = v2.fetch_fs(corp_code, year)
        if payload.get("status") != "000":
            continue
        comps = intan_components(payload)
        for off, d in comps.items():
            y = year - off
            vals = list(d.values())
            if not vals:
                continue
            cands = set(round(v, 2) for v in vals)
            total = round(sum(vals), 2)
            cands.add(total)
            for v in vals:
                r = round(total - v, 2)
                if r > 0:
                    cands.add(r)
            anc.setdefault(y, set()).update(c for c in cands if c > 0)
    return anc


def main():
    imp = pd.read_csv(IMP_IN, dtype={"corp_code": str})
    uni = pd.read_csv(PROCESSED_DIR / "universe_final.csv", dtype={"corp_code": str})
    yearmap = {r["corp_code"]: [int(y) for y in str(r["연도목록"]).split(",")
                                if y.strip().isdigit() and 2015 <= int(y) <= 2025]
               for _, r in uni.iterrows()}

    failed = imp[imp["대사"] != "성공"].copy()
    print(f"재실행 대상 {len(failed)}건 ({failed['corp_code'].nunique()}개사)")

    fixed = {}
    anc_cache = {}
    for i, (_, r) in enumerate(failed.iterrows(), 1):
        cc, year, name = r["corp_code"], int(r["사업연도"]), r["기업명"]
        if cc not in anc_cache:
            anc_cache[cc] = anchors_wide(cc, yearmap.get(cc, []))
        anc = anc_cache[cc]
        cur, prev = anc.get(year, set()), anc.get(year - 1, set())
        if not cur:
            continue

        for rep in v4.list_annual_reports(cc, year):
            rno = rep["rcept_no"]
            if v4.ensure_doc(rno) is None:
                continue
            res = v3.extract(rno, cur, prev)
            if res is None:
                continue
            impair = res["손상액"]
            if impair is None and not res["상각손상합산"]:
                impair = 0.0
            fixed[(cc, year)] = {
                "대사": "성공", "신뢰도": res["신뢰도"],
                "사용보고서": "정정" if "정정" in rep["report_nm"] else "원본",
                "본문무형자산": res["본문금액"], "주석기말": res["주석기말"],
                "합산표시": res["상각손상합산"], "손상액": impair, "상각액": res["상각액"],
            }
            if res["신뢰도"] == "확정":
                break
        if i % 20 == 0:
            print(f"  {i}/{len(failed)} … 회수 {len(fixed)}건", flush=True)

    # 원본에 회수분 반영
    out = imp.copy()
    for (cc, year), vals in fixed.items():
        mask = (out["corp_code"] == cc) & (out["사업연도"] == year)
        for k, v in vals.items():
            out.loc[mask, k] = v
    out.to_csv(IMP_OUT, index=False, encoding="utf-8-sig")

    n = len(out)
    ok = out[out["대사"] == "성공"]
    print(f"\n===== 재실행 결과 =====")
    print(f"회수 {len(fixed)}건")
    print(f"대사 성공 {len(ok)}/{n} ({len(ok)/n:.1%})  (이전 75.8%)")
    print(f"  두 지점 확정 {(ok['신뢰도'] == '확정').sum()}건")
    print(f"저장 → {IMP_OUT}")

    print("\n===== 회수된 기업 =====")
    if fixed:
        rec = pd.DataFrame([{"기업명": imp[(imp.corp_code == cc) & (imp.사업연도 == y)].iloc[0]["기업명"],
                             "연도": y} for (cc, y) in fixed])
        print(rec.groupby("기업명").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
