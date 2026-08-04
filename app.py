"""
콘텐츠 무형자산 손상징후 분석 — 엔터·미디어 상장사

DART 사업보고서 주석에서 무형자산 손상액을 추출해 데이터셋을 구축하고,
손상 인식을 예측할 수 있는지 검증한 프로젝트. 톤은 전체 개조식으로 통일함.

각 섹션은 '한눈에'(비전문가도 읽히는 요약) → '상세'(기술·회계적 서술) 2단 구조로 배치함.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
RES = BASE / "data" / "results"

st.set_page_config(page_title="콘텐츠 무형자산 손상징후 분석", page_icon="📉", layout="wide")

# 다크 배경에서 대비가 확보되도록 밝기를 올린 팔레트
ACCENT = "#5B9BD5"
ACCENT_SOFT = "#3D6E9E"
WARN = "#E0736F"
MUTED = "#6E7A8C"
GOLD = "#C9A13B"   # 본문 강조(볼드)용 — 짙은 금색
RED = "#E4635E"    # 부정적 결론 강조용
BG = "#0F1420"
PANEL = "#1A2233"
TEXT = "#E4E9F2"
GRID = "#2A3446"

# Plotly 기본 템플릿을 다크로 고정함. 배경은 투명하게 두어 페이지와 이어지게 함
pio.templates["impair_dark"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, size=13),
    title=dict(font=dict(color=TEXT, size=15)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    legend=dict(font=dict(color=TEXT)),
    hoverlabel=dict(bgcolor=PANEL, font=dict(color=TEXT)),
))
pio.templates.default = "plotly_dark+impair_dark"

st.markdown(f"""
<style>
.stApp {{ background-color:{BG}; }}
.lead {{ font-size:1.28rem; line-height:1.62; font-weight:600; color:{TEXT};
        border-left:5px solid {ACCENT}; padding:0.55rem 0 0.55rem 0.95rem;
        margin:0.2rem 0 1.1rem 0; background:linear-gradient(90deg,#18202F 0%,rgba(24,32,47,0) 85%); }}
.lead b {{ color:{GOLD}; }}
.lead .red, .lead b.red {{ color:{RED}; }}
.plain {{ font-size:1.0rem; line-height:1.72; color:#C9D2E0; }}
.plain b {{ color:{GOLD}; }}
.plain .red {{ color:{RED}; }}
.tag {{ display:inline-block; background:#22304A; color:{ACCENT}; font-size:0.78rem;
       font-weight:700; padding:0.16rem 0.6rem; border-radius:0.7rem; margin-bottom:0.45rem;
       letter-spacing:0.02em; }}
.bigq {{ font-size:1.72rem; line-height:1.45; font-weight:700; color:{TEXT};
       text-align:center; padding:1.5rem 1rem 1.35rem 1rem; margin:0.2rem 0 1.4rem 0;
       background:linear-gradient(180deg,#18202F 0%,#131A28 100%);
       border:1px solid #2A3446; border-radius:0.7rem; }}
.bigq .q {{ color:{GOLD}; }}
.concl {{ font-size:1.02rem; line-height:1.68; color:#D6DDE9; margin:1.1rem 0 0.4rem 0;
       padding:0.85rem 1.1rem; background:#171D2B; border-left:5px solid {RED};
       border-radius:0.35rem; }}
.concl b {{ color:{GOLD}; }}
.concl .red {{ color:{RED}; }}
[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid #26314A;
       border-radius:0.6rem; padding:0.7rem 0.9rem; }}
[data-testid="stMetricLabel"] p {{ color:#9BA8BD !important; }}
hr {{ border-color:#26314A !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:0.4rem; }}
.stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:0.5rem 0.5rem 0 0;
       padding:0.45rem 1.0rem; }}
.stTabs [aria-selected="true"] {{ background:#22304A; }}
</style>
""", unsafe_allow_html=True)


def lead(text):
    st.markdown(f'<div class="lead">{text}</div>', unsafe_allow_html=True)


def plain(text):
    st.markdown(f'<div class="plain">{text}</div>', unsafe_allow_html=True)


def tag(text):
    st.markdown(f'<span class="tag">{text}</span>', unsafe_allow_html=True)


def detail_header(text):
    st.divider()
    st.markdown(f"#### 상세 — {text}")


# ============================== 데이터 로드 ==============================
@st.cache_data
def load():
    panel = pd.read_csv(PROC / "panel.csv", dtype={"corp_code": str})
    feat = pd.read_csv(PROC / "features_v2.csv", dtype={"corp_code": str})
    uni = pd.read_csv(PROC / "universe_final.csv", dtype={"corp_code": str})
    tests = pd.read_csv(RES / "tests.csv")
    models = pd.read_csv(RES / "models.csv")
    coef = pd.read_csv(RES / "coef.csv")
    pipe = json.loads((RES / "pipeline.json").read_text(encoding="utf-8"))
    panel = panel.merge(uni[["corp_code", "업종"]], on="corp_code", how="left")
    return panel, feat, uni, tests, models, coef, pipe


panel, feat, uni, tests, models, coef, pipe = load()
conf = panel[(panel["상태"] == "사용가능") & (panel["신뢰도"] == "확정")].copy()
conf["손상비율"] = conf["손상액"] / conf["무형자산"]


def won(v):
    if pd.isna(v):
        return "-"
    return f"{v / 1e8:,.1f}억"


# ============================== 헤더 ==============================
st.title("콘텐츠 무형자산 손상징후 분석")
st.markdown(
    "엔터·미디어·콘텐츠 상장사의 **무형자산 손상차손을 사업보고서 주석에서 추출**해 데이터셋을 "
    "구축하고, 재무데이터로 손상 인식을 예측할 수 있는지 검증한 프로젝트."
)

st.markdown("")

# ===== 요약 블록 — 탭을 열지 않아도 질문·방법·결론이 보이도록 상단에 고정함 =====
st.markdown(
    '<div class="bigq">실적이 나빠진 콘텐츠 기업은,<br>'
    '<span class="q">이듬해 무형자산 손상을 인식하게 되는가?</span></div>',
    unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    tag("왜 이 질문을 했나")
    plain("""
콘텐츠 제작비는 먼저 나가고 성패는 한참 뒤에 드러남.
그 사이 재무제표에는 아직 검증되지 않은 자산이 남음.<br><br>
실적이 안 좋을 때, 손상을 인식하면 장부에 남지만
<b>인식하지 않기로 한다면 그 판단은 흔적을 남기지 않음.</b>
감사인이 볼 수 있는 것은 기록된 것들 뿐.<br><br>
때문에, <b>"이미 공시된 실적 지표만으로 손상 가능성을 미리 가늠할 수 있다면,
감사 계획 단계에서 주의의 우선순위를 세울 수 있지 않을까."</b>라는 질문으로
본 프로젝트를 시작함.
    """)
with s2:
    tag("어떻게 확인했나")
    plain("""
엔터·미디어 상장사 <b>78개사 11개년</b>.<br><br>
공개 API에 나오지 않는 손상 금액을 사업보고서 주석에서 확보하고,
<b>본문 잔액과 맞아떨어지는 값만 채택</b>해 390건을 검증함.<br><br>
그 위에서 <b>가설 4개</b>를 통계 검정한 뒤,
살아남은 변수만 모델에 투입함.
    """)
with s3:
    tag("무엇을 알게 됐나")
    plain("""
4개 가설 중 3개가 기각되고 <b>'수익성' 측면만</b> 남음.
그러나 변수 8개를 넣은 모델도 <b>"전기에 손상을 인식했는가"라는 단일 조건</b>을 넘지는 못함.<br><br>
즉, <span class="red"><b>실적 지표로 손상을 미리 예측하는 것은 성립하지 않았음.</b></span><br><br>
(대신 확인한 것 — 전기 손상 이력의 예측력은 <b>딱 1년까지만</b> 유효함(2년 뒤에는 사라짐).
주목 기간의 실증 근거는 됨.)
    """)

st.markdown(
    '<div class="concl">실적 악화는 손상의 배경이었으나 <span class="red">예측의 근거는 되지 못했음.</span> '
    '손상 인식은 재무데이터로 환원되지 않는 판단의 영역이며, '
    '이것이 감사인의 <b>\'전문가적 판단이 필요\'</b>한 이유임.</div>',
    unsafe_allow_html=True)

st.markdown("")
c = st.columns(5)
c[0].metric("분석 대상", f"{pipe['모집단_확정']}개사", delta="2015–2025 · 11개년", delta_color="off")
c[1].metric("검증 데이터", f"{pipe['확정표본']} firm-year", delta=f"수집 {pipe['수집_firmyear']:,}건 중", delta_color="off")
c[2].metric("검정한 가설", "4개", delta="H1–H4", delta_color="off")
c[3].metric("살아남은 신호", "1개", delta="수익성(영업이익률)", delta_color="off")
c[4].metric("예측 모델", "미성립", delta="단일 조건 기준선 미달", delta_color="inverse")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 문제의식", "② 데이터 확보 과정", "③ 데이터 탐색", "④ 가설검정·모델링", "⑤ 한계와 결론"])

# ============================== ① 문제의식 ==============================
with tab1:
    lead("콘텐츠 자산의 회수가능액이 장부금액에 미달하면 손상을 인식해야 함. "
         "그러나 <b>인식하지 않기로 한 판단은 어떤 계정에도 기록을 남기지 않음.</b>")

    a, b, c3 = st.columns(3)
    with a:
        tag("검증 대상이 '사실'이 아님")
        plain("""
현금·매출채권은 실재성을 확인하면 종결되는 항목임.
반면 <b>콘텐츠 자산의 회수가능액은 경영진의 미래현금흐름 추정에 의존함.</b><br><br>
검증 대상이 사실이 아니라 <b>가정</b>이라는 점에서 성격이 다름.
        """)
    with b:
        tag("엔터·미디어의 구조적 특성")
        plain("""
제작비 투입과 수익 실현 사이의 시차로 <b>자산이 선행 계상되는 구간</b>이 구조적으로 발생함.<br><br>
흥행 실패 시 회수가능액이 급격히 하락하는 반면,
손상 인식은 곧바로 당기손익에 반영되므로 <b>인식 시점을 이연할 유인이 존재함.</b>
        """)
    with c3:
        tag("검증한 것")
        plain("""
손상 인식 <b>이전 연도의 재무데이터만으로</b> 인식 여부를 예측할 수 있는지 확인함.<br><br>
예측이 성립한다면 <b>인식했어야 함에도 인식하지 않은 기업</b>을 식별하는 데까지 확장 가능하다고 보았음.
        """)

    detail_header("회계·감사 관점")
    d1, d2 = st.columns([3, 2])
    with d1:
        st.markdown("""
**감사에서 가장 다루기 어려운 것은 '사실'이 아니라 '추정'임**
- 현금·매출채권은 실재성을 확인하면 끝나지만, 무형자산 손상은 경영진이 미래현금흐름을
  어떻게 가정하느냐에 따라 금액이 달라짐
- K-IFRS 1036은 매 보고기간말 손상징후를 검토하도록 요구하나, "징후가 있는가"의 1차 판단
  주체가 경영진임
- 따라서 손상을 인식하지 않기로 한 결정은 회계처리 기록에 남지 않음.
  감사인은 "무엇이 기록되었는가"가 아니라 "**무엇이 기록되지 않았는가**"를 의심해야 함

**엔터·미디어 산업에서 이 문제가 가장 첨예함**
- 콘텐츠제작비·판권은 자산 비중이 크지만, 흥행 실패 시 회수가능액이 급격히 0에 수렴함
- 제작비 투입(자산화)과 수익 실현(방영·개봉) 사이에 시차가 있어, 자산은 이미 쌓였는데
  실적은 아직 안 나온 구간이 구조적으로 발생함
- 손상을 인식하면 당기순손실로 직결되므로 인식 시점을 늦출 유인이 명확히 존재함
        """)
    with d2:
        st.info("""
**연구질문**

손상차손이 실제로 인식되기 이전 연도의 재무데이터만으로 그 인식을 예측할 수 있는가?

그리고 그 모델이 "인식했어야 하는데 하지 않은 기업"을 지목할 수 있는가?
        """)
        st.markdown("**설계의 핵심 아이디어 — 오탐을 산출물로 재해석**")
        st.markdown("""
손상 '미인식'은 정답 레이블이 존재하지 않음(관측 불가능).
따라서 관측 가능한 실제 인식 사례로 학습한 뒤,
**학습된 패턴을 보이지만 인식하지 않은 기업을 역으로 추출**하는 구조를 설계함.
        """)
        st.dataframe(pd.DataFrame({
            "모델 예측": ["위험 높음", "위험 높음", "위험 낮음", "위험 낮음"],
            "실제 인식": ["인식함", "인식 안 함", "인식함", "인식 안 함"],
            "통상 해석": ["True Positive", "False Positive", "False Negative", "True Negative"],
            "이 프로젝트의 해석": ["모델 타당성 검증", "감사인이 주목할 대상", "모델이 놓친 사례", "—"],
        }), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### 상세 — 모집단 정의")
    m1, m2 = st.columns([2, 3])
    with m1:
        st.markdown(f"""
- DART 전 상장사 **{pipe['상장사_전체']:,}개사**의 업종코드를 수집
- 업종 핵심 대역(영상·방송프로그램 제작/배급 591, 음반 592, 방송 60x, 창작예술 90x) 추출
- 웹툰·웹소설 IP 6개사 추가 — 콘텐츠 자산화·상각 구조가 드라마·영화와 동일함
- 게임(582x) 제외 — 개발비 자산 성격이 다름
- **상장폐지 기업도 포함** — 사라진 기업을 빼면 생존편향이 생겨 정작 위험했던 사례가 누락됨
- 최종 **{pipe['모집단_확정']}개사 / 2015–2025 사업연도**
        """)
    with m2:
        ind = uni.groupby("업종").size().reset_index(name="기업수").sort_values("기업수")
        fig = px.bar(ind, x="기업수", y="업종", orientation="h",
                     color_discrete_sequence=[ACCENT], text="기업수")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="기업 수")
        st.plotly_chart(fig, use_container_width=True)

    st.warning("""
**폐기한 설계 하나 — 무형자산 비중 게이트**

당초 "무형자산/자산총계 5% 미만이면 적용 대상 아님"으로 모집단을 거르려 했으나, 실제로 재보니
이 비율은 콘텐츠 자산 집약도가 아니라 **영업권 비중**을 재고 있었음.
와이지(4.1%)·쇼박스(2.4%)·큐브(3.6%) 같은 실제 엔터사는 탈락하고,
네이버(8.3%)·카카오(18.8%)·축산물 유통업체(7.3%)가 통과함.
핵심 업종 72개사 중 37개사(51%)가 미달로 나온 것 자체가 판별력이 없다는 증거임.
→ 필터에서 제외하고 설명변수로만 사용함.
    """)

# ============================== ② 데이터 확보 ==============================
with tab2:
    lead("가장 큰 과제는 모델이 아니라 <b>정답 데이터의 확보</b>. "
         "공개 API가 제공하는 재무제표 본문에는 대형사의 손상차손이 나타나지 않음.")

    a, b, c3 = st.columns(3)
    with a:
        tag("본문에 없는 이유")
        plain("""
현금흐름표 조정항목을 본문에 표시하지 않고 <b>주석으로 이관한 회사가 다수임.</b><br><br>
하이브·스튜디오드래곤이 여기 해당하며, API는 본문만 반환함.
결과적으로 <b>대형사일수록 데이터가 비는 구조.</b>
        """)
    with b:
        tag("주석 파싱만으로 부족한 이유")
        plain("""
주석은 표 구조가 회사마다 다르고 <b>동일 회사도 연도별로 변동함.</b><br><br>
표를 잘못 선택하면 전기 수치를 당기로, 천원을 원으로,
유형자산을 무형자산으로 읽게 됨. <b>오류를 인지하지 못하는 것이 가장 큰 위험.</b>
        """)
    with c3:
        tag("채택한 검증 방식")
        plain("""
<b>본문 금액과 앞뒤가 맞는 표만 채택</b>하는 방식으로 전환함.<br><br>
주석의 <b>기말 금액이 당기말 잔액과 일치하고, 기초 금액이 전기말 잔액과 일치할 때만</b> 인정.
불일치 시 채택하지 않고 '실패'로 기록함.
        """)

    st.markdown("")
    k = st.columns(3)
    k[0].metric("본문 API 방식", "0%", delta="레이블 확보 불가", delta_color="inverse")
    k[1].metric("주석 파싱 + 대사 도입", "60%")
    k[2].metric("최종 방식", "96%", delta="+96%p")

    detail_header("파이프라인과 검증 방식")
    st.markdown("""
DART 재무제표 API(`fnlttSinglAcntAll`)로 손상차손을 받으려 했으나 **절반도 잡히지 않았고,
그 이유가 결정적이었음.** 하이브·스튜디오드래곤 등은 현금흐름표를
`영업으로부터 창출된 현금` 한 줄로만 표시하고 상각비·손상차손 등 조정항목을 전부 주석으로 뺐음.
API는 본문만 반환하므로 볼 수가 없음.
    """)
    e1, e2 = st.columns(2)
    e1.error("""
**결측이 무작위가 아니었음**

조정항목을 생략한 쪽이 전부 대형사(하이브·에스엠·와이지·콘텐트리중앙).
본문만 쓰면 **소형사만으로 학습한 모델**이 되고 정작 중요한 대형사가 구조적으로 빠짐.
같은 회사도 연도별로 표시 방식이 바뀜(스튜디오드래곤은 2022년까지 표시, 2023년부터 생략).
    """)
    e2.success("""
**해결 — 재무상태표를 기준점으로 삼는 2점 대사**

사업보고서 원문의 무형자산 주석 변동표를 파싱하고,
**기말 장부금액 = 당기말 잔액**이면서 **기초 장부금액 = 전기말 잔액**인 표만 채택함.
단위(원/천원/백만원)도 어느 배수에서 일치하는지로 역산함.
→ 당기/전기 혼동, 연결/별도 혼동, 단위 오인, 취득원가/장부금액 혼동이 한꺼번에 걸러짐.
    """)

    st.markdown("##### 파서 개선 이력 (시범 10개사 · 50 firm-year 기준)")
    hist = pd.DataFrame(pipe["파서개선"])
    fig = go.Figure(go.Bar(
        x=hist["단계"], y=hist["성공률"],
        marker_color=[WARN] + [MUTED] * (len(hist) - 2) + [ACCENT],
        text=[f"{v:.0f}%" for v in hist["성공률"]], textposition="outside"))
    fig.update_layout(height=340, yaxis_title="레이블 확보율 (%)", xaxis_title="",
                      yaxis_range=[0, 108], margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hist.rename(columns={"성공률": "성공률(%)"}), hide_index=True,
                 use_container_width=True)

    st.markdown("""
> 진단 과정에서 확인한 함정들
> - **'대손상각비'가 '손상' 부분일치에 걸려 들어옴** — 매출채권 관련이므로 반드시 배제해야 함
> - **표 첫 행이 `(단위: 천원)` 안내문**인 경우, 이를 헤더로 오인하면 합계 열을 못 찾아
>   모든 열을 더하게 되고 **금액이 정확히 2배**가 됨 (하이브 5개 연도 전부 이 원인)
> - **[기재정정]사업보고서는 고친 부분만 담아** 무형자산 주석이 통째로 없을 수 있음.
>   가장 최근 접수번호만 쓰면 안 되고 정정본→원본 순으로 재시도해야 함
    """)

    st.divider()
    st.markdown("#### 상세 — 표본이 걸러지는 과정과 결측 편향 점검")
    steps = pd.DataFrame({
        "단계": ["전 상장사", "업종 필터", "사업보고서 보유", "수집 firm-year",
               "주석 대사 성공", "적용대상·분리가능", "2점 대사 확정"],
        "값": [pipe["상장사_전체"], pipe["업종필터_통과"], pipe["사업보고서_보유"],
              pipe["수집_firmyear"], pipe["대사성공_보완후"], pipe["사용가능"], pipe["확정표본"]],
        "단위": ["개사", "개사", "개사", "firm-year", "firm-year", "firm-year", "firm-year"],
    })
    st.dataframe(steps, hide_index=True, use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        by_year = panel.groupby("사업연도").agg(
            전체=("대사", "size"), 성공=("대사", lambda s: (s == "성공").sum())).reset_index()
        by_year["성공률"] = by_year["성공"] / by_year["전체"] * 100
        fig = px.line(by_year, x="사업연도", y="성공률", markers=True,
                      color_discrete_sequence=[ACCENT])
        fig.update_layout(height=300, yaxis_range=[0, 100], yaxis_title="대사 성공률 (%)",
                          title="연도별 — 특정 연도에 쏠리지 않음",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        tgt = panel[panel["적용대상"] & panel["자산총계"].notna()].copy()
        tgt["규모"] = pd.qcut(tgt["자산총계"], 4, labels=["소형", "중소", "중대", "대형"])
        sz = tgt.groupby("규모", observed=True)["대사"].apply(
            lambda s: (s == "성공").mean() * 100).reset_index(name="성공률")
        fig = px.bar(sz, x="규모", y="성공률", color_discrete_sequence=[ACCENT],
                     text=sz["성공률"].map("{:.1f}%".format))
        fig.update_layout(height=300, yaxis_range=[0, 105], yaxis_title="대사 성공률 (%)",
                          title="규모별 — 대형사 편향이 남아 있음(한계)",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("규모 편향은 완전히 제거하지 못했으며 한계로 명시함. "
               "시범 10개사에서 96%가 나온 것은 그 10개사가 대형사 위주였기 때문임.")

# ============================== ③ 데이터 탐색 ==============================
with tab3:
    n_pos = int((conf["손상액"] > 0).sum())
    lead(f"56개사 390개 사업연도에서 <b>손상 인식 {n_pos}건, 누적 "
         f"{conf['손상액'].sum() / 1e12:.2f}조 원</b> 규모의 관측치를 확보. "
         "본문 공시만으로는 상당 부분이 포착되지 않는 데이터임.")

    a, b, c3 = st.columns(3)
    with a:
        tag("조회 가능한 항목")
        plain("""
기업별로 <b>무형자산 잔액, 당기 손상액, 당기 상각액</b>의 연도별 추이를 함께 확인 가능함.<br><br>
모든 수치는 <b>연결재무제표(CFS) 기준</b>이며, 연결 미작성 기간은 별도(OFS)로 대체함.
        """)
    with b:
        tag("분포상 특징")
        plain("""
손상액이 <b>기말 무형자산 잔액을 초과하는 사례</b>가 다수 관측됨.
기중에 자산을 사실상 전액 손상한 경우로, <b>콘텐츠 자산의 회수 위험이 드러나는 구간임.</b>
        """)
    with c3:
        tag("해석 시 유의사항")
        plain("""
연결·별도 기준이 전환되는 연도에는 무형자산이 <b>수 배로 변동함.</b>
실질적 증감이 아니라 <b>연결범위 변경에 따른 것</b>이므로 해당 연도를 별도 표시함.
        """)

    detail_header("확보한 손상 데이터")
    k = st.columns(4)
    k[0].metric("검증 표본", f"{len(conf)} firm-year")
    k[1].metric("기업 수", f"{conf['corp_code'].nunique()}개사")
    k[2].metric("손상 인식(>0)", f"{n_pos}건")
    k[3].metric("손상액 합계", won(conf["손상액"].sum()))

    y1, y2 = st.columns(2)
    with y1:
        byy = conf.groupby("사업연도").agg(
            손상총액=("손상액", "sum"),
            인식기업수=("손상액", lambda s: int((s > 0).sum())),
            관측수=("손상액", "size")).reset_index()
        byy["인식률"] = byy["인식기업수"] / byy["관측수"] * 100
        fig = go.Figure()
        fig.add_bar(x=byy["사업연도"], y=byy["손상총액"] / 1e8, name="손상총액(억원)",
                    marker_color=ACCENT)
        fig.add_scatter(x=byy["사업연도"], y=byy["인식률"], name="인식률(%)",
                        yaxis="y2", mode="lines+markers", line=dict(color=WARN))
        # 제목과 범례가 겹치지 않도록 범례를 그래프 아래로 내림
        fig.update_layout(height=400, title="연도별 손상 인식 규모와 인식률",
                          yaxis_title="손상총액(억원)",
                          yaxis2=dict(title="인식률(%)", overlaying="y", side="right",
                                      range=[0, 100]),
                          margin=dict(l=0, r=0, t=50, b=70),
                          legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
        st.plotly_chart(fig, use_container_width=True)
    with y2:
        nz = conf[conf["손상비율"] > 0].copy()
        nz["구간"] = pd.cut(nz["손상비율"],
                          bins=[0, 0.01, 0.05, 0.1, 0.3, 1, np.inf],
                          labels=["<1%", "1-5%", "5-10%", "10-30%", "30-100%", ">100%"])
        cnt = nz.groupby("구간", observed=True).size().reset_index(name="건수")
        fig = px.bar(cnt, x="구간", y="건수", color_discrete_sequence=[ACCENT], text="건수")
        fig.update_layout(height=360, title="손상액 / 무형자산 잔액 분포",
                          xaxis_title="손상 비율", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
**'>100%' 구간은 오류가 아님** — 기중에 자산을 거의 전액 손상한 정상 사례임.
예: 씨씨에스충북방송 2022년 — 기초 8.1억 → 손상 5.2억 → 기말 0.3억.
원표 6건을 대조해 확인했으며, 오히려 모델이 잡아내야 할 사건임.

**모든 수치는 연결(CFS) 기준임.** 별도재무제표 주석과는 금액이 다를 수 있음.
예: 하이브 2025년 무형자산은 연결 1조 9,866.9억 / 별도 189.0억으로 약 105배 차이가 남
(연결에는 인수한 종속기업의 영업권·무형자산이 포함됨).
    """)

    st.divider()
    st.markdown("#### 상세 — 기업별 조회")
    names = sorted(conf["기업명"].unique())
    default = names.index("(주)콘텐트리중앙") if "(주)콘텐트리중앙" in names else 0
    sel = st.selectbox("기업 선택 (검증 표본 보유 기업만)", names, index=default)
    d = conf[conf["기업명"] == sel].sort_values("사업연도")

    s1, s2 = st.columns([3, 2])
    with s1:
        fig = go.Figure()
        fig.add_bar(x=d["사업연도"], y=d["무형자산"] / 1e8, name="무형자산(억원)",
                    marker_color=ACCENT, opacity=0.85)
        fig.add_bar(x=d["사업연도"], y=d["손상액"] / 1e8, name="손상액(억원)",
                    marker_color=WARN)
        fig.add_scatter(x=d["사업연도"], y=d["상각액"] / 1e8, name="상각액(억원)",
                        mode="lines+markers", line=dict(color="#A9B4C4", dash="dot"))
        fig.update_layout(height=420, barmode="overlay", yaxis_title="억원",
                          title=f"{sel} — 무형자산·손상·상각 추이 (연결 기준)",
                          margin=dict(l=0, r=0, t=50, b=70),
                          legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
        st.plotly_chart(fig, use_container_width=True)
    with s2:
        show = d[["사업연도", "무형자산", "손상액", "상각액", "손상비율", "재무제표"]].copy()
        for c_ in ["무형자산", "손상액", "상각액"]:
            show[c_] = show[c_].map(won)
        show["손상비율"] = show["손상비율"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        st.dataframe(show, hide_index=True, use_container_width=True, height=420)
        st.caption("`재무제표` 열의 CFS는 연결, OFS는 별도 기준임. "
                   "연결 미작성 기간에 한해 별도로 대체함.")

    sw = panel[(panel["기업명"] == sel) & (panel["기준전환"] == True)]
    if len(sw):
        st.warning(f"**연결/별도 전환 발생 연도: {', '.join(map(str, sw['사업연도'].tolist()))}** — "
                   "재무제표 기준이 바뀌면 무형자산 증가율이 실제 증가가 아니라 기준 변경으로 튀므로, "
                   "해당 구간은 증가율 계산에서 끊었음. 전체 모집단에서 22개사 30건 발생함.")

# ============================== ④ 가설검정·모델링 ==============================
with tab4:
    lead("<b>예상한 신호는 <span class='red'>부재</span>.</b> 설명력이 가장 높았던 규칙은 "
         "<b>'전기 손상 인식 여부' 단일 변수</b>이며, 변수 8개를 투입한 모델도 이를 넘지 못함.")

    a, b, c3 = st.columns(3)
    with a:
        tag("설정한 가설")
        plain("""
<b>무형자산은 계속 증가하는데 매출이 따라오지 않으면 손상이 임박한다</b>
— 회계적으로 가장 개연성이 높은 신호로 설정함.
        """)
    with b:
        tag("검정 결과")
        plain("""
<b>두 집단 간 차이 없음.</b> 차기 손상 인식군과 미인식군의 증가괴리가 사실상 동일함.<br><br>
1년 괴리·3년 누적 괴리, t+1·t+2·2년 내 — <b>모든 조합에서 기각됨.</b>
        """)
    with c3:
        tag("실제 신호")
        plain("""
<b>영업이익률</b>(낮을수록 위험)과 <b>전기 손상 인식 여부</b>.<br><br>
다만 후자의 설명력은 <b>1년에 한정됨</b> — t+2 시점에서는 유의성이 사라짐(p=0.18).
        """)

    detail_header("가설 → 검정 → 모델 순서")
    st.markdown("""
변수를 '넣어보니 잘 나와서'가 아니라 **'회계적 가설이 검증되어서'** 넣는 순서를 유지함.
연속형은 Levene 검정으로 등분산을 확인한 뒤 t-검정 또는 Welch 검정을 적용하고,
분포가 비정규인 경우를 대비해 Mann-Whitney U 검정을 함께 봄.
    """)
    hyp = pd.DataFrame({
        "가설": ["H1", "H2", "H3", "H4"],
        "내용": ["자산은 쌓이는데 수익이 안 따라오면 손상 위험이 크다",
               "상각을 늦추면(유효상각률 하락) 비용 이연이고 뒤에 손상으로 터진다",
               "이익의 질이 나쁘면(발생액 괴리) 손상 인식이 임박했다",
               "무형자산 비중이 클수록 손상 시 충격이 크고 유인도 크다"],
        "변수": ["무형자산증가율 − 매출성장률", "유효상각률과 그 변화",
               "(당기순이익−영업활동현금흐름)/자산총계", "무형자산/자산총계, 무형자산/매출액"],
        "결과": ["기각", "기각", "약함", "부분"],
    })
    st.dataframe(hyp, hide_index=True, use_container_width=True)

    horizon = st.radio("예측 시계", ["2년 내", "t+1년", "t+2년"], horizontal=True)
    t = tests[tests["시계"] == horizon].copy()
    t = t[t["p값"].notna()]
    t["최소p"] = t[["p값", "MWU_p"]].min(axis=1)
    t = t.sort_values("최소p")

    h1, h2 = st.columns([3, 2])
    with h1:
        fig = px.bar(t, x="최소p", y="변수", orientation="h",
                     color=t["최소p"] < 0.05,
                     color_discrete_map={True: ACCENT, False: MUTED},
                     text=t["최소p"].map("{:.4f}".format))
        fig.add_vline(x=0.05, line_dash="dash", line_color=WARN, annotation_text="p=0.05")
        fig.update_layout(height=460, showlegend=False,
                          xaxis_title="p값 (t/Welch와 MWU 중 최소)",
                          yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with h2:
        show = t[["변수", "양성n", "음성n", "양성중앙", "음성중앙", "검정", "p값", "MWU_p", "유의"]]
        st.dataframe(show.round(4), hide_index=True, use_container_width=True, height=460)

    st.error("""
**핵심 가설이 기각됨** — 이 프로젝트 설계의 중심이었던 "무형자산증가율 − 매출성장률 괴리"는
1년 괴리·3년 누적 괴리 모두, t+1·t+2·2년 내 **모든 시계에서 유의하지 않음(p=0.39–0.78)**.
손상을 인식하게 될 기업과 아닌 기업의 증가괴리가 사실상 동일함.
    """)
    st.markdown("""
**업종·연도 중앙값 대비 상대화도 효과가 없었음.** 상대변수는 원변수와 VIF 90–290으로 사실상
같은 변수였음(업종군이 작고 기업 간 편차가 업종 간 편차보다 훨씬 큼). 모델에서 제외함.

**살아남은 신호** — 영업이익률(전 시계에서 가장 강함), 당기 손상 인식 여부(t+1·2년내에서만),
무형자산/영업현금흐름·무형자산/매출(약함).
단 **당기 손상 인식 여부는 t+2 단독으로는 유의하지 않음(p=0.18)** — 손상 지속성은 1년짜리 현상임.
    """)

    st.divider()
    st.markdown("#### 상세 — 모델링과 지표 선택 근거")
    st.markdown("""
손상을 놓치는 것(2종 오류)이 정상 기업을 잘못 지목하는 것(1종 오류)보다 손실이 훨씬 큼.
감사기준도 감사위험을 **"중요한 왜곡표시가 있는 재무제표에 적정의견을 주는 위험"** 으로만 정의함.
따라서 Accuracy를 배제하고 Recall과 Average Precision(PR-AUC)을 기준으로 봄.

분할도 두 가지를 모두 확인함 — **연도 분할**(과거로 학습해 미래를 예측, 실제 운영과 동일)과
**기업 분할**(같은 기업이 학습·검증에 함께 들어가는 누수 제거).
    """)

    st.markdown("**모델에 투입한 변수 8개** — 위 검정에서 살아남았거나 통제 목적으로 필요한 것만 선별함")
    st.dataframe(pd.DataFrame({
        "변수": ["영업이익률", "당기 손상 인식 여부", "무형자산/매출액", "무형자산/영업활동현금흐름",
               "발생액 괴리", "부채비율", "자산규모(로그)", "증가괴리(1년)"],
        "선정 사유": ["검정에서 전 시계 유의 — 수익성 신호",
                  "검정에서 유의 — 손상 지속성",
                  "약하게 유의 — 매출 대비 자산 과다",
                  "약하게 유의 — 현금창출력 대비 자산 과다",
                  "이익의 질 통제", "재무 안정성 통제", "규모 효과 통제",
                  "기각된 핵심 가설이지만 대조군으로 포함"],
    }), hide_index=True, use_container_width=True)
    st.caption("업종·연도 상대화 변수는 원변수와 VIF 90–290으로 사실상 동일해 제외함. "
               "결측이 있는 변수는 중앙값으로 대체하고 결측 여부 플래그를 함께 투입함.")

    split = st.radio("분할 방식", ["연도분할", "기업분할", "t+1·연도분할"], horizontal=True)
    m = models[models["모델"].str.startswith(f"[{split}]")].copy()
    m["모델명"] = m["모델"].str.replace(f"[{split}] ", "", regex=False)

    v1, v2 = st.columns([3, 2])
    with v1:
        fig = go.Figure()
        fig.add_bar(x=m["모델명"], y=m["AP"], name="Average Precision", marker_color=ACCENT)
        fig.add_bar(x=m["모델명"], y=m["Recall"], name="Recall", marker_color=ACCENT_SOFT)
        fig.update_layout(height=380, barmode="group", yaxis_range=[0, 1.05],
                          margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        st.dataframe(m[["모델명", "Recall", "Precision", "AP", "ROC-AUC"]].round(3),
                     hide_index=True, use_container_width=True, height=380)

    st.error("""
**변수 8개를 넣은 모델이 "작년에 손상했나?" 한 줄짜리 규칙을 이기지 못함.**

기업 분할에서 로지스틱 회귀 AP는 0.653, 단일 변수 기준선은 0.649로 사실상 동일하며,
ROC-AUC는 오히려 단일 변수가 더 높음(0.679 vs 0.597).
지표나 임계값을 조정해 좋아 보이게 만들 수는 있으나, 그렇게 하지 않음.
    """)

    st.markdown("##### 로지스틱 회귀 계수 (표준화 후) — 해석 불가로 판단한 근거")
    cf = coef[~coef["변수"].str.endswith("_결측")].copy()
    fig = px.bar(cf.sort_values("계수"), x="계수", y="변수", orientation="h",
                 color=cf.sort_values("계수")["계수"] > 0,
                 color_discrete_map={True: WARN, False: ACCENT})
    fig.update_layout(height=320, showlegend=False, yaxis_title="",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.warning("""
**영업이익률 계수가 양수(오즈비 1.29)로 나옴** — 수익성이 높을수록 손상 위험이 크다는 뜻인데,
단변량 검정에서는 정반대였음(양성 중앙값 −1.3% vs 음성 +3.2%).
**단변량 결과와 다변량 계수의 부호가 뒤집힘.** 표본 322건에 변수 13개를 넣어 노이즈를 학습한 것으로
판단함. 이 프로젝트의 목적이 "왜 이 기업이 위험한지 계수로 설명하는 것"이었으므로,
계수를 신뢰할 수 없다면 모델을 쓸 수 없다고 결론함.
    """)

# ============================== ⑤ 한계와 결론 ==============================
with tab5:
    lead("재무제표 수준 데이터에 기반한 손상 예측은 <b class='red'>성립하지 않음.</b> "
         "예측 가능성의 한계를 규명한 것이 이 프로젝트의 결론.")

    a, b, c3 = st.columns(3)
    with a:
        tag("결론")
        plain("""
전기 손상 이력을 넘어서는 예측력을 확보하지 못함.
손상 인식은 <b>재무비율로 환원되지 않는 경영진 재량의 영역</b>이며,
이것이 감사인의 <b>'전문가적 판단이 필요'</b>한 이유임.
        """)
    with b:
        tag("이 결과를 유지한 이유")
        plain("""
레이블 기준·임계값·평가지표를 조정하면 성능을 개선된 것처럼 제시할 수 있었으나
그렇게 하지 않음. <b>감사 영역에서 검증되지 않은 결론을 제시하는 것이 더 큰 위험</b>이라고 판단함.
        """)
    with c3:
        tag("남은 산출물")
        plain("""
모델이 아니라 <b>데이터 파이프라인</b>.
공개 API가 제공하지 않는 손상·상각 수치를
주석·본문 대사로 검증해 <b>78개사 661 firm-year</b> 규모로 구축함.
        """)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 이 프로젝트의 실질적 성과")
        st.markdown("""
1. **공개 API가 주지 않는 데이터를 확보함.**
   재무제표 본문으로는 대형사의 손상차손을 볼 수 없음을 확인하고,
   사업보고서 주석을 파싱해 **본문과 2점 대사**로 검증하는 방식을 설계함
2. **검증을 통과하지 못한 값을 버림.**
   대사가 맞지 않으면 채택하지 않고 '실패'로 명시함.
   틀린 값을 성공으로 위장하지 않는 것을 원칙으로 유지함
3. **음성 결과를 지표 조정으로 감추지 않음.**
   레이블 기준·임계값·평가지표를 바꾸면 좋아 보이게 만들 수 있었으나 하지 않음
        """)
    with c2:
        st.markdown("#### 한계 (명시)")
        st.markdown("""
- **규모 편향**: 대사 성공률이 대형사 95.7% vs 소형–중대 78–82%. 완전히 제거하지 못함
- **표본 규모**: 검증 표본 390 firm-year / 56개사. 복잡한 모델을 쓸 수 없는 크기임
- **관측 불가능성**: 손상 인식 여부만 관측되고 '인식했어야 함'은 관측 불가.
  당초 설계했던 '미인식 위험군 추출'은 모델이 성립하지 않아 수행하지 못함
- **해상도**: 개별 콘텐츠 단위가 아닌 기업 단위 집계 데이터임.
  판권·콘텐츠를 별도 컬럼으로 구분 공시하는 기업은 24%뿐이어서 무형자산 총액으로 분석함
- **회계정책 차이**: 콘텐츠를 무형자산이 아니라 **재고자산·선급금으로 분류**하는 제작사가 있음
  (에이스토리 무형자산 비중 중앙값 0.026%). 같은 드라마 제작사라도 적용 대상이 아닐 수 있음
        """)

    st.divider()
    st.subheader("인사이트")
    i1, i2, i3 = st.columns(3)
    i1.info("""
**① 공시 상세도 자체가 기업마다 다름**

같은 회사도 연도별로 현금흐름표 조정항목 표시 여부가 바뀜.
스튜디오드래곤은 2022년까지 표시했다가 2023년부터 생략함.
**본문만 보고 비교하면 회사 간·연도 간 비교가능성이 성립하지 않음.**
    """)
    i2.info("""
**② 주석과 본문의 대사가 유일한 검증 수단이었음**

당기/전기, 연결/별도, 단위, 취득원가/장부금액을 구분할 방법이 없었으나,
**본문 금액과 맞춰보는 것**만으로 네 가지가 한꺼번에 해결됨.
감사에서 주석과 본문을 맞춰보는 절차가 왜 필요한지를 데이터로 경험함.
    """)
    i3.info("""
**③ 손상 지속성은 1년짜리 현상임**

당기 손상 인식은 t+1년 손상을 강하게 예측하지만
t+2년에는 유의하지 않음(p=0.18).
**손상을 인식한 기업은 다음 해까지 위험하지만 그 이후 정상화되는 패턴**으로,
감사 계획 시 주목 기간의 근거가 될 수 있음.
    """)

    st.divider()
    with st.expander("분석 절차 전체 (재현 가능)"):
        st.markdown("""
| 단계 | 스크립트 | 내용 |
|---|---|---|
| 1 | `01_fetch_corp_codes` | DART 기업코드 마스터 수집 |
| 2 | `02–05_pilot_*` | 본문 API로 손상 레이블 확보 가능성 검증 → **불가 판정** |
| 3 | `06–09_note_*` | 사업보고서 주석 파싱 시도 → 검증 없는 값이 나오는 문제 확인 |
| 4 | `10_extractor_reconciled` | 본문 대사 도입 (60%) |
| 5 | `12_extractor_v3` | 느슨한 검출 + 2점 대사, 헤더 보정 (90%) |
| 6 | `14_extractor_v4` | 정정보고서 폴백 (96%) |
| 7 | `15–17_universe*` | 업종코드 기반 모집단 확정, 무형비중 게이트 폐기 |
| 8 | `18_collect_main` | 78개사 661 firm-year 본 수집 |
| 9 | `19–22_diagnose/verify` | 결측 편향 점검, 손상액 원표 대조 검증 |
| 10 | `21_build_panel` | 연결/별도 전환 플래그, 적용범위 판정 |
| 11 | `23–24_features*` | 파생변수 1·2차 설계, 가설검정 |
| 12 | `25_model` | 로지스틱·랜덤포레스트·XGBoost, 두 가지 분할 검증 |
        """)

st.divider()
st.caption("데이터: DART 전자공시 OpenAPI (금융감독원) · 2015–2025 사업연도 · "
           "본 자료는 개인 학습·포트폴리오 목적의 분석이며 특정 기업의 회계처리 적정성에 대한 "
           "판단이나 투자 권유가 아님.")
