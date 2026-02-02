import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# Utils
# =========================
def pct_to_rate(x: float) -> float:
    return max(0.0, float(x)) / 100.0

def krw_round(x: float, unit: int = 100) -> int:
    if x is None or np.isnan(x) or np.isinf(x):
        return 0
    return int(round(x / unit) * unit)

def guardrail_min_price(landed_cost: float, channel_cost_rate: float, min_margin_rate: float) -> float:
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return np.inf
    return landed_cost / denom

def apply_import_cap(price: float, import_ref: float, premium_rate: float) -> float:
    cap = import_ref * (1.0 + premium_rate)
    return min(price, cap)

# =========================
# Engines: compute LIST(공식가)
# =========================
def list_price_market_index(market_mid: float, market_high: float, position: str) -> float:
    # 공식가를 시장에서 어디에 둘지 (단순 규칙: 중간/중상/상단)
    if position == "중앙":
        return market_mid
    if position == "중상":
        return (market_mid + market_high) / 2
    return market_high  # 상단

def list_price_list_discount(comp_list_mid: float, comp_list_high: float, position: str) -> float:
    # 경쟁 정가 분포 기반 공식가
    if position == "비슷":
        return comp_list_mid
    if position == "높게":
        return comp_list_high
    return (comp_list_mid + comp_list_high) / 2  # 살짝 높게

def list_price_margin_builder(landed_cost: float, channel_cost_rate: float, target_margin_rate: float) -> float:
    # 공식가를 '목표마진'으로 만드는 방식 (채널별로 공식가가 달라질 수 있어 정책 선택 필요)
    denom = 1.0 - channel_cost_rate - target_margin_rate
    if denom <= 0:
        return np.inf
    return landed_cost / denom

# =========================
# Price ladder types
# =========================
PRICE_TYPES = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]

DEFAULT_LADDER = pd.DataFrame([
    {"가격타입": "상시할인가",   "MinDisc%": 15.0, "TargetDisc%": 20.0, "MaxDisc%": 25.0},
    {"가격타입": "홈사할인가",   "MinDisc%": 18.0, "TargetDisc%": 25.0, "MaxDisc%": 32.0},
    {"가격타입": "모바일라방가", "MinDisc%": 20.0, "TargetDisc%": 28.0, "MaxDisc%": 35.0},
    {"가격타입": "브랜드위크가", "MinDisc%": 25.0, "TargetDisc%": 33.0, "MaxDisc%": 40.0},
    {"가격타입": "원데이 특가",  "MinDisc%": 30.0, "TargetDisc%": 40.0, "MaxDisc%": 50.0},
])

def ladder_prices(list_price: float, min_disc: float, tgt_disc: float, max_disc: float):
    # 할인율이 클수록 가격은 낮아짐
    p_max = list_price * (1.0 - pct_to_rate(min_disc))   # 가장 덜 할인 = 상단
    p_tgt = list_price * (1.0 - pct_to_rate(tgt_disc))
    p_min = list_price * (1.0 - pct_to_rate(max_disc))   # 가장 많이 할인 = 하단
    return p_min, p_tgt, p_max

# =========================
# Main compute
# =========================
def compute_all(
    landed_cost: float,
    min_margin_rate: float,
    channels_df: pd.DataFrame,
    engine: str,
    # engine inputs
    market_mid: float | None,
    market_high: float | None,
    market_pos: str,
    comp_list_mid: float | None,
    comp_list_high: float | None,
    comp_list_pos: str,
    target_margin_rate_for_list: float | None,
    # import cap
    use_import_cap: bool,
    import_ref: float | None,
    import_premium_rate: float,
    # ladder
    ladder_df: pd.DataFrame,
    # list price policy
    list_price_policy: str,  # "전채널 동일" or "채널별(원가반영)"
):
    df = channels_df.copy()
    for col in ["수수료율", "프로모션율", "반품율"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["채널비용율(합)"] = df["수수료율"] + df["프로모션율"] + df["반품율"]

    rows = []

    # 1) compute list price per channel (or shared)
    shared_list_price = None
    if list_price_policy == "전채널 동일":
        if engine == "시장앵커형":
            shared_list_price = list_price_market_index(market_mid, market_high, market_pos)
        elif engine == "정가-할인형":
            shared_list_price = list_price_list_discount(comp_list_mid, comp_list_high, comp_list_pos)
        else:  # 원가-마진형
            # 전채널 동일로 쓰려면 기준 채널(대표 채널) 하나를 정해야 하는데,
            # MVP에선 '온라인(오픈마켓)' 첫 행을 기준으로 삼음.
            base_rate = float(df.iloc[0]["채널비용율(합)"]) if len(df) else 0.0
            shared_list_price = list_price_margin_builder(landed_cost, base_rate, target_margin_rate_for_list or (min_margin_rate + 0.15))

        if use_import_cap and import_ref is not None:
            shared_list_price = apply_import_cap(shared_list_price, import_ref, import_premium_rate)

    # 2) per channel compute guardrail min
    df["GuardrailMin"] = df["채널비용율(합)"].apply(lambda r: guardrail_min_price(landed_cost, r, min_margin_rate))

    # 3) build ladder bands per channel
    for _, ch in df.iterrows():
        channel = ch["채널"]
        cost_rate = float(ch["채널비용율(합)"])
        guard_min = float(ch["GuardrailMin"])

        if list_price_policy == "전채널 동일":
            list_price = float(shared_list_price)
        else:
            # 채널별 공식가 (특히 원가-마진형일 때 의미가 큼)
            if engine == "시장앵커형":
                list_price = list_price_market_index(market_mid, market_high, market_pos)
            elif engine == "정가-할인형":
                list_price = list_price_list_discount(comp_list_mid, comp_list_high, comp_list_pos)
            else:
                list_price = list_price_margin_builder(landed_cost, cost_rate, target_margin_rate_for_list or (min_margin_rate + 0.15))

            if use_import_cap and import_ref is not None:
                list_price = apply_import_cap(list_price, import_ref, import_premium_rate)

        # official exposure band (원하면 공식가도 밴드로 만들 수 있지만 MVP는 단일값)
        rows.append({
            "채널": channel, "가격타입": "공식가(노출)", "Min": list_price, "Target": list_price, "Max": list_price
        })

        for _, lad in ladder_df.iterrows():
            pmin, ptgt, pmax = ladder_prices(
                list_price,
                lad["MinDisc%"], lad["TargetDisc%"], lad["MaxDisc%"]
            )
            # guardrail 적용
            pmin = max(pmin, guard_min)
            ptgt = max(ptgt, pmin)
            pmax = max(pmax, ptgt)

            rows.append({
                "채널": channel,
                "가격타입": lad["가격타입"],
                "Min": pmin, "Target": ptgt, "Max": pmax
            })

    out = pd.DataFrame(rows)
    for c in ["Min", "Target", "Max"]:
        out[c] = out[c].apply(lambda x: krw_round(float(x), 100))
    return out

# =========================
# Chart
# =========================
def range_chart(df: pd.DataFrame):
    # y label: 채널 | 가격타입
    d = df.copy()
    d["y"] = d["채널"] + " | " + d["가격타입"]

    fig = go.Figure()

    # Range segments
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["Min"], r["Max"]],
            y=[r["y"], r["y"]],
            mode="lines",
            line=dict(width=10),
            showlegend=False,
            hovertemplate=f"{r['y']}<br>Min: {r['Min']:,}원<br>Target: {r['Target']:,}원<br>Max: {r['Max']:,}원<extra></extra>"
        ))
        # Target marker
        fig.add_trace(go.Scatter(
            x=[r["Target"]],
            y=[r["y"]],
            mode="markers",
            marker=dict(size=10),
            showlegend=False,
            hovertemplate=f"{r['y']}<br>Target: {r['Target']:,}원<extra></extra>"
        ))

    fig.update_layout(
        height=min(900, 40 * len(d) + 120),
        xaxis_title="가격(원)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def cannibal_warnings(df: pd.DataFrame):
    # 간단 룰: 같은 채널에서 가격타입 Target이 단계적으로 내려가야 함
    order = ["공식가(노출)"] + PRICE_TYPES
    warnings = []

    for ch in df["채널"].unique():
        sub = df[df["채널"] == ch].set_index("가격타입")
        prev = None
        for t in order:
            if t not in sub.index:
                continue
            cur = int(sub.loc[t, "Target"])
            if prev is not None and cur > prev:
                warnings.append(f"[{ch}] '{t}' Target({cur:,})가 상위 단계보다 높음 → 레벨/할인율 역전 가능")
            prev = cur

    return warnings

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Dynamic Pricing Simulator", layout="wide")
st.title("Dynamic Pricing Simulator (공식가 + 가격레벨 밴드 + 카니발 도식화)")

with st.sidebar:
    st.header("공통 입력")
    landed_cost = st.number_input("Landed Cost (원)", min_value=0, value=12000, step=100)
    min_margin = st.number_input("최소 마진(%) [하한용]", min_value=0.0, max_value=80.0, value=15.0, step=0.5)

    st.divider()
    st.header("가격 엔진(공식가 산정)")
    engine = st.radio("선택", ["시장앵커형", "정가-할인형", "원가-마진형"], index=0)

    list_price_policy = st.radio("공식가 정책", ["전채널 동일", "채널별(원가반영)"], index=0)

    st.divider()
    st.header("직구캡(옵션)")
    use_import_cap = st.checkbox("직구가 캡 적용", value=True)
    import_ref = st.number_input("직구 실구매가 기준(원)", min_value=0, value=35000, step=500) if use_import_cap else None
    import_premium = st.number_input("허용 프리미엄(%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5) if use_import_cap else 0.0

# 채널 테이블
st.subheader("채널 입력 (행 추가로 채널 확장)")
default_channels = pd.DataFrame([
    {"채널": "온라인(오픈마켓)", "수수료율": 0.12, "프로모션율": 0.05, "반품율": 0.02},
    {"채널": "자사몰", "수수료율": 0.03, "프로모션율": 0.04, "반품율": 0.02},
    {"채널": "공구", "수수료율": 0.08, "프로모션율": 0.03, "반품율": 0.02},
    {"채널": "홈쇼핑", "수수료율": 0.30, "프로모션율": 0.05, "반품율": 0.03},
])

if "channels_df" not in st.session_state:
    st.session_state["channels_df"] = default_channels

channels_df = st.data_editor(
    st.session_state["channels_df"],
    num_rows="dynamic",
    use_container_width=True
)
st.session_state["channels_df"] = channels_df

# 엔진별 입력
st.subheader("리서치/엔진 입력")
c1, c2, c3 = st.columns(3)

market_mid = market_high = None
comp_list_mid = comp_list_high = None
target_margin_for_list = None

with c1:
    if engine == "시장앵커형":
        market_mid = st.number_input("국내 시장가 중앙/평균(원)", min_value=0, value=39000, step=500)
        market_high = st.number_input("국내 시장가 상단(원)", min_value=0, value=49000, step=500)
        market_pos = st.selectbox("공식가 포지션", ["중앙", "중상", "상단"], index=1)
    else:
        market_pos = "중앙"
        st.caption("시장앵커형 선택 시 입력")

with c2:
    if engine == "정가-할인형":
        comp_list_mid = st.number_input("경쟁 정가 중앙/평균(원)", min_value=0, value=49000, step=500)
        comp_list_high = st.number_input("경쟁 정가 상단(원)", min_value=0, value=59000, step=500)
        comp_list_pos = st.selectbox("우리 공식가(정가) 포지션", ["비슷", "높게", "중간"], index=2)
    else:
        comp_list_pos = "비슷"
        st.caption("정가-할인형 선택 시 입력")

with c3:
    if engine == "원가-마진형":
        target_margin_for_list = st.number_input("공식가용 목표마진(%)", min_value=0.0, max_value=90.0, value=30.0, step=0.5)
    st.caption("원가-마진형에서만 사용")

# 가격레벨(할인율 밴드)
st.subheader("가격레벨(판매가 종류) 할인율 밴드 입력")
if "ladder_df" not in st.session_state:
    st.session_state["ladder_df"] = DEFAULT_LADDER.copy()

ladder_df = st.data_editor(
    st.session_state["ladder_df"],
    num_rows="fixed",
    use_container_width=True
)
st.session_state["ladder_df"] = ladder_df

# compute
min_margin_rate = pct_to_rate(min_margin)
import_premium_rate = pct_to_rate(import_premium)

out = compute_all(
    landed_cost=float(landed_cost),
    min_margin_rate=min_margin_rate,
    channels_df=channels_df,
    engine=engine,
    market_mid=market_mid,
    market_high=market_high,
    market_pos=market_pos,
    comp_list_mid=comp_list_mid,
    comp_list_high=comp_list_high,
    comp_list_pos=comp_list_pos,
    target_margin_rate_for_list=pct_to_rate(target_margin_for_list) if target_margin_for_list is not None else None,
    use_import_cap=use_import_cap,
    import_ref=float(import_ref) if use_import_cap else None,
    import_premium_rate=import_premium_rate,
    ladder_df=ladder_df,
    list_price_policy=list_price_policy
)

st.divider()
st.subheader("결과 테이블 (채널 x 가격타입 밴드)")
st.dataframe(out, use_container_width=True)

st.subheader("가격 범위 도식화 (카니발/역전 체크)")
fig = range_chart(out)
st.plotly_chart(fig, use_container_width=True)

warns = cannibal_warnings(out)
if warns:
    st.warning("카니발/레벨 역전 가능 경고:\n- " + "\n- ".join(warns))
else:
    st.success("기본 레벨 순서(공식가 → 상시 → … → 원데이)가 정상 범위로 보입니다.")
