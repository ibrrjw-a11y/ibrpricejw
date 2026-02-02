import streamlit as st
import pandas as pd
import numpy as np

# =========================
# Helpers
# =========================
def pct(x: float) -> float:
    """percent input (e.g., 10) -> ratio (0.10)"""
    return max(0.0, float(x)) / 100.0

def krw_round(x: float, unit: int = 100) -> int:
    """KRW rounding to unit (e.g., 100원 단위)"""
    if np.isnan(x) or np.isinf(x):
        return 0
    return int(round(x / unit) * unit)

def guardrail_min_price(landed_cost: float, channel_total_cost_rate: float, min_margin_rate: float) -> float:
    """
    손익/마진 하한선 가격.
    Price * (1 - channel_cost_rate) - landed_cost >= Price * min_margin_rate
    => Price * (1 - channel_cost_rate - min_margin_rate) >= landed_cost
    => Price >= landed_cost / (1 - channel_cost_rate - min_margin_rate)
    """
    denom = 1.0 - channel_total_cost_rate - min_margin_rate
    if denom <= 0:
        return np.inf
    return landed_cost / denom

def pick_market_anchor(market_low: float, market_mid: float, market_high: float, position: str) -> float:
    if position == "하단":
        return market_low
    if position == "상단":
        return market_high
    return market_mid  # 중앙

def list_price_from_position(market_mid: float, market_high: float, pos: str) -> float:
    # 정가 포지션: 낮게/비슷/높게를 아주 단순 룰로 매핑 (나중에 교체 가능)
    if pos == "낮게":
        return market_mid
    if pos == "높게":
        return market_high
    return (market_mid + market_high) / 2

def apply_import_cap(price: float, import_ref: float, premium_rate: float) -> float:
    """직구 캡: price <= import_ref * (1+premium)"""
    cap = import_ref * (1.0 + premium_rate)
    return min(price, cap)

# =========================
# Core compute (MVP rules; replace later in Step 3)
# =========================
def compute_price_band(
    landed_cost: float,
    min_margin_rate: float,
    channels_df: pd.DataFrame,
    mode: str,  # "소비자가 일관형" or "순마진 일관형"
    # Market-Index inputs
    market_low: float | None,
    market_mid: float | None,
    market_high: float | None,
    market_position: str,
    # Import-Parity inputs
    import_avg: float | None,
    import_min: float | None,
    import_premium_rate: float,
    use_import_cap: bool,
    # List-then-Discount inputs
    use_list_discount: bool,
    comp_disc_avg_rate: float | None,
    comp_disc_max_rate: float | None,
    list_position: str,
):
    df = channels_df.copy()

    # channel total cost rate
    df["수수료율"] = df["수수료율"].astype(float)
    df["프로모션율"] = df["프로모션율"].astype(float)
    df["반품율"] = df["반품율"].astype(float)

    df["채널비용율(합)"] = df["수수료율"] + df["프로모션율"] + df["반품율"]

    # 1) Guardrail min per channel
    df["Min(하한)"] = df["채널비용율(합)"].apply(lambda r: guardrail_min_price(landed_cost, r, min_margin_rate))

    # 2) Target / Max logic
    # Anchor base target from market index (if provided)
    base_target = None
    if market_low is not None and market_mid is not None and market_high is not None:
        base_target = pick_market_anchor(market_low, market_mid, market_high, market_position)

    # List-then-Discount if enabled (derive list price + discount-driven target/min)
    base_list = None
    if use_list_discount and market_mid is not None and market_high is not None:
        base_list = list_price_from_position(market_mid, market_high, list_position)

    results = []
    for _, row in df.iterrows():
        ch = row["채널"]
        cost_rate = row["채널비용율(합)"]
        min_price = row["Min(하한)"]

        # Determine channel target/max
        if use_list_discount and base_list is not None and comp_disc_avg_rate is not None and comp_disc_max_rate is not None:
            # 정가 기반 운영
            list_price = base_list
            target_price = list_price * (1.0 - comp_disc_avg_rate)  # 일상 판매가
            max_price = list_price  # 정가를 상단으로
            # 딜 하한: 최대 할인율 적용 (단, guardrail보다 낮아지면 guardrail로 올림)
            min_candidate = list_price * (1.0 - comp_disc_max_rate)
            min_price_final = max(min_price, min_candidate)
        else:
            # 시장 앵커 기반
            if base_target is None:
                # 아무 시장 입력이 없으면 "원가 기반 목표마진 가격"을 Target으로 사용 (임시)
                target_price = guardrail_min_price(landed_cost, cost_rate, min_margin_rate + 0.10)
            else:
                target_price = base_target
            # Max는 시장 상단이 있으면 시장 상단, 없으면 Target의 1.15배(임시)
            max_price = market_high if market_high is not None else target_price * 1.15
            min_price_final = min_price

        # Import-Parity cap (optional)
        if use_import_cap and import_avg is not None:
            target_price = apply_import_cap(target_price, import_avg, import_premium_rate)
            max_price = apply_import_cap(max_price, import_avg, import_premium_rate)
            # Min도 직구캡에 걸릴 순 있지만 보통은 하한이 더 낮으니 그대로 둠 (원하면 cap 적용 가능)

        # Mode: consumer-consistent vs margin-consistent
        if mode == "순마진 일관형":
            # 채널별로 목표 Target이 동일하게 유지되는 게 아니라,
            # '동일 마진'이 되도록 Target을 원가/비용율 기반으로 재계산.
            # 여기서는 base target을 '목표 마진(min_margin + 0.10)'로 통일하는 예시.
            target_price = guardrail_min_price(landed_cost, cost_rate, min_margin_rate + 0.10)
            max_price = guardrail_min_price(landed_cost, cost_rate, min_margin_rate + 0.18)
            # import cap 적용은 다시
            if use_import_cap and import_avg is not None:
                target_price = apply_import_cap(target_price, import_avg, import_premium_rate)
                max_price = apply_import_cap(max_price, import_avg, import_premium_rate)

        # Ensure ordering
        target_price = max(target_price, min_price_final)
        max_price = max(max_price, target_price)

        results.append({
            "채널": ch,
            "Min(하한)": min_price_final,
            "Target(목표)": target_price,
            "Max(상한)": max_price,
        })

    out = pd.DataFrame(results)
    # KRW rounding
    for c in ["Min(하한)", "Target(목표)", "Max(상한)"]:
        out[c] = out[c].apply(lambda x: krw_round(x, 100))
    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Dynamic Pricing Simulator (MVP)", layout="wide")
st.title("Dynamic Pricing Simulator (Streamlit MVP)")

with st.sidebar:
    st.header("공통 입력")
    landed_cost = st.number_input("Landed Cost (원)", min_value=0, value=12000, step=100)
    min_margin = st.number_input("최소 마진(%)", min_value=0.0, max_value=80.0, value=15.0, step=0.5)
    mode = st.radio("가격 정책 모드", ["소비자가 일관형", "순마진 일관형"], index=0)

    st.divider()
    st.header("로직 활성화")
    use_market_index = st.checkbox("Market-Index (국내 시장가 앵커)", value=True)
    use_import_cap = st.checkbox("Import-Parity (직구가 캡 적용)", value=True)
    use_list_discount = st.checkbox("List-then-Discount (정가→할인율 운영)", value=False)

# Channels input table
st.subheader("채널 입력 (채널 추가는 여기서 행 추가)")
default_channels = pd.DataFrame([
    {"채널": "온라인(오픈마켓)", "수수료율": 0.12, "프로모션율": 0.05, "반품율": 0.02},
    {"채널": "자사몰", "수수료율": 0.03, "프로모션율": 0.04, "반품율": 0.02},
    {"채널": "공구", "수수료율": 0.08, "프로모션율": 0.03, "반품율": 0.02},
    {"채널": "홈쇼핑", "수수료율": 0.30, "프로모션율": 0.05, "반품율": 0.03},
])

if "channels_df" not in st.session_state:
    st.session_state["channels_df"] = default_channels

edited = st.data_editor(
    st.session_state["channels_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "채널": st.column_config.TextColumn(required=True),
        "수수료율": st.column_config.NumberColumn(help="예: 0.12 = 12%", min_value=0.0, max_value=1.0, step=0.01),
        "프로모션율": st.column_config.NumberColumn(help="예: 0.05 = 5%", min_value=0.0, max_value=1.0, step=0.01),
        "반품율": st.column_config.NumberColumn(help="예: 0.02 = 2%", min_value=0.0, max_value=1.0, step=0.01),
    }
)
st.session_state["channels_df"] = edited

colA, colB = st.columns(2)

# Market-Index inputs
with colA:
    st.subheader("국내 시장 리서치 입력 (Market-Index)")
    if use_market_index:
        market_low = st.number_input("국내 가격 하단(원)", min_value=0, value=29000, step=500, key="m_low")
        market_mid = st.number_input("국내 가격 중앙값/평균(원)", min_value=0, value=39000, step=500, key="m_mid")
        market_high = st.number_input("국내 가격 상단(원)", min_value=0, value=49000, step=500, key="m_high")
        market_position = st.selectbox("포지션", ["하단", "중앙", "상단"], index=1)
    else:
        market_low = market_mid = market_high = None
        market_position = "중앙"
        st.caption("비활성화됨")

# Import-Parity inputs
with colB:
    st.subheader("직구 리서치 입력 (Import-Parity)")
    if use_import_cap:
        import_avg = st.number_input("직구 실구매가 평균/중앙(원)", min_value=0, value=35000, step=500, key="i_avg")
        import_min = st.number_input("직구 실구매가 최저(원)", min_value=0, value=32000, step=500, key="i_min")
        import_premium = st.number_input("직구 대비 허용 프리미엄(%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    else:
        import_avg = import_min = None
        import_premium = 0.0
        st.caption("비활성화됨")

# List-then-Discount inputs
st.subheader("정가→할인율 운영 (List-then-Discount)")
if use_list_discount:
    c1, c2, c3 = st.columns(3)
    with c1:
        comp_disc_avg = st.number_input("경쟁 평균 표기 할인율(%)", min_value=0.0, max_value=90.0, value=20.0, step=0.5)
    with c2:
        comp_disc_max = st.number_input("경쟁 최대/딜 할인율(%)", min_value=0.0, max_value=90.0, value=35.0, step=0.5)
    with c3:
        list_position = st.selectbox("우리 정가 포지션", ["낮게", "비슷", "높게"], index=1)
else:
    comp_disc_avg = comp_disc_max = None
    list_position = "비슷"
    st.caption("비활성화됨 (필요시 사이드바에서 켜기)")

# Compute
channels_df = st.session_state["channels_df"].copy()
# convert sidebar % to ratios
min_margin_rate = pct(min_margin)
import_premium_rate = pct(import_premium)
comp_disc_avg_rate = pct(comp_disc_avg) if comp_disc_avg is not None else None
comp_disc_max_rate = pct(comp_disc_max) if comp_disc_max is not None else None

# Safety: ensure numeric rates
for col in ["수수료율", "프로모션율", "반품율"]:
    if col in channels_df.columns:
        channels_df[col] = pd.to_numeric(channels_df[col], errors="coerce").fillna(0.0)

out = compute_price_band(
    landed_cost=float(landed_cost),
    min_margin_rate=min_margin_rate,
    channels_df=channels_df,
    mode=mode,
    market_low=market_low if use_market_index else None,
    market_mid=market_mid if use_market_index else None,
    market_high=market_high if use_market_index else None,
    market_position=market_position,
    import_avg=import_avg if use_import_cap else None,
    import_min=import_min if use_import_cap else None,
    import_premium_rate=import_premium_rate,
    use_import_cap=use_import_cap,
    use_list_discount=use_list_discount,
    comp_disc_avg_rate=comp_disc_avg_rate,
    comp_disc_max_rate=comp_disc_max_rate,
    list_position=list_position,
)

st.divider()
st.subheader("결과: 채널별 가격 밴드 (Min ~ Target ~ Max)")
st.dataframe(out, use_container_width=True)

with st.expander("디버그/해석(간단)"):
    st.write("- Min(하한)은 Landed Cost + 채널 비용 + 최소 마진을 만족하는 손익 방어선입니다.")
    st.write("- Target/Max는 (시장 앵커 / 직구 캡 / 정가-할인 운영 / 정책 모드) 조합에 따라 바뀝니다.")
    st.write("- Step 3에서 원하시는 산식으로 compute_price_band() 내부만 교체하면 UI는 그대로 유지됩니다.")
