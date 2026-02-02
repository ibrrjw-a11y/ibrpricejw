# streamlit_app.py
# ============================================================
# Dynamic Pricing Simulator (No external viz libs)
# - 공식가(노출용) + 온라인 가격레벨(상시/홈사/라방/브위/원데이) 범위
# - 홈쇼핑 채널은 "홈쇼핑가" 단일
# - 홈쇼핑 기간(start~end) 입력 시: 온라인 가격레벨을
#   "홈쇼핑가 대비 +X% 이상"으로 자동 상향한 기간 운영가 표시
# - 도식화: HTML/CSS range bar (no matplotlib/plotly)
# - 추가: "계산식/로직 설명" 탭 (최대한 자세히)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Dynamic Pricing Simulator", layout="wide")

PRICE_TYPES_ONLINE = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]

DEFAULT_LADDER = pd.DataFrame([
    {"가격타입": "상시할인가",   "MinDisc%": 15.0, "TargetDisc%": 20.0, "MaxDisc%": 25.0},
    {"가격타입": "홈사할인가",   "MinDisc%": 18.0, "TargetDisc%": 25.0, "MaxDisc%": 32.0},
    {"가격타입": "모바일라방가", "MinDisc%": 20.0, "TargetDisc%": 28.0, "MaxDisc%": 35.0},
    {"가격타입": "브랜드위크가", "MinDisc%": 25.0, "TargetDisc%": 33.0, "MaxDisc%": 40.0},
    {"가격타입": "원데이 특가",  "MinDisc%": 30.0, "TargetDisc%": 40.0, "MaxDisc%": 50.0},
])

DEFAULT_CHANNELS = pd.DataFrame([
    {"채널": "온라인(오픈마켓)", "채널구분": "온라인", "수수료율": 0.12, "프로모션율": 0.05, "반품율": 0.02},
    {"채널": "자사몰",           "채널구분": "온라인", "수수료율": 0.03, "프로모션율": 0.04, "반품율": 0.02},
    {"채널": "공구",             "채널구분": "온라인", "수수료율": 0.08, "프로모션율": 0.03, "반품율": 0.02},
    {"채널": "홈쇼핑",           "채널구분": "홈쇼핑", "수수료율": 0.30, "프로모션율": 0.05, "반품율": 0.03},
])

# ----------------------------
# Helpers
# ----------------------------
def pct_to_rate(x: float) -> float:
    try:
        return max(0.0, float(x)) / 100.0
    except Exception:
        return 0.0

def krw_round(x: float, unit: int = 100) -> int:
    if x is None:
        return 0
    try:
        x = float(x)
    except Exception:
        return 0
    if np.isnan(x) or np.isinf(x):
        return 0
    return int(round(x / unit) * unit)

def safe_num(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def guardrail_min_price(landed_cost: float, channel_cost_rate: float, min_margin_rate: float) -> float:
    """
    Guardrail(하한) 수식:
    Price*(1-채널비용율) - LandedCost >= Price*최소마진
    => Price*(1-채널비용율-최소마진) >= LandedCost
    => Price >= LandedCost / (1-채널비용율-최소마진)
    """
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

def apply_import_cap(price: float, import_ref: float, premium_rate: float) -> float:
    """
    직구캡:
    Price <= ImportRef*(1+Premium)
    """
    cap = import_ref * (1.0 + premium_rate)
    return min(price, cap)

def ladder_prices(list_price: float, min_disc_pct: float, tgt_disc_pct: float, max_disc_pct: float):
    """
    할인율 밴드 -> 가격 밴드 변환
    - 상단(Max): 최소 할인율(min_disc) 적용 (덜 깎임)
    - 타겟(Target): 목표 할인율(tgt_disc)
    - 하단(Min): 최대 할인율(max_disc) 적용 (가장 많이 깎임)
    """
    p_max = list_price * (1.0 - pct_to_rate(min_disc_pct))
    p_tgt = list_price * (1.0 - pct_to_rate(tgt_disc_pct))
    p_min = list_price * (1.0 - pct_to_rate(max_disc_pct))
    return p_min, p_tgt, p_max

# ----------------------------
# 공식가 산정 엔진
# ----------------------------
def engine_market_anchor(market_mid: float, market_high: float, position: str) -> float:
    """
    시장앵커형: 시장 분포로 공식가를 잡는 매우 단순한 MVP 룰
    - 중앙: MarketMid
    - 중상: (MarketMid+MarketHigh)/2
    - 상단: MarketHigh
    """
    if position == "중앙":
        return market_mid
    if position == "중상":
        return (market_mid + market_high) / 2
    return market_high

def engine_list_discount(comp_list_mid: float, comp_list_high: float, position: str) -> float:
    """
    정가-할인형: 경쟁 '정가' 분포 기반 공식가 MVP 룰
    - 비슷: CompListMid
    - 중간: (CompListMid+CompListHigh)/2
    - 높게: CompListHigh
    """
    if position == "비슷":
        return comp_list_mid
    if position == "높게":
        return comp_list_high
    return (comp_list_mid + comp_list_high) / 2

def engine_cost_margin(landed_cost: float, channel_cost_rate: float, target_margin_rate: float) -> float:
    """
    원가-마진형(공식가용):
    Price*(1-채널비용율-목표마진) >= LandedCost
    => Price = LandedCost/(1-채널비용율-목표마진)
    """
    denom = 1.0 - channel_cost_rate - target_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

# ----------------------------
# Core compute
# ----------------------------
def ladder_validate(df: pd.DataFrame):
    d = df.copy()
    for col in ["MinDisc%", "TargetDisc%", "MaxDisc%"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    # MinDisc <= TargetDisc <= MaxDisc
    d["TargetDisc%"] = np.maximum(d["TargetDisc%"], d["MinDisc%"])
    d["MaxDisc%"] = np.maximum(d["MaxDisc%"], d["TargetDisc%"])
    return d

def compute_base_prices(
    landed_cost: float,
    min_margin_rate: float,
    channels_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    engine_name: str,         # "시장앵커형" | "정가-할인형" | "원가-마진형"
    list_price_policy: str,   # "전채널 동일" | "채널별(원가반영)"
    market_mid: float | None,
    market_high: float | None,
    market_pos: str,
    comp_list_mid: float | None,
    comp_list_high: float | None,
    comp_list_pos: str,
    target_margin_for_list_rate: float | None,
    use_import_cap: bool,
    import_ref: float | None,
    import_premium_rate: float,
    hs_price: float,
):
    df = channels_df.copy()

    if "채널구분" not in df.columns:
        df["채널구분"] = "온라인"

    for col in ["수수료율", "프로모션율", "반품율"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 채널 비용율(합) = 수수료 + 프로모션 + 반품비용(%)  (MVP에서는 %로 단순 합산)
    df["채널비용율(합)"] = df["수수료율"] + df["프로모션율"] + df["반품율"]

    # 하한(guardrail) 계산
    df["GuardrailMin"] = df["채널비용율(합)"].apply(lambda r: guardrail_min_price(landed_cost, float(r), min_margin_rate))

    # 공식가가 "전채널 동일"일 경우 shared_list 먼저 계산
    shared_list = None
    if list_price_policy == "전채널 동일":
        if engine_name == "시장앵커형":
            shared_list = engine_market_anchor(safe_num(market_mid), safe_num(market_high), market_pos)
        elif engine_name == "정가-할인형":
            shared_list = engine_list_discount(safe_num(comp_list_mid), safe_num(comp_list_high), comp_list_pos)
        else:
            # 원가-마진형의 전채널 동일은 대표 채널(첫 행)의 비용율로 계산 (MVP)
            base_rate = float(df.iloc[0]["채널비용율(합)"]) if len(df) else 0.0
            tmr = target_margin_for_list_rate if target_margin_for_list_rate is not None else (min_margin_rate + 0.15)
            shared_list = engine_cost_margin(landed_cost, base_rate, tmr)

        if use_import_cap and import_ref is not None:
            shared_list = apply_import_cap(shared_list, float(import_ref), import_premium_rate)

    ladder_df = ladder_validate(ladder_df)

    rows = []
    for _, r in df.iterrows():
        channel = str(r.get("채널", "")).strip() or "채널"
        group = str(r.get("채널구분", "온라인")).strip() or "온라인"
        cost_rate = float(r["채널비용율(합)"])
        guard_min = float(r["GuardrailMin"])

        # 홈쇼핑 채널: 홈쇼핑가 단일 (하한과 비교해 더 큰 값)
        if group == "홈쇼핑":
            hs_final = max(float(hs_price), guard_min)
            hs_final = krw_round(hs_final, 100)
            rows.append({"채널": channel, "채널구분": group, "가격타입": "홈쇼핑가", "Min": hs_final, "Target": hs_final, "Max": hs_final})
            continue

        # 공식가(노출) 계산
        if list_price_policy == "전채널 동일":
            list_price = float(shared_list)
        else:
            if engine_name == "시장앵커형":
                list_price = engine_market_anchor(safe_num(market_mid), safe_num(market_high), market_pos)
            elif engine_name == "정가-할인형":
                list_price = engine_list_discount(safe_num(comp_list_mid), safe_num(comp_list_high), comp_list_pos)
            else:
                tmr = target_margin_for_list_rate if target_margin_for_list_rate is not None else (min_margin_rate + 0.15)
                list_price = engine_cost_margin(landed_cost, cost_rate, tmr)

            if use_import_cap and import_ref is not None:
                list_price = apply_import_cap(list_price, float(import_ref), import_premium_rate)

        # 공식가도 guardrail 이하로 내려가면 의미 없으므로 guardrail로 끌어올림(MVP 안전장치)
        list_price = max(float(list_price), guard_min)
        list_price = krw_round(list_price, 100)

        # 공식가(노출) 행
        rows.append({"채널": channel, "채널구분": group, "가격타입": "공식가(노출)", "Min": list_price, "Target": list_price, "Max": list_price})

        # 온라인 채널에만 5단계 가격레벨 적용
        if group == "온라인":
            for _, lad in ladder_df.iterrows():
                t = str(lad.get("가격타입", "")).strip()
                if t not in PRICE_TYPES_ONLINE:
                    continue

                pmin, ptgt, pmax = ladder_prices(
                    list_price,
                    safe_num(lad.get("MinDisc%", 0.0)),
                    safe_num(lad.get("TargetDisc%", 0.0)),
                    safe_num(lad.get("MaxDisc%", 0.0)),
                )

                # 각 가격레벨도 guardrail 아래로 못 내려가게 방어
                pmin = max(float(pmin), guard_min)
                ptgt = max(float(ptgt), pmin)
                pmax = max(float(pmax), ptgt)

                rows.append({
                    "채널": channel,
                    "채널구분": group,
                    "가격타입": t,
                    "Min": krw_round(pmin, 100),
                    "Target": krw_round(ptgt, 100),
                    "Max": krw_round(pmax, 100),
                })

    out = pd.DataFrame(rows)

    # 정렬용
    type_order = ["홈쇼핑가", "공식가(노출)"] + PRICE_TYPES_ONLINE
    out["__type_order"] = out["가격타입"].apply(lambda x: type_order.index(x) if x in type_order else 999)
    out = out.sort_values(by=["채널구분", "채널", "__type_order"]).drop(columns=["__type_order"]).reset_index(drop=True)
    return out

def apply_home_shopping_period_uplift(
    out_base: pd.DataFrame,
    channels_df: pd.DataFrame,
    hs_price: float,
    uplift_pct: float,
    apply_to_types: list[str] | None = None,
):
    """
    홈쇼핑 기간 운영가:
    온라인 가격레벨(상시/홈사/라방/브위/원데이)에 대해
    floor = 홈쇼핑가 * (1 + uplift_pct) 를 적용하여 Min/Target/Max를 바닥값으로 끌어올림
    """
    if apply_to_types is None:
        apply_to_types = PRICE_TYPES_ONLINE

    floor = float(hs_price) * (1.0 + float(uplift_pct) / 100.0)

    if "채널구분" not in channels_df.columns:
        online_channels = set(channels_df["채널"].astype(str).tolist())
    else:
        online_channels = set(channels_df.loc[channels_df["채널구분"] == "온라인", "채널"].astype(str).tolist())

    out = out_base.copy()
    mask = out["채널"].astype(str).isin(online_channels) & out["가격타입"].astype(str).isin(apply_to_types)

    for col in ["Min", "Target", "Max"]:
        out.loc[mask, col] = (
            out.loc[mask, col]
            .astype(float)
            .apply(lambda x: max(x, floor))
            .apply(lambda x: krw_round(x, 100))
        )

    # 단조성 보정 (Min <= Target <= Max)
    out.loc[mask, "Target"] = np.maximum(out.loc[mask, "Target"].astype(float), out.loc[mask, "Min"].astype(float))
    out.loc[mask, "Max"] = np.maximum(out.loc[mask, "Max"].astype(float), out.loc[mask, "Target"].astype(float))
    out.loc[mask, "Target"] = out.loc[mask, "Target"].apply(lambda x: krw_round(x, 100))
    out.loc[mask, "Max"] = out.loc[mask, "Max"].apply(lambda x: krw_round(x, 100))

    return out

# ----------------------------
# Cannibal warnings
# ----------------------------
def cannibal_warnings(df: pd.DataFrame):
    """
    같은 채널 내에서 Target은
    공식가 >= 상시 >= 홈사 >= 라방 >= 브위 >= 원데이
    형태로 내려가야 정상(할인이 더 커지니까).
    역전되면 경고.
    """
    order = ["공식가(노출)"] + PRICE_TYPES_ONLINE
    warnings = []

    for ch in df["채널"].unique():
        sub = df[(df["채널"] == ch) & (df["가격타입"].isin(order))].copy()
        if sub.empty:
            continue
        sub = sub.set_index("가격타입")

        prev = None
        for t in order:
            if t not in sub.index:
                continue
            cur = int(sub.loc[t, "Target"])
            if prev is not None and cur > prev:
                warnings.append(f"[{ch}] '{t}' Target({cur:,}원)가 상위 단계보다 높음 → 레벨/할인율 역전 가능")
            prev = cur

    return warnings

# ----------------------------
# HTML/CSS Range Visualization (theme-aware)
# ----------------------------
def render_range_bars(df: pd.DataFrame, title: str, height_px: int = 18):
    st.markdown(f"### {title}")

    d = df.copy()
    d["label"] = d["채널"].astype(str) + " | " + d["가격타입"].astype(str)

    gmin = float(d["Min"].min()) if len(d) else 0.0
    gmax = float(d["Max"].max()) if len(d) else 1.0
    span = max(1.0, gmax - gmin)

    # Streamlit theme base: "light"/"dark" (may be None)
    theme_base = str(st.get_option("theme.base") or "").lower()

    # 대비 색상: 다크면 밝게, 라이트면 어둡게
    if theme_base == "dark":
        label_color = "rgba(255,255,255,0.88)"
        num_color = "rgba(255,255,255,0.70)"
        box_bg = "rgba(255,255,255,0.10)"
        seg_bg = "rgba(255,255,255,0.35)"
        seg_border = "rgba(255,255,255,0.20)"
        dot_color = "rgba(255,255,255,0.95)"
        grid_color = "rgba(255,255,255,0.12)"
    else:
        label_color = "rgba(0,0,0,0.85)"
        num_color = "rgba(0,0,0,0.55)"
        box_bg = "rgba(0,0,0,0.06)"
        seg_bg = "rgba(0,0,0,0.28)"
        seg_border = "rgba(0,0,0,0.10)"
        dot_color = "rgba(0,0,0,0.85)"
        grid_color = "rgba(0,0,0,0.08)"

    st.markdown(
        f"""
<style>
.range-wrap {{ border-top: 1px solid {grid_color}; padding-top: 6px; }}
.range-row {{ display:flex; align-items:center; gap:12px; padding:6px 0; }}
.range-label {{
  width: 320px; font-size: 13px; color: {label_color};
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.range-box {{
  position: relative; height: {height_px}px; flex: 1;
  background: {box_bg}; border-radius: 999px;
  outline: 1px solid {grid_color};
}}
.range-seg {{
  position: absolute; height: 100%; border-radius: 999px;
  background: {seg_bg};
  outline: 1px solid {seg_border};
}}
.range-dot {{
  position: absolute; top: -4px; width: 0; height: 0;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-bottom: 9px solid {dot_color};
}}
.range-nums {{
  width: 290px; font-size: 12px; color: {num_color}; text-align:right; white-space: nowrap;
}}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="range-wrap">', unsafe_allow_html=True)

    for _, r in d.iterrows():
        label = r["label"]
        vmin = float(r["Min"])
        vtgt = float(r["Target"])
        vmax = float(r["Max"])

        left_pct = (vmin - gmin) / span * 100.0
        width_pct = (vmax - vmin) / span * 100.0
        dot_pct = (vtgt - gmin) / span * 100.0

        nums = f"{int(vmin):,} / {int(vtgt):,} / {int(vmax):,}원"

        html = f"""
<div class="range-row">
  <div class="range-label" title="{label}">{label}</div>
  <div class="range-box">
    <div class="range-seg" style="left:{left_pct:.2f}%; width:{max(0.8, width_pct):.2f}%;"></div>
    <div class="range-dot" style="left: calc({dot_pct:.2f}% - 6px);"></div>
  </div>
  <div class="range-nums">{nums}</div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# UI
# ============================================================
st.title("Dynamic Pricing Simulator")
st.caption("공식가(노출) + 온라인 가격레벨 범위 + 홈쇼핑 기간 운영가(+X%) / 도식화는 외부 라이브러리 없이 HTML/CSS")

# Session init
if "channels_df" not in st.session_state:
    st.session_state["channels_df"] = DEFAULT_CHANNELS.copy()
if "ladder_df" not in st.session_state:
    st.session_state["ladder_df"] = DEFAULT_LADDER.copy()

# Sidebar
with st.sidebar:
    st.header("공통 입력")
    landed_cost = st.number_input("Landed Cost (원)", min_value=0, value=12000, step=100)
    min_margin_pct = st.number_input("최소 마진(%) [하한용]", min_value=0.0, max_value=80.0, value=15.0, step=0.5)

    st.divider()
    st.header("공식가 산정 엔진")
    engine_name = st.radio("엔진 선택", ["시장앵커형", "정가-할인형", "원가-마진형"], index=0)
    list_price_policy = st.radio("공식가 정책", ["전채널 동일", "채널별(원가반영)"], index=0)

    st.divider()
    st.header("직구 캡(옵션)")
    use_import_cap = st.checkbox("직구가 캡 적용", value=True)
    import_ref = None
    import_premium_pct = 0.0
    if use_import_cap:
        import_ref = st.number_input("직구 실구매가 기준(원)", min_value=0, value=35000, step=500)
        import_premium_pct = st.number_input("허용 프리미엄(%)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)

    st.divider()
    st.header("홈쇼핑 이벤트(기간 운영)")
    hs_enable = st.checkbox("홈쇼핑 기간 운영가 계산", value=True)
    hs_start = st.date_input("홈쇼핑 시작일", value=date.today())
    hs_end = st.date_input("홈쇼핑 종료일", value=date.today())
    hs_price = st.number_input("홈쇼핑 판매가(원) [직접 입력]", min_value=0, value=29900, step=100)
    hs_uplift_pct = st.number_input("홈쇼핑가 대비 온라인 최소 +%(업)", min_value=0.0, max_value=300.0, value=10.0, step=0.5)

# Channel input
st.subheader("채널 입력 (온라인만 가격레벨 적용, 홈쇼핑은 홈쇼핑가 단일)")
channels_df = st.data_editor(
    st.session_state["channels_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "채널": st.column_config.TextColumn(required=True),
        "채널구분": st.column_config.SelectboxColumn("채널구분", options=["온라인", "홈쇼핑", "기타"], required=True),
        "수수료율": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, help="예: 0.12=12%"),
        "프로모션율": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
        "반품율": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    },
)
st.session_state["channels_df"] = channels_df

# Ladder input (online only)
st.subheader("온라인 가격레벨 할인율 밴드 (온라인 채널에만 적용)")
ladder_df = st.data_editor(
    st.session_state["ladder_df"],
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "가격타입": st.column_config.TextColumn(disabled=True),
        "MinDisc%": st.column_config.NumberColumn(min_value=0.0, max_value=95.0, step=0.5, help="최소 할인율(=상단 가격)"),
        "TargetDisc%": st.column_config.NumberColumn(min_value=0.0, max_value=95.0, step=0.5),
        "MaxDisc%": st.column_config.NumberColumn(min_value=0.0, max_value=95.0, step=0.5, help="최대 할인율(=하단 가격)"),
    },
)
st.session_state["ladder_df"] = ladder_df

# Engine-specific inputs
st.subheader("리서치/엔진 입력")
c1, c2, c3 = st.columns(3)

market_mid = market_high = None
market_pos = "중앙"
comp_list_mid = comp_list_high = None
comp_list_pos = "중간"
target_margin_for_list_pct = None

with c1:
    if engine_name == "시장앵커형":
        market_mid = st.number_input("국내 시장가 중앙/평균(원)", min_value=0, value=39000, step=500)
        market_high = st.number_input("국내 시장가 상단(원)", min_value=0, value=49000, step=500)
        market_pos = st.selectbox("공식가 포지션", ["중앙", "중상", "상단"], index=1)
    else:
        st.caption("시장앵커형 선택 시 입력")

with c2:
    if engine_name == "정가-할인형":
        comp_list_mid = st.number_input("경쟁 정가 중앙/평균(원)", min_value=0, value=49000, step=500)
        comp_list_high = st.number_input("경쟁 정가 상단(원)", min_value=0, value=59000, step=500)
        comp_list_pos = st.selectbox("우리 공식가(정가) 포지션", ["비슷", "중간", "높게"], index=1)
    else:
        st.caption("정가-할인형 선택 시 입력")

with c3:
    if engine_name == "원가-마진형":
        target_margin_for_list_pct = st.number_input("공식가용 목표마진(%)", min_value=0.0, max_value=90.0, value=30.0, step=0.5)
        st.caption("전채널 동일이면 '대표 채널(첫 행)' 비용율 기준으로 공식가를 산정합니다.")
    else:
        st.caption("원가-마진형 선택 시 입력")

# Compute
min_margin_rate = pct_to_rate(min_margin_pct)
import_premium_rate = pct_to_rate(import_premium_pct)
target_margin_for_list_rate = pct_to_rate(target_margin_for_list_pct) if target_margin_for_list_pct is not None else None

out_base = compute_base_prices(
    landed_cost=float(landed_cost),
    min_margin_rate=min_margin_rate,
    channels_df=channels_df,
    ladder_df=ladder_df,
    engine_name=engine_name,
    list_price_policy=list_price_policy,
    market_mid=market_mid,
    market_high=market_high,
    market_pos=market_pos,
    comp_list_mid=comp_list_mid,
    comp_list_high=comp_list_high,
    comp_list_pos=comp_list_pos,
    target_margin_for_list_rate=target_margin_for_list_rate,
    use_import_cap=use_import_cap,
    import_ref=float(import_ref) if (use_import_cap and import_ref is not None) else None,
    import_premium_rate=import_premium_rate,
    hs_price=float(hs_price),
)

out_hs = None
if hs_enable:
    out_hs = apply_home_shopping_period_uplift(
        out_base=out_base,
        channels_df=channels_df,
        hs_price=float(hs_price),
        uplift_pct=float(hs_uplift_pct),
        apply_to_types=PRICE_TYPES_ONLINE,
    )

# Output tables
st.divider()
st.subheader("결과 테이블: 평시(기본) 가격 밴드")
st.dataframe(out_base, use_container_width=True)

warns_base = cannibal_warnings(out_base[out_base["채널구분"] == "온라인"])
if warns_base:
    st.warning("평시(기본) 레벨 역전 가능 경고:\n- " + "\n- ".join(warns_base))
else:
    st.success("평시(기본) 온라인 레벨 순서가 정상 범위로 보입니다.")

if hs_enable and out_hs is not None:
    st.subheader(f"결과 테이블: 홈쇼핑 기간 운영가 (기간: {hs_start} ~ {hs_end})")
    st.caption(f"온라인 가격레벨은 홈쇼핑가({krw_round(hs_price):,}원) 대비 최소 +{hs_uplift_pct}% 이상으로 자동 상향됩니다.")
    st.dataframe(out_hs, use_container_width=True)

    warns_hs = cannibal_warnings(out_hs[out_hs["채널구분"] == "온라인"])
    if warns_hs:
        st.warning("홈쇼핑 기간 운영가 레벨 역전 가능 경고:\n- " + "\n- ".join(warns_hs))
    else:
        st.success("홈쇼핑 기간 운영가에서도 온라인 레벨 순서가 정상 범위로 보입니다.")

# Tabs: viz + formula
st.divider()
tab1, tab2, tab3 = st.tabs(["도식화: 평시(기본)", "도식화: 홈쇼핑 기간 운영가", "계산식/로직 설명"])

with tab1:
    render_range_bars(out_base, title="평시(기본) 가격 범위 도식화")

with tab2:
    if hs_enable and out_hs is not None:
        render_range_bars(out_hs, title=f"홈쇼핑 기간 운영가 도식화 (+{hs_uplift_pct}% 룰)")
    else:
        st.info("사이드바에서 '홈쇼핑 기간 운영가 계산'을 켜면 표시됩니다.")

with tab3:
    st.markdown("## 1) 입력값 정의(모델이 실제로 쓰는 값)")
    st.markdown(
        """
- **Landed Cost (LC)**: 한국 창고 입고 기준 총원가(원)  
- **채널 비용율(%)**:  
  - 수수료율 = 플랫폼/중개/PG 등  
  - 프로모션율 = 쿠폰/광고/딜 비용을 %로 환산한 값  
  - 반품율 = 반품/클레임 비용을 %로 환산한 값  
- **채널비용율 합**:  \\( C = 수수료율 + 프로모션율 + 반품율 \\)  
- **최소마진(%)**: 하한(손익 방어선)을 만들기 위한 최소 GP(혹은 GM)  
- **직구캡**: 직구 실구매가(배송+관부가 포함) 기준으로 공식가 상단 제한  
- **홈쇼핑가**: 홈쇼핑 채널 판매가(원)  
- **홈쇼핑 uplift**: 홈쇼핑 기간 동안 온라인 가격레벨을 홈쇼핑가 대비 +X% 이상으로 유지하기 위한 규칙
"""
    )

    st.markdown("## 2) 채널 손익 하한(Guardrail) 공식")
    st.markdown(
        r"""
### 목적
어떤 가격전략을 쓰든 **절대 이 이하로는 팔면 안 되는 가격 하한선**을 만듭니다.

### 식
- \\(LC\\): Landed Cost (원)  
- \\(C\\): 채널비용율 합(%) → 비율로 사용(예: 0.12)  
- \\(m\\): 최소마진(%) → 비율로 사용(예: 0.15)  
- \\(P\\): 판매가(원)

채널 비용 차감 후 남는 금액에서 원가를 빼고도 최소마진을 만족해야 하므로,

\\[
P(1 - C) - LC \ge Pm
\\]

정리하면,

\\[
P(1 - C - m) \ge LC
\\]

따라서 **하한 가격(guardrail)**은,

\\[
P_{min} = \frac{LC}{1 - C - m}
\\]

### 코드 적용
- `GuardrailMin = landed_cost / (1 - channel_cost_rate - min_margin_rate)`  
- 분모가 0 이하이면 가격이 무한대로 튀므로(구조적으로 불가능), `inf`로 처리합니다.
"""
    )

    st.markdown("## 3) 공식가(노출) 산정 로직")
    st.markdown(
        """
공식가는 “가격레벨(상시/홈사/라방/브위/원데이)”의 **기준점**입니다.  
현재 MVP에는 3가지 엔진이 있으며, 공식가 정책은 2가지입니다.

### 3-1. 공식가 엔진(선택 1)
1) **시장앵커형**  
- 입력: 국내 시장 중앙값/평균(`MarketMid`), 상단(`MarketHigh`), 포지션(`중앙/중상/상단`)  
- 산식(MVP):
  - 중앙: `List = MarketMid`  
  - 중상: `List = (MarketMid + MarketHigh)/2`  
  - 상단: `List = MarketHigh`

2) **정가-할인형**  
- 입력: 경쟁 정가 중앙/평균(`CompListMid`), 경쟁 정가 상단(`CompListHigh`), 포지션(`비슷/중간/높게`)  
- 산식(MVP):
  - 비슷: `List = CompListMid`
  - 중간: `List = (CompListMid + CompListHigh)/2`
  - 높게: `List = CompListHigh`

3) **원가-마진형(공식가용)**  
- 입력: 공식가용 목표마진(`TargetMarginForList`)  
- 산식:
  - `List = LandedCost / (1 - 채널비용율 - TargetMarginForList)`
"""
    )

    st.markdown(
        """
### 3-2. 공식가 정책(선택 1)
- **전채널 동일**: 모든 비(홈쇼핑) 채널에서 공식가를 동일하게 사용  
  - 원가-마진형인 경우, MVP에선 “첫 번째 채널”의 비용율을 대표로 사용합니다. (추후 대표 채널 선택 UI로 확장 가능)
- **채널별(원가반영)**: 채널마다 공식가를 따로 계산(특히 원가-마진형에서 의미가 큼)

### 3-3. 공식가 안전장치(하한 적용)
- 공식가도 `GuardrailMin`보다 낮게 나오면 의미가 없으므로,
  - `List = max(List, GuardrailMin)` 로 끌어올립니다. (MVP 안전장치)

### 3-4. 직구캡(옵션)
- 입력: 직구 실구매가 기준(`ImportRef`), 허용 프리미엄(`Premium%`)
- 산식:
  - `Cap = ImportRef * (1 + Premium)`
  - `List = min(List, Cap)`  
"""
    )

    st.markdown("## 4) 온라인 가격레벨(상시/홈사/라방/브위/원데이) 밴드 산정")
    st.markdown(
        r"""
온라인 채널에서만 적용됩니다.

### 4-1. 할인율 밴드 입력
가격레벨별로 3개의 할인율을 입력합니다.
- MinDisc%: 최소 할인(덜 깎는 구간) → **가격 상단(Max)**
- TargetDisc%: 목표 할인 → **가격 타겟(Target)**
- MaxDisc%: 최대 할인(가장 많이 깎는 구간) → **가격 하단(Min)**

### 4-2. 할인율 → 가격 변환
- \\(L\\): 공식가(노출)
- \\(d_{min}\\), \\(d_{tgt}\\), \\(d_{max}\\): 할인율(비율)

\\[
P_{max} = L(1-d_{min})
\\]
\\[
P_{tgt} = L(1-d_{tgt})
\\]
\\[
P_{min} = L(1-d_{max})
\\]

### 4-3. 하한(Guardrail) 적용
각 가격레벨도 원가 방어를 위해:
- `Pmin = max(Pmin, GuardrailMin)`
- 그리고 항상 `Min <= Target <= Max`가 되도록:
  - `Target = max(Target, Min)`
  - `Max = max(Max, Target)`
"""
    )

    st.markdown("## 5) 홈쇼핑 채널 가격 산정(단일)")
    st.markdown(
        r"""
홈쇼핑 채널은 온라인 레벨 구조를 적용하지 않고 “홈쇼핑가” 1개로 관리합니다.

- 입력: 홈쇼핑가 \\(H\\)
- 하한(Guardrail) 적용:
\\[
H_{final} = \max(H, GuardrailMin)
\\]

그리고 Min/Target/Max 모두 동일하게 표시합니다.
"""
    )

    st.markdown("## 6) 홈쇼핑 기간 운영가(온라인 상향 규칙)")
    st.markdown(
        r"""
홈쇼핑 기간 동안 온라인 가격 카니발을 방지하기 위해, 온라인 가격레벨을 홈쇼핑가 대비 일정 비율 이상으로 상향합니다.

- 입력:
  - 홈쇼핑가 \\(H\\)
  - uplift \\(u\\) (비율, 예: 0.10)
- 바닥값(Floor):
\\[
Floor = H(1+u)
\\]

온라인 가격레벨(상시/홈사/라방/브위/원데이)의 Min/Target/Max에 대해:
- `Value = max(Value, Floor)` 를 적용합니다.
- 이후에도 `Min <= Target <= Max`를 유지하도록 동일한 단조성 보정을 수행합니다.
"""
    )

    st.markdown("## 7) 라운딩(가격 단위 정리)")
    st.markdown(
        """
현재 코드는 모든 계산 결과를 **100원 단위 반올림**합니다.
- `krw_round(x, unit=100)`  
원하면 10원/1000원 단위로 쉽게 변경 가능.
"""
    )

    st.markdown("## 8) 카니발(레벨 역전) 경고 로직")
    st.markdown(
        """
온라인 채널에서 **Target 기준**으로 아래 순서가 깨지면 경고합니다.

- 정상 기대 순서(가격이 내려가야 정상):
  - 공식가 ≥ 상시할인가 ≥ 홈사할인가 ≥ 모바일라방가 ≥ 브랜드위크가 ≥ 원데이 특가

만약 어떤 레벨의 Target이 상위 레벨보다 높으면(즉, 할인 구조가 역전되면)
- 경고 메시지로 표시합니다.

※ 현재는 “같은 채널 내부”만 체크합니다.  
원하면 다음 단계로 “채널 간 교차 카니발(홈쇼핑 vs 온라인)” 규칙도 추가할 수 있습니다.
"""
    )

    st.markdown("## 9) 현재 MVP 모델의 한계와 확장 포인트(중요)")
    st.markdown(
        """
이 MVP는 “키인/운영이 쉬운 것”을 목표로 일부를 단순화했습니다.

### 단순화된 부분
- 채널비용율을 `수수료+프로모션+반품`의 단순 합으로 처리
- 프로모션/반품은 본래 “정액 비용/건당 비용”이 섞이는데, 현재는 %로 환산해 입력 받음
- 원가-마진형에서 ‘전채널 동일 공식가’는 첫 채널 비용율을 대표로 사용

### 확장 포인트(필요 시)
- 프로모션 비용을 “정액+% 혼합”으로 입력 가능
- 반품을 “반품률 × (회수비+폐기비+재판매율)” 구조로 쪼개기
- 홈쇼핑 기간 uplift 적용 대상을 체크박스로 선택(상시/홈사만 등)
- 채널 간 카니발 체크(예: 온라인 원데이 Max가 홈쇼핑가보다 낮으면 경고)
"""
    )

st.info("✅ 도식화가 까맣게 보이던 문제는 다크모드용 대비색을 적용해서 해결했습니다. (라벨/바/점 모두 테마 대응)")
