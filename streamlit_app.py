# streamlit_app.py
# ============================================================
# Pricing Simulator (Strict Channel Taxonomy)
# - 온라인: 공식가(노출) + 5레벨(상시/홈사/라방/브위/원데이)
# - 공구: 공구가만
# - 홈쇼핑: 홈쇼핑가만
# - 홈쇼핑 기간: 온라인 5레벨을 "홈쇼핑가 대비 +X%" 이상으로 바닥 상향
# - 외부 시각화 라이브러리 없음 (HTML/CSS)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(page_title="Pricing Simulator", layout="wide")

ONLINE_LEVELS = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]
ONLINE_ONLY_TYPES = ["공식가(노출)"] + ONLINE_LEVELS

DEFAULT_LADDER = pd.DataFrame([
    {"가격타입": "상시할인가",   "MinDisc%": 15.0, "TargetDisc%": 20.0, "MaxDisc%": 25.0},
    {"가격타입": "홈사할인가",   "MinDisc%": 18.0, "TargetDisc%": 25.0, "MaxDisc%": 32.0},
    {"가격타입": "모바일라방가", "MinDisc%": 20.0, "TargetDisc%": 28.0, "MaxDisc%": 35.0},
    {"가격타입": "브랜드위크가", "MinDisc%": 25.0, "TargetDisc%": 33.0, "MaxDisc%": 40.0},
    {"가격타입": "원데이 특가",  "MinDisc%": 30.0, "TargetDisc%": 40.0, "MaxDisc%": 50.0},
])

DEFAULT_CHANNELS = pd.DataFrame([
    {"채널": "온라인(오픈마켓)", "채널구분": "온라인", "고정가(원)": "", "수수료율": 0.12, "프로모션율": 0.05, "반품율": 0.02},
    {"채널": "자사몰",           "채널구분": "온라인", "고정가(원)": "", "수수료율": 0.03, "프로모션율": 0.04, "반품율": 0.02},
    {"채널": "공구",             "채널구분": "공구",   "고정가(원)": 29900, "수수료율": 0.08, "프로모션율": 0.03, "반품율": 0.02},
    {"채널": "홈쇼핑",           "채널구분": "홈쇼핑", "고정가(원)": 29900, "수수료율": 0.30, "프로모션율": 0.05, "반품율": 0.03},
])

# ----------------------------
# Helpers
# ----------------------------
def pct_to_rate(x) -> float:
    try:
        return max(0.0, float(x)) / 100.0
    except Exception:
        return 0.0

def krw_round(x: float, unit: int = 100) -> int:
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return 0
        return int(round(x / unit) * unit)
    except Exception:
        return 0

def safe_num(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def guardrail_min_price(landed_cost: float, channel_cost_rate: float, min_margin_rate: float) -> float:
    # Price*(1-C) - LC >= Price*m  ->  Price >= LC / (1-C-m)
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

def ladder_prices(list_price: float, min_disc_pct: float, tgt_disc_pct: float, max_disc_pct: float):
    # 할인율이 클수록 가격은 낮아짐
    p_max = list_price * (1.0 - pct_to_rate(min_disc_pct))   # 상단(덜 할인)
    p_tgt = list_price * (1.0 - pct_to_rate(tgt_disc_pct))
    p_min = list_price * (1.0 - pct_to_rate(max_disc_pct))   # 하단(많이 할인)
    return p_min, p_tgt, p_max

def apply_import_cap(price: float, import_ref: float, premium_rate: float) -> float:
    cap = import_ref * (1.0 + premium_rate)
    return min(price, cap)

def ladder_validate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["MinDisc%", "TargetDisc%", "MaxDisc%"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    # MinDisc <= TargetDisc <= MaxDisc
    d["TargetDisc%"] = np.maximum(d["TargetDisc%"], d["MinDisc%"])
    d["MaxDisc%"] = np.maximum(d["MaxDisc%"], d["TargetDisc%"])
    return d

# ----------------------------
# Official price engines (ONLINE ONLY)
# ----------------------------
def engine_market_anchor(market_mid: float, market_high: float, position: str) -> float:
    if position == "중앙":
        return market_mid
    if position == "중상":
        return (market_mid + market_high) / 2
    return market_high

def engine_list_anchor(comp_list_mid: float, comp_list_high: float, position: str) -> float:
    if position == "비슷":
        return comp_list_mid
    if position == "높게":
        return comp_list_high
    return (comp_list_mid + comp_list_high) / 2

def engine_cost_margin(landed_cost: float, channel_cost_rate: float, target_margin_rate: float) -> float:
    denom = 1.0 - channel_cost_rate - target_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

# ----------------------------
# Core compute
# ----------------------------
def compute_prices(
    landed_cost: float,
    min_margin_rate: float,
    channels_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    # official price (online only)
    engine_name: str,        # "시장앵커형" | "정가앵커형" | "원가-마진형"
    list_price_policy: str,  # "온라인채널별" | "온라인공통"
    market_mid: float | None,
    market_high: float | None,
    market_pos: str,
    comp_list_mid: float | None,
    comp_list_high: float | None,
    comp_list_pos: str,
    target_margin_for_list_rate: float | None,
    # import cap (optional)
    use_import_cap: bool,
    import_ref: float | None,
    import_premium_rate: float,
):
    df = channels_df.copy()

    # normalize
    if "채널구분" not in df.columns:
        df["채널구분"] = "온라인"
    if "고정가(원)" not in df.columns:
        df["고정가(원)"] = ""

    for col in ["수수료율", "프로모션율", "반품율"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["채널비용율(합)"] = df["수수료율"] + df["프로모션율"] + df["반품율"]
    df["GuardrailMin"] = df["채널비용율(합)"].apply(lambda r: guardrail_min_price(landed_cost, float(r), min_margin_rate))

    ladder_df = ladder_validate(ladder_df)

    # online common official price (optional)
    online_common_list = None
    if list_price_policy == "온라인공통":
        # 대표 온라인 채널: 첫 번째 온라인 행
        online_rows = df[df["채널구분"] == "온라인"]
        base_rate = float(online_rows.iloc[0]["채널비용율(합)"]) if len(online_rows) else 0.0

        if engine_name == "시장앵커형":
            online_common_list = engine_market_anchor(safe_num(market_mid), safe_num(market_high), market_pos)
        elif engine_name == "정가앵커형":
            online_common_list = engine_list_anchor(safe_num(comp_list_mid), safe_num(comp_list_high), comp_list_pos)
        else:
            tmr = target_margin_for_list_rate if target_margin_for_list_rate is not None else (min_margin_rate + 0.15)
            online_common_list = engine_cost_margin(landed_cost, base_rate, tmr)

        if use_import_cap and import_ref is not None:
            online_common_list = apply_import_cap(online_common_list, float(import_ref), import_premium_rate)

    rows = []

    for _, r in df.iterrows():
        ch = str(r.get("채널", "")).strip() or "채널"
        group = str(r.get("채널구분", "온라인")).strip() or "온라인"
        guard_min = float(r["GuardrailMin"])
        cost_rate = float(r["채널비용율(합)"])

        # ---------- NON-ONLINE: single price only ----------
        if group in ["공구", "홈쇼핑", "기타"]:
            fixed = safe_num(r.get("고정가(원)"), 0.0)
            final = max(fixed, guard_min) if fixed > 0 else guard_min
            final = krw_round(final, 100)

            price_type = "공구가" if group == "공구" else ("홈쇼핑가" if group == "홈쇼핑" else "기타채널가")
            rows.append({"채널": ch, "채널구분": group, "가격타입": price_type, "Min": final, "Target": final, "Max": final})
            continue

        # ---------- ONLINE: official + 5 levels ----------
        # official calc (ONLINE ONLY)
        if list_price_policy == "온라인공통" and online_common_list is not None:
            list_price = float(online_common_list)
        else:
            if engine_name == "시장앵커형":
                list_price = engine_market_anchor(safe_num(market_mid), safe_num(market_high), market_pos)
            elif engine_name == "정가앵커형":
                list_price = engine_list_anchor(safe_num(comp_list_mid), safe_num(comp_list_high), comp_list_pos)
            else:
                tmr = target_margin_for_list_rate if target_margin_for_list_rate is not None else (min_margin_rate + 0.15)
                list_price = engine_cost_margin(landed_cost, cost_rate, tmr)

            if use_import_cap and import_ref is not None:
                list_price = apply_import_cap(list_price, float(import_ref), import_premium_rate)

        # official guardrail
        list_price = max(float(list_price), guard_min)
        list_price = krw_round(list_price, 100)

        # ONLINE OFFICIAL (online only)
        rows.append({"채널": ch, "채널구분": group, "가격타입": "공식가(노출)", "Min": list_price, "Target": list_price, "Max": list_price})

        # ONLINE LEVELS
        for _, lad in ladder_df.iterrows():
            t = str(lad.get("가격타입", "")).strip()
            if t not in ONLINE_LEVELS:
                continue

            pmin, ptgt, pmax = ladder_prices(
                list_price,
                safe_num(lad.get("MinDisc%", 0.0)),
                safe_num(lad.get("TargetDisc%", 0.0)),
                safe_num(lad.get("MaxDisc%", 0.0)),
            )

            # guardrail + monotonic
            pmin = max(float(pmin), guard_min)
            ptgt = max(float(ptgt), pmin)
            pmax = max(float(pmax), ptgt)

            rows.append({
                "채널": ch,
                "채널구분": group,
                "가격타입": t,
                "Min": krw_round(pmin, 100),
                "Target": krw_round(ptgt, 100),
                "Max": krw_round(pmax, 100),
            })

    out = pd.DataFrame(rows)

    # sort
    type_order = ["공구가", "홈쇼핑가", "기타채널가"] + ["공식가(노출)"] + ONLINE_LEVELS
    out["__type_order"] = out["가격타입"].apply(lambda x: type_order.index(x) if x in type_order else 999)
    out = out.sort_values(by=["채널구분", "채널", "__type_order"]).drop(columns=["__type_order"]).reset_index(drop=True)
    return out

def apply_hs_period_uplift(out_base: pd.DataFrame, uplift_pct: float) -> pd.DataFrame:
    """
    홈쇼핑 기간: 온라인 레벨(5개)만 바닥 상향
    기준 홈쇼핑가: out_base의 '홈쇼핑가' 중 최저값(여러 홈쇼핑 채널이 있으면 최저를 기준)
    """
    hs_rows = out_base[out_base["가격타입"] == "홈쇼핑가"]
    if hs_rows.empty:
        return out_base.copy()

    hs_ref = float(hs_rows["Target"].min())
    floor = hs_ref * (1.0 + float(uplift_pct) / 100.0)

    out = out_base.copy()
    mask = (out["채널구분"] == "온라인") & (out["가격타입"].isin(ONLINE_LEVELS))

    for col in ["Min", "Target", "Max"]:
        out.loc[mask, col] = out.loc[mask, col].astype(float).apply(lambda x: krw_round(max(x, floor), 100))

    # monotonic per row
    out.loc[mask, "Target"] = np.maximum(out.loc[mask, "Target"].astype(float), out.loc[mask, "Min"].astype(float))
    out.loc[mask, "Max"] = np.maximum(out.loc[mask, "Max"].astype(float), out.loc[mask, "Target"].astype(float))
    out.loc[mask, "Target"] = out.loc[mask, "Target"].apply(lambda x: krw_round(x, 100))
    out.loc[mask, "Max"] = out.loc[mask, "Max"].apply(lambda x: krw_round(x, 100))

    return out

# ----------------------------
# Theme-safe HTML range bars
# ----------------------------
def render_range_bars(df: pd.DataFrame, title: str):
    st.subheader(title)

    d = df.copy()
    d["label"] = d["채널"].astype(str) + " | " + d["가격타입"].astype(str)

    gmin = float(d["Min"].min()) if len(d) else 0.0
    gmax = float(d["Max"].max()) if len(d) else 1.0
    span = max(1.0, gmax - gmin)

    st.markdown(
        """
<style>
.dp-wrap { border-top: 1px solid rgba(128,128,128,0.35); padding-top: 6px; }
.dp-row { display:flex; align-items:center; gap:12px; padding:6px 0; }
.dp-label {
  width: 360px; font-size: 13px;
  color: var(--text-color) !important;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dp-box {
  position: relative; flex: 1; height: 18px; border-radius: 999px;
  background: var(--secondary-background-color) !important;
  border: 1px solid rgba(128,128,128,0.35);
}
.dp-seg {
  position: absolute; height: 100%; border-radius: 999px;
  background: rgba(255,255,255,0.35);
  border: 1px solid rgba(255,255,255,0.25);
}
.dp-dot {
  position: absolute; top: -4px; width: 0; height: 0;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-bottom: 9px solid var(--text-color) !important;
  filter: drop-shadow(0 1px 1px rgba(0,0,0,0.25));
}
.dp-nums {
  width: 320px; font-size: 12px;
  color: var(--text-color) !important;
  opacity: 0.78;
  text-align: right; white-space: nowrap;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dp-wrap">', unsafe_allow_html=True)

    for _, r in d.iterrows():
        label = r["label"]
        vmin = float(r["Min"]); vtgt = float(r["Target"]); vmax = float(r["Max"])

        left_pct = (vmin - gmin) / span * 100.0
        width_pct = (vmax - vmin) / span * 100.0
        dot_pct = (vtgt - gmin) / span * 100.0

        nums = f"{int(vmin):,} / {int(vtgt):,} / {int(vmax):,}원"

        html = f"""
<div class="dp-row">
  <div class="dp-label" title="{label}">{label}</div>
  <div class="dp-box">
    <div class="dp-seg" style="left:{left_pct:.2f}%; width:{max(0.8, width_pct):.2f}%;"></div>
    <div class="dp-dot" style="left: calc({dot_pct:.2f}% - 6px);"></div>
  </div>
  <div class="dp-nums">{nums}</div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# UI
# ============================================================
st.title("Pricing Simulator")
st.caption("온라인(공식가+5레벨) / 공구(공구가만) / 홈쇼핑(홈쇼핑가만) — 온라인 용어는 온라인에만 생성")

if "channels_df" not in st.session_state:
    st.session_state["channels_df"] = DEFAULT_CHANNELS.copy()
if "ladder_df" not in st.session_state:
    st.session_state["ladder_df"] = DEFAULT_LADDER.copy()

with st.sidebar:
    st.header("공통 입력")
    landed_cost = st.number_input("Landed Cost (원)", min_value=0, value=12000, step=100)
    min_margin_pct = st.number_input("최소 마진(%) [하한용]", min_value=0.0, max_value=80.0, value=15.0, step=0.5)

    st.divider()
    st.header("온라인 공식가 산정")
    engine_name = st.radio("엔진 선택", ["시장앵커형", "정가앵커형", "원가-마진형"], index=0)
    list_price_policy = st.radio("공식가 정책(온라인만)", ["온라인공통", "온라인채널별"], index=0)

    st.divider()
    st.header("직구 캡(옵션, 온라인 공식가에만 적용)")
    use_import_cap = st.checkbox("직구가 캡 적용", value=True)
    import_ref = None
    import_premium_pct = 0.0
    if use_import_cap:
        import_ref = st.number_input("직구 실구매가 기준(원)", min_value=0, value=35000, step=500)
        import_premium_pct = st.number_input("허용 프리미엄(%)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)

    st.divider()
    st.header("홈쇼핑 기간(온라인 카니발 방지)")
    hs_enable = st.checkbox("홈쇼핑 기간 운영가 계산", value=True)
    hs_start = st.date_input("홈쇼핑 시작일", value=date.today())
    hs_end = st.date_input("홈쇼핑 종료일", value=date.today())
    hs_uplift_pct = st.number_input("홈쇼핑가 대비 온라인 최소 +%(업)", min_value=0.0, max_value=300.0, value=10.0, step=0.5)

st.subheader("채널 입력")
channels_df = st.data_editor(
    st.session_state["channels_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "채널": st.column_config.TextColumn(required=True),
        "채널구분": st.column_config.SelectboxColumn("채널구분", options=["온라인", "공구", "홈쇼핑", "기타"], required=True),
        "고정가(원)": st.column_config.NumberColumn(help="공구/홈쇼핑/기타는 여기 입력값만 사용(온라인은 비워도 됨).", step=100),
        "수수료율": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
        "프로모션율": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
        "반품율": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    },
)
st.session_state["channels_df"] = channels_df

st.subheader("온라인 가격레벨 할인율 밴드(온라인에만 적용)")
ladder_df = st.data_editor(
    st.session_state["ladder_df"],
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "가격타입": st.column_config.TextColumn(disabled=True),
        "MinDisc%": st.column_config.NumberColumn(help="최소 할인율(=가격 상단 Max)", min_value=0.0, max_value=95.0, step=0.5),
        "TargetDisc%": st.column_config.NumberColumn(help="목표 할인율(=대표 운영가 Target)", min_value=0.0, max_value=95.0, step=0.5),
        "MaxDisc%": st.column_config.NumberColumn(help="최대 할인율(=가격 하단 Min)", min_value=0.0, max_value=95.0, step=0.5),
    },
)
st.session_state["ladder_df"] = ladder_df

st.subheader("리서치 입력(온라인 공식가 산정용)")
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
        st.caption("시장앵커형일 때만 필요")

with c2:
    if engine_name == "정가앵커형":
        comp_list_mid = st.number_input("경쟁 정가 중앙/평균(원)", min_value=0, value=49000, step=500)
        comp_list_high = st.number_input("경쟁 정가 상단(원)", min_value=0, value=59000, step=500)
        comp_list_pos = st.selectbox("우리 공식가 포지션", ["비슷", "중간", "높게"], index=1)
    else:
        st.caption("정가앵커형일 때만 필요")

with c3:
    if engine_name == "원가-마진형":
        target_margin_for_list_pct = st.number_input("공식가용 목표마진(%)", min_value=0.0, max_value=90.0, value=30.0, step=0.5)
    else:
        st.caption("원가-마진형일 때만 필요")

min_margin_rate = pct_to_rate(min_margin_pct)
import_premium_rate = pct_to_rate(import_premium_pct)
target_margin_for_list_rate = pct_to_rate(target_margin_for_list_pct) if target_margin_for_list_pct is not None else None

out_base = compute_prices(
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
)

out_hs = None
if hs_enable:
    out_hs = apply_hs_period_uplift(out_base, uplift_pct=float(hs_uplift_pct))

st.divider()
st.subheader("결과 테이블: 평시(기본)")
st.dataframe(out_base, use_container_width=True)

if hs_enable and out_hs is not None:
    st.subheader(f"결과 테이블: 홈쇼핑 기간 운영가 (기간: {hs_start} ~ {hs_end})")
    st.caption("※ 홈쇼핑가(최저 홈쇼핑가 기준) 대비 +X% 이상으로 온라인 5레벨만 상향 적용")
    st.dataframe(out_hs, use_container_width=True)

st.divider()
tab1, tab2 = st.tabs(["도식화: 평시(기본)", "도식화: 홈쇼핑 기간 운영가"])

with tab1:
    render_range_bars(out_base, "평시(기본) 가격 범위 도식화")

with tab2:
    if hs_enable and out_hs is not None:
        render_range_bars(out_hs, f"홈쇼핑 기간 운영가 도식화 (+{hs_uplift_pct}% 룰)")
    else:
        st.info("홈쇼핑 기간 운영가 계산을 켜면 표시됩니다.")
