import streamlit as st
import pandas as pd
import numpy as np
import re

# ============================================================
# Pricing Rules Simulator (3 Engines)
# - Online-only terms: 공식가, 상시/홈사/라방/브위/원데이 (온라인 채널에만 생성)
# - Groupbuy: 공구가만
# - Homeshopping: 홈쇼핑가만
# - 3 Engines:
#   1) 공구 앵커 사다리
#   2) 홈쇼핑-공구 최저라인 + 온라인 업룰(카니발 방지)
#   3) 마진 Guardrail + (1 또는 2) 결과 보정
# - No plotly/matplotlib; HTML/CSS range visualization
# ============================================================

st.set_page_config(page_title="Pricing Rules Simulator (3안)", layout="wide")

ONLINE_LEVELS = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]
ONLINE_TYPES = ["공식가(노출)"] + ONLINE_LEVELS
OTHER_TYPES = ["공구가", "홈쇼핑가"]

DEFAULT_PRODUCTS = pd.DataFrame([
    {"브랜드": "아티키", "상품명": "타임 블루 병 250g", "구분": "단품", "구성수량(Q)": 1,
     "랜디드코스트(총원가)": 9351, "홈쇼핑가(세트가)": 50000, "공구가(세트가)": 29600, "공식가(세트가)": 54000},
    {"브랜드": "아티키", "상품명": "[2구] 타임 블루 250g 2병", "구분": "세트", "구성수량(Q)": 2,
     "랜디드코스트(총원가)": 23675, "홈쇼핑가(세트가)": 219000, "공구가(세트가)": 56800, "공식가(세트가)": 99000},
])

# ----------------------------
# Helpers
# ----------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def pct_to_rate(x):
    return float(x) / 100.0

def krw_round(x, unit):
    try:
        return int(round(float(x) / unit) * unit)
    except Exception:
        return 0

def make_band(target_price, band_pct, rounding_unit):
    """
    band_pct = 6% 라면, Target 기준 +-3%로 Min/Max 생성.
    """
    t = float(target_price)
    half = pct_to_rate(band_pct) / 2.0
    pmin = t * (1.0 - half)
    pmax = t * (1.0 + half)
    return krw_round(pmin, rounding_unit), krw_round(t, rounding_unit), krw_round(pmax, rounding_unit)

def guardrail_min_price(landed_cost, channel_cost_rate, min_margin_rate):
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

def parse_units_from_text(text):
    """
    아주 러프한 구성 파서.
    - '2병', '4박스', '6통', 'x 2개' 등에서 숫자를 하나라도 찾으면 그걸 Q로 제안
    - 실패하면 None
    """
    if not isinstance(text, str) or not text.strip():
        return None
    m = re.search(r'(\d+)\s*(병|박스|통|개|세트)', text)
    if m:
        return int(m.group(1))
    m2 = re.search(r'x\s*(\d+)', text.lower())
    if m2:
        return int(m2.group(1))
    return None

# ----------------------------
# Visualization: Theme-safe HTML range bars
# ----------------------------
def render_range_bars(df, title):
    st.subheader(title)

    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    d = df.copy()
    d["label"] = d["브랜드"].astype(str) + " | " + d["상품명"].astype(str) + " | " + d["가격타입"].astype(str)

    gmin = float(d["Min"].min())
    gmax = float(d["Max"].max())
    span = max(1.0, gmax - gmin)

    st.markdown(
        """
<style>
.dp-wrap { border-top: 1px solid rgba(128,128,128,0.35); padding-top: 6px; }
.dp-row { display:flex; align-items:center; gap:12px; padding:6px 0; }
.dp-label {
  width: 520px; font-size: 13px;
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
# 3 Engines (룰 3안)
# ============================================================

def engine1_groupbuy_ladder(unit_G, a, b, c, hs_premium_over_G):
    """
    1안: 공구 앵커 사다리 (단위가격 기준)
    - 공구(Unit) = G
    - 모바일(Unit) = G*(1+a)
    - 행사(Unit) = G*(1+b)
    - 상시(Unit) = G*(1+c)
    - 홈쇼핑(Unit) = 공구(Unit)*(1+hs_premium_over_G)  (공구가 홈쇼핑보다 싸야하면 premium을 +로 둠)
    """
    G = float(unit_G)
    hs = G * (1.0 + hs_premium_over_G)
    mob = G * (1.0 + a)
    evt = G * (1.0 + b)
    always = G * (1.0 + c)
    return {
        "공구가": G,
        "홈쇼핑가": hs,
        "모바일라방가": mob,
        "브랜드위크가": evt,     # 이벤트성(브위)을 행사로 매핑
        "원데이 특가": max(G, evt * 0.85),  # 원데이를 행사보다 더 아래로 기본 설정 (조정 가능)
        "홈사할인가": max(mob, evt * 0.95), # 홈사 = 라방~행사 사이 기본
        "상시할인가": always,
    }

def engine2_hs_floor_with_uplift(unit_H, d, u, a, b, c):
    """
    2안: 홈쇼핑-공구 최저라인 + 온라인 업룰
    - 홈쇼핑(Unit) = H (입력)
    - 공구(Unit) = H*(1-d)  (공구가 홈쇼핑보다 더 싸야 함)
    - 온라인최저허용선(Unit) = H*(1+u)  (카니발 방지 하한)
    - 온라인 레벨들은 공구 기반 사다리 + 온라인최저허용선으로 바닥 상향
    """
    H = float(unit_H)
    G = H * (1.0 - d)
    floor_online = H * (1.0 + u)

    mob = max(G * (1.0 + a), floor_online)
    evt = max(G * (1.0 + b), floor_online)
    always = max(G * (1.0 + c), floor_online)

    # 나머지 레벨도 순서 유지
    oneday = max(floor_online, evt * 0.85)      # 기본: 행사보다 낮게
    homesale = max(mob * 0.98, floor_online)    # 기본: 라방 근처
    brandweek = evt

    return {
        "공구가": G,
        "홈쇼핑가": H,
        "모바일라방가": mob,
        "브랜드위크가": brandweek,
        "원데이 특가": oneday,
        "홈사할인가": homesale,
        "상시할인가": always,
        "온라인최저허용선": floor_online,
    }

def derive_official_from_online(unit_prices, official_mode, official_margin_over_always):
    """
    온라인 '공식가(노출)' 생성:
    - 옵션 1) 상시할인가를 사실상 노출가로 쓰는 경우: 공식가 = 상시할인가
    - 옵션 2) 상시할인가 위에 '정가 프레이밍'을 만들기: 공식가 = 상시*(1+margin)
    """
    always = unit_prices["상시할인가"]
    if official_mode == "상시=공식":
        return always
    return always * (1.0 + official_margin_over_always)

def apply_guardrail(unit_target, unit_guard):
    return max(float(unit_target), float(unit_guard))

# ============================================================
# App UI
# ============================================================

st.title("Pricing Rules Simulator (3안)")
st.caption("온라인 용어(공식가/상시/홈사/라방/브위/원데이)는 온라인에만 생성 | 공구=공구가만 | 홈쇼핑=홈쇼핑가만")

# Session defaults
if "products_df" not in st.session_state:
    st.session_state["products_df"] = DEFAULT_PRODUCTS.copy()

# Sidebar: Engine selection + global params
with st.sidebar:
    st.header("엔진 선택(3안)")
    engine = st.radio(
        "가격 룰",
        [
            "1안) 공구 앵커 사다리",
            "2안) 홈쇼핑-공구 최저라인 + 온라인 업룰",
            "3안) 마진 Guardrail + (1/2) 보정",
        ],
        index=1
    )

    st.divider()
    st.header("공통 설정")
    rounding_unit = st.selectbox("반올림 단위", [10, 100, 1000], index=1)
    band_pct = st.slider("가격 밴드폭(%) (Min~Max)", min_value=0, max_value=20, value=6, step=1)
    st.caption("예: 6%면 Target 기준 ±3%로 Min/Max 생성")

    st.divider()
    st.header("온라인 공식가 생성 방식")
    official_mode = st.radio("공식가(노출) 규칙", ["상시=공식", "상시 위에 정가 프레이밍 추가"], index=1)
    official_margin_over_always = st.slider("정가 프레이밍(%): 공식가=상시*(1+%)", 0, 80, 20, 1) / 100.0

    st.divider()
    st.header("온라인 레벨 사다리(기본 비율)")
    st.caption("공구 기반 사다리용(1안/2안 공통). 네 데이터 패턴에 맞춘 기본값.")
    a = st.slider("모바일라방: 공구 대비 +%", 0, 30, 5, 1) / 100.0
    b = st.slider("행사/브위: 공구 대비 +%", 0, 80, 25, 1) / 100.0
    c = st.slider("상시: 공구 대비 +%", 0, 120, 60, 1) / 100.0

    if engine.startswith("1안"):
        st.divider()
        st.header("1안 파라미터")
        hs_premium_over_G = st.slider("홈쇼핑: 공구 대비 +% (공구가 더 싸게)", 0, 20, 5, 1) / 100.0

    if engine.startswith("2안") or engine.startswith("3안"):
        st.divider()
        st.header("2안 파라미터(홈쇼핑 기반)")
        d = st.slider("공구가: 홈쇼핑 대비 -% (공구가 더 싸게)", 0, 20, 4, 1) / 100.0
        u = st.slider("온라인 하한: 홈쇼핑 대비 +% (카니발 방지)", 0, 60, 15, 1) / 100.0

    if engine.startswith("3안"):
        st.divider()
        st.header("3안 Guardrail(마진/비용)")
        st.caption("채널별 최소 허용가(=원가 방어) 적용 후 룰 가격을 끌어올립니다.")
        min_margin_pct = st.slider("최소 마진(%)", 0, 50, 15, 1) / 100.0
        fee_online = st.slider("온라인 채널비용율(수수료+프로모션+반품)", 0, 60, 20, 1) / 100.0
        fee_groupbuy = st.slider("공구 채널비용율", 0, 60, 15, 1) / 100.0
        fee_hs = st.slider("홈쇼핑 채널비용율", 0, 80, 35, 1) / 100.0

# Main: product input
st.subheader("상품 입력(배치)")
st.caption("구성수량(Q)은 '세트가→단위가' 환산에 사용. 비어있으면 상품명에서 숫자 힌트를 자동 제안합니다(가능할 때만).")

products_df = st.data_editor(
    st.session_state["products_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "브랜드": st.column_config.TextColumn(required=True),
        "상품명": st.column_config.TextColumn(required=True),
        "구분": st.column_config.SelectboxColumn(options=["단품", "세트"], required=True),
        "구성수량(Q)": st.column_config.NumberColumn(min_value=1, step=1, help="세트 구성 수량(병/개/박스 등)."),
        "랜디드코스트(총원가)": st.column_config.NumberColumn(min_value=0, step=10),
        "홈쇼핑가(세트가)": st.column_config.NumberColumn(min_value=0, step=100),
        "공구가(세트가)": st.column_config.NumberColumn(min_value=0, step=100),
        "공식가(세트가)": st.column_config.NumberColumn(min_value=0, step=100, help="2-Anchor 방식(홈쇼핑+공식가)을 쓰고 싶을 때 활용. 엔진에 따라 사용/무시됩니다."),
    },
)
st.session_state["products_df"] = products_df

# Auto-suggest Q if blank or 0
df = products_df.copy()
if "구성수량(Q)" not in df.columns:
    df["구성수량(Q)"] = 1

suggested = []
for i, r in df.iterrows():
    q = safe_float(r.get("구성수량(Q)"), np.nan)
    if not np.isnan(q) and q >= 1:
        suggested.append(int(q))
        continue
    hint = parse_units_from_text(str(r.get("상품명", "")))
    suggested.append(int(hint) if hint else 1)
df["구성수량(Q)"] = suggested

# Compute
def compute_all(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df_in.iterrows():
        brand = str(r.get("브랜드", "")).strip()
        name = str(r.get("상품명", "")).strip()
        q = max(1, int(safe_float(r.get("구성수량(Q)"), 1)))
        landed_total = safe_float(r.get("랜디드코스트(총원가)"), 0.0)
        unit_cost = landed_total / q if q > 0 else 0.0

        hs_set = safe_float(r.get("홈쇼핑가(세트가)"), np.nan)
        gb_set = safe_float(r.get("공구가(세트가)"), np.nan)
        off_set = safe_float(r.get("공식가(세트가)"), np.nan)

        hs_unit = hs_set / q if not np.isnan(hs_set) and q > 0 else np.nan
        gb_unit = gb_set / q if not np.isnan(gb_set) and q > 0 else np.nan
        off_unit = off_set / q if not np.isnan(off_set) and q > 0 else np.nan

        # ---- Engine-specific: generate unit targets ----
        if engine.startswith("1안"):
            # Need 공구가
            if np.isnan(gb_unit):
                continue
            unit_prices = engine1_groupbuy_ladder(
                unit_G=gb_unit,
                a=a, b=b, c=c,
                hs_premium_over_G=hs_premium_over_G
            )
            official_unit = derive_official_from_online(unit_prices, official_mode, official_margin_over_always)

        elif engine.startswith("2안"):
            # Need 홈쇼핑가
            if np.isnan(hs_unit):
                continue
            unit_prices = engine2_hs_floor_with_uplift(
                unit_H=hs_unit,
                d=d, u=u, a=a, b=b, c=c
            )
            official_unit = derive_official_from_online(unit_prices, official_mode, official_margin_over_always)

        else:
            # 3안: use 2안(홈쇼핑 기반)을 기본으로, 없으면 1안 fallback
            if not np.isnan(hs_unit):
                unit_prices = engine2_hs_floor_with_uplift(unit_H=hs_unit, d=d, u=u, a=a, b=b, c=c)
            elif not np.isnan(gb_unit):
                unit_prices = engine1_groupbuy_ladder(unit_G=gb_unit, a=a, b=b, c=c, hs_premium_over_G=0.05)
            else:
                continue

            official_unit = derive_official_from_online(unit_prices, official_mode, official_margin_over_always)

            # Guardrail per channel (unit basis)
            guard_online = guardrail_min_price(unit_cost, fee_online, min_margin_pct)
            guard_gb = guardrail_min_price(unit_cost, fee_groupbuy, min_margin_pct)
            guard_hs = guardrail_min_price(unit_cost, fee_hs, min_margin_pct)

            # Apply guardrails (targets)
            # 공구가/홈쇼핑가/온라인레벨/공식가(노출)
            unit_prices["공구가"] = apply_guardrail(unit_prices.get("공구가", 0), guard_gb)
            unit_prices["홈쇼핑가"] = apply_guardrail(unit_prices.get("홈쇼핑가", 0), guard_hs)
            for k in ONLINE_LEVELS:
                if k in unit_prices:
                    unit_prices[k] = apply_guardrail(unit_prices[k], guard_online)
            official_unit = apply_guardrail(official_unit, guard_online)

        # ---- Enforce ordering constraints (practical safety) ----
        # 홈쇼핑은 최저(또는 공구가 더 싸게) 정책은 파라미터로 이미 설계됨.
        # 온라인 레벨은 공식가 이하로 캡.
        for k in ONLINE_LEVELS:
            if k in unit_prices:
                unit_prices[k] = min(unit_prices[k], official_unit)

        # ---- Build output rows (set price = unit * Q), then band ----
        # Groupbuy only
        if "공구가" in unit_prices:
            tgt = unit_prices["공구가"] * q
            pmin, ptgt, pmax = make_band(tgt, band_pct, rounding_unit)
            rows.append({"브랜드": brand, "상품명": name, "구분": r.get("구분", ""), "Q": q, "가격타입": "공구가",
                         "Min": pmin, "Target": ptgt, "Max": pmax})

        # Homeshopping only
        if "홈쇼핑가" in unit_prices:
            tgt = unit_prices["홈쇼핑가"] * q
            pmin, ptgt, pmax = make_band(tgt, band_pct, rounding_unit)
            rows.append({"브랜드": brand, "상품명": name, "구분": r.get("구분", ""), "Q": q, "가격타입": "홈쇼핑가",
                         "Min": pmin, "Target": ptgt, "Max": pmax})

        # Online only
        tgt = official_unit * q
        pmin, ptgt, pmax = make_band(tgt, band_pct, rounding_unit)
        rows.append({"브랜드": brand, "상품명": name, "구분": r.get("구분", ""), "Q": q, "가격타입": "공식가(노출)",
                     "Min": pmin, "Target": ptgt, "Max": pmax})

        for k in ONLINE_LEVELS:
            if k in unit_prices:
                tgt = unit_prices[k] * q
                pmin, ptgt, pmax = make_band(tgt, band_pct, rounding_unit)
                rows.append({"브랜드": brand, "상품명": name, "구분": r.get("구분", ""), "Q": q, "가격타입": k,
                             "Min": pmin, "Target": ptgt, "Max": pmax})

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # sort order
    order = {"공구가": 0, "홈쇼핑가": 1, "공식가(노출)": 2}
    for i, k in enumerate(ONLINE_LEVELS, start=3):
        order[k] = i
    out["__ord"] = out["가격타입"].map(order).fillna(999).astype(int)
    out = out.sort_values(["브랜드", "상품명", "__ord"]).drop(columns="__ord").reset_index(drop=True)
    return out

out = compute_all(df)

st.divider()
st.subheader("산출 결과")
if out.empty:
    st.warning("필수 입력값이 부족한 상품이 있습니다. (1안은 공구가 필요 / 2안은 홈쇼핑가 필요 / 3안은 둘 중 하나 + 원가 권장)")
else:
    st.dataframe(out, use_container_width=True)

    # Pivot view for quick scan (Target)
    st.subheader("요약(타겟가 피벗)")
    pv = out.pivot_table(index=["브랜드", "상품명"], columns="가격타입", values="Target", aggfunc="first")
    pv = pv.reindex(columns=OTHER_TYPES + ["공식가(노출)"] + ONLINE_LEVELS, fill_value=np.nan)
    st.dataframe(pv.reset_index(), use_container_width=True)

    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("도식화(선택 상품만)")
        options = (out["브랜드"] + " | " + out["상품명"]).drop_duplicates().tolist()
        sel = st.multiselect("도식화할 상품 선택", options=options, default=options[:3])
        if sel:
            mask = (out["브랜드"] + " | " + out["상품명"]).isin(sel)
            render_range_bars(out[mask], "가격 범위 도식화 (Min / Target / Max)")
        else:
            st.info("상품을 선택하면 도식화가 나옵니다.")
    with c2:
        st.subheader("룰 빠른 점검")
        st.write("현재 엔진:", engine)
        st.write("밴드폭:", f"{band_pct}%")
        st.write("반올림:", f"{rounding_unit:,}원")

        with st.expander("⚠️ 체크 포인트(자동 경고는 다음 단계에 추가 가능)"):
            st.markdown(
                """
- **1안**: 공구가 입력이 없으면 계산 불가  
- **2안**: 홈쇼핑가 입력이 없으면 계산 불가  
- **3안**: 홈쇼핑가가 있으면 그걸 우선 사용, 없으면 공구가로 대체  
- **공식가(노출)**는 온라인에만 생성됩니다.
- **공구/홈쇼핑에는 온라인 용어가 절대 생성되지 않습니다.**
                """
            )
