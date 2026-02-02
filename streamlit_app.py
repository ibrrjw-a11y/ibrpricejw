import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

# ============================================================
# IBR Pricing Simulator (Upload Master → Search Select → Simulate)
# - Online-only terms: 공식가(노출), 상시/홈사/라방/브위/원데이
# - Groupbuy: 공구가만
# - Homeshopping: 홈쇼핑가만
# - 3 Methods (intuitive names):
#   A) 공구 기준 온라인 사다리(비율)
#   B) 홈쇼핑 기준 + 공구 더 싸게 + 온라인 방어
#   C) 손익 방어 포함(원가/비용/마진 Guardrail)
# - NEW: Validation & Auto-correction rules
#   "홈쇼핑은 항상 최저, 공구는 그보다 더 최저" + "온라인은 홈쇼핑보다 비싸게"
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator", layout="wide")

ONLINE_LEVELS = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]
ONLINE_TYPES = ["공식가(노출)"] + ONLINE_LEVELS
CHANNEL_TYPES = ["공구가", "홈쇼핑가"] + ONLINE_TYPES

METHODS = {
    "A) 공구 기준 온라인 사다리(비율)": "A",
    "B) 홈쇼핑 기준 + 공구 더 싸게 + 온라인 방어": "B",
    "C) 손익 방어 포함(원가/비용/마진 가드레일)": "C",
}

# ----------------------------
# Styling (visibility fix)
# ----------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 14px !important; }
.block-container { padding-top: 1.1rem; }

thead tr th { position: sticky; top: 0; z-index: 1; }

.dp-wrap { border-top: 1px solid rgba(128,128,128,0.35); padding-top: 6px; }
.dp-row { display:flex; align-items:center; gap:12px; padding:7px 0; }
.dp-label {
  width: 560px;
  font-size: 13px;
  color: var(--text-color) !important;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  opacity: 0.95;
}
.dp-box {
  position: relative; flex: 1; height: 18px; border-radius: 999px;
  background: var(--secondary-background-color) !important;
  border: 1px solid rgba(128,128,128,0.45);
}
.dp-seg {
  position: absolute; height: 100%; border-radius: 999px;
  background: var(--primary-color);
  opacity: 0.22;
  border: 1px solid rgba(128,128,128,0.25);
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
  opacity: 0.85;
  text-align: right; white-space: nowrap;
}
.hint { color: var(--text-color); opacity: 0.78; }
.small { font-size: 12px; opacity: 0.78; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Utils
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

def pct_to_rate(pct: float) -> float:
    return float(pct) / 100.0

def krw_round(x, unit=100):
    try:
        return int(round(float(x) / unit) * unit)
    except Exception:
        return 0

def make_band(target_price: float, band_pct: float, rounding_unit: int):
    # band_pct=6 => Target 기준 ±3%
    t = float(target_price)
    half = pct_to_rate(band_pct) / 2.0
    pmin = t * (1.0 - half)
    pmax = t * (1.0 + half)
    return krw_round(pmin, rounding_unit), krw_round(t, rounding_unit), krw_round(pmax, rounding_unit)

def guardrail_min_price(landed_cost: float, channel_cost_rate: float, min_margin_rate: float) -> float:
    # Price >= landed_cost / (1 - cost_rate - min_margin)
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

def infer_brand_from_name(name: str) -> str:
    # 브랜드가 제품명에 녹아있다: 첫 토큰을 브랜드 후보로 단순 추정
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0]

def render_range_bars(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    d = df.copy()
    d["label"] = d["품번"].astype(str) + " | " + d["신규품명"].astype(str) + " | " + d["가격타입"].astype(str)

    gmin = float(d["Min"].min())
    gmax = float(d["Max"].max())
    span = max(1.0, gmax - gmin)

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

def to_excel_bytes(df_dict: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()

# ----------------------------
# Methods (unit basis)
# ----------------------------
def method_A_from_groupbuy(G_unit: float, a: float, b: float, c: float, hs_premium_over_G: float):
    """
    A) 공구 기준 온라인 사다리(비율)
    """
    G = float(G_unit)
    hs = G * (1.0 + hs_premium_over_G)
    mob = G * (1.0 + a)
    brandweek = G * (1.0 + b)
    always = G * (1.0 + c)

    homesale = max(mob * 0.98, G)
    oneday = max(brandweek * 0.85, G)

    return {
        "공구가": G,
        "홈쇼핑가": hs,
        "모바일라방가": mob,
        "브랜드위크가": brandweek,
        "상시할인가": always,
        "홈사할인가": homesale,
        "원데이 특가": oneday,
    }

def method_B_from_hs(H_unit: float, d: float, u: float, a: float, b: float, c: float):
    """
    B) 홈쇼핑 기준 + 공구 더 싸게 + 온라인 방어
    """
    H = float(H_unit)
    G = H * (1.0 - d)
    online_floor = H * (1.0 + u)

    mob = max(G * (1.0 + a), online_floor)
    brandweek = max(G * (1.0 + b), online_floor)
    always = max(G * (1.0 + c), online_floor)

    homesale = max(mob * 0.98, online_floor)
    oneday = max(brandweek * 0.85, online_floor)

    return {
        "홈쇼핑가": H,
        "공구가": G,
        "모바일라방가": mob,
        "브랜드위크가": brandweek,
        "상시할인가": always,
        "홈사할인가": homesale,
        "원데이 특가": oneday,
        "온라인최저허용선": online_floor,
    }

def derive_official_from_always(always_unit: float, mode: str, premium: float):
    if mode == "상시=공식":
        return float(always_unit)
    return float(always_unit) * (1.0 + premium)

# ----------------------------
# NEW: Validation & Auto-correction rules
# ----------------------------
def enforce_price_rules(
    unit_prices: dict,
    official_unit: float,
    official_mode: str,
    gb_under_hs_min: float,          # e.g. 0.03 => 공구는 HS보다 최소 3% 싸게
    online_over_hs_min: float,       # e.g. 0.15 => 온라인은 HS보다 최소 15% 비싸게
    enforce_monotonic_online: bool,  # online ladder ordering
    auto_correct: bool,              # auto correction vs warning only
):
    """
    Required policy:
    - 공구가 < 홈쇼핑가 (and preferably at least gb_under_hs_min cheaper)
    - 온라인(모든 레벨) > 홈쇼핑가 (at least online_over_hs_min uplift)
    - 온라인 레벨 순서: 상시 >= 홈사 >= 라방 >= 브위 >= 원데이
    - 온라인 레벨은 공식가 이하, 공식가는 온라인 중 최고 이상
    """
    warnings = []
    p = dict(unit_prices)  # copy

    gb = p.get("공구가", np.nan)
    hs = p.get("홈쇼핑가", np.nan)

    # If missing required anchors, just return
    if np.isnan(gb) or np.isnan(hs):
        return p, official_unit, warnings

    # ---- Rule 1: 공구가 is strictly lower than 홈쇼핑가
    gb_target_max = hs * (1.0 - gb_under_hs_min)
    if gb > gb_target_max:
        msg = f"[위반] 공구가({gb:,.0f})가 홈쇼핑가({hs:,.0f})보다 충분히 싸지 않음. 목표: 공구 ≤ 홈쇼핑×(1-{gb_under_hs_min*100:.0f}%) = {gb_target_max:,.0f}"
        if auto_correct:
            p["공구가"] = gb_target_max
            warnings.append(msg + " → 자동보정: 공구가를 낮춤")
        else:
            warnings.append(msg)

    # recompute after possible correction
    gb = p.get("공구가", gb)
    hs = p.get("홈쇼핑가", hs)

    # ---- Rule 2: 온라인은 홈쇼핑보다 비싸게(카니발 방지 하한)
    online_floor = hs * (1.0 + online_over_hs_min)
    for k in ONLINE_LEVELS:
        if k in p and not np.isnan(p[k]):
            if p[k] < online_floor:
                msg = f"[위반] {k}({p[k]:,.0f})가 홈쇼핑 방어선({online_floor:,.0f})보다 낮음 (홈쇼핑×(1+{online_over_hs_min*100:.0f}%))."
                if auto_correct:
                    p[k] = online_floor
                    warnings.append(msg + " → 자동보정: 온라인 레벨을 방어선으로 상향")
                else:
                    warnings.append(msg)

    # ---- Rule 3: 온라인 레벨 순서 보정(상시 ≥ 홈사 ≥ 라방 ≥ 브위 ≥ 원데이)
    if enforce_monotonic_online:
        order_high_to_low = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]
        # Work from low to high to enforce non-decreasing
        low_to_high = list(reversed(order_high_to_low))  # 원데이→브위→라방→홈사→상시
        prev = None
        for k in low_to_high:
            if k not in p or np.isnan(p[k]):
                continue
            if prev is None:
                prev = p[k]
                continue
            if p[k] < prev:
                msg = f"[위반] 온라인 레벨 순서 깨짐: {k}({p[k]:,.0f}) < 이전 하위레벨({prev:,.0f})."
                if auto_correct:
                    p[k] = prev
                    warnings.append(msg + " → 자동보정: 상위레벨을 하위레벨 이상으로 상향")
                else:
                    warnings.append(msg)
            prev = p[k]

    # ---- Rule 4: 온라인 레벨은 공식가 이하, 공식가는 온라인 최고 이상
    max_online = max([p.get(k, -np.inf) for k in ONLINE_LEVELS if k in p and not np.isnan(p[k])] + [-np.inf])
    if max_online > official_unit:
        msg = f"[위반] 온라인 레벨 최고값({max_online:,.0f})이 공식가({official_unit:,.0f})를 초과."
        if auto_correct:
            if official_mode == "상시=공식":
                # 공식가=상시이므로, 공식가를 올리면 상시도 같이 올라야 함
                # 가장 간단한 해결: 상시를 max_online으로 올리고, 공식가도 동일하게
                p["상시할인가"] = max(p.get("상시할인가", max_online), max_online)
                official_unit = p["상시할인가"]
                # 다른 상위레벨도 공식가 이하 캡 필요 없음(공식가=상시)
                warnings.append(msg + " → 자동보정: 상시(=공식)를 온라인 최고값 이상으로 상향")
            else:
                official_unit = max_online
                warnings.append(msg + " → 자동보정: 공식가를 온라인 최고값 이상으로 상향")
        else:
            warnings.append(msg)

    # ---- Rule 5: 온라인 레벨을 공식가 이하로 캡
    for k in ONLINE_LEVELS:
        if k in p and not np.isnan(p[k]):
            if p[k] > official_unit:
                msg = f"[위반] {k}({p[k]:,.0f})가 공식가({official_unit:,.0f})보다 높음."
                if auto_correct:
                    p[k] = official_unit
                    warnings.append(msg + " → 자동보정: 레벨을 공식가로 캡")
                else:
                    warnings.append(msg)

    return p, official_unit, warnings

# ----------------------------
# Load master file
# ----------------------------
def load_master(file) -> pd.DataFrame:
    df0 = pd.read_excel(file)

    cols = [str(c).strip() for c in df0.columns]
    df0.columns = cols

    code_col = None
    name_col = None
    for c in cols:
        if c in ["품번", "상품코드", "제품코드", "SKU", "코드"]:
            code_col = c
        if c in ["신규품명", "통일제품명", "통일 제품명", "제품명", "상품명"]:
            name_col = c

    if code_col is not None:
        df0 = df0.rename(columns={code_col: "품번"})
    if name_col is not None:
        df0 = df0.rename(columns={name_col: "신규품명"})

    if "품번" not in df0.columns:
        df0["품번"] = ""
    if "신규품명" not in df0.columns:
        df0["신규품명"] = ""

    df0["브랜드(추정)"] = df0["신규품명"].apply(infer_brand_from_name)
    df0 = df0[["품번", "신규품명", "브랜드(추정)"]].dropna(how="all")
    df0["품번"] = df0["품번"].astype(str).str.strip()
    df0["신규품명"] = df0["신규품명"].astype(str).str.strip()
    df0 = df0[df0["품번"].ne("") | df0["신규품명"].ne("")]
    df0 = df0.drop_duplicates(subset=["품번"]).reset_index(drop=True)
    return df0

# ----------------------------
# App
# ----------------------------
st.title("IBR 가격 시뮬레이터")
st.caption("상품 마스터 업로드 → 검색 선택 → 3가지 방법론 → 가격 밴드/도식화 + 룰 위반 검증/자동보정")

tab_sim, tab_formula, tab_data = st.tabs(["시뮬레이터", "계산식(로직)", "데이터 업로드/선택"])

if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame(columns=["품번", "신규품명", "브랜드(추정)"])
if "inputs_df" not in st.session_state:
    st.session_state["inputs_df"] = pd.DataFrame(columns=[
        "품번", "신규품명", "브랜드(추정)",
        "구성수량(Q)", "랜디드코스트(총원가)",
        "홈쇼핑가(세트가)", "공구가(세트가)"
    ])

# ----------------------------
# DATA TAB
# ----------------------------
with tab_data:
    st.subheader("1) 상품 마스터 업로드")
    up = st.file_uploader("상품정보 엑셀 업로드 (예: No/품번/신규품명)", type=["xlsx", "xls"])

    colA, colB = st.columns([2, 1])
    with colA:
        if up is not None:
            try:
                master = load_master(up)
                st.session_state["master_df"] = master
                st.success(f"업로드 완료: {len(master):,}개 상품 로드")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류: {e}")
        else:
            st.info("업로드하면 품번/신규품명을 읽어옵니다. (현재는 비어있음)")

    with colB:
        if not st.session_state["master_df"].empty:
            st.metric("로드된 상품 수", f"{len(st.session_state['master_df']):,}")

    st.divider()
    st.subheader("2) 검색해서 상품 선택")
    master_df = st.session_state["master_df"]

    if master_df.empty:
        st.warning("먼저 상품정보 파일을 업로드하세요.")
    else:
        q = st.text_input("검색 (품번 또는 신규품명)", value="")
        view = master_df.copy()
        if q.strip():
            mask = view["품번"].str.contains(q, case=False, na=False) | view["신규품명"].str.contains(q, case=False, na=False)
            view = view[mask].copy()

        st.caption(f"검색 결과: {len(view):,}개")
        st.dataframe(view, use_container_width=True, height=320)

        options = (view["품번"].fillna("") + " | " + view["신규품명"].fillna("")).tolist()
        picked = st.multiselect("선택(멀티 가능)", options=options, default=[])

        if st.button("선택 상품을 입력 테이블로 추가", type="primary", disabled=(len(picked) == 0)):
            sel_codes = [p.split(" | ", 1)[0].strip() for p in picked]
            sel_df = master_df[master_df["품번"].isin(sel_codes)].copy()

            base = st.session_state["inputs_df"].copy()
            if base.empty:
                base = pd.DataFrame(columns=[
                    "품번", "신규품명", "브랜드(추정)",
                    "구성수량(Q)", "랜디드코스트(총원가)",
                    "홈쇼핑가(세트가)", "공구가(세트가)"
                ])

            sel_df["구성수량(Q)"] = 1
            sel_df["랜디드코스트(총원가)"] = np.nan
            sel_df["홈쇼핑가(세트가)"] = np.nan
            sel_df["공구가(세트가)"] = np.nan

            merged = pd.concat([base, sel_df[base.columns]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["품번"], keep="first").reset_index(drop=True)
            st.session_state["inputs_df"] = merged
            st.success(f"입력 테이블에 {len(sel_df)}개 추가 완료 (총 {len(merged)}개)")

    st.divider()
    st.subheader("3) (옵션) 입력값 엑셀 업로드로 채우기")
    st.caption("품번 기준으로 랜디드코스트/홈쇼핑가/공구가 등을 엑셀로 업로드해 자동 매칭합니다.")
    up2 = st.file_uploader("입력값 엑셀 업로드(품번 포함 권장)", type=["xlsx", "xls"], key="input_uploader")

    if up2 is not None and not st.session_state["inputs_df"].empty:
        try:
            add = pd.read_excel(up2)
            add.columns = [str(c).strip() for c in add.columns]

            col_map = {
                "품번": None,
                "랜디드코스트(총원가)": None,
                "홈쇼핑가(세트가)": None,
                "공구가(세트가)": None,
                "구성수량(Q)": None,
            }
            for c in add.columns:
                if c in ["품번", "상품코드", "제품코드", "SKU"]:
                    col_map["품번"] = c
                if c in ["랜디드코스트(총원가)", "원가", "랜디드코스트", "LandedCost"]:
                    col_map["랜디드코스트(총원가)"] = c
                if c in ["홈쇼핑가(세트가)", "홈쇼핑 판매가", "홈쇼핑가", "방송가"]:
                    col_map["홈쇼핑가(세트가)"] = c
                if c in ["공구가(세트가)", "공구 엔드가", "공구가"]:
                    col_map["공구가(세트가)"] = c
                if c in ["구성수량(Q)", "구성수량", "수량", "Q"]:
                    col_map["구성수량(Q)"] = c

            if col_map["품번"] is None:
                st.error("업로드한 입력값 파일에 '품번'(또는 SKU/상품코드/제품코드)이 필요합니다.")
            else:
                add2 = add.rename(columns={
                    col_map["품번"]: "품번",
                    col_map["랜디드코스트(총원가)"]: "랜디드코스트(총원가)" if col_map["랜디드코스트(총원가)"] else col_map["랜디드코스트(총원가)"],
                    col_map["홈쇼핑가(세트가)"]: "홈쇼핑가(세트가)" if col_map["홈쇼핑가(세트가)"] else col_map["홈쇼핑가(세트가)"],
                    col_map["공구가(세트가)"]: "공구가(세트가)" if col_map["공구가(세트가)"] else col_map["공구가(세트가)"],
                    col_map["구성수량(Q)"]: "구성수량(Q)" if col_map["구성수량(Q)"] else col_map["구성수량(Q)"],
                })

                base = st.session_state["inputs_df"].copy()
                base = base.merge(add2, on="품번", how="left", suffixes=("", "_new"))

                for f in ["구성수량(Q)", "랜디드코스트(총원가)", "홈쇼핑가(세트가)", "공구가(세트가)"]:
                    if f"{f}_new" in base.columns:
                        base[f] = base[f].where(base[f].notna(), base[f"{f}_new"])
                        base = base.drop(columns=[f"{f}_new"])

                st.session_state["inputs_df"] = base
                st.success("입력값 병합 완료(품번 기준).")

        except Exception as e:
            st.error(f"입력값 업로드 처리 오류: {e}")

# ----------------------------
# SIM TAB
# ----------------------------
with tab_sim:
    st.subheader("가격 방법론 선택")
    left, right = st.columns([2, 1])

    with left:
        method_label = st.selectbox("가격 산정 방법(3안)", list(METHODS.keys()), index=1)
        method_key = METHODS[method_label]
        st.markdown(f"<div class='hint'>선택된 방법: <b>{method_label}</b></div>", unsafe_allow_html=True)

    with right:
        rounding_unit = st.selectbox("반올림 단위", [10, 100, 1000], index=1)
        band_pct = st.slider("가격 밴드폭(%)", 0, 20, 6, 1)
        st.caption("Target 기준 ±(밴드폭/2)%로 Min/Max")

    st.divider()
    st.subheader("입력 테이블(선택 상품만)")
    if st.session_state["inputs_df"].empty:
        st.warning("데이터 업로드/선택 탭에서 상품을 선택해 입력 테이블에 추가해주세요.")
    else:
        inputs_df = st.data_editor(
            st.session_state["inputs_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "품번": st.column_config.TextColumn(disabled=True),
                "신규품명": st.column_config.TextColumn(disabled=True),
                "브랜드(추정)": st.column_config.TextColumn(disabled=True),
                "구성수량(Q)": st.column_config.NumberColumn(min_value=1, step=1),
                "랜디드코스트(총원가)": st.column_config.NumberColumn(min_value=0, step=10),
                "홈쇼핑가(세트가)": st.column_config.NumberColumn(min_value=0, step=100),
                "공구가(세트가)": st.column_config.NumberColumn(min_value=0, step=100),
            },
            height=320,
        )
        st.session_state["inputs_df"] = inputs_df

        st.divider()
        st.subheader("파라미터")
        p1, p2, p3 = st.columns(3)

        with p1:
            st.caption("온라인 레벨 사다리(공구 기준 비율)")
            a = st.slider("모바일라방: 공구 대비 +%", 0, 30, 5, 1) / 100.0
            b = st.slider("브위(행사): 공구 대비 +%", 0, 80, 25, 1) / 100.0
            c = st.slider("상시: 공구 대비 +%", 0, 120, 60, 1) / 100.0

        with p2:
            st.caption("홈쇼핑/공구 관계 & 온라인 방어")
            hs_premium_over_G = st.slider("홈쇼핑: 공구 대비 +% (A에서 사용)", 0, 20, 5, 1) / 100.0
            d = st.slider("공구: 홈쇼핑 대비 -% (B/C에서 사용)", 0, 20, 4, 1) / 100.0
            u = st.slider("온라인 하한: 홈쇼핑 대비 +% (B/C)", 0, 60, 15, 1) / 100.0

        with p3:
            st.caption("공식가(노출) 생성 규칙(온라인 전용)")
            official_mode = st.radio("공식가 생성", ["상시=공식", "상시 위에 정가 프레이밍"], index=1)
            official_premium = st.slider("정가 프레이밍(%)", 0, 80, 20, 1) / 100.0

        use_guardrail = (method_key == "C")
        if use_guardrail:
            st.divider()
            st.subheader("손익 방어(Guardrail) 설정 (C안 전용)")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                min_margin = st.slider("최소 마진(%)", 0, 50, 15, 1) / 100.0
            with g2:
                fee_online = st.slider("온라인 비용율(%)", 0, 60, 20, 1) / 100.0
            with g3:
                fee_gb = st.slider("공구 비용율(%)", 0, 60, 15, 1) / 100.0
            with g4:
                fee_hs = st.slider("홈쇼핑 비용율(%)", 0, 80, 35, 1) / 100.0

        st.divider()
        st.subheader("룰 검증/자동보정(카니발 방지)")
        v1, v2, v3 = st.columns([1.2, 1.2, 1])
        with v1:
            auto_correct = st.toggle("위반 시 자동보정", value=True)
            st.caption("OFF면 경고만 표시하고 가격은 건드리지 않습니다.")
        with v2:
            enforce_monotonic_online = st.toggle("온라인 레벨 순서 강제(상시≥홈사≥라방≥브위≥원데이)", value=True)
        with v3:
            st.caption("최소 격차(정책값)")
            gb_under_hs_min = st.slider("공구는 HS보다 최소 -%", 0, 20, 3, 1) / 100.0
            online_over_hs_min = st.slider("온라인은 HS보다 최소 +%", 0, 80, 15, 1) / 100.0

        # ----------------------------
        # Compute
        # ----------------------------
        def compute_prices(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            rows = []
            warn_rows = []

            for _, r in df_in.iterrows():
                code = str(r.get("품번", "")).strip()
                name = str(r.get("신규품명", "")).strip()
                brand_guess = str(r.get("브랜드(추정)", "")).strip()

                q = int(max(1, safe_float(r.get("구성수량(Q)"), 1)))
                landed_total = safe_float(r.get("랜디드코스트(총원가)"), np.nan)
                hs_set = safe_float(r.get("홈쇼핑가(세트가)"), np.nan)
                gb_set = safe_float(r.get("공구가(세트가)"), np.nan)

                unit_cost = (landed_total / q) if (not np.isnan(landed_total) and q > 0) else np.nan
                hs_unit = (hs_set / q) if (not np.isnan(hs_set) and q > 0) else np.nan
                gb_unit = (gb_set / q) if (not np.isnan(gb_set) and q > 0) else np.nan

                unit_prices = None

                # 1) generate base unit prices by method
                if method_key == "A":
                    if np.isnan(gb_unit):
                        continue
                    unit_prices = method_A_from_groupbuy(gb_unit, a, b, c, hs_premium_over_G)

                elif method_key == "B":
                    if np.isnan(hs_unit):
                        continue
                    unit_prices = method_B_from_hs(hs_unit, d, u, a, b, c)

                else:
                    # C: prefer HS (B), else fallback to A
                    if not np.isnan(hs_unit):
                        unit_prices = method_B_from_hs(hs_unit, d, u, a, b, c)
                    elif not np.isnan(gb_unit):
                        unit_prices = method_A_from_groupbuy(gb_unit, a, b, c, 0.05)
                    else:
                        continue

                    # apply guardrails if cost exists
                    if not np.isnan(unit_cost):
                        guard_online = guardrail_min_price(unit_cost, fee_online, min_margin)
                        guard_gb2 = guardrail_min_price(unit_cost, fee_gb, min_margin)
                        guard_hs2 = guardrail_min_price(unit_cost, fee_hs, min_margin)

                        if "공구가" in unit_prices:
                            unit_prices["공구가"] = max(unit_prices["공구가"], guard_gb2)
                        if "홈쇼핑가" in unit_prices:
                            unit_prices["홈쇼핑가"] = max(unit_prices["홈쇼핑가"], guard_hs2)
                        for k in ONLINE_LEVELS:
                            if k in unit_prices:
                                unit_prices[k] = max(unit_prices[k], guard_online)

                # 2) derive official (online only)
                always_unit = unit_prices.get("상시할인가", np.nan)
                if np.isnan(always_unit):
                    continue
                official_unit = derive_official_from_always(always_unit, official_mode, official_premium)

                # 3) Enforce policy rules (validate / auto-correct)
                unit_prices2, official_unit2, warns = enforce_price_rules(
                    unit_prices=unit_prices,
                    official_unit=official_unit,
                    official_mode=official_mode,
                    gb_under_hs_min=gb_under_hs_min,
                    online_over_hs_min=online_over_hs_min,
                    enforce_monotonic_online=enforce_monotonic_online,
                    auto_correct=auto_correct
                )

                for w in warns:
                    warn_rows.append({
                        "품번": code,
                        "신규품명": name,
                        "메시지": w
                    })

                unit_prices = unit_prices2
                official_unit = official_unit2

                # 4) Build output rows (set-level prices) + bands
                def add_row(price_type: str, unit_target: float):
                    tgt = unit_target * q
                    pmin, ptgt, pmax = make_band(tgt, band_pct, rounding_unit)
                    rows.append({
                        "품번": code,
                        "신규품명": name,
                        "브랜드(추정)": brand_guess,
                        "Q": q,
                        "가격타입": price_type,
                        "Min": pmin,
                        "Target": ptgt,
                        "Max": pmax,
                    })

                # 공구/홈쇼핑: 각각 1줄만
                if "공구가" in unit_prices and not np.isnan(unit_prices["공구가"]):
                    add_row("공구가", unit_prices["공구가"])
                if "홈쇼핑가" in unit_prices and not np.isnan(unit_prices["홈쇼핑가"]):
                    add_row("홈쇼핑가", unit_prices["홈쇼핑가"])

                # 온라인: 공식가 + 5레벨
                add_row("공식가(노출)", official_unit)
                for k in ONLINE_LEVELS:
                    if k in unit_prices and not np.isnan(unit_prices[k]):
                        add_row(k, unit_prices[k])

            out = pd.DataFrame(rows)
            warn_df = pd.DataFrame(warn_rows)

            if out.empty:
                return out, warn_df

            order = {t: i for i, t in enumerate(CHANNEL_TYPES)}
            out["__ord"] = out["가격타입"].map(order).fillna(999).astype(int)
            out = out.sort_values(["품번", "__ord"]).drop(columns="__ord").reset_index(drop=True)

            return out, warn_df

        out, warn_df = compute_prices(st.session_state["inputs_df"])

        st.divider()
        st.subheader("결과")
        if out.empty:
            need = "공구가" if method_key == "A" else "홈쇼핑가"
            st.warning(f"계산 가능한 상품이 없습니다. ({method_label}은 기본적으로 '{need}(세트가)' 입력이 필요합니다)")
        else:
            if not warn_df.empty:
                st.warning(f"룰 위반/보정 메시지: {len(warn_df):,}건")
                st.dataframe(warn_df, use_container_width=True, height=180)
            else:
                st.success("룰 위반 없음 (현재 설정 기준)")

            st.dataframe(out, use_container_width=True, height=360)

            st.subheader("요약(타겟가 피벗)")
            pv = out.pivot_table(index=["품번", "신규품명"], columns="가격타입", values="Target", aggfunc="first")
            pv = pv.reindex(columns=CHANNEL_TYPES, fill_value=np.nan)
            st.dataframe(pv.reset_index(), use_container_width=True, height=320)

            st.divider()
            st.subheader("가격 범위 도식화 (Min / Target / Max)")
            options = (out["품번"] + " | " + out["신규품명"]).drop_duplicates().tolist()
            picked = st.multiselect("도식화할 상품 선택", options=options, default=options[: min(6, len(options))])
            if picked:
                mask = (out["품번"] + " | " + out["신규품명"]).isin(picked)
                render_range_bars(out[mask], "선택 상품 가격 밴드")

            st.divider()
            xbytes = to_excel_bytes({
                "result_long": out,
                "result_pivot": pv.reset_index(),
                "violations": warn_df if not warn_df.empty else pd.DataFrame(columns=["품번","신규품명","메시지"])
            })
            st.download_button(
                "결과 엑셀 다운로드",
                data=xbytes,
                file_name="pricing_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ----------------------------
# FORMULA TAB (more intuitive + rationale)
# ----------------------------
with tab_formula:
    st.subheader("계산식(로직) — 더 직관적으로 + 근거 포함")
    st.caption("수학 기호를 최소화하고, 운영자가 ‘왜 이렇게 되는지’ 바로 이해하게 만드는 설명 방식입니다.")

    st.markdown("## 먼저: 우리가 강제하는 ‘가격 질서(계약/카니발 방지 룰)’")
    st.markdown(
        """
### 가격 질서(Policy Order)
- **공구가**: 전체에서 **가장 낮은 가격** (최저가)
- **홈쇼핑가**: 공구보다는 높지만, **온라인보다 낮아야 하는 가격** (방송 최저가)
- **온라인 가격(상시/홈사/라방/브위/원데이)**: 홈쇼핑과 가격 충돌(카니발)을 막기 위해 **홈쇼핑보다 높아야 함**
- **공식가(노출)**: 온라인에서 **가장 높은 앵커(정가 프레이밍용)**

#### 근거(왜 이 질서를 강제하나?)
- 홈쇼핑은 방송 편성/수수료/반품/CS 비용 구조가 다르고, **“방송에서 제일 싸게 판다”는 소비자 기대**가 거의 계약 수준으로 작동합니다.
- 온라인이 홈쇼핑보다 싸지면, 방송 중/방송 직후에 **카니발(검색 비교/CS 폭주/정산 이슈)**이 발생합니다.
- 공구는 “한 번에 크게 터는” 구조라, 홈쇼핑보다 **조금 더 낮은 가격**이 현실적으로 필요합니다(대량/선결제/트래픽 성격).
"""
    )

    st.divider()
    st.markdown("## 공통: 세트/구성 가격을 ‘단위가’로 환산한 뒤 룰 적용")
    st.markdown(
        """
- 홈쇼핑은 4병+스틱, 온라인은 1병 등 **구성이 다르기 때문에**,  
  비교와 룰 적용은 반드시 **단위가(=1병/1개 환산가)**로 합니다.

### 단위가 환산
- 구성수량을 Q라고 하면:
  - **단위가 = 세트가 / Q**
  - 계산이 끝나면 다시 **세트가 = 단위가 × Q** 로 복원합니다.

#### 근거
- ‘세트 구성’이 바뀌면 세트가 자체는 변동이 큰데,  
  실제로 채널 간 가격 질서를 결정하는 건 **“1개 기준의 체감 단가”**입니다.
"""
    )

    st.divider()
    st.markdown("## A) 공구 기준 온라인 사다리(비율)")
    st.markdown(
        """
### 입력(핵심)
- 공구 단위가(G)
- 온라인 비율 3개:  
  - 라방 = 공구 대비 +a%  
  - 브위(행사) = 공구 대비 +b%  
  - 상시 = 공구 대비 +c%
- 홈쇼핑은 공구보다 비싸야 하므로: **홈쇼핑 = 공구 대비 +h%**

### 계산(운영 언어로)
1) **공구가(단위)** = G  
2) **홈쇼핑가(단위)** = 공구가 × (1 + h%)  
3) **라방가(단위)** = 공구가 × (1 + a%)  
4) **브위가(단위)** = 공구가 × (1 + b%)  
5) **상시가(단위)** = 공구가 × (1 + c%)  
6) 홈사/원데이는 “라방~브위 사이 / 브위보다 조금 아래” 같은 운영 관행을 기본값으로 둠(필요하면 다음 단계에서 파라미터로 분리 가능)

### 근거
- 온라인 데이터가 이미 “공구 < 라방 < 행사 < 상시”로 움직이고 있어 **비율 사다리**가 가장 빠르고 재현성이 좋습니다.
"""
    )

    st.divider()
    st.markdown("## B) 홈쇼핑 기준 + 공구 더 싸게 + 온라인 방어(카니발 방지)")
    st.markdown(
        """
### 입력(핵심)
- 홈쇼핑 단위가(H)
- 공구가 할인율(d%): 공구가가 홈쇼핑보다 더 싸게  
- 온라인 방어 업율(u%): 온라인이 홈쇼핑보다 항상 비싸게(카니발 방지)
- 온라인 사다리 비율(a,b,c)

### 계산(운영 언어로)
1) **홈쇼핑가(단위)** = H  
2) **공구가(단위)** = 홈쇼핑가 × (1 - d%)  
3) **온라인 최저 방어선(단위)** = 홈쇼핑가 × (1 + u%)  
4) 온라인 가격은 “공구 사다리로 계산한 값”과 “방어선” 중 **더 큰 값**을 채택
   - 라방 = max(공구×(1+a%), 방어선)
   - 브위 = max(공구×(1+b%), 방어선)
   - 상시 = max(공구×(1+c%), 방어선)

### 근거
- 홈쇼핑 기간/방송 중에는 온라인이 조금만 낮아도 바로 CS/검색비교로 터집니다.
- 그래서 온라인은 “사다리”가 아니라 **방어선**으로 하단을 띄워야 실무적으로 안전합니다.
"""
    )

    st.divider()
    st.markdown("## C) 손익 방어 포함(원가/비용/마진 가드레일)")
    st.markdown(
        """
### 입력(추가)
- 단위 원가(랜디드코스트/Q)
- 채널별 비용율(수수료/프로모션/반품 등)과 최소마진

### 계산(운영 언어로)
1) 먼저 A 또는 B로 “목표 가격”을 만든다  
2) 그 다음, 손익이 깨지는 가격은 **채널별 최소 허용가**로 끌어올린다

### 채널별 최소 허용가(단위)
- “이 가격 밑으로 팔면 비용/마진 구조상 적자”인 선
- **최소허용가 = 단위원가 / (1 - 채널비용율 - 최소마진율)**

### 근거
- 가격 질서(카니발 방지)만 맞추면 “그럴듯하지만 적자”가 나올 수 있습니다.
- C안은 가격 질서 위에 **손익 하단**을 깔아서 ‘구조적으로 불가능한 가격’을 자동 차단합니다.
"""
    )

    st.divider()
    st.markdown("## 추가: 결과 검증/자동보정 룰(이번에 추가된 기능)")
    st.markdown(
        """
### 우리가 강제하는 3가지 검증(선택: 경고만 / 자동보정)
1) **공구가 ≤ 홈쇼핑가 × (1 - 최소차이%)**  
2) **온라인 모든 레벨 ≥ 홈쇼핑가 × (1 + 최소업율%)**  
3) **온라인 레벨 순서 유지**: 상시 ≥ 홈사 ≥ 라방 ≥ 브위 ≥ 원데이  
+ 공식가는 온라인 최고값 이상 / 온라인은 공식가 이하

#### 근거
- 이 룰이 없으면 “입력/비율”이 살짝만 어긋나도  
  홈쇼핑-온라인 가격 충돌(카니발), 공구-홈쇼핑 역전이 쉽게 발생합니다.
- 자동보정은 “정책을 어기는 값을 정책 안으로 밀어 넣는” 방식이라  
  운영자가 의도한 가격 질서를 빠르게 확보할 수 있습니다.
"""
    )
