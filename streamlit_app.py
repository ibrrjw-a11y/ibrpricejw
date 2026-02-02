import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v2 (Research-driven)
# 1안) 온라인(시장단가) 먼저 → HS/공구 feasibility 체크
# 2안) 홈쇼핑 먼저 → 온라인을 시장밴드 내에서 최대 확보(온라인이점 방어)
# 3안) 패키지 리디자인(구성 Q + 사은품가치) → 1/2안 불가 시 구조로 해결
#
# - Online-only: 공식가(노출), 상시/홈사/라방/브위/원데이
# - Groupbuy: 공구가만
# - Homeshopping: 홈쇼핑가만
# - 리서치 입력(국내/해외/경쟁) → Market Reference Band 생성
# - 결과: Target + 추천 밴드(Min~Target~Max) + 룰검증/자동보정 + 진단
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator", layout="wide")

# ----------------------------
# Constants
# ----------------------------
ONLINE_LEVELS = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]
ONLINE_TYPES = ["공식가(노출)"] + ONLINE_LEVELS
CHANNEL_TYPES = ["공구가", "홈쇼핑가"] + ONLINE_TYPES

MODES = {
    "1안) 온라인(시장단가) 먼저 → HS/공구 가능성 체크": "M1",
    "2안) 홈쇼핑 먼저 → 온라인을 시장밴드 내에서 최대 확보": "M2",
    "3안) 패키지 리디자인(Q+사은품) → 구조로 해결": "M3",
}

# ----------------------------
# Styling (dark-mode visibility)
# ----------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 14px !important; }
.block-container { padding-top: 1.1rem; }
thead tr th { position: sticky; top: 0; z-index: 1; }

.hint { color: var(--text-color); opacity: 0.78; }
.small { font-size: 12px; opacity: 0.78; }

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

def krw_round(x, unit=100):
    try:
        return int(round(float(x) / unit) * unit)
    except Exception:
        return 0

def pct_to_rate(pct):
    return float(pct) / 100.0

def make_band(target_price, band_pct, rounding_unit):
    """
    band_pct=6이면 Target 기준 ±3%
    """
    t = float(target_price)
    half = pct_to_rate(band_pct) / 2.0
    pmin = t * (1.0 - half)
    pmax = t * (1.0 + half)
    return krw_round(pmin, rounding_unit), krw_round(t, rounding_unit), krw_round(pmax, rounding_unit)

def recommended_band_pct_from_ref(ref_min, ref_max, default=6):
    """
    시장 밴드 폭이 넓을수록(변동성/불확실성↑) 밴드폭 추천을 넓힘
    - 비율폭 = ref_max/ref_min
    """
    if ref_min <= 0 or ref_max <= 0:
        return default
    ratio = ref_max / ref_min
    # 경험칙(단순): 1.15 이하면 6%, 1.30 이하면 8%, 그 이상 10%
    if ratio <= 1.15:
        return 6
    if ratio <= 1.30:
        return 8
    return 10

def infer_brand_from_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0]

def to_excel_bytes(df_dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()

def render_range_bars(df, title):
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

# ----------------------------
# Research → Reference Band
# (입력은 "요약값"만 받는다: min/max/avg 일부만으로 밴드 생성)
# ----------------------------
def compute_reference_band(row):
    """
    입력 가능한 필드(있는 것만 사용):
      - 국내: domestic_min, domestic_max
      - 경쟁: comp_min, comp_max, comp_avg
      - 해외: overseas_rrp, overseas_sale_min, overseas_sale_max
      - 직구: direct_buy_price
    출력:
      ref_min, ref_mid, ref_max + explain
    """
    # mins candidates
    mins = []
    maxs = []
    mids = []

    dom_min = safe_float(row.get("국내관측_min", np.nan))
    dom_max = safe_float(row.get("국내관측_max", np.nan))
    if not np.isnan(dom_min): mins.append(dom_min)
    if not np.isnan(dom_max): maxs.append(dom_max)
    if not np.isnan(dom_min) and not np.isnan(dom_max):
        mids.append((dom_min + dom_max) / 2.0)

    comp_min = safe_float(row.get("경쟁사_min", np.nan))
    comp_max = safe_float(row.get("경쟁사_max", np.nan))
    comp_avg = safe_float(row.get("경쟁사_avg", np.nan))
    if not np.isnan(comp_min): mins.append(comp_min)
    if not np.isnan(comp_max): maxs.append(comp_max)
    if not np.isnan(comp_avg): mids.append(comp_avg)
    elif (not np.isnan(comp_min) and not np.isnan(comp_max)):
        mids.append((comp_min + comp_max) / 2.0)

    ov_rrp = safe_float(row.get("해외정가_RRP", np.nan))
    ov_min = safe_float(row.get("해외실판매_min", np.nan))
    ov_max = safe_float(row.get("해외실판매_max", np.nan))
    if not np.isnan(ov_min): mins.append(ov_min)
    if not np.isnan(ov_max): maxs.append(ov_max)
    if not np.isnan(ov_rrp): maxs.append(ov_rrp)
    if not np.isnan(ov_min) and not np.isnan(ov_max):
        mids.append((ov_min + ov_max) / 2.0)
    elif not np.isnan(ov_rrp):
        mids.append(ov_rrp)

    direct_buy = safe_float(row.get("직구가", np.nan))
    if not np.isnan(direct_buy):
        mins.append(direct_buy)
        mids.append(direct_buy)

    # fallback
    if len(mins) == 0 and len(maxs) == 0 and len(mids) == 0:
        return np.nan, np.nan, np.nan, "리서치 입력 없음"

    ref_min = np.nanmin(mins) if len(mins) else (np.nanmin(mids) if len(mids) else np.nan)
    ref_max = np.nanmax(maxs) if len(maxs) else (np.nanmax(mids) if len(mids) else np.nan)
    # mid: 중앙값(robust)
    if len(mids):
        ref_mid = float(np.nanmedian(mids))
    else:
        ref_mid = (ref_min + ref_max) / 2.0 if (not np.isnan(ref_min) and not np.isnan(ref_max)) else np.nan

    explain = []
    explain.append(f"RefMin={ref_min:,.0f} (입력된 min/직구 중 최저)")
    explain.append(f"RefMid={ref_mid:,.0f} (입력된 중심값들의 중앙값)")
    explain.append(f"RefMax={ref_max:,.0f} (입력된 max/해외정가 중 최고)")
    return ref_min, ref_mid, ref_max, " / ".join(explain)

# ----------------------------
# Economics Guardrail (optional; keep minimal)
# ----------------------------
def guardrail_min_unit_price(unit_cost, channel_cost_rate, min_margin_rate):
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return unit_cost / denom

# ----------------------------
# Online ladder (from a base "always" unit price)
# ----------------------------
def build_online_from_always(always_unit, discounts):
    """
    discounts: dict for each level: e.g. {"브랜드위크가":0.25, "모바일라방가":0.15 ...}
    interpret as 'discount from always'
    """
    A = float(always_unit)
    out = {}
    out["상시할인가"] = A
    out["브랜드위크가"] = A * (1.0 - discounts["브랜드위크가"])
    out["모바일라방가"] = A * (1.0 - discounts["모바일라방가"])
    out["홈사할인가"] = A * (1.0 - discounts["홈사할인가"])
    out["원데이 특가"] = A * (1.0 - discounts["원데이 특가"])
    return out

def derive_official_from_always(always_unit, list_discount_rate):
    """
    정가 프레이밍을 가장 직관적으로:
    - 상시는 공식가 대비 '상시할인율' 만큼 깎인 가격이다
    => 공식가 = 상시 / (1 - 상시할인율)
    """
    A = float(always_unit)
    d = float(list_discount_rate)
    if d >= 0.95:
        d = 0.95
    if d < 0:
        d = 0.0
    return A / (1.0 - d)

# ----------------------------
# Policy validation (unit-based, more realistic)
# - HS is lowest (unit) among channels
# - GB lower than HS by at least gb_under_hs_min
# - Online levels are higher than HS by at least online_over_hs_min
# - Online ladder monotonic: Always >= Homesale >= Live >= BrandWeek >= OneDay
# - Online levels <= Official, Official >= max(online levels)
# ----------------------------
def validate_and_autocorrect(
    unit_prices,
    official_unit,
    gb_under_hs_min,
    online_over_hs_min,
    enforce_monotonic_online=True,
    auto_correct=True,
):
    p = dict(unit_prices)
    warnings = []

    hs = p.get("홈쇼핑가", np.nan)
    gb = p.get("공구가", np.nan)

    # Need HS anchor for policy checks
    if np.isnan(hs):
        return p, official_unit, warnings

    # ---- GB < HS
    if not np.isnan(gb):
        gb_max = hs * (1.0 - gb_under_hs_min)
        if gb > gb_max:
            msg = f"[위반] 공구 단위가({gb:,.0f})가 홈쇼핑 단위가({hs:,.0f})보다 충분히 낮지 않음. 목표 공구≤HS×(1-{gb_under_hs_min*100:.0f}%)={gb_max:,.0f}"
            if auto_correct:
                p["공구가"] = gb_max
                warnings.append(msg + " → 자동보정: 공구를 하향")
            else:
                warnings.append(msg)

    # ---- Online must be above HS by floor
    online_floor = hs * (1.0 + online_over_hs_min)
    for k in ONLINE_LEVELS:
        if k in p and not np.isnan(p[k]):
            if p[k] < online_floor:
                msg = f"[위반] {k} 단위가({p[k]:,.0f})가 HS 방어선({online_floor:,.0f})보다 낮음 (HS×(1+{online_over_hs_min*100:.0f}%))."
                if auto_correct:
                    p[k] = online_floor
                    warnings.append(msg + " → 자동보정: 온라인 레벨 상향")
                else:
                    warnings.append(msg)

    # ---- Online monotonic (high→low)
    if enforce_monotonic_online:
        order_high_to_low = ["상시할인가", "홈사할인가", "모바일라방가", "브랜드위크가", "원데이 특가"]
        low_to_high = list(reversed(order_high_to_low))
        prev = None
        for k in low_to_high:
            if k not in p or np.isnan(p[k]):
                continue
            if prev is None:
                prev = p[k]
                continue
            if p[k] < prev:
                msg = f"[위반] 온라인 레벨 순서: {k}({p[k]:,.0f}) < 하위레벨({prev:,.0f})"
                if auto_correct:
                    p[k] = prev
                    warnings.append(msg + " → 자동보정: 상위레벨을 하위레벨 이상으로 상향")
                else:
                    warnings.append(msg)
            prev = p[k]

    # ---- Official relations
    max_online = max([p.get(k, -np.inf) for k in ONLINE_LEVELS if k in p and not np.isnan(p[k])] + [-np.inf])
    if max_online > official_unit:
        msg = f"[위반] 온라인 최고 단위가({max_online:,.0f})가 공식가 단위가({official_unit:,.0f})를 초과"
        if auto_correct:
            official_unit = max_online
            warnings.append(msg + " → 자동보정: 공식가를 상향(온라인 최고 이상)")
        else:
            warnings.append(msg)

    for k in ONLINE_LEVELS:
        if k in p and not np.isnan(p[k]) and p[k] > official_unit:
            msg = f"[위반] {k}({p[k]:,.0f})가 공식가({official_unit:,.0f})보다 높음"
            if auto_correct:
                p[k] = official_unit
                warnings.append(msg + " → 자동보정: 레벨을 공식가로 캡")
            else:
                warnings.append(msg)

    return p, official_unit, warnings

# ----------------------------
# Master loader
# ----------------------------
def load_master(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]

    code_col = None
    name_col = None
    for c in df.columns:
        if c in ["품번", "상품코드", "제품코드", "SKU", "코드"]:
            code_col = c
        if c in ["신규품명", "통일제품명", "통일 제품명", "제품명", "상품명"]:
            name_col = c

    if code_col is not None:
        df = df.rename(columns={code_col: "품번"})
    if name_col is not None:
        df = df.rename(columns={name_col: "신규품명"})

    if "품번" not in df.columns:
        df["품번"] = ""
    if "신규품명" not in df.columns:
        df["신규품명"] = ""

    df["브랜드(추정)"] = df["신규품명"].apply(infer_brand_from_name)
    df = df[["품번", "신규품명", "브랜드(추정)"]].dropna(how="all")
    df["품번"] = df["품번"].astype(str).str.strip()
    df["신규품명"] = df["신규품명"].astype(str).str.strip()
    df = df[df["품번"].ne("") | df["신규품명"].ne("")]
    df = df.drop_duplicates(subset=["품번"]).reset_index(drop=True)
    return df

# ============================================================
# UI
# ============================================================
st.title("IBR 가격 시뮬레이터 (리서치 기반)")
st.caption("리서치 입력 → 시장단가 밴드 산출 → 1/2/3안으로 가격 추천(밴드 포함) + 룰검증/자동보정")

tab_sim, tab_formula, tab_data = st.tabs(["시뮬레이터", "계산식(로직)", "데이터 업로드/선택"])

# session
if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame(columns=["품번", "신규품명", "브랜드(추정)"])
if "inputs_df" not in st.session_state:
    st.session_state["inputs_df"] = pd.DataFrame(columns=[
        "품번", "신규품명", "브랜드(추정)",
        "온라인기준수량(Q_online)", "홈쇼핑구성(Q_hs)", "공구구성(Q_gb)",
        "랜디드코스트(총원가)",
        # research inputs (summary)
        "국내관측_min", "국내관측_max",
        "경쟁사_min", "경쟁사_max", "경쟁사_avg",
        "해외정가_RRP", "해외실판매_min", "해외실판매_max",
        "직구가",
        # optional anchors (if known)
        "홈쇼핑가(세트가)_입력", "공구가(세트가)_입력",
        # 3안
        "사은품가치(원)_hs", "사은품가치(원)_gb",
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
            st.info("업로드하면 품번/신규품명을 읽어옵니다.")

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
                base = st.session_state["inputs_df"]

            # defaults
            sel_df["온라인기준수량(Q_online)"] = 1
            sel_df["홈쇼핑구성(Q_hs)"] = 1
            sel_df["공구구성(Q_gb)"] = 1
            sel_df["랜디드코스트(총원가)"] = np.nan

            for c in ["국내관측_min", "국내관측_max", "경쟁사_min", "경쟁사_max", "경쟁사_avg",
                      "해외정가_RRP", "해외실판매_min", "해외실판매_max", "직구가",
                      "홈쇼핑가(세트가)_입력", "공구가(세트가)_입력",
                      "사은품가치(원)_hs", "사은품가치(원)_gb"]:
                sel_df[c] = np.nan

            merged = pd.concat([base, sel_df[base.columns]], ignore_index=True)
            merged = merged.drop_duplicates(subset=["품번"], keep="first").reset_index(drop=True)
            st.session_state["inputs_df"] = merged
            st.success(f"입력 테이블에 {len(sel_df)}개 추가 완료 (총 {len(merged)}개)")

    st.divider()
    st.subheader("3) (옵션) 입력값 엑셀 업로드로 채우기")
    st.caption("품번 기준으로 랜디드/리서치/앵커 입력을 업로드해 자동 병합합니다.")
    up2 = st.file_uploader("입력값 엑셀 업로드(품번 포함 권장)", type=["xlsx", "xls"], key="input_uploader")

    if up2 is not None and not st.session_state["inputs_df"].empty:
        try:
            add = pd.read_excel(up2)
            add.columns = [str(c).strip() for c in add.columns]
            # 품번 컬럼 자동 탐지
            code_col = None
            for c in add.columns:
                if c in ["품번", "상품코드", "제품코드", "SKU"]:
                    code_col = c
                    break
            if code_col is None:
                st.error("업로드 파일에 '품번'(또는 SKU/상품코드/제품코드) 컬럼이 필요합니다.")
            else:
                add = add.rename(columns={code_col: "품번"})
                base = st.session_state["inputs_df"].copy()
                base = base.merge(add, on="품번", how="left", suffixes=("", "_new"))

                # 업데이트 가능한 필드들
                updatable = [c for c in base.columns if c.endswith("_new")]
                for c_new in updatable:
                    c = c_new.replace("_new", "")
                    if c in base.columns:
                        base[c] = base[c].where(base[c].notna(), base[c_new])
                    base = base.drop(columns=[c_new])

                st.session_state["inputs_df"] = base
                st.success("입력값 병합 완료(품번 기준).")
        except Exception as e:
            st.error(f"입력값 업로드 처리 오류: {e}")

# ----------------------------
# SIM TAB
# ----------------------------
with tab_sim:
    st.subheader("모드 선택(1안/2안/3안)")
    mode_label = st.selectbox("가격 추천 모드", list(MODES.keys()), index=0)
    mode = MODES[mode_label]
    st.markdown(f"<div class='hint'>선택 모드: <b>{mode_label}</b></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("입력 테이블(리서치 포함)")
    if st.session_state["inputs_df"].empty:
        st.warning("데이터 업로드/선택 탭에서 상품을 선택해 입력 테이블에 추가해주세요.")
        st.stop()

    # Show editable table
    # ✅ 입력(편집)과 계산을 분리: form + 버튼
with st.form("input_form", clear_on_submit=False):
    edited_df = st.data_editor(
        st.session_state["inputs_df"],
        key="inputs_editor",  # ✅ 고정 key: 입력값 튐/지워짐 방지
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            # ✅ 여기에는 네가 원래 쓰던 column_config 내용을 그대로 붙여 넣으면 됨
            # (즉, 아까 삭제한 블록의 column_config 딕셔너리를 그대로 복붙)
        },
        height=360,
    )

    c_save, c_calc = st.columns([1, 1])
    save_clicked = c_save.form_submit_button("입력값 적용(저장)", type="secondary")
    calc_clicked = c_calc.form_submit_button("계산 실행", type="primary")

# ✅ 저장/계산 버튼 눌렀을 때만 session_state 갱신
if save_clicked or calc_clicked:
    st.session_state["inputs_df"] = edited_df.copy()

# ✅ '계산 실행' 누르기 전에는 아래 계산/도식화가 돌아가지 않게 막음(속도 개선 핵심)
if not calc_clicked:
    st.info("입력값을 수정한 뒤 '계산 실행'을 눌러 결과를 업데이트하세요.")
    st.stop()
out, warn_df, diag_df = compute_for_all(inputs_df)
# ✅ 계산은 여기부터 (오직 calc_clicked일 때만 실행)
inputs_df = st.session_state["inputs_df"]


st.divider()
    st.subheader("공통 파라미터(추천 밴드 포함)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        rounding_unit = st.selectbox("반올림 단위", [10, 100, 1000], index=1)
    with c2:
        auto_band = st.toggle("밴드폭 자동추천", value=True)
    with c3:
        band_pct_manual = st.slider("밴드폭 수동(%)", 0, 20, 6, 1)
        st.caption("자동추천 ON이면 상품별로 밴드폭을 추천")
    with c4:
        positioning = st.slider("시장 밴드 내 포지셔닝(%)", 0, 100, 50, 5)
        st.caption("0=밴드 하단, 50=중앙, 100=상단")

    # online discount ladder (from always)
    st.divider()
    st.subheader("온라인 가격 구조(상시 기준 할인율)")
    o1, o2, o3, o4, o5, o6 = st.columns([1, 1, 1, 1, 1, 1])
    with o1:
        d_brandweek = st.slider("브랜드위크 할인율(%)", 0, 70, 25, 1) / 100.0
    with o2:
        d_live = st.slider("라방 할인율(%)", 0, 70, 15, 1) / 100.0
    with o3:
        d_homesale = st.slider("홈사 할인율(%)", 0, 70, 10, 1) / 100.0
    with o4:
        d_oneday = st.slider("원데이 할인율(%)", 0, 80, 35, 1) / 100.0
    with o5:
        list_disc = st.slider("상시할인율(정가 프레이밍) (%)", 0, 80, 20, 1) / 100.0
    with o6:
        st.caption("정가=상시/(1-상시할인율)")
        pass

    discounts = {
        "브랜드위크가": d_brandweek,
        "모바일라방가": d_live,
        "홈사할인가": d_homesale,
        "원데이 특가": d_oneday,
    }

    st.divider()
    st.subheader("정책 룰(카니발 방지/채널 질서)")
    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        auto_correct = st.toggle("위반 시 자동보정", value=True)
    with r2:
        enforce_monotonic = st.toggle("온라인 레벨 순서 강제(상시≥홈사≥라방≥브위≥원데이)", value=True)
    with r3:
        gb_under_hs_min = st.slider("공구는 HS보다 최소 -%", 0, 20, 3, 1) / 100.0
        online_over_hs_min = st.slider("온라인은 HS보다 최소 +%", 0, 80, 15, 1) / 100.0

    # minimal profitability (optional, keep close)
    st.divider()
    st.subheader("손익 가드레일(선택)")
    use_guard = st.toggle("원가/비용/최소마진으로 불가능 가격 차단", value=False)
    if use_guard:
        g1, g2, g3, g4 = st.columns([1, 1, 1, 1])
        with g1:
            min_margin = st.slider("최소마진(%)", 0, 50, 15, 1) / 100.0
        with g2:
            fee_online = st.slider("온라인 비용율(%)", 0, 60, 20, 1) / 100.0
        with g3:
            fee_hs = st.slider("홈쇼핑 비용율(%)", 0, 80, 35, 1) / 100.0
        with g4:
            fee_gb = st.slider("공구 비용율(%)", 0, 60, 15, 1) / 100.0

    # ------------------------------------------------------------
    # Mode-specific params
    # ------------------------------------------------------------
    if mode == "M1":
        st.divider()
        st.subheader("1안 추가 파라미터 (온라인→HS/공구 설계)")
        m1a, m1b = st.columns([1, 1])
        with m1a:
            hs_under_online = st.slider("HS 단위가는 온라인 최저(원데이/브위 등) 대비 -%(목표)", 0, 40, 10, 1) / 100.0
            st.caption("홈쇼핑이 ‘최저가’가 되도록 온라인 최저 대비 추가 할인")
        with m1b:
            st.caption("공구는 HS보다 더 최저 (정책 룰에서 자동 적용)")
            pass

    if mode == "M2":
        st.divider()
        st.subheader("2안 추가 파라미터 (HS→온라인 방어)")
        m2a, m2b = st.columns([1, 1])
        with m2a:
            hs_anchor_source = st.radio("HS 앵커", ["입력값(홈쇼핑가)", "시장밴드에서 추천"], index=0)
        with m2b:
            hs_position_in_band = st.slider("HS 포지셔닝(밴드 내) (%)", 0, 100, 15, 5)
            st.caption("방송은 보통 하단에 두는 편(예: 10~20%)")

    if mode == "M3":
        st.divider()
        st.subheader("3안 추가 파라미터 (패키지 리디자인: Q + 사은품)")
        m3a, m3b, m3c = st.columns([1, 1, 1])
        with m3a:
            hs_q_candidates = st.multiselect(
                "HS 구성 후보(Q_hs) (추천)",
                options=[1, 2, 3, 4, 5, 6, 8, 10],
                default=[2, 4, 6, 8],
            )
        with m3b:
            gb_q_candidates = st.multiselect(
                "공구 구성 후보(Q_gb) (추천)",
                options=[1, 2, 3, 4, 5, 6, 8, 10],
                default=[2, 4, 6],
            )
        with m3c:
            hs_under_online = st.slider("HS 단위가 목표: 온라인 최저 대비 -%", 0, 40, 8, 1) / 100.0
            st.caption("3안은 가격을 깎기보다 Q/사은품으로 체감단가를 설계")

    # ------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------
    def choose_in_band(ref_min, ref_mid, ref_max, position_pct):
        if np.isnan(ref_min) or np.isnan(ref_max):
            return ref_mid
        p = float(position_pct) / 100.0
        return ref_min + (ref_max - ref_min) * p

    def compute_for_all(df_in):
        rows = []
        warn_rows = []
        diag_rows = []

        for _, r in df_in.iterrows():
            code = str(r.get("품번", "")).strip()
            name = str(r.get("신규품명", "")).strip()
            brand = str(r.get("브랜드(추정)", "")).strip()

            q_online = int(max(1, safe_float(r.get("온라인기준수량(Q_online)", 1))))
            q_hs_in = int(max(1, safe_float(r.get("홈쇼핑구성(Q_hs)", 1))))
            q_gb_in = int(max(1, safe_float(r.get("공구구성(Q_gb)", 1))))

            landed_total = safe_float(r.get("랜디드코스트(총원가)", np.nan))
            unit_cost = (landed_total / q_online) if (not np.isnan(landed_total) and q_online > 0) else np.nan

            # reference band (unit basis of online Q)
            ref_min, ref_mid, ref_max, ref_explain = compute_reference_band(r)

            if np.isnan(ref_mid) and (np.isnan(ref_min) or np.isnan(ref_max)):
                diag_rows.append({
                    "품번": code, "신규품명": name,
                    "진단": "리서치 입력 부족으로 시장밴드 산출 불가",
                    "해결": "국내관측/경쟁/해외/직구 중 최소 1개 이상 입력",
                })
                continue

            # band recommendation per product
            band_pct_reco = recommended_band_pct_from_ref(ref_min, ref_max, default=band_pct_manual)
            band_pct_use = band_pct_reco if auto_band else band_pct_manual

            # --- Online: choose "always" within band
            always_unit = choose_in_band(ref_min, ref_mid, ref_max, positioning)
            online_units = build_online_from_always(always_unit, discounts)
            official_unit = derive_official_from_always(online_units["상시할인가"], list_disc)

            # clamp online into band (soft): if outside, warn & clip
            def clip_to_band(x, label):
                if np.isnan(ref_min) or np.isnan(ref_max):
                    return x
                if x < ref_min:
                    warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[클립] {label} 단위가({x:,.0f})가 시장 하한({ref_min:,.0f}) 미만 → 하한으로 상향"})
                    return ref_min
                if x > ref_max:
                    warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[클립] {label} 단위가({x:,.0f})가 시장 상한({ref_max:,.0f}) 초과 → 상한으로 하향"})
                    return ref_max
                return x

            # Keep online within band (recommended behavior)
            for k in ONLINE_LEVELS:
                online_units[k] = clip_to_band(online_units[k], k)
            official_unit = max(official_unit, max(online_units.values()))

            # ----------------------------------------------------
            # Anchor depending on mode
            # ----------------------------------------------------
            hs_unit = np.nan
            gb_unit = np.nan
            hs_q = q_hs_in
            gb_q = q_gb_in
            free_hs = safe_float(r.get("사은품가치(원)_hs", 0), 0.0)
            free_gb = safe_float(r.get("사은품가치(원)_gb", 0), 0.0)

            # online lowest in unit terms
            online_lowest = min([online_units[k] for k in ONLINE_LEVELS if k in online_units and not np.isnan(online_units[k])])

            if mode == "M1":
                # 1안: 온라인 먼저, HS/GB는 온라인최저 대비 더 싸게 만들고 feasibility 체크
                hs_unit = online_lowest * (1.0 - hs_under_online)
                gb_unit = hs_unit * (1.0 - gb_under_hs_min)

            elif mode == "M2":
                # 2안: HS anchor 먼저
                hs_set_input = safe_float(r.get("홈쇼핑가(세트가)_입력", np.nan))
                if hs_anchor_source == "입력값(홈쇼핑가)":
                    if np.isnan(hs_set_input):
                        diag_rows.append({
                            "품번": code, "신규품명": name,
                            "진단": "2안(HS 입력) 선택했지만 홈쇼핑가 입력이 없음",
                            "해결": "홈쇼핑가(세트가)_입력 또는 '시장밴드에서 추천'으로 전환",
                        })
                        continue
                    hs_unit = hs_set_input / max(1, hs_q)
                else:
                    # HS as low position in band
                    hs_unit = choose_in_band(ref_min, ref_mid, ref_max, hs_position_in_band)

                # GB from HS
                gb_unit = hs_unit * (1.0 - gb_under_hs_min)

                # Online must be above HS by policy; push online up if needed
                floor = hs_unit * (1.0 + online_over_hs_min)
                # rebase always at least floor but within band if possible
                always_unit2 = max(online_units["상시할인가"], floor)
                always_unit2 = clip_to_band(always_unit2, "상시할인가(방어)")
                online_units = build_online_from_always(always_unit2, discounts)
                for k in ONLINE_LEVELS:
                    online_units[k] = clip_to_band(online_units[k], k)
                official_unit = derive_official_from_always(online_units["상시할인가"], list_disc)
                official_unit = max(official_unit, max(online_units.values()))

            else:
                # 3안: 패키지 리디자인(Q + 사은품)
                # 목표 HS unit = 온라인최저*(1-hs_under_online), 단 손익가드레일이 있으면 그 이상으로
                hs_unit_target = online_lowest * (1.0 - hs_under_online)

                # if use guardrail, enforce min unit price
                hs_unit_min = -np.inf
                gb_unit_min = -np.inf
                if use_guard and not np.isnan(unit_cost):
                    hs_unit_min = guardrail_min_unit_price(unit_cost, fee_hs, min_margin)
                    gb_unit_min = guardrail_min_unit_price(unit_cost, fee_gb, min_margin)

                hs_unit = max(hs_unit_target, hs_unit_min)
                gb_unit = max(hs_unit * (1.0 - gb_under_hs_min), gb_unit_min)

                # choose Q that makes offer "sellable" but still policy-ordered
                # Score = (단위가 낮을수록) + (사은품 가치로 체감단가 낮을수록) + (시장밴드 중앙 근접)
                def score_set(unit_price, q, free_value, ref_mid_local):
                    set_price = unit_price * q
                    # 체감단가(사은품을 '가치 단위'로 환산)
                    # free_equiv_units = free_value / 상시단위가 (보수적으로 상시 기준)
                    denom = max(1e-6, online_units["상시할인가"])
                    free_units = free_value / denom
                    effective_unit = set_price / (q + free_units)  # 체감 단위가
                    # prefer: effective_unit low, but unit_price not too far from ref_mid (too cheap = 브랜드손상/마진 리스크)
                    dist = abs(unit_price - ref_mid_local) / max(1.0, ref_mid_local)
                    return -effective_unit + (-dist * 2000)

                # pick HS Q
                best_hs = None
                for q in hs_q_candidates:
                    sc = score_set(hs_unit, q, free_hs, ref_mid if not np.isnan(ref_mid) else hs_unit)
                    if (best_hs is None) or (sc > best_hs[0]):
                        best_hs = (sc, q)
                if best_hs is not None:
                    hs_q = best_hs[1]

                # pick GB Q
                best_gb = None
                for q in gb_q_candidates:
                    sc = score_set(gb_unit, q, free_gb, ref_mid if not np.isnan(ref_mid) else gb_unit)
                    if (best_gb is None) or (sc > best_gb[0]):
                        best_gb = (sc, q)
                if best_gb is not None:
                    gb_q = best_gb[1]

            # ----------------------------------------------------
            # Optional guardrails: ensure online & HS/GB not below min feasible
            # ----------------------------------------------------
            if use_guard and not np.isnan(unit_cost):
                online_min_unit = guardrail_min_unit_price(unit_cost, fee_online, min_margin)
                # online levels
                for k in ONLINE_LEVELS:
                    if online_units[k] < online_min_unit:
                        warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[손익] {k} 단위가({online_units[k]:,.0f}) < 온라인 최소허용({online_min_unit:,.0f}) → 상향"})
                        online_units[k] = online_min_unit
                official_unit = max(official_unit, max(online_units.values()))

                # HS/GB
                hs_min_unit = guardrail_min_unit_price(unit_cost, fee_hs, min_margin)
                gb_min_unit = guardrail_min_unit_price(unit_cost, fee_gb, min_margin)
                if not np.isnan(hs_unit) and hs_unit < hs_min_unit:
                    warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[손익] HS 단위가({hs_unit:,.0f}) < 홈쇼핑 최소허용({hs_min_unit:,.0f}) → 상향"})
                    hs_unit = hs_min_unit
                if not np.isnan(gb_unit) and gb_unit < gb_min_unit:
                    warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[손익] 공구 단위가({gb_unit:,.0f}) < 공구 최소허용({gb_min_unit:,.0f}) → 상향"})
                    gb_unit = gb_min_unit

            # ----------------------------------------------------
            # Policy validation / auto-correct
            # ----------------------------------------------------
            unit_prices = {
                "홈쇼핑가": hs_unit,
                "공구가": gb_unit,
                **online_units,
            }
            unit_prices2, official_unit2, warns = validate_and_autocorrect(
                unit_prices=unit_prices,
                official_unit=official_unit,
                gb_under_hs_min=gb_under_hs_min,
                online_over_hs_min=online_over_hs_min,
                enforce_monotonic_online=enforce_monotonic,
                auto_correct=auto_correct,
            )
            official_unit = official_unit2

            for w in warns:
                warn_rows.append({"품번": code, "신규품명": name, "메시지": w})

            # ----------------------------------------------------
            # Diagnostics: if still conflicting with market band (indicates 1/2 may fail)
            # ----------------------------------------------------
            # online advantage: always vs ref_mid
            always_final = unit_prices2.get("상시할인가", np.nan)
            if not np.isnan(ref_mid) and not np.isnan(always_final):
                online_adv = (always_final - ref_mid) / max(1.0, ref_mid)
            else:
                online_adv = np.nan

            # feasibility notes for HS/GB vs online lowest (policy)
            hs_final = unit_prices2.get("홈쇼핑가", np.nan)
            gb_final = unit_prices2.get("공구가", np.nan)

            diag = []
            if not np.isnan(online_adv):
                if online_adv < -0.05:
                    diag.append("온라인 상시가 시장중앙 대비 낮음(브랜드/마진 여지 약함)")
                elif online_adv > 0.20:
                    diag.append("온라인 상시가 시장중앙 대비 높음(판매 저항 가능)")
                else:
                    diag.append("온라인 상시가 시장중앙과 균형")

            if mode in ["M1", "M3"]:
                # if HS got pushed up due to guardrail and violates 'HS is attractive' relative to online lowest by too small gap
                if not np.isnan(hs_final) and hs_final > online_lowest * 0.98:
                    diag.append("HS가 온라인 최저와 거의 비슷(방송 메리트 약할 수 있음) → Q/사은품 강화 권장")

            diag_rows.append({
                "품번": code,
                "신규품명": name,
                "모드": mode_label,
                "시장밴드": f"{ref_min:,.0f}~{ref_mid:,.0f}~{ref_max:,.0f}",
                "밴드폭추천(%)": band_pct_reco,
                "온라인이점(상시 vs 시장중앙)": (f"{online_adv*100:+.1f}%" if not np.isnan(online_adv) else ""),
                "진단메모": " / ".join(diag) if diag else "",
            })

            # ----------------------------------------------------
            # Build output rows (set prices) + band
            # - online set price = unit * Q_online
            # - HS set price = unit * Q_hs (mode3 can recommend Q)
            # - GB set price = unit * Q_gb
            # ----------------------------------------------------
            def add(price_type, unit_val, q_set):
                if np.isnan(unit_val):
                    return
                target = unit_val * q_set
                pmin, ptgt, pmax = make_band(target, band_pct_use, rounding_unit)
                rows.append({
                    "품번": code,
                    "신규품명": name,
                    "브랜드(추정)": brand,
                    "가격타입": price_type,
                    "구성Q": q_set,
                    "Target": krw_round(ptgt, rounding_unit),
                    "Min": krw_round(pmin, rounding_unit),
                    "Max": krw_round(pmax, rounding_unit),
                    "밴드폭(%)": band_pct_use,
                })

            # channel types
            add("공구가", unit_prices2.get("공구가", np.nan), gb_q)
            add("홈쇼핑가", unit_prices2.get("홈쇼핑가", np.nan), hs_q)

            # online
            add("상시할인가", unit_prices2.get("상시할인가", np.nan), q_online)
            add("홈사할인가", unit_prices2.get("홈사할인가", np.nan), q_online)
            add("모바일라방가", unit_prices2.get("모바일라방가", np.nan), q_online)
            add("브랜드위크가", unit_prices2.get("브랜드위크가", np.nan), q_online)
            add("원데이 특가", unit_prices2.get("원데이 특가", np.nan), q_online)
            add("공식가(노출)", official_unit, q_online)

            # mode3 extras
            if mode == "M3":
                # show perceived effective unit price with freebies
                # effective unit: set_price / (Q + free_value / always_unit)
                try:
                    denom = max(1e-6, unit_prices2.get("상시할인가", always_unit))
                    hs_set_price = unit_prices2.get("홈쇼핑가", np.nan) * hs_q
                    gb_set_price = unit_prices2.get("공구가", np.nan) * gb_q
                    hs_free_units = free_hs / denom
                    gb_free_units = free_gb / denom
                    hs_eff_unit = hs_set_price / (hs_q + hs_free_units) if (hs_q + hs_free_units) > 0 else np.nan
                    gb_eff_unit = gb_set_price / (gb_q + gb_free_units) if (gb_q + gb_free_units) > 0 else np.nan
                    diag_rows[-1]["3안_사은품가치"] = f"HS {free_hs:,.0f}원 / GB {free_gb:,.0f}원"
                    diag_rows[-1]["3안_체감단위가"] = f"HS {hs_eff_unit:,.0f} / GB {gb_eff_unit:,.0f}"
                    diag_rows[-1]["3안_추천구성Q"] = f"HS Q={hs_q}, GB Q={gb_q}"
                except Exception:
                    pass

        out = pd.DataFrame(rows)
        warn_df = pd.DataFrame(warn_rows)
        diag_df = pd.DataFrame(diag_rows)

        if not out.empty:
            order = {t: i for i, t in enumerate(CHANNEL_TYPES)}
            out["__ord"] = out["가격타입"].map(order).fillna(999).astype(int)
            out = out.sort_values(["품번", "__ord"]).drop(columns="__ord").reset_index(drop=True)

        return out, warn_df, diag_df

    

    st.divider()
    st.subheader("진단 요약(추천 밴드 포함)")
    if diag_df.empty:
        st.info("진단 데이터가 없습니다.")
    else:
        st.dataframe(diag_df, use_container_width=True, height=260)

    st.divider()
    st.subheader("룰 위반/클립/손익 경고 로그")
    if warn_df.empty:
        st.success("경고 없음(현재 설정 기준)")
    else:
        st.warning(f"{len(warn_df):,}건")
        st.dataframe(warn_df, use_container_width=True, height=220)

    st.divider()
    st.subheader("추천 가격 밴드 결과 (Min / Target / Max)")
    if out.empty:
        st.warning("결과가 없습니다. (리서치 입력 부족 또는 앵커 입력 누락 가능)")
    else:
        st.dataframe(out, use_container_width=True, height=360)

        st.subheader("요약(타겟가 피벗)")
        pv = out.pivot_table(index=["품번", "신규품명"], columns="가격타입", values="Target", aggfunc="first")
        pv = pv.reindex(columns=CHANNEL_TYPES, fill_value=np.nan)
        st.dataframe(pv.reset_index(), use_container_width=True, height=300)

        st.divider()
        st.subheader("가격 범위 도식화")
        options = (out["품번"] + " | " + out["신규품명"]).drop_duplicates().tolist()
        picked = st.multiselect("도식화할 상품 선택", options=options, default=options[: min(6, len(options))])
        if picked:
            mask = (out["품번"] + " | " + out["신규품명"]).isin(picked)
            render_range_bars(out[mask], "선택 상품 가격 밴드(추천)")

        st.divider()
        xbytes = to_excel_bytes({
            "result_long": out,
            "result_pivot": pv.reset_index(),
            "diagnosis": diag_df,
            "warnings": warn_df if not warn_df.empty else pd.DataFrame(columns=["품번", "신규품명", "메시지"]),
        })
        st.download_button(
            "결과 엑셀 다운로드",
            data=xbytes,
            file_name="pricing_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ----------------------------
# FORMULA TAB (intuitive + rationale)
# ----------------------------
with tab_formula:
    st.subheader("계산식(로직) — 운영 언어 + 근거 (현재 코드 기준)")

    st.markdown("## 0) 핵심 전제: 리서치 기반 ‘시장 허용 구간(Reference Band)’")
    st.markdown(
        """
### Reference Band(시장단가 밴드) 만들기
- 입력은 “요약값”만 받습니다: 국내 관측(min/max), 경쟁사(min/max/avg), 해외(RRP, 실판매 min/max), 직구가
- 엔진은 그 값들로 아래를 산출합니다:

**RefMin(시장 하한)**  
- 입력된 `min` 값들과 `직구가` 중 가장 낮은 값  
→ “이 가격 아래면 소비자가 이미 더 싼 선택지를 보고 있다”는 의미

**RefMax(시장 상한)**  
- 입력된 `max` 값들과 `해외정가(RRP)` 중 가장 높은 값  
→ “이 가격 위면 구매저항이 급격히 커지는 구간”으로 해석

**RefMid(시장 중심)**  
- 입력된 평균/중앙값 후보들의 ‘중앙값(중간값)’  
→ 특정 값 1개가 튀어도 흔들리지 않게(robust)

#### 근거
- 실무 가격은 단일 숫자가 아니라 “허용 가능한 구간” 안에서 결정됩니다.
- 이 밴드는 온라인/홈쇼핑/공구를 설계할 때 **정합성(시장 일치)을 보장하는 레일** 역할을 합니다.
"""
    )

    st.divider()
    st.markdown("## 1) 온라인 가격 구조(상시 기준 할인율로 생성)")
    st.markdown(
        """
### 온라인은 ‘상시’가 기준(중심축)
- 엔진은 상시 단위가(1개 기준)를 먼저 잡습니다:
  - RefMin~RefMax 사이에서 포지셔닝(0~100%)으로 결정
  - 예: 50%면 밴드 중앙(RefMid 근처)

### 온라인 레벨은 상시에서 할인율로 자동 생성
- 브랜드위크 = 상시 × (1 - 브랜드위크 할인율)
- 라방 = 상시 × (1 - 라방 할인율)
- 홈사 = 상시 × (1 - 홈사 할인율)
- 원데이 = 상시 × (1 - 원데이 할인율)

#### 근거
- 온라인은 상시 운영이 핵심이고, 프로모션은 “상시에서 얼마나 깎느냐”로 움직입니다.
- 실무에서 가장 많이 쓰는 운영 언어가 ‘상시 대비 할인율’입니다.
"""
    )

    st.divider()
    st.markdown("## 2) 공식가(노출)는 ‘정가 프레이밍’ 앵커")
    st.markdown(
        """
### 공식가 = 상시 / (1 - 상시할인율)
- 예: 상시가 40,000원이고 상시할인율 20%면
  - 공식가 = 40,000 / 0.8 = 50,000원

#### 근거
- 소비자가 ‘할인 중’이라고 인지하려면 정가(앵커)가 필요합니다.
- 상시가를 실판매 기준으로 두고, 그 위에 노출용 정가를 얹는 게 가장 흔한 프레이밍입니다.
"""
    )

    st.divider()
    st.markdown("## 3) 정책 룰(카니발 방지/채널 질서) — 이 코드가 강제하는 것")
    st.markdown(
        """
### 채널 질서(단위가 기준)
- 홈쇼핑 단위가 < 온라인 모든 레벨 단위가  (최저가 정책/카니발 방지)
- 공구 단위가 < 홈쇼핑 단위가  (공구 메리트 확보)
- 온라인 레벨 순서: 상시 ≥ 홈사 ≥ 라방 ≥ 브랜드위크 ≥ 원데이
- 공식가 ≥ 온라인 최고 / 온라인은 공식가 이하

#### 근거
- 홈쇼핑은 방송 최저 기대가 강하고, 온라인이 더 싸면 즉시 카니발이 납니다.
- 공구는 “대량/선결제/한정” 성격으로 홈쇼핑보다 더 싸게 잡는 편이 일반적입니다.
"""
    )

    st.divider()
    st.markdown("## 4) 1안/2안/3안 차이(현재 구현)")
    st.markdown(
        """
### 1안) 온라인(시장단가) 먼저
- 온라인을 시장 밴드에 맞춰 ‘잘 팔리는 구조’로 만든다
- HS/공구는 온라인 최저 대비 추가로 싸게 설계
- 이후 룰/손익(옵션)으로 가능 여부 확인

**근거:** 온라인이 메인 채널일 때 ‘온라인 이점’이 가장 중요하기 때문

### 2안) 홈쇼핑 먼저 + 온라인 방어
- HS가 확정/우선인 경우
- 온라인은 HS 대비 +% 방어선 위로 올려서 카니발 방지 + 온라인이점 확보
- 동시에 시장 밴드에서 벗어나지 않도록 클립(경고/보정)

**근거:** 방송 확정이 내려오는 실무 상황을 반영하되 온라인을 망치지 않기 위해

### 3안) 패키지 리디자인(Q + 사은품)
- 가격을 더 깎는 대신:
  - HS/공구 **구성 Q**를 후보 중 추천(세트가↑)
  - **사은품가치(원)**로 체감단가(효과)까지 보여줌

**근거:** 1/2안이 마진/정책/시장캡에 막힐 때 실무 해법은 ‘구성/혜택 설계’이기 때문
"""
    )

    st.divider()
    st.markdown("## 5) 추천 ‘가격 밴드폭(%)’")
    st.markdown(
        """
- 밴드폭은 “불확실성/변동성”을 반영합니다.
- 시장 밴드 폭이 좁으면(RefMax/RefMin이 작으면) 밴드도 좁게
- 시장 밴드 폭이 넓으면 밴드도 넓게

현재 코드의 단순 추천:
- RefMax/RefMin ≤ 1.15 → 6%
- ≤ 1.30 → 8%
- 그 이상 → 10%

**근거:** 리서치가 흔들릴수록 단일가격 고정은 위험하고, 운영 여지를 남기는 게 안전합니다.
"""
    )
