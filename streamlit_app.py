import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v2 (Research-driven)
# - Online-only levels: 공식가(노출), 상시/홈사/브위/원데이/라방
# - Groupbuy: 공구가만
# - Homeshopping: 홈쇼핑가만
# - 리서치 입력(국내/해외/경쟁) → Market Reference Band 생성
# - 결과: Target + 추천 밴드(Min~Target~Max) + 룰검증/자동보정 + 진단
#
# ✅ 변경사항(2026-02-02):
# - 라방이 원데이특가보다 더 싸야 함
#   => 온라인 레벨 순서(고가→저가):
#      상시 ≥ 홈사 ≥ 브랜드위크 ≥ 원데이특가 ≥ 모바일라방가(최저)
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator", layout="wide")

# ----------------------------
# Constants
# ----------------------------
# Online-only price levels
ONLINE_LEVELS = ["상시할인가", "홈사할인가", "브랜드위크가", "원데이 특가", "모바일라방가"]
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
# ----------------------------
def compute_reference_band(row):
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

    if len(mins) == 0 and len(maxs) == 0 and len(mids) == 0:
        return np.nan, np.nan, np.nan, "리서치 입력 없음"

    ref_min = np.nanmin(mins) if len(mins) else (np.nanmin(mids) if len(mids) else np.nan)
    ref_max = np.nanmax(maxs) if len(maxs) else (np.nanmax(mids) if len(mids) else np.nan)
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
# Economics Guardrail
# ----------------------------
def guardrail_min_unit_price(unit_cost, channel_cost_rate, min_margin_rate):
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return unit_cost / denom

# ----------------------------
# Online ladder
# ----------------------------
def build_online_from_always(always_unit, discounts):
    A = float(always_unit)
    out = {}
    out["상시할인가"] = A
    out["홈사할인가"] = A * (1.0 - discounts["홈사할인가"])
    out["브랜드위크가"] = A * (1.0 - discounts["브랜드위크가"])
    out["원데이 특가"] = A * (1.0 - discounts["원데이 특가"])
    out["모바일라방가"] = A * (1.0 - discounts["모바일라방가"])
    return out

def derive_official_from_always(always_unit, list_discount_rate):
    A = float(always_unit)
    d = float(list_discount_rate)
    if d >= 0.95:
        d = 0.95
    if d < 0:
        d = 0.0
    return A / (1.0 - d)

# ----------------------------
# Policy validation / autocorrect
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

    if np.isnan(hs):
        return p, official_unit, warnings

    # GB < HS
    if not np.isnan(gb):
        gb_max = hs * (1.0 - gb_under_hs_min)
        if gb > gb_max:
            msg = f"[위반] 공구 단위가({gb:,.0f})가 홈쇼핑 단위가({hs:,.0f})보다 충분히 낮지 않음. 목표 공구≤HS×(1-{gb_under_hs_min*100:.0f}%)={gb_max:,.0f}"
            if auto_correct:
                p["공구가"] = gb_max
                warnings.append(msg + " → 자동보정: 공구를 하향")
            else:
                warnings.append(msg)

    # Online must be above HS by floor
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

    # ✅ Online monotonic (high→low): 상시 ≥ 홈사 ≥ 브랜드위크 ≥ 원데이 ≥ 라방(최저)
    if enforce_monotonic_online:
        order_high_to_low = ["상시할인가", "홈사할인가", "브랜드위크가", "원데이 특가", "모바일라방가"]
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

    # Official relations
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

if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame(columns=["품번", "신규품명", "브랜드(추정)"])
if "inputs_df" not in st.session_state:
    st.session_state["inputs_df"] = pd.DataFrame(columns=[
        "품번", "신규품명", "브랜드(추정)",
        "온라인기준수량(Q_online)", "홈쇼핑구성(Q_hs)", "공구구성(Q_gb)",
        "랜디드코스트(총원가)",
        "국내관측_min", "국내관측_max",
        "경쟁사_min", "경쟁사_max", "경쟁사_avg",
        "해외정가_RRP", "해외실판매_min", "해외실판매_max",
        "직구가",
        "홈쇼핑가(세트가)_입력", "공구가(세트가)_입력",
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

# ============================================================
# Compute engine
# ============================================================
def compute_for_all(df_in,
                    mode_label,
                    mode,
                    rounding_unit,
                    auto_band,
                    band_pct_manual,
                    positioning,
                    discounts,
                    list_disc,
                    auto_correct,
                    enforce_monotonic,
                    gb_under_hs_min,
                    online_over_hs_min,
                    use_guard,
                    min_margin=0.15,
                    fee_online=0.20,
                    fee_hs=0.35,
                    fee_gb=0.15,
                    # M1/M3
                    hs_under_online=0.10,
                    # M2
                    hs_anchor_source="입력값(홈쇼핑가)",
                    hs_position_in_band=15,
                    # M3
                    hs_q_candidates=(2, 4, 6, 8),
                    gb_q_candidates=(2, 4, 6),
                    ):
    rows = []
    warn_rows = []
    diag_rows = []

    def choose_in_band(ref_min, ref_mid, ref_max, position_pct):
        if np.isnan(ref_min) or np.isnan(ref_max):
            return ref_mid
        p = float(position_pct) / 100.0
        return ref_min + (ref_max - ref_min) * p

    def clip_to_band(x, ref_min, ref_max, code, name, label):
        if np.isnan(ref_min) or np.isnan(ref_max):
            return x
        if x < ref_min:
            warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[클립] {label} 단위가({x:,.0f})가 시장 하한({ref_min:,.0f}) 미만 → 하한으로 상향"})
            return ref_min
        if x > ref_max:
            warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[클립] {label} 단위가({x:,.0f})가 시장 상한({ref_max:,.0f}) 초과 → 상한으로 하향"})
            return ref_max
        return x

    for _, r in df_in.iterrows():
        code = str(r.get("품번", "")).strip()
        name = str(r.get("신규품명", "")).strip()
        brand = str(r.get("브랜드(추정)", "")).strip()

        q_online = int(max(1, safe_float(r.get("온라인기준수량(Q_online)", 1))))
        q_hs_in = int(max(1, safe_float(r.get("홈쇼핑구성(Q_hs)", 1))))
        q_gb_in = int(max(1, safe_float(r.get("공구구성(Q_gb)", 1))))

        landed_total = safe_float(r.get("랜디드코스트(총원가)", np.nan))
        unit_cost = (landed_total / q_online) if (not np.isnan(landed_total) and q_online > 0) else np.nan

        ref_min, ref_mid, ref_max, _ = compute_reference_band(r)
        if np.isnan(ref_mid) and (np.isnan(ref_min) or np.isnan(ref_max)):
            diag_rows.append({
                "품번": code, "신규품명": name,
                "진단": "리서치 입력 부족으로 시장밴드 산출 불가",
                "해결": "국내관측/경쟁/해외/직구 중 최소 1개 이상 입력",
            })
            continue

        band_pct_reco = recommended_band_pct_from_ref(ref_min, ref_max, default=band_pct_manual)
        band_pct_use = band_pct_reco if auto_band else band_pct_manual

        # Online: choose always within band
        always_unit = choose_in_band(ref_min, ref_mid, ref_max, positioning)
        online_units = build_online_from_always(always_unit, discounts)

        # clip online to band
        for k in ONLINE_LEVELS:
            online_units[k] = clip_to_band(online_units[k], ref_min, ref_max, code, name, k)

        official_unit = derive_official_from_always(online_units["상시할인가"], list_disc)
        official_unit = max(official_unit, max(online_units.values()))

        # online lowest (now likely 라방)
        online_lowest = min([online_units[k] for k in ONLINE_LEVELS if not np.isnan(online_units[k])])

        hs_unit = np.nan
        gb_unit = np.nan
        hs_q = q_hs_in
        gb_q = q_gb_in
        free_hs = safe_float(r.get("사은품가치(원)_hs", 0), 0.0)
        free_gb = safe_float(r.get("사은품가치(원)_gb", 0), 0.0)

        if mode == "M1":
            hs_unit = online_lowest * (1.0 - hs_under_online)
            gb_unit = hs_unit * (1.0 - gb_under_hs_min)

        elif mode == "M2":
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
                hs_unit = choose_in_band(ref_min, ref_mid, ref_max, hs_position_in_band)

            gb_unit = hs_unit * (1.0 - gb_under_hs_min)

            # Online must be above HS floor: push always up (then rebuild)
            floor = hs_unit * (1.0 + online_over_hs_min)
            always_unit2 = max(online_units["상시할인가"], floor)
            always_unit2 = clip_to_band(always_unit2, ref_min, ref_max, code, name, "상시할인가(방어)")
            online_units = build_online_from_always(always_unit2, discounts)
            for k in ONLINE_LEVELS:
                online_units[k] = clip_to_band(online_units[k], ref_min, ref_max, code, name, k)
            official_unit = derive_official_from_always(online_units["상시할인가"], list_disc)
            official_unit = max(official_unit, max(online_units.values()))
            online_lowest = min([online_units[k] for k in ONLINE_LEVELS if not np.isnan(online_units[k])])

        else:
            # M3: package redesign (Q + freebies)
            hs_unit_target = online_lowest * (1.0 - hs_under_online)

            hs_unit_min = -np.inf
            gb_unit_min = -np.inf
            if use_guard and not np.isnan(unit_cost):
                hs_unit_min = guardrail_min_unit_price(unit_cost, fee_hs, min_margin)
                gb_unit_min = guardrail_min_unit_price(unit_cost, fee_gb, min_margin)

            hs_unit = max(hs_unit_target, hs_unit_min)
            gb_unit = max(hs_unit * (1.0 - gb_under_hs_min), gb_unit_min)

            def score_set(unit_price, q, free_value, ref_mid_local):
                set_price = unit_price * q
                denom = max(1e-6, online_units["상시할인가"])
                free_units = free_value / denom
                effective_unit = set_price / (q + free_units) if (q + free_units) > 0 else np.inf
                dist = abs(unit_price - ref_mid_local) / max(1.0, ref_mid_local)
                return -effective_unit + (-dist * 2000)

            best_hs = None
            for q in hs_q_candidates:
                sc = score_set(hs_unit, q, free_hs, ref_mid if not np.isnan(ref_mid) else hs_unit)
                if (best_hs is None) or (sc > best_hs[0]):
                    best_hs = (sc, q)
            if best_hs is not None:
                hs_q = int(best_hs[1])

            best_gb = None
            for q in gb_q_candidates:
                sc = score_set(gb_unit, q, free_gb, ref_mid if not np.isnan(ref_mid) else gb_unit)
                if (best_gb is None) or (sc > best_gb[0]):
                    best_gb = (sc, q)
            if best_gb is not None:
                gb_q = int(best_gb[1])

        # Optional guardrails
        if use_guard and not np.isnan(unit_cost):
            online_min_unit = guardrail_min_unit_price(unit_cost, fee_online, min_margin)
            for k in ONLINE_LEVELS:
                if online_units[k] < online_min_unit:
                    warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[손익] {k} 단위가({online_units[k]:,.0f}) < 온라인 최소허용({online_min_unit:,.0f}) → 상향"})
                    online_units[k] = online_min_unit
            official_unit = max(official_unit, max(online_units.values()))

            hs_min_unit = guardrail_min_unit_price(unit_cost, fee_hs, min_margin)
            gb_min_unit = guardrail_min_unit_price(unit_cost, fee_gb, min_margin)
            if not np.isnan(hs_unit) and hs_unit < hs_min_unit:
                warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[손익] HS 단위가({hs_unit:,.0f}) < 홈쇼핑 최소허용({hs_min_unit:,.0f}) → 상향"})
                hs_unit = hs_min_unit
            if not np.isnan(gb_unit) and gb_unit < gb_min_unit:
                warn_rows.append({"품번": code, "신규품명": name, "메시지": f"[손익] 공구 단위가({gb_unit:,.0f}) < 공구 최소허용({gb_min_unit:,.0f}) → 상향"})
                gb_unit = gb_min_unit

        # Policy validation/autocorrect
        unit_prices = {"홈쇼핑가": hs_unit, "공구가": gb_unit, **online_units}
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

        always_final = unit_prices2.get("상시할인가", np.nan)
        hs_final = unit_prices2.get("홈쇼핑가", np.nan)
        gb_final = unit_prices2.get("공구가", np.nan)

        if not np.isnan(ref_mid) and not np.isnan(always_final):
            online_adv = (always_final - ref_mid) / max(1.0, ref_mid)
        else:
            online_adv = np.nan

        diag = []
        if not np.isnan(online_adv):
            if online_adv < -0.05:
                diag.append("온라인 상시가 시장중앙 대비 낮음(브랜드/마진 여지 약함)")
            elif online_adv > 0.20:
                diag.append("온라인 상시가 시장중앙 대비 높음(판매 저항 가능)")
            else:
                diag.append("온라인 상시가 시장중앙과 균형")

        if mode in ["M1", "M3"]:
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

        add("공구가", unit_prices2.get("공구가", np.nan), gb_q)
        add("홈쇼핑가", unit_prices2.get("홈쇼핑가", np.nan), hs_q)

        # Online (Q_online)
        add("상시할인가", unit_prices2.get("상시할인가", np.nan), q_online)
        add("홈사할인가", unit_prices2.get("홈사할인가", np.nan), q_online)
        add("브랜드위크가", unit_prices2.get("브랜드위크가", np.nan), q_online)
        add("원데이 특가", unit_prices2.get("원데이 특가", np.nan), q_online)
        add("모바일라방가", unit_prices2.get("모바일라방가", np.nan), q_online)
        add("공식가(노출)", official_unit, q_online)

    out = pd.DataFrame(rows)
    warn_df = pd.DataFrame(warn_rows)
    diag_df = pd.DataFrame(diag_rows)

    if not out.empty:
        order = {t: i for i, t in enumerate(CHANNEL_TYPES)}
        out["__ord"] = out["가격타입"].map(order).fillna(999).astype(int)
        out = out.sort_values(["품번", "__ord"]).drop(columns="__ord").reset_index(drop=True)

    return out, warn_df, diag_df

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

    # 공통 파라미터 (계산 버튼 눌렀을 때만 반영되도록 form 아래로 내려도 되지만,
    # 너 UX 기준: 편집 먼저 하고 계산 버튼 누르기 흐름이면 여기에 둬도 OK
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

    st.divider()
    st.subheader("온라인 가격 구조(상시 기준 할인율)")
    o1, o2, o3, o4, o5, o6 = st.columns([1, 1, 1, 1, 1, 1])

    # ✅ 기본값 변경: 라방이 원데이보다 싸도록(라방 할인율 > 원데이 할인율)
    with o1:
        d_brandweek = st.slider("브랜드위크 할인율(%)", 0, 70, 25, 1) / 100.0
    with o2:
        d_homesale = st.slider("홈사 할인율(%)", 0, 70, 10, 1) / 100.0
    with o3:
        d_oneday = st.slider("원데이 할인율(%)", 0, 80, 30, 1) / 100.0
    with o4:
        d_live = st.slider("라방 할인율(%)", 0, 80, 40, 1) / 100.0
    with o5:
        list_disc = st.slider("상시할인율(정가 프레이밍) (%)", 0, 80, 20, 1) / 100.0
    with o6:
        st.caption("정가=상시/(1-상시할인율)")

    discounts = {
        "브랜드위크가": d_brandweek,
        "홈사할인가": d_homesale,
        "원데이 특가": d_oneday,
        "모바일라방가": d_live,
    }

    st.divider()
    st.subheader("정책 룰(카니발 방지/채널 질서)")
    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        auto_correct = st.toggle("위반 시 자동보정", value=True)
    with r2:
        enforce_monotonic = st.toggle("온라인 레벨 순서 강제(상시≥홈사≥브위≥원데이≥라방)", value=True)
    with r3:
        gb_under_hs_min = st.slider("공구는 HS보다 최소 -%", 0, 20, 3, 1) / 100.0
        online_over_hs_min = st.slider("온라인은 HS보다 최소 +%", 0, 80, 15, 1) / 100.0

    st.divider()
    st.subheader("손익 가드레일(선택)")
    use_guard = st.toggle("원가/비용/최소마진으로 불가능 가격 차단", value=False)
    min_margin = 0.15
    fee_online = 0.20
    fee_hs = 0.35
    fee_gb = 0.15
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

    # Mode-specific params
    if mode == "M1":
        st.divider()
        st.subheader("1안 추가 파라미터 (온라인→HS/공구 설계)")
        hs_under_online = st.slider("HS 단위가는 온라인 최저(라방) 대비 -%(목표)", 0, 40, 10, 1) / 100.0
        st.caption("라방이 온라인 최저이므로, HS는 라방보다 더 낮게 설계됨(정책).")

        hs_anchor_source = "입력값(홈쇼핑가)"
        hs_position_in_band = 15
        hs_q_candidates = (2, 4, 6, 8)
        gb_q_candidates = (2, 4, 6)

    elif mode == "M2":
        st.divider()
        st.subheader("2안 추가 파라미터 (HS→온라인 방어)")
        hs_anchor_source = st.radio("HS 앵커", ["입력값(홈쇼핑가)", "시장밴드에서 추천"], index=0)
        hs_position_in_band = st.slider("HS 포지셔닝(밴드 내) (%)", 0, 100, 15, 5)

        hs_under_online = 0.10
        hs_q_candidates = (2, 4, 6, 8)
        gb_q_candidates = (2, 4, 6)

    else:
        st.divider()
        st.subheader("3안 추가 파라미터 (패키지 리디자인: Q + 사은품)")
        hs_q_candidates = st.multiselect("HS 구성 후보(Q_hs)", options=[1,2,3,4,5,6,8,10], default=[2,4,6,8])
        gb_q_candidates = st.multiselect("공구 구성 후보(Q_gb)", options=[1,2,3,4,5,6,8,10], default=[2,4,6])

        hs_under_online = st.slider("HS 단위가 목표: 온라인 최저(라방) 대비 -%", 0, 40, 8, 1) / 100.0
        hs_anchor_source = "입력값(홈쇼핑가)"
        hs_position_in_band = 15

    # ✅ 입력(편집)과 계산을 분리: form + 버튼
    with st.form("input_form", clear_on_submit=False):
        edited_df = st.data_editor(
            st.session_state["inputs_df"],
            key="inputs_editor",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                # ⚠️ 기존 column_config를 여기에 그대로 붙여넣어야 함
            },
            height=360,
        )

        c_save, c_calc = st.columns([1, 1])
        save_clicked = c_save.form_submit_button("입력값 적용(저장)", type="secondary")
        calc_clicked = c_calc.form_submit_button("계산 실행", type="primary")

    if save_clicked or calc_clicked:
        st.session_state["inputs_df"] = edited_df.copy()

    if not calc_clicked:
        st.info("입력값을 수정한 뒤 '계산 실행'을 눌러 결과를 업데이트하세요.")
        st.stop()

    out, warn_df, diag_df = compute_for_all(
        st.session_state["inputs_df"],
        mode_label=mode_label,
        mode=mode,
        rounding_unit=rounding_unit,
        auto_band=auto_band,
        band_pct_manual=band_pct_manual,
        positioning=positioning,
        discounts=discounts,
        list_disc=list_disc,
        auto_correct=auto_correct,
        enforce_monotonic=enforce_monotonic,
        gb_under_hs_min=gb_under_hs_min,
        online_over_hs_min=online_over_hs_min,
        use_guard=use_guard,
        min_margin=min_margin,
        fee_online=fee_online,
        fee_hs=fee_hs,
        fee_gb=fee_gb,
        hs_under_online=hs_under_online,
        hs_anchor_source=hs_anchor_source,
        hs_position_in_band=hs_position_in_band,
        hs_q_candidates=tuple(hs_q_candidates),
        gb_q_candidates=tuple(gb_q_candidates),
    )

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
# FORMULA TAB
# ----------------------------
with tab_formula:
    st.subheader("계산식(로직) — 운영 언어 + 근거 (현재 코드 기준)")

    st.markdown("## 온라인 레벨 순서(카니발 방지)")
    st.markdown(
        """
### 온라인 레벨은 아래 ‘가격 질서’를 강제합니다(고가→저가)
**상시할인가 ≥ 홈사할인가 ≥ 브랜드위크가 ≥ 원데이 특가 ≥ 모바일라방가(최저)**

#### 근거
- 라방은 ‘방송 한정/즉시 구매 유도’ 성격이 강해서, 일반적으로 온라인 내에서도 최저로 떨어지는 경우가 많음.
- 원데이 특가는 기간 한정이지만, 라방보다 “즉시성/전환유도”가 약한 경우가 있어 라방을 더 강하게 내릴 수 있음.
"""
    )

    st.divider()
    st.markdown("## 공식가(노출)")
    st.markdown(
        """
- 공식가 = 상시 / (1 - 상시할인율)
- 공식가는 온라인 레벨 중 최고가 이상으로 유지(정가 앵커 역할)
"""
    )

# ----------------------------
# DATA TAB already defined above
# ----------------------------
