import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

# ============================================================
# IBR Dynamic Pricing - Rule Simulator (3 Methods)
# - Upload product master (품번/신규품명)
# - Search & select products, then price inputs editable
# - Strict vocabulary:
#   * 온라인에서만: 공식가(노출), 상시/홈사/라방/브위/원데이
#   * 공구: 공구가만
#   * 홈쇼핑: 홈쇼핑가만
# - 3 Methods (renamed for clarity)
#   A) 공구 기준 온라인 사다리(비율)
#   B) 홈쇼핑(기준) + 공구는 더 싸게 + 온라인 방어(Uplift)
#   C) 손익 방어 포함(원가/비용/최소마진 Guardrail + A/B 결과 보정)
# - No plotly/matplotlib (HTML/CSS range bars)
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator", layout="wide")

# ----------------------------
# Constants
# ----------------------------
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
/* 전체 가독성: 다크/라이트 모두 대비 확보 */
html, body, [class*="css"] {
  font-size: 14px !important;
}

/* 표/라벨 대비 강화 */
.block-container { padding-top: 1.2rem; }

/* 데이터프레임 헤더 고정 느낌 */
thead tr th {
  position: sticky;
  top: 0;
  z-index: 1;
}

/* Range bar styles */
.dp-wrap { border-top: 1px solid rgba(128,128,128,0.35); padding-top: 6px; }
.dp-row { display:flex; align-items:center; gap:12px; padding:7px 0; }
.dp-label {
  width: 520px;
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
.hint { color: var(--text-color); opacity: 0.75; }
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
    """
    band_pct=6 이면 Target 기준 ±3%로 Min/Max
    """
    t = float(target_price)
    half = pct_to_rate(band_pct) / 2.0
    pmin = t * (1.0 - half)
    pmax = t * (1.0 + half)
    return krw_round(pmin, rounding_unit), krw_round(t, rounding_unit), krw_round(pmax, rounding_unit)

def guardrail_min_price(landed_cost: float, channel_cost_rate: float, min_margin_rate: float) -> float:
    """
    Price*(1 - cost_rate) - landed_cost >= Price*min_margin
    => Price >= landed_cost / (1 - cost_rate - min_margin)
    """
    denom = 1.0 - channel_cost_rate - min_margin_rate
    if denom <= 0:
        return float("inf")
    return landed_cost / denom

def infer_brand_from_name(name: str) -> str:
    """
    브랜드가 제품명에 녹아있다는 전제에서 아주 단순 추정:
    - 첫 토큰(첫 공백 전)을 브랜드 후보로 사용
    """
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
    """
    df_dict: {sheet_name: dataframe}
    """
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()

# ----------------------------
# Engines (unit basis)
# ----------------------------
def method_A_from_groupbuy(G_unit: float, a: float, b: float, c: float, hs_premium_over_G: float):
    """
    A) 공구 기준 온라인 사다리(비율)
    - 공구(Unit)=G
    - 홈쇼핑(Unit)=G*(1+hs_premium)  (공구가 홈쇼핑보다 싸게 => premium 양수)
    - 모바일(Unit)=G*(1+a)
    - 브위(Unit)=G*(1+b)
    - 상시(Unit)=G*(1+c)
    - 홈사(Unit)=라방 근처(기본 라방*0.98)
    - 원데이(Unit)=브위보다 낮게(기본 브위*0.85) but 최소 공구 이상
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
    - 홈쇼핑(Unit)=H
    - 공구(Unit)=H*(1-d)  (공구가 홈쇼핑보다 더 싸게)
    - 온라인 하한(Unit)=H*(1+u)  (카니발 방지, 온라인은 항상 HS보다 비싸게)
    - 온라인 레벨=공구 기반 사다리 + 하한으로 바닥 상향
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
    """
    공식가(노출) 생성:
    - mode = "상시=공식"  -> 공식가 = 상시
    - mode = "정가 프레이밍" -> 공식가 = 상시*(1+premium)
    """
    if mode == "상시=공식":
        return float(always_unit)
    return float(always_unit) * (1.0 + premium)

# ----------------------------
# Load master file & mapping
# ----------------------------
def load_master(file) -> pd.DataFrame:
    df0 = pd.read_excel(file)
    # expected: No / 품번 / 신규품명
    cols = list(df0.columns)

    # auto-detect
    code_col = None
    name_col = None
    for c in cols:
        if str(c).strip() in ["품번", "상품코드", "제품코드", "SKU", "코드"]:
            code_col = c
        if str(c).strip() in ["신규품명", "통일제품명", "통일 제품명", "제품명", "상품명"]:
            name_col = c

    # if missing, user mapping later
    if code_col is None or name_col is None:
        df0 = df0.copy()
        df0.columns = [str(c) for c in df0.columns]
    else:
        df0 = df0.rename(columns={code_col: "품번", name_col: "신규품명"})

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
st.caption("파일 업로드 → 품번/제품명 검색 선택 → 3가지 방법론 중 선택 → 가격밴드 & 도식화")

tab_sim, tab_formula, tab_data = st.tabs(["시뮬레이터", "계산식(로직)", "데이터 업로드/선택"])

# Session state
if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame(columns=["품번", "신규품명", "브랜드(추정)"])

if "inputs_df" not in st.session_state:
    st.session_state["inputs_df"] = pd.DataFrame(columns=[
        "품번", "신규품명", "브랜드(추정)",
        "구성수량(Q)", "랜디드코스트(총원가)",
        "홈쇼핑가(세트가)", "공구가(세트가)"
    ])

# ----------------------------
# DATA TAB: Upload and select
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
            st.info("업로드하면 자동으로 품번/신규품명을 읽어옵니다. (현재는 비어있음)")

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

        # selection via multiselect (works reliably)
        options = (view["품번"].fillna("") + " | " + view["신규품명"].fillna("")).tolist()
        picked = st.multiselect("선택(멀티 가능)", options=options, default=[])

        if st.button("선택 상품을 입력 테이블로 추가", type="primary", disabled=(len(picked) == 0)):
            sel_codes = [p.split(" | ", 1)[0].strip() for p in picked]
            sel_df = master_df[master_df["품번"].isin(sel_codes)].copy()

            # merge into inputs_df
            base = st.session_state["inputs_df"].copy()
            if base.empty:
                base = pd.DataFrame(columns=[
                    "품번", "신규품명", "브랜드(추정)",
                    "구성수량(Q)", "랜디드코스트(총원가)",
                    "홈쇼핑가(세트가)", "공구가(세트가)"
                ])

            # defaults
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

            # flexible mapping
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
                st.error("업로드한 입력값 파일에 '품번' 컬럼(또는 상품코드/제품코드/SKU)이 필요합니다.")
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

                # update only if new exists
                for f in ["구성수량(Q)", "랜디드코스트(총원가)", "홈쇼핑가(세트가)", "공구가(세트가)"]:
                    if f"{f}_new" in base.columns:
                        base[f] = base[f].where(base[f].notna(), base[f"{f}_new"])
                        base = base.drop(columns=[f"{f}_new"])

                st.session_state["inputs_df"] = base
                st.success("입력값 병합 완료(품번 기준).")

        except Exception as e:
            st.error(f"입력값 업로드 처리 오류: {e}")

# ----------------------------
# SIM TAB: Inputs & compute
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
        # editable inputs
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
        st.subheader("방법론 파라미터(공통/선택)")
        p1, p2, p3 = st.columns(3)

        with p1:
            st.caption("온라인 레벨 사다리(공구 기준 비율)")
            a = st.slider("모바일라방: 공구 대비 +%", 0, 30, 5, 1) / 100.0
            b = st.slider("브위(행사): 공구 대비 +%", 0, 80, 25, 1) / 100.0
            c = st.slider("상시: 공구 대비 +%", 0, 120, 60, 1) / 100.0

        with p2:
            st.caption("홈쇼핑/공구 관계 & 온라인 방어(B/C에서 주로 사용)")
            hs_premium_over_G = st.slider("홈쇼핑: 공구 대비 +% (A에서 사용)", 0, 20, 5, 1) / 100.0
            d = st.slider("공구: 홈쇼핑 대비 -% (B/C에서 사용)", 0, 20, 4, 1) / 100.0
            u = st.slider("온라인 하한: 홈쇼핑 대비 +% (B/C)", 0, 60, 15, 1) / 100.0

        with p3:
            st.caption("공식가(노출) 생성 규칙(온라인에서만)")
            official_mode = st.radio("공식가 생성", ["상시=공식", "상시 위에 정가 프레이밍"], index=1)
            official_premium = st.slider("정가 프레이밍(%)", 0, 80, 20, 1) / 100.0

        # C: guardrail params
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

        # ----------------------------
        # Compute
        # ----------------------------
        def compute_prices(df_in: pd.DataFrame) -> pd.DataFrame:
            rows = []
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

                # method selection
                unit_prices = None
                if method_key == "A":
                    if np.isnan(gb_unit):
                        continue
                    unit_prices = method_A_from_groupbuy(gb_unit, a, b, c, hs_premium_over_G)
                elif method_key == "B":
                    if np.isnan(hs_unit):
                        continue
                    unit_prices = method_B_from_hs(hs_unit, d, u, a, b, c)
                else:
                    # C: prefer HS, fallback GB
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

                        # push up targets
                        if "공구가" in unit_prices:
                            unit_prices["공구가"] = max(unit_prices["공구가"], guard_gb2)
                        if "홈쇼핑가" in unit_prices:
                            unit_prices["홈쇼핑가"] = max(unit_prices["홈쇼핑가"], guard_hs2)
                        for k in ONLINE_LEVELS:
                            if k in unit_prices:
                                unit_prices[k] = max(unit_prices[k], guard_online)

                        unit_prices["__guard_online"] = guard_online

                # official (online only)
                always_unit = unit_prices.get("상시할인가", np.nan)
                if np.isnan(always_unit):
                    continue
                official_unit = derive_official_from_always(always_unit, official_mode, official_premium)

                # (safety) online prices <= official
                for k in ONLINE_LEVELS:
                    if k in unit_prices:
                        unit_prices[k] = min(unit_prices[k], official_unit)

                # build output rows (set-level)
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

                # 공구/홈쇼핑은 각각 1줄만
                if "공구가" in unit_prices:
                    add_row("공구가", unit_prices["공구가"])
                if "홈쇼핑가" in unit_prices:
                    add_row("홈쇼핑가", unit_prices["홈쇼핑가"])

                # 온라인은 공식가 + 5레벨
                add_row("공식가(노출)", official_unit)
                for k in ONLINE_LEVELS:
                    if k in unit_prices:
                        add_row(k, unit_prices[k])

            out = pd.DataFrame(rows)
            if out.empty:
                return out

            order = {t: i for i, t in enumerate(CHANNEL_TYPES)}
            out["__ord"] = out["가격타입"].map(order).fillna(999).astype(int)
            out = out.sort_values(["품번", "__ord"]).drop(columns="__ord").reset_index(drop=True)
            return out

        out = compute_prices(st.session_state["inputs_df"])

        st.divider()
        st.subheader("결과")
        if out.empty:
            need = "공구가" if method_key == "A" else "홈쇼핑가"
            st.warning(f"계산 가능한 상품이 없습니다. ({method_label}은 기본적으로 '{need}(세트가)' 입력이 필요합니다)")
        else:
            st.dataframe(out, use_container_width=True, height=360)

            # Pivot view (Target)
            st.subheader("요약(타겟가 피벗)")
            pv = out.pivot_table(index=["품번", "신규품명"], columns="가격타입", values="Target", aggfunc="first")
            pv = pv.reindex(columns=CHANNEL_TYPES, fill_value=np.nan)
            st.dataframe(pv.reset_index(), use_container_width=True, height=320)

            # Visualization
            st.divider()
            st.subheader("가격 범위 도식화 (Min / Target / Max)")
            options = (out["품번"] + " | " + out["신규품명"]).drop_duplicates().tolist()
            picked = st.multiselect("도식화할 상품 선택", options=options, default=options[: min(6, len(options))])
            if picked:
                mask = (out["품번"] + " | " + out["신규품명"]).isin(picked)
                render_range_bars(out[mask], "선택 상품 가격 밴드")

            # Download
            st.divider()
            xbytes = to_excel_bytes({"result_long": out, "result_pivot": pv.reset_index()})
            st.download_button(
                "결과 엑셀 다운로드",
                data=xbytes,
                file_name="pricing_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ----------------------------
# FORMULA TAB: show formulas per method
# ----------------------------
with tab_formula:
    st.subheader("계산식(로직) 보기")
    st.caption("선택된 3안별로 어떤 입력을 기반으로 어떤 순서로 계산되는지 표시합니다.")

    st.markdown("### 공통: 가격 밴드(Min~Target~Max)")
    st.markdown(
        r"""
- Target 가격이 \(P\), 밴드폭이 \(B\%\)일 때 (예: 6% = ±3%):
\[
Min = P \times (1 - \frac{B}{2})
\]
\[
Max = P \times (1 + \frac{B}{2})
\]
- 반올림 단위(10/100/1000원)로 Min/Target/Max를 반올림
"""
    )

    st.markdown("### 공통: 공식가(노출) 생성(온라인 전용)")
    st.markdown(
        r"""
- \(A\) = 상시할인가(Target 기준, 단위가)
- 모드 1) **상시=공식**:
\[
List = A
\]
- 모드 2) **정가 프레이밍** (프리미엄 \(p\)):
\[
List = A \times (1+p)
\]
"""
    )

    st.divider()
    st.markdown("## A) 공구 기준 온라인 사다리(비율)")
    st.markdown(
        r"""
입력(핵심):
- 공구 단위가 \(G\)
- 비율 \(a,b,c\): (모바일/브위/상시)
- 홈쇼핑 프리미엄 \(h\): 홈쇼핑이 공구보다 비싸도록

계산:
\[
공구 = G
\]
\[
홈쇼핑 = G \times (1+h)
\]
\[
모바일 = G \times (1+a)
\]
\[
브랜드위크(행사) = G \times (1+b)
\]
\[
상시 = G \times (1+c)
\]
보조 레벨(기본 규칙):
- 홈사: \(홈사 \approx 모바일 \times 0.98\) (단, 공구 이상)
- 원데이: \(원데이 \approx 브랜드위크 \times 0.85\) (단, 공구 이상)

그 후:
- 공식가(List) 생성 → 온라인 레벨은 List 이하로 캡
- 세트가는 단위가 × 구성수량(Q)
"""
    )

    st.divider()
    st.markdown("## B) 홈쇼핑 기준 + 공구 더 싸게 + 온라인 방어")
    st.markdown(
        r"""
입력(핵심):
- 홈쇼핑 단위가 \(H\)
- 공구 할인 \(d\): 공구가 홈쇼핑보다 더 싸게
- 온라인 방어 업율 \(u\): 온라인이 홈쇼핑보다 항상 비싸게(카니발 방지)
- 비율 \(a,b,c\): (모바일/브위/상시)

계산:
\[
홈쇼핑 = H
\]
\[
공구 = H \times (1-d)
\]
온라인 최저 허용선(방어선):
\[
OnlineFloor = H \times (1+u)
\]
온라인 레벨(사다리 + 방어선 바닥 상향):
\[
모바일 = \max(공구 \times (1+a), OnlineFloor)
\]
\[
브랜드위크 = \max(공구 \times (1+b), OnlineFloor)
\]
\[
상시 = \max(공구 \times (1+c), OnlineFloor)
\]
보조 레벨:
- 홈사: \(\max(모바일 \times 0.98, OnlineFloor)\)
- 원데이: \(\max(브랜드위크 \times 0.85, OnlineFloor)\)

그 후:
- 공식가(List) 생성 → 온라인 레벨은 List 이하로 캡
- 세트가는 단위가 × 구성수량(Q)
"""
    )

    st.divider()
    st.markdown("## C) 손익 방어 포함(원가/비용/마진 Guardrail)")
    st.markdown(
        r"""
C안은 A 또는 B로 먼저 가격을 만든 뒤, **원가/비용/최소마진 기반 하한(Guardrail)**로 끌어올립니다.

입력(추가):
- 랜디드코스트 단위가 \(LC\)
- 채널 비용율 \(C_{online}, C_{gb}, C_{hs}\)
- 최소마진 \(m\)

채널별 최소 허용가:
\[
Guardrail = \frac{LC}{1 - C - m}
\]
적용:
- 공구가 ≥ Guardrail(공구)
- 홈쇼핑가 ≥ Guardrail(홈쇼핑)
- 온라인 레벨/공식가 ≥ Guardrail(온라인)

그 후:
- 다시 공식가(List) 생성/캡 규칙 적용
- 세트가는 단위가 × 구성수량(Q)
"""
    )

    st.info("원하면 다음 단계로: (1) '홈쇼핑이 항상 최저/공구가 더 최저'를 강제하는 체크 룰, (2) 레벨별 정확한 위치(홈사/원데이)를 파라미터로 분리해서 더 정밀하게 만들 수 있어요.")

