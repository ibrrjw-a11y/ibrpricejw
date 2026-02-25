import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v6.1 (Cost-only → Auto prices → You adjust)
#
# ✅ 원가 파일(상품명 통일 + 원가)만 업로드하면:
#   - SKU별 레인지(Min~Max=MSRP) 자동 생성
#   - 레인지 안을 밴드(가격영역)로 분할하여 채널별 추천가(Target) 자동 산출
#   - 채널 비용(수수료/PG/배송/마케팅/반품) 바꾸면 즉시 재계산
#
# ✅ 이후 사용자는:
#   - Min(최저가), Max(최고가/MSRP)를 수동으로 올리거나 내릴 수 있음 → 전체 가격이 같이 이동
#   - 특정 가격영역(채널)의 가격을 직접 override 가능(룰 위반도 허용) → 경고만 표시
#   - 마진(기여이익/기여이익률)과 Floor(손익하한) 침범 여부를 즉시 확인
#   - 세트는 앱에서 BOM(구성품 리스트업) → 자동 추천가 + 구성품별 실질단가/할인율 + 카니발 체크
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator v6.1", layout="wide")

PRICE_ZONES = ["공구", "홈쇼핑", "폐쇄몰", "모바일라방", "원데이", "브랜드위크", "홈사", "상시", "오프라인", "MSRP"]

DEFAULT_CHANNELS = [
    ("오프라인",     0.50, 0.00, 0.0,    0.00, 0.00, 0.0),
    ("자사몰",       0.05, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("스마트스토어", 0.05, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("쿠팡",         0.25, 0.00, 3000.0, 0.00, 0.00, 0.0),
    ("오픈마켓",     0.15, 0.00, 3000.0, 0.00, 0.00, 0.0),
    ("홈사",         0.30, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("공구",         0.50, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("홈쇼핑",       0.55, 0.00, 0.0,    0.00, 0.00, 0.0),  # 홈쇼핑 택배부담(기본)
    ("모바일라이브", 0.40, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("폐쇄몰",       0.25, 0.00, 3000.0, 0.00, 0.00, 0.0),
]

DEFAULT_ZONE_MAP = {
    "공구": "공구",
    "홈쇼핑": "홈쇼핑",
    "폐쇄몰": "폐쇄몰",
    "모바일라방": "모바일라이브",
    "원데이": "자사몰",
    "브랜드위크": "자사몰",
    "홈사": "홈사",
    "상시": "자사몰",
    "오프라인": "오프라인",
    "MSRP": "자사몰",
}

# 기본 밴드 경계(0~100): 마우스로 조정 가능
DEFAULT_BOUNDARIES = [0, 10, 20, 30, 42, 52, 62, 72, 84, 94, 100]

def default_zone_target_pos(boundaries):
    out = {}
    for i, z in enumerate(PRICE_ZONES):
        out[z] = (boundaries[i] + boundaries[i+1]) / 2
    return out

# -----------------------------
# Utilities
# -----------------------------
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

def krw_ceil(x, unit=100):
    try:
        x = float(x)
        return int(np.ceil(x / unit) * unit)
    except Exception:
        return 0

def to_excel_bytes(df_dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()

# -----------------------------
# Load cost master (상품명 통일 + 원가)
# -----------------------------
def find_cost_sheet(xls: pd.ExcelFile):
    candidates = []
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sh, header=2)
            cols = [str(c).strip() for c in df.columns]
            colset = set(cols)
            if ("상품코드" in colset) and any(c in colset for c in ["원가 (vat-)", "원가", "매입원가", "랜디드코스트", "랜디드코스트(총원가)"]):
                candidates.append(sh)
        except Exception:
            continue
    return candidates[0] if candidates else xls.sheet_names[0]

def load_products_from_cost_master(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    sh = find_cost_sheet(xls)
    df = pd.read_excel(xls, sheet_name=sh, header=2)
    df.columns = [str(c).strip() for c in df.columns]

    code_col = "상품코드" if "상품코드" in df.columns else None
    name_col = "상품명" if "상품명" in df.columns else None
    brand_col = "브랜드" if "브랜드" in df.columns else None

    cost_col = None
    for c in ["원가 (vat-)", "원가", "매입원가", "랜디드코스트", "랜디드코스트(총원가)"]:
        if c in df.columns:
            cost_col = c
            break

    out = pd.DataFrame({
        "품번": df[code_col].astype(str).str.strip() if code_col else "",
        "상품명": df[name_col].astype(str).str.strip() if name_col else "",
        "브랜드": df[brand_col].astype(str).str.strip() if brand_col else "",
        "원가": pd.to_numeric(df[cost_col], errors="coerce") if cost_col else np.nan,
    })
    out = out[out["품번"].ne("")].drop_duplicates(subset=["품번"]).reset_index(drop=True)

    # optional overrides
    out["MSRP_오버라이드"] = np.nan
    out["Min_오버라이드"] = np.nan
    out["Max_오버라이드"] = np.nan
    out["운영여부"] = True
    return out

# -----------------------------
# Economics
# -----------------------------
def floor_price(cost_total, q_orders, fee, pg, mkt, ship_per_order, ret_rate, ret_cost_order, min_cm):
    denom = 1.0 - (fee + pg + mkt + min_cm)
    if denom <= 0:
        return float("inf")
    ship_unit = ship_per_order / max(1, q_orders)
    ret_unit = (ret_rate * ret_cost_order) / max(1, q_orders)
    return (cost_total + ship_unit + ret_unit) / denom

def contrib_metrics(price, cost_total, q_orders, fee, pg, mkt, ship_per_order, ret_rate, ret_cost_order):
    if price <= 0:
        return np.nan, np.nan
    ship_unit = ship_per_order / max(1, q_orders)
    ret_unit = (ret_rate * ret_cost_order) / max(1, q_orders)
    net = price * (1.0 - fee - pg - mkt) - ship_unit - ret_unit - cost_total
    return net, net / price

# -----------------------------
# Auto MSRP / Auto range from cost
# -----------------------------
def compute_auto_range_from_cost(
    cost_total: float,
    channels_df: pd.DataFrame,
    zone_map: dict,
    boundaries: list,
    rounding_unit: int,
    min_cm: float,
    max_cost_ratio: float,
    include_zones: list,
    min_zone: str = "공구",
    msrp_override=np.nan,
):
    """
    원가만으로 Min~Max(=MSRP) 자동 생성.
    - Min = min_zone Floor
    - MSRP_base = 원가/max_cost_ratio
    - 각 존의 Floor가 그 존의 BandHigh 안에 들어오도록 Max를 자동 상향
    """
    ch_map = channels_df.set_index("채널명").to_dict("index")

    def zone_floor(z):
        ch = zone_map.get(z, "자사몰")
        p = ch_map.get(ch, None)
        if p is None:
            return np.nan
        return floor_price(
            cost_total=cost_total,
            q_orders=1,
            fee=p["수수료율"],
            pg=p["PG"],
            mkt=p["마케팅비"],
            ship_per_order=p["배송비(주문당)"],
            ret_rate=p["반품률"],
            ret_cost_order=p["반품비(주문당)"],
            min_cm=min_cm
        )

    min_auto = zone_floor(min_zone)
    if min_auto != min_auto or min_auto <= 0:
        return np.nan, np.nan, {"note": "Min(Floor) 산출 불가: 원가/채널 파라미터를 확인하세요."}

    min_auto = krw_round(min_auto, rounding_unit)

    msrp_base = krw_ceil(cost_total / max_cost_ratio, rounding_unit) if (cost_total == cost_total and cost_total > 0) else np.nan

    # Max requirement: make sure floors fit into each zone's BandHigh
    max_req = []
    for i, z in enumerate(PRICE_ZONES):
        if z not in include_zones:
            continue
        fz = zone_floor(z)
        if fz != fz:
            continue
        end = boundaries[i+1] / 100.0
        if end <= 0:
            continue
        if fz > min_auto:
            max_needed = min_auto + (fz - min_auto) / end
            max_req.append(max_needed)

    candidates = []
    if msrp_base == msrp_base:
        candidates.append(msrp_base)
    if max_req:
        candidates.append(max(max_req))
    if msrp_override == msrp_override and msrp_override is not None and msrp_override > 0:
        candidates.append(msrp_override)

    max_auto = krw_ceil(max(candidates), rounding_unit) if candidates else np.nan

    note = ""
    if max_req and (max_auto > (msrp_base if msrp_base==msrp_base else 0)):
        note = "채널 손익하한이 밴드에 들어오도록 MSRP(=Max)를 자동 상향"
    return min_auto, max_auto, {"note": note, "msrp_base": msrp_base}

# -----------------------------
# Build zone table with adjustable Min/Max and optional overrides
# -----------------------------
def build_zone_table(
    cost_total: float,
    min_price: float,
    max_price: float,
    channels_df: pd.DataFrame,
    zone_map: dict,
    boundaries: list,
    target_pos: dict,
    rounding_unit: int,
    min_cm: float,
    overrides_df: pd.DataFrame,
    item_type: str,
    item_id: str
):
    ch_map = channels_df.set_index("채널명").to_dict("index")
    rows = []
    if min_price != min_price or max_price != max_price or max_price <= min_price:
        return pd.DataFrame()

    span = max_price - min_price
    for i, z in enumerate(PRICE_ZONES):
        start = boundaries[i] / 100.0
        end = boundaries[i+1] / 100.0
        band_low = min_price + span * start
        band_high = min_price + span * end
        pos = target_pos.get(z, (boundaries[i]+boundaries[i+1])/2) / 100.0
        target_raw = min_price + span * pos

        ch = zone_map.get(z, "자사몰")
        p = ch_map.get(ch, None)
        if p is None:
            continue

        floor = floor_price(
            cost_total=cost_total,
            q_orders=1,
            fee=p["수수료율"],
            pg=p["PG"],
            mkt=p["마케팅비"],
            ship_per_order=p["배송비(주문당)"],
            ret_rate=p["반품률"],
            ret_cost_order=p["반품비(주문당)"],
            min_cm=min_cm
        )

        # default recommended within band + floor clamp
        status = "OK"
        target = max(target_raw, floor)
        if floor > band_high:
            status = "불가(Floor>BandHigh)"
            target = band_high
        elif target > band_high:
            status = "클립(Target→BandHigh)"
            target = band_high

        # apply user override (allowed even if it violates floor/order; we just flag)
        ov = overrides_df[
            (overrides_df["오퍼타입"] == item_type) &
            (overrides_df["오퍼ID"] == item_id) &
            (overrides_df["가격영역"] == z)
        ]
        override_price = np.nan
        if not ov.empty:
            override_price = safe_float(ov.iloc[0]["가격_오버라이드"], np.nan)

        effective = override_price if (override_price == override_price and override_price > 0) else target

        # rounding
        band_low_r = krw_round(band_low, rounding_unit)
        band_high_r = krw_round(band_high, rounding_unit)
        floor_r = krw_round(floor, rounding_unit)
        target_r = krw_round(target, rounding_unit)
        eff_r = krw_round(effective, rounding_unit) if (effective == effective and effective > 0) else np.nan

        cm, cmr = contrib_metrics(
            price=eff_r if eff_r == eff_r else 0,
            cost_total=cost_total,
            q_orders=1,
            fee=p["수수료율"],
            pg=p["PG"],
            mkt=p["마케팅비"],
            ship_per_order=p["배송비(주문당)"],
            ret_rate=p["반품률"],
            ret_cost_order=p["반품비(주문당)"]
        )

        flags = []
        if eff_r == eff_r and eff_r < floor_r:
            flags.append("⚠️Floor 미만(손익 위험)")
        if eff_r == eff_r and eff_r < band_low_r:
            flags.append("⚠️BandLow 미만")
        if eff_r == eff_r and eff_r > band_high_r and z != "MSRP":
            flags.append("⚠️BandHigh 초과")

        rows.append({
            "가격영역": z,
            "비용채널": ch,
            "BandLow": band_low_r,
            "BandHigh": band_high_r,
            "Floor(손익하한)": floor_r,
            "추천가(Target)": target_r,
            "가격_오버라이드(원)": (krw_round(override_price, rounding_unit) if override_price == override_price else np.nan),
            "최종가격(원)": eff_r,
            "상태": status,
            "경고": " / ".join(flags),
            "마진룸(원)=최종-Floor": (eff_r - floor_r) if (eff_r == eff_r) else np.nan,
            "기여이익(원)": int(round(cm)) if cm == cm else np.nan,
            "기여이익률(%)": round(cmr*100, 1) if cmr == cmr else np.nan,
        })
    return pd.DataFrame(rows)

def check_order_violations(zdf: pd.DataFrame, gap_pct: float, gap_won: float):
    """가격영역 순서(저→고)에서 갭부족/역전 경고만 계산 (자동보정 없음)"""
    if zdf.empty:
        return pd.DataFrame()
    order = {z:i for i,z in enumerate(PRICE_ZONES)}
    g = zdf[["가격영역","최종가격(원)"]].copy()
    g["ord"] = g["가격영역"].map(order)
    g = g.sort_values("ord")
    viol = []
    prev_p = None
    prev_z = None
    for _, r in g.iterrows():
        p = safe_float(r["최종가격(원)"], np.nan)
        if p != p:
            continue
        if prev_p is None:
            prev_p = p
            prev_z = r["가격영역"]
            continue
        need = max(prev_p*(1+gap_pct), prev_p + gap_won)
        if p < need:
            viol.append({
                "하위영역": prev_z,
                "하위가격": prev_p,
                "상위영역": r["가격영역"],
                "상위가격": p,
                "필요최소상위가": need,
                "갭부족": need - p
            })
        prev_p = p
        prev_z = r["가격영역"]
    return pd.DataFrame(viol)

# -----------------------------
# Set helpers
# -----------------------------
def compute_set_cost(set_id: str, bom_df: pd.DataFrame, products_df: pd.DataFrame) -> float:
    b = bom_df[bom_df["세트ID"] == set_id].copy()
    if b.empty:
        return np.nan
    b["수량"] = pd.to_numeric(b["수량"], errors="coerce").fillna(0).astype(int)
    b = b.merge(products_df[["품번","원가"]], on="품번", how="left")
    b["원가"] = pd.to_numeric(b["원가"], errors="coerce").fillna(0.0)
    return float((b["원가"] * b["수량"]).sum())

def allocate_set_price_to_components(set_id, zone, set_price, products_df, bom_df, sku_always_prices):
    b = bom_df[bom_df["세트ID"] == set_id].copy()
    if b.empty:
        return pd.DataFrame()
    b["수량"] = pd.to_numeric(b["수량"], errors="coerce").fillna(0).astype(int)
    b = b.merge(products_df[["품번","상품명"]], on="품번", how="left")
    b["상시가_ref"] = b["품번"].astype(str).map(sku_always_prices).astype(float).fillna(0.0)
    b["ref_value"] = b["상시가_ref"] * b["수량"]
    total_ref = float(b["ref_value"].sum())
    b["w"] = (b["ref_value"] / total_ref) if total_ref > 0 else (1.0 / len(b))
    b["배분매출"] = set_price * b["w"]
    b["실질단가"] = b["배분매출"] / b["수량"].replace(0, np.nan)
    b["상시대비할인율(%)"] = np.where(b["상시가_ref"]>0, (1.0 - (b["실질단가"] / b["상시가_ref"])) * 100.0, np.nan)
    b["세트ID"] = set_id
    b["가격영역"] = zone
    return b[["세트ID","가격영역","품번","상품명","수량","상시가_ref","실질단가","상시대비할인율(%)"]]

# -----------------------------
# Session state init
# -----------------------------
if "products_df" not in st.session_state:
    st.session_state["products_df"] = pd.DataFrame(columns=["품번","상품명","브랜드","원가","MSRP_오버라이드","Min_오버라이드","Max_오버라이드","운영여부"])
if "channels_df" not in st.session_state:
    st.session_state["channels_df"] = pd.DataFrame(DEFAULT_CHANNELS, columns=["채널명","수수료율","PG","배송비(주문당)","마케팅비","반품률","반품비(주문당)"])
if "zone_map" not in st.session_state:
    st.session_state["zone_map"] = DEFAULT_ZONE_MAP.copy()
if "boundaries" not in st.session_state:
    st.session_state["boundaries"] = DEFAULT_BOUNDARIES.copy()
if "target_pos" not in st.session_state:
    st.session_state["target_pos"] = default_zone_target_pos(st.session_state["boundaries"])
if "sets_df" not in st.session_state:
    st.session_state["sets_df"] = pd.DataFrame(columns=["세트ID","세트명","MSRP_오버라이드"])
if "bom_df" not in st.session_state:
    st.session_state["bom_df"] = pd.DataFrame(columns=["세트ID","품번","수량"])
if "plan_df" not in st.session_state:
    st.session_state["plan_df"] = pd.DataFrame(columns=["가격영역","오퍼타입","오퍼ID","가격_오버라이드"])
if "overrides_df" not in st.session_state:
    st.session_state["overrides_df"] = pd.DataFrame(columns=["오퍼타입","오퍼ID","가격영역","가격_오버라이드"])

# -----------------------------
# UI
# -----------------------------
st.title("IBR 가격 시뮬레이터 v6.1")
st.caption("원가만 업로드 → 자동 가격 생성 → Min/Max/채널별 가격은 사용자가 수정(룰 위반 가능) → 마진/카니발은 경고로 표시")

tab_up, tab_sku, tab_set, tab_plan, tab_logic = st.tabs(
    ["1) 업로드/설정", "2) 단품(자동→수정)", "3) 세트(BOM)", "4) 운영플랜/카니발", "5) 로직(문장)"]
)

# =============================
# 1) Upload & Settings
# =============================
with tab_up:
    st.subheader("A. 원가/상품명 통일 파일 업로드(필수)")
    up = st.file_uploader("원가/상품마스터 업로드(.xlsx)", type=["xlsx","xls"])
    if up is not None:
        try:
            new_df = load_products_from_cost_master(up)
            st.session_state["products_df"] = new_df
            st.success(f"업로드 완료: {len(new_df):,}개 SKU")
        except Exception as e:
            st.error(f"업로드 오류: {e}")

    st.metric("현재 SKU 수", f"{len(st.session_state['products_df']):,}")

    st.divider()
    st.subheader("B. 채널 비용(키인 즉시 반영)")
    st.session_state["channels_df"] = st.data_editor(
        st.session_state["channels_df"],
        use_container_width=True,
        num_rows="dynamic",
        height=260
    )

    st.divider()
    st.subheader("C. 가격영역(밴드) ↔ 비용채널 매핑")
    zone_map = st.session_state["zone_map"].copy()
    channel_names = st.session_state["channels_df"]["채널명"].dropna().astype(str).tolist()
    cols = st.columns(5)
    for i, z in enumerate(PRICE_ZONES):
        with cols[i % 5]:
            zone_map[z] = st.selectbox(
                f"{z}",
                options=channel_names,
                index=channel_names.index(zone_map.get(z, channel_names[0])) if zone_map.get(z) in channel_names else 0,
                key=f"zmap_{z}"
            )
    st.session_state["zone_map"] = zone_map

    st.divider()
    st.subheader("D. 밴드 경계(마우스로 조정)")
    b = st.session_state["boundaries"].copy()
    prev = 0
    new_b = [0]
    for idx in range(1, 10):
        left_zone = PRICE_ZONES[idx-1]
        right_zone = PRICE_ZONES[idx]
        minv = prev + 1
        maxv = 100 - (10-idx)
        val = int(b[idx])
        val = max(minv, min(maxv, val))
        val = st.slider(f"경계 {idx}: {left_zone} | {right_zone} (%)", min_value=minv, max_value=maxv, value=val, step=1, key=f"b_{idx}")
        new_b.append(val)
        prev = val
    new_b.append(100)
    st.session_state["boundaries"] = new_b

    with st.expander("각 영역 내 Target 위치(%) (기본=중앙)", expanded=False):
        tp = st.session_state["target_pos"].copy()
        cols = st.columns(5)
        for i, z in enumerate(PRICE_ZONES):
            s = new_b[i]; e = new_b[i+1]
            mid = int(round((s+e)/2))
            with cols[i%5]:
                tp[z] = st.slider(f"{z}", min_value=int(s), max_value=int(e), value=int(tp.get(z, mid)), step=1, key=f"tp_{z}")
        st.session_state["target_pos"] = tp

    st.divider()
    st.subheader("E. (선택) SKU 오버라이드 저장용 테이블")
    st.caption("기본은 '원가만'으로 자동 산출. 다만 SKU별로 MSRP/Min/Max를 고정하고 싶으면 여기서 입력.")
    if st.session_state["products_df"].empty:
        st.info("먼저 업로드하세요.")
    else:
        st.session_state["products_df"] = st.data_editor(st.session_state["products_df"], use_container_width=True, height=300, num_rows="dynamic")

# =============================
# 2) SKU Auto → Adjust
# =============================
with tab_sku:
    st.subheader("단품: 원가 기반 자동 산출 → Min/Max/채널가격 수정")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("업로드/설정 탭에서 원가 파일을 업로드하세요.")
    else:
        p1, p2, p3, p4 = st.columns([1,1,1,1])
        with p1:
            rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2)
        with p2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한(예:0.30)", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f")
        with p3:
            min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1) / 100.0
        with p4:
            gap_pct = st.slider("서열 갭(%)", 0, 20, 5, 1) / 100.0
            gap_won = st.number_input("서열 갭(원)", min_value=0, value=2000, step=500)

        include_zones = PRICE_ZONES

        options = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
        picked = st.selectbox("SKU 선택", options, index=0)
        sku = picked.split(" | ",1)[0].strip()
        row = prod[prod["품번"].astype(str)==sku].iloc[0]
        cost = safe_float(row["원가"], np.nan)

        if cost != cost or cost <= 0:
            st.error("원가가 비어있거나 0 이하입니다. 업로드 파일의 원가(vat-)를 확인하세요.")
        else:
            min_zone = st.selectbox("자동 Min 기준 존", PRICE_ZONES, index=0)
            min_auto, max_auto, meta = compute_auto_range_from_cost(
                cost_total=cost,
                channels_df=st.session_state["channels_df"],
                zone_map=st.session_state["zone_map"],
                boundaries=st.session_state["boundaries"],
                rounding_unit=rounding_unit,
                min_cm=min_cm,
                max_cost_ratio=max_cost_ratio,
                include_zones=include_zones,
                min_zone=min_zone,
                msrp_override=safe_float(row.get("MSRP_오버라이드", np.nan), np.nan),
            )

            st.markdown(f"**SKU:** `{sku}` — {row.get('상품명','')}")
            if meta.get("note"):
                st.info(meta["note"])

            # User adjustments (THIS is what you asked for)
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                min_user = st.number_input("Min(최저가) 수정", min_value=0, value=int(min_auto), step=rounding_unit)
            with c2:
                max_user = st.number_input("Max(최고가/MSRP) 수정", min_value=0, value=int(max_auto), step=rounding_unit)
            with c3:
                st.caption("Min/Max를 바꾸면 밴드 전체가 같이 이동합니다. (원가만으로 산출된 기본값을 수동 조정)")

            if max_user <= min_user:
                st.warning("Max가 Min 이하입니다. 밴드가 펼쳐지지 않으므로 Max를 올려주세요.")
                max_user = min_user + max(rounding_unit*10, int(min_user*0.15))

            st.write(f"- 원가: **{int(cost):,}원** | 레인지(적용): **{int(min_user):,} ~ {int(max_user):,}원**")

            zdf = build_zone_table(
                cost_total=cost,
                min_price=float(min_user),
                max_price=float(max_user),
                channels_df=st.session_state["channels_df"],
                zone_map=st.session_state["zone_map"],
                boundaries=st.session_state["boundaries"],
                target_pos=st.session_state["target_pos"],
                rounding_unit=rounding_unit,
                min_cm=min_cm,
                overrides_df=st.session_state["overrides_df"],
                item_type="SKU",
                item_id=sku,
            )

            # Editable overrides per zone (user can break rules)
            st.markdown("### 채널별 가격 수정(오버라이드) — 룰 위반 허용, 경고만 표시")
            if zdf.empty:
                st.info("결과가 없습니다.")
            else:
                edit_cols = ["가격영역","가격_오버라이드(원)"]
                edit_view = zdf[edit_cols].copy()
                edit_view = st.data_editor(edit_view, use_container_width=True, height=260, num_rows="fixed", key="sku_override_editor")

                # write back overrides
                ov = st.session_state["overrides_df"].copy()
                # remove existing overrides for this sku
                ov = ov[~((ov["오퍼타입"]=="SKU") & (ov["오퍼ID"]==sku))].copy()
                # add new ones
                for _, rr in edit_view.iterrows():
                    p = safe_float(rr.get("가격_오버라이드(원)"), np.nan)
                    if p == p and p > 0:
                        ov = pd.concat([ov, pd.DataFrame([{
                            "오퍼타입":"SKU",
                            "오퍼ID":sku,
                            "가격영역":rr["가격영역"],
                            "가격_오버라이드":float(p)
                        }])], ignore_index=True)
                st.session_state["overrides_df"] = ov

                # rebuild to reflect overrides
                zdf = build_zone_table(
                    cost_total=cost,
                    min_price=float(min_user),
                    max_price=float(max_user),
                    channels_df=st.session_state["channels_df"],
                    zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"],
                    target_pos=st.session_state["target_pos"],
                    rounding_unit=rounding_unit,
                    min_cm=min_cm,
                    overrides_df=st.session_state["overrides_df"],
                    item_type="SKU",
                    item_id=sku,
                )

                st.dataframe(zdf, use_container_width=True, height=360)

                # order violations
                viol = check_order_violations(zdf, gap_pct=gap_pct, gap_won=gap_won)
                if not viol.empty:
                    st.warning("서열/갭 위반(카니발 위험 가능성) — 허용은 되지만 참고용 경고입니다.")
                    st.dataframe(viol, use_container_width=True, height=220)
                else:
                    st.success("서열/갭 위반 없음(현재 최종가격 기준).")

                xb = to_excel_bytes({"sku_result": zdf, "order_violations": viol})
                st.download_button("이 SKU 결과 엑셀 다운로드", xb, file_name=f"{sku}_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================
# 3) Set (BOM)
# =============================
with tab_set:
    st.subheader("세트(BOM): 앱에서 구성하면 자동 추천가 + 구성품 배분단가/할인율")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("원가 파일을 먼저 업로드하세요.")
    else:
        c1, c2, c3 = st.columns([1,2,1])
        with c1:
            new_id = st.text_input("세트ID", value="")
        with c2:
            new_name = st.text_input("세트명", value="")
        with c3:
            if st.button("세트 추가", type="primary", disabled=(not new_id.strip() or not new_name.strip())):
                sets = st.session_state["sets_df"].copy()
                if (sets["세트ID"] == new_id.strip()).any():
                    st.warning("이미 존재하는 세트ID")
                else:
                    sets = pd.concat([sets, pd.DataFrame([{"세트ID":new_id.strip(),"세트명":new_name.strip(),"MSRP_오버라이드":np.nan}])], ignore_index=True)
                    st.session_state["sets_df"] = sets
                    st.success("세트 추가 완료")

        if st.session_state["sets_df"].empty:
            st.info("세트를 먼저 추가하세요.")
        else:
            st.session_state["sets_df"] = st.data_editor(st.session_state["sets_df"], use_container_width=True, height=160, num_rows="dynamic")

            set_opts = (st.session_state["sets_df"]["세트ID"].astype(str) + " | " + st.session_state["sets_df"]["세트명"].astype(str)).tolist()
            picked = st.selectbox("편집할 세트 선택", set_opts, index=0)
            set_id = picked.split(" | ",1)[0].strip()

            # BOM add
            st.markdown("### BOM(구성품) 추가")
            sku_opts = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
            a1,a2,a3 = st.columns([3,1,1])
            with a1:
                sku_pick = st.selectbox("구성품 SKU", sku_opts, index=0, key=f"bom_sku_{set_id}")
                sku = sku_pick.split(" | ",1)[0].strip()
            with a2:
                qty = st.number_input("수량", min_value=1, value=1, step=1, key=f"bom_qty_{set_id}")
            with a3:
                if st.button("추가", key=f"bom_add_{set_id}"):
                    bom = st.session_state["bom_df"].copy()
                    bom = pd.concat([bom, pd.DataFrame([{"세트ID":set_id,"품번":sku,"수량":int(qty)}])], ignore_index=True)
                    st.session_state["bom_df"] = bom
                    st.success("추가 완료")

            bom_view = st.session_state["bom_df"][st.session_state["bom_df"]["세트ID"]==set_id].copy()
            if bom_view.empty:
                st.info("BOM이 비어있습니다.")
            else:
                bom_view = bom_view.merge(prod[["품번","상품명","원가"]], on="품번", how="left")
                st.dataframe(bom_view, use_container_width=True, height=220)

            if st.button("이 세트 BOM 전체 삭제", type="secondary", key=f"bom_clear_{set_id}"):
                bom = st.session_state["bom_df"].copy()
                bom = bom[bom["세트ID"]!=set_id].copy()
                st.session_state["bom_df"] = bom
                st.success("삭제 완료")

            st.divider()
            st.markdown("### 세트 추천가(자동) → Min/Max/채널가격 수정")
            p1, p2, p3, p4 = st.columns([1,1,1,1])
            with p1:
                rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="set_round")
            with p2:
                max_cost_ratio = st.number_input("MSRP 원가율 상한", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="set_ratio")
            with p3:
                min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="set_cm") / 100.0
            with p4:
                gap_pct = st.slider("서열 갭(%)", 0, 20, 5, 1, key="set_gap") / 100.0
                gap_won = st.number_input("서열 갭(원)", min_value=0, value=2000, step=500, key="set_gap_won")

            if bom_view.empty:
                st.info("BOM을 구성하면 자동 가격이 나옵니다.")
            else:
                cost_total = compute_set_cost(set_id, st.session_state["bom_df"], prod)
                srow = st.session_state["sets_df"][st.session_state["sets_df"]["세트ID"]==set_id].iloc[0]

                min_auto, max_auto, meta = compute_auto_range_from_cost(
                    cost_total=cost_total,
                    channels_df=st.session_state["channels_df"],
                    zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"],
                    rounding_unit=rounding_unit,
                    min_cm=min_cm,
                    max_cost_ratio=max_cost_ratio,
                    include_zones=PRICE_ZONES,
                    min_zone="공구",
                    msrp_override=safe_float(srow.get("MSRP_오버라이드", np.nan), np.nan),
                )

                st.write(f"- 세트 원가합: **{int(cost_total):,}원** | 자동 레인지: **{int(min_auto):,} ~ {int(max_auto):,}원**")
                if meta.get("note"):
                    st.info(meta["note"])

                c1,c2,c3 = st.columns([1,1,2])
                with c1:
                    min_user = st.number_input("Min 수정(세트)", min_value=0, value=int(min_auto), step=rounding_unit, key="set_min_user")
                with c2:
                    max_user = st.number_input("Max 수정(세트)", min_value=0, value=int(max_auto), step=rounding_unit, key="set_max_user")
                with c3:
                    st.caption("Min/Max를 바꾸면 밴드 전체가 같이 이동합니다.")

                if max_user <= min_user:
                    max_user = min_user + max(rounding_unit*10, int(min_user*0.15))

                zdf = build_zone_table(
                    cost_total=cost_total,
                    min_price=float(min_user),
                    max_price=float(max_user),
                    channels_df=st.session_state["channels_df"],
                    zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"],
                    target_pos=st.session_state["target_pos"],
                    rounding_unit=rounding_unit,
                    min_cm=min_cm,
                    overrides_df=st.session_state["overrides_df"],
                    item_type="SET",
                    item_id=set_id,
                )

                st.markdown("#### 채널별 가격 오버라이드(세트)")
                edit_view = zdf[["가격영역","가격_오버라이드(원)"]].copy()
                edit_view = st.data_editor(edit_view, use_container_width=True, height=260, num_rows="fixed", key="set_override_editor")

                ov = st.session_state["overrides_df"].copy()
                ov = ov[~((ov["오퍼타입"]=="SET") & (ov["오퍼ID"]==set_id))].copy()
                for _, rr in edit_view.iterrows():
                    p = safe_float(rr.get("가격_오버라이드(원)"), np.nan)
                    if p == p and p > 0:
                        ov = pd.concat([ov, pd.DataFrame([{"오퍼타입":"SET","오퍼ID":set_id,"가격영역":rr["가격영역"],"가격_오버라이드":float(p)}])], ignore_index=True)
                st.session_state["overrides_df"] = ov

                zdf = build_zone_table(
                    cost_total=cost_total,
                    min_price=float(min_user),
                    max_price=float(max_user),
                    channels_df=st.session_state["channels_df"],
                    zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"],
                    target_pos=st.session_state["target_pos"],
                    rounding_unit=rounding_unit,
                    min_cm=min_cm,
                    overrides_df=st.session_state["overrides_df"],
                    item_type="SET",
                    item_id=set_id,
                )

                st.dataframe(zdf, use_container_width=True, height=340)

                viol = check_order_violations(zdf, gap_pct=gap_pct, gap_won=gap_won)
                if not viol.empty:
                    st.warning("세트 가격영역 서열/갭 위반(참고용)")
                    st.dataframe(viol, use_container_width=True, height=200)

                # Allocation (component effective unit price) for each zone
                # Build SKU '상시' effective prices for weighting
                sku_always = {}
                for sku in st.session_state["bom_df"][st.session_state["bom_df"]["세트ID"]==set_id]["품번"].astype(str).unique():
                    r = prod[prod["품번"].astype(str)==sku]
                    if r.empty:
                        continue
                    rcost = safe_float(r.iloc[0]["원가"], np.nan)
                    if rcost != rcost or rcost <= 0:
                        continue
                    min_s, max_s, _ = compute_auto_range_from_cost(
                        cost_total=rcost,
                        channels_df=st.session_state["channels_df"],
                        zone_map=st.session_state["zone_map"],
                        boundaries=st.session_state["boundaries"],
                        rounding_unit=rounding_unit,
                        min_cm=min_cm,
                        max_cost_ratio=max_cost_ratio,
                        include_zones=PRICE_ZONES,
                        min_zone="공구",
                        msrp_override=safe_float(r.iloc[0].get("MSRP_오버라이드", np.nan), np.nan),
                    )
                    z_sku = build_zone_table(
                        cost_total=rcost,
                        min_price=float(min_s),
                        max_price=float(max_s),
                        channels_df=st.session_state["channels_df"],
                        zone_map=st.session_state["zone_map"],
                        boundaries=st.session_state["boundaries"],
                        target_pos=st.session_state["target_pos"],
                        rounding_unit=rounding_unit,
                        min_cm=min_cm,
                        overrides_df=st.session_state["overrides_df"],
                        item_type="SKU",
                        item_id=str(sku),
                    )
                    ar = z_sku[z_sku["가격영역"]=="상시"]
                    if not ar.empty:
                        sku_always[str(sku)] = float(ar.iloc[0]["최종가격(원)"])

                st.markdown("#### 구성품 배분단가/할인율(상시가 가중치)")
                alloc_rows = []
                for _, rr in zdf.iterrows():
                    alloc = allocate_set_price_to_components(set_id, rr["가격영역"], rr["최종가격(원)"], prod, st.session_state["bom_df"], sku_always)
                    if not alloc.empty:
                        alloc_rows.append(alloc)
                if alloc_rows:
                    alloc_df = pd.concat(alloc_rows, ignore_index=True)
                    st.dataframe(alloc_df, use_container_width=True, height=320)
                    xb = to_excel_bytes({"set_result": zdf, "alloc": alloc_df, "order_viol": viol})
                    st.download_button("세트 결과 엑셀 다운로드", xb, file_name=f"{set_id}_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.info("구성품 상시가(가중치) 산출이 부족해 배분 계산이 제한됩니다(구성품 원가 필요).")

# =============================
# 4) Plan / Cannibal (simple)
# =============================
with tab_plan:
    st.subheader("운영플랜(채널별 오퍼 배치) → 카니발(역전/갭부족) 체크")
    st.caption("이 탭은 간단 버전입니다. (SKU/SET을 채널에 배치 → SKU별 최저 실질단가로 역전/갭부족 탐지)")

    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("원가 파일을 먼저 업로드하세요.")
    else:
        plan = st.session_state["plan_df"].copy()
        c1,c2,c3,c4 = st.columns([1,1,2,1])
        with c1:
            zone = st.selectbox("가격영역", PRICE_ZONES, index=0, key="plan_zone")
        with c2:
            otype = st.selectbox("오퍼타입", ["SKU","SET"], index=0, key="plan_otype")
        with c3:
            if otype=="SKU":
                opts = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
                pick = st.selectbox("오퍼 선택", opts, index=0, key="plan_pick_sku")
                oid = pick.split(" | ",1)[0].strip()
            else:
                sets_df = st.session_state["sets_df"].copy()
                if sets_df.empty:
                    st.warning("세트가 없습니다(세트 탭에서 생성).")
                    oid = ""
                else:
                    opts = (sets_df["세트ID"].astype(str) + " | " + sets_df["세트명"].astype(str)).tolist()
                    pick = st.selectbox("오퍼 선택", opts, index=0, key="plan_pick_set")
                    oid = pick.split(" | ",1)[0].strip()
        with c4:
            p_override = st.number_input("가격 오버라이드(0=미사용)", min_value=0, value=0, step=1000, key="plan_price")
        if st.button("플랜 추가", type="primary", key="plan_add", disabled=(otype=="SET" and oid=="")):
            plan = pd.concat([plan, pd.DataFrame([{"가격영역":zone,"오퍼타입":otype,"오퍼ID":oid,"가격_오버라이드": (float(p_override) if p_override>0 else np.nan)}])], ignore_index=True)
            st.session_state["plan_df"] = plan

        st.divider()
        st.markdown("### 현재 플랜")
        if st.session_state["plan_df"].empty:
            st.info("플랜이 비어 있습니다.")
        else:
            st.session_state["plan_df"] = st.data_editor(st.session_state["plan_df"], use_container_width=True, height=220, num_rows="dynamic")
            plan = st.session_state["plan_df"].copy()

        st.divider()
        st.markdown("### 카니발 체크")
        p1,p2,p3 = st.columns([1,1,1])
        with p1:
            rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="plan_round")
        with p2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="plan_ratio")
        with p3:
            min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="plan_cm") / 100.0
        gap_pct = st.slider("최소갭(%)", 0, 20, 5, 1, key="plan_gap_pct") / 100.0
        gap_won = st.number_input("최소갭(원)", min_value=0, value=2000, step=500, key="plan_gap_won")

        if st.button("카니발 체크 실행", type="primary", key="plan_run"):
            if plan.empty:
                st.error("플랜이 비어 있습니다.")
            else:
                # Build effective SKU price per zone from overrides if set; else recompute quickly with auto range
                eff_rows = []
                sets_df = st.session_state["sets_df"].copy()
                bom_df = st.session_state["bom_df"].copy()
                overrides_df = st.session_state["overrides_df"].copy()

                # helper: compute sku final price in zone using auto range and overrides
                def sku_zone_price(sku, zone):
                    r = prod[prod["품번"].astype(str)==sku].iloc[0]
                    cost = safe_float(r["원가"], np.nan)
                    if cost != cost or cost <= 0:
                        return np.nan
                    min_auto, max_auto, _ = compute_auto_range_from_cost(cost, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                                                                         rounding_unit, min_cm, max_cost_ratio, PRICE_ZONES, "공구",
                                                                         safe_float(r.get("MSRP_오버라이드", np.nan), np.nan))
                    zdf = build_zone_table(cost, min_auto, max_auto, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                                           st.session_state["target_pos"], rounding_unit, min_cm, overrides_df, "SKU", sku)
                    rr = zdf[zdf["가격영역"]==zone]
                    return float(rr.iloc[0]["최종가격(원)"]) if not rr.empty else np.nan

                # sku '상시' for set allocation weights
                sku_always = {}
                for sku in prod["품번"].astype(str).tolist():
                    p = sku_zone_price(sku, "상시")
                    if p == p:
                        sku_always[sku] = p

                for _, pr in plan.iterrows():
                    zone = pr["가격영역"]
                    otype = pr["오퍼타입"]
                    oid = str(pr["오퍼ID"])
                    p_override = safe_float(pr.get("가격_오버라이드", np.nan), np.nan)

                    if otype == "SKU":
                        p = sku_zone_price(oid, zone)
                        if p_override == p_override and p_override > 0:
                            p = p_override
                        if p == p:
                            eff_rows.append({"가격영역":zone, "품번":oid, "실질단가":p, "오퍼":f"SKU:{oid}"})
                    else:
                        # set: allocate to components
                        cost_total = compute_set_cost(oid, bom_df, prod)
                        if cost_total != cost_total:
                            continue
                        srow = sets_df[sets_df["세트ID"]==oid]
                        msrp_ov = safe_float(srow.iloc[0].get("MSRP_오버라이드", np.nan), np.nan) if not srow.empty else np.nan
                        min_auto, max_auto, _ = compute_auto_range_from_cost(cost_total, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                                                                             rounding_unit, min_cm, max_cost_ratio, PRICE_ZONES, "공구", msrp_ov)
                        z_set = build_zone_table(cost_total, min_auto, max_auto, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                                                 st.session_state["target_pos"], rounding_unit, min_cm, overrides_df, "SET", oid)
                        rr = z_set[z_set["가격영역"]==zone]
                        if rr.empty:
                            continue
                        set_price = float(rr.iloc[0]["최종가격(원)"])
                        if p_override == p_override and p_override > 0:
                            set_price = p_override
                        alloc = allocate_set_price_to_components(oid, zone, set_price, prod, bom_df, sku_always)
                        for _, ar in alloc.iterrows():
                            eff_rows.append({"가격영역":zone, "품번":str(ar["품번"]), "실질단가":float(ar["실질단가"]), "오퍼":f"SET:{oid}"})

                eff = pd.DataFrame(eff_rows)
                if eff.empty:
                    st.error("실질단가 계산 결과가 없습니다.")
                else:
                    min_eff = eff.groupby(["가격영역","품번"], as_index=False)["실질단가"].min()
                    min_eff = min_eff.merge(prod[["품번","상품명"]], on="품번", how="left")
                    st.markdown("#### 채널별 SKU 최저 실질단가")
                    st.dataframe(min_eff.sort_values(["품번","가격영역"]), use_container_width=True, height=320)

                    # check across zones per sku
                    order = {z:i for i,z in enumerate(PRICE_ZONES)}
                    viol = []
                    for sku, g in min_eff.groupby("품번"):
                        g2 = g.copy()
                        g2["ord"] = g2["가격영역"].map(order)
                        g2 = g2.sort_values("ord")
                        prev = None
                        prev_zone = None
                        for _, rr in g2.iterrows():
                            cur = rr["실질단가"]
                            if prev is None:
                                prev = cur; prev_zone = rr["가격영역"]; continue
                            need = max(prev*(1+gap_pct), prev+gap_won)
                            if cur < need:
                                viol.append({"품번":sku,"상품명":rr.get("상품명",""),"하위영역":prev_zone,"하위가격":prev,
                                             "상위영역":rr["가격영역"],"상위가격":cur,"필요최소상위가":need,"갭부족":need-cur})
                            prev = cur; prev_zone = rr["가격영역"]
                    viol_df = pd.DataFrame(viol)
                    st.divider()
                    st.markdown("#### 카니발(갭부족/역전) 경고")
                    if viol_df.empty:
                        st.success("카니발 경고 없음(현재 플랜 기준).")
                    else:
                        st.warning(f"{len(viol_df):,}건")
                        st.dataframe(viol_df, use_container_width=True, height=260)

                    xb = to_excel_bytes({"min_eff": min_eff, "violations": viol_df})
                    st.download_button("카니발 결과 엑셀 다운로드", xb, file_name="cannibal_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================
# 5) Logic (plain language)
# =============================
with tab_logic:
    st.subheader("로직(문장) — '원가만'으로 자동 가격이 나오는 방식")
    st.markdown(
        """
### 1) 입력은 원가만(필수)
- 업로드 파일(상품코드=품번, 상품명, 원가(vat-))만 있으면 자동 가격이 산출됩니다.
- 마케팅비/반품률은 **0으로 둔 상태에서도** 돌아가며, 나중에 키인하면 즉시 재계산됩니다.

### 2) 손익하한(Floor)
각 가격영역은 비용채널(수수료/PG/배송/마케팅/반품)을 갖습니다.
- Floor = (원가 + 배송비/주문 + 반품률×반품비/주문) / (1 - 수수료 - PG - 마케팅비 - 최소기여이익률)

### 3) 자동 레인지(Min~Max=MSRP)
- Min(자동) = 선택한 최저 영역(기본 공구)의 Floor
- MSRP_base = 원가 / 원가율상한(예: 30%)
- 그리고 중요한 자동 보정:
  - 각 가격영역 z의 Floor가 그 영역의 BandHigh 안에 들어오도록 Max를 자동 상향합니다.
  - 그래서 '원가만'으로도 밴드가 0폭으로 붙지 않게 설계합니다.
- 사용자가 MSRP_오버라이드를 넣으면 그 값이 최우선 Max가 됩니다.

### 4) 밴드(가격영역)
- Min~Max 레인지(0~100%)를 10개 가격영역으로 분할합니다.
- 경계는 슬라이더(마우스로 드래그)로 조정합니다.
- 각 영역의 추천가(Target)는 영역 내부 위치(기본 중앙)에서 찍되,
  - Target = max(영역내 위치값, Floor)
  - Floor가 BandHigh를 넘으면 '불가'로 표시합니다.

### 5) 사용자는 마음대로 조정 가능(요구사항 반영)
- Min을 올리면 모든 가격이 함께 올라갑니다(레인지 기반).
- Max를 올리면 모든 가격이 함께 올라갑니다.
- 특정 채널 가격은 오버라이드로 직접 입력할 수 있고, 서열/갭/Floor 위반은 경고만 표시합니다.

### 6) 세트
- 앱에서 세트ID/세트명 생성 후 BOM(구성품 리스트업)을 추가하면
  - 세트원가 = Σ(구성품원가×수량)
  - 동일하게 자동 레인지/밴드/추천가를 산출합니다.
- 세트가격을 구성품별로 배분할 때는 구성품의 상시가(ref)를 가중치로 사용하여 실질단가/할인율을 계산합니다.

### 7) 운영플랜/카니발
- 채널별로 SKU/세트를 배치하면 SKU 기준 '최저 실질단가'를 만들고
- 가격영역 서열(저→고)에서 갭부족/역전을 경고로 표시합니다.
"""
    )