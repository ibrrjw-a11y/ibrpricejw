import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v4 (Band-first, Set/BOM + Cannibalization)
# 목표:
# 1) '최저가 ~ 최고가' 하나의 레인지 안에 채널별 가격영역(밴드)을 고정
# 2) 최저/최고는 SKU/세트별로 조정 가능
# 3) 채널별 수수료/PG/배송/마케팅/반품을 키인하면 마진/하한/추천가가 즉시 재계산
# 4) 세트(BOM)를 입력하면 채널별 세트가 추천 + 마진 + 구성품별 실질 단가(상시가 비율 배분)
# 5) 운영 플랜(채널별로 어떤 세트/단품을 팔지) 입력 → 카니발(역전) 자동 체크
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator v4", layout="wide")

# ----------------------------
# Constants
# ----------------------------
PRICE_ORDER_LOW_TO_HIGH = [
    "공구",
    "홈쇼핑",
    "폐쇄몰",
    "모바일라방",
    "원데이",
    "브랜드위크",
    "홈사",
    "상시",
    "오프라인",
    "MSRP",
]

ONLINE_PLATFORMS = ["자사몰", "스마트스토어", "쿠팡", "오픈마켓"]

DEFAULT_FEES = {
    # user provided
    "오프라인": 0.50,
    "자사몰": 0.05,
    "스마트스토어": 0.05,
    "쿠팡": 0.25,
    "오픈마켓": 0.15,
    "홈사": 0.30,
    "공구": 0.50,
    "홈쇼핑": 0.55,
    "모바일라방": 0.40,
    "폐쇄몰": 0.25,
}

DEFAULT_PG = 0.03
DEFAULT_SHIP_PER_ORDER = 3000

# Price-type → which channel costs to use when computing floor/margin
# (온라인 레벨은 플랫폼별 마진도 보여줄 수 있지만, floor는 대표 채널로 계산)
PRICE_TO_CHANNEL_DEFAULT = {
    "공구": "공구",
    "홈쇼핑": "홈쇼핑",
    "폐쇄몰": "폐쇄몰",
    "모바일라방": "모바일라방",
    "원데이": "자사몰",
    "브랜드위크": "자사몰",
    "홈사": "홈사",
    "상시": "자사몰",
    "오프라인": "오프라인",
    "MSRP": "자사몰",
}

# ----------------------------
# Styling (band bar)
# ----------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 14px !important; }
.block-container { padding-top: 1.0rem; }

.band-wrap { border-top: 1px solid rgba(128,128,128,0.35); padding-top: 10px; }
.band-row { display:flex; align-items:center; gap:12px; padding:8px 0; }
.band-label { width: 420px; font-size: 13px; opacity: 0.95; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.band-box { position: relative; flex: 1; height: 18px; border-radius: 999px; background: var(--secondary-background-color) !important; border: 1px solid rgba(128,128,128,0.45); }
.band-seg { position:absolute; height:100%; border-radius:999px; background: var(--primary-color); border: 1px solid rgba(128,128,128,0.18); }
.band-tick { position:absolute; top:-2px; width:1px; height:22px; background: rgba(128,128,128,0.55); }
.band-dot { position:absolute; top:-6px; width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:10px solid var(--text-color) !important; filter: drop-shadow(0 1px 1px rgba(0,0,0,0.25)); }
.band-nums { width: 360px; font-size: 12px; opacity: 0.82; text-align:right; white-space: nowrap; }
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


def krw_round(x, unit=100):
    try:
        return int(round(float(x) / unit) * unit)
    except Exception:
        return 0


def pct(x):
    return float(x) / 100.0


def normalize_band_widths(df):
    d = df.copy()
    d["width_pct"] = d["width_pct"].astype(float).clip(lower=0)
    total = d["width_pct"].sum()
    if total <= 0:
        # fallback
        d["width_pct"] = 100.0 / len(d)
        total = 100.0
    # normalize to 100
    d["width_pct"] = d["width_pct"] / total * 100.0

    # build start/end
    starts = []
    ends = []
    cur = 0.0
    for w in d["width_pct"].tolist():
        starts.append(cur)
        cur += w
        ends.append(cur)
    # force end=100
    ends[-1] = 100.0
    d["start_pct"] = starts
    d["end_pct"] = ends
    return d


def make_price_from_band(min_price, max_price, start_pct, end_pct, pos_pct, rounding_unit):
    min_p = float(min_price)
    max_p = float(max_price)
    span = max(0.0, max_p - min_p)
    lo = min_p + span * (float(start_pct) / 100.0)
    hi = min_p + span * (float(end_pct) / 100.0)
    pos = float(pos_pct) / 100.0
    tgt = lo + (hi - lo) * pos
    return (
        krw_round(lo, rounding_unit),
        krw_round(tgt, rounding_unit),
        krw_round(hi, rounding_unit),
    )


def econ_floor(order_cost, channel_row, min_margin_rate):
    """Return minimum feasible selling price for an order (set or single)."""
    fee = safe_float(channel_row.get("fee_rate", 0.0), 0.0)
    pg = safe_float(channel_row.get("pg_rate", 0.0), 0.0)
    mkt = safe_float(channel_row.get("mkt_rate", 0.0), 0.0)

    ship = safe_float(channel_row.get("ship_per_order", 0.0), 0.0)
    rrate = safe_float(channel_row.get("return_rate", 0.0), 0.0)
    rcost = safe_float(channel_row.get("return_cost_per_order", 0.0), 0.0)

    denom = 1.0 - (fee + pg + mkt + min_margin_rate)
    if denom <= 0:
        return float("inf")
    numer = float(order_cost) + ship + (rrate * rcost)
    return numer / denom


def contribution_margin(selling_price, order_cost, channel_row):
    """Return (cm_won, cm_rate) based on channel costs."""
    p = float(selling_price)
    if p <= 0:
        return (np.nan, np.nan)

    fee = safe_float(channel_row.get("fee_rate", 0.0), 0.0)
    pg = safe_float(channel_row.get("pg_rate", 0.0), 0.0)
    mkt = safe_float(channel_row.get("mkt_rate", 0.0), 0.0)

    ship = safe_float(channel_row.get("ship_per_order", 0.0), 0.0)
    rrate = safe_float(channel_row.get("return_rate", 0.0), 0.0)
    rcost = safe_float(channel_row.get("return_cost_per_order", 0.0), 0.0)

    net = p * (1.0 - fee - pg - mkt) - (float(order_cost) + ship + (rrate * rcost))
    return net, net / p


def infer_brand_from_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0]


def to_excel_bytes(sheets: dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()


def render_band_bar(item_label, min_price, max_price, band_df, markers=None):
    """Render one horizontal bar segmented by price bands.

    markers: dict label -> price
    """
    if markers is None:
        markers = {}

    min_p = float(min_price)
    max_p = float(max_price)
    span = max(1.0, max_p - min_p)

    # build segments HTML
    seg_html = ""
    for i, r in band_df.iterrows():
        left = float(r["start_pct"])
        width = float(r["end_pct"] - r["start_pct"])
        # vary opacity by index (no fixed colors)
        op = 0.18 + (i % 5) * 0.06
        seg_html += (
            f"<div class='band-seg' title='{r['price_type']}: {left:.1f}%~{r['end_pct']:.1f}%' "
            f"style='left:{left:.2f}%; width:{max(0.8, width):.2f}%; opacity:{op:.2f};'></div>"
        )

    # ticks at boundaries
    ticks_html = ""
    for _, r in band_df.iterrows():
        ticks_html += f"<div class='band-tick' style='left:{float(r['start_pct']):.2f}%;'></div>"
    ticks_html += f"<div class='band-tick' style='left:100%;'></div>"

    # markers
    dot_html = ""
    for k, v in markers.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        pct_pos = (float(v) - min_p) / span * 100.0
        pct_pos = min(100.0, max(0.0, pct_pos))
        dot_html += f"<div class='band-dot' title='{k}: {int(v):,}원' style='left: calc({pct_pos:.2f}% - 6px);'></div>"

    nums = f"{int(min_p):,}원  ~  {int(max_p):,}원"

    html = f"""
<div class="band-row">
  <div class="band-label" title="{item_label}">{item_label}</div>
  <div class="band-box">
    {seg_html}
    {ticks_html}
    {dot_html}
  </div>
  <div class="band-nums">{nums}</div>
</div>
"""
    st.markdown('<div class="band-wrap">', unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Session State init
# ----------------------------
if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame(
        columns=[
            "품번",
            "신규품명",
            "브랜드",
            "원가",
            "MSRP",
            "최저가(override)",
            "최고가(override)",
            "메모",
        ]
    )

if "sets_bom" not in st.session_state:
    st.session_state["sets_bom"] = pd.DataFrame(
        columns=["세트명", "품번", "수량"]
    )

if "plan_df" not in st.session_state:
    st.session_state["plan_df"] = pd.DataFrame(
        columns=["채널(가격영역)", "오퍼유형", "오퍼명", "판매가(override)"]
    )

# Channel params
if "channel_params" not in st.session_state:
    # channels include online platforms + special channels
    ch_rows = []
    # online platforms
    for ch in ONLINE_PLATFORMS:
        ch_rows.append(
            {
                "channel": ch,
                "fee_rate": DEFAULT_FEES.get(ch, 0.0),
                "pg_rate": DEFAULT_PG,
                "mkt_rate": 0.0,
                "ship_per_order": DEFAULT_SHIP_PER_ORDER,
                "return_rate": 0.0,
                "return_cost_per_order": 0.0,
            }
        )
    # special channels
    for ch in ["홈사", "폐쇄몰", "공구", "홈쇼핑", "모바일라방", "오프라인"]:
        if ch in ONLINE_PLATFORMS:
            continue
        # 홈쇼핑은 기본 배송비 0 가정
        ship = 0.0 if ch == "홈쇼핑" else DEFAULT_SHIP_PER_ORDER
        # 오프라인은 PG 0으로 두는 게 일반적
        pg = 0.0 if ch == "오프라인" else DEFAULT_PG
        ch_rows.append(
            {
                "channel": ch,
                "fee_rate": DEFAULT_FEES.get(ch, 0.0),
                "pg_rate": pg,
                "mkt_rate": 0.0,
                "ship_per_order": ship,
                "return_rate": 0.0,
                "return_cost_per_order": 0.0,
            }
        )

    st.session_state["channel_params"] = pd.DataFrame(ch_rows)

# Band config
if "band_config" not in st.session_state:
    # Default widths: user can edit
    widths = [
        10,  # 공구
        10,  # 홈쇼핑
        10,  # 폐쇄몰
        12,  # 모바일라방
        10,  # 원데이
        10,  # 브랜드위크
        10,  # 홈사
        12,  # 상시
        10,  # 오프라인
        6,   # MSRP
    ]
    band_rows = []
    for pt, w in zip(PRICE_ORDER_LOW_TO_HIGH, widths):
        band_rows.append({"price_type": pt, "width_pct": float(w), "target_pos_pct": 50.0})
    st.session_state["band_config"] = pd.DataFrame(band_rows)


# ----------------------------
# Core compute: single item price map
# ----------------------------

def compute_item_pricemap(
    item_cost: float,
    msrp_input: float,
    min_override: float,
    max_override: float,
    band_cfg: pd.DataFrame,
    channel_params: pd.DataFrame,
    rounding_unit: int,
    min_margin_rate: float,
    cost_ratio_cap: float,
):
    """Compute min/max for item then each price_type: band range, floor, recommended target, margin."""

    cost = safe_float(item_cost, np.nan)
    msrp = safe_float(msrp_input, np.nan)

    # MSRP policy floor
    msrp_floor = np.nan
    if not np.isnan(cost) and cost_ratio_cap > 0:
        msrp_floor = cost / cost_ratio_cap

    if np.isnan(msrp):
        msrp = msrp_floor
    else:
        if not np.isnan(msrp_floor) and msrp < msrp_floor:
            msrp = msrp_floor

    # max
    max_price = safe_float(max_override, np.nan)
    if np.isnan(max_price):
        max_price = msrp

    # min
    min_price = safe_float(min_override, np.nan)
    if np.isnan(min_price):
        # default: economics floor for lowest price type (공구) with Q=1
        # use 공구 channel row
        ch_map = PRICE_TO_CHANNEL_DEFAULT.get("공구", "공구")
        row = channel_params[channel_params["channel"] == ch_map]
        if not row.empty and not np.isnan(cost):
            floor = econ_floor(cost, row.iloc[0].to_dict(), min_margin_rate)
            min_price = floor
        else:
            min_price = max(0.0, safe_float(cost, 0.0))

    # sanity
    min_price = float(min_price)
    max_price = float(max_price) if not np.isnan(max_price) else float(min_price)
    if max_price < min_price:
        max_price = min_price

    # build normalized band
    band_norm = normalize_band_widths(band_cfg)

    # compute per price type
    rows = []
    for _, br in band_norm.iterrows():
        pt = br["price_type"]
        start_pct = br["start_pct"]
        end_pct = br["end_pct"]
        pos_pct = br["target_pos_pct"]

        band_low, band_tgt, band_high = make_price_from_band(
            min_price, max_price, start_pct, end_pct, pos_pct, rounding_unit
        )

        # economics floor for this price_type (use mapped channel)
        ch = PRICE_TO_CHANNEL_DEFAULT.get(pt, "자사몰")
        ch_row = channel_params[channel_params["channel"] == ch]
        floor = np.nan
        if (not ch_row.empty) and (not np.isnan(cost)):
            floor = econ_floor(cost, ch_row.iloc[0].to_dict(), min_margin_rate)
            floor = krw_round(floor, rounding_unit)

        # recommended target: inside band but not below floor
        tgt = band_tgt
        reason = "밴드 중심"
        if not np.isnan(floor):
            if floor > band_high:
                # impossible in this band
                tgt = floor
                reason = "불가(하한>밴드상단)"
            elif floor > tgt:
                tgt = floor
                reason = "하한 방어"

        # margin at target
        cm_won = np.nan
        cm_rate = np.nan
        room_won = np.nan
        if (not ch_row.empty) and (not np.isnan(cost)):
            cm_won, cm_rate = contribution_margin(tgt, cost, ch_row.iloc[0].to_dict())
            if not np.isnan(floor):
                room_won = tgt - floor

        rows.append(
            {
                "가격영역": pt,
                "BandLow": band_low,
                "BandHigh": band_high,
                "Floor(손익하한)": floor,
                "추천가(Target)": krw_round(tgt, rounding_unit),
                "추천근거": reason,
                "기여이익(원)": (np.nan if np.isnan(cm_won) else int(round(cm_won))),
                "기여이익률": (np.nan if np.isnan(cm_rate) else round(cm_rate * 100.0, 1)),
                "마진룸(원)": (np.nan if np.isnan(room_won) else int(round(room_won))),
                "사용채널(손익)": ch,
            }
        )

    out = pd.DataFrame(rows)

    return {
        "min_price": krw_round(min_price, rounding_unit),
        "max_price": krw_round(max_price, rounding_unit),
        "msrp": krw_round(msrp, rounding_unit) if not np.isnan(msrp) else np.nan,
        "band_norm": band_norm,
        "table": out,
    }


# ----------------------------
# Set/BOM helpers
# ----------------------------

def compute_set_cost_and_reference(master_df, bom_df, set_name):
    """Return (set_cost, reference_value_always, detail_df)

    reference_value_always will be computed later (needs per-SKU always price), so here we only return cost + component list.
    """
    b = bom_df[bom_df["세트명"] == set_name].copy()
    if b.empty:
        return np.nan, pd.DataFrame(columns=["품번", "신규품명", "수량", "원가", "MSRP"])

    b["수량"] = b["수량"].apply(lambda x: int(max(1, safe_float(x, 1))))
    m = master_df[["품번", "신규품명", "원가", "MSRP"]].copy()
    m["품번"] = m["품번"].astype(str)
    b["품번"] = b["품번"].astype(str)

    d = b.merge(m, on="품번", how="left")

    d["원가"] = d["원가"].apply(safe_float)
    d["MSRP"] = d["MSRP"].apply(safe_float)

    d["원가합"] = d["원가"].fillna(0.0) * d["수량"]

    set_cost = d["원가합"].sum()

    return set_cost, d


def compute_set_pricemap(
    set_name: str,
    set_cost: float,
    set_components_df: pd.DataFrame,
    set_min_override: float,
    set_max_override: float,
    band_cfg: pd.DataFrame,
    channel_params: pd.DataFrame,
    rounding_unit: int,
    min_margin_rate: float,
    default_max_from_components: str = "MSRP합",
):
    """Compute band-based prices for a set (order-level)."""

    cost = safe_float(set_cost, np.nan)

    # default max: sum component MSRP (if available)
    comp_msrp_sum = np.nan
    if not set_components_df.empty:
        comp_msrp_sum = np.nansum(set_components_df["MSRP"].values * set_components_df["수량"].values)

    max_price = safe_float(set_max_override, np.nan)
    if np.isnan(max_price):
        if default_max_from_components == "MSRP합" and not np.isnan(comp_msrp_sum) and comp_msrp_sum > 0:
            max_price = comp_msrp_sum
        else:
            # fallback: cost-based
            max_price = cost / 0.30 if (not np.isnan(cost) and cost > 0) else np.nan

    min_price = safe_float(set_min_override, np.nan)
    if np.isnan(min_price):
        # default min: economics floor for lowest price type (공구)
        ch = PRICE_TO_CHANNEL_DEFAULT.get("공구", "공구")
        row = channel_params[channel_params["channel"] == ch]
        if not row.empty and not np.isnan(cost):
            min_price = econ_floor(cost, row.iloc[0].to_dict(), min_margin_rate)
        else:
            min_price = cost

    # sanity
    min_price = float(min_price) if not np.isnan(min_price) else 0.0
    max_price = float(max_price) if not np.isnan(max_price) else min_price
    if max_price < min_price:
        max_price = min_price

    band_norm = normalize_band_widths(band_cfg)

    rows = []
    for _, br in band_norm.iterrows():
        pt = br["price_type"]
        band_low, band_tgt, band_high = make_price_from_band(
            min_price, max_price, br["start_pct"], br["end_pct"], br["target_pos_pct"], rounding_unit
        )

        ch = PRICE_TO_CHANNEL_DEFAULT.get(pt, "자사몰")
        ch_row = channel_params[channel_params["channel"] == ch]
        floor = np.nan
        if not ch_row.empty and not np.isnan(cost):
            floor = econ_floor(cost, ch_row.iloc[0].to_dict(), min_margin_rate)
            floor = krw_round(floor, rounding_unit)

        tgt = band_tgt
        reason = "밴드 중심"
        if not np.isnan(floor):
            if floor > band_high:
                tgt = floor
                reason = "불가(하한>밴드상단)"
            elif floor > tgt:
                tgt = floor
                reason = "하한 방어"

        cm_won = np.nan
        cm_rate = np.nan
        room_won = np.nan
        if not ch_row.empty and not np.isnan(cost):
            cm_won, cm_rate = contribution_margin(tgt, cost, ch_row.iloc[0].to_dict())
            if not np.isnan(floor):
                room_won = tgt - floor

        rows.append(
            {
                "가격영역": pt,
                "BandLow": band_low,
                "BandHigh": band_high,
                "Floor(손익하한)": floor,
                "추천가(Target)": krw_round(tgt, rounding_unit),
                "추천근거": reason,
                "기여이익(원)": (np.nan if np.isnan(cm_won) else int(round(cm_won))),
                "기여이익률": (np.nan if np.isnan(cm_rate) else round(cm_rate * 100.0, 1)),
                "마진룸(원)": (np.nan if np.isnan(room_won) else int(round(room_won))),
                "사용채널(손익)": ch,
            }
        )

    out = pd.DataFrame(rows)
    return {
        "min_price": krw_round(min_price, rounding_unit),
        "max_price": krw_round(max_price, rounding_unit),
        "band_norm": band_norm,
        "table": out,
    }


def allocate_set_price_to_components(
    set_price: float,
    components_df: pd.DataFrame,
    always_price_map: dict,
    rounding_unit: int,
):
    """Allocate set selling price to components by their ALWAYS(상시) value share.

    returns detail df with effective unit price and discount vs always.
    """

    d = components_df.copy()
    if d.empty:
        return d

    d["상시가(단품)"] = d["품번"].map(always_price_map).apply(safe_float)
    d["상시가_가치"] = d["상시가(단품)"].fillna(0.0) * d["수량"].astype(float)

    total_ref = d["상시가_가치"].sum()
    if total_ref <= 0:
        # fallback: MSRP 기준 배분
        d["MSRP_가치"] = d["MSRP"].fillna(0.0) * d["수량"].astype(float)
        total_ref = d["MSRP_가치"].sum()
        d["배분가중치"] = d["MSRP_가치"] / total_ref if total_ref > 0 else 0.0
    else:
        d["배분가중치"] = d["상시가_가치"] / total_ref

    d["배분매출(원)"] = d["배분가중치"] * float(set_price)
    d["실질단가(원)"] = d["배분매출(원)"] / d["수량"].astype(float)

    # discount vs always
    d["상시대비_할인율(%)"] = np.where(
        d["상시가(단품)"].fillna(0.0) > 0,
        (1.0 - (d["실질단가(원)"] / d["상시가(단품)"])) * 100.0,
        np.nan,
    )

    # rounding
    d["배분매출(원)"] = d["배분매출(원)"].apply(lambda x: krw_round(x, rounding_unit))
    d["실질단가(원)"] = d["실질단가(원)"].apply(lambda x: krw_round(x, rounding_unit))
    d["상시대비_할인율(%)"] = d["상시대비_할인율(%)"].apply(lambda x: (np.nan if np.isnan(x) else round(x, 1)))

    show_cols = [
        "품번",
        "신규품명",
        "수량",
        "상시가(단품)",
        "배분가중치",
        "배분매출(원)",
        "실질단가(원)",
        "상시대비_할인율(%)",
    ]
    d["배분가중치"] = d["배분가중치"].apply(lambda x: round(float(x), 4) if not np.isnan(x) else np.nan)

    return d[show_cols]


# ----------------------------
# Title & Tabs
# ----------------------------

st.title("IBR 가격 시뮬레이터 v4 (밴드형 + 세트/BOM + 카니발 체크)")
st.caption("최저~최고 레인지 안에서 채널별 가격영역(밴드)을 정의하고, 수수료/비용 입력에 따라 추천가·마진·카니발을 직관적으로 확인합니다.")


tab_policy, tab_products, tab_sets, tab_plan, tab_logic = st.tabs(
    ["밴드/비용 설정", "단품(제품)", "세트(BOM)", "운영 플랜 & 카니발", "문장형 로직(설명)"]
)

# ============================================================
# Tab 1: Policy / Band / Channel params
# ============================================================
with tab_policy:
    st.subheader("1) 채널 비용/차감 구조(키인하면 전체 재계산)")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        rounding_unit = st.selectbox("반올림 단위", [10, 100, 1000], index=1)
    with c2:
        min_margin_rate = st.slider("최소 기여이익률(%)", 0, 50, 15, 1) / 100.0
    with c3:
        cost_ratio_cap = st.slider("MSRP 원가율 상한(%)", 10, 80, 30, 1) / 100.0
        st.caption("예: 30%면 MSRP ≥ 원가/0.30")
    with c4:
        gap_pct = st.slider("카니발 최소 간격(%)", 0, 30, 5, 1) / 100.0
        gap_won = st.number_input("카니발 최소 간격(원)", min_value=0, value=2000, step=500)

    st.session_state["_rounding_unit"] = rounding_unit
    st.session_state["_min_margin_rate"] = min_margin_rate
    st.session_state["_cost_ratio_cap"] = cost_ratio_cap
    st.session_state["_gap_pct"] = gap_pct
    st.session_state["_gap_won"] = gap_won

    st.caption("채널별 수수료/PG/배송/마케팅/반품을 수정하면, 손익하한(Floor)·추천가(Target)·마진룸이 즉시 바뀝니다.")

    ch_df = st.session_state["channel_params"].copy()
    ch_df = st.data_editor(
        ch_df,
        use_container_width=True,
        num_rows="dynamic",
        height=330,
        column_config={
            "channel": st.column_config.TextColumn("채널", disabled=True),
            "fee_rate": st.column_config.NumberColumn("수수료율", format="%.2f"),
            "pg_rate": st.column_config.NumberColumn("PG", format="%.2f"),
            "mkt_rate": st.column_config.NumberColumn("마케팅비", format="%.2f"),
            "ship_per_order": st.column_config.NumberColumn("배송비/주문(원)", format="%d"),
            "return_rate": st.column_config.NumberColumn("반품률", format="%.2f"),
            "return_cost_per_order": st.column_config.NumberColumn("반품비/주문(원)", format="%d"),
        },
    )
    st.session_state["channel_params"] = ch_df

    st.divider()
    st.subheader("2) 가격 밴드(영역) 설정: 최저~최고 사이에 '어디부터 어디까지가 어떤 가격인지' 정의")
    st.caption("각 가격영역의 폭(width)을 조정하면 전체 레인지(0~100%)에서 영역이 자동으로 재분배됩니다. Target 위치는 각 영역 내부에서 추천가가 찍히는 위치입니다.")

    band_df = st.session_state["band_config"].copy()
    band_df = st.data_editor(
        band_df,
        use_container_width=True,
        height=330,
        column_config={
            "price_type": st.column_config.TextColumn("가격영역", disabled=True),
            "width_pct": st.column_config.NumberColumn("영역폭(%)", min_value=0.0, step=1.0),
            "target_pos_pct": st.column_config.NumberColumn("Target 위치(영역 내 %)", min_value=0.0, max_value=100.0, step=5.0),
        },
    )
    band_norm = normalize_band_widths(band_df)
    st.session_state["band_config"] = band_df

    st.caption("정규화된 밴드(자동 계산):")
    view = band_norm[["price_type", "start_pct", "end_pct", "width_pct", "target_pos_pct"]].copy()
    view["start_pct"] = view["start_pct"].map(lambda x: round(float(x), 1))
    view["end_pct"] = view["end_pct"].map(lambda x: round(float(x), 1))
    view["width_pct"] = view["width_pct"].map(lambda x: round(float(x), 1))
    st.dataframe(view, use_container_width=True, height=260)

    st.divider()
    st.subheader("3) 다운로드(설정/마스터/세트/BOM/플랜)")

    xbytes = to_excel_bytes(
        {
            "channel_params": st.session_state["channel_params"],
            "band_config": band_norm,
            "master_products": st.session_state["master_df"],
            "sets_bom": st.session_state["sets_bom"],
            "plan": st.session_state["plan_df"],
        }
    )
    st.download_button(
        "현재 상태 엑셀로 다운로드",
        data=xbytes,
        file_name="ibr_pricing_v4_state.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ============================================================
# Tab 2: Products
# ============================================================
with tab_products:
    st.subheader("1) 제품 마스터 업로드 (이 파일은 '제품명 통일/품번 기준'으로 사용)")

    up = st.file_uploader("엑셀 업로드 (품번/신규품명 필수, 원가/MSRP 있으면 자동 인식)", type=["xlsx", "xls"], key="master_up")

    def load_master(file):
        df = pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]

        # map columns
        col_map = {}
        for c in df.columns:
            if c in ["품번", "상품코드", "제품코드", "SKU", "코드"]:
                col_map[c] = "품번"
            if c in ["신규품명", "제품명", "상품명", "통일제품명", "통일 제품명"]:
                col_map[c] = "신규품명"
            if c in ["원가", "매입원가", "랜디드코스트", "랜디드코스트(총원가)", "Cost", "cost"]:
                col_map[c] = "원가"
            if c in ["소비자가", "정가", "MSRP", "msrp", "RRP"]:
                col_map[c] = "MSRP"

        df = df.rename(columns=col_map)
        if "품번" not in df.columns:
            df["품번"] = ""
        if "신규품명" not in df.columns:
            df["신규품명"] = ""
        if "원가" not in df.columns:
            df["원가"] = np.nan
        if "MSRP" not in df.columns:
            df["MSRP"] = np.nan

        df["품번"] = df["품번"].astype(str).str.strip()
        df["신규품명"] = df["신규품명"].astype(str).str.strip()
        df = df[df["품번"].ne("") | df["신규품명"].ne("")].copy()

        df["브랜드"] = df["신규품명"].apply(infer_brand_from_name)
        df["최저가(override)"] = np.nan
        df["최고가(override)"] = np.nan
        df["메모"] = ""

        df = df[["품번", "신규품명", "브랜드", "원가", "MSRP", "최저가(override)", "최고가(override)", "메모"]]
        df = df.drop_duplicates(subset=["품번"], keep="first").reset_index(drop=True)
        return df

    if up is not None:
        try:
            st.session_state["master_df"] = load_master(up)
            st.success(f"업로드 완료: {len(st.session_state['master_df']):,}개")
        except Exception as e:
            st.error(f"업로드 처리 오류: {e}")

    st.divider()
    st.subheader("2) 마스터 수정 (원가/MSRP/최저/최고 오버라이드)")

    mdf = st.session_state["master_df"].copy()
    if mdf.empty:
        st.info("먼저 제품 마스터 파일을 업로드하세요.")
    else:
        mdf = st.data_editor(
            mdf,
            use_container_width=True,
            height=360,
            num_rows="dynamic",
            column_config={
                "원가": st.column_config.NumberColumn("원가", format="%d"),
                "MSRP": st.column_config.NumberColumn("MSRP", format="%d"),
                "최저가(override)": st.column_config.NumberColumn("최저가 override", format="%d"),
                "최고가(override)": st.column_config.NumberColumn("최고가 override", format="%d"),
            },
        )
        st.session_state["master_df"] = mdf

        st.divider()
        st.subheader("3) 단품 가격 밴드/추천가/마진(직관형)")

        options = (mdf["품번"].fillna("") + " | " + mdf["신규품명"].fillna("")).tolist()
        pick = st.selectbox("상품 선택", options=options, index=0)
        sku = pick.split(" | ", 1)[0].strip()
        row = mdf[mdf["품번"] == sku].iloc[0].to_dict()

        band_cfg = st.session_state["band_config"].copy()
        channel_params = st.session_state["channel_params"].copy()

        result = compute_item_pricemap(
            item_cost=row.get("원가"),
            msrp_input=row.get("MSRP"),
            min_override=row.get("최저가(override)"),
            max_override=row.get("최고가(override)"),
            band_cfg=band_cfg,
            channel_params=channel_params,
            rounding_unit=st.session_state.get("_rounding_unit", 100),
            min_margin_rate=st.session_state.get("_min_margin_rate", 0.15),
            cost_ratio_cap=st.session_state.get("_cost_ratio_cap", 0.30),
        )

        # show band bar with key markers
        # markers: 공구/상시/MSRP target
        tdf = result["table"].set_index("가격영역")
        markers = {}
        for k in ["공구", "홈쇼핑", "폐쇄몰", "모바일라방", "원데이", "브랜드위크", "홈사", "상시", "오프라인", "MSRP"]:
            if k in tdf.index:
                markers[k] = tdf.loc[k, "추천가(Target)"]

        render_band_bar(
            item_label=f"{sku} | {row.get('신규품명','')}",
            min_price=result["min_price"],
            max_price=result["max_price"],
            band_df=result["band_norm"],
            markers={"공구": markers.get("공구"), "상시": markers.get("상시"), "MSRP": markers.get("MSRP")},
        )

        st.caption("※ 위 바는 하나의 레인지(최저~최고) 안에서 밴드(영역)가 어떻게 나뉘는지 보여줍니다. (표시는 공구/상시/MSRP Target만 찍었습니다.)")

        st.dataframe(result["table"], use_container_width=True, height=360)

        # also show online platforms margin at key online prices (상시/브위/원데이)
        st.divider()
        st.subheader("4) 온라인 플랫폼별 마진 비교(같은 가격이라도 수수료가 달라 마진이 달라짐)")
        online_pts = ["상시", "브랜드위크", "원데이"]
        rows2 = []
        for pt in online_pts:
            if pt not in tdf.index:
                continue
            sell_price = float(tdf.loc[pt, "추천가(Target)"])
            for platform in ONLINE_PLATFORMS:
                ch_row = channel_params[channel_params["channel"] == platform]
                if ch_row.empty:
                    continue
                cost = safe_float(row.get("원가"), np.nan)
                if np.isnan(cost):
                    continue
                floor = econ_floor(cost, ch_row.iloc[0].to_dict(), st.session_state.get("_min_margin_rate", 0.15))
                cm_won, cm_rate = contribution_margin(sell_price, cost, ch_row.iloc[0].to_dict())
                rows2.append(
                    {
                        "가격영역": pt,
                        "플랫폼": platform,
                        "판매가": int(sell_price),
                        "Floor": krw_round(floor, rounding_unit),
                        "CM(원)": int(round(cm_won)),
                        "CM%": round(cm_rate * 100.0, 1),
                    }
                )
        df2 = pd.DataFrame(rows2)
        if df2.empty:
            st.info("원가 입력이 없거나, 계산할 데이터가 없습니다.")
        else:
            st.dataframe(df2, use_container_width=True, height=260)

# ============================================================
# Tab 3: Sets/BOM
# ============================================================
with tab_sets:
    st.subheader("1) 세트(BOM) 구성 입력")
    st.caption("세트명-품번-수량 형태로 구성품을 입력하면, 세트 가격을 자동 추천합니다.")

    mdf = st.session_state["master_df"].copy()
    if mdf.empty:
        st.info("먼저 단품(제품) 탭에서 제품 마스터를 업로드하세요.")
    else:
        # Upload BOM
        st.markdown("**(옵션) 세트 BOM 엑셀 업로드**")
        st.caption("컬럼 예시: 세트명, 품번, 수량")
        up_bom = st.file_uploader("세트 BOM 업로드", type=["xlsx", "xls"], key="bom_up")
        if up_bom is not None:
            try:
                b = pd.read_excel(up_bom)
                b.columns = [str(c).strip() for c in b.columns]
                # rename
                col_map = {}
                for c in b.columns:
                    if c in ["세트명", "set", "set_name", "Set"]:
                        col_map[c] = "세트명"
                    if c in ["품번", "SKU", "상품코드", "제품코드"]:
                        col_map[c] = "품번"
                    if c in ["수량", "qty", "Q"]:
                        col_map[c] = "수량"
                b = b.rename(columns=col_map)
                if not set(["세트명", "품번", "수량"]).issubset(set(b.columns)):
                    st.error("업로드 파일에 '세트명/품번/수량' 컬럼이 필요합니다.")
                else:
                    b = b[["세트명", "품번", "수량"]].copy()
                    b["품번"] = b["품번"].astype(str).str.strip()
                    b["세트명"] = b["세트명"].astype(str).str.strip()
                    b["수량"] = b["수량"].apply(lambda x: int(max(1, safe_float(x, 1))))
                    st.session_state["sets_bom"] = b
                    st.success(f"BOM 업로드 완료: {b['세트명'].nunique():,}개 세트")
            except Exception as e:
                st.error(f"BOM 업로드 오류: {e}")

        st.divider()
        st.subheader("2) BOM 직접 편집")

        bom = st.session_state["sets_bom"].copy()
        bom = st.data_editor(
            bom,
            use_container_width=True,
            height=320,
            num_rows="dynamic",
            column_config={
                "세트명": st.column_config.TextColumn("세트명"),
                "품번": st.column_config.TextColumn("품번"),
                "수량": st.column_config.NumberColumn("수량", min_value=1, step=1),
            },
        )
        st.session_state["sets_bom"] = bom

        st.divider()
        st.subheader("3) 세트 가격 추천 + 구성품 실질단가(상시가 비율 배분)")

        set_names = sorted(bom["세트명"].dropna().astype(str).unique().tolist())
        if len(set_names) == 0:
            st.info("먼저 BOM에 세트 구성을 입력하세요.")
        else:
            set_pick = st.selectbox("세트 선택", options=set_names)

            # set overrides
            cA, cB = st.columns([1, 1])
            with cA:
                set_min_override = st.number_input("세트 최저가 override(미입력=자동)", min_value=0, value=0, step=1000)
            with cB:
                set_max_override = st.number_input("세트 최고가 override(미입력=자동)", min_value=0, value=0, step=1000)

            set_min_override = np.nan if set_min_override == 0 else float(set_min_override)
            set_max_override = np.nan if set_max_override == 0 else float(set_max_override)

            set_cost, comp_df = compute_set_cost_and_reference(mdf, bom, set_pick)

            st.markdown(f"**세트 원가 합계:** {int(round(set_cost)):,}원")
            st.dataframe(comp_df[["품번", "신규품명", "수량", "원가", "MSRP"]], use_container_width=True, height=220)

            band_cfg = st.session_state["band_config"].copy()
            channel_params = st.session_state["channel_params"].copy()

            set_res = compute_set_pricemap(
                set_name=set_pick,
                set_cost=set_cost,
                set_components_df=comp_df,
                set_min_override=set_min_override,
                set_max_override=set_max_override,
                band_cfg=band_cfg,
                channel_params=channel_params,
                rounding_unit=st.session_state.get("_rounding_unit", 100),
                min_margin_rate=st.session_state.get("_min_margin_rate", 0.15),
            )

            # markers for a few
            tdf = set_res["table"].set_index("가격영역")
            render_band_bar(
                item_label=f"SET | {set_pick}",
                min_price=set_res["min_price"],
                max_price=set_res["max_price"],
                band_df=set_res["band_norm"],
                markers={"공구": float(tdf.loc["공구", "추천가(Target)"]) if "공구" in tdf.index else None,
                         "상시": float(tdf.loc["상시", "추천가(Target)"]) if "상시" in tdf.index else None,
                         "MSRP": float(tdf.loc["MSRP", "추천가(Target)"]) if "MSRP" in tdf.index else None},
            )

            st.dataframe(set_res["table"], use_container_width=True, height=360)

            # Build always price map from product pricemap (상시 target)
            # compute for all SKUs once (small datasets ok)
            always_price_map = {}
            for _, pr in mdf.iterrows():
                sku = str(pr.get("품번", "")).strip()
                if not sku:
                    continue
                r = compute_item_pricemap(
                    item_cost=pr.get("원가"),
                    msrp_input=pr.get("MSRP"),
                    min_override=pr.get("최저가(override)"),
                    max_override=pr.get("최고가(override)"),
                    band_cfg=band_cfg,
                    channel_params=channel_params,
                    rounding_unit=st.session_state.get("_rounding_unit", 100),
                    min_margin_rate=st.session_state.get("_min_margin_rate", 0.15),
                    cost_ratio_cap=st.session_state.get("_cost_ratio_cap", 0.30),
                )
                t = r["table"].set_index("가격영역")
                if "상시" in t.index:
                    always_price_map[sku] = float(t.loc["상시", "추천가(Target)"])

            st.divider()
            st.subheader("4) 구성품별 실질단가/할인율(상시가 비율 배분)")

            # choose a channel price type to analyze allocation
            price_type_for_alloc = st.selectbox("어느 가격영역의 세트가로 배분할까요?", options=PRICE_ORDER_LOW_TO_HIGH, index=PRICE_ORDER_LOW_TO_HIGH.index("브랜드위크") if "브랜드위크" in PRICE_ORDER_LOW_TO_HIGH else 0)
            set_price = float(tdf.loc[price_type_for_alloc, "추천가(Target)"]) if price_type_for_alloc in tdf.index else np.nan

            alloc_df = allocate_set_price_to_components(
                set_price=set_price,
                components_df=comp_df[["품번", "신규품명", "수량", "MSRP"]].copy(),
                always_price_map=always_price_map,
                rounding_unit=rounding_unit,
            )
            if alloc_df.empty:
                st.info("배분할 데이터가 없습니다.")
            else:
                st.dataframe(alloc_df, use_container_width=True, height=260)

            st.divider()
            st.subheader("5) 세트 비교(공통 구성품이 있을 때 어느 세트가 더 저렴한가)")

            col1, col2 = st.columns([1, 1])
            with col1:
                set_a = st.selectbox("세트 A", options=set_names, index=set_names.index(set_pick))
            with col2:
                set_b = st.selectbox("세트 B", options=set_names, index=min(len(set_names)-1, set_names.index(set_pick)+1) if len(set_names) > 1 else 0)

            if set_a and set_b:
                cost_a, comp_a = compute_set_cost_and_reference(mdf, bom, set_a)
                cost_b, comp_b = compute_set_cost_and_reference(mdf, bom, set_b)

                # for fair comparison: use same price_type_for_alloc and recommended set price for each
                res_a = compute_set_pricemap(
                    set_name=set_a,
                    set_cost=cost_a,
                    set_components_df=comp_a,
                    set_min_override=np.nan,
                    set_max_override=np.nan,
                    band_cfg=band_cfg,
                    channel_params=channel_params,
                    rounding_unit=rounding_unit,
                    min_margin_rate=min_margin_rate,
                )
                res_b = compute_set_pricemap(
                    set_name=set_b,
                    set_cost=cost_b,
                    set_components_df=comp_b,
                    set_min_override=np.nan,
                    set_max_override=np.nan,
                    band_cfg=band_cfg,
                    channel_params=channel_params,
                    rounding_unit=rounding_unit,
                    min_margin_rate=min_margin_rate,
                )

                tA = res_a["table"].set_index("가격영역")
                tB = res_b["table"].set_index("가격영역")

                price_a = float(tA.loc[price_type_for_alloc, "추천가(Target)"]) if price_type_for_alloc in tA.index else np.nan
                price_b = float(tB.loc[price_type_for_alloc, "추천가(Target)"]) if price_type_for_alloc in tB.index else np.nan

                alloc_a = allocate_set_price_to_components(price_a, comp_a[["품번", "신규품명", "수량", "MSRP"]].copy(), always_price_map, rounding_unit)
                alloc_b = allocate_set_price_to_components(price_b, comp_b[["품번", "신규품명", "수량", "MSRP"]].copy(), always_price_map, rounding_unit)

                if alloc_a.empty or alloc_b.empty:
                    st.info("비교할 데이터가 부족합니다.")
                else:
                    # compare only shared products
                    shared = set(alloc_a["품번"].astype(str)) & set(alloc_b["품번"].astype(str))
                    if len(shared) == 0:
                        st.info("공통 구성품이 없습니다.")
                    else:
                        aa = alloc_a[alloc_a["품번"].astype(str).isin(shared)].copy()
                        bb = alloc_b[alloc_b["품번"].astype(str).isin(shared)].copy()
                        cmp = aa.merge(bb, on="품번", suffixes=("_A", "_B"))
                        cmp["A가 더 저렴?"] = np.where(
                            cmp["실질단가(원)_A"].astype(float) < cmp["실질단가(원)_B"].astype(float),
                            "A",
                            np.where(
                                cmp["실질단가(원)_A"].astype(float) > cmp["실질단가(원)_B"].astype(float),
                                "B",
                                "동일",
                            ),
                        )
                        show = cmp[[
                            "품번",
                            "신규품명_A",
                            "실질단가(원)_A",
                            "상시대비_할인율(%)_A",
                            "실질단가(원)_B",
                            "상시대비_할인율(%)_B",
                            "A가 더 저렴?",
                        ]].rename(columns={"신규품명_A": "제품명"})
                        st.dataframe(show, use_container_width=True, height=260)

# ============================================================
# Tab 4: Plan & Cannibalization
# ============================================================
with tab_plan:
    st.subheader("1) 운영 플랜 입력")
    st.caption("채널(가격영역)별로 어떤 오퍼(단품/세트)를 운영할지 입력하면, 공통 SKU 기준으로 가격 역전(카니발)을 잡아냅니다.")

    mdf = st.session_state["master_df"].copy()
    bom = st.session_state["sets_bom"].copy()

    if mdf.empty:
        st.info("먼저 제품 마스터를 업로드하세요.")
    else:
        plan = st.session_state["plan_df"].copy()
        plan = st.data_editor(
            plan,
            use_container_width=True,
            height=320,
            num_rows="dynamic",
            column_config={
                "채널(가격영역)": st.column_config.SelectboxColumn("채널(가격영역)", options=PRICE_ORDER_LOW_TO_HIGH),
                "오퍼유형": st.column_config.SelectboxColumn("오퍼유형", options=["단품", "세트"]),
                "판매가(override)": st.column_config.NumberColumn("판매가 override", format="%d"),
            },
        )
        st.session_state["plan_df"] = plan

        st.divider()
        st.subheader("2) 카니발 분석")

        if plan.empty:
            st.info("플랜 테이블에 최소 1개 이상의 오퍼를 추가하세요.")
        else:
            band_cfg = st.session_state["band_config"].copy()
            channel_params = st.session_state["channel_params"].copy()
            rounding_unit = st.session_state.get("_rounding_unit", 100)
            min_margin_rate = st.session_state.get("_min_margin_rate", 0.15)
            cost_ratio_cap = st.session_state.get("_cost_ratio_cap", 0.30)
            gap_pct = st.session_state.get("_gap_pct", 0.05)
            gap_won = st.session_state.get("_gap_won", 2000)

            # compute always map
            always_price_map = {}
            sku_name_map = {}
            sku_cost_map = {}
            for _, pr in mdf.iterrows():
                sku = str(pr.get("품번", "")).strip()
                if not sku:
                    continue
                sku_name_map[sku] = pr.get("신규품명", "")
                sku_cost_map[sku] = safe_float(pr.get("원가"), np.nan)
                r = compute_item_pricemap(
                    item_cost=pr.get("원가"),
                    msrp_input=pr.get("MSRP"),
                    min_override=pr.get("최저가(override)"),
                    max_override=pr.get("최고가(override)"),
                    band_cfg=band_cfg,
                    channel_params=channel_params,
                    rounding_unit=rounding_unit,
                    min_margin_rate=min_margin_rate,
                    cost_ratio_cap=cost_ratio_cap,
                )
                t = r["table"].set_index("가격영역")
                if "상시" in t.index:
                    always_price_map[sku] = float(t.loc["상시", "추천가(Target)"])

            # helper: get recommended price for single SKU at a given price_type
            def get_single_price(sku, price_type):
                pr = mdf[mdf["품번"].astype(str) == str(sku)].iloc[0].to_dict()
                r = compute_item_pricemap(
                    item_cost=pr.get("원가"),
                    msrp_input=pr.get("MSRP"),
                    min_override=pr.get("최저가(override)"),
                    max_override=pr.get("최고가(override)"),
                    band_cfg=band_cfg,
                    channel_params=channel_params,
                    rounding_unit=rounding_unit,
                    min_margin_rate=min_margin_rate,
                    cost_ratio_cap=cost_ratio_cap,
                )
                t = r["table"].set_index("가격영역")
                if price_type in t.index:
                    return float(t.loc[price_type, "추천가(Target)"])
                return np.nan

            # helper: get recommended price for set at price_type
            def get_set_price(set_name, price_type):
                set_cost, comp_df = compute_set_cost_and_reference(mdf, bom, set_name)
                res = compute_set_pricemap(
                    set_name=set_name,
                    set_cost=set_cost,
                    set_components_df=comp_df,
                    set_min_override=np.nan,
                    set_max_override=np.nan,
                    band_cfg=band_cfg,
                    channel_params=channel_params,
                    rounding_unit=rounding_unit,
                    min_margin_rate=min_margin_rate,
                )
                t = res["table"].set_index("가격영역")
                if price_type in t.index:
                    return float(t.loc[price_type, "추천가(Target)"])
                return np.nan

            # build effective price records
            eff_rows = []
            for _, r in plan.iterrows():
                price_type = str(r.get("채널(가격영역)", "")).strip()
                offer_type = str(r.get("오퍼유형", "")).strip()
                offer_name = str(r.get("오퍼명", "")).strip()
                override_price = safe_float(r.get("판매가(override)", np.nan), np.nan)

                if not price_type or not offer_type or not offer_name:
                    continue

                if offer_type == "단품":
                    sku = offer_name
                    if sku not in sku_name_map:
                        continue
                    sell = override_price if not np.isnan(override_price) else get_single_price(sku, price_type)
                    eff_rows.append(
                        {
                            "채널(가격영역)": price_type,
                            "오퍼유형": offer_type,
                            "오퍼명": offer_name,
                            "품번": sku,
                            "제품명": sku_name_map.get(sku, ""),
                            "실질단가(원)": krw_round(sell, rounding_unit),
                        }
                    )

                else:  # set
                    set_name = offer_name
                    if bom[bom["세트명"] == set_name].empty:
                        continue
                    set_cost, comp_df = compute_set_cost_and_reference(mdf, bom, set_name)
                    sell_set = override_price if not np.isnan(override_price) else get_set_price(set_name, price_type)
                    alloc = allocate_set_price_to_components(
                        set_price=sell_set,
                        components_df=comp_df[["품번", "신규품명", "수량", "MSRP"]].copy(),
                        always_price_map=always_price_map,
                        rounding_unit=rounding_unit,
                    )
                    for _, ar in alloc.iterrows():
                        sku = str(ar["품번"])
                        eff_rows.append(
                            {
                                "채널(가격영역)": price_type,
                                "오퍼유형": offer_type,
                                "오퍼명": set_name,
                                "품번": sku,
                                "제품명": ar.get("신규품명", sku_name_map.get(sku, "")),
                                "실질단가(원)": safe_float(ar.get("실질단가(원)", np.nan), np.nan),
                            }
                        )

            eff = pd.DataFrame(eff_rows)
            if eff.empty:
                st.warning("플랜에서 유효한 오퍼/품번을 찾지 못했습니다. 오퍼명(단품=품번, 세트=세트명) 입력을 확인하세요.")
            else:
                # pivot: min effective price per SKU per price_type
                pv = eff.pivot_table(
                    index=["품번", "제품명"],
                    columns="채널(가격영역)",
                    values="실질단가(원)",
                    aggfunc="min",
                )
                pv = pv.reindex(columns=PRICE_ORDER_LOW_TO_HIGH)

                st.markdown("**채널별 최저 실질단가(세트 배분 포함)**")
                st.dataframe(pv.reset_index(), use_container_width=True, height=360)

                # detect cannibalization (higher band cheaper than lower band - gap)
                violations = []
                cols = [c for c in PRICE_ORDER_LOW_TO_HIGH if c in pv.columns]
                for (sku, pname), rowv in pv.iterrows():
                    prev_price = None
                    prev_ch = None
                    for ch in cols:
                        cur = rowv.get(ch, np.nan)
                        if np.isnan(cur):
                            continue
                        if prev_price is None:
                            prev_price = float(cur)
                            prev_ch = ch
                            continue
                        # requirement: cur >= max(prev*(1+gap_pct), prev + gap_won)
                        thresh = max(prev_price * (1.0 + gap_pct), prev_price + gap_won)
                        if float(cur) < thresh:
                            violations.append(
                                {
                                    "품번": sku,
                                    "제품명": pname,
                                    "하위채널": prev_ch,
                                    "하위가격": int(prev_price),
                                    "상위채널": ch,
                                    "상위가격": int(float(cur)),
                                    "요구최소상위가격": int(thresh),
                                    "갭부족(원)": int(thresh - float(cur)),
                                }
                            )
                        prev_price = float(cur)
                        prev_ch = ch

                st.divider()
                st.subheader("3) 카니발(가격 역전/갭 부족) 경고")
                if len(violations) == 0:
                    st.success("카니발 위반 없음(현재 플랜 + 갭 룰 기준).")
                else:
                    vdf = pd.DataFrame(violations)
                    st.warning(f"{len(vdf):,}건")
                    st.dataframe(vdf, use_container_width=True, height=260)

                st.divider()
                st.subheader("4) 상세 근거(어떤 오퍼가 최저를 만들었는지)")
                st.caption("아래 테이블은 채널별 실질단가를 만든 오퍼 레벨 데이터를 포함합니다(세트의 경우 배분 단가).")
                st.dataframe(eff, use_container_width=True, height=320)

# ============================================================
# Tab 5: Logic (Korean explanation)
# ============================================================
with tab_logic:
    st.subheader("가격 로직 요약(문장형)")

    st.markdown(
        """
### 1) 가격은 ‘최저가 ~ 최고가’ 하나의 레인지 안에서만 움직입니다.
- 각 SKU(또는 세트)는 **최저가(min)**와 **최고가(max)**를 가집니다.
- 이 레인지 안에서 **공구/홈쇼핑/폐쇄몰/라방/원데이/브랜드위크/홈사/상시/오프라인/MSRP**가 차지하는 영역(밴드)을 미리 정의합니다.
- 즉, “어디부터 어디까지가 어떤 가격인지”가 **비율(0~100%)로 고정**되고, 숫자는 min/max에 의해 자동 스케일링 됩니다.

### 2) 채널별 손익 하한(Floor)을 먼저 계산합니다.
각 가격영역은 실제 판매 채널의 차감 구조(수수료/PG/마케팅/배송/반품)를 반영해, 아래의 최소 허용 판매가를 가집니다.

- **Floor(손익하한)**
\[
Floor = \frac{(원가 + 배송비/주문 + 반품률\times반품비/주문)}{1-(수수료+PG+마케팅비+최소기여이익률)}
\]

- Floor를 밑으로 내려가면, 목표 최소 기여이익률을 만족할 수 없습니다.

### 3) 추천가(Target)는 ‘밴드’ 안에서 정하되, Floor를 침범하지 않습니다.
- 각 가격영역 밴드에는 **BandLow ~ BandHigh**가 있고,
- Target은 밴드 내부의 위치(예: 중앙 50%)를 기본으로 찍습니다.
- 단, **Floor > Target**이면 Target을 Floor까지 끌어올려 ‘마진 방어’를 합니다.
- **Floor > BandHigh**이면 해당 영역은 구조적으로 불가능(불가)로 표시됩니다.

### 4) 마진룸(Margin Room)은 Target과 Floor의 차이입니다.
- **마진룸(원)** = Target - Floor
- 이 값이 클수록, 동일 채널에서 쿠폰/광고비/추가할인을 쓸 여지가 큽니다.

### 5) 세트(BOM)는 구성품 합산 원가로 손익을 계산하고, 가격은 동일한 밴드 로직으로 추천합니다.
- 세트 원가 = Σ(구성품 원가 × 수량)
- 세트는 하나의 오더로 판매되므로, 배송/반품은 주문당 비용으로 반영됩니다.

### 6) 세트 구성품별 ‘실질 단가’는 상시가 비율로 배분해 근사합니다.
- 세트 A와 세트 B가 같은 SKU를 포함하면, 각 세트에서 해당 SKU의 ‘실질 단가’를 비교해
  **어느 세트가 더 싸게 팔리고 있는지**를 직관적으로 확인할 수 있습니다.

### 7) 운영 플랜을 입력하면 카니발(가격 역전)을 자동으로 잡습니다.
- 채널(가격영역)별로 운영할 단품/세트를 지정하면,
- 세트의 경우 배분 단가까지 반영해 SKU 단위의 최저 실질단가를 계산합니다.
- 그리고 낮은 채널→높은 채널 순서에서
  **상위 채널 가격이 하위 채널 가격보다 충분히 높지 않으면**(예: +5% 또는 +2,000원 미만)
  카니발 경고로 표시합니다.
"""
    )

    st.info("이 버전은 ‘밴드’를 1차 UI로 두었고, 수수료/비용 입력이 손익 하한과 마진룸을 통해 즉시 가격에 반영되도록 구성했습니다.\n\n다음 단계로는: (1) 세트 전용 SKU/울타리(구성 고정) 체크, (2) 채널별 쿠폰/적립 중첩까지 포함한 ‘최종가’ 카니발 체크로 확장할 수 있습니다.")

