import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# IBR Pricing Simulator v6.2 — UX Redesign
# ============================================================

st.set_page_config(
    page_title="IBR 가격 시뮬레이터",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💰"
)

# ────────────────────────────────────────────────────────────
# GLOBAL CSS
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── 전체 기본 ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── 헤더/타이틀 영역 ── */
.app-header {
    background: linear-gradient(135deg, #1a1f36 0%, #0f172a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    border-left: 5px solid #6366f1;
    box-shadow: 0 4px 24px rgba(99,102,241,0.18);
}
.app-header h1 {
    color: #f8fafc;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.6;
}
.app-header .badge-row {
    margin-top: 14px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-indigo { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.badge-amber  { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }

/* ── 섹션 헤더 ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
}
.section-icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.icon-indigo { background: rgba(99,102,241,0.12); }
.icon-green  { background: rgba(16,185,129,0.12); }
.icon-amber  { background: rgba(245,158,11,0.12); }
.icon-red    { background: rgba(239,68,68,0.12); }
.icon-blue   { background: rgba(59,130,246,0.12); }
.section-header h3 {
    font-size: 1.02rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
}
.section-header .sub {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0;
}

/* ── 스텝 카드 (업로드 가이드) ── */
.step-card {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.step-card:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 14px rgba(99,102,241,0.1);
}
.step-card.active {
    border-color: #6366f1;
    background: #fafafe;
    box-shadow: 0 4px 14px rgba(99,102,241,0.12);
}
.step-card.done {
    border-color: #10b981;
    background: #f0fdf9;
}
.step-num {
    position: absolute;
    top: -10px;
    left: 18px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    background: #6366f1;
    color: white;
}
.step-num.done-num { background: #10b981; }
.step-card h4 { font-size: 0.88rem; font-weight: 700; color: #1e293b; margin: 6px 0 4px; }
.step-card p  { font-size: 0.78rem; color: #64748b; margin: 0; line-height: 1.5; }

/* ── KPI 카드 ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 0 0 20px 0;
}
.kpi-card {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.kpi-indigo::before { background: #6366f1; }
.kpi-card.kpi-green::before  { background: #10b981; }
.kpi-card.kpi-amber::before  { background: #f59e0b; }
.kpi-card.kpi-red::before    { background: #ef4444; }
.kpi-card.kpi-blue::before   { background: #3b82f6; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 3px;
}

/* ── 가격 밴드 시각화 ── */
.price-band-container {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 22px;
    margin: 14px 0;
}
.price-band-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 14px;
}

/* ── 상태 배지 ── */
.status-ok    { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; background:rgba(16,185,129,0.12); color:#059669; border:1px solid rgba(16,185,129,0.3); }
.status-warn  { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; background:rgba(245,158,11,0.12); color:#d97706; border:1px solid rgba(245,158,11,0.3); }
.status-error { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; background:rgba(239,68,68,0.12); color:#dc2626; border:1px solid rgba(239,68,68,0.3); }

/* ── 인포 박스 ── */
.info-box {
    background: linear-gradient(135deg, #f0f4ff, #fafafe);
    border: 1.5px solid #c7d2fe;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
}
.info-box p { font-size: 0.82rem; color: #4338ca; margin: 0; line-height: 1.6; }

.warn-box {
    background: linear-gradient(135deg, #fffbeb, #fefce8);
    border: 1.5px solid #fde68a;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
}
.warn-box p { font-size: 0.82rem; color: #92400e; margin: 0; line-height: 1.6; }

.success-box {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1.5px solid #a7f3d0;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
}
.success-box p { font-size: 0.82rem; color: #065f46; margin: 0; line-height: 1.6; }

/* ── 구분선 ── */
.divider { border: none; border-top: 1.5px solid #f1f5f9; margin: 22px 0; }

/* ── 채널 그리드 ── */
.channel-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 10px 0;
}
.channel-item {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
    font-size: 0.8rem;
}
.channel-item .ch-name { font-weight: 700; color: #1e293b; margin-bottom: 2px; }
.channel-item .ch-val  { font-size: 0.72rem; color: #6366f1; font-weight: 600; }

/* ── 로직 문서 스타일 ── */
.logic-card {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 12px 0;
}
.logic-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.logic-card p, .logic-card li {
    font-size: 0.83rem;
    color: #475569;
    line-height: 1.7;
    margin: 4px 0;
}
.logic-card code {
    background: #f1f5f9;
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 0.8rem;
    color: #6366f1;
    font-family: 'Courier New', monospace;
}

/* ── Streamlit 기본 요소 오버라이드 ── */
div[data-testid="stTabs"] > div > div > div > div > button {
    font-weight: 600;
    font-size: 0.86rem;
    padding: 10px 20px;
    border-radius: 8px 8px 0 0;
}
div[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stAlert { border-radius: 10px; }

/* ── 파일 업로더 커스텀 ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed #c7d2fe;
    border-radius: 14px;
    background: #fafafe;
    padding: 8px;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;
}

/* ── 버튼 스타일 ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 20px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
    transition: transform 0.1s, box-shadow 0.1s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99,102,241,0.4);
}
div[data-testid="stDownloadButton"] > button {
    border-radius: 10px;
    font-weight: 600;
}

/* ── 선택 영역 하이라이트 ── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px;
}
div[data-testid="stNumberInput"] > div {
    border-radius: 10px;
}

/* ── 탭 전체 컨텐츠 여백 ── */
div[data-testid="stTabsContent"] {
    padding-top: 20px;
}

/* ── 범례 행 ── */
.legend-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    align-items: center;
    margin: 8px 0;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    color: #64748b;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────
PRICE_ZONES = ["공구", "홈쇼핑", "폐쇄몰", "모바일라방", "원데이", "브랜드위크", "홈사", "상시", "오프라인", "MSRP"]

ZONE_COLORS = {
    "공구":      "#ef4444",
    "홈쇼핑":    "#f97316",
    "폐쇄몰":    "#eab308",
    "모바일라방": "#84cc16",
    "원데이":    "#22c55e",
    "브랜드위크": "#14b8a6",
    "홈사":      "#3b82f6",
    "상시":      "#6366f1",
    "오프라인":  "#8b5cf6",
    "MSRP":     "#ec4899",
}

DEFAULT_CHANNELS = [
    ("오프라인",     0.50, 0.00, 0.0,    0.00, 0.00, 0.0),
    ("자사몰",       0.05, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("스마트스토어", 0.05, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("쿠팡",         0.25, 0.00, 3000.0, 0.00, 0.00, 0.0),
    ("오픈마켓",     0.15, 0.00, 3000.0, 0.00, 0.00, 0.0),
    ("홈사",         0.30, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("공구",         0.50, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("홈쇼핑",       0.55, 0.00, 0.0,    0.00, 0.00, 0.0),
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

DEFAULT_BOUNDARIES = [0, 10, 20, 30, 42, 52, 62, 72, 84, 94, 100]


# ────────────────────────────────────────────────────────────
# 유틸리티
# ────────────────────────────────────────────────────────────
def default_zone_target_pos(boundaries):
    out = {}
    for i, z in enumerate(PRICE_ZONES):
        out[z] = (boundaries[i] + boundaries[i+1]) / 2
    return out

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

def fmt_krw(v):
    try:
        return f"₩{int(v):,}"
    except:
        return "—"

def fmt_pct(v):
    try:
        return f"{v:.1f}%"
    except:
        return "—"


# ────────────────────────────────────────────────────────────
# 원가 마스터 로드
# ────────────────────────────────────────────────────────────
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

    code_col  = "상품코드" if "상품코드" in df.columns else None
    name_col  = "상품명"   if "상품명"   in df.columns else None
    brand_col = "브랜드"   if "브랜드"   in df.columns else None

    cost_col = None
    for c in ["원가 (vat-)", "원가", "매입원가", "랜디드코스트", "랜디드코스트(총원가)"]:
        if c in df.columns:
            cost_col = c
            break

    out = pd.DataFrame({
        "품번":  df[code_col].astype(str).str.strip() if code_col else "",
        "상품명": df[name_col].astype(str).str.strip() if name_col else "",
        "브랜드": df[brand_col].astype(str).str.strip() if brand_col else "",
        "원가":  pd.to_numeric(df[cost_col], errors="coerce") if cost_col else np.nan,
    })
    out = out[out["품번"].ne("")].drop_duplicates(subset=["품번"]).reset_index(drop=True)
    out["MSRP_오버라이드"] = np.nan
    out["Min_오버라이드"]  = np.nan
    out["Max_오버라이드"]  = np.nan
    out["운영여부"]         = True
    return out


# ────────────────────────────────────────────────────────────
# 경제 계산
# ────────────────────────────────────────────────────────────
def floor_price(cost_total, q_orders, fee, pg, mkt, ship_per_order, ret_rate, ret_cost_order, min_cm):
    denom = 1.0 - (fee + pg + mkt + min_cm)
    if denom <= 0:
        return float("inf")
    ship_unit = ship_per_order / max(1, q_orders)
    ret_unit  = (ret_rate * ret_cost_order) / max(1, q_orders)
    return (cost_total + ship_unit + ret_unit) / denom

def contrib_metrics(price, cost_total, q_orders, fee, pg, mkt, ship_per_order, ret_rate, ret_cost_order):
    if price <= 0:
        return np.nan, np.nan
    ship_unit = ship_per_order / max(1, q_orders)
    ret_unit  = (ret_rate * ret_cost_order) / max(1, q_orders)
    net = price * (1.0 - fee - pg - mkt) - ship_unit - ret_unit - cost_total
    return net, net / price


# ────────────────────────────────────────────────────────────
# 자동 레인지
# ────────────────────────────────────────────────────────────
def compute_auto_range_from_cost(cost_total, channels_df, zone_map, boundaries, rounding_unit, min_cm,
                                  max_cost_ratio, include_zones, min_zone="공구", msrp_override=np.nan):
    ch_map = channels_df.set_index("채널명").to_dict("index")

    def zone_floor(z):
        ch = zone_map.get(z, "자사몰")
        p  = ch_map.get(ch, None)
        if p is None:
            return np.nan
        return floor_price(cost_total, 1, p["수수료율"], p["PG"], p["마케팅비"],
                           p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"], min_cm)

    min_auto = zone_floor(min_zone)
    if min_auto != min_auto or min_auto <= 0:
        return np.nan, np.nan, {"note": "Min(Floor) 산출 불가"}
    min_auto = krw_round(min_auto, rounding_unit)

    msrp_base = krw_ceil(cost_total / max_cost_ratio, rounding_unit) if (cost_total == cost_total and cost_total > 0) else np.nan

    max_req = []
    for i, z in enumerate(PRICE_ZONES):
        if z not in include_zones:
            continue
        fz  = zone_floor(z)
        end = boundaries[i+1] / 100.0
        if fz != fz or end <= 0:
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
    if max_req and (max_auto > (msrp_base if msrp_base == msrp_base else 0)):
        note = "채널 손익하한이 밴드에 들어오도록 MSRP(=Max)를 자동 상향했습니다."
    return min_auto, max_auto, {"note": note, "msrp_base": msrp_base}


# ────────────────────────────────────────────────────────────
# 존 테이블 빌드
# ────────────────────────────────────────────────────────────
def build_zone_table(cost_total, min_price, max_price, channels_df, zone_map, boundaries,
                      target_pos, rounding_unit, min_cm, overrides_df, item_type, item_id):
    ch_map = channels_df.set_index("채널명").to_dict("index")
    rows   = []
    if min_price != min_price or max_price != max_price or max_price <= min_price:
        return pd.DataFrame()

    span = max_price - min_price
    for i, z in enumerate(PRICE_ZONES):
        start      = boundaries[i] / 100.0
        end        = boundaries[i+1] / 100.0
        band_low   = min_price + span * start
        band_high  = min_price + span * end
        pos        = target_pos.get(z, (boundaries[i]+boundaries[i+1])/2) / 100.0
        target_raw = min_price + span * pos

        ch = zone_map.get(z, "자사몰")
        p  = ch_map.get(ch, None)
        if p is None:
            continue

        floor = floor_price(cost_total, 1, p["수수료율"], p["PG"], p["마케팅비"],
                            p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"], min_cm)

        status = "정상"
        target = max(target_raw, floor)
        if floor > band_high:
            status = "불가(Floor>BandHigh)"
            target = band_high
        elif target > band_high:
            status = "클립(Target→BandHigh)"
            target = band_high

        ov = overrides_df[
            (overrides_df["오퍼타입"] == item_type) &
            (overrides_df["오퍼ID"]   == item_id) &
            (overrides_df["가격영역"] == z)
        ]
        override_price = np.nan
        if not ov.empty:
            override_price = safe_float(ov.iloc[0]["가격_오버라이드"], np.nan)

        effective = override_price if (override_price == override_price and override_price > 0) else target

        band_low_r  = krw_round(band_low,  rounding_unit)
        band_high_r = krw_round(band_high, rounding_unit)
        floor_r     = krw_round(floor,     rounding_unit)
        target_r    = krw_round(target,    rounding_unit)
        eff_r       = krw_round(effective, rounding_unit) if (effective == effective and effective > 0) else np.nan

        cm, cmr = contrib_metrics(eff_r if eff_r == eff_r else 0, cost_total, 1,
                                   p["수수료율"], p["PG"], p["마케팅비"],
                                   p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"])

        flags = []
        if eff_r == eff_r and eff_r < floor_r:
            flags.append("⚠️ Floor 미만(손익위험)")
        if eff_r == eff_r and eff_r < band_low_r:
            flags.append("⚠️ BandLow 미만")
        if eff_r == eff_r and eff_r > band_high_r and z != "MSRP":
            flags.append("⚠️ BandHigh 초과")

        rows.append({
            "가격영역":            z,
            "비용채널":            ch,
            "BandLow":             band_low_r,
            "BandHigh":            band_high_r,
            "Floor(손익하한)":     floor_r,
            "추천가(Target)":      target_r,
            "가격_오버라이드(원)": (krw_round(override_price, rounding_unit) if override_price == override_price else np.nan),
            "최종가격(원)":        eff_r,
            "상태":                status,
            "경고":                " / ".join(flags),
            "마진룸(원)":          (eff_r - floor_r) if (eff_r == eff_r) else np.nan,
            "기여이익(원)":        int(round(cm)) if cm == cm else np.nan,
            "기여이익률(%)":       round(cmr*100, 1) if cmr == cmr else np.nan,
        })
    return pd.DataFrame(rows)

def check_order_violations(zdf, gap_pct, gap_won):
    if zdf.empty:
        return pd.DataFrame()
    order = {z: i for i, z in enumerate(PRICE_ZONES)}
    g = zdf[["가격영역", "최종가격(원)"]].copy()
    g["ord"] = g["가격영역"].map(order)
    g = g.sort_values("ord")
    viol = []
    prev_p = prev_z = None
    for _, r in g.iterrows():
        p = safe_float(r["최종가격(원)"], np.nan)
        if p != p:
            continue
        if prev_p is None:
            prev_p = p; prev_z = r["가격영역"]; continue
        need = max(prev_p*(1+gap_pct), prev_p + gap_won)
        if p < need:
            viol.append({"하위영역": prev_z, "하위가격": prev_p,
                          "상위영역": r["가격영역"], "상위가격": p,
                          "필요최소상위가": need, "갭부족": need - p})
        prev_p = p; prev_z = r["가격영역"]
    return pd.DataFrame(viol)


# ────────────────────────────────────────────────────────────
# 세트 헬퍼
# ────────────────────────────────────────────────────────────
def compute_set_cost(set_id, bom_df, products_df):
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
    b["ref_value"]  = b["상시가_ref"] * b["수량"]
    total_ref       = float(b["ref_value"].sum())
    b["w"]          = (b["ref_value"] / total_ref) if total_ref > 0 else (1.0 / len(b))
    b["배분매출"]    = set_price * b["w"]
    b["실질단가"]    = b["배분매출"] / b["수량"].replace(0, np.nan)
    b["상시대비할인율(%)"] = np.where(b["상시가_ref"]>0, (1.0 - (b["실질단가"] / b["상시가_ref"])) * 100.0, np.nan)
    b["세트ID"]     = set_id
    b["가격영역"]   = zone
    return b[["세트ID","가격영역","품번","상품명","수량","상시가_ref","실질단가","상시대비할인율(%)"]]


# ────────────────────────────────────────────────────────────
# 세션 초기화
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# 헬퍼 컴포넌트
# ────────────────────────────────────────────────────────────
def section_header(icon, title, subtitle="", color="indigo"):
    st.markdown(f"""
    <div class="section-header">
      <div class="section-icon icon-{color}">{icon}</div>
      <div>
        <h3>{title}</h3>
        {"" if not subtitle else f'<p class="sub">{subtitle}</p>'}
      </div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(label, value, sub="", color="indigo"):
    return f"""
    <div class="kpi-card kpi-{color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {"" if not sub else f'<div class="kpi-sub">{sub}</div>'}
    </div>
    """

def render_kpis(cards):
    html = '<div class="kpi-grid">'
    for c in cards:
        html += kpi_card(*c)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def info_box(msg):
    st.markdown(f'<div class="info-box"><p>ℹ️&nbsp;&nbsp;{msg}</p></div>', unsafe_allow_html=True)

def warn_box(msg):
    st.markdown(f'<div class="warn-box"><p>⚠️&nbsp;&nbsp;{msg}</p></div>', unsafe_allow_html=True)

def success_box(msg):
    st.markdown(f'<div class="success-box"><p>✅&nbsp;&nbsp;{msg}</p></div>', unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def build_price_band_chart(zdf, cost, min_price, max_price):
    """가격 밴드 시각화 — 가로 막대 차트"""
    if zdf.empty:
        return None

    zones  = zdf["가격영역"].tolist()
    finals = zdf["최종가격(원)"].tolist()
    floors = zdf["Floor(손익하한)"].tolist()
    band_h = zdf["BandHigh"].tolist()
    band_l = zdf["BandLow"].tolist()
    cmrs   = zdf["기여이익률(%)"].tolist()

    colors = [ZONE_COLORS.get(z, "#6366f1") for z in zones]

    fig = go.Figure()

    # 밴드 범위 (BandLow ~ BandHigh) 배경
    for i, z in enumerate(zones):
        fig.add_shape(
            type="rect",
            x0=band_l[i], x1=band_h[i],
            y0=i - 0.35, y1=i + 0.35,
            fillcolor="rgba(226,232,240,0.5)",
            line=dict(width=0),
            layer="below"
        )

    # Floor 선
    for i, f in enumerate(floors):
        if f and not np.isnan(f):
            fig.add_shape(
                type="line",
                x0=f, x1=f,
                y0=i - 0.38, y1=i + 0.38,
                line=dict(color="rgba(239,68,68,0.8)", width=2, dash="dot")
            )

    # 최종가 막대
    fig.add_trace(go.Bar(
        x=finals,
        y=zones,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color="white", width=1.5)
        ),
        text=[f"{fmt_krw(v)}<br><span style='font-size:10px'>{fmt_pct(c)}</span>" if v and not np.isnan(v) else ""
              for v, c in zip(finals, cmrs)],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=11, color="white", family="Noto Sans KR"),
        hovertemplate="<b>%{y}</b><br>최종가: %{x:,.0f}원<extra></extra>",
        name="최종가격",
        width=0.65,
    ))

    # 원가 라인
    fig.add_vline(x=cost, line_color="rgba(100,116,139,0.6)", line_width=1.5,
                  line_dash="longdash",
                  annotation_text=f"원가 {fmt_krw(cost)}", annotation_position="top",
                  annotation_font=dict(size=10, color="#64748b"))

    fig.update_layout(
        height=max(320, len(zones) * 42 + 80),
        margin=dict(l=10, r=30, t=30, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="가격 (원)",
            tickformat=",d",
            gridcolor="#f1f5f9",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(zones)),
            tickfont=dict(size=12, family="Noto Sans KR"),
            showgrid=False,
        ),
        showlegend=False,
        bargap=0.2,
        font=dict(family="Noto Sans KR"),
    )
    return fig


def build_margin_chart(zdf):
    """채널별 기여이익률 도넛/막대"""
    if zdf.empty:
        return None
    sub = zdf[zdf["기여이익률(%)"].notna()].copy()
    if sub.empty:
        return None
    colors = [ZONE_COLORS.get(z, "#6366f1") for z in sub["가격영역"]]
    fig = go.Figure(go.Bar(
        x=sub["가격영역"],
        y=sub["기여이익률(%)"],
        marker=dict(color=colors, opacity=0.85, line=dict(color="white", width=1)),
        text=[f"{v:.1f}%" for v in sub["기여이익률(%)"]],
        textposition="outside",
        textfont=dict(size=11, family="Noto Sans KR"),
        hovertemplate="<b>%{x}</b><br>기여이익률: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(239,68,68,0.6)", line_width=1.5)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickfont=dict(size=11, family="Noto Sans KR"), showgrid=False),
        yaxis=dict(title="기여이익률(%)", gridcolor="#f1f5f9", showgrid=True),
        showlegend=False,
        font=dict(family="Noto Sans KR"),
    )
    return fig


# ────────────────────────────────────────────────────────────
# 앱 헤더
# ────────────────────────────────────────────────────────────
prod_count = len(st.session_state["products_df"])
uploaded   = prod_count > 0

st.markdown(f"""
<div class="app-header">
  <h1>💰 IBR 가격 시뮬레이터 <span style="font-size:1rem;font-weight:500;color:#6366f1;">v6.2</span></h1>
  <p>원가 파일만 업로드하면 → 채널별 가격 자동 산출 · Min/Max 조정 · 마진 즉시 확인 · 카니발 경고</p>
  <div class="badge-row">
    <span class="badge badge-indigo">📦 {"SKU " + str(prod_count) + "개 로드됨" if uploaded else "파일 미업로드"}</span>
    <span class="badge badge-green">📡 채널 {len(st.session_state["channels_df"])}개 설정</span>
    <span class="badge badge-amber">🎯 가격영역 {len(PRICE_ZONES)}개</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# 탭
# ────────────────────────────────────────────────────────────
tab_up, tab_sku, tab_set, tab_plan, tab_logic = st.tabs([
    "⚙️ 설정 · 업로드",
    "📦 단품(SKU) 시뮬레이터",
    "🗂️ 세트(BOM)",
    "📋 운영플랜 · 카니발",
    "📖 로직 안내"
])


# ════════════════════════════════════════════════════════════
# 탭 1 : 설정 · 업로드
# ════════════════════════════════════════════════════════════
with tab_up:

    # ── STEP 가이드 (상단 요약) ──
    step1_done = uploaded
    step2_done = True  # 채널은 기본값 있음
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px;">
      <div class="step-card {'done' if step1_done else 'active'}">
        <div class="step-num {'done-num' if step1_done else ''}">{'✓' if step1_done else '1'}</div>
        <h4>📁 원가 파일 업로드</h4>
        <p>상품코드 · 상품명 · 원가(VAT-) 컬럼이 있는 .xlsx 파일</p>
      </div>
      <div class="step-card {'done' if step2_done else ''}">
        <div class="step-num {'done-num' if step2_done else ''}">2</div>
        <h4>📡 채널 비용 확인</h4>
        <p>수수료 · PG · 배송비 · 마케팅비 · 반품률 입력(기본값 제공)</p>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <h4>🎯 단품 탭으로 이동</h4>
        <p>SKU 선택 → 자동 가격 확인 → Min/Max 조정 → 마진 확인</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── A. 파일 업로드 ──
    section_header("📁", "원가 · 상품마스터 파일 업로드", "상품코드 / 상품명 / 원가(VAT-) 컬럼이 포함된 엑셀 파일", "indigo")

    col_up, col_info = st.columns([3, 2])
    with col_up:
        up = st.file_uploader(
            "엑셀 파일을 드래그하거나 클릭해서 선택하세요 (.xlsx / .xls)",
            type=["xlsx","xls"],
            label_visibility="collapsed"
        )
        if up is not None:
            try:
                new_df = load_products_from_cost_master(up)
                st.session_state["products_df"] = new_df
                success_box(f"업로드 완료 — {len(new_df):,}개 SKU가 로드되었습니다.")
            except Exception as e:
                st.error(f"파일 읽기 오류: {e}")

    with col_info:
        prod = st.session_state["products_df"]
        if not prod.empty:
            valid_cost = prod["원가"].notna().sum()
            render_kpis([
                ("총 SKU 수",    f"{len(prod):,}",       "로드된 전체 품번", "indigo"),
                ("원가 보유",    f"{valid_cost:,}",       "원가 값 있는 SKU", "green"),
                ("원가 누락",    f"{len(prod)-valid_cost:,}", "원가 없는 SKU", "amber"),
            ])
        else:
            info_box("파일을 업로드하면 SKU 통계가 여기에 표시됩니다.")

    divider()

    # ── B. 채널 비용 ──
    section_header("📡", "채널별 비용 설정", "수수료 · PG · 배송비 · 마케팅비 · 반품률을 채널별로 입력하세요", "blue")
    info_box("값을 수정하면 모든 가격 계산에 즉시 반영됩니다. 비율은 0~1 사이 소수(예: 5% → 0.05)로 입력하세요.")

    st.session_state["channels_df"] = st.data_editor(
        st.session_state["channels_df"],
        use_container_width=True,
        num_rows="dynamic",
        height=290,
        column_config={
            "채널명":        st.column_config.TextColumn("채널명", width="medium"),
            "수수료율":      st.column_config.NumberColumn("수수료율", format="%.2f", min_value=0.0, max_value=1.0),
            "PG":            st.column_config.NumberColumn("PG 수수료", format="%.2f", min_value=0.0, max_value=0.1),
            "배송비(주문당)":st.column_config.NumberColumn("배송비(원)", format="%d", min_value=0),
            "마케팅비":      st.column_config.NumberColumn("마케팅비율", format="%.2f", min_value=0.0, max_value=1.0),
            "반품률":        st.column_config.NumberColumn("반품률", format="%.2f", min_value=0.0, max_value=1.0),
            "반품비(주문당)":st.column_config.NumberColumn("반품비(원)", format="%d", min_value=0),
        }
    )

    divider()

    # ── C. 가격영역 ↔ 채널 매핑 ──
    section_header("🗺️", "가격영역 ↔ 비용채널 매핑", "각 가격영역에 적용할 채널 비용을 선택합니다", "amber")

    zone_map     = st.session_state["zone_map"].copy()
    channel_names = st.session_state["channels_df"]["채널명"].dropna().astype(str).tolist()

    cols = st.columns(5)
    for i, z in enumerate(PRICE_ZONES):
        with cols[i % 5]:
            color_dot = ZONE_COLORS.get(z, "#6366f1")
            st.markdown(f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;"><div style="width:10px;height:10px;border-radius:50%;background:{color_dot};"></div><span style="font-size:0.8rem;font-weight:600;color:#1e293b;">{z}</span></div>', unsafe_allow_html=True)
            zone_map[z] = st.selectbox(
                z, options=channel_names,
                index=channel_names.index(zone_map.get(z, channel_names[0])) if zone_map.get(z) in channel_names else 0,
                key=f"zmap_{z}",
                label_visibility="collapsed"
            )
    st.session_state["zone_map"] = zone_map

    divider()

    # ── D. 밴드 경계 ──
    section_header("📊", "가격밴드 경계 설정", "Min~Max 레인지를 10개 영역으로 나누는 경계값을 조정합니다 (0 ~ 100%)", "green")

    b     = st.session_state["boundaries"].copy()
    prev  = 0
    new_b = [0]

    col_left, col_right = st.columns([3, 2])
    with col_left:
        for idx in range(1, 10):
            lz   = PRICE_ZONES[idx-1]
            rz   = PRICE_ZONES[idx]
            minv = prev + 1
            maxv = 100 - (10 - idx)
            val  = max(minv, min(maxv, int(b[idx])))
            lc   = ZONE_COLORS.get(lz, "#94a3b8")
            rc   = ZONE_COLORS.get(rz, "#6366f1")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:-4px;">'
                f'<span style="font-size:0.75rem;font-weight:600;color:{lc};min-width:56px;text-align:right;">{lz}</span>'
                f'<span style="font-size:0.7rem;color:#94a3b8;">│</span>'
                f'<span style="font-size:0.75rem;font-weight:600;color:{rc};min-width:56px;">{rz}</span>'
                f'</div>', unsafe_allow_html=True
            )
            val = st.slider(
                f"경계 {idx}", min_value=minv, max_value=maxv, value=val,
                step=1, key=f"b_{idx}", label_visibility="collapsed"
            )
            new_b.append(val)
            prev = val
        new_b.append(100)
        st.session_state["boundaries"] = new_b

    with col_right:
        # 밴드 시각화 (미니 막대)
        widths = [new_b[i+1]-new_b[i] for i in range(len(PRICE_ZONES))]
        fig_b  = go.Figure()
        for i, z in enumerate(PRICE_ZONES):
            fig_b.add_trace(go.Bar(
                name=z, x=[widths[i]], y=["밴드"],
                orientation="h",
                marker=dict(color=ZONE_COLORS.get(z, "#6366f1"), opacity=0.85,
                            line=dict(color="white", width=1.5)),
                text=f"{z}<br>{widths[i]}%",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=9, color="white"),
                hovertemplate=f"<b>{z}</b><br>폭: {widths[i]}%<extra></extra>",
            ))
        fig_b.update_layout(
            barmode="stack", height=90,
            margin=dict(l=0, r=0, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False, range=[0, 100]),
            yaxis=dict(showticklabels=False, showgrid=False),
        )
        st.markdown("<p style='font-size:0.78rem;font-weight:600;color:#475569;margin:0 0 6px;'>밴드 분포 미리보기</p>", unsafe_allow_html=True)
        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

        # 수치 목록
        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:#475569;margin:10px 0 4px;'>각 영역 범위</p>", unsafe_allow_html=True)
        for i, z in enumerate(PRICE_ZONES):
            color_dot = ZONE_COLORS.get(z, "#6366f1")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;">'
                f'<span style="display:flex;align-items:center;gap:5px;font-size:0.76rem;">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{color_dot};display:inline-block;"></span>'
                f'{z}</span>'
                f'<span style="font-size:0.76rem;color:#6366f1;font-weight:600;">{new_b[i]}% ~ {new_b[i+1]}%</span>'
                f'</div>', unsafe_allow_html=True
            )

    with st.expander("🎯 각 영역 내 Target 위치 세부 조정", expanded=False):
        tp   = st.session_state["target_pos"].copy()
        cols = st.columns(5)
        for i, z in enumerate(PRICE_ZONES):
            s   = new_b[i];  e = new_b[i+1]
            mid = int(round((s+e)/2))
            with cols[i % 5]:
                st.markdown(f'<p style="font-size:0.78rem;font-weight:600;color:{ZONE_COLORS.get(z,"#6366f1")};margin:0 0 2px;">{z}</p>', unsafe_allow_html=True)
                tp[z] = st.slider(z, min_value=int(s), max_value=int(e),
                                   value=int(tp.get(z, mid)), step=1,
                                   key=f"tp_{z}", label_visibility="collapsed")
        st.session_state["target_pos"] = tp

    divider()

    # ── E. SKU 오버라이드 ──
    section_header("✏️", "SKU별 MSRP / Min / Max 수동 지정 (선택)", "기본은 원가에서 자동 산출합니다. 특정 SKU에 고정값이 필요할 때만 입력하세요.", "red")

    if st.session_state["products_df"].empty:
        info_box("원가 파일을 먼저 업로드하면 SKU 목록이 표시됩니다.")
    else:
        st.session_state["products_df"] = st.data_editor(
            st.session_state["products_df"],
            use_container_width=True,
            height=280,
            num_rows="dynamic",
            column_config={
                "품번":           st.column_config.TextColumn("품번", width="small"),
                "상품명":          st.column_config.TextColumn("상품명", width="large"),
                "브랜드":          st.column_config.TextColumn("브랜드", width="small"),
                "원가":            st.column_config.NumberColumn("원가(원)", format="%d"),
                "MSRP_오버라이드": st.column_config.NumberColumn("MSRP 고정값", format="%d"),
                "Min_오버라이드":  st.column_config.NumberColumn("Min 고정값", format="%d"),
                "Max_오버라이드":  st.column_config.NumberColumn("Max 고정값", format="%d"),
                "운영여부":        st.column_config.CheckboxColumn("운영"),
            }
        )


# ════════════════════════════════════════════════════════════
# 탭 2 : 단품(SKU) 시뮬레이터
# ════════════════════════════════════════════════════════════
with tab_sku:
    prod = st.session_state["products_df"].copy()

    if prod.empty:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:3rem;margin-bottom:16px;">📂</div>
          <h3 style="color:#1e293b;margin-bottom:8px;">파일을 먼저 업로드해 주세요</h3>
          <p style="color:#64748b;font-size:0.88rem;">설정·업로드 탭에서 원가 파일을 업로드하면 자동으로 가격이 산출됩니다.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── 파라미터 설정 행 ──
        section_header("⚙️", "시뮬레이션 파라미터", "", "indigo")

        pc1, pc2, pc3, pc4, pc5 = st.columns([1, 1, 1, 1, 1])
        with pc1:
            rounding_unit = st.selectbox("반올림 단위", [10, 100, 1000], index=2,
                                          help="가격 계산 후 반올림 단위")
        with pc2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한", min_value=0.05, max_value=0.80,
                                              value=0.30, step=0.01, format="%.2f",
                                              help="원가 ÷ 이 값 = MSRP 기준 (예: 30% → 원가의 3.3배)")
        with pc3:
            min_cm = st.slider("최소 기여이익률 (%)", 0, 50, 15, 1,
                                help="Floor 계산 시 보장할 최소 기여이익률") / 100.0
        with pc4:
            gap_pct = st.slider("서열 갭 (%)", 0, 20, 5, 1,
                                 help="가격영역 서열 위반 감지 기준 (%)") / 100.0
        with pc5:
            gap_won = st.number_input("서열 갭 (원)", min_value=0, value=2000, step=500,
                                       help="가격영역 서열 위반 감지 기준 (원)")

        divider()

        # ── SKU 선택 ──
        section_header("📦", "SKU 선택", "", "blue")

        options = (prod["품번"].astype(str) + "  |  " + prod["상품명"].astype(str)).tolist()

        sc1, sc2 = st.columns([3, 1])
        with sc1:
            picked = st.selectbox("분석할 SKU를 선택하세요", options, index=0,
                                   label_visibility="collapsed")
        with sc2:
            min_zone = st.selectbox("자동 Min 기준 존", PRICE_ZONES, index=0,
                                     help="어떤 가격영역의 Floor를 최저가(Min)로 사용할지 선택")

        sku  = picked.split("  |  ", 1)[0].strip()
        row  = prod[prod["품번"].astype(str) == sku].iloc[0]
        cost = safe_float(row["원가"], np.nan)

        # SKU 정보 카드
        st.markdown(f"""
        <div style="background:#f8fafc;border:1.5px solid #e2e8f0;border-radius:12px;padding:14px 18px;margin:10px 0 20px;">
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div><span style="font-size:0.72rem;color:#64748b;font-weight:600;">품번</span><br>
                 <span style="font-size:0.95rem;font-weight:700;color:#1e293b;">{row.get('품번','')}</span></div>
            <div style="color:#e2e8f0;">│</div>
            <div><span style="font-size:0.72rem;color:#64748b;font-weight:600;">상품명</span><br>
                 <span style="font-size:0.95rem;font-weight:700;color:#1e293b;">{row.get('상품명','')}</span></div>
            <div style="color:#e2e8f0;">│</div>
            <div><span style="font-size:0.72rem;color:#64748b;font-weight:600;">브랜드</span><br>
                 <span style="font-size:0.88rem;font-weight:600;color:#475569;">{row.get('브랜드','')}</span></div>
            <div style="color:#e2e8f0;">│</div>
            <div><span style="font-size:0.72rem;color:#64748b;font-weight:600;">원가 (VAT-)</span><br>
                 <span style="font-size:1.05rem;font-weight:700;color:#6366f1;">{fmt_krw(cost) if cost==cost else "미입력"}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if cost != cost or cost <= 0:
            st.error("⛔ 원가가 비어 있거나 0 이하입니다. 업로드 파일의 '원가(vat-)' 컬럼을 확인하세요.")
        else:
            min_auto, max_auto, meta = compute_auto_range_from_cost(
                cost_total=cost,
                channels_df=st.session_state["channels_df"],
                zone_map=st.session_state["zone_map"],
                boundaries=st.session_state["boundaries"],
                rounding_unit=rounding_unit,
                min_cm=min_cm,
                max_cost_ratio=max_cost_ratio,
                include_zones=PRICE_ZONES,
                min_zone=min_zone,
                msrp_override=safe_float(row.get("MSRP_오버라이드", np.nan), np.nan),
            )

            if meta.get("note"):
                info_box(meta["note"])

            # ── Min / Max 조정 ──
            section_header("🎚️", "Min / Max 가격 조정", "Min/Max를 바꾸면 모든 가격영역이 함께 이동합니다", "green")

            mc1, mc2, mc3 = st.columns([1, 1, 2])
            with mc1:
                min_user = st.number_input(
                    "Min — 최저가 (원)", min_value=0, value=int(min_auto),
                    step=rounding_unit, help="공구 등 최저 가격영역 기준선"
                )
            with mc2:
                max_user = st.number_input(
                    "Max — 최고가 / MSRP (원)", min_value=0, value=int(max_auto),
                    step=rounding_unit, help="정가(MSRP) 기준선"
                )
            with mc3:
                cost_ratio = cost / max_user if max_user > 0 else 0
                span_val   = max_user - min_user
                render_kpis([
                    ("레인지 폭", fmt_krw(span_val), f"Max - Min", "indigo"),
                    ("원가율(Max 기준)", fmt_pct(cost_ratio*100), f"원가/MSRP", "green" if cost_ratio <= max_cost_ratio else "red"),
                ])

            if max_user <= min_user:
                warn_box("Max가 Min 이하입니다. 밴드가 펼쳐지지 않으므로 Max 값을 높여주세요.")
                max_user = min_user + max(rounding_unit*10, int(min_user*0.15))

            # ── 존 테이블 빌드 ──
            section_header("✏️", "채널별 가격 직접 수정 (오버라이드)", "추천가를 수정해도 경고만 표시되며 저장됩니다", "amber")

            zdf_init = build_zone_table(
                cost_total=cost, min_price=float(min_user), max_price=float(max_user),
                channels_df=st.session_state["channels_df"], zone_map=st.session_state["zone_map"],
                boundaries=st.session_state["boundaries"], target_pos=st.session_state["target_pos"],
                rounding_unit=rounding_unit, min_cm=min_cm,
                overrides_df=st.session_state["overrides_df"], item_type="SKU", item_id=sku,
            )

            if zdf_init.empty:
                info_box("결과가 없습니다. 파라미터를 확인하세요.")
            else:
                info_box("'가격_오버라이드(원)' 컬럼에 원하는 가격을 직접 입력하면 해당 가격이 최종가로 적용됩니다. Floor 미만이어도 경고만 표시됩니다.")

                edit_cols  = ["가격영역", "가격_오버라이드(원)"]
                edit_view  = zdf_init[edit_cols].copy()
                edit_view2 = st.data_editor(
                    edit_view, use_container_width=True, height=290, num_rows="fixed",
                    key="sku_override_editor",
                    column_config={
                        "가격영역":          st.column_config.TextColumn("가격영역", disabled=True),
                        "가격_오버라이드(원)": st.column_config.NumberColumn("직접 입력 가격 (원)", format="%d"),
                    }
                )

                # 오버라이드 반영
                ov = st.session_state["overrides_df"].copy()
                ov = ov[~((ov["오퍼타입"]=="SKU") & (ov["오퍼ID"]==sku))].copy()
                for _, rr in edit_view2.iterrows():
                    p = safe_float(rr.get("가격_오버라이드(원)"), np.nan)
                    if p == p and p > 0:
                        ov = pd.concat([ov, pd.DataFrame([{
                            "오퍼타입":"SKU","오퍼ID":sku,
                            "가격영역":rr["가격영역"],"가격_오버라이드":float(p)
                        }])], ignore_index=True)
                st.session_state["overrides_df"] = ov

                zdf = build_zone_table(
                    cost_total=cost, min_price=float(min_user), max_price=float(max_user),
                    channels_df=st.session_state["channels_df"], zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"], target_pos=st.session_state["target_pos"],
                    rounding_unit=rounding_unit, min_cm=min_cm,
                    overrides_df=st.session_state["overrides_df"], item_type="SKU", item_id=sku,
                )

                # ── 결과 시각화 ──
                section_header("📊", "가격 시뮬레이션 결과", "", "indigo")

                # KPI 요약
                avg_final = zdf["최종가격(원)"].mean()
                avg_cmr   = zdf["기여이익률(%)"].mean()
                warn_cnt  = (zdf["경고"] != "").sum()
                render_kpis([
                    ("원가",         fmt_krw(cost),                 "VAT 미포함",         "blue"),
                    ("Min (자동)",   fmt_krw(min_auto),             "공구 Floor 기준",     "green"),
                    ("Max / MSRP",  fmt_krw(max_user),             "정가 기준선",          "indigo"),
                    ("평균 최종가",  fmt_krw(avg_final) if avg_final==avg_final else "—", "10개 영역 평균", "amber"),
                    ("평균 기여이익률", fmt_pct(avg_cmr) if avg_cmr==avg_cmr else "—", "10개 영역 평균", "green" if (avg_cmr==avg_cmr and avg_cmr>=min_cm*100) else "red"),
                    ("경고 항목",    str(warn_cnt),                 "Floor/BandHigh 위반", "red" if warn_cnt>0 else "green"),
                ])

                # 차트
                ch1, ch2 = st.columns([3, 2])
                with ch1:
                    fig_band = build_price_band_chart(zdf, cost, float(min_user), float(max_user))
                    if fig_band:
                        st.markdown("""
                        <div class="price-band-container">
                          <div class="price-band-title">채널별 최종가격 & 밴드 범위</div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="legend-row">
                          <div class="legend-item"><div class="legend-dot" style="background:#e2e8f0;border-radius:2px;width:14px;height:8px;"></div>밴드 범위</div>
                          <div class="legend-item"><div style="width:14px;border-top:2px dashed rgba(239,68,68,0.7);"></div>Floor(손익하한)</div>
                          <div class="legend-item"><div style="width:14px;border-top:2px dashed #94a3b8;"></div>원가</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(fig_band, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)
                with ch2:
                    fig_cm = build_margin_chart(zdf)
                    if fig_cm:
                        st.markdown("""
                        <div class="price-band-container">
                          <div class="price-band-title">채널별 기여이익률 (%)</div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)

                # 상세 테이블
                with st.expander("📋 전체 가격 테이블 상세 보기", expanded=False):
                    # 경고 컬럼 강조
                    display_df = zdf.copy()
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=360,
                        column_config={
                            "가격영역":         st.column_config.TextColumn("가격영역"),
                            "비용채널":         st.column_config.TextColumn("비용채널"),
                            "BandLow":          st.column_config.NumberColumn("밴드 하한", format="%d"),
                            "BandHigh":         st.column_config.NumberColumn("밴드 상한", format="%d"),
                            "Floor(손익하한)":  st.column_config.NumberColumn("Floor(원)", format="%d"),
                            "추천가(Target)":   st.column_config.NumberColumn("추천가(원)", format="%d"),
                            "최종가격(원)":     st.column_config.NumberColumn("최종가(원)", format="%d"),
                            "기여이익(원)":     st.column_config.NumberColumn("기여이익(원)", format="%d"),
                            "기여이익률(%)":    st.column_config.NumberColumn("기여이익률(%)", format="%.1f"),
                            "마진룸(원)":       st.column_config.NumberColumn("마진룸(원)", format="%d"),
                        }
                    )

                # 서열/갭 위반
                viol = check_order_violations(zdf, gap_pct=gap_pct, gap_won=gap_won)
                if not viol.empty:
                    warn_box(f"{len(viol)}건의 서열·갭 위반이 감지되었습니다. 카니발(가격역전) 위험이 있으므로 확인 후 조정하세요.")
                    st.dataframe(viol, use_container_width=True, height=200)
                else:
                    success_box("서열·갭 위반 없음 — 현재 설정에서 가격영역 순서가 올바릅니다.")

                # 다운로드
                xb = to_excel_bytes({"sku_result": zdf, "order_violations": viol})
                st.download_button(
                    "⬇️  이 SKU 결과 엑셀 다운로드", xb,
                    file_name=f"{sku}_pricing_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )


# ════════════════════════════════════════════════════════════
# 탭 3 : 세트(BOM)
# ════════════════════════════════════════════════════════════
with tab_set:
    prod = st.session_state["products_df"].copy()

    if prod.empty:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:3rem;margin-bottom:16px;">📂</div>
          <h3 style="color:#1e293b;margin-bottom:8px;">파일을 먼저 업로드해 주세요</h3>
          <p style="color:#64748b;font-size:0.88rem;">설정·업로드 탭에서 원가 파일을 업로드하면 세트를 구성할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── 세트 생성 ──
        section_header("➕", "세트 생성", "세트 ID와 이름을 입력해 세트를 추가합니다", "indigo")

        sc1, sc2, sc3 = st.columns([2, 3, 1])
        with sc1:
            new_id = st.text_input("세트 ID", placeholder="예: SET-001", label_visibility="collapsed")
        with sc2:
            new_name = st.text_input("세트명", placeholder="예: 기초케어 세트 3종", label_visibility="collapsed")
        with sc3:
            if st.button("세트 추가", type="primary", disabled=(not new_id.strip() or not new_name.strip())):
                sets = st.session_state["sets_df"].copy()
                if (sets["세트ID"] == new_id.strip()).any():
                    warn_box("이미 존재하는 세트 ID입니다.")
                else:
                    sets = pd.concat([sets, pd.DataFrame([{"세트ID": new_id.strip(), "세트명": new_name.strip(), "MSRP_오버라이드": np.nan}])], ignore_index=True)
                    st.session_state["sets_df"] = sets
                    success_box(f"세트 '{new_name.strip()}'이 추가되었습니다.")

        if st.session_state["sets_df"].empty:
            info_box("아직 세트가 없습니다. 위에서 세트를 추가해 주세요.")
        else:
            st.session_state["sets_df"] = st.data_editor(
                st.session_state["sets_df"],
                use_container_width=True, height=140, num_rows="dynamic",
                column_config={
                    "세트ID":          st.column_config.TextColumn("세트 ID"),
                    "세트명":           st.column_config.TextColumn("세트명", width="large"),
                    "MSRP_오버라이드": st.column_config.NumberColumn("MSRP 고정값", format="%d"),
                }
            )

            divider()

            # ── 세트 선택 & BOM 편집 ──
            set_opts = (st.session_state["sets_df"]["세트ID"].astype(str) + "  |  " + st.session_state["sets_df"]["세트명"].astype(str)).tolist()
            picked   = st.selectbox("편집할 세트 선택", set_opts, index=0, label_visibility="collapsed")
            set_id   = picked.split("  |  ", 1)[0].strip()

            section_header("🧩", "BOM (구성품) 추가", "이 세트에 포함할 단품 SKU와 수량을 추가합니다", "blue")

            sku_opts = (prod["품번"].astype(str) + "  |  " + prod["상품명"].astype(str)).tolist()
            b1, b2, b3 = st.columns([4, 1, 1])
            with b1:
                sku_pick = st.selectbox("구성품 SKU", sku_opts, index=0, key=f"bom_sku_{set_id}",
                                         label_visibility="collapsed")
                bom_sku  = sku_pick.split("  |  ", 1)[0].strip()
            with b2:
                qty = st.number_input("수량", min_value=1, value=1, step=1, key=f"bom_qty_{set_id}",
                                       label_visibility="collapsed")
            with b3:
                if st.button("구성품 추가", key=f"bom_add_{set_id}", type="primary"):
                    bom = st.session_state["bom_df"].copy()
                    bom = pd.concat([bom, pd.DataFrame([{"세트ID": set_id, "품번": bom_sku, "수량": int(qty)}])], ignore_index=True)
                    st.session_state["bom_df"] = bom
                    success_box("구성품이 추가되었습니다.")

            bom_view = st.session_state["bom_df"][st.session_state["bom_df"]["세트ID"] == set_id].copy()

            if bom_view.empty:
                info_box("BOM이 비어 있습니다. 구성품을 추가해 주세요.")
            else:
                bom_view_merged = bom_view.merge(prod[["품번","상품명","원가"]], on="품번", how="left")
                bom_view_merged["소계(원)"] = (bom_view_merged["원가"] * bom_view_merged["수량"]).round(0).astype("Int64")
                total_cost_bom = bom_view_merged["소계(원)"].sum()
                st.dataframe(
                    bom_view_merged,
                    use_container_width=True, height=200,
                    column_config={
                        "원가":    st.column_config.NumberColumn("단품 원가(원)", format="%d"),
                        "소계(원)": st.column_config.NumberColumn("소계(원)", format="%d"),
                    }
                )
                st.markdown(f'<p style="text-align:right;font-size:0.85rem;font-weight:700;color:#6366f1;">세트 원가 합계: {fmt_krw(total_cost_bom)}</p>', unsafe_allow_html=True)

                if st.button("🗑️ 이 세트 BOM 전체 삭제", type="secondary", key=f"bom_clear_{set_id}"):
                    bom = st.session_state["bom_df"].copy()
                    bom = bom[bom["세트ID"] != set_id].copy()
                    st.session_state["bom_df"] = bom
                    success_box("BOM이 삭제되었습니다.")

            divider()

            # ── 세트 가격 시뮬레이션 ──
            section_header("📊", "세트 가격 시뮬레이션", "", "green")

            sp1, sp2, sp3, sp4 = st.columns([1, 1, 1, 1])
            with sp1:
                rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="set_round")
            with sp2:
                max_cost_ratio = st.number_input("MSRP 원가율 상한", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="set_ratio")
            with sp3:
                min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="set_cm") / 100.0
            with sp4:
                gap_pct = st.slider("서열 갭(%)", 0, 20, 5, 1, key="set_gap") / 100.0
            gap_won = st.number_input("서열 갭(원)", min_value=0, value=2000, step=500, key="set_gap_won")

            if bom_view.empty:
                info_box("BOM을 구성하면 자동 가격이 산출됩니다.")
            else:
                cost_total = compute_set_cost(set_id, st.session_state["bom_df"], prod)
                srow = st.session_state["sets_df"][st.session_state["sets_df"]["세트ID"] == set_id].iloc[0]

                min_auto, max_auto, meta = compute_auto_range_from_cost(
                    cost_total=cost_total,
                    channels_df=st.session_state["channels_df"],
                    zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"],
                    rounding_unit=rounding_unit,
                    min_cm=min_cm, max_cost_ratio=max_cost_ratio,
                    include_zones=PRICE_ZONES, min_zone="공구",
                    msrp_override=safe_float(srow.get("MSRP_오버라이드", np.nan), np.nan),
                )

                if meta.get("note"):
                    info_box(meta["note"])

                render_kpis([
                    ("세트 원가 합계", fmt_krw(cost_total), "구성품 합산", "blue"),
                    ("자동 Min",       fmt_krw(min_auto),   "공구 Floor 기준", "green"),
                    ("자동 Max",       fmt_krw(max_auto),   "MSRP 기준선", "indigo"),
                ])

                sm1, sm2, sm3 = st.columns([1, 1, 2])
                with sm1:
                    min_user = st.number_input("Min 수정(세트)", min_value=0, value=int(min_auto), step=rounding_unit, key="set_min_user")
                with sm2:
                    max_user = st.number_input("Max 수정(세트)", min_value=0, value=int(max_auto), step=rounding_unit, key="set_max_user")
                with sm3:
                    st.caption("Min/Max를 바꾸면 모든 가격영역이 함께 이동합니다.")

                if max_user <= min_user:
                    max_user = min_user + max(rounding_unit*10, int(min_user*0.15))

                zdf = build_zone_table(
                    cost_total=cost_total, min_price=float(min_user), max_price=float(max_user),
                    channels_df=st.session_state["channels_df"], zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"], target_pos=st.session_state["target_pos"],
                    rounding_unit=rounding_unit, min_cm=min_cm,
                    overrides_df=st.session_state["overrides_df"], item_type="SET", item_id=set_id,
                )

                section_header("✏️", "채널별 가격 오버라이드 (세트)", "", "amber")
                edit_view = zdf[["가격영역","가격_오버라이드(원)"]].copy()
                edit_view2 = st.data_editor(
                    edit_view, use_container_width=True, height=270, num_rows="fixed",
                    key="set_override_editor",
                    column_config={
                        "가격영역":          st.column_config.TextColumn("가격영역", disabled=True),
                        "가격_오버라이드(원)": st.column_config.NumberColumn("직접 입력 가격 (원)", format="%d"),
                    }
                )

                ov = st.session_state["overrides_df"].copy()
                ov = ov[~((ov["오퍼타입"]=="SET") & (ov["오퍼ID"]==set_id))].copy()
                for _, rr in edit_view2.iterrows():
                    p = safe_float(rr.get("가격_오버라이드(원)"), np.nan)
                    if p == p and p > 0:
                        ov = pd.concat([ov, pd.DataFrame([{"오퍼타입":"SET","오퍼ID":set_id,"가격영역":rr["가격영역"],"가격_오버라이드":float(p)}])], ignore_index=True)
                st.session_state["overrides_df"] = ov

                zdf = build_zone_table(
                    cost_total=cost_total, min_price=float(min_user), max_price=float(max_user),
                    channels_df=st.session_state["channels_df"], zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"], target_pos=st.session_state["target_pos"],
                    rounding_unit=rounding_unit, min_cm=min_cm,
                    overrides_df=st.session_state["overrides_df"], item_type="SET", item_id=set_id,
                )

                # 차트
                fig_s = build_price_band_chart(zdf, cost_total, float(min_user), float(max_user))
                if fig_s:
                    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

                with st.expander("📋 세트 가격 테이블 상세", expanded=False):
                    st.dataframe(zdf, use_container_width=True, height=340)

                viol = check_order_violations(zdf, gap_pct=gap_pct, gap_won=gap_won)
                if not viol.empty:
                    warn_box(f"{len(viol)}건 서열·갭 위반")
                    st.dataframe(viol, use_container_width=True, height=200)

                # 구성품 배분 단가
                section_header("🔢", "구성품 배분단가 · 할인율", "상시가 가중치 기준으로 세트가격을 구성품에 배분합니다", "blue")
                sku_always = {}
                for s_sku in st.session_state["bom_df"][st.session_state["bom_df"]["세트ID"]==set_id]["품번"].astype(str).unique():
                    r_s = prod[prod["품번"].astype(str)==s_sku]
                    if r_s.empty: continue
                    rcost = safe_float(r_s.iloc[0]["원가"], np.nan)
                    if rcost != rcost or rcost <= 0: continue
                    min_s, max_s, _ = compute_auto_range_from_cost(
                        rcost, st.session_state["channels_df"], st.session_state["zone_map"],
                        st.session_state["boundaries"], rounding_unit, min_cm, max_cost_ratio,
                        PRICE_ZONES, "공구", safe_float(r_s.iloc[0].get("MSRP_오버라이드", np.nan), np.nan),
                    )
                    z_sku = build_zone_table(
                        rcost, float(min_s), float(max_s),
                        st.session_state["channels_df"], st.session_state["zone_map"],
                        st.session_state["boundaries"], st.session_state["target_pos"],
                        rounding_unit, min_cm, st.session_state["overrides_df"], "SKU", str(s_sku),
                    )
                    ar = z_sku[z_sku["가격영역"]=="상시"]
                    if not ar.empty:
                        sku_always[str(s_sku)] = float(ar.iloc[0]["최종가격(원)"])

                alloc_rows = []
                for _, rr in zdf.iterrows():
                    alloc = allocate_set_price_to_components(set_id, rr["가격영역"], rr["최종가격(원)"], prod, st.session_state["bom_df"], sku_always)
                    if not alloc.empty:
                        alloc_rows.append(alloc)

                if alloc_rows:
                    alloc_df = pd.concat(alloc_rows, ignore_index=True)
                    st.dataframe(
                        alloc_df, use_container_width=True, height=300,
                        column_config={
                            "상시가_ref":          st.column_config.NumberColumn("상시가(원)", format="%d"),
                            "실질단가":            st.column_config.NumberColumn("실질단가(원)", format="%.0f"),
                            "상시대비할인율(%)":   st.column_config.NumberColumn("상시대비할인율(%)", format="%.1f"),
                        }
                    )
                    xb = to_excel_bytes({"set_result": zdf, "alloc": alloc_df, "order_viol": viol})
                    st.download_button("⬇️  세트 결과 엑셀 다운로드", xb, file_name=f"{set_id}_result.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        type="primary")
                else:
                    info_box("구성품 상시가 산출이 부족해 배분 계산이 제한됩니다. 구성품 원가를 확인하세요.")


# ════════════════════════════════════════════════════════════
# 탭 4 : 운영플랜 · 카니발
# ════════════════════════════════════════════════════════════
with tab_plan:
    prod = st.session_state["products_df"].copy()

    if prod.empty:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:3rem;margin-bottom:16px;">📂</div>
          <h3 style="color:#1e293b;margin-bottom:8px;">파일을 먼저 업로드해 주세요</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        section_header("📋", "운영플랜 구성", "가격영역별로 SKU/세트를 배치합니다", "indigo")
        info_box("가격영역에 배치된 SKU/세트의 '실질단가'를 기준으로 가격 역전(카니발)을 탐지합니다.")

        plan = st.session_state["plan_df"].copy()
        pl1, pl2, pl3, pl4 = st.columns([1, 1, 3, 1])
        with pl1:
            zone  = st.selectbox("가격영역", PRICE_ZONES, index=0, key="plan_zone")
        with pl2:
            otype = st.selectbox("오퍼타입", ["SKU","SET"], index=0, key="plan_otype")
        with pl3:
            if otype == "SKU":
                opts = (prod["품번"].astype(str) + "  |  " + prod["상품명"].astype(str)).tolist()
                pick = st.selectbox("오퍼 선택", opts, index=0, key="plan_pick_sku",
                                     label_visibility="collapsed")
                oid  = pick.split("  |  ", 1)[0].strip()
            else:
                sets_df = st.session_state["sets_df"].copy()
                if sets_df.empty:
                    warn_box("세트 탭에서 먼저 세트를 생성해 주세요.")
                    oid = ""
                else:
                    opts = (sets_df["세트ID"].astype(str) + "  |  " + sets_df["세트명"].astype(str)).tolist()
                    pick = st.selectbox("오퍼 선택", opts, index=0, key="plan_pick_set",
                                         label_visibility="collapsed")
                    oid  = pick.split("  |  ", 1)[0].strip()
        with pl4:
            p_override = st.number_input("가격 오버라이드(0=미사용)", min_value=0, value=0,
                                          step=1000, key="plan_price", label_visibility="collapsed")

        if st.button("➕ 플랜에 추가", type="primary", key="plan_add", disabled=(otype=="SET" and oid=="")):
            plan = pd.concat([plan, pd.DataFrame([{
                "가격영역": zone, "오퍼타입": otype, "오퍼ID": oid,
                "가격_오버라이드": (float(p_override) if p_override > 0 else np.nan)
            }])], ignore_index=True)
            st.session_state["plan_df"] = plan

        divider()
        section_header("📑", "현재 운영플랜", "", "blue")

        if st.session_state["plan_df"].empty:
            info_box("플랜이 비어 있습니다. 위에서 항목을 추가해 주세요.")
        else:
            st.session_state["plan_df"] = st.data_editor(
                st.session_state["plan_df"],
                use_container_width=True, height=220, num_rows="dynamic",
                column_config={
                    "가격영역":       st.column_config.TextColumn("가격영역"),
                    "오퍼타입":       st.column_config.TextColumn("타입"),
                    "오퍼ID":         st.column_config.TextColumn("SKU/세트 ID"),
                    "가격_오버라이드":st.column_config.NumberColumn("가격 오버라이드(원)", format="%d"),
                }
            )
            plan = st.session_state["plan_df"].copy()

        divider()
        section_header("🔍", "카니발(가격역전) 탐지", "플랜 내 SKU별 최저 실질단가 기준으로 역전·갭부족을 탐지합니다", "red")

        cp1, cp2, cp3 = st.columns([1, 1, 1])
        with cp1:
            rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="plan_round")
        with cp2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한", min_value=0.05, max_value=0.80,
                                              value=0.30, step=0.01, format="%.2f", key="plan_ratio")
        with cp3:
            min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="plan_cm") / 100.0

        cc1, cc2 = st.columns(2)
        with cc1:
            gap_pct = st.slider("최소 갭 (%)", 0, 20, 5, 1, key="plan_gap_pct") / 100.0
        with cc2:
            gap_won = st.number_input("최소 갭 (원)", min_value=0, value=2000, step=500, key="plan_gap_won")

        if st.button("🔍 카니발 탐지 실행", type="primary", key="plan_run"):
            if plan.empty:
                st.error("플랜이 비어 있습니다.")
            else:
                with st.spinner("분석 중..."):
                    eff_rows  = []
                    sets_df   = st.session_state["sets_df"].copy()
                    bom_df    = st.session_state["bom_df"].copy()
                    overrides_df = st.session_state["overrides_df"].copy()

                    def sku_zone_price(sku, zone):
                        r_s = prod[prod["품번"].astype(str)==sku].iloc[0]
                        c_s = safe_float(r_s["원가"], np.nan)
                        if c_s != c_s or c_s <= 0: return np.nan
                        mn, mx, _ = compute_auto_range_from_cost(c_s, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"], rounding_unit, min_cm, max_cost_ratio, PRICE_ZONES, "공구", safe_float(r_s.get("MSRP_오버라이드", np.nan), np.nan))
                        z2 = build_zone_table(c_s, mn, mx, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"], st.session_state["target_pos"], rounding_unit, min_cm, overrides_df, "SKU", sku)
                        rr = z2[z2["가격영역"]==zone]
                        return float(rr.iloc[0]["최종가격(원)"]) if not rr.empty else np.nan

                    sku_always = {sku: p for sku in prod["품번"].astype(str).tolist() if (p := sku_zone_price(sku, "상시")) == p}

                    for _, pr in plan.iterrows():
                        pz = pr["가격영역"]; pt = pr["오퍼타입"]; pid = str(pr["오퍼ID"])
                        p_ov = safe_float(pr.get("가격_오버라이드", np.nan), np.nan)
                        if pt == "SKU":
                            p_v = sku_zone_price(pid, pz)
                            if p_ov == p_ov and p_ov > 0: p_v = p_ov
                            if p_v == p_v:
                                eff_rows.append({"가격영역":pz,"품번":pid,"실질단가":p_v,"오퍼":f"SKU:{pid}"})
                        else:
                            ct = compute_set_cost(pid, bom_df, prod)
                            if ct != ct: continue
                            sr = sets_df[sets_df["세트ID"]==pid]
                            mo = safe_float(sr.iloc[0].get("MSRP_오버라이드",np.nan),np.nan) if not sr.empty else np.nan
                            mn, mx, _ = compute_auto_range_from_cost(ct, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"], rounding_unit, min_cm, max_cost_ratio, PRICE_ZONES, "공구", mo)
                            zs = build_zone_table(ct, mn, mx, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"], st.session_state["target_pos"], rounding_unit, min_cm, overrides_df, "SET", pid)
                            rr = zs[zs["가격영역"]==pz]
                            if rr.empty: continue
                            sp = float(rr.iloc[0]["최종가격(원)"])
                            if p_ov == p_ov and p_ov > 0: sp = p_ov
                            alloc = allocate_set_price_to_components(pid, pz, sp, prod, bom_df, sku_always)
                            for _, ar in alloc.iterrows():
                                eff_rows.append({"가격영역":pz,"품번":str(ar["품번"]),"실질단가":float(ar["실질단가"]),"오퍼":f"SET:{pid}"})

                    eff = pd.DataFrame(eff_rows)
                    if eff.empty:
                        st.error("실질단가 계산 결과가 없습니다.")
                    else:
                        min_eff = eff.groupby(["가격영역","품번"], as_index=False)["실질단가"].min()
                        min_eff = min_eff.merge(prod[["품번","상품명"]], on="품번", how="left")

                        section_header("📊", "채널별 SKU 최저 실질단가", "", "blue")
                        st.dataframe(
                            min_eff.sort_values(["품번","가격영역"]),
                            use_container_width=True, height=300,
                            column_config={
                                "실질단가": st.column_config.NumberColumn("최저 실질단가(원)", format="%d"),
                            }
                        )

                        order = {z:i for i,z in enumerate(PRICE_ZONES)}
                        viol  = []
                        for sku_v, g in min_eff.groupby("품번"):
                            g2 = g.copy(); g2["ord"] = g2["가격영역"].map(order); g2 = g2.sort_values("ord")
                            prev = prev_zone = None
                            for _, rr in g2.iterrows():
                                cur = rr["실질단가"]
                                if prev is None: prev=cur; prev_zone=rr["가격영역"]; continue
                                need = max(prev*(1+gap_pct), prev+gap_won)
                                if cur < need:
                                    viol.append({"품번":sku_v,"상품명":rr.get("상품명",""),"하위영역":prev_zone,"하위가격":prev,"상위영역":rr["가격영역"],"상위가격":cur,"필요최소상위가":need,"갭부족":need-cur})
                                prev=cur; prev_zone=rr["가격영역"]
                        viol_df = pd.DataFrame(viol)

                        section_header("⚠️", "카니발(역전·갭부족) 경고", "", "red")
                        if viol_df.empty:
                            success_box("카니발 경고 없음 — 현재 플랜에서 가격 역전이 없습니다.")
                        else:
                            warn_box(f"{len(viol_df):,}건의 역전·갭부족이 감지되었습니다. 아래 항목을 확인해 주세요.")
                            st.dataframe(viol_df, use_container_width=True, height=260,
                                          column_config={"갭부족": st.column_config.NumberColumn("갭부족(원)", format="%d")})

                        xb = to_excel_bytes({"min_eff": min_eff, "violations": viol_df})
                        st.download_button("⬇️  카니발 결과 엑셀 다운로드", xb,
                                            file_name="cannibal_result.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            type="primary")


# ════════════════════════════════════════════════════════════
# 탭 5 : 로직 안내
# ════════════════════════════════════════════════════════════
with tab_logic:
    section_header("📖", "시뮬레이터 로직 안내", "원가만으로 자동 가격이 산출되는 방식을 설명합니다", "indigo")

    logic_items = [
        ("📁", "입력: 원가만 필요",
         "상품코드 · 상품명 · 원가(VAT-) 컬럼이 있는 엑셀 파일만 업로드하면 자동으로 가격이 산출됩니다. "
         "마케팅비 · 반품률은 0으로 두어도 작동하며, 나중에 값을 입력하면 즉시 재계산됩니다.",
         ""),
        ("📐", "손익하한(Floor) 계산 공식",
         "각 가격영역은 비용채널(수수료 · PG · 배송 · 마케팅 · 반품)을 가지며, 아래 공식으로 Floor를 산출합니다:",
         "Floor = (원가 + 배송비/주문 + 반품률×반품비/주문) ÷ (1 − 수수료 − PG − 마케팅비 − 최소기여이익률)"),
        ("🎯", "자동 레인지(Min ~ Max = MSRP)",
         "아래 세 가지 기준 중 가장 큰 값을 Max로 자동 설정합니다:",
         "• Min(자동) = 선택한 최저 영역(기본 공구)의 Floor\n• MSRP_base = 원가 ÷ 원가율 상한(예: 30%)\n• 각 영역 Floor가 해당 BandHigh 안에 들어오도록 Max를 자동 상향"),
        ("📊", "밴드(가격영역) 분할",
         "Min~Max 레인지(0~100%)를 10개 영역으로 분할합니다. 경계는 슬라이더로 조정하며, 각 영역의 추천가(Target)는 영역 내 위치(기본 중앙)와 Floor 중 큰 값입니다.",
         ""),
        ("✏️", "사용자 자유 조정",
         "아래 항목은 모두 자유롭게 수정할 수 있습니다. 룰 위반 시 경고만 표시되며 저장됩니다:",
         "• Min/Max 조정 → 모든 가격영역이 함께 이동\n• 특정 채널 가격 직접 입력(오버라이드) → Floor 미만이어도 경고만\n• 서열·갭 위반 → 경고 표시"),
        ("🗂️", "세트 가격",
         "세트 원가 = Σ(구성품 원가 × 수량)으로 자동 계산됩니다. 구성품별 배분 단가는 상시가(개별 SKU 상시 채널 최종가)를 가중치로 사용합니다.",
         ""),
        ("⚠️", "카니발(역전) 탐지",
         "운영플랜 탭에서 채널별 SKU/세트를 배치하면, SKU 기준 '최저 실질단가'를 구한 뒤 가격영역 서열(저→고)에서 갭부족 · 역전을 경고로 표시합니다.",
         ""),
    ]

    for icon, title, desc, code in logic_items:
        st.markdown(f"""
        <div class="logic-card">
          <h4>{icon} {title}</h4>
          <p>{desc.replace(chr(10), '<br>')}</p>
          {"" if not code else f'<p style="margin-top:8px;"><code>{code.replace(chr(10), "</code><br><code>")}</code></p>'}
        </div>
        """, unsafe_allow_html=True)
