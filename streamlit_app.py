import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v3 (Policy + Economics-driven)
# - Reflects channel fee/PG/shipping/marketing inputs
# - Enforces price ladder to prevent cannibalization
# - Set pricing rules (reference-based discount range + profit floor)
# - Shows logic in sentences (dynamic, parameterized)
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator v3", layout="wide")

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

def krw_round(x, unit=100):
    try:
        return int(round(float(x) / unit) * unit)
    except Exception:
        return 0

def pct_to_rate(pct):
    return float(pct) / 100.0

def rate_to_pct(rate):
    return float(rate) * 100.0

def to_excel_bytes(df_dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()

# ----------------------------
# Policy / Defaults
# ----------------------------
PRICE_LADDER_LOW_TO_HIGH = [
    "공구가",
    "홈쇼핑가",
    "폐쇄몰가",
    "모바일라방가",
    "원데이특가",
    "브랜드위크가",
    "홈사할인가",
    "상시할인가",
    "오프라인가",
    "소비자가(MSRP)",
]

ONLINE_LADDER_HIGH_TO_LOW = ["상시할인가", "홈사할인가", "브랜드위크가", "원데이특가", "모바일라방가"]

DEFAULT_CHANNEL_CONFIG = pd.DataFrame(
    [
        # 채널명, 수수료%, PG적용, 배송비(원/주문), 마케팅%, 반품률%, 반품비(원/주문), 비고
        ["자사몰", 5.0, True, 3000, 0.0, 0.0, 0, "온라인 앵커(기준)"],
        ["스마트스토어", 5.0, True, 3000, 0.0, 0.0, 0, "온라인 유통"],
        ["쿠팡", 25.0, True, 3000, 0.0, 0.0, 0, "온라인 유통(높은 수수료)"],
        ["오픈마켓", 15.0, True, 3000, 0.0, 0.0, 0, "온라인 유통"],
        ["홈사", 30.0, True, 3000, 0.0, 0.0, 0, "제휴/홈사몰"],
        ["폐쇄몰", 25.0, True, 3000, 0.0, 0.0, 0, "폐쇄몰(특판)"],
        ["공구", 50.0, True, 3000, 0.0, 0.0, 0, "공동구매/딜"],
        ["모바일라이브", 40.0, True, 3000, 0.0, 0.0, 0, "라이브커머스"],
        ["홈쇼핑", 55.0, False, 0, 0.0, 0.0, 0, "수수료+송출/제작+택배: 홈쇼핑 부담(가정)"],
        ["오프라인", 50.0, False, 0, 0.0, 0.0, 0, "사입/리테일 마진(가정)"],
    ],
    columns=["채널", "수수료율(%)", "PG적용", "배송비(원/주문)", "마케팅비(%)", "반품률(%)", "반품비(원/주문)", "비고"],
)

# price type -> channel mapping (경제 구조)
PRICE_TYPE_TO_CHANNEL_DEFAULT = {
    "상시할인가": "자사몰",
    "브랜드위크가": "자사몰",
    "원데이특가": "자사몰",
    "홈사할인가": "홈사",
    "모바일라방가": "모바일라이브",
    "폐쇄몰가": "폐쇄몰",
    "공구가": "공구",
    "홈쇼핑가": "홈쇼핑",
    "오프라인가": "오프라인",
}

# 세트 할인율 범위(Reference 대비) - 회사 룰(기본값)
SET_DISCOUNT_RULE = pd.DataFrame(
    [
        ["공구가", 0.20, 0.35, "전용 SKU 권장/딜 강도 높음"],
        ["홈쇼핑가", 0.20, 0.35, "대용량/다구성 전제"],
        ["폐쇄몰가", 0.20, 0.30, "전용 SKU 권장"],
        ["모바일라방가", 0.15, 0.25, "쿠폰 포함 최종가 기준 관리"],
        ["원데이특가", 0.12, 0.20, ""],
        ["브랜드위크가", 0.12, 0.20, ""],
        ["홈사할인가", 0.08, 0.15, ""],
        ["상시할인가", 0.08, 0.15, ""],
        ["오프라인가", 0.08, 0.15, "오프라인은 과할인 지양"],
    ],
    columns=["가격타입", "세트할인_min", "세트할인_max", "메모"],
)

# ----------------------------
# Market Reference (optional)
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
        return np.nan, np.nan, np.nan

    ref_min = np.nanmin(mins) if len(mins) else (np.nanmin(mids) if len(mids) else np.nan)
    ref_max = np.nanmax(maxs) if len(maxs) else (np.nanmax(mids) if len(mids) else np.nan)

    if len(mids):
        ref_mid = float(np.nanmedian(mids))
    else:
        ref_mid = (ref_min + ref_max) / 2.0 if (not np.isnan(ref_min) and not np.isnan(ref_max)) else np.nan

    return float(ref_min), float(ref_mid), float(ref_max)

def pick_within_band(ref_min, ref_max, positioning_pct):
    if np.isnan(ref_min) or np.isnan(ref_max):
        return np.nan
    p = float(positioning_pct) / 100.0
    return ref_min + (ref_max - ref_min) * p

# ----------------------------
# Economics
# ----------------------------
def calc_min_unit_price(
    unit_cost: float,
    q_in_order: int,
    channel_row: dict,
    pg_rate: float,
    min_cm: float,
):
    """
    최소허용 '단위가'(consumer-facing unit price) 계산.
    - unit_cost: 단위 원가(원)
    - q_in_order: 1주문 내 수량(Q) (세트는 2/4/6 등)
    - channel_row: 채널 파라미터(수수료/배송비/마케팅/반품 등)
    - pg_rate: 결제수수료(%) -> rate
    - min_cm: 최소 기여이익률(%) -> rate

    모델(단순 기대값):
      NetRevenue = Price*(1 - fee - pg - mkt)
      ExpectedCost = unit_cost + ship_per_unit + (return_rate*return_cost_per_order)/q
      Require: (NetRevenue - ExpectedCost) / NetRevenue >= min_cm
      => Price >= ExpectedCost / (1 - fee - pg - mkt - min_cm)
    """
    fee = pct_to_rate(channel_row.get("수수료율(%)", 0.0))
    mkt = pct_to_rate(channel_row.get("마케팅비(%)", 0.0))
    pg = pct_to_rate(pg_rate) if bool(channel_row.get("PG적용", True)) else 0.0
    ship_per_order = safe_float(channel_row.get("배송비(원/주문)", 0.0), 0.0)
    ship_per_unit = ship_per_order / max(1, int(q_in_order))

    ret_rate = pct_to_rate(channel_row.get("반품률(%)", 0.0))
    ret_cost_per_order = safe_float(channel_row.get("반품비(원/주문)", 0.0), 0.0)
    expected_ret_cost_per_unit = (ret_rate * ret_cost_per_order) / max(1, int(q_in_order))

    denom = 1.0 - (fee + mkt + pg + min_cm)
    if denom <= 0:
        return float("inf")

    expected_cost = float(unit_cost) + float(ship_per_unit) + float(expected_ret_cost_per_unit)
    return expected_cost / denom

def enforce_ladder_with_gap(prices, order_low_to_high, gap_rate, gap_abs, rounding_unit, auto_correct=True):
    """
    prices: dict(price_type -> value)
    Ensures: next >= max(prev*(1+gap_rate), prev+gap_abs)
    """
    p = dict(prices)
    warnings = []

    prev_key = None
    prev_val = None
    for k in order_low_to_high:
        if k not in p or np.isnan(p[k]):
            continue
        if prev_key is None:
            prev_key, prev_val = k, float(p[k])
            continue

        need = max(prev_val * (1.0 + gap_rate), prev_val + gap_abs)
        if float(p[k]) < need:
            msg = f"[서열/간격] {k}({p[k]:,.0f}) < {prev_key} 대비 최소허용({need:,.0f}) (gap {gap_rate*100:.0f}% or {gap_abs:,.0f}원)"
            if auto_correct:
                p[k] = krw_round(need, rounding_unit)
                warnings.append(msg + " → 자동보정: 상향")
            else:
                warnings.append(msg)
        prev_key, prev_val = k, float(p[k])

    # MSRP must be >= max other
    if "소비자가(MSRP)" in p:
        max_other = max([v for kk, v in p.items() if kk != "소비자가(MSRP)" and (v is not None and not np.isnan(v))] + [0])
        if p["소비자가(MSRP)"] < max_other:
            msg = f"[앵커] MSRP({p['소비자가(MSRP)']:,.0f}) < 타 채널 최고가({max_other:,.0f})"
            if auto_correct:
                p["소비자가(MSRP)"] = krw_round(max_other, rounding_unit)
                warnings.append(msg + " → 자동보정: MSRP 상향")
            else:
                warnings.append(msg)

    return p, warnings

def make_band(target_price, band_pct, rounding_unit):
    t = float(target_price)
    half = pct_to_rate(band_pct) / 2.0
    pmin = t * (1.0 - half)
    pmax = t * (1.0 + half)
    return krw_round(pmin, rounding_unit), krw_round(t, rounding_unit), krw_round(pmax, rounding_unit)

# ----------------------------
# Master loader (simple)
# ----------------------------
def infer_brand_from_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0]

def load_master(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]

    # Try map common columns
    rename_map = {}
    for c in df.columns:
        if c in ["품번", "상품코드", "제품코드", "SKU", "코드"]:
            rename_map[c] = "품번"
        if c in ["신규품명", "통일제품명", "통일 제품명", "제품명", "상품명"]:
            rename_map[c] = "신규품명"
        if c in ["원가", "랜디드코스트", "랜디드코스트(총원가)", "총원가", "COGS"]:
            rename_map[c] = "랜디드코스트(총원가)"
        if c in ["소비자가", "MSRP", "정가"]:
            rename_map[c] = "소비자가_참고"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "품번" not in df.columns:
        df["품번"] = ""
    if "신규품명" not in df.columns:
        df["신규품명"] = ""

    # Keep some optional columns if exist
    keep = ["품번", "신규품명"]
    for c in ["랜디드코스트(총원가)", "소비자가_참고"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].copy()
    df["브랜드(추정)"] = df["신규품명"].apply(infer_brand_from_name)
    df["품번"] = df["품번"].astype(str).str.strip()
    df["신규품명"] = df["신규품명"].astype(str).str.strip()
    df = df[df["품번"].ne("") | df["신규품명"].ne("")].drop_duplicates(subset=["품번"]).reset_index(drop=True)
    return df

# ============================================================
# Compute engine per SKU
# ============================================================
def compute_prices_for_row(
    row: pd.Series,
    channel_cfg: pd.DataFrame,
    online_anchor_channel: str,
    pg_rate: float,
    min_cm: float,
    cost_ratio_limit: float,  # e.g., 0.30
    rounding_unit: int,
    gap_rate: float,
    gap_abs: int,
    auto_correct: bool,
    # online discounts (based on 상시)
    d_homesale: float,
    d_brandweek: float,
    d_oneday: float,
    d_live: float,
    list_disc: float,  # 상시→MSRP 프레이밍
    # HS/GB/Closed position rules
    hs_under_live: float,  # HS unit = live * (1 - hs_under_live)
    gb_under_hs: float,    # GB unit = HS * (1 - gb_under_hs)
    closed_pos_pct: float, # closed between HS and live (0=HS, 100=live)
    offline_disc_from_msrp: float, # offline = MSRP * (1 - offline_disc)
    # market band option
    use_market_band: bool,
    market_positioning_pct: float,
    band_pct: float,
    set_ref_price_type: str, # "상시할인가" or "브랜드위크가"
    set_discount_rule_df: pd.DataFrame,
):
    warn = []
    out_rows = []

    code = str(row.get("품번", "")).strip()
    name = str(row.get("신규품명", "")).strip()
    brand = str(row.get("브랜드(추정)", "")).strip()

    q_online = int(max(1, safe_float(row.get("온라인기준수량(Q_online)", 1), 1)))
    q_hs = int(max(1, safe_float(row.get("홈쇼핑구성(Q_hs)", 1), 1)))
    q_gb = int(max(1, safe_float(row.get("공구구성(Q_gb)", 1), 1)))
    q_closed = int(max(1, safe_float(row.get("폐쇄몰구성(Q_closed)", q_online), q_online)))

    landed_total = safe_float(row.get("랜디드코스트(총원가)", np.nan))
    if np.isnan(landed_total) or landed_total <= 0:
        return [], [{"품번": code, "신규품명": name, "메시지": "[입력] 랜디드코스트(총원가)가 없어서 계산 불가"}], []

    unit_cost = float(landed_total) / max(1, q_online)

    # MSRP floor from cost ratio rule
    msrp_floor_unit = unit_cost / max(1e-6, cost_ratio_limit)  # cost_ratio_limit=0.30 means MSRP >= cost/0.30
    msrp_input = safe_float(row.get("소비자가_입력", np.nan))
    if not np.isnan(msrp_input) and msrp_input > 0:
        # 입력 MSRP는 "총 세트가"로 들어온다고 가정 -> 단위 MSRP로 환산
        msrp_floor_unit = max(msrp_floor_unit, float(msrp_input) / max(1, q_online))

    # Build channel map
    cfg = channel_cfg.set_index("채널").to_dict(orient="index")

    def ch(name_):
        if name_ not in cfg:
            # fallback to first row
            return list(cfg.values())[0]
        return cfg[name_]

    # Economics floors for each price type (unit)
    floors = {}
    # Online tiers on anchor channel
    anchor_ch = online_anchor_channel
    for pt in ["상시할인가", "브랜드위크가", "원데이특가"]:
        floors[pt] = calc_min_unit_price(unit_cost, q_online, ch(anchor_ch), pg_rate, min_cm)
    # 홈사
    floors["홈사할인가"] = calc_min_unit_price(unit_cost, q_online, ch("홈사"), pg_rate, min_cm)
    # live / closed / hs / gb / offline
    floors["모바일라방가"] = calc_min_unit_price(unit_cost, q_online, ch("모바일라이브"), pg_rate, min_cm)
    floors["폐쇄몰가"] = calc_min_unit_price(unit_cost, q_closed, ch("폐쇄몰"), pg_rate, min_cm)
    floors["홈쇼핑가"] = calc_min_unit_price(unit_cost, q_hs, ch("홈쇼핑"), pg_rate, min_cm)
    floors["공구가"] = calc_min_unit_price(unit_cost, q_gb, ch("공구"), pg_rate, min_cm)
    floors["오프라인가"] = calc_min_unit_price(unit_cost, q_online, ch("오프라인"), pg_rate, min_cm)

    # Always is decision variable. Define each price type as coef * always_unit (linear model).
    # Online ladder
    coef = {}
    coef["상시할인가"] = 1.0
    coef["홈사할인가"] = 1.0 - float(d_homesale)
    coef["브랜드위크가"] = 1.0 - float(d_brandweek)
    coef["원데이특가"] = 1.0 - float(d_oneday)
    coef["모바일라방가"] = 1.0 - float(d_live)

    # Ensure discount monotonic high->low; if violates, warn and clip discounts
    # (상시 >= 홈사 >= 브위 >= 원데이 >= 라방)
    # Convert to "prices": higher coef means higher price.
    # If coef order breaks, we adjust by making offending coef equal to previous.
    online_order = ONLINE_LADDER_HIGH_TO_LOW
    prev_coef = None
    for k in online_order:
        if prev_coef is None:
            prev_coef = coef[k]
            continue
        if coef[k] > prev_coef:
            warn.append({"품번": code, "신규품명": name, "메시지": f"[할인율] {k}가 상위레벨보다 비쌈(할인율 설정 위반) → {k}를 상위레벨 수준으로 캡"})
            coef[k] = prev_coef
        prev_coef = coef[k]

    # MSRP derived from always
    coef_msrp = 1.0 / max(1e-6, (1.0 - float(list_disc)))
    coef["소비자가(MSRP)"] = coef_msrp

    # Offline derived from MSRP
    coef["오프라인가"] = coef_msrp * (1.0 - float(offline_disc_from_msrp))

    # HS/GB derived from live
    coef["홈쇼핑가"] = coef["모바일라방가"] * (1.0 - float(hs_under_live))
    coef["공구가"] = coef["홈쇼핑가"] * (1.0 - float(gb_under_hs))

    # Closed mall between HS and live
    pos = float(closed_pos_pct) / 100.0
    coef["폐쇄몰가"] = coef["홈쇼핑가"] * (1.0 - pos) + coef["모바일라방가"] * pos

    # Required always from floors
    always_reqs = []

    # MSRP floor: msrp_floor_unit <= always*coef_msrp => always >= msrp_floor_unit/coef_msrp
    always_reqs.append(msrp_floor_unit / coef_msrp)

    for pt, floor in floors.items():
        if pt not in coef:
            continue
        c = coef[pt]
        if c <= 0:
            continue
        always_reqs.append(float(floor) / c)

    # Optional: market anchor for always
    if use_market_band:
        ref_min, ref_mid, ref_max = compute_reference_band(row)
        market_always = pick_within_band(ref_min, ref_max, market_positioning_pct)
        if not np.isnan(market_always):
            always_reqs.append(market_always)

    always_unit = max(always_reqs) if always_reqs else np.nan
    if np.isnan(always_unit) or always_unit <= 0:
        return [], [{"품번": code, "신규품명": name, "메시지": "[계산] always_unit 산출 실패"}], []

    # Build unit prices
    prices_unit = {pt: always_unit * coef[pt] for pt in coef.keys()}

    # Round all unit prices
    for k in prices_unit:
        prices_unit[k] = krw_round(prices_unit[k], rounding_unit)

    # Enforce ladder order with gap on "총 가격"(세트는 set price로 비교), but we can approximate by unit then.
    # We'll do ladder on unit prices first (except MSRP).
    ladder_prices = {k: prices_unit.get(k, np.nan) for k in PRICE_LADDER_LOW_TO_HIGH}
    ladder_prices, w_ladder = enforce_ladder_with_gap(
        ladder_prices,
        PRICE_LADDER_LOW_TO_HIGH,
        gap_rate=gap_rate,
        gap_abs=gap_abs,
        rounding_unit=rounding_unit,
        auto_correct=auto_correct,
    )
    for w in w_ladder:
        warn.append({"품번": code, "신규품명": name, "메시지": w})

    # Apply ladder-corrected to prices_unit
    prices_unit.update(ladder_prices)

    # Prepare result rows (Target is total price = unit * Q)
    def add_row(price_type, q, channel_name):
        unit_val = prices_unit.get(price_type, np.nan)
        if np.isnan(unit_val):
            return
        target_total = float(unit_val) * int(q)
        pmin, ptgt, pmax = make_band(target_total, band_pct, rounding_unit)

        # Economics check: compute expected CM% at Target
        ch_row = ch(channel_name)
        fee = pct_to_rate(ch_row.get("수수료율(%)", 0.0))
        mkt = pct_to_rate(ch_row.get("마케팅비(%)", 0.0))
        pg = pct_to_rate(pg_rate) if bool(ch_row.get("PG적용", True)) else 0.0
        ship = safe_float(ch_row.get("배송비(원/주문)", 0.0), 0.0)
        ret_rate = pct_to_rate(ch_row.get("반품률(%)", 0.0))
        ret_cost = safe_float(ch_row.get("반품비(원/주문)", 0.0), 0.0)

        net_sales = target_total * (1.0 - fee - mkt - pg)
        expected_cost_total = float(unit_cost) * int(q) + ship + ret_rate * ret_cost
        cm = net_sales - expected_cost_total
        cm_rate = (cm / net_sales) if net_sales > 0 else np.nan

        out_rows.append({
            "품번": code,
            "신규품명": name,
            "브랜드(추정)": brand,
            "가격타입": price_type,
            "적용채널(경제)": channel_name,
            "구성Q": int(q),
            "Target": krw_round(ptgt, rounding_unit),
            "Min": krw_round(pmin, rounding_unit),
            "Max": krw_round(pmax, rounding_unit),
            "예상정산액(원)": krw_round(net_sales, 1),
            "예상기여이익(원)": krw_round(cm, 1),
            "예상기여이익률": (f"{cm_rate*100:.1f}%" if not np.isnan(cm_rate) else ""),
        })

    # Map price types to q and channel
    add_row("공구가", q_gb, "공구")
    add_row("홈쇼핑가", q_hs, "홈쇼핑")
    add_row("폐쇄몰가", q_closed, "폐쇄몰")
    add_row("모바일라방가", q_online, "모바일라이브")
    add_row("원데이특가", q_online, online_anchor_channel)
    add_row("브랜드위크가", q_online, online_anchor_channel)
    add_row("홈사할인가", q_online, "홈사")
    add_row("상시할인가", q_online, online_anchor_channel)
    add_row("오프라인가", q_online, "오프라인")
    add_row("소비자가(MSRP)", q_online, online_anchor_channel)

    # Set discount rule check (Reference 대비)
    diag_rows = []

    ref_unit = prices_unit.get(set_ref_price_type, np.nan)
    if not np.isnan(ref_unit):
        rules = set_discount_rule_df.set_index("가격타입").to_dict(orient="index")
        for pt, q in [("공구가", q_gb), ("홈쇼핑가", q_hs), ("폐쇄몰가", q_closed), ("모바일라방가", q_online)]:
            if pt not in prices_unit:
                continue
            set_price = float(prices_unit[pt]) * int(q)
            ref_total = float(ref_unit) * int(q)
            disc = 1.0 - (set_price / ref_total) if ref_total > 0 else np.nan

            rinfo = rules.get(pt, None)
            if rinfo is not None and not np.isnan(disc):
                dmin = float(rinfo["세트할인_min"])
                dmax = float(rinfo["세트할인_max"])
                if disc < dmin:
                    warn.append({"품번": code, "신규품명": name, "메시지": f"[세트체감] {pt} 세트할인 {disc*100:.1f}% < 권장 최소 {dmin*100:.0f}% → 고객 체감 약할 수 있음(사은품/구성 강화 고려)"})
                if disc > dmax:
                    warn.append({"품번": code, "신규품명": name, "메시지": f"[세트충돌] {pt} 세트할인 {disc*100:.1f}% > 권장 최대 {dmax*100:.0f}% → 가격질서 붕괴 위험(전용SKU/구성울타리 필수)"})

            diag_rows.append({
                "품번": code,
                "신규품명": name,
                "Reference(기준가)": set_ref_price_type,
                "가격타입": pt,
                "세트Q": int(q),
                "Reference총액": krw_round(ref_total, rounding_unit),
                "세트판매가": krw_round(set_price, rounding_unit),
                "세트할인율": (f"{disc*100:.1f}%" if not np.isnan(disc) else ""),
            })

    return out_rows, warn, diag_rows

def compute_for_all(df_in: pd.DataFrame, **kwargs):
    rows = []
    warn_rows = []
    diag_rows = []
    for _, r in df_in.iterrows():
        out, warns, diags = compute_prices_for_row(r, **kwargs)
        rows.extend(out)
        warn_rows.extend(warns)
        diag_rows.extend(diags)
    out_df = pd.DataFrame(rows)
    warn_df = pd.DataFrame(warn_rows)
    diag_df = pd.DataFrame(diag_rows)

    if not out_df.empty:
        order = {t: i for i, t in enumerate(PRICE_LADDER_LOW_TO_HIGH)}
        out_df["__ord"] = out_df["가격타입"].map(order).fillna(999).astype(int)
        out_df = out_df.sort_values(["품번", "__ord"]).drop(columns="__ord").reset_index(drop=True)

    return out_df, warn_df, diag_df

# ============================================================
# UI
# ============================================================
st.title("IBR 가격 시뮬레이터 v3 (정책+손익 기반)")
st.caption("채널비용(수수료/PG/배송/마케팅/반품) 입력 → 원가율/최소기여이익 방어 → 채널 서열/간격으로 자동보정 → 밴드(Min~Target~Max) 출력")

tab_sim, tab_policy, tab_data = st.tabs(["시뮬레이터", "정책/로직(문장)", "데이터 업로드/선택"])

# Session defaults
if "master_df" not in st.session_state:
    st.session_state["master_df"] = pd.DataFrame(columns=["품번", "신규품명", "브랜드(추정)"])
if "inputs_df" not in st.session_state:
    st.session_state["inputs_df"] = pd.DataFrame(columns=[
        "품번", "신규품명", "브랜드(추정)",
        "온라인기준수량(Q_online)", "홈쇼핑구성(Q_hs)", "공구구성(Q_gb)", "폐쇄몰구성(Q_closed)",
        "랜디드코스트(총원가)", "소비자가_입력",
        "국내관측_min", "국내관측_max",
        "경쟁사_min", "경쟁사_max", "경쟁사_avg",
        "해외정가_RRP", "해외실판매_min", "해외실판매_max",
        "직구가",
    ])
if "channel_cfg" not in st.session_state:
    st.session_state["channel_cfg"] = DEFAULT_CHANNEL_CONFIG.copy()
if "set_rule" not in st.session_state:
    st.session_state["set_rule"] = SET_DISCOUNT_RULE.copy()

# ----------------------------
# DATA TAB
# ----------------------------
with tab_data:
    st.subheader("1) 상품 마스터 업로드")
    up = st.file_uploader("상품정보 엑셀 업로드 (품번/신규품명 필수, 원가/소비자가는 있으면 자동 반영)", type=["xlsx", "xls"])

    colA, colB = st.columns([2, 1])
    with colA:
        if up is not None:
            try:
                master = load_master(up)
                st.session_state["master_df"] = master

                # 입력테이블 기본 세팅: 원가/소비자가 있으면 채우기
                base = st.session_state["inputs_df"].copy()
                merged = master.copy()
                if "랜디드코스트(총원가)" in merged.columns:
                    merged["랜디드코스트(총원가)"] = merged["랜디드코스트(총원가)"]
                if "소비자가_참고" in merged.columns:
                    merged["소비자가_입력"] = merged["소비자가_참고"]
                # default Q
                merged["온라인기준수량(Q_online)"] = 1
                merged["홈쇼핑구성(Q_hs)"] = 1
                merged["공구구성(Q_gb)"] = 1
                merged["폐쇄몰구성(Q_closed)"] = 1

                # ensure all columns exist
                for c in base.columns:
                    if c not in merged.columns:
                        merged[c] = np.nan

                merged = merged[base.columns]
                st.session_state["inputs_df"] = merged.drop_duplicates(subset=["품번"], keep="first").reset_index(drop=True)
                st.success(f"업로드 완료: {len(master):,}개 상품 로드")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류: {e}")
        else:
            st.info("업로드하면 품번/신규품명(필수)을 읽어옵니다.")

    with colB:
        if not st.session_state["master_df"].empty:
            st.metric("로드된 상품 수", f"{len(st.session_state['master_df']):,}")

    st.divider()
    st.subheader("2) (옵션) 입력값 엑셀 업로드로 보강")
    st.caption("품번 기준으로 랜디드/리서치/구성Q 등을 업로드해 자동 병합합니다.")
    up2 = st.file_uploader("입력값 엑셀 업로드(품번 포함 권장)", type=["xlsx", "xls"], key="input_uploader")

    if up2 is not None and not st.session_state["inputs_df"].empty:
        try:
            add = pd.read_excel(up2)
            add.columns = [str(c).strip() for c in add.columns]
            if "품번" not in add.columns:
                # try common
                for c in add.columns:
                    if c in ["상품코드", "제품코드", "SKU"]:
                        add = add.rename(columns={c: "품번"})
                        break
            if "품번" not in add.columns:
                st.error("업로드 파일에 '품번'(또는 SKU/상품코드/제품코드) 컬럼이 필요합니다.")
            else:
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

    st.divider()
    st.subheader("3) 입력 테이블 미리보기/편집")
    st.caption("여기서 원가/리서치/구성Q를 수정할 수 있습니다.")
    if st.session_state["inputs_df"].empty:
        st.warning("먼저 상품정보 파일을 업로드하세요.")
    else:
        st.session_state["inputs_df"] = st.data_editor(
            st.session_state["inputs_df"],
            use_container_width=True,
            height=420,
            num_rows="dynamic",
        )

# ----------------------------
# POLICY TAB (문장 노출)
# ----------------------------
with tab_policy:
    st.subheader("정책/로직 (문장형) — 입력값에 따라 자동 갱신")

    cfg_df = st.session_state["channel_cfg"]
    pg_default = 3.0

    st.markdown("### A. 기본 원칙(고정)")
    st.markdown(
        """
1) **모든 SKU는 원가율 30% 상한을 만족하도록 MSRP를 설정한다.**  
2) **공식몰·오프라인은 앵커 가격이며, 가격 질서(서열/간격)를 무너뜨리지 않는다.**  
3) **표시가(쿠폰 전)와 최종가(쿠폰 후) 모두에서 채널 서열이 역전되지 않도록 한다.**  
4) **세트는 단품 Reference(기준가) 대비 채널별 할인율 범위 내에서만 운영한다.**
"""
    )

    st.markdown("### B. 채널 비용 구조(입력값 기반)")
    st.caption("아래 표는 시뮬레이터에서 직접 수정 가능하며, 수정 시 모든 결과가 즉시 재계산됩니다.")
    st.dataframe(cfg_df, use_container_width=True, height=360)

    st.markdown("### C. 손익 하한(최소 판매가) 공식")
    st.markdown(
        """
- 채널별 최소허용 단위가(원)는 아래식을 사용합니다.

`최소허용 단위가 = (단위원가 + (배송비/주문)/Q + (반품률×반품비/주문)/Q) ÷ (1 - 수수료 - PG - 마케팅비 - 최소기여이익률)`

- 여기서 Q는 해당 채널의 1주문 구성 수량입니다(세트일수록 Q↑ → 배송비가 단위에 분산).
"""
    )

    st.markdown("### D. 채널 가격 서열(카니발 방지)")
    st.markdown(
        """
- 가격은 아래 순서(저가→고가)를 기본으로 하며, 인접 구간은 **최소 +5% 또는 +2,000원(둘 중 큰 값)** 이상의 간격을 유지합니다.

`공구 < 홈쇼핑 < 폐쇄몰 < 라방 < 원데이 < 브랜드위크 < 홈사 < 상시 < 오프라인 < MSRP`
"""
    )

    st.markdown("### E. 세트 할인율 룰(Reference 대비)")
    st.dataframe(st.session_state["set_rule"], use_container_width=True, height=320)

# ----------------------------
# SIM TAB
# ----------------------------
with tab_sim:
    st.subheader("1) 채널 파라미터(수수료/배송/PG/마케팅/반품)")
    st.caption("여기 값만 수정하면, 아래 추천 가격/밴드/손익이 전부 즉시 바뀝니다.")

    cfg = st.data_editor(
        st.session_state["channel_cfg"],
        use_container_width=True,
        height=330,
        num_rows="fixed",
        column_config={
            "PG적용": st.column_config.CheckboxColumn("PG적용"),
        }
    )
    st.session_state["channel_cfg"] = cfg

    st.divider()
    st.subheader("2) 공통 정책 파라미터")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        rounding_unit = st.selectbox("반올림 단위", [10, 100, 1000], index=1)
    with c2:
        pg_rate = st.number_input("PG 수수료(%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    with c3:
        min_cm = st.slider("최소 기여이익률(%)", 0, 40, 15, 1) / 100.0
    with c4:
        cost_ratio_limit = st.slider("MSRP 원가율 상한(%)", 10, 60, 30, 1) / 100.0
    with c5:
        band_pct = st.slider("추천 밴드폭(%)", 0, 20, 6, 1)

    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        gap_rate = st.slider("인접 밴드 최소 간격(%)", 0, 20, 5, 1) / 100.0
    with g2:
        gap_abs = st.number_input("인접 밴드 최소 간격(원)", min_value=0, max_value=20000, value=2000, step=500)
    with g3:
        auto_correct = st.toggle("규칙 위반 시 자동보정", value=True)

    st.divider()
    st.subheader("3) 온라인 가격 사다리(상시 기준 할인율)")
    o1, o2, o3, o4, o5, o6 = st.columns([1, 1, 1, 1, 1, 1])
    with o1:
        d_brandweek = st.slider("브랜드위크 할인율(%)", 0, 70, 25, 1) / 100.0
    with o2:
        d_homesale = st.slider("홈사 할인율(%)", 0, 70, 10, 1) / 100.0
    with o3:
        d_oneday = st.slider("원데이 할인율(%)", 0, 80, 30, 1) / 100.0
    with o4:
        d_live = st.slider("라방 할인율(%)", 0, 80, 40, 1) / 100.0
    with o5:
        list_disc = st.slider("상시→MSRP 프레이밍(%)", 0, 80, 20, 1) / 100.0
    with o6:
        offline_disc = st.slider("MSRP 대비 오프라인 할인(%)", 0, 30, 5, 1) / 100.0

    st.caption("온라인 질서: 상시 ≥ 홈사 ≥ 브랜드위크 ≥ 원데이 ≥ 라방(최저)")

    st.divider()
    st.subheader("4) 공구/홈쇼핑/폐쇄몰 포지션(정책)")
    p1, p2, p3 = st.columns([1, 1, 1])
    with p1:
        hs_under_live = st.slider("HS는 라방 단위가 대비 -%(목표)", 0, 40, 10, 1) / 100.0
    with p2:
        gb_under_hs = st.slider("공구는 HS 단위가 대비 -%(목표)", 0, 30, 5, 1) / 100.0
    with p3:
        closed_pos = st.slider("폐쇄몰 위치(HS~라방 사이) (%)", 0, 100, 60, 5)

    st.divider()
    st.subheader("5) (선택) 시장밴드(리서치)로 온라인 앵커 보정")
    use_market_band = st.toggle("시장밴드 사용", value=False)
    market_positioning = st.slider("시장 밴드 내 상시가 포지셔닝(%)", 0, 100, 50, 5)
    st.caption("ON이면 리서치 입력(국내/경쟁/해외/직구)으로 RefBand를 만들고, 상시가를 밴드 내 포지셔닝 값으로 끌어옵니다. 단, 손익/정책 하한이 우선합니다.")

    st.divider()
    st.subheader("6) 세트 Reference 기준가")
    set_ref_price_type = st.selectbox("세트 할인율 평가 기준", ["상시할인가", "브랜드위크가"], index=0)

    st.divider()
    st.subheader("6-1) 세트 할인율 룰(키인/수정)")
    st.caption("Reference(기준가) 대비 세트 할인율 권장 범위를 직접 입력합니다. (경고/진단에 사용)")
    st.session_state["set_rule"] = st.data_editor(
        st.session_state["set_rule"],
        use_container_width=True,
        height=240,
        num_rows="fixed",
        column_config={
            "세트할인_min": st.column_config.NumberColumn("세트할인_min", format="%.2f"),
            "세트할인_max": st.column_config.NumberColumn("세트할인_max", format="%.2f"),
        }
    )

    st.divider()
    st.subheader("7) 온라인 기준 채널(상시/브위/원데이 적용 수수료)")
    channels = st.session_state["channel_cfg"]["채널"].tolist()
    default_anchor = "자사몰" if "자사몰" in channels else channels[0]
    online_anchor_channel = st.selectbox("온라인(상시/브위/원데이) 기준 채널", channels, index=channels.index(default_anchor))

    st.divider()
    st.subheader("8) 계산 실행")
    if st.session_state["inputs_df"].empty:
        st.warning("데이터 업로드/선택 탭에서 상품을 먼저 불러와 주세요.")
    else:
        if st.button("계산 실행", type="primary"):
            out, warn_df, set_diag = compute_for_all(
                st.session_state["inputs_df"],
                channel_cfg=st.session_state["channel_cfg"],
                online_anchor_channel=online_anchor_channel,
                pg_rate=pg_rate,
                min_cm=min_cm,
                cost_ratio_limit=cost_ratio_limit,
                rounding_unit=rounding_unit,
                gap_rate=gap_rate,
                gap_abs=gap_abs,
                auto_correct=auto_correct,
                d_homesale=d_homesale,
                d_brandweek=d_brandweek,
                d_oneday=d_oneday,
                d_live=d_live,
                list_disc=list_disc,
                hs_under_live=hs_under_live,
                gb_under_hs=gb_under_hs,
                closed_pos_pct=closed_pos,
                offline_disc_from_msrp=offline_disc,
                use_market_band=use_market_band,
                market_positioning_pct=market_positioning,
                band_pct=band_pct,
                set_ref_price_type=set_ref_price_type,
                set_discount_rule_df=st.session_state["set_rule"],
            )

            st.session_state["result_out"] = out
            st.session_state["result_warn"] = warn_df
            st.session_state["result_setdiag"] = set_diag

    # Results section (render if exists)
    if "result_out" in st.session_state:
        out = st.session_state["result_out"]
        warn_df = st.session_state.get("result_warn", pd.DataFrame())
        set_diag = st.session_state.get("result_setdiag", pd.DataFrame())

        st.divider()
        st.subheader("추천 가격 밴드 결과 (Min / Target / Max)")
        if out.empty:
            st.warning("결과가 없습니다. (원가 입력 누락 등)")
        else:
            st.dataframe(out, use_container_width=True, height=420)

            st.subheader("타겟가 피벗(요약)")
            pv = out.pivot_table(index=["품번", "신규품명"], columns="가격타입", values="Target", aggfunc="first")
            pv = pv.reindex(columns=PRICE_LADDER_LOW_TO_HIGH, fill_value=np.nan)
            st.dataframe(pv.reset_index(), use_container_width=True, height=320)

        st.divider()
        st.subheader("세트 할인율 진단(Reference 대비)")
        if set_diag is None or set_diag.empty:
            st.info("세트 진단 데이터가 없습니다.")
        else:
            st.dataframe(set_diag, use_container_width=True, height=260)

        st.divider()
        st.subheader("룰 위반/자동보정/경고 로그")
        if warn_df is None or warn_df.empty:
            st.success("경고 없음(현재 설정 기준)")
        else:
            st.warning(f"{len(warn_df):,}건")
            st.dataframe(warn_df, use_container_width=True, height=260)

        st.divider()
        st.subheader("결과 엑셀 다운로드")
        xbytes = to_excel_bytes({
            "result_long": out if out is not None else pd.DataFrame(),
            "result_pivot": pv.reset_index() if "pv" in locals() else pd.DataFrame(),
            "set_diagnosis": set_diag if set_diag is not None else pd.DataFrame(),
            "warnings": warn_df if warn_df is not None else pd.DataFrame(columns=["품번", "신규품명", "메시지"]),
            "channel_config": st.session_state["channel_cfg"],
            "set_rule": st.session_state["set_rule"],
        })
        st.download_button(
            "엑셀 다운로드",
            data=xbytes,
            file_name="pricing_result_v3.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )