import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# ============================================================
# IBR Pricing Simulator v6.5
# (Cost-only → Auto prices → You adjust) + Calibration/Validation
#
# v6.3 adds:
# - Upload historical "운영 가격표" (set header + preceding components)
# - Parse into Set_Prices / Set_BOM
# - Calibrate set discount table (Disc) by reverse-engineering from history
# - Validate predicted vs actual (accuracy within tolerance)
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator v6.5", layout="wide")

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

DEFAULT_BOUNDARIES = [0, 10, 20, 30, 42, 52, 62, 72, 84, 94, 100]

def default_zone_target_pos(boundaries):
    return {z: (boundaries[i] + boundaries[i+1]) / 2 for i, z in enumerate(PRICE_ZONES)}

# -----------------------------
# Set Pricing Defaults
# -----------------------------
GIFT_KEYWORDS = ["쇼핑백", "트레이", "틴케이스", "스푼", "선물", "기프트", "포장", "케이스"]
SET_TYPES = ["multi", "assort", "gift"]

def make_default_set_disc_df():
    rows = []
    defaults = {
        "multi": {
            "공구": 45, "홈쇼핑": 55, "폐쇄몰": 35, "모바일라방": 40, "원데이": 30,
            "브랜드위크": 25, "홈사": 25, "상시": 3, "오프라인": 0, "MSRP": 0
        },
        "assort": {
            "공구": 42, "홈쇼핑": 55, "폐쇄몰": 33, "모바일라방": 38, "원데이": 28,
            "브랜드위크": 23, "홈사": 23, "상시": 10, "오프라인": 5, "MSRP": 0
        },
        "gift": {
            "공구": 45, "홈쇼핑": 58, "폐쇄몰": 35, "모바일라방": 40, "원데이": 30,
            "브랜드위크": 25, "홈사": 25, "상시": 12, "오프라인": 5, "MSRP": 0
        },
    }
    for stype in SET_TYPES:
        for z in PRICE_ZONES:
            rows.append({"세트타입": stype, "가격영역": z, "할인율(%)": defaults[stype].get(z, 0)})
    return pd.DataFrame(rows)

DEFAULT_SET_PARAMS = {
    "k_msrp_set_multi": 1.00,
    "k_msrp_set_assort": 0.98,
    "k_msrp_set_gift": 1.03,
    "pack_cost_default": 0.0,
    "pack_cost_gift": 700.0,
    "disc_pack_step_pct": 2.0,   # add = step * log2(pack_units)
    "disc_pack_cap_pct": 6.0,
    "hero_boost": 0.6,
}

# -----------------------------
# Utilities
# -----------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s == "-":
                return default
            s = s.replace(",", "")
            return float(s)
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

import importlib.util, zipfile

def _pick_excel_engine():
    """Return a pandas ExcelWriter engine that exists in the runtime, else None."""
    for eng, mod in [("openpyxl", "openpyxl"), ("xlsxwriter", "xlsxwriter")]:
        try:
            if importlib.util.find_spec(mod) is not None:
                return eng
        except Exception:
            continue
    return None

def to_excel_bytes(df_dict):
    """
    Returns: (bytes, ext, mime)
    - If an Excel engine exists (openpyxl or xlsxwriter): create .xlsx
    - Else: create .zip of UTF-8-SIG CSVs (one file per sheet)
    """
    eng = _pick_excel_engine()
    if eng is not None:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine=eng) as writer:
            for sh, df in df_dict.items():
                df.to_excel(writer, index=False, sheet_name=str(sh)[:31])
        return bio.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    bio = BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sh, df in df_dict.items():
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            zf.writestr(f"{str(sh)[:31]}.csv", csv)
    return bio.getvalue(), "zip", "application/zip"

# -----------------------------
# Market anchor helpers (v6.5)
# -----------------------------
MARKET_TYPES = ["국내(신규)", "해외(동일)"]

def compute_market_anchors_from_row(row: pd.Series, default_always_disc: float):
    """
    (msrp_mkt, always_mkt, disc_used, note) 반환

    입력 우선순위
    1) MSRP_시장 / 상시_시장 (직접 입력)
    2) 없으면:
       - 해외(동일): 동일상품_시장가 × k_brand × k_pack × k_tax
       - 국내(신규): 경쟁카테고리_기준가 × k_pos × k_brand × k_pack
    3) 상시는: 상시_시장 직접입력 or MSRP시장×(1-상시할인율_시장)
    """
    # 할인율(시장) 기본값
    disc_pct = safe_float(row.get("상시할인율_시장(%)", np.nan), np.nan)
    if disc_pct == disc_pct:
        disc_used = min(max(disc_pct / 100.0, 0.0), 0.95)
    else:
        disc_used = min(max(float(default_always_disc or 0.0), 0.0), 0.95)

    msrp_direct = safe_float(row.get("MSRP_시장", np.nan), np.nan)
    always_direct = safe_float(row.get("상시_시장", np.nan), np.nan)

    # 1) MSRP_시장 직접입력
    if msrp_direct == msrp_direct and msrp_direct > 0:
        msrp_mkt = float(msrp_direct)
        note_msrp = "MSRP_시장(직접)"
    else:
        mtype = str(row.get("시장구분", MARKET_TYPES[0]))
        p_same = safe_float(row.get("동일상품_시장가", np.nan), np.nan)
        p_cat = safe_float(row.get("경쟁카테고리_기준가", np.nan), np.nan)
        k_pos = safe_float(row.get("포지셔닝계수(k_pos)", 1.0), 1.0)
        k_brand = safe_float(row.get("브랜드계수(k_brand)", 1.0), 1.0)
        k_pack = safe_float(row.get("패키징계수(k_pack)", 1.0), 1.0)
        k_tax = safe_float(row.get("세금/환율계수(k_tax)", 1.0), 1.0)

        msrp_mkt = np.nan
        note_msrp = "MSRP_시장값 없음"
        if ("해외" in mtype) and (p_same == p_same and p_same > 0):
            msrp_mkt = float(p_same) * float(k_brand) * float(k_pack) * float(k_tax)
            note_msrp = "동일상품_시장가 기반"
        elif ("국내" in mtype) and (p_cat == p_cat and p_cat > 0):
            msrp_mkt = float(p_cat) * float(k_pos) * float(k_brand) * float(k_pack)
            note_msrp = "경쟁카테고리_기준가 기반"

    # 2) 상시_시장 직접입력
    if always_direct == always_direct and always_direct > 0:
        always_mkt = float(always_direct)
        note_always = "상시_시장(직접)"
    else:
        if msrp_mkt == msrp_mkt and msrp_mkt > 0:
            always_mkt = float(msrp_mkt) * (1.0 - disc_used)
            note_always = "MSRP시장×(1-상시할인율)"
        else:
            always_mkt = np.nan
            note_always = "상시 시장값 없음"

    return msrp_mkt, always_mkt, disc_used, f"{note_msrp} / {note_always}"


# -----------------------------
# Excel template (cost master upload)
# -----------------------------
# ---- Cost master template (no Excel deps) ----
_COST_TEMPLATE_B64 = """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"""

def make_cost_master_template_bytes():
    """Return the prebuilt .xlsx template bytes (base64-embedded)."""
    return base64.b64decode(_COST_TEMPLATE_B64.encode("ascii"))


# -----------------------------
# Load cost master
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
    # ---- (선택) 시장 앵커 입력 컬럼(v6.5) ----
    # 직접 입력(가장 우선)
    out["MSRP_시장"] = np.nan
    out["상시_시장"] = np.nan
    out["상시할인율_시장(%)"] = 20.0

    # 근거 입력(직접 입력이 없을 때 사용)
    out["시장구분"] = MARKET_TYPES[0]
    out["동일상품_시장가"] = np.nan
    out["경쟁카테고리_기준가"] = np.nan

    # 보정 계수(기본 1.0)
    out["포지셔닝계수(k_pos)"] = 1.00
    out["브랜드계수(k_brand)"] = 1.00
    out["패키징계수(k_pack)"] = 1.00
    out["세금/환율계수(k_tax)"] = 1.00
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
# Auto range from cost
# -----------------------------
def compute_auto_range_from_cost(
    cost_total: float,
    channels_df: pd.DataFrame,
    zone_map: dict,
    boundaries: list,
    rounding_unit: int,
    min_cm: float,
    min_cost_ratio_cap: float,         # 예: 0.30  (최저가 원가율 상한)
    always_cost_ratio_target: float,   # 예: 0.18  (상시 목표 원가율)
    always_list_disc: float,           # 예: 0.20  (상시할인율: MSRP 대비)
    include_zones: list,
    min_zone: str = "공구",
    msrp_override=np.nan,
    min_override=np.nan,
    max_override=np.nan,
    market_msrp=np.nan,
    market_always=np.nan,
    market_always_disc=np.nan,
    msrp_min_gap_pct=0.15,
):
    """
    ✅ v6.4 정책(중요): '원가율 30% 상한'은 MSRP가 아니라 "최저가(Min) 바닥" 룰로 적용.

    1) Min 후보
       - Min_floor = min_zone의 손익하한(Floor)
       - Min_ratio = 원가 / min_cost_ratio_cap
       => Min = max(Min_floor, Min_ratio) (Min_오버라이드가 있으면 최우선)

    2) MSRP 후보
       - Always_target = 원가 / always_cost_ratio_target  (상시 목표 원가율)
       - MSRP_base = Always_target / (1 - always_list_disc)  (정가 프레이밍)
       - + 채널별 Floor가 각 영역 BandHigh 안에 들어오도록 MSRP 필요 시 상향
       - + Max_오버라이드/MSRP_오버라이드가 있으면 최우선 후보로 포함

    3) 최종
       - Min, Max는 rounding_unit 기준으로 올림(ceil) 처리
       - Max <= Min이면 최소 스팬을 강제로 부여
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
            fee=float(p["수수료율"]),
            pg=float(p["PG"]),
            mkt=float(p["마케팅비"]),
            ship_per_order=float(p["배송비(주문당)"]),
            ret_rate=float(p["반품률"]),
            ret_cost_order=float(p["반품비(주문당)"]),
            min_cm=min_cm,
        )

    # -----------------
    # Min (최저가)
    # -----------------
    min_floor = zone_floor(min_zone)

    min_ratio = np.nan
    if cost_total == cost_total and cost_total > 0 and min_cost_ratio_cap and float(min_cost_ratio_cap) > 0:
        min_ratio = float(cost_total) / float(min_cost_ratio_cap)

    if min_override == min_override and float(min_override) > 0:
        min_auto = float(min_override)
        min_note = "Min 오버라이드 적용"
    else:
        candidates = []
        if min_floor == min_floor and min_floor > 0:
            candidates.append(float(min_floor))
        if min_ratio == min_ratio and min_ratio > 0:
            candidates.append(float(min_ratio))
        if not candidates:
            return np.nan, np.nan, {"note": "Min 산출 불가: 원가/채널 파라미터를 확인하세요."}
        min_auto = max(candidates)
        min_note = "Min = max(손익하한, 원가/원가율상한)"

    min_auto = krw_ceil(min_auto, rounding_unit)

    # -----------------
    # MSRP/상시 앵커 (v6.5: 시장 입력 반영)
    # -----------------
    # 1) 시장 상시/정가가 있으면 그걸 우선 사용
    # 2) 없으면(시장값 미입력) 기존처럼 원가 기반 정책(상시 목표 원가율/상시할인율)로 대체

    # (fallback) 원가 기반 상시 타깃
    always_target = np.nan
    if cost_total == cost_total and cost_total > 0 and always_cost_ratio_target and float(always_cost_ratio_target) > 0:
        always_target = float(cost_total) / float(always_cost_ratio_target)
        always_target = krw_ceil(always_target, rounding_unit)

    # 상시 존 손익하한
    floor_always = zone_floor("상시")

    # 상시할인율(시장 입력이 있으면 그 값을 사용)
    disc_used = float(always_list_disc or 0.0)
    if market_always_disc == market_always_disc:
        disc_used = float(market_always_disc)
    disc_used = min(max(disc_used, 0.0), 0.95)

    # 상시 앵커
    if market_always == market_always and float(market_always) > 0:
        always_anchor = float(market_always)
        always_note = "상시=시장상시 기반"
    else:
        always_anchor = float(always_target) if always_target == always_target and always_target > 0 else np.nan
        always_note = "상시=원가정책 기반(시장값 없을 때)"

    # 손익하한보다 낮으면 올림
    if floor_always == floor_always and floor_always > 0:
        if always_anchor == always_anchor and always_anchor > 0:
            always_anchor = max(float(always_anchor), float(floor_always))
        else:
            always_anchor = float(floor_always)

    # 상시에서 MSRP 프레이밍(정가) 역산
    msrp_base = np.nan
    if always_anchor == always_anchor and always_anchor > 0:
        msrp_base = float(always_anchor) / (1.0 - disc_used)
        msrp_base = krw_ceil(msrp_base, rounding_unit)

    # MSRP 앵커(시장MSRP가 있으면 우선)
    msrp_anchor = np.nan
    if market_msrp == market_msrp and float(market_msrp) > 0:
        cands = [float(market_msrp)]
        if msrp_base == msrp_base and msrp_base > 0:
            cands.append(float(msrp_base))
        msrp_anchor = max(cands)
        msrp_note = "MSRP=시장MSRP 기반"
    else:
        msrp_anchor = msrp_base
        msrp_note = "MSRP=원가정책 기반(시장값 없을 때)"

    # 최소 간격(밴드 의미 확보): MSRP >= Min*(1+gap)
    try:
        gap = float(msrp_min_gap_pct or 0.15)
    except Exception:
        gap = 0.15
    gap = max(0.0, min(0.95, gap))
    if msrp_anchor == msrp_anchor and msrp_anchor > 0:
        msrp_anchor = max(float(msrp_anchor), float(min_auto) * (1.0 + gap))
    else:
        msrp_anchor = float(min_auto) * (1.0 + gap)
    msrp_anchor = krw_ceil(msrp_anchor, rounding_unit)

# -----------------
    # BandHigh가 각 존 Floor를 담을 수 있도록 Max 자동 상향
    # -----------------
    max_req = []
    for i, z in enumerate(PRICE_ZONES):
        if z not in include_zones:
            continue
        fz = zone_floor(z)
        if fz != fz or fz <= 0:
            continue
        end = boundaries[i+1] / 100.0
        if end <= 0:
            continue
        if fz > min_auto:
            max_needed = min_auto + (fz - min_auto) / end
            max_req.append(max_needed)

    candidates = []
    if msrp_anchor == msrp_anchor and msrp_anchor > 0:
        candidates.append(float(msrp_anchor))
    if max_req:
        candidates.append(float(max(max_req)))
    if max_override == max_override and float(max_override) > 0:
        candidates.append(float(max_override))
    if msrp_override == msrp_override and float(msrp_override) > 0:
        candidates.append(float(msrp_override))

    if not candidates:
        max_auto = min_auto + rounding_unit * 20
        max_note = "MSRP 자동값 부족 → 임시 스팬 부여"
    else:
        max_auto = float(max(candidates))
        max_note = "MSRP = max(상시정책기반, 채널Floor충족, 오버라이드)"

    max_auto = krw_ceil(max_auto, rounding_unit)

    # Ensure spread
    if max_auto <= min_auto:
        max_auto = min_auto + max(rounding_unit * 20, int(min_auto * 0.2))
        max_auto = krw_ceil(max_auto, rounding_unit)

    note = f"{min_note} / {always_note} / {msrp_note} / {max_note}"
    if max_req:
        if msrp_anchor == msrp_anchor and max_auto > msrp_anchor:
            note += " (채널 손익하한이 밴드에 들어오도록 MSRP 자동 상향 포함)"
        if msrp_anchor != msrp_anchor:
            note += " (채널 손익하한이 밴드에 들어오도록 MSRP 자동 상향 포함)"

    meta = {
        "note": note,
        "min_floor": float(min_floor) if min_floor == min_floor else np.nan,
        "min_ratio": float(min_ratio) if min_ratio == min_ratio else np.nan,
        "always_target(fallback)": float(always_target) if always_target == always_target else np.nan,
        "always_anchor": float(always_anchor) if always_anchor == always_anchor else np.nan,
        "msrp_base(from_always)": float(msrp_base) if msrp_base == msrp_base else np.nan,
        "msrp_anchor": float(msrp_anchor) if msrp_anchor == msrp_anchor else np.nan,
        "disc_used": float(disc_used),
    }
    return float(min_auto), float(max_auto), meta

def build_zone_table(
    cost_total: float, min_price: float, max_price: float,
    channels_df: pd.DataFrame, zone_map: dict, boundaries: list, target_pos: dict,
    rounding_unit: int, min_cm: float, overrides_df: pd.DataFrame,
    item_type: str, item_id: str
):
    ch_map = channels_df.set_index("채널명").to_dict("index")
    if min_price != min_price or max_price != max_price or max_price <= min_price:
        return pd.DataFrame(columns=['가격영역', '비용채널', 'BandLow', 'BandHigh', 'Floor(손익하한)', '추천가(Target)', '가격_오버라이드(원)', '최종가격(원)', '상태', '경고', '마진룸(원)=최종-Floor', '기여이익(원)', '기여이익률(%)'])

    rows = []
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

        floor = floor_price(cost_total, 1, p["수수료율"], p["PG"], p["마케팅비"], p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"], min_cm)

        status = "OK"
        target = max(target_raw, floor)
        if floor > band_high:
            status = "불가(손익하한이 너무 높음)"; target = band_high
        elif target > band_high:
            status = "상단에 맞춤"; target = band_high

        ov = overrides_df[(overrides_df["오퍼타입"]==item_type) & (overrides_df["오퍼ID"]==item_id) & (overrides_df["가격영역"]==z)]
        override_price = safe_float(ov.iloc[0]["가격_오버라이드"], np.nan) if not ov.empty else np.nan
        effective = override_price if (override_price == override_price and override_price > 0) else target

        band_low_r = krw_round(band_low, rounding_unit)
        band_high_r = krw_round(band_high, rounding_unit)
        floor_r = krw_round(floor, rounding_unit)
        target_r = krw_round(target, rounding_unit)
        eff_r = krw_round(effective, rounding_unit) if (effective == effective and effective > 0) else np.nan

        cm, cmr = contrib_metrics(eff_r if eff_r==eff_r else 0, cost_total, 1, p["수수료율"], p["PG"], p["마케팅비"], p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"])

        flags = []
        if eff_r == eff_r and eff_r < floor_r: flags.append("⚠️손익하한 미만")
        if eff_r == eff_r and eff_r < band_low_r: flags.append("⚠️영역 하단 미만")
        if eff_r == eff_r and eff_r > band_high_r and z != "MSRP": flags.append("⚠️영역 상단 초과")

        rows.append({
            "가격영역": z, "비용채널": ch,
            "BandLow": band_low_r, "BandHigh": band_high_r,
            "Floor(손익하한)": floor_r,
            "추천가(Target)": target_r,
            "가격_오버라이드(원)": (krw_round(override_price, rounding_unit) if override_price == override_price else np.nan),
            "최종가격(원)": eff_r,
            "상태": status, "경고": " / ".join(flags),
            "마진룸(원)=최종-Floor": (eff_r - floor_r) if (eff_r == eff_r) else np.nan,
            "기여이익(원)": int(round(cm)) if cm == cm else np.nan,
            "기여이익률(%)": round(cmr*100, 1) if cmr == cmr else np.nan,
        })
    return pd.DataFrame(rows)

# -----------------------------
# Set helpers
# -----------------------------
def is_accessory_sku(sku: str, name: str, cost: float) -> bool:
    sku = str(sku or "")
    name = str(name or "")
    if sku.upper().startswith("U"): return True
    for kw in GIFT_KEYWORDS:
        if kw in name: return True
    try:
        if float(cost) > 0 and float(cost) <= 800: return True
    except Exception:
        pass
    return False

def classify_set(set_id: str, bom_df: pd.DataFrame, products_df: pd.DataFrame):
    b = bom_df[bom_df["세트ID"] == set_id].copy()
    if b.empty:
        return {"set_type":"assort", "non_acc_units":0, "hero_sku":None}

    b["수량"] = pd.to_numeric(b["수량"], errors="coerce").fillna(0).astype(int)
    b = b.merge(products_df[["품번","상품명","원가"]], on="품번", how="left")
    b["원가"] = pd.to_numeric(b["원가"], errors="coerce").fillna(0.0)

    b["is_acc"] = b.apply(lambda r: is_accessory_sku(r["품번"], r.get("상품명",""), r.get("원가",0)), axis=1)
    non_acc = b[~b["is_acc"]].copy()
    non_acc_units = int(non_acc["수량"].sum())
    unique_non_acc = int(non_acc["품번"].nunique())
    is_gift = bool(b["is_acc"].any())

    base_type = "multi" if (unique_non_acc <= 1 and non_acc_units > 0) else "assort"
    set_type = "gift" if is_gift else base_type

    hero_sku = None
    if not non_acc.empty:
        hero_sku = str(non_acc.sort_values("원가", ascending=False).iloc[0]["품번"])

    return {"set_type": set_type, "non_acc_units": non_acc_units, "hero_sku": hero_sku, "detail_df": b}


def estimate_sku_msrp_from_cost(
    sku_cost: float,
    channels_df,
    zone_map,
    boundaries,
    rounding_unit,
    min_cm,
    min_cost_ratio_cap,
    always_cost_ratio_target,
    always_list_disc,
    market_msrp=np.nan,
    market_always=np.nan,
    market_always_disc=np.nan,
    msrp_min_gap_pct=0.15,
):
    """원가만 있을 때 SKU의 Max(=MSRP) 추정치 (v6.4: Min-원가율 룰 + 상시정책 기반)"""
    if sku_cost != sku_cost or sku_cost <= 0:
        return np.nan
    _, max_auto, _ = compute_auto_range_from_cost(
        cost_total=float(sku_cost),
        channels_df=channels_df,
        zone_map=zone_map,
        boundaries=boundaries,
        rounding_unit=rounding_unit,
        min_cm=min_cm,
        min_cost_ratio_cap=min_cost_ratio_cap,
        always_cost_ratio_target=always_cost_ratio_target,
        always_list_disc=always_list_disc,
        market_msrp=market_msrp,
        market_always=market_always,
        market_always_disc=market_always_disc,
        msrp_min_gap_pct=msrp_min_gap_pct,
        include_zones=PRICE_ZONES,
        min_zone="공구",
        msrp_override=np.nan,
        min_override=np.nan,
        max_override=np.nan,
    )
    return float(max_auto) if max_auto == max_auto else np.nan

def compute_predicted_sku_always(
    products_df: pd.DataFrame,
    channels_df: pd.DataFrame,
    zone_map: dict,
    boundaries: list,
    rounding_unit: int,
    min_cm: float,
    min_cost_ratio_cap: float,
    always_cost_ratio_target: float,
    always_list_disc: float,
    msrp_min_gap_pct: float,
    overrides_df: pd.DataFrame,
):
    """
    세트 BASE 산출용: SKU의 '상시' 가격(최종)을 예측.
    - 원가만으로 Min/Max를 생성(v6.4 정책)
    - zone_table에서 '상시' 최종가격을 가져옴
    - zdf가 비거나 컬럼이 없으면 안전하게 skip (KeyError 방지)
    """
    sku_always = {}

    if products_df is None or products_df.empty:
        return sku_always

    # SKU별 상시 예측은 "기본 Target 위치"로 산출 (오버라이드는 그대로 반영)
    default_tp = default_zone_target_pos(boundaries)

    for _, rr in products_df.iterrows():
        sku = str(rr.get("품번", "")).strip()
        if not sku:
            continue
        cost = safe_float(rr.get("원가", np.nan), np.nan)
        if cost != cost or cost <= 0:
            continue

        msrp_mkt, always_mkt, disc_used, _mnote = compute_market_anchors_from_row(rr, default_always_disc=always_list_disc)

        min_auto, max_auto, _ = compute_auto_range_from_cost(
            cost_total=cost,
            channels_df=channels_df,
            zone_map=zone_map,
            boundaries=boundaries,
            rounding_unit=rounding_unit,
            min_cm=min_cm,
            min_cost_ratio_cap=min_cost_ratio_cap,
            always_cost_ratio_target=always_cost_ratio_target,
            always_list_disc=always_list_disc,
            market_msrp=msrp_mkt,
            market_always=always_mkt,
            market_always_disc=disc_used,
            msrp_min_gap_pct=msrp_min_gap_pct,
            include_zones=PRICE_ZONES,
            min_zone="공구",
            msrp_override=safe_float(rr.get("MSRP_오버라이드", np.nan), np.nan),
            min_override=safe_float(rr.get("Min_오버라이드", np.nan), np.nan),
            max_override=safe_float(rr.get("Max_오버라이드", np.nan), np.nan),
        )

        zdf = build_zone_table(
            cost_total=cost,
            min_price=float(min_auto),
            max_price=float(max_auto),
            channels_df=channels_df,
            zone_map=zone_map,
            boundaries=boundaries,
            target_pos=default_tp,
            rounding_unit=rounding_unit,
            min_cm=min_cm,
            overrides_df=overrides_df,
            item_type="SKU",
            item_id=sku,
        )

        if zdf is None or zdf.empty or "가격영역" not in zdf.columns:
            continue

        ar = zdf[zdf["가격영역"] == "상시"]
        if ar.empty:
            continue

        p = safe_float(ar.iloc[0].get("최종가격(원)", np.nan), np.nan)
        if p == p and p > 0:
            sku_always[sku] = float(p)

    return sku_always

def compute_set_anchors(set_id: str, bom_df: pd.DataFrame, products_df: pd.DataFrame,
                        sku_always: dict, params: dict,
                        channels_df, zone_map, boundaries, rounding_unit, min_cm,
                        min_cost_ratio_cap, always_cost_ratio_target, always_list_disc, msrp_min_gap_pct=0.15):
    cls = classify_set(set_id, bom_df, products_df)
    b = cls.get("detail_df", pd.DataFrame()).copy()
    if b.empty: return None

    # SKU별 시장 앵커 조회용
    prod_map = products_df.set_index("품번").to_dict("index") if (products_df is not None and not products_df.empty) else {}

    set_type = cls["set_type"]
    pack_cost = float(params.get("pack_cost_gift", 700.0)) if set_type=="gift" else float(params.get("pack_cost_default", 0.0))

    b["is_acc"] = b.apply(lambda r: is_accessory_sku(r["품번"], r.get("상품명",""), r.get("원가",0)), axis=1)
    b["상시_ref"] = b["품번"].astype(str).map(sku_always).astype(float).fillna(0.0)
    b.loc[b["is_acc"], "상시_ref"] = 0.0
    base_sum = float((b["상시_ref"] * b["수량"]).sum())

    # msrp sum estimate (exclude accessories)
    msrp_sum = 0.0
    for _, rr in b.iterrows():
        if rr["is_acc"]: 
            continue
        sku_key = str(rr.get("품번", "")).strip()
        prow = prod_map.get(sku_key, None)
        if prow is None:
            msrp_mkt = np.nan
            always_mkt = np.nan
            disc_used = np.nan
        else:
            msrp_mkt, always_mkt, disc_used, _ = compute_market_anchors_from_row(pd.Series(prow), default_always_disc=always_list_disc)

        sku_msrp = estimate_sku_msrp_from_cost(
            safe_float(rr.get("원가", np.nan), np.nan),
            channels_df, zone_map, boundaries, rounding_unit, min_cm,
            min_cost_ratio_cap, always_cost_ratio_target, always_list_disc,
            market_msrp=msrp_mkt,
            market_always=always_mkt,
            market_always_disc=disc_used,
            msrp_min_gap_pct=msrp_min_gap_pct,
        )
        if sku_msrp == sku_msrp:
            msrp_sum += float(sku_msrp) * int(rr["수량"])

    k = float(params.get("k_msrp_set_multi", 1.00)) if set_type=="multi" else (
        float(params.get("k_msrp_set_assort", 0.98)) if set_type=="assort" else float(params.get("k_msrp_set_gift", 1.03))
    )
    msrp_set_sum = msrp_sum * k
    pack_units = int(cls.get("non_acc_units", 0)) if cls.get("non_acc_units",0) > 0 else 1

    if base_sum <= 0 and msrp_sum > 0:
        base_sum = msrp_sum * 0.85

    return {"set_type": set_type, "pack_cost": pack_cost, "pack_units": pack_units,
            "base_sum": base_sum, "msrp_sum_est": msrp_sum, "msrp_set_sum": msrp_set_sum,
            "hero_sku": cls.get("hero_sku"), "detail_df": b}

def compute_set_range(cost_total: float, anchors: dict, channels_df, zone_map, boundaries, rounding_unit, min_cm,
                  min_cost_ratio_cap, always_cost_ratio_target, always_list_disc,
                  msrp_min_gap_pct=0.15,
                  msrp_override=np.nan, min_override=np.nan, max_override=np.nan):
    min_auto, max_cost, meta = compute_auto_range_from_cost(
        cost_total=cost_total, channels_df=channels_df, zone_map=zone_map, boundaries=boundaries,
        rounding_unit=rounding_unit, min_cm=min_cm, min_cost_ratio_cap=min_cost_ratio_cap,
                always_cost_ratio_target=always_cost_ratio_target,
                always_list_disc=always_list_disc,
                msrp_min_gap_pct=msrp_min_gap_pct,

        include_zones=PRICE_ZONES, min_zone="공구", msrp_override=np.nan
    )
    candidates = [max_cost] if max_cost==max_cost else []
    if anchors and anchors.get("msrp_set_sum", np.nan) == anchors.get("msrp_set_sum", np.nan):
        candidates.append(float(anchors["msrp_set_sum"]))
    if msrp_override == msrp_override and msrp_override > 0:
        candidates.append(float(msrp_override))
    max_auto = krw_ceil(max(candidates), rounding_unit) if candidates else max_cost
    return float(min_auto), float(max_auto), meta

def build_zone_table_set(cost_total: float, min_price: float, max_price: float, anchors: dict,
                         channels_df, zone_map, boundaries, rounding_unit, min_cm,
                         overrides_df, disc_df, params, item_id: str):
    ch_map = channels_df.set_index("채널명").to_dict("index")
    if min_price != min_price or max_price != max_price or max_price <= min_price or anchors is None:
        return pd.DataFrame(columns=['가격영역', '세트타입', '팩수량(부자재제외)', 'Disc(%)', '비용채널', 'BandLow', 'BandHigh', 'Floor(손익하한)', '추천가(Target)', '가격_오버라이드(원)', '최종가격(원)', '상태', '경고', '마진룸(원)=최종-Floor', '기여이익(원)', '기여이익률(%)'])
    rows = []
    span = max_price - min_price
    set_type = anchors["set_type"]
    base_sum = float(anchors.get("base_sum", 0.0))
    pack_units = int(anchors.get("pack_units", 1))

    for i, z in enumerate(PRICE_ZONES):
        start = boundaries[i] / 100.0
        end = boundaries[i+1] / 100.0
        band_low = min_price + span * start
        band_high = min_price + span * end

        ch = zone_map.get(z, "자사몰")
        p = ch_map.get(ch, None)
        if p is None:
            continue
        floor = floor_price(cost_total, 1, p["수수료율"], p["PG"], p["마케팅비"], p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"], min_cm)

        status = "OK"
        if z == "MSRP":
            target = max_price
            disc_pct = 0.0
        else:
            disc_pct = get_set_disc_pct(set_type, z, pack_units, disc_df, params)
            target_raw = base_sum * (1.0 - disc_pct/100.0)
            target = max(target_raw, floor)
            if floor > band_high:
                status = "불가(손익하한이 너무 높음)"; target = band_high
            else:
                if target > band_high:
                    status = "상단에 맞춤"; target = band_high
                if target < band_low:
                    status = "하단에 맞춤"; target = band_low

        ov = overrides_df[(overrides_df["오퍼타입"]=="SET") & (overrides_df["오퍼ID"]==item_id) & (overrides_df["가격영역"]==z)]
        override_price = safe_float(ov.iloc[0]["가격_오버라이드"], np.nan) if not ov.empty else np.nan
        effective = override_price if (override_price==override_price and override_price>0) else target

        band_low_r = krw_round(band_low, rounding_unit)
        band_high_r = krw_round(band_high, rounding_unit)
        floor_r = krw_round(floor, rounding_unit)
        target_r = krw_round(target, rounding_unit)
        eff_r = krw_round(effective, rounding_unit) if (effective==effective and effective>0) else np.nan

        cm, cmr = contrib_metrics(eff_r if eff_r==eff_r else 0, cost_total, 1, p["수수료율"], p["PG"], p["마케팅비"], p["배송비(주문당)"], p["반품률"], p["반품비(주문당)"])

        flags = []
        if eff_r == eff_r and eff_r < floor_r: flags.append("⚠️손익하한 미만")
        if eff_r == eff_r and eff_r < band_low_r: flags.append("⚠️영역 하단 미만")
        if eff_r == eff_r and eff_r > band_high_r and z != "MSRP": flags.append("⚠️영역 상단 초과")

        rows.append({
            "가격영역": z, "세트타입": set_type, "팩수량(부자재제외)": pack_units, "Disc(%)": round(float(disc_pct),1),
            "비용채널": ch, "BandLow": band_low_r, "BandHigh": band_high_r,
            "Floor(손익하한)": floor_r, "추천가(Target)": target_r,
            "가격_오버라이드(원)": (krw_round(override_price, rounding_unit) if override_price==override_price else np.nan),
            "최종가격(원)": eff_r, "상태": status, "경고": " / ".join(flags),
            "마진룸(원)=최종-Floor": (eff_r - floor_r) if eff_r==eff_r else np.nan,
            "기여이익(원)": int(round(cm)) if cm==cm else np.nan,
            "기여이익률(%)": round(cmr*100,1) if cmr==cmr else np.nan,
        })
    return pd.DataFrame(rows)

# -----------------------------
# History parsing & calibration
# -----------------------------
def load_history_table(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_history_to_tables(df_raw: pd.DataFrame):
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # choose product name column (handles duplicated "신규품명")
    name_col = None
    for cand in ["신규품명.1", "신규품명_1", "신규품명 (2)", "신규품명"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        for c in df.columns:
            if "신규품명" in c:
                name_col = c
                break
    if name_col is None:
        name_col = "상품명" if "상품명" in df.columns else None

    def parse_no(x):
        v = safe_float(x, np.nan)
        if v != v: return np.nan
        return int(v)

    df["_no"] = df["No"].apply(parse_no) if "No" in df.columns else np.nan
    df["_sku"] = df["품번"].astype(str).str.strip() if "품번" in df.columns else ""
    df["_name"] = df[name_col].astype(str).str.strip() if (name_col and name_col in df.columns) else ""

    money_cols = ["원가","폐쇄몰","공구가","홈쇼핑","모바일방송가","원데이특가","브랜드위크가","오프라인","상시할인가","소비자가"]
    for c in money_cols:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    def is_set_row(r):
        no = r["_no"]
        sku = str(r["_sku"]).strip()
        nm = str(r["_name"])
        if no != no: 
            return False
        if ("세트" in nm) or ("_세트_" in nm):
            return True
        if sku == "" or sku.lower() in ["nan", "none"]:
            return True
        return False

    def is_component_row(r):
        sku = str(r["_sku"]).strip()
        no = r["_no"]
        if sku == "" or sku.lower() in ["nan", "none"]:
            return False
        return not (no == no)

    components = []
    sets = []
    boms = []
    block_idx = 0

    for _, r in df.iterrows():
        if is_component_row(r):
            components.append({
                "품번": r["_sku"], "상품명": r["_name"],
                "원가": safe_float(r.get("원가", np.nan), np.nan),
                "소비자가": safe_float(r.get("소비자가", np.nan), np.nan),
                "폐쇄몰": safe_float(r.get("폐쇄몰", np.nan), np.nan),
                "공구가": safe_float(r.get("공구가", np.nan), np.nan),
                "홈쇼핑": safe_float(r.get("홈쇼핑", np.nan), np.nan),
                "모바일방송가": safe_float(r.get("모바일방송가", np.nan), np.nan),
                "원데이특가": safe_float(r.get("원데이특가", np.nan), np.nan),
                "브랜드위크가": safe_float(r.get("브랜드위크가", np.nan), np.nan),
                "오프라인": safe_float(r.get("오프라인", np.nan), np.nan),
                "상시할인가": safe_float(r.get("상시할인가", np.nan), np.nan),
            })
            continue

        if is_set_row(r):
            block_idx += 1
            set_id = f"S{block_idx:04d}"
            sets.append({
                "set_id": set_id, "source_No": r["_no"], "set_name": r["_name"],
                "원가": safe_float(r.get("원가", np.nan), np.nan),
                "소비자가": safe_float(r.get("소비자가", np.nan), np.nan),
                "폐쇄몰": safe_float(r.get("폐쇄몰", np.nan), np.nan),
                "공구가": safe_float(r.get("공구가", np.nan), np.nan),
                "홈쇼핑": safe_float(r.get("홈쇼핑", np.nan), np.nan),
                "모바일방송가": safe_float(r.get("모바일방송가", np.nan), np.nan),
                "원데이특가": safe_float(r.get("원데이특가", np.nan), np.nan),
                "브랜드위크가": safe_float(r.get("브랜드위크가", np.nan), np.nan),
                "오프라인": safe_float(r.get("오프라인", np.nan), np.nan),
                "상시할인가": safe_float(r.get("상시할인가", np.nan), np.nan),
            })
            if components:
                comp_df = pd.DataFrame(components)
                comp_df["품번"] = comp_df["품번"].astype(str).str.strip()
                g = comp_df.groupby("품번", as_index=False).agg(qty=("품번","size"), 상품명=("상품명","first"), 원가=("원가","median"))
                for _, cr in g.iterrows():
                    boms.append({"set_id": set_id, "품번": cr["품번"], "수량": int(cr["qty"]), "상품명": cr["상품명"], "원가": cr["원가"]})
            components = []
            continue

    return pd.DataFrame(components), pd.DataFrame(sets), pd.DataFrame(boms)

def zone_from_history_column(col: str) -> str:
    mapping = {
        "폐쇄몰": "폐쇄몰",
        "공구가": "공구",
        "홈쇼핑": "홈쇼핑",
        "모바일방송가": "모바일라방",
        "원데이특가": "원데이",
        "브랜드위크가": "브랜드위크",
        "오프라인": "오프라인",
        "상시할인가": "상시",
        "소비자가": "MSRP",
    }
    return mapping.get(col, "")

def calibrate_set_disc_from_history(set_df, bom_df_hist, products_df, sku_always_pred, params, disc_df):
    if set_df.empty or bom_df_hist.empty:
        return disc_df, pd.DataFrame()

    # convert bom for classifier
    bom_app = bom_df_hist.rename(columns={"set_id":"세트ID"})[["세트ID","품번","수량"]].copy()
    obs_rows = []

    for _, sr in set_df.iterrows():
        sid = sr["set_id"]
        cls = classify_set(sid, bom_app, products_df[["품번","상품명","원가"]].copy())
        set_type = cls.get("set_type","assort")
        pack_units = int(cls.get("non_acc_units",0)) if cls.get("non_acc_units",0)>0 else 1
        detail = cls.get("detail_df", pd.DataFrame()).copy()
        if detail.empty:
            continue
        detail["is_acc"] = detail.apply(lambda r: is_accessory_sku(r["품번"], r.get("상품명",""), r.get("원가",0)), axis=1)
        detail["상시_pred"] = detail["품번"].astype(str).map(sku_always_pred).astype(float).fillna(0.0)
        detail.loc[detail["is_acc"], "상시_pred"] = 0.0
        base_sum = float((detail["상시_pred"] * detail["수량"]).sum())
        if base_sum <= 0:
            continue

        for col in ["폐쇄몰","공구가","홈쇼핑","모바일방송가","원데이특가","브랜드위크가","오프라인","상시할인가"]:
            p = safe_float(sr.get(col, np.nan), np.nan)
            if p != p or p <= 0:
                continue
            zone = zone_from_history_column(col)
            disc_obs = 1.0 - (p / base_sum)
            step = float(params.get("disc_pack_step_pct", 2.0))
            cap = float(params.get("disc_pack_cap_pct", 6.0))
            add = 0.0 if pack_units<=1 else min(cap, step*np.log2(pack_units))
            base_disc = disc_obs*100.0 - add
            obs_rows.append({"set_id":sid,"set_type":set_type,"pack_units":pack_units,"zone":zone,
                             "price_actual":p,"base_sum_pred":base_sum,
                             "disc_obs_pct":disc_obs*100.0,"add_pct":add,"base_disc_pct":base_disc})

    obs = pd.DataFrame(obs_rows)
    if obs.empty:
        return disc_df, obs

    new_disc = disc_df.copy()
    for stype in SET_TYPES:
        for z in PRICE_ZONES:
            if z == "MSRP":
                continue
            sub = obs[(obs["set_type"]==stype) & (obs["zone"]==z)]
            if sub.empty:
                continue
            med = float(np.nanmedian(sub["base_disc_pct"].values))
            new_disc.loc[(new_disc["세트타입"]==stype) & (new_disc["가격영역"]==z), "할인율(%)"] = round(max(0.0, min(95.0, med)), 1)

    return new_disc, obs

# -----------------------------
# Session state
# -----------------------------
if "products_df" not in st.session_state:
    st.session_state["products_df"] = pd.DataFrame(columns=["품번","상품명","브랜드","원가","MSRP_시장","상시_시장","상시할인율_시장(%)","시장구분","동일상품_시장가","경쟁카테고리_기준가","포지셔닝계수(k_pos)","브랜드계수(k_brand)","패키징계수(k_pack)","세금/환율계수(k_tax)","MSRP_오버라이드","Min_오버라이드","Max_오버라이드","운영여부"])
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
if "overrides_df" not in st.session_state:
    st.session_state["overrides_df"] = pd.DataFrame(columns=["오퍼타입","오퍼ID","가격영역","가격_오버라이드"])
if "set_disc_df" not in st.session_state:
    st.session_state["set_disc_df"] = make_default_set_disc_df()
if "set_params" not in st.session_state:
    st.session_state["set_params"] = DEFAULT_SET_PARAMS.copy()
if "history_set_df" not in st.session_state:
    st.session_state["history_set_df"] = pd.DataFrame()
if "history_bom_df" not in st.session_state:
    st.session_state["history_bom_df"] = pd.DataFrame()

# -----------------------------
# UI
# -----------------------------
st.title("IBR 가격 시뮬레이터 v6.5")
st.caption("원가만 업로드 → 자동 가격 생성 → (추가) 기존 운영 데이터로 세트 할인율 역산/검증")

tab_up, tab_cal, tab_sku, tab_set, tab_logic = st.tabs(
    ["1) 업로드/설정", "2) 캘리브레이션(세트 할인율)", "3) 단품(자동→수정)", "4) 세트(BOM)", "5) 로직(문장)"]
)

with tab_up:
    st.subheader("A. 원가/상품명 통일 파일 업로드(필수)")

    st.download_button(
        "원가 업로드 템플릿 다운로드(.xlsx)",
        data=make_cost_master_template_bytes(),
        file_name="원가_상품마스터_업로드양식.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="상품코드/상품명/브랜드/원가(vat-)만 채워서 업로드하면 됩니다."
    )

    up = st.file_uploader("원가/상품마스터 업로드(.xlsx)", type=["xlsx","xls"])
    if up is not None:
        try:
            st.session_state["products_df"] = load_products_from_cost_master(up)
            st.success(f"업로드 완료: {len(st.session_state['products_df']):,}개 SKU")
        except Exception as e:
            st.error(f"업로드 오류: {e}")

    st.metric("현재 SKU 수", f"{len(st.session_state['products_df']):,}")
    st.divider()
    st.subheader("A-1. (선택) 시장 앵커/오버라이드 입력(키인)")
    st.caption("MSRP/상시는 '시장 앵커(동일상품/경쟁카테고리)'가 있으면 우선 반영됩니다. 값 입력 후 자동 가격/세트/캘리브레이션까지 전체에 반영됩니다.")
    if st.session_state["products_df"].empty:
        st.info("먼저 원가 파일을 업로드하세요.")
    else:
        st.session_state["products_df"] = st.data_editor(
            st.session_state["products_df"],
            use_container_width=True,
            height=260,
            num_rows="dynamic",
        )

    st.divider()
    st.subheader("B. 채널 비용(키인 즉시 반영)")
    st.session_state["channels_df"] = st.data_editor(st.session_state["channels_df"], use_container_width=True, num_rows="dynamic", height=260)

    st.divider()
    st.subheader("C. 가격영역(밴드) ↔ 비용채널 매핑")
    zone_map = st.session_state["zone_map"].copy()
    channel_names = st.session_state["channels_df"]["채널명"].dropna().astype(str).tolist()
    cols = st.columns(5)
    for i, z in enumerate(PRICE_ZONES):
        with cols[i % 5]:
            zone_map[z] = st.selectbox(f"{z}", options=channel_names, index=channel_names.index(zone_map.get(z, channel_names[0])) if zone_map.get(z) in channel_names else 0, key=f"zmap_{z}")
    st.session_state["zone_map"] = zone_map

    st.divider()
    st.subheader("D. 밴드 경계(마우스로 조정)")
    b = st.session_state["boundaries"].copy()
    prev = 0
    new_b = [0]
    for idx in range(1, 10):
        minv = prev + 1
        maxv = 100 - (10-idx)
        val = int(b[idx])
        val = max(minv, min(maxv, val))
        val = st.slider(f"경계 {idx}: {PRICE_ZONES[idx-1]} | {PRICE_ZONES[idx]} (%)", min_value=minv, max_value=maxv, value=val, step=1, key=f"b_{idx}")
        new_b.append(val); prev = val
    new_b.append(100)
    st.session_state["boundaries"] = new_b

    with st.expander("각 영역 내 Target 위치(%) (기본=중앙) — SKU 전용", expanded=False):
        tp = st.session_state["target_pos"].copy()
        cols = st.columns(5)
        for i, z in enumerate(PRICE_ZONES):
            s = new_b[i]; e = new_b[i+1]
            mid = int(round((s+e)/2))
            with cols[i%5]:
                tp[z] = st.slider(f"{z}", min_value=int(s), max_value=int(e), value=int(tp.get(z, mid)), step=1, key=f"tp_{z}")
        st.session_state["target_pos"] = tp

with tab_cal:
    st.subheader("기존 운영 가격표 업로드 → 세트 Disc(할인율) 자동 역산")
    st.caption("세트는 '세트(No 있는 행) 직전 누적된 SKU행이 BOM' 규칙으로 파싱됩니다.")
    up_hist = st.file_uploader("운영 가격표 업로드(.xlsx/.csv)", type=["xlsx","xls","csv"], key="hist_up")

    if up_hist is not None:
        try:
            raw = load_history_table(up_hist)
            _, set_hist, bom_hist = parse_history_to_tables(raw)
            st.session_state["history_set_df"] = set_hist
            st.session_state["history_bom_df"] = bom_hist
            st.success(f"파싱 완료: 세트 {len(set_hist):,} / BOM라인 {len(bom_hist):,}")
        except Exception as e:
            st.error(f"파싱 오류: {e}")

    set_hist = st.session_state["history_set_df"]
    bom_hist = st.session_state["history_bom_df"]

    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        rounding_unit = st.selectbox("반올림 단위(캘)", [10,100,1000], index=2, key="cal_round")
    with c2:
        min_cost_ratio_cap = st.number_input("최저가 원가율 상한", min_value=0.05, max_value=0.95, value=0.30, step=0.01, format="%.2f", key="cal_min_ratio")
        always_cost_ratio_target = st.number_input("상시 목표 원가율", min_value=0.05, max_value=0.95, value=0.18, step=0.01, format="%.2f", key="cal_always_ratio")
        always_list_disc = st.slider("상시할인율(MSRP 대비) %", 0, 80, 20, 1, key="cal_list_disc") / 100.0


    c4,c5 = st.columns([1,1])
    with c4:
        msrp_min_gap_pct = st.slider("MSRP 최소 여유(=Min 대비 +%)", 0, 80, 15, 1, key="cal_gap") / 100.0
    with c5:
        min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="cal_cm") / 100.0


    tol = st.slider("일치 허용오차(±%)", 1, 20, 5, 1, key="cal_tol") / 100.0

    st.markdown("**현재 Disc 테이블(세트타입×가격영역)**")
    st.session_state["set_disc_df"] = st.data_editor(st.session_state["set_disc_df"], use_container_width=True, height=260, num_rows="dynamic", key="disc_editor")

    if st.session_state["products_df"].empty:
        st.warning("먼저 원가/상품마스터를 업로드해야 캘리브레이션이 가능합니다.")
    else:
        if st.button("✅ 캘리브레이션 실행(Disc 자동 채움)", type="primary"):
            sku_always_pred = compute_predicted_sku_always(
                products_df=st.session_state["products_df"],
                channels_df=st.session_state["channels_df"],
                zone_map=st.session_state["zone_map"],
                boundaries=st.session_state["boundaries"],
                rounding_unit=rounding_unit,
                min_cm=min_cm,
                min_cost_ratio_cap=min_cost_ratio_cap,
                always_cost_ratio_target=always_cost_ratio_target,
                always_list_disc=always_list_disc,
                msrp_min_gap_pct=msrp_min_gap_pct,

                overrides_df=st.session_state["overrides_df"],
            )
            new_disc, obs = calibrate_set_disc_from_history(
                set_df=set_hist,
                bom_df_hist=bom_hist,
                products_df=st.session_state["products_df"],
                sku_always_pred=sku_always_pred,
                params=st.session_state["set_params"],
                disc_df=st.session_state["set_disc_df"]
            )
            st.session_state["set_disc_df"] = new_disc
            st.session_state["cal_obs_df"] = obs
            st.success("Disc 테이블 업데이트 완료")

        with st.expander("역산 로그(Disc_obs / base_disc)", expanded=False):
            obs = st.session_state.get("cal_obs_df", pd.DataFrame())
            st.dataframe(obs.sort_values(["set_type","zone"]).head(300) if not obs.empty else obs, use_container_width=True, height=260)

        st.divider()
        st.subheader("검증: 예측 vs 실제(세트) 일치율")
        if set_hist.empty or bom_hist.empty:
            st.info("세트/구성 데이터가 없습니다. 운영 가격표 업로드 후 실행하세요.")
        else:
            if st.button("📊 검증 실행", type="secondary", key="run_validate_sets"):
                # 1) BASE 정합을 위해, 현재 파라미터 기준 SKU 상시(예측) 생성
                sku_always_pred = compute_predicted_sku_always(
                    products_df=st.session_state["products_df"],
                    channels_df=st.session_state["channels_df"],
                    zone_map=st.session_state["zone_map"],
                    boundaries=st.session_state["boundaries"],
                    rounding_unit=rounding_unit,
                    min_cm=min_cm,
                    min_cost_ratio_cap=min_cost_ratio_cap,
                always_cost_ratio_target=always_cost_ratio_target,
                always_list_disc=always_list_disc,
                msrp_min_gap_pct=msrp_min_gap_pct,

                    overrides_df=st.session_state["overrides_df"],
                )

                bom_app = bom_hist.rename(columns={"set_id":"세트ID"})[["세트ID","품번","수량"]].copy()

                rows = []
                actual_cols = ["공구가","홈쇼핑","폐쇄몰","모바일방송가","원데이특가","브랜드위크가","오프라인","상시할인가"]
                for _, sr in set_hist.iterrows():
                    sid = sr["set_id"]
                    anchors = compute_set_anchors(
                        set_id=sid,
                        bom_df=bom_app,
                        products_df=st.session_state["products_df"],
                        sku_always=sku_always_pred,
                        params=st.session_state["set_params"],
                        channels_df=st.session_state["channels_df"],
                        zone_map=st.session_state["zone_map"],
                        boundaries=st.session_state["boundaries"],
                        rounding_unit=rounding_unit,
                        min_cm=min_cm,
                        min_cost_ratio_cap=min_cost_ratio_cap,
                always_cost_ratio_target=always_cost_ratio_target,
                always_list_disc=always_list_disc,

                    )
                    if anchors is None:
                        continue
                    cost_total = compute_set_cost(sid, bom_app, st.session_state["products_df"], anchors["pack_cost"])
                    if cost_total != cost_total:
                        continue
                    min_auto, max_auto, _ = compute_set_range(
                        cost_total, anchors,
                        st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                        rounding_unit, min_cm, min_cost_ratio_cap, always_cost_ratio_target, always_list_disc, msrp_min_gap_pct
                    )
                    zdf = build_zone_table_set(
                        cost_total, min_auto, max_auto, anchors,
                        st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                        rounding_unit, min_cm,
                        st.session_state["overrides_df"], st.session_state["set_disc_df"], st.session_state["set_params"],
                        sid
                    )
                    if zdf.empty:
                        continue

                    for col in actual_cols:
                        actual = safe_float(sr.get(col, np.nan), np.nan)
                        if actual != actual or actual <= 0:
                            continue
                        zone = zone_from_history_column(col)
                        pr = zdf[zdf["가격영역"]==zone]
                        if pr.empty:
                            continue
                        pred = float(pr.iloc[0]["최종가격(원)"])
                        err_pct = abs(pred - actual) / max(1.0, actual)
                        rows.append({
                            "set_id": sid,
                            "set_name": sr.get("set_name",""),
                            "set_type": anchors.get("set_type",""),
                            "zone": zone,
                            "actual": actual,
                            "pred": pred,
                            "err_pct": err_pct,
                            "match": err_pct <= tol,
                        })

                cmp_df = pd.DataFrame(rows)
                if cmp_df.empty:
                    st.error("비교 가능한 데이터가 없습니다. (세트 가격 컬럼이 비어있거나 매핑 문제일 수 있음)")
                else:
                    overall = float(cmp_df["match"].mean()) * 100.0
                    st.metric("전체 일치율", f"{overall:.1f}% (N={len(cmp_df):,})")

                    by_zone = cmp_df.groupby("zone", as_index=False).agg(N=("match","size"), Acc=("match","mean"), MAPE=("err_pct","mean"))
                    by_zone["Acc(%)"] = (by_zone["Acc"]*100).round(1)
                    by_zone["MAPE(%)"] = (by_zone["MAPE"]*100).round(1)
                    st.markdown("**가격영역별**")
                    st.dataframe(by_zone.sort_values("N", ascending=False)[["zone","N","Acc(%)","MAPE(%)"]], use_container_width=True, height=260)

                    by_type = cmp_df.groupby("set_type", as_index=False).agg(N=("match","size"), Acc=("match","mean"), MAPE=("err_pct","mean"))
                    by_type["Acc(%)"] = (by_type["Acc"]*100).round(1)
                    by_type["MAPE(%)"] = (by_type["MAPE"]*100).round(1)
                    st.markdown("**세트타입별**")
                    st.dataframe(by_type.sort_values("N", ascending=False)[["set_type","N","Acc(%)","MAPE(%)"]], use_container_width=True, height=180)

                    st.markdown("**오차 큰 TOP 30**")
                    st.dataframe(cmp_df.sort_values("err_pct", ascending=False).head(30), use_container_width=True, height=320)

                    xb, xb_ext, xb_mime = to_excel_bytes({"pred_vs_actual": cmp_df, "by_zone": by_zone, "by_type": by_type})
        st.download_button("검증 결과 다운로드", xb, file_name=f"validation_sets.{xb_ext}",
                          mime=xb_mime)

with tab_sku:
    st.subheader("단품: 원가 기반 자동 산출 → Min/Max/채널가격 수정")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("업로드/설정 탭에서 원가 파일을 업로드하세요.")
    else:
        p1,p2,p3,p4 = st.columns([1,1,1,1])
        with p1:
            rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="sku_round")
        with p2:
            min_cost_ratio_cap = st.number_input("최저가 원가율 상한(예:0.30)", min_value=0.05, max_value=0.95, value=0.30, step=0.01, format="%.2f", key="sku_min_ratio")
            always_cost_ratio_target = st.number_input("상시 목표 원가율(예:0.18)", min_value=0.05, max_value=0.95, value=0.18, step=0.01, format="%.2f", key="sku_always_ratio")
            always_list_disc = st.slider("상시할인율(MSRP 대비) %", 0, 80, 20, 1, key="sku_list_disc") / 100.0
        with p3:
            msrp_min_gap_pct = st.slider("MSRP 최소 여유(=Min 대비 +%)", 0, 80, 15, 1, key="sku_gap") / 100.0
        with p4:
            min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="sku_cm") / 100.0

        options = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
        picked = st.selectbox("SKU 선택", options, index=0, key="sku_pick")
        sku = picked.split(" | ",1)[0].strip()
        row = prod[prod["품번"].astype(str)==sku].iloc[0]
        cost = safe_float(row["원가"], np.nan)

        if cost != cost or cost <= 0:
            st.error("원가가 비어있거나 0 이하입니다.")
        else:
            msrp_mkt, always_mkt, disc_used, mnote = compute_market_anchors_from_row(row, default_always_disc=always_list_disc)
            min_auto, max_auto, meta = compute_auto_range_from_cost(
                cost_total=cost,
                channels_df=st.session_state["channels_df"],
                zone_map=st.session_state["zone_map"],
                boundaries=st.session_state["boundaries"],
                rounding_unit=rounding_unit,
                min_cm=min_cm,
                min_cost_ratio_cap=min_cost_ratio_cap,
                always_cost_ratio_target=always_cost_ratio_target,
                always_list_disc=always_list_disc,
                market_msrp=msrp_mkt,
                market_always=always_mkt,
                market_always_disc=disc_used,
                msrp_min_gap_pct=msrp_min_gap_pct,

                include_zones=PRICE_ZONES,
                min_zone="공구",
                msrp_override=safe_float(row.get("MSRP_오버라이드", np.nan), np.nan),
            )

            st.markdown(f"**SKU:** `{sku}` — {row.get('상품명','')}")
            if meta.get("note"):
                st.info(meta["note"])

            c1,c2 = st.columns(2)
            with c1:
                min_user = st.number_input("Min(최저가) 수정", min_value=0, value=int(min_auto), step=rounding_unit)
            with c2:
                max_user = st.number_input("Max(최고가/MSRP) 수정", min_value=0, value=int(max_auto), step=rounding_unit)

            if max_user <= min_user:
                st.warning("Max가 Min 이하입니다. Max를 올려주세요.")
                max_user = min_user + max(rounding_unit*10, int(min_user*0.15))

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

with tab_set:
    st.subheader("세트(BOM): 구성하면 자동 추천가 + (Disc 캘리브레이션 반영)")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("원가 파일을 먼저 업로드하세요.")
    else:
        c1,c2,c3 = st.columns([1,2,1])
        with c1:
            new_id = st.text_input("세트ID", value="", key="new_set_id")
        with c2:
            new_name = st.text_input("세트명", value="", key="new_set_name")
        with c3:
            if st.button("세트 추가", type="primary", disabled=(not new_id.strip() or not new_name.strip()), key="add_set"):
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
            st.session_state["sets_df"] = st.data_editor(st.session_state["sets_df"], use_container_width=True, height=160, num_rows="dynamic", key="sets_editor")
            set_opts = (st.session_state["sets_df"]["세트ID"].astype(str) + " | " + st.session_state["sets_df"]["세트명"].astype(str)).tolist()
            picked = st.selectbox("편집할 세트 선택", set_opts, index=0, key="set_pick")
            set_id = picked.split(" | ",1)[0].strip()

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
                bom_view["is_acc(부자재)"] = bom_view.apply(lambda r: is_accessory_sku(r["품번"], r.get("상품명",""), r.get("원가",0)), axis=1)
                st.dataframe(bom_view, use_container_width=True, height=220)

            st.divider()
            st.markdown("### 세트 추천가")
            p1,p2,p3,p4 = st.columns([1,1,1,1])
            with p1:
                rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="set_round")
            with p2:
                min_cost_ratio_cap = st.number_input("최저가 원가율 상한", min_value=0.05, max_value=0.95, value=0.30, step=0.01, format="%.2f", key="set_min_ratio")
                always_cost_ratio_target = st.number_input("상시 목표 원가율", min_value=0.05, max_value=0.95, value=0.18, step=0.01, format="%.2f", key="set_always_ratio")
                always_list_disc = st.slider("상시할인율(MSRP 대비) %", 0, 80, 20, 1, key="set_list_disc") / 100.0
            with p3:
                msrp_min_gap_pct = st.slider("MSRP 최소 여유(=Min 대비 +%)", 0, 80, 15, 1, key="set_gap") / 100.0
            with p4:
                min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="set_cm") / 100.0

            if not bom_view.empty:
                sku_always = compute_predicted_sku_always(
                    st.session_state["products_df"],
                    st.session_state["channels_df"],
                    st.session_state["zone_map"],
                    st.session_state["boundaries"],
                    rounding_unit, min_cm, min_cost_ratio_cap, always_cost_ratio_target, always_list_disc, msrp_min_gap_pct,
                    st.session_state["overrides_df"]
                )
                anchors = compute_set_anchors(
                    set_id, st.session_state["bom_df"], prod, sku_always, st.session_state["set_params"],
                    st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                    rounding_unit, min_cm, min_cost_ratio_cap, always_cost_ratio_target, always_list_disc, msrp_min_gap_pct
                )
                if anchors is None:
                    st.error("세트 앵커 계산 실패")
                else:
                    cost_total = compute_set_cost(set_id, st.session_state["bom_df"], prod, anchors["pack_cost"])
                    min_auto, max_auto, meta = compute_set_range(cost_total, anchors, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"], rounding_unit, min_cm, min_cost_ratio_cap, always_cost_ratio_target, always_list_disc, msrp_min_gap_pct)
                    st.write(f"- 세트 원가합(+pack_cost): **{int(cost_total):,}원** | 자동 레인지: **{int(min_auto):,} ~ {int(max_auto):,}원**")
                    if meta.get("note"): st.info(meta["note"])

                    c1,c2 = st.columns(2)
                    with c1:
                        min_user = st.number_input("Min 수정(세트)", min_value=0, value=int(min_auto), step=rounding_unit, key="set_min_user")
                    with c2:
                        max_user = st.number_input("Max 수정(세트)", min_value=0, value=int(max_auto), step=rounding_unit, key="set_max_user")

                    if max_user <= min_user:
                        max_user = min_user + max(rounding_unit*10, int(min_user*0.15))

                    zdf = build_zone_table_set(
                        cost_total, float(min_user), float(max_user), anchors,
                        st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                        rounding_unit, min_cm,
                        st.session_state["overrides_df"], st.session_state["set_disc_df"], st.session_state["set_params"],
                        set_id
                    )
                    st.dataframe(zdf, use_container_width=True, height=360)

with tab_logic:
    st.subheader("v6.3 로직 요약")
    st.markdown(
        """
- **원가만 업로드**하면 SKU/세트가 자동으로 가격을 생성합니다.
- **세트 추천가**는 `BASE(구성품 상시예측 합) × (1 - Disc)`를 기본으로 하며, Floor/Band로 클램프합니다.
- **운영 가격표 업로드** 후 캘리브레이션을 실행하면,
  - 세트 헤더 행(No 있는 행) 직전 누적된 SKU행을 BOM으로 확정하고,
  - `Disc_obs = 1 - (세트실제가 / BASE_pred)`로 Disc를 역산해 **Disc 테이블을 자동 채움**합니다.
        """
    )