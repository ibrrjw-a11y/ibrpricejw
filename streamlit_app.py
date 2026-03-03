import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v6.3
# (Cost-only → Auto prices → You adjust) + Calibration/Validation
#
# v6.3 adds:
# - Upload historical "운영 가격표" (set header + preceding components)
# - Parse into Set_Prices / Set_BOM
# - Calibrate set discount table (Disc) by reverse-engineering from history
# - Validate predicted vs actual (accuracy within tolerance)
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator v6.3", layout="wide")

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

def to_excel_bytes(df_dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sh, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sh[:31])
    return bio.getvalue()

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
    max_cost_ratio: float,
    include_zones: list,
    min_zone: str = "공구",
    msrp_override=np.nan,
):
    ch_map = channels_df.set_index("채널명").to_dict("index")

    def zone_floor(z):
        ch = zone_map.get(z, "자사몰")
        p = ch_map.get(ch, None)
        if p is None:
            return np.nan
        return floor_price(
            cost_total=cost_total, q_orders=1,
            fee=p["수수료율"], pg=p["PG"], mkt=p["마케팅비"],
            ship_per_order=p["배송비(주문당)"],
            ret_rate=p["반품률"], ret_cost_order=p["반품비(주문당)"],
            min_cm=min_cm
        )

    min_auto = zone_floor(min_zone)
    if min_auto != min_auto or min_auto <= 0:
        return np.nan, np.nan, {"note": "Min(Floor) 산출 불가"}

    min_auto = krw_round(min_auto, rounding_unit)
    msrp_base = krw_ceil(cost_total / max_cost_ratio, rounding_unit) if (cost_total == cost_total and cost_total > 0) else np.nan

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
# SKU zone table
# -----------------------------
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
            status = "불가(Floor>BandHigh)"; target = band_high
        elif target > band_high:
            status = "클립(Target→BandHigh)"; target = band_high

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
        if eff_r == eff_r and eff_r < floor_r: flags.append("⚠️Floor 미만")
        if eff_r == eff_r and eff_r < band_low_r: flags.append("⚠️BandLow 미만")
        if eff_r == eff_r and eff_r > band_high_r and z != "MSRP": flags.append("⚠️BandHigh 초과")

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

def estimate_sku_msrp_from_cost(sku_cost: float, channels_df, zone_map, boundaries, rounding_unit, min_cm, max_cost_ratio):
    if sku_cost != sku_cost or sku_cost <= 0: return np.nan
    _, max_auto, _ = compute_auto_range_from_cost(
        cost_total=sku_cost, channels_df=channels_df, zone_map=zone_map, boundaries=boundaries,
        rounding_unit=rounding_unit, min_cm=min_cm, max_cost_ratio=max_cost_ratio,
        include_zones=PRICE_ZONES, min_zone="공구", msrp_override=np.nan
    )
    return float(max_auto) if max_auto == max_auto else np.nan

def compute_set_cost(set_id: str, bom_df: pd.DataFrame, products_df: pd.DataFrame, pack_cost: float):
    b = bom_df[bom_df["세트ID"] == set_id].copy()
    if b.empty: return np.nan
    b["수량"] = pd.to_numeric(b["수량"], errors="coerce").fillna(0).astype(int)
    b = b.merge(products_df[["품번","원가"]], on="품번", how="left")
    b["원가"] = pd.to_numeric(b["원가"], errors="coerce").fillna(0.0)
    return float((b["원가"] * b["수량"]).sum() + float(pack_cost or 0.0))

def get_set_disc_pct(set_type: str, zone: str, pack_units: int, disc_df: pd.DataFrame, params: dict):
    base = 0.0
    rr = disc_df[(disc_df["세트타입"]==set_type) & (disc_df["가격영역"]==zone)]
    if not rr.empty: base = float(rr.iloc[0]["할인율(%)"])
    step = float(params.get("disc_pack_step_pct", 2.0))
    cap = float(params.get("disc_pack_cap_pct", 6.0))
    add = 0.0 if (pack_units is None or pack_units <= 1) else min(cap, step * np.log2(pack_units))
    return max(0.0, min(95.0, base + add))

def compute_predicted_sku_always(products_df, channels_df, zone_map, boundaries, rounding_unit, min_cm, max_cost_ratio, overrides_df):
    sku_always = {}
    tp_mid = default_zone_target_pos(boundaries)
    for _, r in products_df.iterrows():
        sku = str(r["품번"]).strip()
        cost = safe_float(r.get("원가", np.nan), np.nan)
        if cost != cost or cost <= 0: 
            continue
        min_s, max_s, _ = compute_auto_range_from_cost(
            cost_total=cost, channels_df=channels_df, zone_map=zone_map, boundaries=boundaries,
            rounding_unit=rounding_unit, min_cm=min_cm, max_cost_ratio=max_cost_ratio,
            include_zones=PRICE_ZONES, min_zone="공구", msrp_override=safe_float(r.get("MSRP_오버라이드", np.nan), np.nan)
        )
        zdf = build_zone_table(cost, min_s, max_s, channels_df, zone_map, boundaries, tp_mid, rounding_unit, min_cm, overrides_df, "SKU", sku)
        if zdf is None or zdf.empty or ("가격영역" not in zdf.columns):
            continue
        ar = zdf[zdf["가격영역"]=="상시"]
        if not ar.empty:
            sku_always[sku] = float(ar.iloc[0]["최종가격(원)"])
    return sku_always

def compute_set_anchors(set_id: str, bom_df: pd.DataFrame, products_df: pd.DataFrame,
                        sku_always: dict, params: dict,
                        channels_df, zone_map, boundaries, rounding_unit, min_cm, max_cost_ratio):
    cls = classify_set(set_id, bom_df, products_df)
    b = cls.get("detail_df", pd.DataFrame()).copy()
    if b.empty: return None

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
        sku_msrp = estimate_sku_msrp_from_cost(
            safe_float(rr.get("원가", np.nan), np.nan),
            channels_df, zone_map, boundaries, rounding_unit, min_cm, max_cost_ratio
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

def compute_set_range(cost_total: float, anchors: dict, channels_df, zone_map, boundaries, rounding_unit, min_cm, max_cost_ratio, msrp_override=np.nan):
    min_auto, max_cost, meta = compute_auto_range_from_cost(
        cost_total=cost_total, channels_df=channels_df, zone_map=zone_map, boundaries=boundaries,
        rounding_unit=rounding_unit, min_cm=min_cm, max_cost_ratio=max_cost_ratio,
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
                status = "불가(Floor>BandHigh)"; target = band_high
            else:
                if target > band_high:
                    status = "클립(Target→BandHigh)"; target = band_high
                if target < band_low:
                    status = "클립(Target→BandLow)"; target = band_low

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
        if eff_r == eff_r and eff_r < floor_r: flags.append("⚠️Floor 미만")
        if eff_r == eff_r and eff_r < band_low_r: flags.append("⚠️BandLow 미만")
        if eff_r == eff_r and eff_r > band_high_r and z != "MSRP": flags.append("⚠️BandHigh 초과")

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
st.title("IBR 가격 시뮬레이터 v6.3")
st.caption("원가만 업로드 → 자동 가격 생성 → (추가) 기존 운영 데이터로 세트 할인율 역산/검증")

tab_up, tab_cal, tab_sku, tab_set, tab_logic = st.tabs(
    ["1) 업로드/설정", "2) 캘리브레이션(세트 할인율)", "3) 단품(자동→수정)", "4) 세트(BOM)", "5) 로직(문장)"]
)

with tab_up:
    st.subheader("A. 원가/상품명 통일 파일 업로드(필수)")
    up = st.file_uploader("원가/상품마스터 업로드(.xlsx)", type=["xlsx","xls"])
    if up is not None:
        try:
            st.session_state["products_df"] = load_products_from_cost_master(up)
            st.success(f"업로드 완료: {len(st.session_state['products_df']):,}개 SKU")
        except Exception as e:
            st.error(f"업로드 오류: {e}")

    st.metric("현재 SKU 수", f"{len(st.session_state['products_df']):,}")

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
        max_cost_ratio = st.number_input("MSRP 원가율 상한(캘)", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="cal_ratio")
    with c3:
        min_cm = st.slider("최소 기여이익률(캘) %", 0, 50, 15, 1, key="cal_cm") / 100.0

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
                max_cost_ratio=max_cost_ratio,
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
                    max_cost_ratio=max_cost_ratio,
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
                        max_cost_ratio=max_cost_ratio,
                    )
                    if anchors is None:
                        continue
                    cost_total = compute_set_cost(sid, bom_app, st.session_state["products_df"], anchors["pack_cost"])
                    if cost_total != cost_total:
                        continue
                    min_auto, max_auto, _ = compute_set_range(
                        cost_total, anchors,
                        st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                        rounding_unit, min_cm, max_cost_ratio
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

                    xb = to_excel_bytes({"pred_vs_actual": cmp_df, "by_zone": by_zone, "by_type": by_type})
                    st.download_button("검증 결과 다운로드", xb, file_name="validation_sets.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_sku:
    st.subheader("단품: 원가 기반 자동 산출 → Min/Max/채널가격 수정")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("업로드/설정 탭에서 원가 파일을 업로드하세요.")
    else:
        p1,p2,p3 = st.columns([1,1,1])
        with p1:
            rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="sku_round")
        with p2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한(예:0.30)", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="sku_ratio")
        with p3:
            min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="sku_cm") / 100.0

        options = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
        picked = st.selectbox("SKU 선택", options, index=0, key="sku_pick")
        sku = picked.split(" | ",1)[0].strip()
        row = prod[prod["품번"].astype(str)==sku].iloc[0]
        cost = safe_float(row["원가"], np.nan)

        if cost != cost or cost <= 0:
            st.error("원가가 비어있거나 0 이하입니다.")
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
            p1,p2,p3 = st.columns([1,1,1])
            with p1:
                rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=2, key="set_round")
            with p2:
                max_cost_ratio = st.number_input("MSRP 원가율 상한", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="set_ratio")
            with p3:
                min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1, key="set_cm") / 100.0

            if not bom_view.empty:
                sku_always = compute_predicted_sku_always(
                    st.session_state["products_df"],
                    st.session_state["channels_df"],
                    st.session_state["zone_map"],
                    st.session_state["boundaries"],
                    rounding_unit, min_cm, max_cost_ratio,
                    st.session_state["overrides_df"]
                )
                anchors = compute_set_anchors(
                    set_id, st.session_state["bom_df"], prod, sku_always, st.session_state["set_params"],
                    st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"],
                    rounding_unit, min_cm, max_cost_ratio
                )
                if anchors is None:
                    st.error("세트 앵커 계산 실패")
                else:
                    cost_total = compute_set_cost(set_id, st.session_state["bom_df"], prod, anchors["pack_cost"])
                    min_auto, max_auto, meta = compute_set_range(cost_total, anchors, st.session_state["channels_df"], st.session_state["zone_map"], st.session_state["boundaries"], rounding_unit, min_cm, max_cost_ratio)
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