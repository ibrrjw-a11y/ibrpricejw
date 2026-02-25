import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# IBR Pricing Simulator v5 (Band-first / Set-first / Intuitive)
# 핵심 컨셉:
# 1) SKU/SET마다 "최저가(Min)~최고가(Max)" 레인지 먼저 설정(자동+수동 오버라이드)
# 2) 레인지 안을 "가격영역 밴드(구간)"로 분할 -> 각 영역의 Target은 밴드 내부에서 선택
# 3) 채널 비용(수수료/PG/배송/마케팅/반품)과 최소기여이익률로 "손익하한(Floor)" 계산
# 4) Target은 Floor보다 낮을 수 없음(자동 클립/불가 표시)
# 5) 세트는 BOM(구성품 리스트업)로 원가/레인지/밴드/채널별 가격 자동추천 + 구성품별 배분단가/할인율 + 카니발 체크
# ============================================================

st.set_page_config(page_title="IBR Pricing Simulator v5", layout="wide")

# -----------------------------
# Defaults (based on our discussion)
# -----------------------------
PRICE_ZONES = ["공구", "홈쇼핑", "폐쇄몰", "모바일라방", "원데이", "브랜드위크", "홈사", "상시", "오프라인", "MSRP"]

DEFAULT_CHANNELS = [
    # 채널명, 수수료, PG, 배송(주문당), 마케팅, 반품률, 반품비(주문당)
    ("오프라인",   0.50, 0.00, 0.0,    0.00, 0.00, 0.0),
    ("자사몰",     0.05, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("스마트스토어",0.05, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("쿠팡",       0.25, 0.00, 3000.0, 0.00, 0.00, 0.0),
    ("오픈마켓",   0.15, 0.00, 3000.0, 0.00, 0.00, 0.0),
    ("홈사",       0.30, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("공구",       0.50, 0.03, 3000.0, 0.00, 0.00, 0.0),
    ("홈쇼핑",     0.55, 0.00, 0.0,    0.00, 0.00, 0.0),  # 홈쇼핑이 송출/제작/택배 부담 가정
    ("모바일라이브",0.40,0.03, 3000.0, 0.00, 0.00, 0.0),
    ("폐쇄몰",     0.25, 0.00, 3000.0, 0.00, 0.00, 0.0),
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

# 밴드(구간) 기본 경계값 (0~100)
# 공구~홈쇼핑~...~MSRP 순서로 연속 구간
DEFAULT_BOUNDARIES = [0, 10, 20, 30, 42, 52, 62, 72, 84, 94, 100]
# 각 영역 내 Target 위치(%) 기본값: 구간 중앙
def default_zone_target_pos(boundaries):
    out = {}
    for i, z in enumerate(PRICE_ZONES):
        out[z] = (boundaries[i] + boundaries[i+1]) / 2
    return out

# -----------------------------
# Helpers
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

def find_cost_sheet(xls: pd.ExcelFile):
    """원가 파일(예: 2026_원가_상품마스터.xlsx)에서 '상품코드'+'원가' 컬럼이 있는 시트 우선 탐색"""
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
    if candidates:
        return candidates[0]
    return xls.sheet_names[0]

def load_products_from_cost_master(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    sh = find_cost_sheet(xls)
    df = pd.read_excel(xls, sheet_name=sh, header=2)
    df.columns = [str(c).strip() for c in df.columns]

    # resolve columns
    code_col = "상품코드" if "상품코드" in df.columns else None
    name_col = "상품명" if "상품명" in df.columns else None

    cost_col = None
    for c in ["원가 (vat-)", "원가", "매입원가", "랜디드코스트", "랜디드코스트(총원가)"]:
        if c in df.columns:
            cost_col = c
            break

    out = pd.DataFrame({
        "품번": df[code_col].astype(str).str.strip() if code_col else "",
        "상품명": df[name_col].astype(str).str.strip() if name_col else "",
        "브랜드": df["브랜드"].astype(str).str.strip() if "브랜드" in df.columns else "",
        "원가": pd.to_numeric(df[cost_col], errors="coerce") if cost_col else np.nan,
    })

    out = out[out["품번"].ne("")].drop_duplicates(subset=["품번"]).reset_index(drop=True)
    # optional MSRP columns in source
    for c in ["MSRP", "소비자가", "정가", "RRP"]:
        if c in df.columns:
            tmp = pd.to_numeric(df[c], errors="coerce")
            out["MSRP_입력"] = tmp
            break
    if "MSRP_입력" not in out.columns:
        out["MSRP_입력"] = np.nan

    # override fields for app
    out["MSRP_오버라이드"] = np.nan
    out["Min_오버라이드"] = np.nan
    out["Max_오버라이드"] = np.nan
    out["운영여부"] = True
    return out

def compute_msrp(cost, max_cost_ratio=0.30, rounding_unit=1000, msrp_input=np.nan, msrp_override=np.nan):
    """MSRP 자동/입력/오버라이드 결합: 최종 MSRP"""
    c = safe_float(cost, np.nan)
    if np.isnan(c) or c <= 0:
        base = np.nan
    else:
        base = c / max_cost_ratio
    candidates = []
    for v in [msrp_input, base, msrp_override]:
        vv = safe_float(v, np.nan)
        if not np.isnan(vv) and vv > 0:
            candidates.append(vv)
    if not candidates:
        return np.nan
    return krw_ceil(max(candidates), rounding_unit)

def floor_price(cost_total, q, fee, pg, mkt, ship_per_order, ret_rate, ret_cost_order, min_cm):
    """세트/단품 공통 손익하한(판매가 기준)"""
    denom = 1.0 - (fee + pg + mkt + min_cm)
    if denom <= 0:
        return float("inf")
    ship_unit = ship_per_order / max(1, q)
    ret_unit = (ret_rate * ret_cost_order) / max(1, q)
    return (cost_total + ship_unit + ret_unit) / denom

def contrib_metrics(price, cost_total, q, fee, pg, mkt, ship_per_order, ret_rate, ret_cost_order):
    """판매가 기준 기여이익(원), 기여이익률"""
    if price <= 0:
        return np.nan, np.nan
    ship_unit = ship_per_order / max(1, q)
    ret_unit = (ret_rate * ret_cost_order) / max(1, q)
    net = price * (1.0 - fee - pg - mkt) - ship_unit - ret_unit - cost_total
    return net, net / price

def build_zone_table_for_item(item_type, item_id, products_df, sets_df, bom_df,
                             channels_df, zone_map, boundaries, target_pos, rounding_unit,
                             min_cm, max_cost_ratio, min_zone_for_range, max_zone_for_range,
                             min_override=np.nan, max_override=np.nan):
    """
    item_type: 'SKU' or 'SET'
    returns: zone_table (per-zone), meta dict(min,max,msrp,...)
    """
    # cost_total and name
    if item_type == "SKU":
        row = products_df.loc[products_df["품번"] == item_id].iloc[0]
        name = row["상품명"]
        cost = safe_float(row["원가"], np.nan)
        msrp = compute_msrp(cost, max_cost_ratio=max_cost_ratio, rounding_unit=rounding_unit,
                            msrp_input=row.get("MSRP_입력", np.nan),
                            msrp_override=row.get("MSRP_오버라이드", np.nan))
        cost_total = cost
        q_item = 1
    else:
        srow = sets_df.loc[sets_df["세트ID"] == item_id].iloc[0]
        name = srow["세트명"]
        # compute set cost from BOM
        b = bom_df[bom_df["세트ID"] == item_id].merge(products_df[["품번","상품명","원가"]], on="품번", how="left")
        b["수량"] = pd.to_numeric(b["수량"], errors="coerce").fillna(0).astype(int)
        b["원가"] = pd.to_numeric(b["원가"], errors="coerce")
        cost_total = float((b["원가"].fillna(0) * b["수량"]).sum())
        q_item = 1  # set sold as 1 order unit; shipping/return per order, not per component
        # set MSRP default = sum(component MSRP) or override
        # We'll use sum(cost/max_cost_ratio) as safe proxy if component MSRP missing.
        msrp = safe_float(srow.get("MSRP_오버라이드", np.nan), np.nan)
        if np.isnan(msrp) or msrp <= 0:
            # build from components
            msrp_sum = 0.0
            for _, rr in b.iterrows():
                cc = safe_float(rr["원가"], np.nan)
                if np.isnan(cc) or cc <= 0:
                    continue
                msrp_sum += (cc / max_cost_ratio) * rr["수량"]
            msrp = krw_ceil(msrp_sum, rounding_unit) if msrp_sum>0 else np.nan

    # channel lookup
    ch_map = channels_df.set_index("채널명").to_dict("index")

    # determine range Min/Max
    # Min base: floor of selected min_zone_for_range
    min_zone = min_zone_for_range
    max_zone = max_zone_for_range

    def zone_floor(z):
        ch = zone_map.get(z, "자사몰")
        p = ch_map.get(ch, None)
        if p is None:
            return np.nan
        return floor_price(
            cost_total=cost_total,
            q=1,
            fee=p["수수료율"],
            pg=p["PG"],
            mkt=p["마케팅비"],
            ship_per_order=p["배송비(주문당)"],
            ret_rate=p["반품률"],
            ret_cost_order=p["반품비(주문당)"],
            min_cm=min_cm
        )

    floor_min_zone = zone_floor(min_zone)
    min_auto = krw_round(floor_min_zone, rounding_unit)

    # Max base: MSRP or floor of max_zone? We'll treat MSRP as max by default
    max_auto = krw_round(msrp, rounding_unit) if not np.isnan(msrp) else np.nan
    if np.isnan(max_auto) and max_zone in PRICE_ZONES:
        max_auto = krw_round(zone_floor(max_zone), rounding_unit)

    # apply overrides (priority: explicit function args > df override columns)
    min_final = safe_float(min_override, np.nan)
    max_final = safe_float(max_override, np.nan)
    if np.isnan(min_final) and item_type=="SKU":
        min_final = safe_float(row.get("Min_오버라이드", np.nan), np.nan)
    if np.isnan(max_final) and item_type=="SKU":
        max_final = safe_float(row.get("Max_오버라이드", np.nan), np.nan)
    if np.isnan(min_final):
        min_final = min_auto
    if np.isnan(max_final):
        max_final = max_auto

    # protect: if max <= min => no room
    note = ""
    if np.isnan(min_final) or np.isnan(max_final):
        note = "원가/정가 정보 부족으로 레인지 산출 불가"
    else:
        if max_final <= min_final:
            # expand max slightly to visualize bands
            bump = max(rounding_unit*10, min_final*0.1)
            max_final = krw_round(min_final + bump, rounding_unit)
            note = f"⚠️ 레인지가 붙어있어(Max<=Min) 자동으로 Max를 확장(+{int(bump):,})했습니다. (공구 단품/홈쇼핑 단품 운영이 구조적으로 어려울 수 있음)"

    # build zone table
    rows = []
    if (not np.isnan(min_final)) and (not np.isnan(max_final)) and (max_final > min_final):
        span = max_final - min_final
        for i, z in enumerate(PRICE_ZONES):
            start = boundaries[i] / 100.0
            end = boundaries[i+1] / 100.0
            band_low = min_final + span * start
            band_high = min_final + span * end

            pos = target_pos.get(z, (boundaries[i]+boundaries[i+1])/2) / 100.0
            target_raw = min_final + span * pos

            # economics
            ch = zone_map.get(z, "자사몰")
            p = ch_map.get(ch, None)
            if p is None:
                continue
            floor = floor_price(
                cost_total=cost_total,
                q=1,
                fee=p["수수료율"],
                pg=p["PG"],
                mkt=p["마케팅비"],
                ship_per_order=p["배송비(주문당)"],
                ret_rate=p["반품률"],
                ret_cost_order=p["반품비(주문당)"],
                min_cm=min_cm
            )

            # clamp and status
            status = "OK"
            target = max(target_raw, floor)
            if floor > band_high:
                status = "불가(Floor>BandHigh)"
                target = band_high
            elif target > band_high:
                status = "클립(Target→BandHigh)"
                target = band_high

            # rounding
            band_low = krw_round(band_low, rounding_unit)
            band_high = krw_round(band_high, rounding_unit)
            floor_r = krw_round(floor, rounding_unit)
            target_r = krw_round(target, rounding_unit)

            cm, cmr = contrib_metrics(
                price=target_r,
                cost_total=cost_total,
                q=1,
                fee=p["수수료율"],
                pg=p["PG"],
                mkt=p["마케팅비"],
                ship_per_order=p["배송비(주문당)"],
                ret_rate=p["반품률"],
                ret_cost_order=p["반품비(주문당)"]
            )

            rows.append({
                "가격영역": z,
                "비용채널": ch,
                "BandLow": band_low,
                "BandHigh": band_high,
                "Floor(손익하한)": floor_r,
                "추천가(Target)": target_r,
                "상태": status,
                "마진룸(원)=Target-Floor": target_r - floor_r,
                "기여이익(원)": krw_round(cm, 1) if not np.isnan(cm) else np.nan,
                "기여이익률(%)": round(cmr*100, 1) if not np.isnan(cmr) else np.nan,
            })
    zdf = pd.DataFrame(rows)
    meta = {
        "아이템": item_id,
        "이름": name,
        "원가합": cost_total,
        "Min": min_final,
        "Max": max_final,
        "MSRP": msrp if item_type=="SKU" else max_auto,
        "노트": note,
        "min_zone_for_range": min_zone,
        "max_zone_for_range": max_zone,
    }
    return zdf, meta

def allocate_set_price_to_components(set_id, zone, set_price, products_df, sets_df, bom_df, sku_always_prices):
    """
    set_price를 구성품에 상시가 가중치로 배분해서 구성품별 실질단가/상시대비할인율 계산
    sku_always_prices: dict {품번: 상시추천가(Target)}  (단품 기준)
    """
    b = bom_df[bom_df["세트ID"] == set_id].copy()
    if b.empty:
        return pd.DataFrame()

    b["수량"] = pd.to_numeric(b["수량"], errors="coerce").fillna(0).astype(int)
    b = b.merge(products_df[["품번","상품명"]], on="품번", how="left")
    b["상시가_ref"] = b["품번"].map(sku_always_prices).astype(float)
    b["상시가_ref"] = b["상시가_ref"].fillna(0.0)

    b["ref_value"] = b["상시가_ref"] * b["수량"]
    total_ref = float(b["ref_value"].sum())
    if total_ref <= 0:
        # fallback: equal weight
        b["w"] = 1.0 / len(b)
    else:
        b["w"] = b["ref_value"] / total_ref

    b["배분매출"] = set_price * b["w"]
    b["실질단가"] = b["배분매출"] / b["수량"].replace(0, np.nan)
    b["상시대비할인율(%)"] = np.where(
        b["상시가_ref"]>0,
        (1.0 - (b["실질단가"] / b["상시가_ref"])) * 100.0,
        np.nan
    )
    b["세트ID"] = set_id
    b["가격영역"] = zone
    return b[["세트ID","가격영역","품번","상품명","수량","상시가_ref","실질단가","상시대비할인율(%)"]]

# -----------------------------
# Session state init
# -----------------------------
if "products_df" not in st.session_state:
    st.session_state["products_df"] = pd.DataFrame(columns=["품번","상품명","브랜드","원가","MSRP_입력","MSRP_오버라이드","Min_오버라이드","Max_오버라이드","운영여부"])
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

# -----------------------------
# UI
# -----------------------------
st.title("IBR 가격 시뮬레이터 v5 (밴드 기반 + 세트 빌더 + 카니발 체크)")
st.caption("레인지(Min~Max) → 밴드(영역) → 채널 손익하한(Floor) → 추천가(Target) / 마진룸 / 세트 배분 / 카니발")

tab_up, tab_single, tab_set, tab_plan, tab_logic = st.tabs(
    ["1) 업로드/기본설정", "2) 단품 밴드(직관형)", "3) 세트 빌더", "4) 운영플랜 & 카니발", "5) 로직(문장)"]
)

# =============================
# 1) Upload & Settings
# =============================
with tab_up:
    st.subheader("A. 원가/상품명 통일 파일 업로드")
    st.caption("업로드 파일은 '상품명 통일 + 원가' 목적. (예: 2026_원가_상품마스터.xlsx)")
    up = st.file_uploader("원가/상품마스터 업로드(.xlsx)", type=["xlsx","xls"])

    c1, c2 = st.columns([1,1])
    with c1:
        if up is not None:
            try:
                new_df = load_products_from_cost_master(up)
                base = st.session_state["products_df"].copy()
                if base.empty:
                    merged = new_df
                else:
                    merged = base.merge(new_df[["품번","상품명","브랜드","원가","MSRP_입력"]], on="품번", how="outer", suffixes=("", "_new"))
                    # prefer new name/cost if present
                    for col in ["상품명","브랜드","원가","MSRP_입력"]:
                        merged[col] = merged[col+"_new"].where(merged[col+"_new"].notna(), merged[col])
                        merged = merged.drop(columns=[col+"_new"])
                    # keep overrides
                    for col in ["MSRP_오버라이드","Min_오버라이드","Max_오버라이드","운영여부"]:
                        if col not in merged.columns:
                            merged[col] = np.nan if "오버라이드" in col else True
                merged["운영여부"] = merged["운영여부"].fillna(True).astype(bool)
                st.session_state["products_df"] = merged.reset_index(drop=True)
                st.success(f"업로드 완료: {len(merged):,}개 SKU")
            except Exception as e:
                st.error(f"업로드 처리 오류: {e}")
        else:
            st.info("파일을 업로드하면 SKU(상품코드)와 원가(vat-)가 자동 로드됩니다.")

    with c2:
        st.metric("현재 SKU 수", f"{len(st.session_state['products_df']):,}")

    st.divider()
    st.subheader("B. 채널 비용(키인하면 바로 반영)")
    st.caption("수수료/PG/배송/마케팅/반품을 수정하면 Floor/추천가/마진룸이 즉시 바뀝니다.")
    ch_edited = st.data_editor(
        st.session_state["channels_df"],
        use_container_width=True,
        num_rows="dynamic",
        height=280,
        key="channels_editor",
    )
    st.session_state["channels_df"] = ch_edited.copy()

    st.divider()
    st.subheader("C. 가격영역(밴드) → 비용채널 매핑")
    st.caption("가격영역(공구/홈쇼핑/원데이 등)이 실제로 어느 비용채널(자사몰/쿠팡 등)에서 운영되는지 선택합니다.")
    zone_map = st.session_state["zone_map"].copy()
    channel_names = st.session_state["channels_df"]["채널명"].dropna().astype(str).tolist()
    zm_cols = st.columns(5)
    for i, z in enumerate(PRICE_ZONES):
        with zm_cols[i % 5]:
            zone_map[z] = st.selectbox(
                f"{z} 비용채널",
                options=channel_names,
                index=channel_names.index(zone_map.get(z, channel_names[0])) if zone_map.get(z, None) in channel_names else 0,
                key=f"zone_map_{z}"
            )
    st.session_state["zone_map"] = zone_map

    st.divider()
    st.subheader("D. 밴드 경계(마우스로 드래그해서 조정)")
    st.caption("Min~Max 레인지(0~100%)를 10개 가격영역으로 분할하는 경계값입니다. 경계 슬라이더를 드래그하면 밴드 폭이 바뀝니다.")
    b = st.session_state["boundaries"].copy()

    # boundary sliders b1..b9
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

    # target positions (optional)
    with st.expander("각 가격영역 내 Target 위치(%) 조정 (기본=구간 중앙)", expanded=False):
        tp = st.session_state["target_pos"].copy()
        cols = st.columns(5)
        for i, z in enumerate(PRICE_ZONES):
            s = new_b[i]
            e = new_b[i+1]
            mid = (s+e)/2
            with cols[i%5]:
                tp[z] = st.slider(f"{z} Target(%)", min_value=int(s), max_value=int(e), value=int(round(tp.get(z, mid))), step=1, key=f"tp_{z}")
        st.session_state["target_pos"] = tp

    st.divider()
    st.subheader("E. SKU 마스터(원가/MSRP/Min/Max 오버라이드)")
    st.caption("원가/정가를 확인하고, 필요 시 MSRP/Min/Max를 수동 오버라이드하세요. (원가에 배송비를 포함하지 않는 것을 권장)")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("먼저 원가/상품마스터 파일을 업로드하세요.")
    else:
        prod_edit = st.data_editor(prod, use_container_width=True, height=360, num_rows="dynamic", key="prod_editor")
        st.session_state["products_df"] = prod_edit.copy()

# =============================
# 2) Single SKU band view
# =============================
with tab_single:
    st.subheader("단품 가격 밴드/추천가/마진(직관형)")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("업로드 탭에서 원가/상품마스터를 먼저 업로드하세요.")
    else:
        # global policy
        p1, p2, p3, p4 = st.columns([1,1,1,1])
        with p1:
            rounding_unit = st.selectbox("반올림 단위", [10,100,1000], index=1)
        with p2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한(예:30%)", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f")
        with p3:
            min_cm = st.slider("최소 기여이익률(%)", 0, 50, 15, 1) / 100.0
        with p4:
            gap_pct = st.slider("카니발 최소갭(%)", 0, 20, 5, 1) / 100.0
            gap_won = st.number_input("카니발 최소갭(원)", min_value=0, value=2000, step=500)

        options = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
        picked = st.selectbox("상품 선택", options=options, index=0)
        sku = picked.split(" | ",1)[0].strip()

        # range anchor selection
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            min_zone_for_range = st.selectbox("레인지 Min 기준(기본)", PRICE_ZONES, index=0)
        with c2:
            max_zone_for_range = st.selectbox("레인지 Max 기준(기본)", ["MSRP"] + PRICE_ZONES, index=0)
        with c3:
            st.caption("Min/Max는 기본값이며, SKU별 Min_오버라이드/Max_오버라이드로 수동 고정 가능합니다.")

        zdf, meta = build_zone_table_for_item(
            item_type="SKU",
            item_id=sku,
            products_df=prod,
            sets_df=st.session_state["sets_df"],
            bom_df=st.session_state["bom_df"],
            channels_df=st.session_state["channels_df"],
            zone_map=st.session_state["zone_map"],
            boundaries=st.session_state["boundaries"],
            target_pos=st.session_state["target_pos"],
            rounding_unit=rounding_unit,
            min_cm=min_cm,
            max_cost_ratio=max_cost_ratio,
            min_zone_for_range=min_zone_for_range,
            max_zone_for_range=max_zone_for_range,
        )

        st.markdown(f"**선택 SKU:** `{meta['아이템']}`  —  {meta['이름']}")
        st.write(f"- 원가: **{int(meta['원가합']):,}원**  |  레인지: **{int(meta['Min']):,} ~ {int(meta['Max']):,}원**  |  MSRP(자동/입력/오버라이드): **{(int(meta['MSRP']):,}원) if meta['MSRP']==meta['MSRP'] else 'N/A'}**")
        if meta["노트"]:
            st.warning(meta["노트"])

        if zdf.empty:
            st.info("레인지 또는 채널 파라미터가 부족하여 계산 결과가 없습니다.")
        else:
            # show table
            st.dataframe(zdf, use_container_width=True, height=360)

            # quick insight: which zones are impossible
            bad = zdf[zdf["상태"].str.contains("불가", na=False)]
            if not bad.empty:
                st.error("일부 가격영역이 구조적으로 불가능합니다(Floor가 밴드 상단을 초과). 보통 해결책: (1) 해당 채널 제외, (2) Q세트로 전환, (3) MSRP/Max 상향, (4) 원가 개선")

            # export
            xb = to_excel_bytes({"sku_zone": zdf})
            st.download_button("이 SKU 결과 엑셀 다운로드", data=xb, file_name=f"{sku}_band.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================
# 3) Set Builder (BOM)
# =============================
with tab_set:
    st.subheader("세트 빌더 (구성품 리스트업 → 자동 가격 추천)")
    prod = st.session_state["products_df"].copy()
    if prod.empty:
        st.warning("업로드 탭에서 원가/상품마스터를 먼저 업로드하세요.")
    else:
        # Create/edit sets
        st.markdown("### A. 세트 생성")
        c1, c2, c3 = st.columns([1,2,1])
        with c1:
            new_set_id = st.text_input("세트ID", value="")
        with c2:
            new_set_name = st.text_input("세트명", value="")
        with c3:
            if st.button("세트 추가", type="primary", disabled=(not new_set_id.strip() or not new_set_name.strip())):
                sets = st.session_state["sets_df"].copy()
                if (sets["세트ID"] == new_set_id.strip()).any():
                    st.warning("이미 존재하는 세트ID입니다.")
                else:
                    sets = pd.concat([sets, pd.DataFrame([{"세트ID":new_set_id.strip(), "세트명":new_set_name.strip(), "MSRP_오버라이드":np.nan}])], ignore_index=True)
                    st.session_state["sets_df"] = sets
                    st.success("세트 추가 완료")

        st.markdown("### B. 세트 리스트/수정")
        if st.session_state["sets_df"].empty:
            st.info("세트를 추가하세요.")
        else:
            sets_edit = st.data_editor(st.session_state["sets_df"], use_container_width=True, height=180, num_rows="dynamic", key="sets_editor")
            st.session_state["sets_df"] = sets_edit.copy()

            st.markdown("### C. BOM(구성품) 입력")
            set_options = (st.session_state["sets_df"]["세트ID"].astype(str) + " | " + st.session_state["sets_df"]["세트명"].astype(str)).tolist()
            pick_set = st.selectbox("편집할 세트 선택", options=set_options, index=0)
            set_id = pick_set.split(" | ",1)[0].strip()

            # BOM editor for selected set
            bom = st.session_state["bom_df"].copy()
            bom = bom[bom["세트ID"] == set_id].copy()
            # Provide simple add UI
            sku_options = (prod["품번"].astype(str) + " | " + prod["상품명"].astype(str)).tolist()
            c1, c2, c3 = st.columns([3,1,1])
            with c1:
                sel_sku = st.selectbox("구성품 선택", options=sku_options, index=0, key=f"bom_sel_{set_id}")
                sku = sel_sku.split(" | ",1)[0].strip()
            with c2:
                qty = st.number_input("수량", min_value=1, value=1, step=1, key=f"bom_qty_{set_id}")
            with c3:
                if st.button("구성품 추가", key=f"bom_add_{set_id}"):
                    all_bom = st.session_state["bom_df"].copy()
                    all_bom = pd.concat([all_bom, pd.DataFrame([{"세트ID":set_id, "품번":sku, "수량":int(qty)}])], ignore_index=True)
                    st.session_state["bom_df"] = all_bom
                    st.success("추가됨")

            # show and allow edit
            bom_all = st.session_state["bom_df"].copy()
            bom_view = bom_all[bom_all["세트ID"] == set_id].copy()
            if bom_view.empty:
                st.info("BOM이 비어 있습니다. 구성품을 추가하세요.")
            else:
                bom_view = bom_view.merge(prod[["품번","상품명","원가"]], on="품번", how="left")
                st.dataframe(bom_view, use_container_width=True, height=220)

                # quick remove
                if st.button("이 세트 BOM 전체 삭제", type="secondary", key=f"bom_clear_{set_id}"):
                    bom_all = bom_all[bom_all["세트ID"] != set_id].copy()
                    st.session_state["bom_df"] = bom_all
                    st.success("삭제 완료")

            st.divider()
            st.markdown("### D. 세트 가격 추천(채널별) + 구성품 배분단가/할인율")
            # policy inputs
            p1, p2, p3 = st.columns([1,1,1])
            with p1:
                rounding_unit = st.selectbox("반올림 단위(세트)", [10,100,1000], index=2, key="set_round")
            with p2:
                max_cost_ratio = st.number_input("MSRP 원가율 상한(세트)", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="set_costratio")
            with p3:
                min_cm = st.slider("최소 기여이익률(세트, %)", 0, 50, 15, 1, key="set_cm") / 100.0

            min_zone_for_range = st.selectbox("세트 레인지 Min 기준", PRICE_ZONES, index=0, key="set_minz")
            max_zone_for_range = st.selectbox("세트 레인지 Max 기준", ["MSRP"] + PRICE_ZONES, index=0, key="set_maxz")

            if st.button("세트 추천가 계산", type="primary", key="set_calc"):
                if st.session_state["bom_df"][st.session_state["bom_df"]["세트ID"]==set_id].empty:
                    st.error("BOM이 비어 있어 계산할 수 없습니다.")
                else:
                    zdf, meta = build_zone_table_for_item(
                        item_type="SET",
                        item_id=set_id,
                        products_df=prod,
                        sets_df=st.session_state["sets_df"],
                        bom_df=st.session_state["bom_df"],
                        channels_df=st.session_state["channels_df"],
                        zone_map=st.session_state["zone_map"],
                        boundaries=st.session_state["boundaries"],
                        target_pos=st.session_state["target_pos"],
                        rounding_unit=rounding_unit,
                        min_cm=min_cm,
                        max_cost_ratio=max_cost_ratio,
                        min_zone_for_range=min_zone_for_range,
                        max_zone_for_range=max_zone_for_range,
                    )

                    st.markdown(f"**세트:** `{set_id}` — {meta['이름']}")
                    st.write(f"- 세트 원가합: **{int(meta['원가합']):,}원** | 레인지: **{int(meta['Min']):,} ~ {int(meta['Max']):,}원**")
                    if meta["노트"]:
                        st.warning(meta["노트"])
                    st.dataframe(zdf, use_container_width=True, height=320)

                    # Build SKU always price map for allocation
                    # We compute each component SKU's '상시' target quickly using current band defaults.
                    # (If SKU has overrides, it will be reflected.)
                    sku_always = {}
                    for sku in st.session_state["bom_df"][st.session_state["bom_df"]["세트ID"]==set_id]["품번"].unique():
                        z_sku, _ = build_zone_table_for_item(
                            item_type="SKU",
                            item_id=str(sku),
                            products_df=prod,
                            sets_df=st.session_state["sets_df"],
                            bom_df=st.session_state["bom_df"],
                            channels_df=st.session_state["channels_df"],
                            zone_map=st.session_state["zone_map"],
                            boundaries=st.session_state["boundaries"],
                            target_pos=st.session_state["target_pos"],
                            rounding_unit=rounding_unit,
                            min_cm=min_cm,
                            max_cost_ratio=max_cost_ratio,
                            min_zone_for_range="공구",
                            max_zone_for_range="MSRP",
                        )
                        if not z_sku.empty:
                            always_row = z_sku[z_sku["가격영역"]=="상시"]
                            if not always_row.empty:
                                sku_always[str(sku)] = float(always_row.iloc[0]["추천가(Target)"])
                    # Allocation view for selected zones
                    st.markdown("#### 구성품 배분단가/할인율(상시가 기준 가중치)")
                    alloc_rows = []
                    for _, rr in zdf.iterrows():
                        zone = rr["가격영역"]
                        set_price = rr["추천가(Target)"]
                        alloc = allocate_set_price_to_components(set_id, zone, set_price, prod, st.session_state["sets_df"], st.session_state["bom_df"], sku_always)
                        if not alloc.empty:
                            alloc_rows.append(alloc)
                    if alloc_rows:
                        alloc_df = pd.concat(alloc_rows, ignore_index=True)
                        st.dataframe(alloc_df, use_container_width=True, height=320)
                        xb = to_excel_bytes({"set_zone": zdf, "alloc": alloc_df})
                        st.download_button("세트 결과 엑셀 다운로드", data=xb, file_name=f"{set_id}_set_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    else:
                        st.info("구성품 상시가(ref)가 부족하여 배분 계산이 제한됩니다. (구성품 SKU 원가/레인지가 산출되어야 함)")

# =============================
# 4) Plan & Cannibalization
# =============================
with tab_plan:
    st.subheader("운영플랜(채널별 오퍼 배치) → 카니발(역전/갭부족) 직관 체크")
    prod = st.session_state["products_df"].copy()
    sets_df = st.session_state["sets_df"].copy()
    if prod.empty:
        st.warning("업로드 탭에서 원가/상품마스터를 먼저 업로드하세요.")
    else:
        st.caption("플랜에 단품/세트를 채널별로 넣으면, SKU 기준 '실질 최저단가'를 만들고 채널 서열대로 역전/갭부족을 잡아냅니다.")
        plan = st.session_state["plan_df"].copy()

        # Add offer row
        c1, c2, c3, c4 = st.columns([1,1,2,1])
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
                if sets_df.empty:
                    st.warning("세트가 없습니다. 세트 빌더에서 세트를 추가하세요.")
                    oid = ""
                else:
                    opts = (sets_df["세트ID"].astype(str) + " | " + sets_df["세트명"].astype(str)).tolist()
                    pick = st.selectbox("오퍼 선택", opts, index=0, key="plan_pick_set")
                    oid = pick.split(" | ",1)[0].strip()
        with c4:
            price_override = st.number_input("가격 오버라이드(0=추천가)", min_value=0, value=0, step=1000, key="plan_price")
        if st.button("플랜에 추가", type="primary", disabled=(otype=="SET" and oid=="")):
            plan = pd.concat([plan, pd.DataFrame([{
                "가격영역": zone,
                "오퍼타입": otype,
                "오퍼ID": oid,
                "가격_오버라이드": float(price_override) if price_override>0 else np.nan
            }])], ignore_index=True)
            st.session_state["plan_df"] = plan

        st.divider()
        st.markdown("### A. 현재 플랜")
        if st.session_state["plan_df"].empty:
            st.info("아직 플랜이 없습니다. 위에서 채널별 오퍼를 추가하세요.")
        else:
            plan_edit = st.data_editor(st.session_state["plan_df"], use_container_width=True, height=240, num_rows="dynamic", key="plan_editor")
            st.session_state["plan_df"] = plan_edit.copy()
            plan = plan_edit.copy()

        st.divider()
        st.markdown("### B. 카니발 체크 실행")
        p1, p2, p3 = st.columns([1,1,1])
        with p1:
            rounding_unit = st.selectbox("반올림 단위(플랜)", [10,100,1000], index=2, key="plan_round")
        with p2:
            max_cost_ratio = st.number_input("MSRP 원가율 상한(플랜)", min_value=0.05, max_value=0.80, value=0.30, step=0.01, format="%.2f", key="plan_costratio")
        with p3:
            min_cm = st.slider("최소 기여이익률(플랜, %)", 0, 50, 15, 1, key="plan_cm") / 100.0
        gap_pct = st.slider("최소 갭(%)", 0, 20, 5, 1, key="plan_gap_pct") / 100.0
        gap_won = st.number_input("최소 갭(원)", min_value=0, value=2000, step=500, key="plan_gap_won")

        if st.button("카니발 체크", type="primary", key="run_cannibal"):
            if plan.empty:
                st.error("플랜이 비어 있습니다.")
            else:
                # Precompute SKU '상시' prices for allocation baseline
                sku_always = {}
                for sku in prod["품번"].astype(str).tolist():
                    z_sku, _ = build_zone_table_for_item(
                        item_type="SKU",
                        item_id=str(sku),
                        products_df=prod,
                        sets_df=sets_df,
                        bom_df=st.session_state["bom_df"],
                        channels_df=st.session_state["channels_df"],
                        zone_map=st.session_state["zone_map"],
                        boundaries=st.session_state["boundaries"],
                        target_pos=st.session_state["target_pos"],
                        rounding_unit=rounding_unit,
                        min_cm=min_cm,
                        max_cost_ratio=max_cost_ratio,
                        min_zone_for_range="공구",
                        max_zone_for_range="MSRP",
                    )
                    if not z_sku.empty:
                        always_row = z_sku[z_sku["가격영역"]=="상시"]
                        if not always_row.empty:
                            sku_always[str(sku)] = float(always_row.iloc[0]["추천가(Target)"])

                # Build per-offer effective SKU unit prices
                eff_rows = []
                for _, pr in plan.iterrows():
                    zone = pr["가격영역"]
                    otype = pr["오퍼타입"]
                    oid = str(pr["오퍼ID"])
                    price_override = safe_float(pr.get("가격_오버라이드", np.nan), np.nan)

                    if otype == "SKU":
                        z_sku, _ = build_zone_table_for_item(
                            item_type="SKU",
                            item_id=oid,
                            products_df=prod,
                            sets_df=sets_df,
                            bom_df=st.session_state["bom_df"],
                            channels_df=st.session_state["channels_df"],
                            zone_map=st.session_state["zone_map"],
                            boundaries=st.session_state["boundaries"],
                            target_pos=st.session_state["target_pos"],
                            rounding_unit=rounding_unit,
                            min_cm=min_cm,
                            max_cost_ratio=max_cost_ratio,
                            min_zone_for_range="공구",
                            max_zone_for_range="MSRP",
                        )
                        if z_sku.empty:
                            continue
                        rr = z_sku[z_sku["가격영역"]==zone]
                        if rr.empty:
                            continue
                        p = float(rr.iloc[0]["추천가(Target)"])
                        if not np.isnan(price_override) and price_override>0:
                            p = float(price_override)
                        eff_rows.append({"가격영역":zone, "품번":oid, "실질단가":p, "오퍼":f"SKU:{oid}"})
                    else:
                        # SET: allocate price to components
                        z_set, _ = build_zone_table_for_item(
                            item_type="SET",
                            item_id=oid,
                            products_df=prod,
                            sets_df=sets_df,
                            bom_df=st.session_state["bom_df"],
                            channels_df=st.session_state["channels_df"],
                            zone_map=st.session_state["zone_map"],
                            boundaries=st.session_state["boundaries"],
                            target_pos=st.session_state["target_pos"],
                            rounding_unit=rounding_unit,
                            min_cm=min_cm,
                            max_cost_ratio=max_cost_ratio,
                            min_zone_for_range="공구",
                            max_zone_for_range="MSRP",
                        )
                        if z_set.empty:
                            continue
                        rr = z_set[z_set["가격영역"]==zone]
                        if rr.empty:
                            continue
                        set_price = float(rr.iloc[0]["추천가(Target)"])
                        if not np.isnan(price_override) and price_override>0:
                            set_price = float(price_override)

                        alloc = allocate_set_price_to_components(oid, zone, set_price, prod, sets_df, st.session_state["bom_df"], sku_always)
                        if alloc.empty:
                            continue
                        for _, ar in alloc.iterrows():
                            eff_rows.append({
                                "가격영역": zone,
                                "품번": str(ar["품번"]),
                                "실질단가": float(ar["실질단가"]),
                                "오퍼": f"SET:{oid}"
                            })

                eff = pd.DataFrame(eff_rows)
                if eff.empty:
                    st.error("실질단가 계산 결과가 없습니다. (세트 BOM 누락 또는 SKU 계산 불가)")
                else:
                    # per zone, per sku: min effective unit price
                    min_eff = eff.groupby(["가격영역","품번"], as_index=False)["실질단가"].min()
                    min_eff = min_eff.merge(prod[["품번","상품명"]], on="품번", how="left")

                    st.markdown("#### 채널별 SKU 최저 실질단가")
                    st.dataframe(min_eff.sort_values(["품번","가격영역"]), use_container_width=True, height=320)

                    # Cannibal check across ordered zones
                    order = {z:i for i,z in enumerate(PRICE_ZONES)}  # low->high order already
                    # Check each SKU across zones where it exists
                    viol = []
                    for sku, g in min_eff.groupby("품번"):
                        g2 = g.copy()
                        g2["ord"] = g2["가격영역"].map(order)
                        g2 = g2.sort_values("ord")
                        # compare adjacent
                        prev = None
                        prev_zone = None
                        for _, rr in g2.iterrows():
                            if prev is None:
                                prev = rr["실질단가"]
                                prev_zone = rr["가격영역"]
                                continue
                            cur = rr["실질단가"]
                            cur_zone = rr["가격영역"]
                            need = max(prev*(1+gap_pct), prev+gap_won)
                            if cur < need:
                                viol.append({
                                    "품번": sku,
                                    "상품명": rr.get("상품명",""),
                                    "하위영역": prev_zone,
                                    "하위가격": prev,
                                    "상위영역": cur_zone,
                                    "상위가격": cur,
                                    "필요최소상위가": need,
                                    "갭부족": need-cur
                                })
                            prev = cur
                            prev_zone = cur_zone

                    viol_df = pd.DataFrame(viol)
                    st.divider()
                    st.markdown("#### 카니발(역전/갭부족) 경고")
                    if viol_df.empty:
                        st.success("카니발 경고 없음(현재 플랜/갭 기준).")
                    else:
                        st.warning(f"{len(viol_df):,}건")
                        st.dataframe(viol_df, use_container_width=True, height=260)

                    xb = to_excel_bytes({"min_eff": min_eff, "violations": viol_df})
                    st.download_button("카니발 체크 결과 엑셀 다운로드", data=xb, file_name="cannibal_check.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================
# 5) Logic (plain-language)
# =============================
with tab_logic:
    st.subheader("가격 로직(문장) — v5 기준")
    st.markdown(
        """
### 1) 입력의 의미
- 업로드 파일은 **상품명 통일 + 원가**를 위한 파일입니다. (품번=상품코드, 원가(vat-))
- 채널 비용(수수료/PG/배송/마케팅/반품)은 **키인 즉시** 손익하한(Floor)과 추천가(Target)에 반영됩니다.

### 2) 레인지(Min~Max) 먼저
- 각 SKU(또는 세트)는 먼저 **최저가(Min)**와 **최고가(Max)**를 가집니다.
- 기본값:
  - Min = 선택한 최저 영역(예: 공구)의 **손익하한(Floor)**
  - Max = MSRP(소비자가)
- Min/Max는 SKU별로 오버라이드 가능(고정 가능).
- Max가 Min보다 작거나 같으면, 밴드가 펼쳐지지 않으므로 앱이 경고를 띄우고 **Max를 임시 확장**합니다.

### 3) MSRP(소비자가) 산정
- MSRP는 다음 후보 중 **가장 큰 값**을 택하고, 반올림 단위로 올림 처리합니다.
  - (a) 업로드 파일의 MSRP 입력값(있다면)
  - (b) 원가율 상한 정책값: **원가 / 0.30** (원가율 30% 상한)
  - (c) MSRP 오버라이드(수동 입력)
- 즉, **원가가 내려가면 정책 MSRP도 내려갈 수 있으므로**, MSRP를 고정하고 싶다면 MSRP_오버라이드를 입력합니다.

### 4) 밴드(가격영역 구간)
- Min~Max 레인지(0~100%)를 **10개 가격영역**으로 분할합니다.
- 각 영역은 (BandLow~BandHigh) 구간을 가지며, Target은 기본적으로 구간의 중앙(또는 설정한 위치)에 놓습니다.
- 경계값은 슬라이더로 조정(마우스로 드래그).

### 5) 손익하한(Floor)과 Target 결정
- 각 가격영역은 실제 비용채널(자사몰/쿠팡/공구 등)에 매핑됩니다.
- 손익하한(Floor)은 아래 식으로 계산합니다.
  - Floor = (원가 + 배송비/주문 + 반품률×반품비/주문) / (1 - 수수료 - PG - 마케팅비 - 최소기여이익률)
- 최종 Target은:
  - Target = max(밴드 내부 위치값, Floor)
  - 단, Floor가 BandHigh를 넘으면 해당 영역은 **불가(Floor>BandHigh)** 로 표시됩니다.

### 6) 세트(BOM) 가격 추천
- 세트는 BOM(구성품 리스트업)으로 만들며,
  - 세트원가 = Σ(구성품원가×수량)
  - 세트도 동일하게 Min~Max 레인지 → 밴드 → Floor → Target 로 가격을 추천합니다.
- 세트 가격을 구성품별로 배분할 때는,
  - 각 구성품의 상시가(ref)를 가중치로 사용하여 배분 단가/할인율을 계산합니다.
  - 구성은 달라도 같은 SKU가 포함된 세트끼리, 해당 SKU의 **실질단가**를 비교할 수 있습니다.

### 7) 카니발 체크(운영플랜)
- 채널별로 단품/세트를 배치하면, SKU 기준으로 각 채널의 **최저 실질단가**를 만들고
- 가격영역 서열(저가→고가) 기준으로
  - 상위영역 가격 ≥ max(하위가격×(1+갭%), 하위가격+갭원)
  를 만족하지 못하면 카니발 경고로 표시합니다.
"""
    )