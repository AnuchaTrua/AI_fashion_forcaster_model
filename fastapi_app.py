# fastapi_app.py

import os
import json
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

# ================== PATH CONFIG ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

MODEL_PATH = os.path.join(MODELS_DIR, "fashion_lstm.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_x.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")
DAILY_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_store_sales.csv")
PRODUCTS_CSV_PATH = os.path.join(DATA_RAW_DIR, "fashion_boutique_dataset.csv")

# ================== LOAD MODEL & DATA ==================

# 1) โมเดล LSTM
try:
    # compile=False เพราะตอนโหลดเราไม่ต้องการ metric 'mae'/'mape' แล้ว
    model = keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"ไม่สามารถโหลดโมเดลจาก {MODEL_PATH}: {e}")

# 2) scaler
try:
    scaler_x = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"ไม่สามารถโหลด scaler จาก {SCALER_PATH}: {e}")

# 3) config
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception as e:
    raise RuntimeError(f"ไม่สามารถโหลด config จาก {CONFIG_PATH}: {e}")

WINDOW_SIZE = config["window_size"]
FEATURE_COLS = config["feature_cols"]   # เช่น ["total_qty", "total_revenue", ...]
TARGET_COL = config.get("target_col", "total_qty")

# 4) daily time-series (ยอดรวมทั้งร้านรายวัน)
try:
    daily_df = pd.read_csv(DAILY_CSV_PATH, parse_dates=["purchase_date"])
    daily_df = daily_df.sort_values("purchase_date").reset_index(drop=True)
except Exception as e:
    raise RuntimeError(f"ไม่สามารถโหลด daily_store_sales.csv จาก {DAILY_CSV_PATH}: {e}")

# 5) product list (ฐานข้อมูลร้านปลอม)
try:
    products_df = pd.read_csv(PRODUCTS_CSV_PATH)
except Exception as e:
    raise RuntimeError(f"ไม่สามารถโหลด products CSV จาก {PRODUCTS_CSV_PATH}: {e}")

# คอลัมน์ที่เราจะส่งให้ frontend (เพิ่ม markdown_percentage ด้วย)
PRODUCT_EXPORT_COLS = [
    "product_id",
    "category",
    "brand",
    "season",
    "color",
    "size",
    "current_price",
    "markdown_percentage",  # 👈 ใช้โชว์ส่วนลดเดิมใน UI
    "stock_quantity",
]

# ================== FASTAPI APP ==================

app = FastAPI(title="AI Fashion Forecaster API")

# CORS: ให้ React (localhost:5173 ฯลฯ) เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev mode: เปิดหมด (ถ้า deploy จริงค่อยล็อก)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Pydantic Models ==================

class DayFeature(BaseModel):
    total_qty: float
    total_revenue: float
    avg_discount: float
    avg_rating: float
    dayofweek: int
    is_weekend: int
    month: int
    year: int


class PredictRequest(BaseModel):
    last_days: List[DayFeature]


class PredictItemStockRequest(BaseModel):
    product_id: str
    horizon_days: int  # เช่น 7 หรือ 30
    current_stock: Optional[float] = None  # ถ้าไม่ส่ง จะใช้จาก CSV แทน

    # 👇 scenario เสริมสำหรับลองเล่นราคา/ส่วนลด
    scenario_price: Optional[float] = None        # ราคาจำลอง (ต่อชิ้น)
    scenario_discount: Optional[float] = None     # ส่วนลดจำลอง (%)

# ================== Utils ==================

def row_to_serializable(row: pd.Series, cols: list) -> dict:
    """
    แปลง pandas row ให้เป็น dict ที่ JSON-friendly:
    - NaN / inf -> None
    - numpy float/int -> float/int ปกติ
    """
    out = {}
    for c in cols:
        v = row.get(c)
        # float
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                out[c] = None
            else:
                out[c] = float(v)
        # int
        elif isinstance(v, (np.integer, int)):
            out[c] = int(v)
        else:
            # string/อื่น ๆ
            if isinstance(v, str):
                out[c] = v
            else:
                out[c] = None if pd.isna(v) else v
    return out


def forecast_store_multi_step(horizon_days: int) -> list[float]:
    """
    ใช้ LSTM เดิม ทำนาย demand รวม 'ทั้งร้าน' ล่วงหน้า horizon_days วัน
    โดยวน predict ทีละวัน (multi-step forecasting)
    """
    if len(daily_df) < WINDOW_SIZE:
        raise ValueError(
            f"จำนวนข้อมูลรายวัน ({len(daily_df)}) น้อยกว่า window_size={WINDOW_SIZE}"
        )

    # เอา WINDOW_SIZE วันล่าสุดเป็น window เริ่มต้น (ยังไม่ scale)
    window = daily_df.tail(WINDOW_SIZE).copy()
    window_unscaled = window[FEATURE_COLS].values.astype("float32")

    last_date = window["purchase_date"].iloc[-1]
    forecasts: list[float] = []

    for _ in range(horizon_days):
        # scale แล้วส่งเข้าโมเดล
        X_scaled = scaler_x.transform(window_unscaled)
        X_input = np.expand_dims(X_scaled, axis=0)  # (1, window_size, n_features)

        y_pred = model.predict(X_input, verbose=0).flatten()[0]
        forecasts.append(float(y_pred))

        # สร้าง feature สำหรับวันถัดไป
        next_date = last_date + pd.Timedelta(days=1)
        dayofweek = next_date.dayofweek
        is_weekend = 1 if dayofweek >= 5 else 0
        month = next_date.month
        year = next_date.year

        # เอาค่าจากวันล่าสุดมาใช้เป็น approx สำหรับ revenue/discount/rating
        last_unscaled = window_unscaled[-1].copy()
        last_total_revenue = float(last_unscaled[1])
        last_avg_discount = float(last_unscaled[2])
        last_avg_rating = float(last_unscaled[3])

        next_row = np.array(
            [
                y_pred,             # total_qty (predicted)
                last_total_revenue, # ใช้ค่าเดิมไปก่อน
                last_avg_discount,
                last_avg_rating,
                dayofweek,
                is_weekend,
                month,
                year,
            ],
            dtype="float32",
        )

        # slide window ไปข้างหน้า 1 วัน
        window_unscaled = np.vstack([window_unscaled[1:], next_row])
        last_date = next_date

    return forecasts


def get_product_share(product_row: pd.Series) -> float:
    """
    คำนวณ 'สัดส่วน' ของสินค้าเทียบทั้งร้าน แบบง่ายจาก stock_quantity
    เพื่อใช้กระจาย demand รวมลงเป็น demand ต่อสินค้า
    """
    if "stock_quantity" not in products_df.columns:
        return 1.0

    total_stock = products_df["stock_quantity"].sum()
    if total_stock <= 0:
        return 1.0

    stock_i = float(product_row.get("stock_quantity", 0.0) or 0.0)
    share = stock_i / total_stock

    # กำหนดขั้นต่ำของ share เช่น 1%
    min_share = 0.01
    if share <= 0:
        share = min_share
    else:
        share = max(share, min_share)

    return float(share)


def compute_scenario_factor(
    product_row: pd.Series,
    scenario_price: Optional[float],
    scenario_discount: Optional[float],
) -> float:
    """
    คำนวณ factor ปรับ demand ของสินค้านี้จากราคา/ส่วนลดจำลองแบบง่าย ๆ
    - ถ้าลดราคา → factor > 1 (ความต้องการเพิ่ม)
    - ถ้าขึ้นราคา → factor < 1 (ความต้องการลด)
    """

    # ราคาและส่วนลดเดิมจากฐานข้อมูล
    base_price = float(product_row.get("current_price", 0.0) or 0.0)
    base_discount = float(product_row.get("markdown_percentage", 0.0) or 0.0)

    # ถ้าฐานไม่มีราคาเลย → ไม่ปรับอะไร
    if base_price <= 0:
        return 1.0

    # ใช้ scenario ถ้ามี ไม่งั้น fallback เป็นค่าเดิม
    eff_price = scenario_price if scenario_price is not None else base_price
    eff_discount = (
        scenario_discount if scenario_discount is not None else base_discount
    )

    # effective price = price * (1 - discount%)
    eff_base = base_price * (1.0 - base_discount / 100.0)
    eff_scenario = eff_price * (1.0 - eff_discount / 100.0)

    if eff_base <= 0 or eff_scenario <= 0:
        return 1.0

    # ratio > 1 ถ้า scenario ถูกกว่าเดิม
    ratio = eff_base / eff_scenario

    # สมมติ elasticity แบบง่าย ๆ
    elasticity = 0.8  # ยิ่งสูง ความไวต่อราคายิ่งมาก
    factor = ratio ** elasticity

    # กันไม่ให้เยอะหรือน้อยเกินไป
    factor = max(0.3, min(2.5, factor))
    return float(factor)

# ================== Routes ==================

@app.get("/")
def root():
    return {"message": "AI Fashion Forecaster API is running"}


@app.get("/latest_series")
def latest_series():
    """
    ส่งข้อมูล WINDOW_SIZE วันล่าสุด (ยอดรวมทั้งร้าน) ให้ frontend
    ดูแพตเทิร์น และใช้เป็นข้อมูลประกอบ
    """
    if len(daily_df) < WINDOW_SIZE:
        return {
            "success": False,
            "message": f"ข้อมูลใน CSV มี {len(daily_df)} แถว น้อยกว่า window_size={WINDOW_SIZE}",
        }

    last = daily_df.tail(WINDOW_SIZE).copy()
    result = []
    for _, row in last.iterrows():
        result.append(
            {
                "date": row["purchase_date"].strftime("%Y-%m-%d"),
                "total_qty": float(row["total_qty"]),
                "total_revenue": float(row["total_revenue"]),
                "avg_discount": float(row["avg_discount"]),
                "avg_rating": float(row["avg_rating"]),
                "dayofweek": int(row["dayofweek"]),
                "is_weekend": int(row["is_weekend"]),
                "month": int(row["month"]),
                "year": int(row["year"]),
            }
        )

    return {
        "success": True,
        "window_size": WINDOW_SIZE,
        "last_days": result,
    }


@app.post("/predict_next_day")
def predict_next_day(req: PredictRequest):
    """
    endpoint เดิม: ทำนาย demand รวม 'วันถัดไป' จาก sequence WINDOW_SIZE วัน
    (เก็บไว้เผื่ออยากใช้ที่อื่น)
    """
    if len(req.last_days) != WINDOW_SIZE:
        return {
            "success": False,
            "message": f"ต้องส่ง last_days มา {WINDOW_SIZE} วันพอดี (ตอนนี้ {len(req.last_days)})",
        }

    df = pd.DataFrame([d.dict() for d in req.last_days])
    df = df[FEATURE_COLS]

    X = scaler_x.transform(df.values.astype("float32"))
    X_input = np.expand_dims(X, axis=0)
    y_pred = model.predict(X_input, verbose=0).flatten()[0]

    return {"success": True, "predicted_demand_next_day": float(y_pred)}


@app.get("/products")
def list_products():
    """
    ส่ง list สินค้าให้ frontend เลือก (dropdown)
    """
    cols = [c for c in PRODUCT_EXPORT_COLS if c in products_df.columns]
    items = []
    for _, row in products_df[cols].iterrows():
        items.append(row_to_serializable(row, cols))

    return {"success": True, "items": items}


@app.post("/predict_item_stock")
def predict_item_stock(req: PredictItemStockRequest):
    """
    ผู้ใช้เลือกสินค้า + horizon (7/30 วัน) + current_stock
    + (ทางเลือก) scenario_price, scenario_discount
    → ทำนาย demand & stock คงเหลือของสินค้านั้น
    """
    # หา product
    product_row = products_df.loc[products_df["product_id"] == req.product_id]
    if product_row.empty:
        return {
            "success": False,
            "message": f"ไม่พบสินค้า product_id = {req.product_id}",
        }

    product_row = product_row.iloc[0]

    # กำหนดสต็อกตั้งต้น
    if req.current_stock is not None:
        base_stock = float(req.current_stock)
    else:
        base_stock = float(product_row.get("stock_quantity", 0.0) or 0.0)

    horizon = int(req.horizon_days)
    if horizon <= 0:
        return {"success": False, "message": "horizon_days ต้องมากกว่า 0"}

    # 1) ทำนาย demand รวมทั้งร้าน ล่วงหน้า horizon วัน
    try:
        store_forecasts = forecast_store_multi_step(horizon)
    except Exception as e:
        return {
            "success": False,
            "message": f"พยากรณ์ demand รวมทั้งร้านไม่สำเร็จ: {e}",
        }

    # 2) share ของสินค้านี้
    share_i = get_product_share(product_row)

    # 3) factor จาก scenario ราคา/ส่วนลด
    scenario_factor = compute_scenario_factor(
        product_row,
        req.scenario_price,
        req.scenario_discount,
    )

    # 4) demand รายวันของสินค้านี้ (share * factor)
    item_daily_demand = [
        float(f * share_i * scenario_factor) for f in store_forecasts
    ]
    total_future_demand = float(np.sum(item_daily_demand))

    # 5) stock คงเหลือ
    predicted_left = base_stock - total_future_demand
    if predicted_left < 0:
        predicted_left = 0.0

    return {
        "success": True,
        "product_id": req.product_id,
        "horizon_days": horizon,
        "base_stock": base_stock,
        "product_share": share_i,
        "scenario_price": req.scenario_price,
        "scenario_discount": req.scenario_discount,
        "scenario_factor": scenario_factor,
        "forecast_store_daily": store_forecasts,
        "forecast_item_daily": item_daily_demand,
        "total_future_demand": total_future_demand,
        "predicted_stock_left": predicted_left,
    }
