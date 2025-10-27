import os
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from iqoptionapi.stable_api import IQ_Option
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------
# CONFIGURACIÓN
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DelowyssProRender")

IQ_EMAIL = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "")
IQ_MODE = os.getenv("IQ_MODE", "PRACTICE")
PORT = int(os.getenv("PORT", 10000))
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

XGB_FILE = os.path.join(MODEL_DIR, "xgboost_model.joblib")
RF_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
NN_FILE = os.path.join(MODEL_DIR, "nn_model.h5")

ASSETS = ["EURUSD-OTC", "EURUSD"]

# -----------------------
# IQ CONNECTOR (REALTIME)
# -----------------------
class IQConnector:
    def __init__(self, email: str, password: str, mode: str = "PRACTICE"):
        self.email = email
        self.password = password
        self.mode = mode
        self.api = None
        self.connected = False
        self.available_assets = {}
        self.realtime_data = {}
        self.reconnect_lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            logger.info("🔗 [IQ] Conectando a IQ Option...")
            self.api = IQ_Option(self.email, self.password)
            self.api.connect()
            if not self.api.check_connect():
                logger.error("❌ No se pudo conectar a IQ Option")
                self.connected = False
                return
            if self.mode.upper() == "REAL":
                self.api.change_balance("REAL")
                logger.info("💰 [IQ] Modo REAL activado")
            else:
                self.api.change_balance("PRACTICE")
                logger.info("🎯 [IQ] Modo DEMO activado")
            self.connected = True
            self.available_assets = self.api.get_all_open_time()
            logger.info(f"✅ [IQ] Conectado - Activos: {len(self.available_assets)}")
        except Exception as e:
            logger.error(f"❌ [IQ] Error conectando: {e}")
            self.connected = False

    def reconnect(self):
        with self.reconnect_lock:
            if not self.connected:
                logger.warning("🔁 Reintentando conexión con IQ Option...")
                self._connect()

    def start_realtime_stream(self, asset: str = "EURUSD-OTC", timeframe_min: int = 1):
        if not self.connected:
            logger.error("❌ [IQ] No conectado para iniciar stream")
            return
        try:
            period = timeframe_min * 60
            logger.info(f"📡 Iniciando stream en tiempo real para {asset}")
            self.api.start_candles_stream(asset, period, 100)

            def stream_loop():
                while self.connected:
                    try:
                        candles = self.api.get_realtime_candles(asset, period)
                        if candles:
                            df = pd.DataFrame.from_dict(candles, orient="index")
                            df["timestamp"] = pd.to_datetime(df["from"], unit="s")
                            df.sort_values("timestamp", inplace=True)
                            self.realtime_data[asset] = df[["timestamp", "open", "max", "min", "close", "volume"]]
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"⚠️ Error en stream: {e}")
                        self.reconnect()
                        time.sleep(5)

            t = threading.Thread(target=stream_loop, daemon=True)
            t.start()
        except Exception as e:
            logger.error(f"❌ Error iniciando stream: {e}")

    def get_realtime_candles(self, asset: str):
        if asset in self.realtime_data:
            return self.realtime_data[asset].copy()
        logger.warning(f"⚠️ No hay datos en tiempo real para {asset}")
        return None

    def get_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        try:
            if not self.connected:
                self.reconnect()
            candles = self.api.get_candles(asset, timeframe_min * 60, count, time.time())
            df = pd.DataFrame([{
                "timestamp": datetime.fromtimestamp(c["from"]),
                "open": c["open"], "high": c["max"],
                "low": c["min"], "close": c["close"], "volume": c["volume"]
            } for c in candles])
            return df
        except Exception as e:
            logger.error(f"❌ Error obteniendo velas: {e}")
            return None

# -----------------------
# INDICADORES
# -----------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy().sort_values("timestamp")
        df["ema_fast"] = EMAIndicator(close=df["close"], window=5).ema_indicator()
        df["ema_slow"] = EMAIndicator(close=df["close"], window=15).ema_indicator()
        macd = MACD(close=df["close"])
        df["macd"], df["macd_signal"], df["macd_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        df["rsi"] = RSIIndicator(close=df["close"], window=12).rsi()
        adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["adx"] = adx.adx()
        bb = BollingerBands(close=df["close"], window=18, window_dev=2)
        df["bb_upper"], df["bb_lower"] = bb.bollinger_hband(), bb.bollinger_lband()
        df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=12).average_true_range()
        df["body"] = (df["close"] - df["open"]) / (df["open"] + 1e-8)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ Error calculando indicadores: {e}")
        return pd.DataFrame()

# -----------------------
# MODEL MANAGER
# -----------------------
class ModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.rf_model = None
        self.nn_model = None
        self.model_lock = threading.Lock()
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(SCALER_FILE): self.scaler = joblib.load(SCALER_FILE)
            if os.path.exists(XGB_FILE): self.xgb_model = joblib.load(XGB_FILE)
            if os.path.exists(RF_FILE): self.rf_model = joblib.load(RF_FILE)
            if os.path.exists(NN_FILE): self.nn_model = tf.keras.models.load_model(NN_FILE)
            logger.info("✅ Modelos cargados correctamente")
        except Exception as e:
            logger.warning(f"⚠️ Error cargando modelos: {e}")

    def build_and_train(self, df_features: pd.DataFrame):
        with self.model_lock:
            if len(df_features) < 50: return False
            X = df_features.drop(columns=["target"])
            y = df_features["target"].astype(int)
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
            xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1)
            xgb_clf.fit(X_train, y_train)
            rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8)
            rf_clf.fit(X_train, y_train)
            nn = Sequential([
                Dense(64, activation="relu", input_shape=(X_scaled.shape[1],)),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            nn.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            self.xgb_model, self.rf_model, self.nn_model = xgb_clf, rf_clf, nn
            self.save_models()
            return True

    def save_models(self):
        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.xgb_model, XGB_FILE)
        joblib.dump(self.rf_model, RF_FILE)
        self.nn_model.save(NN_FILE)

    def predict(self, df_features: pd.DataFrame):
        with self.model_lock:
            if any(m is None for m in [self.xgb_model, self.rf_model, self.nn_model]):
                return {"error": "Modelos no entrenados"}
            X = df_features.drop(columns=["target"])
            X_scaled = self.scaler.transform(X)
            last = X_scaled[-1:]
            xgb_pred = self.xgb_model.predict_proba(last)[0][1]
            rf_pred = self.rf_model.predict_proba(last)[0][1]
            nn_pred = float(self.nn_model.predict(last, verbose=0).ravel()[0])
            ensemble = 0.4 * xgb_pred + 0.3 * rf_pred + 0.3 * nn_pred
            direction = "ALCISTA" if ensemble >= 0.5 else "BAJISTA"
            return {"direction": direction, "confidence": float(abs(ensemble - 0.5) * 2), "xgb": xgb_pred, "rf": rf_pred, "nn": nn_pred}

# -----------------------
# ASSISTANT
# -----------------------
class DelowyssAssistant:
    def __init__(self, iq_connector: IQConnector, model_manager: ModelManager):
        self.iq = iq_connector
        self.models = model_manager
        self.last_analysis = None
        self.enabled = True

    def run_analysis_once(self, asset="EURUSD-OTC", realtime=False):
        if not self.enabled or not self.iq.connected:
            return {"error": "IQ Option no conectado o sistema desactivado"}
        df = self.iq.get_realtime_candles(asset) if realtime else self.iq.get_candles(asset)
        if df is None or df.empty:
            return {"error": "No se pudieron obtener velas"}
        df_ind = compute_indicators(df)
        if df_ind.empty:
            return {"error": "Sin indicadores válidos"}
        self.models.build_and_train(df_ind)
        prediction = self.models.predict(df_ind)
        self.last_analysis = {"asset": asset, "prediction": prediction, "timestamp": datetime.now().isoformat()}
        return self.last_analysis

# -----------------------
# INICIALIZACIÓN
# -----------------------
logger.info("🚀 Iniciando Delowyss Trading Professional...")
iq_conn = IQConnector(IQ_EMAIL, IQ_PASSWORD, IQ_MODE)
model_mgr = ModelManager()
assistant = DelowyssAssistant(iq_conn, model_mgr)

if iq_conn.connected:
    iq_conn.start_realtime_stream("EURUSD-OTC")

# -----------------------
# FASTAPI
# -----------------------
app = FastAPI(title="Delowyss Pro", version="3.0")

@app.get("/")
def index():
    return HTMLResponse("<h1>🧠 Delowyss Pro Trading IA Activo</h1>")

@app.get("/api/status")
def status():
    return {"iq_connected": iq_conn.connected, "models_ready": all([
        model_mgr.xgb_model, model_mgr.rf_model, model_mgr.nn_model
    ]), "last_analysis": assistant.last_analysis}

@app.post("/api/analyze")
def analyze():
    return assistant.run_analysis_once("EURUSD-OTC")

@app.post("/api/analyze/realtime")
def analyze_realtime():
    return assistant.run_analysis_once("EURUSD-OTC", realtime=True)

# -----------------------
# AUTO LEARNING CADA 30 MIN
# -----------------------
def auto_training():
    logger.info("🔄 Entrenamiento automático iniciado")
    try:
        assistant.run_analysis_once("EURUSD-OTC", realtime=True)
        logger.info("✅ Entrenamiento completado")
    except Exception as e:
        logger.error(f"❌ Error entrenamiento automático: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(auto_training, "interval", minutes=30)
scheduler.start()

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
