# main.py
import os
import time
import logging
from datetime import datetime, timedelta
import threading
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Keras for neural network
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Indicators
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# IQ Option API (non-official)
from iqoptionapi.stable_api import IQ_Option

# Web API
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DelowyssProV2")

# -----------------------
# Config (env vars)
# -----------------------
IQ_EMAIL = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "")
IQ_MODE = os.getenv("IQ_MODE", "PRACTICE")  # PRACTICE or REAL
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Asset list to analyze
ASSETS = ["EURUSD-OTC", "EURUSD"]

# Train schedule (minutes)
RETRAIN_INTERVAL_MIN = 30

# -----------------------
# Helper: model filenames
# -----------------------
XGB_FILE = os.path.join(MODEL_DIR, "xgboost_model.joblib")
RF_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
NN_FILE = os.path.join(MODEL_DIR, "nn_model.h5")

# -----------------------
# Trading / Data class
# -----------------------
class IQConnector:
    def __init__(self, email: str, password: str, mode: str = "PRACTICE"):
        self.email = email
        self.password = password
        self.mode = mode
        self.api = None
        self.connected = False
        self.available_assets = {}
        self._connect()

    def _connect(self):
        try:
            logger.info("Conectando a IQ Option...")
            self.api = IQ_Option(self.email, self.password)
            connected = self.api.connect()
            if not connected:
                logger.error("No se pudo conectar a IQ Option (credenciales o red).")
                self.connected = False
                return
            self.connected = True
            # set mode
            if self.mode.upper() == "REAL":
                self.api.change_balance("REAL")
            else:
                self.api.change_balance("PRACTICE")
            self.available_assets = self.api.get_all_open_time()
            logger.info("Conectado a IQ Option. Balance cargado.")
        except Exception as e:
            logger.exception("Error conectando a IQ Option: %s", e)
            self.connected = False

    def get_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        """
        timeframe_min: 1, 5, etc (minutes)
        count: number of candles
        """
        try:
            if not self.connected:
                logger.error("No conectado a IQ Option")
                return None
            # iqoption get_candles uses seconds timeframe
            candles = self.api.get_candles(asset, timeframe_min * 60, count, time.time())
            # convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(c['from']),
                'open': c['open'],
                'high': c['max'],
                'low': c['min'],
                'close': c['close'],
                'volume': c['volume']
            } for c in candles])
            return df
        except Exception as e:
            logger.exception("Error obteniendo velas: %s", e)
            return None

# -----------------------
# Feature engineering
# -----------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # EMA
    df['ema_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=15).ema_indicator()

    # MACD
    macd = MACD(close=df['close'], window_fast=10, window_slow=20, window_sign=7)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=12).rsi()

    # Stochastic
    st = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=12)
    df['stoch_k'] = st.stoch()
    df['stoch_d'] = st.stoch_signal()

    # ADX
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()

    # Bollinger
    bb = BollingerBands(close=df['close'], window=18, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # ATR
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=12).average_true_range()

    # Candle features
    df['body'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['open'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['open'] + 1e-8)

    # Lags
    for lag in [1,2,3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        df[f'adx_lag_{lag}'] = df['adx'].shift(lag)

    # target: next candle up/down
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df.dropna()

# -----------------------
# Models and training
# -----------------------
class ModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.nn_model: Optional[tf.keras.Model] = None
        self.model_lock = threading.Lock()
        self._load_models_if_exist()

    def _load_models_if_exist(self):
        try:
            if os.path.exists(SCALER_FILE):
                self.scaler = joblib.load(SCALER_FILE)
                logger.info("Scaler cargado.")
            if os.path.exists(XGB_FILE):
                self.xgb_model = joblib.load(XGB_FILE)
                logger.info("XGBoost cargado.")
            if os.path.exists(RF_FILE):
                self.rf_model = joblib.load(RF_FILE)
                logger.info("RandomForest cargado.")
            if os.path.exists(NN_FILE):
                self.nn_model = tf.keras.models.load_model(NN_FILE)
                logger.info("NN cargada.")
        except Exception as e:
            logger.exception("Error cargando modelos: %s", e)

    def save_models(self):
        try:
            with self.model_lock:
                if self.xgb_model is not None:
                    joblib.dump(self.xgb_model, XGB_FILE)
                if self.rf_model is not None:
                    joblib.dump(self.rf_model, RF_FILE)
                joblib.dump(self.scaler, SCALER_FILE)
                if self.nn_model is not None:
                    self.nn_model.save(NN_FILE, include_optimizer=False)
            logger.info("Modelos guardados correctamente.")
        except Exception as e:
            logger.exception("Error guardando modelos: %s", e)

    def build_and_train(self, df_features: pd.DataFrame) -> bool:
        """Entrena XGB, RF y NN en ensemble. Retorna True si entrenó correctamente."""
        try:
            with self.model_lock:
                X = df_features.drop(columns=['target'])
                y = df_features['target'].astype(int)

                # scale
                X_scaled = self.scaler.fit_transform(X)

                # XGBoost
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=120, max_depth=6, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                    eval_metric='logloss', random_state=42, n_jobs=1
                )
                xgb_clf.fit(X_scaled, y)
                self.xgb_model = xgb_clf

                # RandomForest
                rf_clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=1)
                rf_clf.fit(X_scaled, y)
                self.rf_model = rf_clf

                # Neural Network (simple dense)
                input_dim = X_scaled.shape[1]
                nn = Sequential([
                    Dense(128, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.1),
                    Dense(1, activation='sigmoid')
                ])
                nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
                nn.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.15, callbacks=[es], verbose=0)
                self.nn_model = nn

                logger.info("Entrenamiento completado: XGB, RF y NN.")
                self.save_models()
            return True
        except Exception as e:
            logger.exception("Error en entrenamiento: %s", e)
            return False

    def predict(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Predicción por ensemble para la última fila"""
        try:
            with self.model_lock:
                if self.xgb_model is None or self.rf_model is None or self.nn_model is None:
                    return {"error": "Modelos no están listos."}
                X = df_features.drop(columns=['target'])
                X_scaled = self.scaler.transform(X)
                last = X_scaled[-1:].copy()

                xgb_pred = self.xgb_model.predict_proba(last)[0][1]
                rf_pred = self.rf_model.predict_proba(last)[0][1]
                nn_pred = float(self.nn_model.predict(last, verbose=0).ravel()[0])

                # Weighted ensemble
                weights = np.array([0.4, 0.3, 0.3])
                probs = np.array([xgb_pred, rf_pred, nn_pred])
                ensemble_prob = float(np.dot(weights, probs))

                direction = "ALCISTA" if ensemble_prob >= 0.5 else "BAJISTA"
                confidence = ensemble_prob

                return {
                    "direction": direction,
                    "confidence": float(confidence),
                    "xgb": float(xgb_pred),
                    "rf": float(rf_pred),
                    "nn": float(nn_pred),
                    "ensemble": float(ensemble_prob)
                }
        except Exception as e:
            logger.exception("Error en predicción: %s", e)
            return {"error": str(e)}

# -----------------------
# Orchestrator that ties IQ and Models
# -----------------------
class DelowyssAssistant:
    def __init__(self, iq_connector: IQConnector, model_manager: ModelManager):
        self.iq = iq_connector
        self.models = model_manager
        self.analysis_lock = threading.Lock()
        self.last_analysis = None
        self.enabled = True  # toggle for enabling analysis/prediction

    def toggle_enabled(self):
        with self.analysis_lock:
            self.enabled = not self.enabled
            logger.info("Toggled enabled -> %s", self.enabled)
            return self.enabled

    def run_analysis_once(self, asset: str = "EURUSD-OTC", candles: int = 200) -> Dict[str, Any]:
        """Obtener datos, procesar, entrenar si hace falta y predecir siguiente vela."""
        if not self.enabled:
            return {"error": "ANALYSIS_DISABLED"}

        if not self.iq.connected:
            return {"error": "IQ_NOT_CONNECTED"}

        with self.analysis_lock:
            df = self.iq.get_candles(asset, timeframe_min=1, count=candles)
            if df is None or df.empty:
                return {"error": "NO_DATA"}

            df_ind = compute_indicators(df)
            # prepare features set
            feature_cols = [c for c in df_ind.columns if c not in ['timestamp', 'target']]
            features = df_ind[feature_cols + ['target']].dropna()
            if len(features) < 60:
                return {"error": "NOT_ENOUGH_DATA"}

            # Train incremental: use last N rows for training; here we retrain fully (you can enhance incremental)
            trained = self.models.build_and_train(features)
            prediction = self.models.predict(features)

            self.last_analysis = {
                "asset": asset,
                "timestamp": datetime.now(),
                "prediction": prediction
            }
            logger.info("Analysis completed for %s: %s", asset, prediction)
            return self.last_analysis

# -----------------------
# App + scheduler setup
# -----------------------
iq_conn = IQConnector(IQ_EMAIL, IQ_PASSWORD, mode=IQ_MODE)
model_mgr = ModelManager()
assistant = DelowyssAssistant(iq_conn, model_mgr)

app = FastAPI(title="Delowyss Assistant", version="2.0")

# Simple web UI
HTML_INDEX = """
<!doctype html>
<html>
  <head>
    <title>Delowyss Assistant</title>
  </head>
  <body>
    <h1>Delowyss Assistant - Analizador</h1>
    <p>Estado del assistente: <strong id="status">Cargando...</strong></p>
    <button onclick="toggle()">Activar/Desactivar Análisis</button>
    <button onclick="run()">Analizar y predecir siguiente vela (EURUSD-OTC)</button>
    <pre id="output"></pre>
    <script>
      async function status(){
        const r = await fetch('/status');
        const j = await r.json();
        document.getElementById('status').innerText = j.enabled ? "ACTIVO" : "PAUSADO";
      }
      async function toggle(){
        const r = await fetch('/toggle', {method:'POST'});
        const j = await r.json();
        document.getElementById('status').innerText = j.enabled ? "ACTIVO" : "PAUSADO";
      }
      async function run(){
        document.getElementById('output').innerText = "Analizando...";
        const r = await fetch('/analyze', {method:'POST'});
        const j = await r.json();
        document.getElementById('output').innerText = JSON.stringify(j, null, 2);
      }
      status();
      setInterval(status, 5000);
    </script>
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_INDEX

@app.get("/status")
def status():
    return {"enabled": assistant.enabled, "iq_connected": iq_conn.connected, "last_analysis": assistant.last_analysis}

@app.post("/toggle")
def toggle():
    enabled = assistant.toggle_enabled()
    return {"enabled": enabled}

@app.post("/analyze")
def analyze(background_tasks: BackgroundTasks):
    """
    Ejecuta análisis en background y retorna resultado cuando termine.
    """
    # run in background to avoid blocking http thread
    def job():
        res = assistant.run_analysis_once(asset="EURUSD-OTC", candles=300)
        logger.info("Manual analysis finished: %s", res)
    background_tasks.add_task(job)
    return {"status": "analysis_started"}

# Automatic retraining schedule (cada 30 min)
scheduler = BackgroundScheduler()
def scheduled_retrain():
    if not assistant.enabled:
        logger.info("Skips scheduled retrain because assistant disabled.")
        return
    logger.info("Scheduled retrain started.")
    try:
        # we'll retrain on both assets sequentially
        for asset in ASSETS:
            res = assistant.run_analysis_once(asset=asset, candles=300)
            logger.info("Scheduled asset %s result: %s", asset, res)
    except Exception as e:
        logger.exception("Error in scheduled retrain: %s", e)

scheduler.add_job(scheduled_retrain, 'interval', minutes=RETRAIN_INTERVAL_MIN, next_run_time=datetime.now())
scheduler.start()
logger.info("Scheduler started: retrain every %d minutes", RETRAIN_INTERVAL_MIN)

# -----------------------
# Run the app
# -----------------------
if __name__ == "__main__":
    # Run uvicorn when launching directly
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
