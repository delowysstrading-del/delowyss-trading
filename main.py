# main.py
import os
import time
import threading
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from iqoptionapi.api import IQ_Option
from sklearn.ensemble import RandomForestClassifier
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Delowyss")

app = FastAPI(title="Delowyss Trading Professional")

# Permitir CORS (para extensión Chrome o front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# IQ CONNECTOR REALTIME
# -------------------------------------------
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
            logger.info("🔗 Conectando a IQ Option...")
            if not self.email or not self.password:
                logger.error("❌ Email o password no configurados")
                self.connected = False
                return
            self.api = IQ_Option(self.email, self.password)
            self.api.connect()
            if not self.api.check_connect():
                logger.error("❌ No se pudo conectar a IQ Option")
                self.connected = False
                return

            # Cambiar modo de cuenta
            if self.mode.upper() == "REAL":
                self.api.change_balance("REAL")
                logger.info("💰 Modo REAL activado")
            else:
                self.api.change_balance("PRACTICE")
                logger.info("🎯 Modo DEMO activado")

            self.connected = True
            self.available_assets = self.api.get_all_open_time()
            logger.info("✅ Conectado correctamente a IQ Option")

        except Exception as e:
            logger.error(f"❌ Error conectando: {e}")
            self.connected = False

    def reconnect(self):
        with self.reconnect_lock:
            if not self.connected:
                logger.warning("🔁 Intentando reconexión...")
                self._connect()

    def start_realtime_stream(self, asset: str = "EURUSD-OTC", timeframe_min: int = 1):
        if not self.connected:
            logger.error("❌ No conectado - no se puede iniciar stream")
            return
        try:
            period = timeframe_min * 60
            logger.info(f"📡 Iniciando stream en tiempo real para {asset} ({timeframe_min}m)")
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
            logger.error(f"❌ No se pudo iniciar stream: {e}")

    def get_realtime_candles(self, asset: str):
        return self.realtime_data.get(asset, None)

    def get_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        try:
            if not self.connected:
                self.reconnect()
            logger.info(f"📈 Obteniendo {count} velas de {asset}")
            candles = self.api.get_candles(asset, timeframe_min * 60, count, time.time())
            df = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.fromtimestamp(c["from"]),
                        "open": c["open"],
                        "high": c["max"],
                        "low": c["min"],
                        "close": c["close"],
                        "volume": c["volume"],
                    }
                    for c in candles
                ]
            )
            return df
        except Exception as e:
            logger.error(f"❌ Error obteniendo velas: {e}")
            return None

# -------------------------------------------
# MODELO IA Y FUNCIONES DE ANÁLISIS
# -------------------------------------------
class ModelManager:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.is_trained = False

    def train(self, df):
        df = df.dropna()
        df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
        X = df[["rsi", "ema", "macd", "bb_upper", "bb_lower"]]
        y = df["target"]
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("🧠 Modelo entrenado correctamente")

    def predict(self, df):
        if not self.is_trained:
            return {"status": "error", "message": "Modelo no entrenado"}
        last_row = df[["rsi", "ema", "macd", "bb_upper", "bb_lower"]].iloc[-1].values.reshape(1, -1)
        pred = self.model.predict(last_row)[0]
        return {"signal": "CALL" if pred == 1 else "PUT"}

def compute_indicators(df):
    try:
        df["ema"] = df["close"].ewm(span=14, adjust=False).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, min_periods=14).mean()
        avg_loss = loss.ewm(span=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        df["bb_middle"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"❌ Error calculando indicadores: {e}")
        return pd.DataFrame()

# -------------------------------------------
# INICIALIZACIÓN GLOBAL
# -------------------------------------------
iq_conn = IQConnector(
    email=os.getenv("IQ_EMAIL", ""),
    password=os.getenv("IQ_PASSWORD", "")
)
model_mgr = ModelManager()

if iq_conn.connected:
    iq_conn.start_realtime_stream("EURUSD-OTC", timeframe_min=1)

scheduler = BackgroundScheduler()
scheduler.start()

# -------------------------------------------
# ENDPOINTS FASTAPI
# -------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "msg": "🚀 Delowyss Trading Professional activo"}

@app.post("/api/analyze")
def analyze(asset: str = "EURUSD-OTC"):
    df = iq_conn.get_realtime_candles(asset)
    if df is None or df.empty:
        return {"error": "Sin datos en tiempo real"}
    df_ind = compute_indicators(df)
    if df_ind.empty:
        return {"error": "No se pudieron calcular indicadores"}
    prediction = model_mgr.predict(df_ind)
    return {"asset": asset, "prediction": prediction, "timestamp": datetime.now().isoformat()}

@app.get("/api/train")
def train_model(asset: str = "EURUSD-OTC"):
    df = iq_conn.get_candles(asset)
    if df is None or df.empty:
        return {"error": "No se pudo obtener datos históricos"}
    df_ind = compute_indicators(df)
    model_mgr.train(df_ind)
    return {"status": "success", "message": "Modelo entrenado correctamente"}

# -------------------------------------------
# EJECUCIÓN LOCAL / RENDER
# -------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
