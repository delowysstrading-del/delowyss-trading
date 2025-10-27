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
from sklearn.model_selection import train_test_split
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

# IQ Option API
from iqoptionapi.stable_api import IQ_Option

# Web API - CORREGIDO: Usando FastAPI, no Flask
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------
# Configuration for Render
# -----------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DelowyssProRender")

# Environment variables - VERIFICAR QUE ESTÉN SETEADAS EN RENDER
IQ_EMAIL = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "") 
IQ_MODE = os.getenv("IQ_MODE", "PRACTICE")
PORT = int(os.getenv("PORT", 10000))

# Validar variables críticas
if not IQ_EMAIL or not IQ_PASSWORD:
    logger.error("❌ CRÍTICO: Variables IQ_EMAIL e IQ_PASSWORD no configuradas")
    logger.error("💡 Configúralas en Render Dashboard -> Environment Variables")
    # No salir, permitir que el servidor inicie pero mostrar advertencia

# Create models directory
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Assets to analyze
ASSETS = ["EURUSD-OTC", "EURUSD"]
RETRAIN_INTERVAL_MIN = 30

# Model files
XGB_FILE = os.path.join(MODEL_DIR, "xgboost_model.joblib")
RF_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
NN_FILE = os.path.join(MODEL_DIR, "nn_model.h5")

# -----------------------
# IQ Connector (OPTIMIZADO)
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
            logger.info("🔗 [IQ] Conectando a IQ Option...")
            
            if not self.email or not self.password:
                logger.error("❌ [IQ] Email o password no configurados")
                self.connected = False
                return
                
            self.api = IQ_Option(self.email, self.password)
            
            # Conexión con timeout extendido
            connected = self.api.connect(timeout=20)
            
            if not connected:
                logger.error("❌ [IQ] Falló la conexión a IQ Option")
                self.connected = False
                return
                
            self.connected = True
            
            # Set account mode
            if self.mode.upper() == "REAL":
                self.api.change_balance("REAL")
                logger.info("💰 [IQ] Modo: CUENTA REAL")
            else:
                self.api.change_balance("PRACTICE")
                logger.info("🎯 [IQ] Modo: CUENTA DEMO")
                
            # Get available assets
            try:
                self.available_assets = self.api.get_all_open_time()
                logger.info(f"📊 [IQ] Activos disponibles: {len(self.available_assets)}")
            except Exception as e:
                logger.warning(f"⚠️ [IQ] No se pudieron obtener activos: {e}")
                self.available_assets = {}
                
            logger.info("✅ [IQ] Conexión exitosa a IQ Option")
            
        except Exception as e:
            logger.error(f"❌ [IQ] Error de conexión: {e}")
            self.connected = False

    def get_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        try:
            if not self.connected:
                logger.error("❌ [IQ] No conectado para obtener velas")
                return None
                
            logger.info(f"📈 [IQ] Obteniendo velas de {asset}...")
            candles = self.api.get_candles(asset, timeframe_min * 60, count, time.time())
            
            if not candles:
                logger.warning(f"⚠️ [IQ] No se obtuvieron velas para {asset}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(c['from']),
                'open': c['open'],
                'high': c['max'],
                'low': c['min'],
                'close': c['close'],
                'volume': c['volume']
            } for c in candles])
            
            logger.info(f"✅ [IQ] Velas obtenidas: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"❌ [IQ] Error obteniendo velas: {e}")
            return None

# -----------------------
# Feature Engineering 
# -----------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular indicadores técnicos"""
    try:
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

        # ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx.adx()

        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=18, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        # ATR
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=12).average_true_range()

        # Candle features
        df['body'] = (df['close'] - df['open']) / (df['open'] + 1e-8)

        # Target: next candle direction
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        result = df.dropna()
        logger.info(f"📊 Indicadores calculados: {len(result)} muestras")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error calculando indicadores: {e}")
        return pd.DataFrame()

# -----------------------
# Model Manager
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
                logger.info("📊 Scaler cargado")
            if os.path.exists(XGB_FILE):
                self.xgb_model = joblib.load(XGB_FILE)
                logger.info("🌳 XGBoost cargado")
            if os.path.exists(RF_FILE):
                self.rf_model = joblib.load(RF_FILE)
                logger.info("🌲 RandomForest cargado")
            if os.path.exists(NN_FILE):
                self.nn_model = tf.keras.models.load_model(NN_FILE)
                logger.info("🧠 Red Neuronal cargada")
        except Exception as e:
            logger.warning(f"⚠️ Error cargando modelos: {e}")

    def build_and_train(self, df_features: pd.DataFrame) -> bool:
        try:
            with self.model_lock:
                if len(df_features) < 50:
                    logger.error("❌ Datos insuficientes para entrenar")
                    return False

                X = df_features.drop(columns=['target'])
                y = df_features['target'].astype(int)

                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # 1. XGBoost
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric='logloss'
                )
                xgb_clf.fit(X_train, y_train)
                self.xgb_model = xgb_clf

                # 2. RandomForest
                rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf_clf.fit(X_train, y_train)
                self.rf_model = rf_clf

                # 3. Neural Network
                input_dim = X_scaled.shape[1]
                nn = Sequential([
                    Dense(64, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                nn.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.15, verbose=0)
                self.nn_model = nn

                # Save models
                self.save_models()
                logger.info("✅ Modelos entrenados y guardados")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            return False

    def save_models(self):
        try:
            with self.model_lock:
                if self.xgb_model is not None:
                    joblib.dump(self.xgb_model, XGB_FILE)
                if self.rf_model is not None:
                    joblib.dump(self.rf_model, RF_FILE)
                joblib.dump(self.scaler, SCALER_FILE)
                if self.nn_model is not None:
                    self.nn_model.save(NN_FILE)
            logger.info("💾 Modelos guardados")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")

    def predict(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        try:
            with self.model_lock:
                if self.xgb_model is None or self.rf_model is None or self.nn_model is None:
                    return {"error": "Modelos no entrenados"}

                X = df_features.drop(columns=['target'])
                X_scaled = self.scaler.transform(X)
                last = X_scaled[-1:].copy()

                # Individual predictions
                xgb_pred = self.xgb_model.predict_proba(last)[0][1]
                rf_pred = self.rf_model.predict_proba(last)[0][1]
                nn_pred = float(self.nn_model.predict(last, verbose=0).ravel()[0])

                # Ensemble prediction
                ensemble_prob = (xgb_pred * 0.4 + rf_pred * 0.3 + nn_pred * 0.3)
                direction = "ALCISTA" if ensemble_prob >= 0.5 else "BAJISTA"
                confidence = ensemble_prob if ensemble_prob >= 0.5 else 1 - ensemble_prob

                return {
                    "direction": direction,
                    "confidence": float(confidence),
                    "probability": float(ensemble_prob),
                    "xgb": float(xgb_pred),
                    "rf": float(rf_pred),
                    "nn": float(nn_pred),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {"error": str(e)}

# -----------------------
# Trading Assistant
# -----------------------
class DelowyssAssistant:
    def __init__(self, iq_connector: IQConnector, model_manager: ModelManager):
        self.iq = iq_connector
        self.models = model_manager
        self.analysis_lock = threading.Lock()
        self.last_analysis = None
        self.enabled = True

    def run_analysis_once(self, asset: str = "EURUSD-OTC", candles: int = 200) -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "Sistema desactivado"}

        if not self.iq.connected:
            return {"error": "IQ Option no conectado"}

        with self.analysis_lock:
            logger.info(f"🔍 Analizando {asset} con {candles} velas...")
            
            # Obtener datos REALES de IQ Option
            df = self.iq.get_candles(asset, timeframe_min=1, count=candles)
            
            if df is None or df.empty:
                return {"error": "No se pudieron obtener datos de IQ Option"}

            # Compute indicators
            df_ind = compute_indicators(df)
            if df_ind.empty:
                return {"error": "Error calculando indicadores"}

            feature_cols = [c for c in df_ind.columns if c not in ['timestamp', 'target']]
            features = df_ind[feature_cols + ['target']].dropna()
            
            if len(features) < 50:
                return {"error": "Datos insuficientes para análisis"}

            # Train models
            trained = self.models.build_and_train(features)
            if not trained:
                return {"error": "Error en entrenamiento de modelos"}

            # Make prediction
            prediction = self.models.predict(features)

            # Store results
            self.last_analysis = {
                "asset": asset,
                "timestamp": datetime.now(),
                "prediction": prediction,
                "data_points": len(features),
                "current_price": df_ind['close'].iloc[-1]
            }

            logger.info(f"📊 Análisis completado: {prediction.get('direction', 'N/A')}")
            return self.last_analysis

# -----------------------
# Inicialización del Sistema
# -----------------------
logger.info("🚀 INICIANDO DELOWYSS TRADING PROFESSIONAL...")

# Inicializar componentes
iq_conn = IQConnector(IQ_EMAIL, IQ_PASSWORD, mode=IQ_MODE)
model_mgr = ModelManager()
assistant = DelowyssAssistant(iq_conn, model_mgr)

# -----------------------
# FastAPI Application
# -----------------------
app = FastAPI(title="Delowyss Pro", version="2.0")

# Simple Web Interface
HTML_INDEX = """
<!DOCTYPE html>
<html>
<head>
    <title>Delowyss Pro - Trading con IA</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Delowyss Pro Trading AI</h1>
            <p>Sistema avanzado de trading con inteligencia artificial</p>
        </div>
        
        <div id="status-panel">
            <h3>Estado del Sistema</h3>
            <div id="iq-status" class="status">Cargando...</div>
            <div id="models-status" class="status">Cargando...</div>
        </div>
        
        <div>
            <h3>Acciones</h3>
            <button class="btn btn-primary" onclick="runAnalysis()">Ejecutar Análisis</button>
            <button class="btn btn-success" onclick="checkStatus()">Verificar Estado</button>
        </div>
        
        <div id="results" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;"></div>
    </div>

    <script>
        async function checkStatus() {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            document.getElementById('iq-status').textContent = 
                `IQ Option: ${data.iq_connected ? '✅ Conectado' : '❌ Desconectado'}`;
            document.getElementById('iq-status').className = 
                `status ${data.iq_connected ? 'connected' : 'disconnected'}`;
                
            document.getElementById('models-status').textContent = 
                `Modelos IA: ${data.models_ready ? '✅ Cargados' : '⚠️ No cargados'}`;
        }

        async function runAnalysis() {
            document.getElementById('results').textContent = 'Ejecutando análisis...';
            const response = await fetch('/api/analyze', { method: 'POST' });
            const data = await response.json();
            document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            await checkStatus();
        }

        // Inicializar
        checkStatus();
    </script>
</body>
</html>
"""

@app.get("/")
def read_root():
    return HTMLResponse(HTML_INDEX)

@app.get("/api/status")
def get_status():
    return {
        "enabled": assistant.enabled,
        "iq_connected": iq_conn.connected,
        "models_ready": all([
            model_mgr.xgb_model is not None,
            model_mgr.rf_model is not None,
            model_mgr.nn_model is not None
        ]),
        "last_analysis": assistant.last_analysis
    }

@app.post("/api/analyze")
def analyze_asset():
    result = assistant.run_analysis_once(asset="EURUSD-OTC", candles=200)
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# -----------------------
# Inicialización Final
# -----------------------
def initialize_system():
    """Inicialización completa del sistema"""
    logger.info("🌐 [Delowyss] Entorno web detectado - Modo servidor activo")
    logger.info("🤖 [Delowyss] Iniciando plataforma de trading...")
    
    # Verificar conexión IQ Option
    if iq_conn.connected:
        logger.info("✅ [Delowyss] Conexión IQ Option establecida")
        
        # Ejecutar análisis inicial
        try:
            logger.info("🔍 [Delowyss] Ejecutando análisis automático inicial...")
            result = assistant.run_analysis_once(asset="EURUSD-OTC", candles=100)
            if 'error' not in result:
                logger.info("✅ [Delowyss] Análisis inicial completado")
            else:
                logger.warning(f"⚠️ [Delowyss] Análisis inicial falló: {result['error']}")
        except Exception as e:
            logger.error(f"❌ [Delowyss] Error en análisis inicial: {e}")
    else:
        logger.warning("⚠️ [Delowyss] Sistema operando sin conexión IQ Option")
    
    logger.info("🤖 [Delowyss System] Sistema profesional inicializado")
    logger.info("✅ [Delowyss System] Sistema cargado - Listo para análisis profesional")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Inicializar sistema
    initialize_system()
    
    logger.info(f"🌐 [Delowyss Server] Iniciando servidor FastAPI en puerto {PORT}...")
    
    # Iniciar servidor FastAPI
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )
