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

# Web API
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------
# Configuration for Render
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DelowyssProRender")

# Environment variables
IQ_EMAIL = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "") 
IQ_MODE = os.getenv("IQ_MODE", "PRACTICE")
PORT = int(os.getenv("PORT", 10000))

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
# IQ Connector (Optimized)
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
            logger.info("🔗 Conectando a IQ Option...")
            self.api = IQ_Option(self.email, self.password)
            connected = self.api.connect()
            
            if not connected:
                logger.error("❌ No se pudo conectar a IQ Option")
                self.connected = False
                return
                
            self.connected = True
            
            # Set account mode
            if self.mode.upper() == "REAL":
                self.api.change_balance("REAL")
            else:
                self.api.change_balance("PRACTICE")
                
            self.available_assets = self.api.get_all_open_time()
            logger.info("✅ Conectado exitosamente a IQ Option")
            
        except Exception as e:
            logger.error(f"❌ Error de conexión: {e}")
            self.connected = False

    def get_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        try:
            if not self.connected:
                logger.error("No conectado a IQ Option")
                return None
                
            # Get candles using IQ Option API
            candles = self.api.get_candles(asset, timeframe_min * 60, count, time.time())
            
            if not candles:
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo velas: {e}")
            return None

    def get_realtime_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        """Get real-time candles for better accuracy"""
        try:
            if not self.connected:
                return None
                
            # Start real-time stream
            self.api.start_candles_stream(asset, timeframe_min * 60, count)
            time.sleep(2)  # Wait for data
            
            candles_data = self.api.get_realtime_candles(asset, timeframe_min * 60)
            
            if not candles_data:
                return None

            # Convert to DataFrame
            candles = []
            for timestamp in sorted(candles_data.keys()):
                candle = candles_data[timestamp]
                candles.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'open': candle['open'],
                    'high': candle['max'],
                    'low': candle['min'],
                    'close': candle['close'],
                    'volume': candle['volume']
                })
            
            return pd.DataFrame(candles)
            
        except Exception as e:
            logger.error(f"Error en velas tiempo real: {e}")
            return None
        finally:
            # Clean up stream
            try:
                self.api.stop_candles_stream(asset, timeframe_min * 60)
            except:
                pass

# -----------------------
# Feature Engineering
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

    # Target: next candle direction
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df.dropna()

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
            logger.info("💾 Modelos guardados")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")

    def build_and_train(self, df_features: pd.DataFrame) -> bool:
        try:
            with self.model_lock:
                X = df_features.drop(columns=['target'])
                y = df_features['target'].astype(int)

                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # 1. XGBoost
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=120, max_depth=6, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                    eval_metric='logloss', random_state=42
                )
                xgb_clf.fit(X_train, y_train)
                self.xgb_model = xgb_clf

                # 2. RandomForest
                rf_clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
                rf_clf.fit(X_train, y_train)
                self.rf_model = rf_clf

                # 3. Neural Network
                input_dim = X_scaled.shape[1]
                nn = Sequential([
                    Dense(128, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.1),
                    Dense(1, activation='sigmoid')
                ])
                nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
                nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.15, callbacks=[es], verbose=0)
                self.nn_model = nn

                logger.info("✅ Entrenamiento completado: XGB + RF + NN")
                self.save_models()
                return True
                
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            return False

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
                weights = np.array([0.4, 0.3, 0.3])
                probs = np.array([xgb_pred, rf_pred, nn_pred])
                ensemble_prob = float(np.dot(weights, probs))

                # Determine direction and confidence
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

    def toggle_enabled(self):
        with self.analysis_lock:
            self.enabled = not self.enabled
            status = "activado" if self.enabled else "desactivado"
            logger.info(f"🔧 Sistema {status}")
            return self.enabled

    def run_analysis_once(self, asset: str = "EURUSD-OTC", candles: int = 200) -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "Sistema desactivado"}

        if not self.iq.connected:
            return {"error": "IQ Option no conectado"}

        with self.analysis_lock:
            # Try real-time data first, fallback to historical
            df = self.iq.get_realtime_candles(asset, timeframe_min=1, count=candles)
            if df is None or df.empty:
                df = self.iq.get_candles(asset, timeframe_min=1, count=candles)
                
            if df is None or df.empty:
                return {"error": "No se pudieron obtener datos"}

            # Compute indicators
            df_ind = compute_indicators(df)
            feature_cols = [c for c in df_ind.columns if c not in ['timestamp', 'target']]
            features = df_ind[feature_cols + ['target']].dropna()
            
            if len(features) < 60:
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
# FastAPI Application
# -----------------------
iq_conn = IQConnector(IQ_EMAIL, IQ_PASSWORD, mode=IQ_MODE)
model_mgr = ModelManager()
assistant = DelowyssAssistant(iq_conn, model_mgr)

app = FastAPI(title="Delowyss Pro", version="2.0")

# Enhanced Web Interface
HTML_INDEX = """
<!DOCTYPE html>
<html>
<head>
    <title>Delowyss Pro - Trading con IA</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .status { 
            padding: 10px; 
            border-radius: 5px; 
            text-align: center; 
            font-weight: bold;
            margin: 10px 0;
        }
        .status-active { background: #d5f4e6; color: #27ae60; border: 1px solid #27ae60; }
        .status-inactive { background: #fadbd8; color: #e74c3c; border: 1px solid #e74c3c; }
        .signal { 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            font-size: 1.2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .signal-buy { background: #d5f4e6; color: #27ae60; border: 2px solid #27ae60; }
        .signal-sell { background: #fadbd8; color: #e74c3c; border: 2px solid #e74c3c; }
        .signal-neutral { background: #fcf3cf; color: #f39c12; border: 2px solid #f39c12; }
        .analysis-item { 
            display: flex; 
            justify-content: space-between; 
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Delowyss Pro</h1>
            <p>Sistema de Trading con Inteligencia Artificial</p>
        </div>
        
        <div class="grid">
            <!-- System Control -->
            <div class="card">
                <h3>⚙️ Control del Sistema</h3>
                <div id="system-status" class="status status-active">ACTIVO</div>
                <div class="analysis-item">
                    <span>IQ Option:</span>
                    <span id="iq-status">Conectado</span>
                </div>
                <div class="analysis-item">
                    <span>Modelos IA:</span>
                    <span id="models-status">Cargando...</span>
                </div>
                <button class="btn btn-warning" onclick="toggleSystem()">Activar/Desactivar</button>
                <button class="btn btn-primary" onclick="runAnalysis()">Analizar y Predecir</button>
            </div>

            <!-- Current Prediction -->
            <div class="card">
                <h3>🎯 Predicción Actual</h3>
                <div id="current-signal" class="signal signal-neutral">ESPERANDO ANÁLISIS</div>
                <div id="prediction-details">
                    <div class="analysis-item"><span>Dirección:</span><span id="direction">-</span></div>
                    <div class="analysis-item"><span>Confianza:</span><span id="confidence">-</span></div>
                    <div class="analysis-item"><span>Probabilidad:</span><span id="probability">-</span></div>
                    <div class="analysis-item"><span>XGBoost:</span><span id="xgb-score">-</span></div>
                    <div class="analysis-item"><span>Random Forest:</span><span id="rf-score">-</span></div>
                    <div class="analysis-item"><span>Red Neuronal:</span><span id="nn-score">-</span></div>
                </div>
            </div>

            <!-- Asset Selection -->
            <div class="card">
                <h3>📊 Configuración</h3>
                <div class="analysis-item">
                    <span>Activo:</span>
                    <select id="asset-select" style="padding: 5px; border-radius: 5px;">
                        <option value="EURUSD-OTC">EURUSD-OTC</option>
                        <option value="EURUSD">EURUSD</option>
                    </select>
                </div>
                <div class="analysis-item">
                    <span>Velas:</span>
                    <input type="number" id="candles-count" value="200" min="50" max="500" style="padding: 5px; border-radius: 5px; width: 80px;">
                </div>
                <button class="btn btn-success" onclick="trainModels()">Entrenar IA</button>
            </div>

            <!-- Last Analysis -->
            <div class="card">
                <h3>📈 Último Análisis</h3>
                <div id="last-analysis">
                    <div class="analysis-item"><span>Activo:</span><span id="last-asset">-</span></div>
                    <div class="analysis-item"><span>Hora:</span><span id="last-time">-</span></div>
                    <div class="analysis-item"><span>Precio:</span><span id="last-price">-</span></div>
                    <div class="analysis-item"><span>Datos:</span><span id="last-data">-</span></div>
                </div>
            </div>
        </div>

        <!-- Results Display -->
        <div class="card" style="margin: 20px;">
            <h3>📋 Resultados del Análisis</h3>
            <pre id="results" style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow: auto; max-height: 400px;"></pre>
        </div>
    </div>

    <script>
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update system status
                document.getElementById('system-status').textContent = data.enabled ? "ACTIVO" : "INACTIVO";
                document.getElementById('system-status').className = data.enabled ? 'status status-active' : 'status status-inactive';
                
                // Update IQ status
                document.getElementById('iq-status').textContent = data.iq_connected ? "Conectado" : "Desconectado";
                
                // Update models status
                const modelsReady = data.models_ready || false;
                document.getElementById('models-status').textContent = modelsReady ? "Listos" : "Entrenando...";
                
                // Update last analysis
                if (data.last_analysis) {
                    const la = data.last_analysis;
                    document.getElementById('last-asset').textContent = la.asset || '-';
                    document.getElementById('last-time').textContent = new Date(la.timestamp).toLocaleString();
                    document.getElementById('last-price').textContent = la.current_price ? la.current_price.toFixed(4) : '-';
                    document.getElementById('last-data').textContent = la.data_points || '-';
                    
                    // Update prediction if available
                    if (la.prediction && !la.prediction.error) {
                        const pred = la.prediction;
                        const signalElement = document.getElementById('current-signal');
                        signalElement.textContent = `${pred.direction} (${(pred.confidence * 100).toFixed(1)}%)`;
                        
                        if (pred.direction === "ALCISTA") {
                            signalElement.className = 'signal signal-buy';
                        } else {
                            signalElement.className = 'signal signal-sell';
                        }
                        
                        document.getElementById('direction').textContent = pred.direction;
                        document.getElementById('confidence').textContent = (pred.confidence * 100).toFixed(1) + '%';
                        document.getElementById('probability').textContent = (pred.probability * 100).toFixed(1) + '%';
                        document.getElementById('xgb-score').textContent = (pred.xgb * 100).toFixed(1) + '%';
                        document.getElementById('rf-score').textContent = (pred.rf * 100).toFixed(1) + '%';
                        document.getElementById('nn-score').textContent = (pred.nn * 100).toFixed(1) + '%';
                    }
                }
                
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }

        async function toggleSystem() {
            const response = await fetch('/api/toggle', { method: 'POST' });
            await updateStatus();
        }

        async function runAnalysis() {
            const asset = document.getElementById('asset-select').value;
            const candles = document.getElementById('candles-count').value;
            
            document.getElementById('current-signal').textContent = 'ANALIZANDO...';
            document.getElementById('results').textContent = 'Ejecutando análisis...';
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asset: asset, candles: parseInt(candles) })
            });
            
            const data = await response.json();
            document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            await updateStatus();
        }

        async function trainModels() {
            document.getElementById('results').textContent = 'Entrenando modelos de IA...';
            const response = await fetch('/api/train', { method: 'POST' });
            const data = await response.json();
            document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            await updateStatus();
        }

        // Initialize
        updateStatus();
        setInterval(updateStatus, 5000); // Update every 5 seconds
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

@app.post("/api/toggle")
def toggle_system():
    enabled = assistant.toggle_enabled()
    return {"enabled": enabled}

@app.post("/api/analyze")
async def analyze_asset(data: dict = None):
    asset = "EURUSD-OTC"
    candles = 200
    
    if data:
        asset = data.get('asset', 'EURUSD-OTC')
        candles = data.get('candles', 200)
    
    result = assistant.run_analysis_once(asset=asset, candles=candles)
    return result

@app.post("/api/train")
def train_models():
    # Force training with current data
    result = assistant.run_analysis_once(asset="EURUSD-OTC", candles=300)
    if 'error' in result:
        return {"success": False, "message": f"Error: {result['error']}"}
    else:
        return {"success": True, "message": "Modelos entrenados exitosamente"}

# Health check for Render
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "Delowyss Pro Trading AI"
    }

# -----------------------
# Scheduler for Auto-Training
# -----------------------
scheduler = BackgroundScheduler()

def scheduled_training():
    if not assistant.enabled:
        return
        
    logger.info("🔄 Ejecutando entrenamiento programado...")
    try:
        for asset in ASSETS:
            result = assistant.run_analysis_once(asset=asset, candles=300)
            logger.info(f"📊 {asset}: {'✅' if 'error' not in result else '❌'}")
            time.sleep(2)
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento programado: {e}")

scheduler.add_job(scheduled_training, 'interval', minutes=RETRAIN_INTERVAL_MIN)
scheduler.start()
logger.info(f"⏰ Programador iniciado - Entrenamiento cada {RETRAIN_INTERVAL_MIN} minutos")

if __name__ == "__main__":
    logger.info(f"🚀 Iniciando Delowyss Pro en puerto {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
