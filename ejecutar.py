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
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Indicators
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# IQ Option API (non-official)
from iqoptionapi.stable_api import IQ_Option

# Web API
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------
# Enhanced Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('delowyss.log')
    ]
)
logger = logging.getLogger("DelowyssProV3")

# -----------------------
# Enhanced Configuration
# -----------------------
IQ_EMAIL = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "")
IQ_MODE = os.getenv("IQ_MODE", "PRACTICE")

# Render-specific settings
PORT = int(os.getenv("PORT", 10000))
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Enhanced asset list with priorities
ASSETS = [
    {"name": "EURUSD-OTC", "priority": 1, "timeframe": 1},
    {"name": "EURUSD", "priority": 2, "timeframe": 1},
    {"name": "GBPUSD-OTC", "priority": 3, "timeframe": 1},
    {"name": "USDJPY-OTC", "priority": 4, "timeframe": 1}
]

# Training configuration
RETRAIN_INTERVAL_MIN = 30
MIN_TRAINING_SAMPLES = 100

# -----------------------
# Enhanced Model Management
# -----------------------
XGB_FILE = os.path.join(MODEL_DIR, "xgboost_model.joblib")
RF_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
NN_FILE = os.path.join(MODEL_DIR, "nn_model.h5")
ENSEMBLE_FILE = os.path.join(MODEL_DIR, "ensemble_weights.joblib")

class EnhancedIQConnector:
    def __init__(self, email: str, password: str, mode: str = "PRACTICE"):
        self.email = email
        self.password = password
        self.mode = mode
        self.api = None
        self.connected = False
        self.available_assets = {}
        self.balance = 0
        self._connect()

    def _connect(self):
        """Enhanced connection with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Conectando a IQ Option (intento {attempt + 1}/{max_retries})...")
                self.api = IQ_Option(self.email, self.password)
                connected = self.api.connect()
                
                if not connected:
                    logger.warning(f"Intento {attempt + 1} falló")
                    time.sleep(2)
                    continue
                    
                self.connected = True
                
                # Set account mode
                if self.mode.upper() == "REAL":
                    self.api.change_balance("REAL")
                else:
                    self.api.change_balance("PRACTICE")
                
                # Get account info
                self.balance = self.api.get_balance()
                self.available_assets = self.api.get_all_open_time()
                
                logger.info(f"✅ Conectado exitosamente a IQ Option")
                logger.info(f"💰 Balance: {self.balance}")
                logger.info(f"📊 Activos disponibles: {len(self.available_assets)}")
                return
                
            except Exception as e:
                logger.error(f"Error en intento {attempt + 1}: {str(e)}")
                time.sleep(3)
        
        logger.error("❌ No se pudo conectar después de todos los intentos")
        self.connected = False

    def get_realtime_candles(self, asset: str, timeframe_min: int = 1, count: int = 200):
        """Get candles with enhanced error handling"""
        try:
            if not self.connected:
                self._connect()  # Try to reconnect
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
            logger.error(f"Error obteniendo velas en tiempo real: {str(e)}")
            return None
        finally:
            # Always stop the stream
            try:
                self.api.stop_candles_stream(asset, timeframe_min * 60)
            except:
                pass

    def place_trade(self, asset: str, action: str, amount: float, expiration: int = 1):
        """Place trade with enhanced error handling"""
        try:
            if not self.connected:
                return False, "No conectado"
            
            direction = "put" if action.upper() == "PUT" else "call"
            status, order_id = self.api.buy(amount, asset, direction, expiration)
            
            return status, order_id
            
        except Exception as e:
            return False, str(e)

# -----------------------
# Enhanced Feature Engineering
# -----------------------
def compute_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced technical indicators with more features"""
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Multiple EMAs
    for period in [5, 8, 13, 21]:
        df[f'ema_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
        df[f'ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']

    # Enhanced MACD
    macd = MACD(close=df['close'], window_fast=8, window_slow=21, window_sign=5)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['macd_trend'] = (df['macd'] > df['macd_signal']).astype(int)

    # Multiple RSI periods
    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = RSIIndicator(close=df['close'], window=period).rsi()

    # Stochastic
    st = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['stoch_k'] = st.stoch()
    df['stoch_d'] = st.stoch_signal()
    df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)

    # ADX with trend strength
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()
    df['adx_trend'] = (df['di_plus'] > df['di_minus']).astype(int)

    # Multiple Bollinger Bands
    for period in [14, 20]:
        bb = BollingerBands(close=df['close'], window=period, window_dev=2)
        df[f'bb_upper_{period}'] = bb.bollinger_hband()
        df[f'bb_lower_{period}'] = bb.bollinger_lband()
        df[f'bb_pos_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)

    # Volatility features
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['volatility'] = df['close'].rolling(20).std()

    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Candle patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    df['is_doji'] = (df['body_size'] < 0.001).astype(int)

    # Lag features for temporal patterns
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
        df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()

    # Target: next candle direction with confidence
    next_close = df['close'].shift(-1)
    price_change_pct = (next_close - df['close']) / df['close']
    df['target'] = (price_change_pct > 0).astype(int)
    df['target_strength'] = abs(price_change_pct)  # Confidence measure

    return df.dropna()

# -----------------------
# Enhanced Model Manager
# -----------------------
class EnhancedModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.nn_model: Optional[tf.keras.Model] = None
        self.ensemble_weights = np.array([0.4, 0.3, 0.3])  # XGB, RF, NN
        self.model_lock = threading.Lock()
        self.training_history = []
        self._load_models_if_exist()

    def _load_models_if_exist(self):
        """Enhanced model loading with fallbacks"""
        try:
            if os.path.exists(SCALER_FILE):
                self.scaler = joblib.load(SCALER_FILE)
                logger.info("✅ Scaler cargado")
            
            if os.path.exists(XGB_FILE):
                self.xgb_model = joblib.load(XGB_FILE)
                logger.info("✅ XGBoost cargado")
            
            if os.path.exists(RF_FILE):
                self.rf_model = joblib.load(RF_FILE)
                logger.info("✅ RandomForest cargado")
            
            if os.path.exists(NN_FILE):
                self.nn_model = tf.keras.models.load_model(NN_FILE)
                logger.info("✅ Red Neuronal cargada")
                
            if os.path.exists(ENSEMBLE_FILE):
                self.ensemble_weights = joblib.load(ENSEMBLE_FILE)
                logger.info("✅ Pesos del ensemble cargados")
                
        except Exception as e:
            logger.warning(f"⚠️ Error cargando modelos: {e}. Se entrenarán nuevos modelos.")

    def save_models(self):
        """Enhanced model saving"""
        try:
            with self.model_lock:
                if self.xgb_model is not None:
                    joblib.dump(self.xgb_model, XGB_FILE)
                if self.rf_model is not None:
                    joblib.dump(self.rf_model, RF_FILE)
                joblib.dump(self.scaler, SCALER_FILE)
                if self.nn_model is not None:
                    self.nn_model.save(NN_FILE, include_optimizer=False)
                joblib.dump(self.ensemble_weights, ENSEMBLE_FILE)
            logger.info("💾 Modelos guardados correctamente")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")

    def build_enhanced_models(self, df_features: pd.DataFrame) -> bool:
        """Enhanced model training with validation"""
        try:
            with self.model_lock:
                if len(df_features) < MIN_TRAINING_SAMPLES:
                    logger.warning(f"❌ No hay suficientes datos: {len(df_features)}/{MIN_TRAINING_SAMPLES}")
                    return False

                X = df_features.drop(columns=['target', 'target_strength'])
                y = df_features['target'].astype(int)
                
                # Enhanced feature selection
                feature_importance = self._calculate_feature_importance(X, y)
                top_features = feature_importance.head(30)['feature'].tolist()
                X = X[top_features]

                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )

                # 1. XGBoost with hyperparameter tuning
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )
                xgb_clf.fit(X_train, y_train)
                xgb_score = xgb_clf.score(X_test, y_test)
                self.xgb_model = xgb_clf

                # 2. Random Forest
                rf_clf = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                rf_clf.fit(X_train, y_train)
                rf_score = rf_clf.score(X_test, y_test)
                self.rf_model = rf_clf

                # 3. Enhanced Neural Network
                input_dim = X_train.shape[1]
                nn = Sequential([
                    Dense(256, activation='relu', input_shape=(input_dim,)),
                    Dropout(0.3),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.1),
                    Dense(1, activation='sigmoid')
                ])
                nn.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'AUC']
                )
                
                es = EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )
                
                history = nn.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[es],
                    verbose=0
                )
                nn_score = max(history.history['val_accuracy'])
                self.nn_model = nn

                # Dynamic ensemble weighting based on performance
                scores = np.array([xgb_score, rf_score, nn_score])
                self.ensemble_weights = scores / scores.sum()

                # Log training results
                training_info = {
                    'timestamp': datetime.now(),
                    'samples': len(X),
                    'xgb_score': xgb_score,
                    'rf_score': rf_score,
                    'nn_score': nn_score,
                    'ensemble_weights': self.ensemble_weights.tolist()
                }
                self.training_history.append(training_info)

                logger.info(f"🎯 Entrenamiento completado:")
                logger.info(f"   XGBoost: {xgb_score:.3f}")
                logger.info(f"   Random Forest: {rf_score:.3f}") 
                logger.info(f"   Red Neuronal: {nn_score:.3f}")
                logger.info(f"   Pesos Ensemble: {self.ensemble_weights}")

                self.save_models()
                return True

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            return False

    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance for selection"""
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df

    def predict_enhanced(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced prediction with confidence intervals"""
        try:
            with self.model_lock:
                if self.xgb_model is None or self.rf_model is None or self.nn_model is None:
                    return {"error": "Modelos no entrenados"}

                X = df_features.drop(columns=['target', 'target_strength'])
                
                # Use top features
                feature_importance = self._calculate_feature_importance(X, df_features['target'])
                top_features = feature_importance.head(30)['feature'].tolist()
                X = X[top_features]
                
                X_scaled = self.scaler.transform(X)
                last = X_scaled[-1:].copy()

                # Individual predictions
                xgb_pred = self.xgb_model.predict_proba(last)[0][1]
                rf_pred = self.rf_model.predict_proba(last)[0][1]
                nn_pred = float(self.nn_model.predict(last, verbose=0).ravel()[0])

                # Ensemble prediction
                probs = np.array([xgb_pred, rf_pred, nn_pred])
                ensemble_prob = float(np.dot(self.ensemble_weights, probs))

                # Confidence calculation
                confidence = abs(ensemble_prob - 0.5) * 2  # 0-1 scale
                
                # Signal strength
                if ensemble_prob >= 0.6:
                    signal = "FUERTE_COMPRA"
                    direction = "ALCISTA"
                elif ensemble_prob >= 0.55:
                    signal = "COMPRA"
                    direction = "ALCISTA" 
                elif ensemble_prob <= 0.4:
                    signal = "FUERTE_VENTA"
                    direction = "BAJISTA"
                elif ensemble_prob <= 0.45:
                    signal = "VENTA"
                    direction = "BAJISTA"
                else:
                    signal = "NEUTRAL"
                    direction = "LATERAL"

                return {
                    "direction": direction,
                    "signal": signal,
                    "confidence": float(confidence),
                    "probability": float(ensemble_prob),
                    "xgb": float(xgb_pred),
                    "rf": float(rf_pred),
                    "nn": float(nn_pred),
                    "weights": self.ensemble_weights.tolist(),
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {"error": str(e)}

# -----------------------
# Enhanced Assistant
# -----------------------
class EnhancedDelowyssAssistant:
    def __init__(self, iq_connector: EnhancedIQConnector, model_manager: EnhancedModelManager):
        self.iq = iq_connector
        self.models = model_manager
        self.analysis_lock = threading.Lock()
        self.last_analysis = None
        self.enabled = True
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0
        }

    def toggle_enabled(self):
        with self.analysis_lock:
            self.enabled = not self.enabled
            status = "ACTIVADO" if self.enabled else "DESACTIVADO"
            logger.info(f"🔧 Asistente {status}")
            return self.enabled

    def run_enhanced_analysis(self, asset: str = "EURUSD-OTC", candles: int = 300) -> Dict[str, Any]:
        """Enhanced analysis with real-time data"""
        if not self.enabled:
            return {"error": "ANALYSIS_DISABLED"}

        if not self.iq.connected:
            return {"error": "IQ_NOT_CONNECTED"}

        with self.analysis_lock:
            try:
                # Get real-time data
                df = self.iq.get_realtime_candles(asset, timeframe_min=1, count=candles)
                if df is None or df.empty:
                    return {"error": "NO_DATA"}

                # Compute indicators
                df_ind = compute_enhanced_indicators(df)
                if len(df_ind) < MIN_TRAINING_SAMPLES:
                    return {"error": "NOT_ENOUGH_DATA"}

                # Prepare features
                feature_cols = [c for c in df_ind.columns if c not in ['timestamp', 'target', 'target_strength']]
                features = df_ind[feature_cols + ['target', 'target_strength']].dropna()

                # Train models
                trained = self.models.build_enhanced_models(features)
                if not trained:
                    return {"error": "TRAINING_FAILED"}

                # Make prediction
                prediction = self.models.predict_enhanced(features)

                # Store analysis results
                self.last_analysis = {
                    "asset": asset,
                    "timestamp": datetime.now(),
                    "prediction": prediction,
                    "data_points": len(features),
                    "current_price": df_ind['close'].iloc[-1]
                }

                # Update performance metrics
                self._update_performance_metrics(prediction)

                logger.info(f"📊 Análisis completado para {asset}")
                logger.info(f"   Señal: {prediction.get('signal', 'N/A')}")
                logger.info(f"   Confianza: {prediction.get('confidence', 0):.2f}")

                return self.last_analysis

            except Exception as e:
                logger.error(f"❌ Error en análisis: {e}")
                return {"error": str(e)}

    def _update_performance_metrics(self, prediction):
        """Update prediction performance metrics"""
        # This would need actual trade results to calculate real accuracy
        # For now, we'll just track the number of predictions
        self.performance_metrics['total_predictions'] += 1

    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            "enabled": self.enabled,
            "iq_connected": self.iq.connected,
            "balance": self.iq.balance,
            "models_ready": all([
                self.models.xgb_model is not None,
                self.models.rf_model is not None, 
                self.models.nn_model is not None
            ]),
            "last_analysis": self.last_analysis,
            "performance": self.performance_metrics,
            "training_history": len(self.models.training_history)
        }

# -----------------------
# Enhanced FastAPI App
# -----------------------
iq_conn = EnhancedIQConnector(IQ_EMAIL, IQ_PASSWORD, mode=IQ_MODE)
model_mgr = EnhancedModelManager()
assistant = EnhancedDelowyssAssistant(iq_conn, model_mgr)

app = FastAPI(
    title="Delowyss Assistant Pro V3",
    description="Sistema de Trading con IA Avanzada",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced Web Interface
HTML_INDEX = """
<!DOCTYPE html>
<html>
<head>
    <title>Delowyss Assistant Pro V3</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
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
        .progress-bar {
            background: #ecf0f1;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear(90deg, #27ae60, #2ecc71);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Delowyss Assistant Pro V3</h1>
            <p>Sistema de Trading con IA Avanzada y Aprendizaje en Tiempo Real</p>
        </div>
        
        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h3>📊 Estado del Sistema</h3>
                <div id="system-status" class="status status-inactive">Cargando...</div>
                <div class="analysis-item">
                    <span>IQ Option:</span>
                    <span id="iq-status">Desconectado</span>
                </div>
                <div class="analysis-item">
                    <span>Balance:</span>
                    <span id="balance">$0.00</span>
                </div>
                <div class="analysis-item">
                    <span>Modelos IA:</span>
                    <span id="models-status">No listos</span>
                </div>
                <div class="analysis-item">
                    <span>Predicciones:</span>
                    <span id="predictions-count">0</span>
                </div>
                <button class="btn btn-primary" onclick="toggleSystem()">Activar/Desactivar</button>
            </div>

            <!-- Quick Analysis -->
            <div class="card">
                <h3>⚡ Análisis Rápido</h3>
                <div class="input-group">
                    <label>Activo:</label>
                    <select id="asset-select">
                        <option value="EURUSD-OTC">EURUSD-OTC</option>
                        <option value="EURUSD">EURUSD</option>
                        <option value="GBPUSD-OTC">GBPUSD-OTC</option>
                        <option value="USDJPY-OTC">USDJPY-OTC</option>
                    </select>
                </div>
                <button class="btn btn-success" onclick="runAnalysis()">Analizar y Predecir</button>
                <button class="btn btn-warning" onclick="trainModels()">Entrenar IA</button>
            </div>

            <!-- Current Signal -->
            <div class="card">
                <h3>🎯 Señal Actual</h3>
                <div id="current-signal" class="signal signal-neutral">ESPERANDO ANÁLISIS</div>
                <div id="signal-details">
                    <div class="analysis-item"><span>Dirección:</span><span id="direction">-</span></div>
                    <div class="analysis-item"><span>Confianza:</span><span id="confidence">-</span></div>
                    <div class="analysis-item"><span>Probabilidad:</span><span id="probability">-</span></div>
                    <div class="analysis-item"><span>Última Actualización:</span><span id="last-update">-</span></div>
                </div>
            </div>

            <!-- Model Details -->
            <div class="card">
                <h3>🤖 Detalles de la IA</h3>
                <div class="analysis-item"><span>XGBoost:</span><span id="xgb-score">-</span></div>
                <div class="analysis-item"><span>Random Forest:</span><span id="rf-score">-</span></div>
                <div class="analysis-item"><span>Red Neuronal:</span><span id="nn-score">-</span></div>
                <div class="analysis-item"><span>Pesos Ensemble:</span><span id="ensemble-weights">-</span></div>
                <div class="analysis-item"><span>Entrenamientos:</span><span id="training-count">0</span></div>
            </div>

            <!-- Trading Actions -->
            <div class="card">
                <h3>💰 Acciones de Trading</h3>
                <div class="input-group">
                    <label>Monto:</label>
                    <input type="number" id="trade-amount" value="1" min="1" max="10">
                </div>
                <button class="btn btn-success" onclick="placeTrade('CALL')">COMPRAR (CALL)</button>
                <button class="btn btn-danger" onclick="placeTrade('PUT')">VENDER (PUT)</button>
                <div id="trade-result" style="margin-top: 10px; padding: 10px; border-radius: 5px;"></div>
            </div>

            <!-- Performance -->
            <div class="card">
                <h3>📈 Rendimiento</h3>
                <div class="analysis-item"><span>Precisión:</span><span id="accuracy">0%</span></div>
                <div class="analysis-item"><span>Total Predicciones:</span><span id="total-predictions">0</span></div>
                <div class="analysis-item"><span>Correctas:</span><span id="correct-predictions">0</span></div>
                <div class="progress-bar">
                    <div id="accuracy-bar" class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let updateInterval;

        async function updateSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // System status
                document.getElementById('system-status').textContent = data.enabled ? "ACTIVO" : "INACTIVO";
                document.getElementById('system-status').className = data.enabled ? 'status status-active' : 'status status-inactive';
                
                // IQ Status
                document.getElementById('iq-status').textContent = data.iq_connected ? "Conectado" : "Desconectado";
                document.getElementById('balance').textContent = '$' + (data.balance || '0.00');
                
                // Models status
                document.getElementById('models-status').textContent = data.models_ready ? "Listos" : "No listos";
                
                // Performance
                document.getElementById('total-predictions').textContent = data.performance?.total_predictions || 0;
                document.getElementById('correct-predictions').textContent = data.performance?.correct_predictions || 0;
                document.getElementById('accuracy').textContent = (data.performance?.accuracy * 100 || 0).toFixed(1) + '%';
                document.getElementById('accuracy-bar').style.width = (data.performance?.accuracy * 100 || 0) + '%';
                
                // Training history
                document.getElementById('training-count').textContent = data.training_history || 0;
                
                // Last analysis
                if (data.last_analysis) {
                    const pred = data.last_analysis.prediction;
                    if (pred && !pred.error) {
                        // Update signal
                        const signalElement = document.getElementById('current-signal');
                        signalElement.textContent = pred.signal || 'NEUTRAL';
                        
                        if (pred.signal?.includes('COMPRA')) {
                            signalElement.className = 'signal signal-buy';
                        } else if (pred.signal?.includes('VENTA')) {
                            signalElement.className = 'signal signal-sell';
                        } else {
                            signalElement.className = 'signal signal-neutral';
                        }
                        
                        // Update details
                        document.getElementById('direction').textContent = pred.direction || '-';
                        document.getElementById('confidence').textContent = (pred.confidence * 100).toFixed(1) + '%';
                        document.getElementById('probability').textContent = (pred.probability * 100).toFixed(1) + '%';
                        document.getElementById('last-update').textContent = new Date(pred.timestamp).toLocaleString();
                        
                        // Model scores
                        document.getElementById('xgb-score').textContent = (pred.xgb * 100).toFixed(1) + '%';
                        document.getElementById('rf-score').textContent = (pred.rf * 100).toFixed(1) + '%';
                        document.getElementById('nn-score').textContent = (pred.nn * 100).toFixed(1) + '%';
                        document.getElementById('ensemble-weights').textContent = pred.weights?.map(w => w.toFixed(2)).join(', ') || '-';
                    }
                }
                
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }

        async function toggleSystem() {
            const response = await fetch('/api/toggle', { method: 'POST' });
            const data = await response.json();
            updateSystemStatus();
        }

        async function runAnalysis() {
            const asset = document.getElementById('asset-select').value;
            document.getElementById('current-signal').textContent = 'ANALIZANDO...';
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asset: asset })
            });
            const data = await response.json();
            
            updateSystemStatus();
        }

        async function trainModels() {
            const response = await fetch('/api/train', { method: 'POST' });
            const data = await response.json();
            alert(data.message || 'Entrenamiento completado');
            updateSystemStatus();
        }

        async function placeTrade(action) {
            const amount = document.getElementById('trade-amount').value;
            const asset = document.getElementById('asset-select').value;
            
            const response = await fetch('/api/trade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    action: action,
                    amount: parseFloat(amount),
                    asset: asset
                })
            });
            const data = await response.json();
            
            const resultElement = document.getElementById('trade-result');
            if (data.success) {
                resultElement.innerHTML = `<span style="color: green;">✅ ${data.message}</span>`;
            } else {
                resultElement.innerHTML = `<span style="color: red;">❌ ${data.message}</span>`;
            }
        }

        // Initialize
        updateSystemStatus();
        updateInterval = setInterval(updateSystemStatus, 5000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_INDEX

@app.get("/api/status")
def get_status():
    return assistant.get_system_status()

@app.post("/api/toggle")
def toggle_system():
    enabled = assistant.toggle_enabled()
    return {"enabled": enabled, "message": f"Sistema {'activado' if enabled else 'desactivado'}"}

@app.post("/api/analyze")
async def analyze_asset(data: dict):
    asset = data.get('asset', 'EURUSD-OTC')
    result = assistant.run_enhanced_analysis(asset=asset, candles=300)
    return result

@app.post("/api/train")
def train_models():
    # Force training with current data
    asset = "EURUSD-OTC"
    result = assistant.run_enhanced_analysis(asset=asset, candles=300)
    if 'error' in result:
        return {"success": False, "message": f"Error: {result['error']}"}
    else:
        return {"success": True, "message": "Modelos entrenados exitosamente"}

@app.post("/api/trade")
def place_trade(data: dict):
    asset = data.get('asset', 'EURUSD-OTC')
    action = data.get('action', 'CALL')
    amount = data.get('amount', 1.0)
    
    success, message = iq_conn.place_trade(asset, action, amount)
    return {"success": success, "message": message}

# Enhanced Scheduler
scheduler = BackgroundScheduler()

def scheduled_enhanced_training():
    if not assistant.enabled:
        logger.info("⏸️  Entrenamiento programado omitido (sistema desactivado)")
        return
    
    logger.info("🔄 Iniciando entrenamiento programado...")
    try:
        for asset_info in ASSETS:
            asset = asset_info['name']
            result = assistant.run_enhanced_analysis(asset=asset, candles=300)
            logger.info(f"📊 Entrenamiento {asset}: {'Éxito' if 'error' not in result else 'Fallido'}")
            time.sleep(5)  # Wait between assets
            
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento programado: {e}")

# Schedule enhanced training
scheduler.add_job(
    scheduled_enhanced_training,
    'interval',
    minutes=RETRAIN_INTERVAL_MIN,
    next_run_time=datetime.now() + timedelta(minutes=1)
)

scheduler.start()
logger.info(f"⏰ Programador iniciado: entrenamiento cada {RETRAIN_INTERVAL_MIN} minutos")

# Health check endpoint for Render
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0",
        "system": "Delowyss Assistant Pro"
    }

if __name__ == "__main__":
    logger.info(f"🚀 Iniciando Delowyss Assistant Pro V3 en puerto {PORT}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )
