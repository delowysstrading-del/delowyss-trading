# main.py - VERSIÓN 4.0 MEJORADA CON ESTABILIDAD AVANZADA
"""
Delowyss Trading AI — V4.0-ESTABLE (Production)
Sistema mejorado con análisis avanzado, ensemble learning y máxima estabilidad
CEO: Eduardo Solis — © 2025
"""

import os
import time
import threading
import logging
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from iqoptionapi.stable_api import IQ_Option
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN MEJORADA ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = int(os.getenv("TIMEFRAME", "60"))
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "3"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "v40")
TRAINING_CSV = os.getenv("TRAINING_CSV", f"training_data_{MODEL_VERSION}.csv")
PERF_CSV = os.getenv("PERF_CSV", f"performance_{MODEL_VERSION}.csv")
MODEL_PATH = os.getenv("MODEL_PATH", f"delowyss_mlp_{MODEL_VERSION}.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", f"delowyss_scaler_{MODEL_VERSION}.joblib")
ENSEMBLE_PATH = os.getenv("ENSEMBLE_PATH", f"ensemble_{MODEL_VERSION}.pkl")

BATCH_TRAIN_SIZE = int(os.getenv("BATCH_TRAIN_SIZE", "150"))
PARTIAL_FIT_AFTER = int(os.getenv("PARTIAL_FIT_AFTER", "6"))
CONFIDENCE_SAVE_THRESHOLD = float(os.getenv("CONFIDENCE_SAVE_THRESHOLD", "68.0"))

SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "10"))
MAX_TICKS_MEMORY = int(os.getenv("MAX_TICKS_MEMORY", "800"))
MAX_CANDLE_TICKS = int(os.getenv("MAX_CANDLE_TICKS", "400"))

# Nuevos parámetros de aprendizaje mejorado
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
MODEL_UPDATE_FREQUENCY = int(os.getenv("MODEL_UPDATE_FREQUENCY", "15"))
ENSEMBLE_WEIGHT_UPDATE = int(os.getenv("ENSEMBLE_WEIGHT_UPDATE", "30"))
MIN_TICKS_FOR_PREDICTION = int(os.getenv("MIN_TICKS_FOR_PREDICTION", "8"))
MAX_RETRY_CONNECTION = int(os.getenv("MAX_RETRY_CONNECTION", "5"))

# ---------------- LOGGING MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# ------------------ Incremental Scaler MEJORADO ------------------
class IncrementalScaler:
    def __init__(self):
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.var_ = None
        self.is_fitted_ = False

    def partial_fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        batch_size = X.shape[0]
        batch_mean = np.mean(X, axis=0)
        batch_var = np.var(X, axis=0)

        if self.n_samples_seen_ == 0:
            self.mean_ = batch_mean
            self.var_ = batch_var
        else:
            total = self.n_samples_seen_ + batch_size
            delta = batch_mean - self.mean_
            self.mean_ += delta * batch_size / total
            self.var_ = (
                (self.n_samples_seen_ * self.var_) +
                (batch_size * batch_var) +
                (self.n_samples_seen_ * batch_size * (delta ** 2) / total)
            ) / total

        self.n_samples_seen_ += batch_size
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise ValueError("Scaler not fitted")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / np.sqrt(self.var_ + 1e-8)

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

# ------------------ Analyzer AVANZADO ------------------
class AdvancedTickAnalyzer:
    def __init__(self, base_ema_alpha=0.3):
        self.ticks = deque(maxlen=MAX_TICKS_MEMORY)
        self.candle_ticks = deque(maxlen=MAX_CANDLE_TICKS)
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.smoothed_price = None
        self.base_ema_alpha = base_ema_alpha
        self.ema_alpha = base_ema_alpha
        self.last_tick_time = None
        self.last_patterns = deque(maxlen=8)
        self.tick_count = 0
        self.volatility_history = deque(maxlen=20)
        self.price_history = deque(maxlen=50)
        # Nuevos atributos para análisis avanzado
        self.momentum_history = deque(maxlen=15)
        self.volume_profile = deque(maxlen=30)
        self.price_velocity = deque(maxlen=10)

    def _calculate_advanced_indicators(self, price: float) -> Dict:
        """Calcula indicadores técnicos avanzados en tiempo real"""
        indicators = {}
        
        if len(self.price_history) >= 10:
            prices = np.array(list(self.price_history))
            
            # RSI-like indicator mejorado
            changes = np.diff(prices[-10:])
            gains = changes[changes > 0].sum() if len(changes[changes > 0]) > 0 else 0
            losses = -changes[changes < 0].sum() if len(changes[changes < 0]) > 0 else 0
            
            if losses == 0:
                rsi_like = 100
            else:
                rs = gains / losses
                rsi_like = 100 - (100 / (1 + rs))
            indicators['rsi_like'] = rsi_like
            
            # Momentum mejorado multi-timeframe
            if len(prices) >= 5:
                short_momentum = (prices[-1] - prices[-5]) * 10000
                long_momentum = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else short_momentum
                indicators['momentum_acceleration'] = short_momentum - long_momentum
                
                # Velocidad del precio
                if len(self.price_velocity) > 0:
                    indicators['price_velocity'] = np.mean(list(self.price_velocity)) * 10000
                else:
                    indicators['price_velocity'] = 0
            else:
                indicators['momentum_acceleration'] = 0
                indicators['price_velocity'] = 0
                
        return indicators

    def _update_ema_alpha(self, current_volatility):
        """Actualización adaptativa del parámetro EMA"""
        try:
            self.volatility_history.append(current_volatility)
            smoothed_vol = np.mean(list(self.volatility_history))
            if smoothed_vol < 0.4:
                self.ema_alpha = self.base_ema_alpha * 0.5
            elif smoothed_vol < 1.2:
                self.ema_alpha = self.base_ema_alpha
            elif smoothed_vol < 2.5:
                self.ema_alpha = self.base_ema_alpha * 1.4
            else:
                self.ema_alpha = self.base_ema_alpha * 1.8
            self.ema_alpha = max(0.05, min(0.7, self.ema_alpha))
        except Exception:
            self.ema_alpha = self.base_ema_alpha

    def add_tick(self, price: float, volume: float = 1.0):
        """Procesar tick con validación mejorada"""
        price = float(price)
        current_time = time.time()
        
        # Validación robusta de precio
        if price <= 0:
            logging.warning(f"Precio inválido ignorado: {price}")
            return None
            
        # Detección de anomalías mejorada
        if self.ticks and len(self.ticks) > 0:
            last_tick = self.ticks[-1]
            last_price = last_tick['price']
            time_gap = current_time - last_tick['timestamp']
            if last_price > 0 and time_gap < 2.0:
                price_change_pct = abs(price - last_price) / last_price
                if price_change_pct > 0.02:
                    logging.warning(f"Anomaly spike ignorado: {last_price:.5f} -> {price:.5f}")
                    return None

        # Inicializar vela si es necesario
        if self.current_candle_open is None:
            self.current_candle_open = self.current_candle_high = self.current_candle_low = price
        else:
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)

        interval = current_time - self.last_tick_time if self.last_tick_time else 0.1
        self.last_tick_time = current_time

        # Calcular volatilidad y actualizar EMA
        current_volatility = (self.current_candle_high - self.current_candle_low) * 10000
        self._update_ema_alpha(current_volatility)

        # Suavizado de precio adaptativo
        if self.smoothed_price is None:
            self.smoothed_price = price
        else:
            self.smoothed_price = (self.ema_alpha * price + (1 - self.ema_alpha) * self.smoothed_price)

        # Calcular velocidad del precio
        if len(self.ticks) > 0:
            last_price = self.ticks[-1]['price']
            time_diff = current_time - self.ticks[-1]['timestamp']
            if time_diff > 0:
                velocity = (price - last_price) / time_diff
                self.price_velocity.append(velocity)

        # Calcular indicadores avanzados
        advanced_indicators = self._calculate_advanced_indicators(price)

        tick_data = {
            "timestamp": current_time,
            "price": price,
            "volume": volume,
            "interval": interval,
            "smoothed_price": self.smoothed_price,
            **advanced_indicators
        }
        self.ticks.append(tick_data)
        self.candle_ticks.append(tick_data)
        self.sequence.append(price)
        self.price_history.append(price)
        self.tick_count += 1

        # Detección de patrones en tiempo real
        if len(self.sequence) >= 5:
            pattern = self._detect_micro_pattern()
            if pattern:
                self.last_patterns.appendleft((datetime.utcnow().isoformat(), pattern))
                
        # Logging optimizado
        if self.tick_count <= 10 or self.tick_count % 25 == 0:
            logging.info(f"✅ Tick #{self.tick_count} procesado - Precio: {price:.5f}")
        return tick_data

    def get_price_history(self):
        return list(self.price_history)

    def _detect_micro_pattern(self):
        """Detección mejorada de patrones de precio"""
        try:
            arr = np.array(self.sequence)
            if len(arr) < 5:
                return None
            diffs = np.diff(arr)
            pos_diffs = (diffs > 0).sum()
            neg_diffs = (diffs < 0).sum()
            total = len(diffs)
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            
            # Patrones mejorados
            if pos_diffs >= total * 0.8 and mean_diff > 0.00003:
                return "ramp-up"
            elif neg_diffs >= total * 0.8 and mean_diff < -0.00003:
                return "ramp-down"
            elif std_diff < 0.00002 and abs(mean_diff) < 0.00001:
                return "consolidation"
            elif np.sum(np.abs(np.diff(np.sign(diffs))) > 0) > total * 0.5:
                return "oscillation"
            elif abs(mean_diff) < 0.00001 and std_diff > 0.00005:
                return "high_volatility"
        except Exception:
            pass
        return None

    def get_market_metrics(self):
        """Métricas avanzadas del mercado"""
        if len(self.candle_ticks) < 10:
            return {}
            
        try:
            prices = [t['price'] for t in self.candle_ticks]
            if len(prices) < 2:
                return {}
                
            price_array = np.array(prices)
            returns = np.diff(price_array) / price_array[:-1]
            volatility = np.std(returns) * 10000 if len(returns) > 0 else 0
            
            # Niveles de soporte/resistencia
            if len(prices) >= 20:
                resistance = max(prices[-20:])
                support = min(prices[-20:])
            else:
                resistance = max(prices) if prices else 0
                support = min(prices) if prices else 0
            
            return {
                'advanced_volatility': volatility,
                'resistance_level': resistance,
                'support_level': support,
                'price_range': resistance - support
            }
        except Exception as e:
            logging.debug(f"Error en métricas avanzadas: {e}")
            return {}

    def get_candle_metrics(self, seconds_remaining_norm: float = None):
        """Métricas de vela mejoradas"""
        if len(self.candle_ticks) < 2:
            return None
            
        try:
            ticks_array = np.array([(
                t['price'], 
                t['volume'], 
                t['interval'],
                t.get('rsi_like', 50),
                t.get('momentum_acceleration', 0),
                t.get('price_velocity', 0)
            ) for t in self.candle_ticks], dtype=np.float32)
            
            prices = ticks_array[:, 0]
            volumes = ticks_array[:, 1]
            intervals = ticks_array[:, 2]
            rsi_values = ticks_array[:, 3]
            momentum_acc = ticks_array[:, 4]
            velocity_values = ticks_array[:, 5]

            current_price = float(prices[-1])
            open_price = float(self.current_candle_open)
            high_price = float(self.current_candle_high)
            low_price = float(self.current_candle_low)

            price_changes = np.diff(prices)
            up_ticks = np.sum(price_changes > 0)
            down_ticks = np.sum(price_changes < 0)
            total_ticks = max(1, up_ticks + down_ticks)

            buy_pressure = up_ticks / total_ticks
            sell_pressure = down_ticks / total_ticks
            pressure_ratio = buy_pressure / max(0.01, sell_pressure)

            if len(prices) >= 8:
                momentum = (prices[-1] - prices[0]) * 10000
            else:
                momentum = (current_price - open_price) * 10000

            volatility = (high_price - low_price) * 10000
            price_change = (current_price - open_price) * 10000

            valid_intervals = intervals[intervals > 0]
            tick_speed = 1.0 / np.mean(valid_intervals) if len(valid_intervals) > 0 else 0.0

            if len(price_changes) > 1:
                signs = np.sign(price_changes)
                direction_changes = np.sum(np.abs(np.diff(signs)) > 0)
                direction_ratio = direction_changes / len(price_changes)
            else:
                direction_ratio = 0.0

            # Indicadores avanzados promediados
            avg_rsi = np.mean(rsi_values) if len(rsi_values) > 0 else 50
            avg_momentum_acc = np.mean(momentum_acc) if len(momentum_acc) > 0 else 0
            avg_velocity = np.mean(velocity_values) if len(velocity_values) > 0 else 0

            # Métricas de mercado avanzadas
            market_metrics = self.get_market_metrics()
            
            # DETECCIÓN DE FASE DE MERCADO MEJORADA
            advanced_vol = market_metrics.get('advanced_volatility', volatility)
            
            if advanced_vol > 2.0 and abs(momentum) > 2.5:
                market_phase = "strong_trend"
                phase_confidence = 0.9
            elif advanced_vol > 1.0 and abs(momentum) > 1.5:
                market_phase = "moderate_trend"
                phase_confidence = 0.7
            elif advanced_vol < 0.3:
                market_phase = "consolidation"
                phase_confidence = 0.6
            elif advanced_vol > 1.5:
                market_phase = "high_volatility"
                phase_confidence = 0.8
            else:
                market_phase = "neutral"
                phase_confidence = 0.5

            metrics = {
                "open_price": open_price,
                "high_price": high_price,
                "low_price": low_price,
                "current_price": current_price,
                "buy_pressure": buy_pressure,
                "sell_pressure": sell_pressure,
                "pressure_ratio": pressure_ratio,
                "momentum": momentum,
                "volatility": volatility,
                "up_ticks": int(up_ticks),
                "down_ticks": int(down_ticks),
                "total_ticks": len(self.candle_ticks),
                "volume_trend": float(np.mean(volumes)),
                "price_change": price_change,
                "tick_speed": tick_speed,
                "direction_ratio": direction_ratio,
                "market_phase": market_phase,
                "phase_confidence": phase_confidence,
                "rsi_like": avg_rsi,
                "momentum_acceleration": avg_momentum_acc,
                "price_velocity": avg_velocity,
                "last_patterns": list(self.last_patterns)[:4],
                "market_metrics": market_metrics,
                "data_quality": min(1.0, len(self.candle_ticks) / 30.0),
                "timestamp": time.time()
            }
            if seconds_remaining_norm is not None:
                metrics['seconds_remaining_norm'] = float(seconds_remaining_norm)
            return metrics
        except Exception as e:
            logging.error(f"Error calculando métricas avanzadas: {e}")
            return None

    def reset_candle(self):
        """Reinicio completo de la vela"""
        self.candle_ticks.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.sequence.clear()
        self.tick_count = 0
        logging.info("🔄 Vela reiniciada")

# ------------------ Predictor INTELIGENTE con Ensemble Learning ------------------
class IntelligentPredictor:
    def __init__(self):
        self.analyzer = AdvancedTickAnalyzer()
        self.model = None
        self.scaler = None
        self.ensemble_models = {}
        self.ensemble_weights = {}
        self.prev_candle_metrics = None
        self.partial_buffer = []
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent': deque(maxlen=50),
            'model_performance': {},
            'feature_importance': {}
        }
        self.last_prediction = None
        self.prediction_history = deque(maxlen=30)
        self._initialize_enhanced_system()
        self._ensure_files()

    def _feature_names(self):
        return [
            "buy_pressure", "sell_pressure", "pressure_ratio", "momentum",
            "volatility", "up_ticks", "down_ticks", "total_ticks",
            "volume_trend", "price_change", "tick_speed", "direction_ratio",
            "seconds_remaining_norm", "rsi_like", "momentum_acceleration",
            "price_velocity", "phase_confidence", "data_quality"
        ]

    def _ensure_files(self):
        """Garantizar que los archivos necesarios existan"""
        try:
            if not os.path.exists(TRAINING_CSV):
                pd.DataFrame(columns=self._feature_names() + ["label", "timestamp", "pattern"]).to_csv(TRAINING_CSV, index=False)
            if not os.path.exists(PERF_CSV):
                pd.DataFrame(columns=[
                    "timestamp", "prediction", "actual", "correct", "confidence", 
                    "model_used", "ensemble_weight", "market_phase", "price_change_pips"
                ]).to_csv(PERF_CSV, index=False)
        except Exception as e:
            logging.error("Error initializing files: %s", e)

    def _initialize_enhanced_system(self):
        """Sistema de inicialización mejorado con ensemble learning"""
        try:
            # Cargar modelo principal si existe
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                logging.info("✅ Modelo ML existente cargado")
            else:
                self._initialize_new_model()

            # Cargar o inicializar ensemble
            if os.path.exists(ENSEMBLE_PATH):
                with open(ENSEMBLE_PATH, 'rb') as f:
                    ensemble_data = pickle.load(f)
                    self.ensemble_models = ensemble_data.get('models', {})
                    self.ensemble_weights = ensemble_data.get('weights', {})
                logging.info("✅ Ensemble models loaded")
            else:
                self._initialize_ensemble_models()
                
        except Exception as e:
            logging.error(f"❌ Error cargando sistema mejorado: {e}")
            self._initialize_new_model()
            self._initialize_ensemble_models()

    def _initialize_new_model(self):
        """Inicializar nuevo modelo con configuración robusta"""
        try:
            self.scaler = IncrementalScaler()
            self.model = MLPClassifier(
                hidden_layer_sizes=(64,32),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
                learning_rate_init=LEARNING_RATE,
                max_iter=1,
                warm_start=True,
                random_state=42
            )
            n = len(self._feature_names())
            X_dummy = np.random.normal(0, 0.1, (10, n)).astype(np.float32)
            y_dummy = np.random.randint(0,2,10)
            self.scaler.partial_fit(X_dummy)
            Xs = self.scaler.transform(X_dummy)
            try:
                self.model.partial_fit(Xs, y_dummy, classes=[0,1])
            except Exception:
                self.model.fit(Xs, y_dummy)
            self._save_artifacts()
            logging.info("✅ Nuevo modelo ML inicializado")
        except Exception as e:
            logging.error(f"❌ Error init model: {e}")
            self.model = None
            self.scaler = None

    def _initialize_ensemble_models(self):
        """Inicializar modelos ensemble para mejor predicción"""
        try:
            self.ensemble_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=30,
                    max_depth=8,
                    random_state=42
                ),
                'gradient_boost': GradientBoostingClassifier(
                    n_estimators=30,
                    max_depth=5,
                    random_state=42
                )
            }
            
            # Pesos iniciales balanceados
            self.ensemble_weights = {name: 1.0 for name in self.ensemble_models.keys()}
            self.performance_stats['model_performance'] = {name: deque(maxlen=15) for name in self.ensemble_models.keys()}
            
            self._save_ensemble()
            logging.info("✅ Ensemble models initialized")
        except Exception as e:
            logging.error(f"❌ Error initializing ensemble: {e}")

    def _save_ensemble(self):
        """Guardar estado del ensemble"""
        try:
            ensemble_data = {
                'models': self.ensemble_models,
                'weights': self.ensemble_weights,
                'performance': self.performance_stats['model_performance']
            }
            with open(ENSEMBLE_PATH, 'wb') as f:
                pickle.dump(ensemble_data, f)
        except Exception as e:
            logging.error(f"❌ Error saving ensemble: {e}")

    def _save_artifacts(self):
        """Guardar modelos y scalers"""
        try:
            if self.model and self.scaler:
                joblib.dump(self.model, MODEL_PATH)
                joblib.dump(self.scaler, SCALER_PATH)
                logging.info("💾 Modelo guardado")
        except Exception as e:
            logging.error(f"❌ Error guardando artifacts: {e}")

    def extract_features(self, metrics):
        """Extraer características para el modelo"""
        try:
            features = [safe_float(metrics.get(k,0.0)) for k in self._feature_names()]
            return np.array(features, dtype=np.float32)
        except Exception:
            return np.zeros(len(self._feature_names()), dtype=np.float32)

    def append_sample_if_confident(self, metrics, label, confidence):
        """Agregar muestra si la confianza es suficiente"""
        try:
            if confidence < CONFIDENCE_SAVE_THRESHOLD:
                return
            row = {k: metrics.get(k,0.0) for k in self._feature_names()}
            row["label"] = int(label)
            row["timestamp"] = datetime.utcnow().isoformat()
            row["pattern"] = metrics.get("market_phase", "unknown")
            pd.DataFrame([row]).to_csv(TRAINING_CSV, mode="a", header=False, index=False)
            self.partial_buffer.append((row,label))
            logging.info(f"💾 Sample guardado - label={label} conf={confidence}% buffer={len(self.partial_buffer)}")
            if len(self.partial_buffer) >= PARTIAL_FIT_AFTER:
                self._perform_enhanced_partial_fit()
        except Exception as e:
            logging.error(f"❌ Error append sample: {e}")

    def _perform_enhanced_partial_fit(self):
        """Entrenamiento parcial mejorado para múltiples modelos"""
        if not self.partial_buffer or not self.model or not self.scaler:
            self.partial_buffer.clear()
            return
        try:
            X_new = np.array([[r[f] for f in self._feature_names()] for (r,_) in self.partial_buffer], dtype=np.float32)
            y_new = np.array([lbl for (_,lbl) in self.partial_buffer])
            self.scaler.partial_fit(X_new)
            Xs = self.scaler.transform(X_new)
            try:
                self.model.partial_fit(Xs, y_new)
            except Exception:
                self.model.fit(Xs, y_new)
            
            # Entrenar modelos ensemble también
            if len(X_new) >= 8:
                for name, model in self.ensemble_models.items():
                    try:
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(Xs, y_new, classes=[0,1])
                        else:
                            # Reentrenar completo si no soporta partial_fit
                            model.fit(Xs, y_new)
                    except Exception as e:
                        logging.warning(f"⚠️ Error training ensemble model {name}: {e}")
            
            self._save_artifacts()
            self._save_ensemble()
            logging.info(f"🧠 Enhanced partial fit completado con {len(X_new)} samples")
            self.partial_buffer.clear()
        except Exception as e:
            logging.error(f"❌ Error enhanced partial fit: {e}")
            self.partial_buffer.clear()

    def _ensemble_predict(self, features):
        """Predicción por ensemble con pesos dinámicos"""
        if not self.ensemble_models:
            return None
            
        try:
            Xs = self.scaler.transform(features.reshape(1, -1))
            predictions = []
            confidences = []
            model_details = []
            
            for name, model in self.ensemble_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(Xs)[0]
                        up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        confidence = int(max(up_prob, 1 - up_prob) * 100)
                        
                        # Aplicar peso del modelo
                        weight = self.ensemble_weights.get(name, 1.0)
                        weighted_prob = (up_prob - 0.5) * weight + 0.5
                        
                        predictions.append(weighted_prob)
                        confidences.append(confidence)
                        model_details.append({
                            'name': name,
                            'weight': weight,
                            'confidence': confidence,
                            'prob_up': up_prob
                        })
                except Exception as e:
                    logging.debug(f"Ensemble model {name} prediction failed: {e}")
                    continue
            
            if predictions:
                avg_pred = np.mean(predictions)
                avg_confidence = int(np.mean(confidences))
                
                return {
                    "prob_up": avg_pred,
                    "confidence": avg_confidence,
                    "direction": "ALZA" if avg_pred >= 0.5 else "BAJA",
                    "model_type": "ENSEMBLE",
                    "models_used": len(predictions),
                    "model_details": model_details
                }
                
        except Exception as e:
            logging.warning(f"⚠️ Ensemble prediction error: {e}")
            
        return None

    def _update_ensemble_weights(self, correct: bool, model_used: str):
        """Actualizar pesos del ensemble basado en performance"""
        try:
            if model_used in self.ensemble_weights:
                current_weight = self.ensemble_weights[model_used]
                
                if correct:
                    new_weight = min(2.0, current_weight * 1.08)  # Aumento más conservador
                else:
                    new_weight = max(0.1, current_weight * 0.92)  # Decremento más conservador
                    
                self.ensemble_weights[model_used] = new_weight
                
                # Registrar performance
                if model_used in self.performance_stats['model_performance']:
                    self.performance_stats['model_performance'][model_used].append(1 if correct else 0)
                
                # Recalcular pesos periódicamente
                if self.performance_stats['total_predictions'] % ENSEMBLE_WEIGHT_UPDATE == 0:
                    self._rebalance_ensemble_weights()
                    
        except Exception as e:
            logging.error(f"❌ Error updating ensemble weights: {e}")

    def _rebalance_ensemble_weights(self):
        """Rebalancear pesos del ensemble basado en performance"""
        try:
            total_performance = 0
            performances = {}
            
            for model_name, perf_deque in self.performance_stats['model_performance'].items():
                if len(perf_deque) > 5:  # Mínimo de muestras para confianza
                    performance = sum(perf_deque) / len(perf_deque)
                    performances[model_name] = performance
                    total_performance += performance
            
            if total_performance > 0 and len(performances) > 1:
                for model_name, performance in performances.items():
                    normalized_perf = performance / total_performance
                    self.ensemble_weights[model_name] = normalized_perf * len(performances)
                    
                logging.info(f"🔧 Ensemble weights rebalanced: {self.ensemble_weights}")
            
        except Exception as e:
            logging.error(f"❌ Error rebalancing ensemble weights: {e}")

    def validate_previous_prediction(self, current_candle_metrics):
        """Valida si la última predicción fue correcta - MEJORADA"""
        if not self.last_prediction:
            return None
            
        try:
            if self.prev_candle_metrics is None:
                return None
                
            prev_close = float(self.prev_candle_metrics["current_price"])
            current_close = float(current_candle_metrics["current_price"])
            
            # Validación más estricta con umbral mínimo
            price_change = (current_close - prev_close) * 10000
            minimal_change = 0.05  # Cambio mínimo para considerar válido
            
            if abs(price_change) < minimal_change:
                actual_direction = "LATERAL"
                correct = False
            else:
                actual_direction = "ALZA" if current_close > prev_close else "BAJA"
                predicted_direction = self.last_prediction.get("direction", "N/A")
                correct = (actual_direction == predicted_direction)
            
            confidence = self.last_prediction.get("confidence", 0)

            result = {
                "timestamp": now_iso(),
                "predicted": predicted_direction,
                "actual": actual_direction,
                "correct": correct,
                "confidence": confidence,
                "price_change_pips": round(price_change, 2),
                "previous_price": round(prev_close, 5),
                "current_price": round(current_close, 5),
                "model_used": self.last_prediction.get("model_used", "UNKNOWN"),
                "reasons": self.last_prediction.get("reasons", []),
                "minimal_change_threshold": minimal_change
            }
            
            # Actualizar pesos del ensemble si se usó
            model_used = self.last_prediction.get("model_used", "")
            if "ENSEMBLE" in model_used or any(name in model_used for name in self.ensemble_models.keys()):
                self._update_ensemble_weights(correct, model_used)
            
            status = "✅ CORRECTA" if correct else "❌ ERRÓNEA"
            if actual_direction == "LATERAL":
                status = "⚪ LATERAL"
                
            logging.info(f"🎯 VALIDACIÓN: {status} | Pred: {predicted_direction} | Real: {actual_direction} | Conf: {confidence}% | Change: {price_change:.1f}pips")
            
            self._update_global_performance_stats(correct, result)
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error validando predicción: {e}")
            return None

    def _update_global_performance_stats(self, correct, validation_result):
        """Actualiza estadísticas globales de performance"""
        global performance_stats
        
        performance_stats['total_predictions'] += 1
        if correct and validation_result.get('actual') != 'LATERAL':
            performance_stats['correct_predictions'] += 1
            
        performance_stats['last_10'].append(1 if correct else 0)
        performance_stats['last_validation'] = validation_result
        
        if performance_stats['last_10']:
            recent_correct = sum(performance_stats['last_10'])
            performance_stats['recent_accuracy'] = (recent_correct / len(performance_stats['last_10'])) * 100
        
        # Log periódico de performance
        if performance_stats['total_predictions'] % 5 == 0:
            overall_acc = (performance_stats['correct_predictions'] / performance_stats['total_predictions'] * 100)
            logging.info(f"📊 PERFORMANCE: Global: {overall_acc:.1f}% | Reciente: {performance_stats['recent_accuracy']:.1f}% | Total: {performance_stats['total_predictions']}")

    def on_candle_closed(self, closed_metrics):
        """Manejo de cierre de vela con aprendizaje"""
        try:
            if self.prev_candle_metrics is not None:
                prev_close = float(self.prev_candle_metrics["current_price"])
                this_close = float(closed_metrics["current_price"])
                label = 1 if this_close > prev_close else 0
                if self.last_prediction:
                    conf = safe_float(self.last_prediction.get("confidence",0.0))
                    self.append_sample_if_confident(self.prev_candle_metrics, label, conf)
                    self._record_performance(self.last_prediction, label)
                self.prev_candle_metrics = closed_metrics.copy()
                self.last_prediction = None
            else:
                self.prev_candle_metrics = closed_metrics.copy()
        except Exception as e:
            logging.error(f"❌ Error on_candle_closed: {e}")

    def _record_performance(self, pred, actual_label):
        """Registrar performance para análisis"""
        try:
            correct = ((pred.get("direction")=="ALZA" and actual_label==1) or (pred.get("direction")=="BAJA" and actual_label==0))
            price_change = (self.prev_candle_metrics["current_price"] - self.prev_candle_metrics["open_price"]) * 10000
            
            rec = {
                "timestamp": now_iso(), 
                "prediction": pred.get("direction"), 
                "actual": "ALZA" if actual_label==1 else "BAJA",
                "correct": correct, 
                "confidence": pred.get("confidence",0), 
                "model_used": pred.get("model_used","HYBRID"),
                "ensemble_weight": json.dumps(self.ensemble_weights),
                "market_phase": self.prev_candle_metrics.get("market_phase", "unknown") if self.prev_candle_metrics else "unknown",
                "price_change_pips": round(price_change, 2)
            }
            pd.DataFrame([rec]).to_csv(PERF_CSV, mode="a", header=False, index=False)
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['correct_predictions'] += int(correct)
            self.performance_stats['recent'].append(int(correct))
            logging.info(f"📊 Performance registrada - Correcto: {correct}")
        except Exception as e:
            logging.error(f"❌ Error recording performance: {e}")

    def _intelligent_rule_based(self, metrics):
        """SISTEMA DE REGLAS INTELIGENTE - CONFIANZA EN ENTEROS"""
        signals = []
        confidences = []
        reasons = []
        
        pr = metrics.get("pressure_ratio", 1.0)
        mom = metrics.get("momentum", 0.0)
        bp = metrics.get("buy_pressure", 0.5)
        sp = metrics.get("sell_pressure", 0.5)
        vol = metrics.get("volatility", 0.0)
        phase = metrics.get("market_phase", "neutral")
        total_ticks = metrics.get("total_ticks", 0)
        rsi = metrics.get("rsi_like", 50)
        mom_acc = metrics.get("momentum_acceleration", 0.0)
        price_velocity = metrics.get("price_velocity", 0.0)
        data_quality = metrics.get("data_quality", 0.5)
        
        # 1. ANÁLISIS RSI MEJORADO
        if rsi > 75:
            signals.append(0)
            confidences.append(70)
            reasons.append(f"RSI sobrecompra {rsi:.1f}")
        elif rsi > 65:
            signals.append(0)
            confidences.append(60)
            reasons.append(f"RSI elevado {rsi:.1f}")
        elif rsi < 25:
            signals.append(1)
            confidences.append(70)
            reasons.append(f"RSI sobreventa {rsi:.1f}")
        elif rsi < 35:
            signals.append(1)
            confidences.append(60)
            reasons.append(f"RSI bajo {rsi:.1f}")
            
        # 2. ACELERACIÓN DEL MOMENTUM MEJORADA
        if mom_acc > 1.0:
            signals.append(1)
            confidences.append(65 + int(min(mom_acc, 4) * 8))
            reasons.append(f"Aceleración alcista fuerte {mom_acc:.1f}")
        elif mom_acc > 0.5:
            signals.append(1)
            confidences.append(60 + int(min(mom_acc, 2) * 10))
            reasons.append(f"Aceleración alcista {mom_acc:.1f}")
        elif mom_acc < -1.0:
            signals.append(0)
            confidences.append(65 + int(min(abs(mom_acc), 4) * 8))
            reasons.append(f"Aceleración bajista fuerte {mom_acc:.1f}")
        elif mom_acc < -0.5:
            signals.append(0)
            confidences.append(60 + int(min(abs(mom_acc), 2) * 10))
            reasons.append(f"Aceleración bajista {mom_acc:.1f}")
        
        # 3. VELOCIDAD DEL PRECIO
        if abs(price_velocity) > 1.0:
            if price_velocity > 0:
                signals.append(1)
                confidences.append(60 + int(min(price_velocity, 3) * 10))
                reasons.append(f"Velocidad alcista {price_velocity:.1f}")
            else:
                signals.append(0)
                confidences.append(60 + int(min(abs(price_velocity), 3) * 10))
                reasons.append(f"Velocidad bajista {price_velocity:.1f}")
        
        # 4. PRESSURE RATIO - SEÑAL FUERTE (ORIGINAL MEJORADA)
        if pr > 2.5:
            signals.append(1)
            confidences.append(min(85, 55 + int((pr - 2.0) * 12)))
            reasons.append(f"Presión compra muy fuerte {pr:.1f}x")
        elif pr > 2.0:
            signals.append(1)
            confidences.append(min(75, 50 + int((pr - 1.8) * 15)))
            reasons.append(f"Presión compra fuerte {pr:.1f}x")
        elif pr > 1.6:
            signals.append(1)
            confidences.append(min(65, 45 + int((pr - 1.5) * 20)))
            reasons.append(f"Presión compra {pr:.1f}x")
        elif pr < 0.4:
            signals.append(0)
            confidences.append(min(85, 55 + int((0.45 - pr) * 12)))
            reasons.append(f"Presión venta muy fuerte {pr:.1f}x")
        elif pr < 0.6:
            signals.append(0)
            confidences.append(min(75, 50 + int((0.65 - pr) * 15)))
            reasons.append(f"Presión venta fuerte {pr:.1f}x")
        elif pr < 0.8:
            signals.append(0)
            confidences.append(min(65, 45 + int((0.75 - pr) * 20)))
            reasons.append(f"Presión venta {pr:.1f}x")
        
        # 5. MOMENTUM - SEÑAL MEDIA (ORIGINAL MEJORADA)
        if mom > 3.0:
            signals.append(1)
            confidences.append(min(80, 50 + int(min(mom, 10) * 3)))
            reasons.append(f"Momento alcista fuerte {mom:.1f}pips")
        elif mom > 1.5:
            signals.append(1)
            confidences.append(min(70, 45 + int(min(mom, 6) * 4)))
            reasons.append(f"Momento alcista {mom:.1f}pips")
        elif mom < -3.0:
            signals.append(0)
            confidences.append(min(80, 50 + int(min(abs(mom), 10) * 3)))
            reasons.append(f"Momento bajista fuerte {mom:.1f}pips")
        elif mom < -1.5:
            signals.append(0)
            confidences.append(min(70, 45 + int(min(abs(mom), 6) * 4)))
            reasons.append(f"Momento bajista {mom:.1f}pips")
        
        # 6. BUY/SELL PRESSURE - SEÑAL DIRECTA (ORIGINAL MEJORADA)
        if bp > 0.75:
            signals.append(1)
            confidences.append(75)
            reasons.append(f"Dominio compra fuerte {bp:.0%}")
        elif bp > 0.65:
            signals.append(1)
            confidences.append(65)
            reasons.append(f"Dominio compra {bp:.0%}")
        elif sp > 0.75:
            signals.append(0)
            confidences.append(75)
            reasons.append(f"Dominio venta fuerte {sp:.0%}")
        elif sp > 0.65:
            signals.append(0)
            confidences.append(65)
            reasons.append(f"Dominio venta {sp:.0%}")
        
        # DECISIÓN FINAL INTELIGENTE CON CONFIANZA EN ENTEROS
        if signals:
            # Calcular confianza promedio ponderada
            avg_confidence = int(sum(confidences) / len(confidences))
            
            buy_signals = sum(1 for s in signals if s == 1)
            sell_signals = sum(1 for s in signals if s == 0)
            
            signal_strength = abs(buy_signals - sell_signals)
            
            if buy_signals > sell_signals and signal_strength >= 2:
                direction = 1
                final_confidence = min(90, avg_confidence + 15)
                reasons.append("Señales alcistas consistentes y fuertes")
            elif sell_signals > buy_signals and signal_strength >= 2:
                direction = 0
                final_confidence = min(90, avg_confidence + 15)
                reasons.append("Señales bajistas consistentes y fuertes")
            elif buy_signals > sell_signals:
                direction = 1
                final_confidence = min(80, avg_confidence + 8)
                reasons.append("Señales alcistas moderadas")
            elif sell_signals > buy_signals:
                direction = 0
                final_confidence = min(80, avg_confidence + 8)
                reasons.append("Señales bajistas moderadas")
            else:
                # Empate - análisis de desempate inteligente
                if mom_acc > 0.2:
                    direction = 1
                    final_confidence = max(45, avg_confidence - 10)
                    reasons.append("Empate - desempate por aceleración positiva")
                elif mom_acc < -0.2:
                    direction = 0
                    final_confidence = max(45, avg_confidence - 10)
                    reasons.append("Empate - desempate por aceleración negativa")
                else:
                    direction = 1 if mom > 0 else 0
                    final_confidence = max(40, avg_confidence - 15)
                    reasons.append("Empate - desempate por momentum")
        else:
            # Sin señales claras - fallback a movimiento de precio
            price_change = metrics.get("price_change", 0)
            if abs(price_change) > 1.0:
                direction = 1 if price_change > 0 else 0
                final_confidence = 50 + int(min(abs(price_change), 5) * 6)
                reasons.append(f"Basado en movimiento significativo: {price_change:.1f}pips")
            else:
                direction = 1 if metrics.get("price_change", 0) > 0 else 0
                final_confidence = 45
                reasons.append("Mercado lateral - predicción conservadora")
        
        # AJUSTES FINALES DE CONFIANZA
        # Ajuste por calidad de datos
        if data_quality < 0.6:
            final_confidence = max(35, int(final_confidence * data_quality))
            reasons.append(f"Calidad de datos baja: {data_quality:.1f}")
        
        # Ajuste por cantidad de ticks
        if total_ticks < 10:
            final_confidence = max(30, final_confidence - 15)
            reasons.append(f"Pocos datos: {total_ticks} ticks")
        elif total_ticks > 30:
            final_confidence = min(95, final_confidence + 5)
            reasons.append(f"Buena cantidad de datos: {total_ticks} ticks")
        
        # Ajuste por fase de mercado
        if phase == "consolidation":
            final_confidence = max(35, final_confidence - 10)
            reasons.append("Mercado en consolidación - menor confianza")
        elif phase == "strong_trend":
            final_confidence = min(95, final_confidence + 8)
            reasons.append("Tendencia fuerte - mayor confianza")
        
        final_confidence = int(max(25, min(95, final_confidence)))
        
        return {
            "direction": "ALZA" if direction == 1 else "BAJA",
            "confidence": final_confidence,
            "price": round(metrics.get("current_price", 0.0), 5),
            "tick_count": total_ticks,
            "reasons": reasons,
            "model_type": "INTELLIGENT_RULES",
            "signal_strength": signal_strength if 'signal_strength' in locals() else 0
        }

    def _advanced_fusion(self, mlp_pred, ensemble_pred, rules_pred, metrics):
        """FUSIÓN AVANZADA DE MODELOS"""
        if not mlp_pred and not ensemble_pred:
            return rules_pred
            
        vol = metrics.get("volatility", 0.0)
        phase = metrics.get("market_phase", "neutral")
        total_ticks = metrics.get("total_ticks", 0)
        rsi = metrics.get("rsi_like", 50)
        data_quality = metrics.get("data_quality", 0.5)
        
        # Pesos base adaptativos
        base_mlp_weight = 0.5
        base_ensemble_weight = 0.3
        base_rules_weight = 0.2
        
        # Ajustar pesos basado en condiciones de mercado
        if phase == "consolidation":
            # En consolidación, confiar más en reglas
            mlp_weight = 0.3
            ensemble_weight = 0.2
            rules_weight = 0.5
        elif phase == "strong_trend" and total_ticks > 25:
            # En tendencia fuerte con buenos datos, confiar más en ML
            mlp_weight = 0.6
            ensemble_weight = 0.3
            rules_weight = 0.1
        elif rsi > 70 or rsi < 30:
            # En extremos RSI, menos confianza en MLP
            mlp_weight = 0.4
            ensemble_weight = 0.3
            rules_weight = 0.3
        elif data_quality < 0.7:
            # Calidad de datos baja, más reglas
            mlp_weight = 0.3
            ensemble_weight = 0.2
            rules_weight = 0.5
        else:
            mlp_weight = base_mlp_weight
            ensemble_weight = base_ensemble_weight
            rules_weight = base_rules_weight
            
        # Ajustar por confianza individual
        if mlp_pred and mlp_pred.get("confidence", 0) < 55:
            mlp_weight *= 0.7
        if ensemble_pred and ensemble_pred.get("confidence", 0) < 55:
            ensemble_weight *= 0.7
            
        # Normalizar pesos
        total_weight = mlp_weight + ensemble_weight + rules_weight
        mlp_weight /= total_weight
        ensemble_weight /= total_weight
        rules_weight /= total_weight
        
        # Calcular predicción combinada
        combined_up = 0.0
        total_confidence = 0.0
        
        if mlp_pred:
            mlp_up = mlp_pred.get("prob_up", 0.5)
            combined_up += mlp_up * mlp_weight
            total_confidence += mlp_pred.get("confidence", 50) * mlp_weight
            
        if ensemble_pred:
            ensemble_up = ensemble_pred.get("prob_up", 0.5)
            combined_up += ensemble_up * ensemble_weight
            total_confidence += ensemble_pred.get("confidence", 50) * ensemble_weight
            
        if rules_pred:
            rules_up = 0.8 if rules_pred["direction"] == "ALZA" else 0.2
            combined_up += rules_up * rules_weight
            total_confidence += rules_pred.get("confidence", 50) * rules_weight
        
        direction = "ALZA" if combined_up >= 0.5 else "BAJA"
        fused_confidence = int(total_confidence)
        
        # Razones de la fusión
        reasons = ["Fusión avanzada de modelos:"]
        if mlp_pred:
            reasons.append(f"MLP({mlp_pred.get('confidence',0)}%)")
        if ensemble_pred:
            reasons.append(f"Ensemble({ensemble_pred.get('confidence',0)}%)")
        if rules_pred:
            reasons.append(f"Rules({rules_pred.get('confidence',0)}%)")
            
        reasons.append(f"Pesos: MLP:{mlp_weight:.2f}, Ensemble:{ensemble_weight:.2f}, Rules:{rules_weight:.2f}")
        
        # Ajuste final de confianza por consistencia
        consistent_count = 0
        if mlp_pred and mlp_pred.get("direction") == direction:
            consistent_count += 1
        if ensemble_pred and ensemble_pred.get("direction") == direction:
            consistent_count += 1
        if rules_pred and rules_pred.get("direction") == direction:
            consistent_count += 1
            
        if consistent_count >= 2:
            fused_confidence = min(95, fused_confidence + 10)
            reasons.append("Múltiples modelos consistentes")
        elif consistent_count == 1:
            fused_confidence = max(35, fused_confidence - 10)
            reasons.append("Modelos en conflicto")
        
        fused_confidence = max(30, min(95, fused_confidence))
        
        return {
            "direction": direction,
            "confidence": fused_confidence,
            "price": round(metrics.get("current_price", 0.0), 5),
            "tick_count": metrics.get("total_ticks", 0),
            "reasons": reasons,
            "model_used": "ADVANCED_HYBRID",
            "fusion_weights": {
                "mlp": round(mlp_weight, 2),
                "ensemble": round(ensemble_weight, 2),
                "rules": round(rules_weight, 2)
            }
        }

    def predict_next_candle(self, seconds_remaining_norm=None):
        """PREDICCIÓN MEJORADA - ACTIVACIÓN EN ÚLTIMOS 3-5 SEGUNDOS"""
        metrics = self.analyzer.get_candle_metrics(seconds_remaining_norm=seconds_remaining_norm)
        if not metrics:
            return {
                "direction": "N/A", 
                "confidence": 0,
                "reason": "sin_datos",
                "timestamp": now_iso()
            }
            
        total_ticks = metrics.get("total_ticks", 0)
        data_quality = metrics.get("data_quality", 0.0)
        
        if total_ticks < MIN_TICKS_FOR_PREDICTION:
            return {
                "direction": "N/A",
                "confidence": 0,
                "reason": f"solo_{total_ticks}_ticks",
                "timestamp": now_iso()
            }
        
        features = self.extract_features(metrics).reshape(1, -1)
        mlp_pred = None
        ensemble_pred = None
        
        # PREDICCIÓN MLP PRINCIPAL (solo con suficiente calidad de datos)
        if self.model and self.scaler and total_ticks >= 12 and data_quality > 0.4:
            try:
                Xs = self.scaler.transform(features)
                proba = self.model.predict_proba(Xs)[0]
                up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                
                mlp_confidence = int(max(up_prob, 1 - up_prob) * 100)
                
                # Ajustar confianza MLP basado en calidad de datos
                if data_quality < 0.7:
                    mlp_confidence = int(mlp_confidence * data_quality)
                
                mlp_pred = {
                    "direction": "ALZA" if up_prob >= 0.5 else "BAJA",
                    "prob_up": up_prob,
                    "confidence": mlp_confidence,
                    "model_type": "MLP"
                }
            except Exception as e:
                logging.warning(f"⚠️ MLP predict error: {e}")
                mlp_pred = None
        
        # PREDICCIÓN ENSEMBLE MEJORADA (requiere más datos)
        if total_ticks >= 18 and data_quality > 0.5:
            ensemble_pred = self._ensemble_predict(features)

        # PREDICCIÓN POR REGLAS INTELIGENTES (siempre disponible)
        rules_pred = self._intelligent_rule_based(metrics)
        
        # FUSIÓN AVANZADA DE MODELOS
        if mlp_pred and ensemble_pred and total_ticks >= 20:
            # Máxima capacidad - usar todos los modelos
            final_pred = self._advanced_fusion(mlp_pred, ensemble_pred, rules_pred, metrics)
        elif mlp_pred and total_ticks >= 15:
            # Capacidad media - MLP + reglas
            final_pred = self._advanced_fusion(mlp_pred, None, rules_pred, metrics)
        elif ensemble_pred and total_ticks >= 18:
            # Ensemble + reglas
            final_pred = self._advanced_fusion(None, ensemble_pred, rules_pred, metrics)
        else:
            # Mínima capacidad - solo reglas
            final_pred = rules_pred
            if total_ticks < 15:
                final_pred["reasons"].append("Fusión no disponible - datos insuficientes")
        
        # Agregar metadata adicional para trazabilidad
        final_pred.update({
            "total_ticks": total_ticks,
            "market_phase": metrics.get("market_phase", "unknown"),
            "data_quality": round(data_quality, 2),
            "timestamp": now_iso(),
            "prediction_window": PREDICTION_WINDOW
        })
        
        self.last_prediction = final_pred.copy()
        self.prediction_history.append(final_pred)
        
        return final_pred

# -------------- IQ CONNECTION ROBUSTO --------------
class RobustIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.last_tick_time = None
        self.tick_count = 0
        self.last_price = None
        self.actual_pair = None
        self.connection_attempts = 0
        self.last_connection_time = 0
        
    def connect(self):
        """Conectar a IQ Option con manejo robusto de errores"""
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.warning("❌ Credenciales IQ no configuradas")
                return None
                
            # Evitar reconexiones muy frecuentes
            current_time = time.time()
            if current_time - self.last_connection_time < 30:
                return self.connected
                
            logging.info("🔗 Conectando a IQ Option...")
            self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.iq.connect()
            
            if check:
                self.iq.change_balance("PRACTICE")
                self.connected = True
                self.connection_attempts = 0
                self.last_connection_time = current_time
                logging.info("✅ Conectado exitosamente a IQ Option")
                
                self._find_working_pair()
                return True
            else:
                self.connection_attempts += 1
                logging.warning(f"⚠️ Conexión fallida ({self.connection_attempts}/{MAX_RETRY_CONNECTION}): {reason}")
                
                if self.connection_attempts >= MAX_RETRY_CONNECTION:
                    logging.error("❌ Máximo de intentos de conexión alcanzado")
                    self.connected = False
                
                return False
                
        except Exception as e:
            logging.error(f"❌ Error conexión: {e}")
            self.connected = False
            return False

    def _find_working_pair(self):
        """Encontrar un par que funcione con múltiples intentos"""
        test_pairs = ["EURUSD", "EURUSD-OTC", "GBPUSD", "USDJPY"]
        
        for pair in test_pairs:
            try:
                logging.info(f"🔍 Probando par: {pair}")
                candles = self.iq.get_candles(pair, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self.actual_pair = pair
                        logging.info(f"✅ Par funcional encontrado: {pair} - Precio: {price:.5f}")
                        return
            except Exception as e:
                logging.debug(f"Par {pair} falló: {e}")
        
        self.actual_pair = "EURUSD"
        logging.warning(f"⚠️ Usando par por defecto: {self.actual_pair}")

    def get_realtime_ticks(self):
        """Obtener ticks en tiempo real con múltiples métodos de respaldo"""
        try:
            if not self.connected:
                if not self.connect():
                    return self.last_price

            working_pair = self.actual_pair if self.actual_pair else PAR
            
            # Método 1: Candles normales
            try:
                candles = self.iq.get_candles(working_pair, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"get_candles falló: {e}")

            # Método 2: Candles en tiempo real
            try:
                realtime = self.iq.get_realtime_candles(working_pair, TIMEFRAME)
                if realtime:
                    candle_list = list(realtime.values())
                    if candle_list:
                        latest_candle = candle_list[-1]
                        price = float(latest_candle.get('close', 0))
                        if price > 0:
                            self._record_tick(price)
                            return price
            except Exception as e:
                logging.debug(f"get_realtime_candles falló: {e}")

            # Método 3: Candles con más datos históricos
            try:
                candles = self.iq.get_candles(working_pair, TIMEFRAME, 2, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"get_candles_v2 falló: {e}")

            return self.last_price

        except Exception as e:
            logging.error(f"❌ Error crítico obteniendo ticks: {e}")
            self.connected = False
            return self.last_price

    def _record_tick(self, price):
        """Registrar tick recibido con logging optimizado"""
        self.tick_count += 1
        self.last_tick_time = time.time()
        self.last_price = price
        
        # Logging menos verboso para mejor performance
        if self.tick_count <= 5 or self.tick_count % 50 == 0:
            pair_info = f" ({self.actual_pair})" if self.actual_pair else ""
            logging.info(f"💰 Tick #{self.tick_count}{pair_info}: {price:.5f}")

    def check_connection(self):
        """Verificar estado de conexión"""
        try:
            if self.iq and hasattr(self.iq, 'check_connect'):
                return self.iq.check_connect()
            return False
        except:
            return False

# --------------- Enhanced Adaptive Trainer Loop ---------------
def enhanced_adaptive_trainer_loop(predictor: IntelligentPredictor):
    """Loop de entrenamiento mejorado con manejo robusto"""
    while True:
        try:
            time.sleep(30)  # Espera entre entrenamientos
            
            if not os.path.exists(TRAINING_CSV):
                continue
                
            df = pd.read_csv(TRAINING_CSV)
            current_size = len(df)
            
            if current_size >= BATCH_TRAIN_SIZE:
                logging.info(f"🔁 Entrenamiento mejorado con {current_size} samples...")
                
                X = df[predictor._feature_names()].values
                y = df["label"].values.astype(int)
                
                # Validación de datos
                if len(X) < BATCH_TRAIN_SIZE * 0.8:
                    continue
                    
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                scaler = IncrementalScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate="adaptive",
                    learning_rate_init=LEARNING_RATE,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=42
                )
                
                model.fit(X_train_scaled, y_train)
                val_accuracy = model.score(X_val_scaled, y_val)
                
                if val_accuracy >= 0.55:
                    predictor.model = model
                    predictor.scaler = scaler
                    predictor._save_artifacts()
                    logging.info(f"✅ Modelo actualizado (val_acc: {val_accuracy:.3f})")
                    
                    # También reentrenar modelos ensemble
                    if current_size >= 200:
                        for name, ensemble_model in predictor.ensemble_models.items():
                            try:
                                ensemble_model.fit(X_train_scaled, y_train)
                                logging.info(f"✅ Ensemble model {name} retrained")
                            except Exception as e:
                                logging.warning(f"⚠️ Error retraining ensemble {name}: {e}")
                        
                        predictor._save_ensemble()
                        
        except Exception as e:
            logging.error(f"❌ Error entrenamiento mejorado: {e}")
            time.sleep(60)

# --------------- Sistema de Mantenimiento ---------------
def maintenance_loop():
    """Loop de mantenimiento para tareas periódicas"""
    while True:
        try:
            time.sleep(300)  # Cada 5 minutos
            
            # Limpieza de memoria
            if hasattr(predictor, 'prediction_history') and len(predictor.prediction_history) > 50:
                predictor.prediction_history = deque(
                    list(predictor.prediction_history)[-30:], 
                    maxlen=30
                )
            
            # Verificación de conexión
            if iq_connector and not iq_connector.connected:
                logging.info("🔄 Intentando reconexión...")
                iq_connector.connect()
            
            # Log de estado del sistema
            current_accuracy = performance_stats.get('recent_accuracy', 0)
            logging.info(f"🛠️ Mantenimiento - Precisión: {current_accuracy:.1f}% | Ticks: {iq_connector.tick_count}")
            
        except Exception as e:
            logging.error(f"Error en mantenimiento: {e}")

# --------------- Global State ---------------
iq_connector = RobustIQConnector()
predictor = IntelligentPredictor()

# Estadísticas globales de performance
performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_10': deque(maxlen=10),
    'last_validation': None,
    'system_uptime': time.time()
}

current_prediction = {
    "direction":"N/A",
    "confidence":0,
    "price":0.0,
    "tick_count":0,
    "reasons":["Sistema iniciando..."],
    "timestamp":now_iso(),
    "model_used":"INIT",
    "status": "INITIALIZING"
}

# --------------- Main loop MEJORADO CON ESTABILIDAD ---------------
def professional_tick_analyzer():
    global current_prediction
    
    logging.info("🚀 Delowyss AI V4.0-ESTABLE iniciado - SISTEMA DE APRENDIZAJE AVANZADO")
    
    iq_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time()//TIMEFRAME*TIMEFRAME
    consecutive_errors = 0

    while True:
        try:
            # Obtener tick en tiempo real
            price = iq_connector.get_realtime_ticks()
            
            if price is not None and price > 0:
                # Procesar tick
                predictor.analyzer.add_tick(price)
                consecutive_errors = 0  # Resetear contador de errores
                
                # Actualizar estado básico
                current_prediction.update({
                    "price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE"
                })
            else:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    logging.warning("⚠️ Múltiples errores consecutivos, intentando reconectar...")
                    iq_connector.connect()
                    consecutive_errors = 0
            
            # LÓGICA DE VELAS MEJORADA
            now = time.time()
            current_candle_start = now//TIMEFRAME*TIMEFRAME
            seconds_remaining = TIMEFRAME - (now % TIMEFRAME)
            
            # PREDICCIÓN ACTIVA SOLO EN ÚLTIMOS SEGUNDOS CON SUFICIENTES DATOS
            if seconds_remaining <= PREDICTION_WINDOW and seconds_remaining > 1:
                if predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION:
                    if (time.time() - last_prediction_time) > 2:  # Evitar predicciones muy frecuentes
                        pred = predictor.predict_next_candle(seconds_remaining_norm=seconds_remaining/TIMEFRAME)
                        current_prediction.update(pred)
                        last_prediction_time = time.time()
                        
                        if pred['direction'] != 'N/A':
                            logging.info(f"🎯 PREDICCIÓN: {pred['direction']} | Conf: {pred['confidence']}% | Ticks: {pred['tick_count']} | Modelo: {pred.get('model_used', 'UNKNOWN')}")
            
            # CAMBIO DE VELA CON VALIDACIÓN MEJORADA
            if current_candle_start > last_candle_start:
                closed_metrics = predictor.analyzer.get_candle_metrics()
                if closed_metrics:
                    # ✅ VALIDACIÓN AGREGADA
                    validation_result = predictor.validate_previous_prediction(closed_metrics)
                    if validation_result:
                        performance_stats['last_validation'] = validation_result
                    
                    predictor.on_candle_closed(closed_metrics)
                
                predictor.analyzer.reset_candle()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela iniciada - Analizando ticks...")
                
            # Espera adaptativa para mejor performance
            sleep_time = 0.5 if seconds_remaining > 10 else 0.1
            time.sleep(sleep_time)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            consecutive_errors += 1
            time.sleep(5)  # Espera más larga en errores críticos

# --------------- FastAPI COMPLETO CON ENDPOINTS MEJORADOS ---------------
app = FastAPI(title="Delowyss AI V4.0", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    """Interfaz web mejorada"""
    try:
        training_samples = len(pd.read_csv(TRAINING_CSV)) if os.path.exists(TRAINING_CSV) else 0
    except:
        training_samples = 0
        
    try:  
        perf_rows = len(pd.read_csv(PERF_CSV)) if os.path.exists(PERF_CSV) else 0
    except:
        perf_rows = 0
        
    perf_acc = 0.0
    try:
        if perf_rows>0:
            perf_df = pd.read_csv(PERF_CSV)
            if "correct" in perf_df:
                perf_acc = perf_df["correct"].mean()*100
    except Exception:
        perf_acc = 0.0
        
    try:
        metrics = predictor.analyzer.get_candle_metrics()
        phase = metrics.get("market_phase") if metrics else "n/a"
        patterns = [p for (_,p) in predictor.analyzer.last_patterns] if predictor.analyzer.last_patterns else []
    except:
        phase = "n/a"
        patterns = []
        
    direction = current_prediction.get("direction","N/A")
    color = "#00ff88" if direction=="ALZA" else ("#ff4444" if direction=="BAJA" else "#ffbb33")
    
    actual_pair = iq_connector.actual_pair if iq_connector and iq_connector.actual_pair else PAR
    
    # Información del ensemble
    ensemble_info = ""
    if hasattr(predictor, 'ensemble_weights') and predictor.ensemble_weights:
        ensemble_info = f" | Ensemble: {', '.join([f'{k}:{v:.2f}' for k, v in predictor.ensemble_weights.items()])}"
    
    # Calcular tiempo hasta siguiente vela
    current_time = time.time()
    seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
    
    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width'>
        <title>Delowyss AI V4.0-ESTABLE</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255,255,255,0.05);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .card {{
                background: rgba(255,255,255,0.03);
                padding: 24px;
                border-radius: 16px;
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.2s;
            }}
            .card:hover {{
                transform: translateY(-2px);
                background: rgba(255,255,255,0.05);
            }}
            .prediction-card {{
                border-left: 6px solid {color};
                padding: 24px;
                background: rgba(255,255,255,0.05);
            }}
            .countdown {{
                font-size: 3em;
                font-weight: bold;
                text-align: center;
                margin: 30px 0;
                font-family: 'Courier New', monospace;
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid {color};
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric {{
                background: rgba(255,255,255,0.05);
                padding: 15px;
                border-radius: 12px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 1.8em;
                font-weight: bold;
                color: {color};
            }}
            .reasons-list {{
                list-style: none;
                padding: 0;
            }}
            .reasons-list li {{
                padding: 8px 12px;
                margin: 5px 0;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                border-left: 3px solid {color};
            }}
            .status-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: bold;
                background: {color};
                color: #0f172a;
            }}
            .ensemble-info {{
                background: rgba(255,255,255,0.05);
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-size: 0.9em;
            }}
            .direction-arrow {{
                font-size: 4em;
                margin: 10px 0;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Delowyss Trading AI — V4.0-ESTABLE</h1>
                <p>Sistema avanzado con aprendizaje automático mejorado y máxima estabilidad</p>
                <div class="status-badge">ONLINE</div>
            </div>
            
            <div class="ensemble-info">
                <strong>🎯 Sistema de Aprendizaje Avanzado Activado</strong>{ensemble_info}
            </div>
            
            <div class="countdown" id="countdown">{int(seconds_remaining)}s</div>
            
            <div class="direction-arrow" id="direction-arrow">
                {"⬆️" if direction == "ALZA" else "⬇️" if direction == "BAJA" else "⏸️"}
            </div>
            
            <div class="card prediction-card">
                <h2 style="margin: 0 0 15px 0; display: flex; align-items: center; gap: 10px;">
                    <span>Predicción Actual:</span>
                    <span style="color: {color}; font-size: 1.3em;">{direction}</span>
                    <span style="color: {color}; font-size: 1.1em;">{current_prediction.get('confidence', 0)}% confianza</span>
                </h2>
                
                <div class="metrics-grid">
                    <div class="metric">
                        <div>Precio Actual</div>
                        <div class="metric-value">{current_prediction.get('price', 0):.5f}</div>
                    </div>
                    <div class="metric">
                        <div>Ticks Procesados</div>
                        <div class="metric-value">{current_prediction.get('tick_count', 0)}</div>
                    </div>
                    <div class="metric">
                        <div>Fase del Mercado</div>
                        <div class="metric-value">{phase}</div>
                    </div>
                    <div class="metric">
                        <div>Modelo Usado</div>
                        <div class="metric-value">{current_prediction.get('model_used', 'HYBRID')}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📊 Rendimiento del Sistema</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div>Precisión Total</div>
                        <div class="metric-value">{performance_stats.get('recent_accuracy', 0):.1f}%</div>
                    </div>
                    <div class="metric">
                        <div>Predicciones</div>
                        <div class="metric-value">{performance_stats.get('correct_predictions', 0)}/{performance_stats.get('total_predictions', 0)}</div>
                    </div>
                    <div class="metric">
                        <div>Uptime</div>
                        <div class="metric-value">{int((time.time() - performance_stats.get('system_uptime', time.time())) / 60)}min</div>
                    </div>
                    <div class="metric">
                        <div>Par Activo</div>
                        <div class="metric-value">{actual_pair}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🎯 Razones de la Predicción</h3>
                <ul class="reasons-list" id="reasons-list">
                    {"".join([f"<li>📈 {r}</li>" for r in current_prediction.get('reasons', ['Analizando mercado...'])])}
                </ul>
            </div>

            <div class="card">
                <h3>🔍 Análisis Técnico</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div>Momentum</div>
                        <div class="metric-value">{metrics.get('momentum', 0) if metrics else 0:.2f}</div>
                    </div>
                    <div class="metric">
                        <div>Desequilibrio</div>
                        <div class="metric-value">{metrics.get('pressure_ratio', 0) if metrics else 0:.2f}x</div>
                    </div>
                    <div class="metric">
                        <div>Volatilidad</div>
                        <div class="metric-value">{metrics.get('volatility', 0) if metrics else 0:.2f}</div>
                    </div>
                    <div class="metric">
                        <div>Calidad Datos</div>
                        <div class="metric-value">{metrics.get('data_quality', 0)*100 if metrics else 0:.1f}%</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📈 Validación en Tiempo Real</h3>
                <div id="validation-result" style="margin: 10px 0; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    Esperando primera validación...
                </div>
                <div id="performance-stats" style="font-size: 0.9em; color: #ccc;">
                    Cargando estadísticas...
                </div>
            </div>
        </div>

        <script>
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                
                const countdownEl = document.getElementById('countdown');
                countdownEl.textContent = remaining + 's';
                
                if (remaining <= 5) {{
                    countdownEl.style.borderColor = '#ff4444';
                    countdownEl.style.animation = 'pulse 1s infinite';
                }} else {{
                    countdownEl.style.borderColor = '{color}';
                    countdownEl.style.animation = 'none';
                }}
            }}
            
            function updateData() {{
                fetch('/api/prediction')
                    .then(response => response.json())
                    .then(data => {{
                        const direction = data.direction || 'N/A';
                        const confidence = data.confidence || 0;
                        const price = data.price || 0;
                        const tickCount = data.tick_count || 0;
                        const reasons = data.reasons || [];
                        const modelUsed = data.model_used || 'HYBRID';
                        
                        // Actualizar predicción
                        document.querySelector('.prediction-card h2').innerHTML = 
                            `Predicción Actual: <span style="color:{color}; font-size:1.3em;">${{direction}}</span>
                             <span style="color:{color}; font-size:1.1em;">${{confidence}}% confianza</span>`;
                        
                        document.querySelector('.metric-value:nth-child(1)').textContent = price.toFixed(5);
                        document.querySelector('.metric-value:nth-child(2)').textContent = tickCount;
                        document.querySelector('.metric-value:nth-child(4)').textContent = modelUsed;
                        
                        // Actualizar flecha de dirección
                        const arrowEl = document.getElementById('direction-arrow');
                        arrowEl.innerHTML = direction === 'ALZA' ? '⬆️' : (direction === 'BAJA' ? '⬇️' : '⏸️');
                        
                        // Actualizar razones
                        const reasonsList = document.getElementById('reasons-list');
                        reasonsList.innerHTML = reasons.map(r => `<li>📈 ${{r}}</li>`).join('') || 
                                                '<li>🔄 Analizando datos de mercado...</li>';
                    }})
                    .catch(error => console.error('Error:', error));
                    
                // Actualizar cada 2 segundos
                setTimeout(updateData, 2000);
            }}

            function updateValidation() {{
                fetch('/api/validation')
                    .then(response => response.json())
                    .then(data => {{
                        const validation = data.last_validation;
                        const perf = data.performance;
                        
                        if (validation && validation.timestamp) {{
                            const correct = validation.correct;
                            const color = correct ? '#00ff88' : '#ff4444';
                            const icon = correct ? '✅' : '❌';
                            
                            document.getElementById('validation-result').innerHTML = `
                                <div style="color:${{color}}; font-weight:bold; font-size:1.1em;">
                                    ${{icon}} Predicción: <strong>${{validation.predicted}}</strong> 
                                    | Real: <strong>${{validation.actual}}</strong>
                                </div>
                                <div style="font-size:0.9em; margin-top:8px;">
                                    Cambio: ${{validation.price_change_pips}}pips | 
                                    Confianza: ${{validation.confidence}}% |
                                    Modelo: ${{validation.model_used}}
                                </div>
                                <div style="font-size:0.8em; margin-top:5px; color:#ccc;">
                                    ${{new Date(validation.timestamp).toLocaleTimeString()}}
                                </div>
                            `;
                        }}
                        
                        if (perf) {{
                            const overallColor = perf.overall_accuracy > 60 ? '#00ff88' : 
                                               perf.overall_accuracy > 50 ? '#ffbb33' : '#ff4444';
                                               
                            const recentColor = perf.recent_accuracy > 60 ? '#00ff88' : 
                                              perf.recent_accuracy > 50 ? '#ffbb33' : '#ff4444';
                        
                            document.getElementById('performance-stats').innerHTML = `
                                <strong>Precisión Global:</strong> <span style="color:${{overallColor}}">${{perf.overall_accuracy}}%</span> 
                                | <strong>Reciente:</strong> <span style="color:${{recentColor}}">${{perf.recent_accuracy}}%</span>
                                | <strong>Total:</strong> ${{perf.total_predictions}} predicciones
                            `;
                        }}
                    }})
                    .catch(error => console.error('Error:', error));
                    
                // Actualizar cada 3 segundos
                setTimeout(updateValidation, 3000);
            }}
            
            // Inicializar
            setInterval(updateCountdown, 1000);
            updateData();
            updateValidation();
            updateCountdown();
            
            // CSS para animación de pulso
            const style = document.createElement('style');
            style.textContent = `
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                    100% {{ opacity: 1; }}
                }}
            `;
            document.head.appendChild(style);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# --------------- ENDPOINTS API MEJORADOS ---------------

@app.get("/api/prediction")
def api_prediction():
    """Endpoint para obtener la predicción actual"""
    return JSONResponse(current_prediction)

@app.get("/api/validation")
def api_validation():
    """Endpoint mejorado para validaciones"""
    try:
        global performance_stats
        
        last_validation = performance_stats.get('last_validation', {})
        total = performance_stats.get('total_predictions', 0)
        correct = performance_stats.get('correct_predictions', 0)
        recent_acc = performance_stats.get('recent_accuracy', 0.0)
        
        overall_accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return JSONResponse({
            "last_validation": last_validation,
            "performance": {
                "total_predictions": total,
                "correct_predictions": correct,
                "overall_accuracy": round(overall_accuracy, 1),
                "recent_accuracy": round(recent_acc, 1),
                "last_10_results": list(performance_stats.get('last_10', []))
            },
            "system_status": "active",
            "timestamp": now_iso()
        })
    except Exception as e:
        logging.error(f"Error en /api/validation: {e}")
        return JSONResponse({"error": "Error obteniendo validación"}, status_code=500)

@app.get("/api/performance")
def api_performance():
    """Endpoint de performance histórico"""
    try:
        if os.path.exists(PERF_CSV):
            perf_df = pd.read_csv(PERF_CSV)
            total = len(perf_df)
            if total > 0 and "correct" in perf_df:
                accuracy = perf_df["correct"].mean() * 100
                recent_perf = perf_df.tail(min(20, total))
                recent_accuracy = recent_perf["correct"].mean() * 100 if len(recent_perf) > 0 else 0
                
                # Métricas adicionales
                avg_confidence = perf_df["confidence"].mean() if "confidence" in perf_df else 0
                model_distribution = perf_df["model_used"].value_counts().to_dict() if "model_used" in perf_df else {}
                
                return JSONResponse({
                    "total_predictions": total,
                    "overall_accuracy": round(accuracy, 2),
                    "recent_accuracy": round(recent_accuracy, 2),
                    "average_confidence": round(avg_confidence, 2),
                    "model_distribution": model_distribution,
                    "timestamp": now_iso()
                })
    except Exception as e:
        logging.error("Error /api/performance: %s", e)
    
    return JSONResponse({"error": "No performance data available"})

@app.get("/api/status")
def api_status():
    """Endpoint de estado del sistema"""
    connected = iq_connector.connected if iq_connector else False
    training_samples = len(pd.read_csv(TRAINING_CSV)) if os.path.exists(TRAINING_CSV) else 0
    perf_rows = len(pd.read_csv(PERF_CSV)) if os.path.exists(PERF_CSV) else 0
    actual_pair = iq_connector.actual_pair if iq_connector and iq_connector.actual_pair else PAR
    
    # Información del ensemble
    ensemble_info = {}
    if hasattr(predictor, 'ensemble_weights'):
        ensemble_info = predictor.ensemble_weights
    
    return JSONResponse({
        "status": "online",
        "version": "4.0.0",
        "connected": connected,
        "pair": actual_pair,
        "model_loaded": predictor.model is not None,
        "training_samples": training_samples,
        "performance_records": perf_rows,
        "total_ticks_processed": predictor.analyzer.tick_count,
        "ensemble_weights": ensemble_info,
        "system_uptime": int(time.time() - performance_stats.get('system_uptime', time.time())),
        "timestamp": now_iso()
    })

@app.get("/api/analysis")
def api_analysis():
    """Endpoint de análisis técnico detallado"""
    try:
        metrics = predictor.analyzer.get_candle_metrics()
        if metrics:
            return JSONResponse({
                "technical_analysis": metrics,
                "market_metrics": metrics.get('market_metrics', {}),
                "current_patterns": [p for (_, p) in predictor.analyzer.last_patterns],
                "price_history": predictor.analyzer.get_price_history()[-20:],  # Últimos 20 precios
                "timestamp": now_iso()
            })
    except Exception as e:
        logging.error("Error /api/analysis: %s", e)
    
    return JSONResponse({"error": "No analysis data available"})

@app.get("/health")
def health_check():
    """Endpoint de salud para monitoreo"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": now_iso(),
        "version": "4.0.0"
    })

# --------------- Inicialización para Render ---------------
def start_background_tasks():
    """Iniciar todas las tareas en background"""
    # Thread principal de análisis
    analyzer_thread = threading.Thread(target=professional_tick_analyzer, daemon=True, name="MainAnalyzer")
    analyzer_thread.start()
    logging.info("📊 Background analyzer started")
    
    # Thread de entrenamiento
    trainer_thread = threading.Thread(target=enhanced_adaptive_trainer_loop, args=(predictor,), daemon=True, name="EnhancedTrainer")
    trainer_thread.start()
    logging.info("🧠 Background enhanced trainer started")
    
    # Thread de mantenimiento
    maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True, name="Maintenance")
    maintenance_thread.start()
    logging.info("🛠️ Background maintenance started")

# Iniciar tareas en background
start_background_tasks()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logging.info(f"🌐 Iniciando servidor en {host}:{port}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="info",
        access_log=True,
        timeout_keep_alive=5
    )
