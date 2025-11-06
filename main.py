# main.py - VERSIÓN MEJORADA CON AJUSTES FINOS
"""
Delowyss Trading AI — V4.0-80PCT (Production)
Sistema optimizado para >80% precisión - VERSIÓN MEJORADA CON AJUSTES
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

# ---------------- CONFIG MEJORADA ----------------
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

BATCH_TRAIN_SIZE = int(os.getenv("BATCH_TRAIN_SIZE", "150"))
PARTIAL_FIT_AFTER = int(os.getenv("PARTIAL_FIT_AFTER", "6"))
CONFIDENCE_SAVE_THRESHOLD = float(os.getenv("CONFIDENCE_SAVE_THRESHOLD", "68.0"))

SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "10"))
MAX_TICKS_MEMORY = int(os.getenv("MAX_TICKS_MEMORY", "800"))
MAX_CANDLE_TICKS = int(os.getenv("MAX_CANDLE_TICKS", "400"))

# PARÁMETROS OPTIMIZADOS PARA 80%+ PRECISIÓN
HIGH_ACCURACY_MODE = os.getenv("HIGH_ACCURACY_MODE", "true").lower() == "true"
MIN_HIGH_ACCURACY_CONFIDENCE = int(os.getenv("MIN_HIGH_ACCURACY_CONFIDENCE", "72"))  # Aumentado de 70
MIN_TICKS_HIGH_ACCURACY = int(os.getenv("MIN_TICKS_HIGH_ACCURACY", "25"))  # Más ticks para mayor confiabilidad

# NUEVOS PARÁMETROS DE OPTIMIZACIÓN
CONFIDENCE_CALIBRATION = float(os.getenv("CONFIDENCE_CALIBRATION", "0.95"))  # Calibración conservadora
VOLATILITY_FILTER_MAX = float(os.getenv("VOLATILITY_FILTER_MAX", "2.5"))  # Filtrar alta volatilidad
PRESSURE_RATIO_THRESHOLD = float(os.getenv("PRESSURE_RATIO_THRESHOLD", "1.8"))  # Umbral más estricto

# ---------------- LOGGING MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# ------------------ Incremental Scaler ------------------
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

# ------------------ Analyzer MEJORADO ------------------
class ProductionTickAnalyzer:
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
        self.consecutive_direction = deque(maxlen=3)  # Nuevo: track dirección consecutiva

    def _update_ema_alpha(self, current_volatility):
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
        price = float(price)
        current_time = time.time()
        
        if price <= 0:
            logging.warning(f"Precio inválido ignorado: {price}")
            return None
            
        # MEJORA: Detección mejorada de anomalías
        if self.ticks and len(self.ticks) > 0:
            last_tick = self.ticks[-1]
            last_price = last_tick['price']
            time_gap = current_time - last_tick['timestamp']
            if last_price > 0 and time_gap < 2.0:
                price_change_pct = abs(price - last_price) / last_price
                if price_change_pct > 0.02:
                    logging.warning(f"Anomaly spike ignorado: {last_price:.5f} -> {price:.5f}")
                    return None

        if self.current_candle_open is None:
            self.current_candle_open = self.current_candle_high = self.current_candle_low = price
        else:
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)

        interval = current_time - self.last_tick_time if self.last_tick_time else 0.1
        self.last_tick_time = current_time

        current_volatility = (self.current_candle_high - self.current_candle_low) * 10000
        self._update_ema_alpha(current_volatility)

        if self.smoothed_price is None:
            self.smoothed_price = price
        else:
            self.smoothed_price = (self.ema_alpha * price + (1 - self.ema_alpha) * self.smoothed_price)

        # NUEVO: Track dirección del tick
        if len(self.ticks) > 0:
            prev_price = self.ticks[-1]['price']
            if price > prev_price:
                self.consecutive_direction.append(1)  # ALZA
            elif price < prev_price:
                self.consecutive_direction.append(-1)  # BAJA
            else:
                self.consecutive_direction.append(0)  # NEUTRAL

        tick_data = {
            "timestamp": current_time,
            "price": price,
            "volume": volume,
            "interval": interval,
            "smoothed_price": self.smoothed_price
        }
        self.ticks.append(tick_data)
        self.candle_ticks.append(tick_data)
        self.sequence.append(price)
        self.price_history.append(price)
        self.tick_count += 1

        if len(self.sequence) >= 5:
            pattern = self._detect_micro_pattern()
            if pattern:
                self.last_patterns.appendleft((datetime.utcnow().isoformat(), pattern))
                
        if self.tick_count <= 10 or self.tick_count % 10 == 0:
            logging.info(f"✅ Tick #{self.tick_count} procesado - Precio: {price:.5f}")
        return tick_data

    def get_price_history(self):
        return list(self.price_history)

    def get_consecutive_direction(self):
        """NUEVO: Obtener dirección consecutiva de ticks"""
        if len(self.consecutive_direction) == 0:
            return 0
        return sum(self.consecutive_direction) / len(self.consecutive_direction)

    def _detect_micro_pattern(self):
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
            if pos_diffs >= total * 0.8 and mean_diff > 0.00003:
                return "ramp-up"
            elif neg_diffs >= total * 0.8 and mean_diff < -0.00003:
                return "ramp-down"
            elif std_diff < 0.00002 and abs(mean_diff) < 0.00001:
                return "consolidation"
            elif np.sum(np.abs(np.diff(np.sign(diffs))) > 0) > total * 0.5:
                return "oscillation"
        except Exception:
            pass
        return None

    def get_candle_metrics(self, seconds_remaining_norm: float = None):
        if len(self.candle_ticks) < 2:
            return None
            
        try:
            ticks_array = np.array([(t['price'], t['volume'], t['interval']) for t in self.candle_ticks], dtype=np.float32)
            prices = ticks_array[:, 0]
            volumes = ticks_array[:, 1]
            intervals = ticks_array[:, 2]

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

            # MEJORA: Clasificación de fase de mercado más precisa
            if volatility < 0.5 and direction_ratio < 0.15:
                market_phase = "consolidation"
            elif abs(momentum) > 2.5 and volatility > 1.2 and buy_pressure > 0.65:
                market_phase = "strong_trend_up"
            elif abs(momentum) > 2.5 and volatility > 1.2 and sell_pressure > 0.65:
                market_phase = "strong_trend_down"
            elif abs(momentum) > 1.0:
                market_phase = "weak_trend"
            else:
                market_phase = "neutral"

            # NUEVO: Calidad de datos
            data_quality = min(1.0, (total_ticks / 30) * 0.5 + (volatility / 2.0) * 0.3 + (1 - direction_ratio) * 0.2)

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
                "last_patterns": list(self.last_patterns)[:4],
                "data_quality": data_quality,  # NUEVO
                "consecutive_direction": self.get_consecutive_direction(),  # NUEVO
                "timestamp": time.time()
            }
            if seconds_remaining_norm is not None:
                metrics['seconds_remaining_norm'] = float(seconds_remaining_norm)
            return metrics
        except Exception as e:
            logging.error(f"Error calculando métricas: {e}")
            return None

    def reset_candle(self):
        self.candle_ticks.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.sequence.clear()
        self.tick_count = 0
        self.consecutive_direction.clear()  # NUEVO
        logging.info("🔄 Vela reiniciada")

# ------------------ SISTEMAS OPTIMIZADOS PARA 80%+ PRECISIÓN ------------------
class MetaLearningPredictor:
    def __init__(self):
        self.performance_by_condition = {}
        self.consecutive_predictions = deque(maxlen=5)  # NUEVO: track predicciones consecutivas
        
    def analyze_error_patterns(self, prediction, actual, metrics):
        """Análisis de errores para aprendizaje"""
        if not metrics:
            return
            
        condition_key = self._get_condition_key(metrics)
        
        if condition_key not in self.performance_by_condition:
            self.performance_by_condition[condition_key] = {'correct': 0, 'total': 0}
        
        self.performance_by_condition[condition_key]['total'] += 1
        if prediction['direction'] == actual:
            self.performance_by_condition[condition_key]['correct'] += 1

        # NUEVO: Track predicciones consecutivas
        self.consecutive_predictions.append({
            'direction': prediction['direction'],
            'correct': prediction['direction'] == actual,
            'confidence': prediction.get('confidence', 0),
            'timestamp': datetime.utcnow().isoformat()
        })

    def _get_condition_key(self, metrics):
        """Crear clave única para condiciones"""
        phase = metrics.get('market_phase', 'unknown')
        volatility = 'high' if metrics.get('volatility', 0) > 1.5 else 'low'
        pressure = 'high_pressure' if metrics.get('pressure_ratio', 1) > 1.8 else 'low_pressure'
        return f"{phase}_{volatility}_{pressure}"

    def get_condition_accuracy(self, metrics):
        """Obtener accuracy para condiciones actuales"""
        condition_key = self._get_condition_key(metrics)
        performance = self.performance_by_condition.get(condition_key, {'correct': 0, 'total': 0})
        
        if performance['total'] > 0:
            return performance['correct'] / performance['total']
        return 0.5

    def get_consecutive_bias(self):
        """NUEVO: Detectar sesgo en predicciones consecutivas"""
        if len(self.consecutive_predictions) < 3:
            return 0
            
        directions = [p['direction'] for p in self.consecutive_predictions]
        if all(d == 'ALZA' for d in directions):
            return 1  # Sesgo alcista
        elif all(d == 'BAJA' for d in directions):
            return -1  # Sesgo bajista
        return 0  # Sin sesgo claro

class RealTimeErrorAnalysis:
    def __init__(self):
        self.error_log = deque(maxlen=100)
        self.accuracy_insights = {
            'high_confidence': {'correct': 0, 'total': 0},
            'medium_confidence': {'correct': 0, 'total': 0},
            'low_confidence': {'correct': 0, 'total': 0}
        }
        
    def log_prediction_result(self, prediction, actual_direction, metrics):
        """Registrar resultado de predicción"""
        result = {
            'timestamp': now_iso(),
            'predicted': prediction['direction'],
            'actual': actual_direction,
            'correct': prediction['direction'] == actual_direction,
            'confidence': prediction.get('confidence', 0)
        }
        self.error_log.append(result)

        # NUEVO: Análisis por nivel de confianza
        confidence = prediction.get('confidence', 0)
        if confidence >= 70:
            bucket = 'high_confidence'
        elif confidence >= 60:
            bucket = 'medium_confidence'
        else:
            bucket = 'low_confidence'
            
        self.accuracy_insights[bucket]['total'] += 1
        if result['correct']:
            self.accuracy_insights[bucket]['correct'] += 1

    def get_accuracy_insights(self):
        """Obtener insights de precisión - MEJORADO"""
        total = len(self.error_log)
        if total == 0:
            return {
                "accuracy": 0.0, 
                "total_predictions": 0, 
                "correct_predictions": 0,
                "high_confidence_accuracy": 0.0,
                "medium_confidence_accuracy": 0.0,
                "low_confidence_accuracy": 0.0
            }
        
        correct = sum(1 for entry in self.error_log if entry['correct'])
        accuracy = correct / total
        
        # Calcular accuracy por nivel de confianza
        high_acc = 0.0
        if self.accuracy_insights['high_confidence']['total'] > 0:
            high_acc = self.accuracy_insights['high_confidence']['correct'] / self.accuracy_insights['high_confidence']['total']
            
        medium_acc = 0.0
        if self.accuracy_insights['medium_confidence']['total'] > 0:
            medium_acc = self.accuracy_insights['medium_confidence']['correct'] / self.accuracy_insights['medium_confidence']['total']
            
        low_acc = 0.0
        if self.accuracy_insights['low_confidence']['total'] > 0:
            low_acc = self.accuracy_insights['low_confidence']['correct'] / self.accuracy_insights['low_confidence']['total']
        
        return {
            "accuracy": accuracy,
            "total_predictions": total,
            "correct_predictions": correct,
            "high_confidence_accuracy": high_acc,
            "medium_confidence_accuracy": medium_acc,
            "low_confidence_accuracy": low_acc
        }

# ------------------ Predictor OPTIMIZADO ------------------
class ProductionPredictor:
    def __init__(self):
        self.analyzer = ProductionTickAnalyzer()
        self.model = None
        self.scaler = None
        self.prev_candle_metrics = None
        self.partial_buffer = []
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent': deque(maxlen=50)
        }
        self.last_prediction = None
        
        # SISTEMAS OPTIMIZADOS
        self.meta_learner = MetaLearningPredictor()
        self.error_analyzer = RealTimeErrorAnalysis()
        self.high_accuracy_mode = HIGH_ACCURACY_MODE
        self.consecutive_high_confidence = 0  # NUEVO: track alta confianza consecutiva
        
        self._initialize_system()
        self._ensure_files()

    def _feature_names(self):
        return [
            "buy_pressure", "sell_pressure", "pressure_ratio", "momentum",
            "volatility", "up_ticks", "down_ticks", "total_ticks",
            "volume_trend", "price_change", "tick_speed", "direction_ratio",
            "seconds_remaining_norm", "data_quality", "consecutive_direction"  # NUEVAS features
        ]

    def _ensure_files(self):
        try:
            if not os.path.exists(TRAINING_CSV):
                pd.DataFrame(columns=self._feature_names() + ["label", "timestamp"]).to_csv(TRAINING_CSV, index=False)
            if not os.path.exists(PERF_CSV):
                pd.DataFrame(columns=["timestamp", "prediction", "actual", "correct", "confidence", "model_used"]).to_csv(PERF_CSV, index=False)
        except Exception as e:
            logging.error("Error initializing files: %s", e)

    def _initialize_system(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                logging.info("✅ Modelo ML existente cargado")
            else:
                self._initialize_new_model()
        except Exception as e:
            logging.error(f"❌ Error cargando modelo: {e}")
            self._initialize_new_model()

    def _initialize_new_model(self):
        try:
            self.scaler = IncrementalScaler()
            self.model = MLPClassifier(
                hidden_layer_sizes=(64,32),
                activation="relu",
                solver="adam",
                learning_rate="adaptive",
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

    def _save_artifacts(self):
        try:
            if self.model and self.scaler:
                joblib.dump(self.model, MODEL_PATH)
                joblib.dump(self.scaler, SCALER_PATH)
                logging.info("💾 Modelo guardado")
        except Exception as e:
            logging.error(f"❌ Error guardando artifacts: {e}")

    def extract_features(self, metrics):
        try:
            features = [safe_float(metrics.get(k,0.0)) for k in self._feature_names()]
            return np.array(features, dtype=np.float32)
        except Exception:
            return np.zeros(len(self._feature_names()), dtype=np.float32)

    def append_sample_if_confident(self, metrics, label, confidence):
        try:
            # MEJORA: Guardar solo muestras de calidad
            if confidence < CONFIDENCE_SAVE_THRESHOLD:
                return
                
            data_quality = metrics.get('data_quality', 0.5)
            if data_quality < 0.4:  # NUEVO: filtrar por calidad
                return
                
            row = {k: metrics.get(k,0.0) for k in self._feature_names()}
            row["label"] = int(label)
            row["timestamp"] = datetime.utcnow().isoformat()
            pd.DataFrame([row]).to_csv(TRAINING_CSV, mode="a", header=False, index=False)
            self.partial_buffer.append((row,label))
            logging.info(f"💾 Sample guardado - label={label} conf={confidence}% calidad={data_quality:.2f} buffer={len(self.partial_buffer)}")
            if len(self.partial_buffer) >= PARTIAL_FIT_AFTER:
                self._perform_partial_fit()
        except Exception as e:
            logging.error(f"❌ Error append sample: {e}")

    def _perform_partial_fit(self):
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
            self._save_artifacts()
            logging.info(f"🧠 Partial fit completado con {len(X_new)} samples")
            self.partial_buffer.clear()
        except Exception as e:
            logging.error(f"❌ Error partial fit: {e}")
            self.partial_buffer.clear()

    # PREDICCIÓN DE ALTA PRECISIÓN OPTIMIZADA
    def _high_accuracy_prediction(self, metrics):
        """Predicción de alta precisión optimizada"""
        if not metrics or metrics.get('total_ticks', 0) < MIN_TICKS_HIGH_ACCURACY:
            return None
            
        # MEJORA: Filtros más estrictos
        volatility = metrics.get('volatility', 0)
        if volatility > VOLATILITY_FILTER_MAX:
            return None
            
        data_quality = metrics.get('data_quality', 0.5)
        if data_quality < 0.6:
            return None

        # 1. Obtener predicción base
        base_pred = self.predict_next_candle_base(metrics)
        
        # 2. Aplicar calibración mejorada
        calibrated_conf = self._calibrate_confidence(base_pred, metrics)
        
        # 3. Verificar criterios optimizados
        if (calibrated_conf >= MIN_HIGH_ACCURACY_CONFIDENCE and 
            self._meets_high_accuracy_criteria(metrics, base_pred)):
            
            # NUEVO: Track alta confianza consecutiva
            if calibrated_conf >= 80:
                self.consecutive_high_confidence += 1
            else:
                self.consecutive_high_confidence = 0
                
            return {
                'direction': base_pred['direction'],
                'confidence': calibrated_conf,
                'model_used': 'HIGH_ACCURACY',
                'reasons': base_pred.get('reasons', []) + ['Calibración alta precisión'],
                'high_accuracy': True,
                'consecutive_high_confidence': self.consecutive_high_confidence
            }
        
        return None

    def _meets_high_accuracy_criteria(self, metrics, prediction):
        """NUEVO: Criterios más estrictos para alta precisión"""
        pressure_ratio = metrics.get('pressure_ratio', 1.0)
        momentum = abs(metrics.get('momentum', 0))
        market_phase = metrics.get('market_phase', 'neutral')
        
        criteria_met = 0
        
        # Criterio 1: Presión de compra/venta definida
        if pressure_ratio > PRESSURE_RATIO_THRESHOLD or pressure_ratio < (1/PRESSURE_RATIO_THRESHOLD):
            criteria_met += 1
            
        # Criterio 2: Momentum significativo
        if momentum > 1.5:
            criteria_met += 1
            
        # Criterio 3: Fase de mercado favorable
        if market_phase in ['strong_trend_up', 'strong_trend_down']:
            criteria_met += 1
        elif market_phase == 'consolidation' and metrics.get('volatility', 0) < 0.8:
            criteria_met += 1
            
        # Criterio 4: Calidad de datos
        if metrics.get('data_quality', 0) > 0.7:
            criteria_met += 1
            
        return criteria_met >= 3  # Requerir al menos 3 criterios

    def predict_next_candle_base(self, metrics):
        """Predicción base optimizada"""
        features = self.extract_features(metrics).reshape(1, -1)
        mlp_pred = None
        
        if self.model and self.scaler and metrics.get('total_ticks', 0) >= 10:
            try:
                Xs = self.scaler.transform(features)
                proba = self.model.predict_proba(Xs)[0]
                up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                
                mlp_confidence = int(max(up_prob, 1 - up_prob) * 100)
                
                # MEJORA: Calibración de confianza MLP
                if abs(up_prob - 0.5) < 0.15:
                    mlp_confidence = max(40, int(mlp_confidence * 0.8))
                
                mlp_pred = {
                    "direction": "ALZA" if up_prob >= 0.5 else "BAJA",
                    "prob_up": up_prob,
                    "confidence": mlp_confidence,
                    "model_type": "MLP"
                }
            except Exception as e:
                logging.warning(f"⚠️ MLP predict error: {e}")
                mlp_pred = None
        
        rules_pred = self._rule_based(metrics)
        
        if mlp_pred and metrics.get('total_ticks', 0) >= 12:
            final_pred = self._fuse(mlp_pred, rules_pred, metrics)
        else:
            final_pred = rules_pred
        
        return final_pred

    def _calibrate_confidence(self, prediction, metrics):
        """Calibración de confianza mejorada"""
        base_confidence = prediction['confidence']
        
        # Factor de accuracy histórica
        historical_accuracy = self.meta_learner.get_condition_accuracy(metrics)
        accuracy_factor = max(0.7, min(1.3, historical_accuracy / 0.5))
        
        # Factor de calidad de datos
        data_quality = metrics.get('data_quality', 0.5)
        quality_factor = 0.6 + (data_quality * 0.4)
        
        # NUEVO: Factor de sesgo consecutivo
        bias_factor = 1.0
        consecutive_bias = self.meta_learner.get_consecutive_bias()
        if consecutive_bias != 0 and prediction['direction'] == ('ALZA' if consecutive_bias > 0 else 'BAJA'):
            bias_factor = 0.9  # Reducir confianza en predicciones con sesgo
        
        # NUEVO: Factor de volatilidad
        volatility = metrics.get('volatility', 0)
        volatility_factor = 1.0
        if volatility > 2.0:
            volatility_factor = 0.8
        elif volatility < 0.3:
            volatility_factor = 0.9
            
        calibrated = base_confidence * accuracy_factor * quality_factor * bias_factor * volatility_factor * CONFIDENCE_CALIBRATION
        return int(max(25, min(95, calibrated)))

    def _assess_data_quality(self, metrics):
        """Evaluación de calidad de datos mejorada"""
        total_ticks = metrics.get('total_ticks', 0)
        volatility = metrics.get('volatility', 0)
        direction_ratio = metrics.get('direction_ratio', 0)
        
        quality_score = 0.0
        
        # Peso de ticks (40%)
        if total_ticks >= 30:
            quality_score += 0.4
        elif total_ticks >= 20:
            quality_score += 0.3
        elif total_ticks >= 10:
            quality_score += 0.2
        else:
            quality_score += 0.1
            
        # Peso de volatilidad (30%)
        if 0.5 <= volatility <= 2.0:
            quality_score += 0.3
        elif 0.3 <= volatility <= 2.5:
            quality_score += 0.2
        else:
            quality_score += 0.1
            
        # Peso de consistencia (30%)
        if direction_ratio < 0.3:  # Menos cambios de dirección = más consistente
            quality_score += 0.3
        elif direction_ratio < 0.5:
            quality_score += 0.2
        else:
            quality_score += 0.1
            
        return min(1.0, quality_score)

    def validate_previous_prediction(self, current_candle_metrics):
        """Valida si la última predicción fue correcta - MEJORADO"""
        if not self.last_prediction:
            return None
            
        try:
            if self.prev_candle_metrics is None:
                return None
                
            prev_close = float(self.prev_candle_metrics["current_price"])
            current_close = float(current_candle_metrics["current_price"])
            
            actual_direction = "ALZA" if current_close > prev_close else "BAJA"
            predicted_direction = self.last_prediction.get("direction", "N/A")
            
            correct = (actual_direction == predicted_direction)
            confidence = self.last_prediction.get("confidence", 0)
            
            price_change = (current_close - prev_close) * 10000

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
                "high_accuracy": self.last_prediction.get("high_accuracy", False)
            }
            
            # ANÁLISIS DE ERRORES MEJORADO
            self.error_analyzer.log_prediction_result(self.last_prediction, actual_direction, self.prev_candle_metrics)
            self.meta_learner.analyze_error_patterns(self.last_prediction, actual_direction, self.prev_candle_metrics)
            
            # ACTUALIZAR ESTADÍSTICAS
            self._update_global_performance_stats(correct, result)
            
            status = "✅ CORRECTA" if correct else "❌ ERRÓNEA"
            accuracy_icon = "🎯" if self.last_prediction.get("high_accuracy", False) else ""
            logging.info(f"🎯 VALIDACIÓN: {status} {accuracy_icon} | Pred: {predicted_direction} | Real: {actual_direction} | Conf: {confidence}% | Change: {price_change:.1f}pips")
            
            # NUEVO: Log de insights de precisión periódico
            if performance_stats['total_predictions'] % 10 == 0:
                insights = self.error_analyzer.get_accuracy_insights()
                logging.info(f"📊 PRECISIÓN DEL SISTEMA: Global: {insights['accuracy']:.1%} | Total: {insights['total_predictions']}")
                if insights['high_confidence_accuracy'] > 0:
                    logging.info(f"    📈 Precisión ALTA confianza: {insights['high_confidence_accuracy']:.1%}")
                if insights['medium_confidence_accuracy'] > 0:
                    logging.info(f"    📈 Precisión MEDIA confianza: {insights['medium_confidence_accuracy']:.1%}")
                if insights['low_confidence_accuracy'] > 0:
                    logging.info(f"    📈 Precisión BAJA confianza: {insights['low_confidence_accuracy']:.1%}")
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error validando predicción: {e}")
            return None

    def _update_global_performance_stats(self, correct, validation_result):
        """Actualiza estadísticas globales"""
        global performance_stats
        
        performance_stats['total_predictions'] += 1
        performance_stats['correct_predictions'] += 1 if correct else 0
        performance_stats['last_10'].append(1 if correct else 0)
        performance_stats['last_validation'] = validation_result
        
        if performance_stats['last_10']:
            recent_correct = sum(performance_stats['last_10'])
            performance_stats['recent_accuracy'] = (recent_correct / len(performance_stats['last_10'])) * 100

    def on_candle_closed(self, closed_metrics):
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
        try:
            correct = ((pred.get("direction")=="ALZA" and actual_label==1) or (pred.get("direction")=="BAJA" and actual_label==0))
            rec = {
                "timestamp": now_iso(), 
                "prediction": pred.get("direction"), 
                "actual": "ALZA" if actual_label==1 else "BAJA",
                "correct": correct, 
                "confidence": pred.get("confidence",0), 
                "model_used": pred.get("model_used","HYBRID"),
                "high_accuracy": pred.get("high_accuracy", False)
            }
            pd.DataFrame([rec]).to_csv(PERF_CSV, mode="a", header=False, index=False)
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['correct_predictions'] += int(correct)
            self.performance_stats['recent'].append(int(correct))
        except Exception as e:
            logging.error(f"❌ Error recording performance: {e}")

    def _rule_based(self, metrics):
        """SISTEMA DE REGLAS OPTIMIZADO"""
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
        data_quality = metrics.get("data_quality", 0.5)
        
        # REGLAS OPTIMIZADAS
        if pr > PRESSURE_RATIO_THRESHOLD:
            signals.append(1)
            confidences.append(min(80, 50 + int((pr - 2.0) * 15)))
            reasons.append(f"Presión compra fuerte {pr:.1f}x")
        elif pr > 1.6:
            signals.append(1)
            confidences.append(min(65, 40 + int((pr - 1.5) * 20)))
            reasons.append(f"Presión compra {pr:.1f}x")
        elif pr < (1/PRESSURE_RATIO_THRESHOLD):
            signals.append(0)
            confidences.append(min(80, 50 + int((0.5 - pr) * 15)))
            reasons.append(f"Presión venta fuerte {pr:.1f}x")
        elif pr < 0.65:
            signals.append(0)
            confidences.append(min(65, 40 + int((0.7 - pr) * 20)))
            reasons.append(f"Presión venta {pr:.1f}x")
        
        if mom > 2.0:
            signals.append(1)
            confidences.append(min(75, 45 + int(min(mom, 8) * 3)))
            reasons.append(f"Momento alcista {mom:.1f}pips")
        elif mom < -2.0:
            signals.append(0)
            confidences.append(min(75, 45 + int(min(abs(mom), 8) * 3)))
            reasons.append(f"Momento bajista {mom:.1f}pips")
        elif abs(mom) > 0.8:
            direction = 1 if mom > 0 else 0
            signals.append(direction)
            confidences.append(55)
            reasons.append(f"Momento leve {mom:.1f}pips")
        
        if bp > 0.70:
            signals.append(1)
            confidences.append(70)
            reasons.append(f"Dominio compra {bp:.0%}")
        elif sp > 0.70:
            signals.append(0)
            confidences.append(70)
            reasons.append(f"Dominio venta {sp:.0%}")
        
        # NUEVO: Reglas basadas en fase de mercado
        if phase == "strong_trend_up":
            signals.append(1)
            confidences.append(75)
            reasons.append("Tendencia alcista fuerte")
        elif phase == "strong_trend_down":
            signals.append(0)
            confidences.append(75)
            reasons.append("Tendencia bajista fuerte")
        elif phase == "consolidation" and vol < 0.8:
            # En consolidación, ser más conservador
            if signals:
                confidences = [int(c * 0.8) for c in confidences]
            reasons.append("Mercado en consolidación - confianza reducida")
        
        # DECISIÓN FINAL OPTIMIZADA
        if signals:
            avg_confidence = int(sum(confidences) / len(confidences))
            
            buy_signals = sum(1 for s in signals if s == 1)
            sell_signals = sum(1 for s in signals if s == 0)
            
            if buy_signals > 0 and sell_signals == 0:
                direction = 1
                final_confidence = min(90, avg_confidence + 10)
                reasons.append("Señales alcistas consistentes")
            elif sell_signals > 0 and buy_signals == 0:
                direction = 0
                final_confidence = min(90, avg_confidence + 10)
                reasons.append("Señales bajistas consistentes")
            else:
                direction = 1 if buy_signals > sell_signals else 0
                final_confidence = max(40, avg_confidence - 15)
                reasons.append("Señales mixtas")
        else:
            price_change = metrics.get("price_change", 0)
            if abs(price_change) > 0.5:
                direction = 1 if price_change > 0 else 0
                final_confidence = 45 + int(min(abs(price_change), 3) * 8)
                reasons.append(f"Basado en movimiento: {price_change:.1f}pips")
            else:
                direction = 1 if metrics.get("price_change", 0) > 0 else 0
                final_confidence = 40
                reasons.append("Mercado lateral")
        
        # AJUSTES FINALES OPTIMIZADOS
        if total_ticks < 8:
            final_confidence = max(30, final_confidence - 15)
            reasons.append(f"Pocos datos: {total_ticks} ticks")
        elif total_ticks > 25:
            final_confidence = min(95, final_confidence + 5)
            
        # Ajuste por calidad de datos
        if data_quality < 0.4:
            final_confidence = max(30, int(final_confidence * 0.8))
            reasons.append("Calidad de datos baja")
        elif data_quality > 0.8:
            final_confidence = min(95, int(final_confidence * 1.1))
            reasons.append("Calidad de datos alta")

        final_confidence = int(max(25, min(95, final_confidence)))
        
        return {
            "direction": "ALZA" if direction == 1 else "BAJA",
            "confidence": final_confidence,
            "price": round(metrics.get("current_price", 0.0), 5),
            "tick_count": total_ticks,
            "reasons": reasons,
            "model_type": "RULES"
        }

    def _fuse(self, mlp_pred, rules_pred, metrics):
        """FUSIÓN OPTIMIZADA"""
        if not mlp_pred:
            return rules_pred
            
        vol = metrics.get("volatility", 0.0)
        phase = metrics.get("market_phase", "neutral")
        total_ticks = metrics.get("total_ticks", 0)
        data_quality = metrics.get("data_quality", 0.5)
        
        base_mlp_weight = 0.6
        
        # MEJORA: Peso dinámico basado en múltiples factores
        if phase == "consolidation":
            mlp_weight = 0.4
        elif phase in ["strong_trend_up", "strong_trend_down"] and total_ticks > 20:
            mlp_weight = 0.7
        elif data_quality > 0.7:
            mlp_weight = base_mlp_weight * 1.1
        elif data_quality < 0.4:
            mlp_weight = base_mlp_weight * 0.7
        else:
            mlp_weight = base_mlp_weight
            
        mlp_confidence = mlp_pred.get("confidence", 50)
        if mlp_confidence < 55:
            mlp_weight *= 0.7
            
        rules_weight = 1.0 - mlp_weight
        
        rules_up = 0.8 if rules_pred["direction"] == "ALZA" else 0.2
        combined_up = mlp_pred["prob_up"] * mlp_weight + rules_up * rules_weight
        
        direction = "ALZA" if combined_up >= 0.5 else "BAJA"
        
        mlp_conf = mlp_pred.get("confidence", 50)
        rules_conf = rules_pred.get("confidence", 50)
        
        fused_confidence = int(mlp_conf * mlp_weight + rules_conf * rules_weight)
        
        if mlp_pred["direction"] != rules_pred["direction"]:
            fused_confidence = max(35, int(fused_confidence * 0.7))
            reasons = [f"Conflicto: MLP({mlp_pred.get('prob_up', 0):.2f}) vs Rules"]
        else:
            reasons = [f"Consenso: MLP {mlp_conf}% + Rules {rules_conf}%"]
        
        reasons.extend(rules_pred.get("reasons", []))
        
        # MEJORA: Ajuste final por calidad de datos
        if data_quality < 0.5:
            fused_confidence = max(30, int(fused_confidence * 0.9))
            reasons.append("Fusión: calidad de datos moderada")
        
        fused_confidence = max(30, min(95, fused_confidence))
        
        return {
            "direction": direction,
            "confidence": fused_confidence,
            "price": round(metrics.get("current_price", 0.0), 5),
            "tick_count": metrics.get("total_ticks", 0),
            "reasons": reasons,
            "model_used": "HYBRID",
            "mlp_confidence": mlp_conf,
            "rules_confidence": rules_conf
        }

    def predict_next_candle(self, seconds_remaining_norm=None):
        """PREDICCIÓN OPTIMIZADA CON MODO ALTA PRECISIÓN"""
        metrics = self.analyzer.get_candle_metrics(seconds_remaining_norm=seconds_remaining_norm)
        if not metrics:
            return {
                "direction": "N/A", 
                "confidence": 0,
                "reason": "sin_datos",
                "timestamp": now_iso()
            }
            
        total_ticks = metrics.get("total_ticks", 0)
        
        if total_ticks < 5:
            return {
                "direction": "N/A",
                "confidence": 0,
                "reason": f"solo_{total_ticks}_ticks",
                "timestamp": now_iso()
            }
        
        # PREDICCIÓN DE ALTA PRECISIÓN OPTIMIZADA
        if self.high_accuracy_mode and seconds_remaining_norm and seconds_remaining_norm <= 0.1:
            high_acc_pred = self._high_accuracy_prediction(metrics)
            if high_acc_pred:
                accuracy_icon = "🎯" if high_acc_pred.get('consecutive_high_confidence', 0) > 1 else ""
                logging.info(f"🎯 PREDICCIÓN ALTA PRECISIÓN {accuracy_icon}: {high_acc_pred['direction']} | Confianza: {high_acc_pred['confidence']}% | Ticks: {total_ticks}")
                self.last_prediction = high_acc_pred.copy()
                return high_acc_pred
        
        # PREDICCIÓN ESTÁNDAR OPTIMIZADA
        features = self.extract_features(metrics).reshape(1, -1)
        mlp_pred = None
        
        if self.model and self.scaler and total_ticks >= 10:
            try:
                Xs = self.scaler.transform(features)
                proba = self.model.predict_proba(Xs)[0]
                up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                
                mlp_confidence = int(max(up_prob, 1 - up_prob) * 100)
                
                if abs(up_prob - 0.5) < 0.15:
                    mlp_confidence = max(40, int(mlp_confidence * 0.8))
                
                mlp_pred = {
                    "direction": "ALZA" if up_prob >= 0.5 else "BAJA",
                    "prob_up": up_prob,
                    "confidence": mlp_confidence,
                    "model_type": "MLP"
                }
            except Exception as e:
                logging.warning(f"⚠️ MLP predict error: {e}")
                mlp_pred = None
        
        rules_pred = self._rule_based(metrics)
        
        if mlp_pred and total_ticks >= 12:
            final_pred = self._fuse(mlp_pred, rules_pred, metrics)
        else:
            final_pred = rules_pred
        
        self.last_prediction = final_pred.copy()
        
        # Log de predicción estándar
        if final_pred['confidence'] >= 70:
            confidence_level = "ALTA"
        elif final_pred['confidence'] >= 60:
            confidence_level = "MEDIA"
        else:
            confidence_level = "BAJA"
            
        logging.info(f"🎯 PREDICCIÓN VELA SIGUIENTE: {final_pred['direction']} | Confianza: {final_pred['confidence']}% ({confidence_level}) | Garrapatas: {total_ticks}")
        
        return final_pred

# -------------- IQ CONNECTION OPTIMIZADA --------------
class IQOptionConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.last_tick_time = None
        self.tick_count = 0
        self.last_price = None
        self.actual_pair = None
        
    def connect(self):
        """Conectar a IQ Option"""
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.warning("❌ Credenciales IQ no configuradas")
                return None
                
            logging.info("🔗 Conectando a IQ Option...")
            self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.iq.connect()
            
            if check:
                self.iq.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conectado exitosamente a IQ Option")
                
                self._find_working_pair()
                
                return self.iq
            else:
                logging.warning(f"⚠️ Conexión fallida: {reason}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Error conexión: {e}")
            return None

    def _find_working_pair(self):
        """Encontrar un par que funcione"""
        test_pairs = ["EURUSD", "EURUSD-OTC", "EURUSD"]
        
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
        """Obtener ticks en tiempo real"""
        try:
            if not self.connected or not self.iq:
                return None

            working_pair = self.actual_pair if self.actual_pair else "EURUSD"
            
            try:
                candles = self.iq.get_candles(working_pair, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"get_candles falló: {e}")

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

            if self.last_price:
                return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo ticks: {e}")
            
        return None

    def _record_tick(self, price):
        """Registrar tick recibido"""
        self.tick_count += 1
        self.last_tick_time = time.time()
        self.last_price = price
        
        if self.tick_count <= 10 or self.tick_count % 5 == 0:
            pair_info = f" ({self.actual_pair})" if self.actual_pair else ""
            logging.info(f"💰 Tick #{self.tick_count}{pair_info}: {price:.5f}")

    def check_connection(self):
        """Verificar conexión"""
        try:
            if self.iq and hasattr(self.iq, 'check_connect'):
                return self.iq.check_connect()
            return False
        except:
            return False

# --------------- Adaptive Trainer Loop ---------------
def adaptive_trainer_loop(predictor: ProductionPredictor):
    """Loop de entrenamiento optimizado"""
    while True:
        try:
            time.sleep(30)
            if not os.path.exists(TRAINING_CSV):
                continue
                
            df = pd.read_csv(TRAINING_CSV)
            current_size = len(df)
            
            if current_size >= BATCH_TRAIN_SIZE:
                logging.info(f"🔁 Entrenamiento con {current_size} samples...")
                
                X = df[predictor._feature_names()].values
                y = df["label"].values.astype(int)
                
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
                        
        except Exception as e:
            logging.error(f"❌ Error entrenamiento: {e}")
            time.sleep(60)

# --------------- Global State ---------------
iq_connector = IQOptionConnector()
predictor = ProductionPredictor()

# Estadísticas globales de performance
performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_10': deque(maxlen=10),
    'last_validation': None
}

current_prediction = {
    "direction":"N/A",
    "confidence":0,
    "price":0.0,
    "tick_count":0,
    "reasons":[],
    "timestamp":now_iso(),
    "model_used":"INIT"
}

# --------------- Main loop OPTIMIZADO ---------------
def professional_tick_analyzer():
    global current_prediction
    
    logging.info("🚀 Delowyss AI V4.0-80PCT iniciado - MODO ALTA PRECISIÓN ACTIVADO")
    logging.info("🎯 OBJETIVO: 80%+ precisión con calibración optimizada")
    
    last_prediction_time = 0
    last_candle_start = time.time()//TIMEFRAME*TIMEFRAME

    iq_connector.connect()

    while True:
        try:
            # Obtener tick en tiempo real
            price = iq_connector.get_realtime_ticks()
            
            if price is not None and price > 0:
                # Procesar tick
                predictor.analyzer.add_tick(price)
                
                # Actualizar estado básico
                current_prediction.update({
                    "price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "CONECTADO"
                })
                
            # LÓGICA DE VELAS
            now = time.time()
            current_candle_start = now//TIMEFRAME*TIMEFRAME
            seconds_remaining = TIMEFRAME - (now % TIMEFRAME)
            
            # PREDICCIÓN ACTIVA SOLO EN ÚLTIMOS SEGUNDOS
            if seconds_remaining <= PREDICTION_WINDOW and seconds_remaining > 1:
                if predictor.analyzer.tick_count >= 8:
                    if (time.time() - last_prediction_time) > 2:
                        pred = predictor.predict_next_candle(seconds_remaining_norm=seconds_remaining/TIMEFRAME)
                        current_prediction.update(pred)
                        last_prediction_time = time.time()
            
            # CAMBIO DE VELA CON VALIDACIÓN
            if current_candle_start > last_candle_start:
                closed_metrics = predictor.analyzer.get_candle_metrics()
                if closed_metrics:
                    validation_result = predictor.validate_previous_prediction(closed_metrics)
                    if validation_result:
                        performance_stats['last_validation'] = validation_result
                    
                    predictor.on_candle_closed(closed_metrics)
                
                predictor.analyzer.reset_candle()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela iniciada - Analizando ticks...")
                
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(2)

# --------------- FastAPI OPTIMIZADO ---------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
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
        data_quality = metrics.get("data_quality", 0) if metrics else 0
    except:
        phase = "n/a"
        patterns = []
        data_quality = 0
        
    direction = current_prediction.get("direction","N/A")
    color = "#00ff88" if direction=="ALZA" else ("#ff4444" if direction=="BAJA" else "#ffbb33")
    
    actual_pair = iq_connector.actual_pair if iq_connector and iq_connector.actual_pair else PAR
    
    # INSIGHTS MEJORADOS
    high_accuracy_info = ""
    if predictor.high_accuracy_mode:
        insights = predictor.error_analyzer.get_accuracy_insights()
        if insights and 'accuracy' in insights:
            high_accuracy_info = f" | Precisión Global: {insights['accuracy']:.1%}"
            if insights['high_confidence_accuracy'] > 0:
                high_accuracy_info += f" | Alta Conf: {insights['high_confidence_accuracy']:.1%}"
    
    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width'>
        <title>Delowyss AI V4.0-80PCT OPTIMIZADO</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background: #0f172a;
                color: #fff;
                padding: 18px;
                margin: 0;
            }}
            .card {{
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255,255,255,0.03);
                padding: 20px;
                border-radius: 12px;
            }}
            .prediction-card {{
                border-left: 6px solid {color};
                padding: 20px;
                margin: 15px 0;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                text-align: center;
            }}
            .validation-card {{
                border-left: 4px solid #ffbb33;
                padding: 15px;
                margin: 15px 0;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 12px;
                margin: 20px 0;
            }}
            .metric-cell {{
                background: rgba(255,255,255,0.03);
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }}
            .accuracy-high {{ color: #00ff88; }}
            .accuracy-medium {{ color: #ffbb33; }}
            .accuracy-low {{ color: #ff4444; }}
            
            .countdown {{
                font-size: 3em;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
            }}
            .countdown.critical {{
                color: #ff4444;
                animation: pulse 1s infinite;
            }}
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            
            .direction-arrow {{
                font-size: 4em;
                margin: 10px 0;
            }}
            .arrow-up {{ color: #00ff88; }}
            .arrow-down {{ color: #ff4444; }}
            
            .status-connected {{ color: #00ff88; }}
            .status-disconnected {{ color: #ff4444; }}
            
            .high-accuracy-badge {{
                background: linear-gradient(45deg, #ff0080, #00ff88);
                padding: 5px 10px;
                border-radius: 20px;
                font-weight: bold;
                margin-left: 10px;
            }}
            
            .quality-indicator {{
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }}
            .quality-high {{ background: #00ff88; }}
            .quality-medium {{ background: #ffbb33; }}
            .quality-low {{ background: #ff4444; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🤖 Delowyss Trading AI — V4.0-80PCT OPTIMIZADO {'<span class="high-accuracy-badge">ALTA PRECISIÓN</span>' if predictor.high_accuracy_mode else ''}</h1>
            <p>Par: <strong>{actual_pair}</strong> • UTC: <span id="current-time">{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</span>
            • Estado: <span id="connection-status" class="{'status-connected' if iq_connector.connected else 'status-disconnected'}">{'CONECTADO' if iq_connector.connected else 'DISCONNECTED'}</span>
            {high_accuracy_info}
            </p>
            
            <div class="countdown" id="countdown">--</div>
            
            <div class="direction-arrow" id="direction-arrow">
                {"⬆️" if direction == "ALZA" else "⬇️" if direction == "BAJA" else "⏸️"}
            </div>
            
            <div class="prediction-card">
                <h2 style="color:{color}; margin:0">{direction} — {current_prediction.get('confidence',0)}% de confianza {'🎯' if current_prediction.get('high_accuracy') else ''}</h2>
                <p>Modelo: {current_prediction.get('model_used','HYBRID')} • Precio: {current_prediction.get('price',0)}</p>
                <p>Fase: <strong>{phase}</strong> • Calidad: <span class="quality-indicator {'quality-high' if data_quality > 0.7 else 'quality-medium' if data_quality > 0.4 else 'quality-low'}"></span>{data_quality:.0%} • Patrones: {', '.join(patterns[:3]) if patterns else 'ninguno'}</p>
                <p>Marcas evaluadas: <strong>{current_prediction.get('tick_count',0)}</strong></p>
            </div>

            <div class="validation-card">
                <h3>📊 Validación en Tiempo Real</h3>
                <div id="validation-result" style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 5px;">
                    Esperando primera validación...
                </div>
                <div id="performance-stats" style="font-size: 0.9em; color: #ccc;">
                    Cargando estadísticas...
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-cell">
                    <strong>Ejemplos de entrenamiento</strong>
                    <div>{training_samples}</div>
                </div>
                <div class="metric-cell">
                    <strong>Filas de rendimiento</strong>
                    <div>{perf_rows}</div>
                </div>
                <div class="metric-cell">
                    <strong>Precisión histórica</strong>
                    <div class="{'accuracy-high' if perf_acc > 60 else 'accuracy-medium' if perf_acc > 50 else 'accuracy-low'}">
                        {perf_acc:.1f}%
                    </div>
                </div>
                <div class="metric-cell">
                    <strong>Timbres actuales</strong>
                    <div>{current_prediction.get('tick_count',0)}</div>
                </div>
            </div>
            
            <div class="metric-cell">
                <h3>Razones de predicción</h3>
                <ul id="reasons-list">
                    {"".join([f"<li>✅ {r}</li>" for r in current_prediction.get('reasons',[])]) if current_prediction.get('reasons') else "<li>🔄 Analizando datos de mercado...</li>"}
                </ul>
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
                    countdownEl.classList.add('critical');
                }} else {{
                    countdownEl.classList.remove('critical');
                }}
                
                document.getElementById('current-time').textContent = 
                    now.toISOString().replace('T', ' ').substr(0, 19);
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
                        const highAccuracy = data.high_accuracy || false;
                        
                        document.querySelector('.prediction-card h2').textContent = 
                            `${{direction}} — ${{confidence}}% de confianza ${{highAccuracy ? '🎯' : ''}}`;
                        document.querySelector('.prediction-card p:nth-child(2)').innerHTML = 
                            `Modelo: ${{modelUsed}} • Precio: ${{price.toFixed(5)}}`;
                        document.querySelector('.prediction-card p:nth-child(4)').innerHTML = 
                            `Marcas evaluadas: <strong>${{tickCount}}</strong>`;
                            
                        const arrowEl = document.getElementById('direction-arrow');
                        arrowEl.innerHTML = direction === 'ALZA' ? '⬆️' : (direction === 'BAJA' ? '⬇️' : '⏸️');
                        
                        const color = direction === 'ALZA' ? '#00ff88' : (direction === 'BAJA' ? '#ff4444' : '#ffbb33');
                        document.querySelector('.prediction-card').style.borderLeftColor = color;
                        document.querySelector('.prediction-card h2').style.color = color;
                        
                        const reasonsList = document.getElementById('reasons-list');
                        reasonsList.innerHTML = reasons.map(r => `<li>✅ ${{r}}</li>`).join('') || 
                                                '<li>🔄 Analizando datos de mercado...</li>';
                    }})
                    .catch(error => console.error('Error:', error));
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
                            const highAccuracyIcon = validation.high_accuracy ? '🎯' : '';
                            
                            document.getElementById('validation-result').innerHTML = `
                                <div style="color:${{color}}; font-weight:bold;">
                                    ${{icon}} ${{highAccuracyIcon}} Predicción: <strong>${{validation.predicted}}</strong> 
                                    | Real: <strong>${{validation.actual}}</strong>
                                </div>
                                <div style="font-size:0.9em; margin-top:5px;">
                                    Cambio: ${{validation.price_change_pips}}pips | 
                                    Confianza: ${{validation.confidence}}% |
                                    Modelo: ${{validation.model_used}}
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
            }}
            
            setInterval(updateCountdown, 1000);
            setInterval(updateData, 2000);
            setInterval(updateValidation, 3000);
            updateCountdown();
            updateData();
            updateValidation();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/validation")
def api_validation():
    """Endpoint para validaciones"""
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
                "recent_accuracy": round(recent_acc, 1)
            },
            "timestamp": now_iso()
        })
    except Exception as e:
        logging.error(f"Error en /api/validation: {e}")
        return JSONResponse({"error": "Error obteniendo validación"})

@app.get("/api/status")
def api_status():
    connected = iq_connector.connected if iq_connector else False
    training_samples = len(pd.read_csv(TRAINING_CSV)) if os.path.exists(TRAINING_CSV) else 0
    perf_rows = len(pd.read_csv(PERF_CSV)) if os.path.exists(PERF_CSV) else 0
    actual_pair = iq_connector.actual_pair if iq_connector and iq_connector.actual_pair else PAR
    
    accuracy_insights = predictor.error_analyzer.get_accuracy_insights()
    
    return JSONResponse({
        "status": "online",
        "connected": connected,
        "pair": actual_pair,
        "model_loaded": predictor.model is not None,
        "training_samples": training_samples,
        "perf_rows": perf_rows,
        "total_ticks_processed": predictor.analyzer.tick_count,
        "high_accuracy_mode": predictor.high_accuracy_mode,
        "accuracy_insights": accuracy_insights,
        "timestamp": now_iso()
    })

# --------------- Inicialización para Render ---------------
def start_background_tasks():
    """Iniciar todas las tareas en background"""
    analyzer_thread = threading.Thread(target=professional_tick_analyzer, daemon=True)
    analyzer_thread.start()
    logging.info("📊 Background analyzer started")
    
    trainer_thread = threading.Thread(target=adaptive_trainer_loop, args=(predictor,), daemon=True)
    trainer_thread.start()
    logging.info("🧠 Background trainer started")

start_background_tasks()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
