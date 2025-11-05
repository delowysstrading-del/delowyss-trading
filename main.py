# main.py
"""
Delowyss Trading AI — V3.8-Full (Production)
Assistant-only (no autotrading). Analiza vela actual tick-by-tick y predice la siguiente 3-5s antes del cierre.
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from iqoptionapi.stable_api import IQ_Option
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = int(os.getenv("TIMEFRAME", "60"))
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "3"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "v38")
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

# ---------------- LOGGING ----------------
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

# ------------------ Analyzer ------------------
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

            if volatility < 0.5 and direction_ratio < 0.15:
                market_phase = "consolidation"
            elif abs(momentum) > 2.5 and volatility > 1.2:
                market_phase = "strong_trend"
            elif abs(momentum) > 1.0:
                market_phase = "weak_trend"
            else:
                market_phase = "neutral"

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
        logging.info("🔄 Vela reiniciada")

# ------------------ Predictor CON VALIDACIÓN ------------------
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
        self._initialize_system()
        self._ensure_files()

    def _feature_names(self):
        return [
            "buy_pressure", "sell_pressure", "pressure_ratio", "momentum",
            "volatility", "up_ticks", "down_ticks", "total_ticks",
            "volume_trend", "price_change", "tick_speed", "direction_ratio",
            "seconds_remaining_norm"
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
            if confidence < CONFIDENCE_SAVE_THRESHOLD:
                return
            row = {k: metrics.get(k,0.0) for k in self._feature_names()}
            row["label"] = int(label)
            row["timestamp"] = datetime.utcnow().isoformat()
            pd.DataFrame([row]).to_csv(TRAINING_CSV, mode="a", header=False, index=False)
            self.partial_buffer.append((row,label))
            logging.info(f"💾 Sample guardado - label={label} conf={confidence}% buffer={len(self.partial_buffer)}")
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

    def validate_previous_prediction(self, current_candle_metrics):
        """Valida si la última predicción fue correcta - MANTIENE ORIGINALIDAD"""
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
                "reasons": self.last_prediction.get("reasons", [])
            }
            
            status = "✅ CORRECTA" if correct else "❌ ERRÓNEA"
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
        performance_stats['correct_predictions'] += 1 if correct else 0
        performance_stats['last_10'].append(1 if correct else 0)
        performance_stats['last_validation'] = validation_result
        
        if performance_stats['last_10']:
            recent_correct = sum(performance_stats['last_10'])
            performance_stats['recent_accuracy'] = (recent_correct / len(performance_stats['last_10'])) * 100
        
        if performance_stats['total_predictions'] % 5 == 0:
            overall_acc = (performance_stats['correct_predictions'] / performance_stats['total_predictions'] * 100)
            logging.info(f"📊 PERFORMANCE ACUMULADA: Global: {overall_acc:.1f}% | Reciente: {performance_stats['recent_accuracy']:.1f}% | Total: {performance_stats['total_predictions']}")

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
                "model_used": pred.get("model_used","HYBRID")
            }
            pd.DataFrame([rec]).to_csv(PERF_CSV, mode="a", header=False, index=False)
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['correct_predictions'] += int(correct)
            self.performance_stats['recent'].append(int(correct))
            logging.info(f"📊 Performance registrada - Correcto: {correct}")
        except Exception as e:
            logging.error(f"❌ Error recording performance: {e}")

    def _rule_based(self, metrics):
        """SISTEMA DE REGLAS MEJORADO - CONFIANZA EN ENTEROS"""
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
        
        # 1. PRESSURE RATIO - SEÑAL FUERTE
        if pr > 2.2:
            signals.append(1)
            confidences.append(min(80, 50 + int((pr - 2.0) * 15)))
            reasons.append(f"Presión compra fuerte {pr:.1f}x")
        elif pr > 1.6:
            signals.append(1)
            confidences.append(min(65, 40 + int((pr - 1.5) * 20)))
            reasons.append(f"Presión compra {pr:.1f}x")
        elif pr < 0.45:
            signals.append(0)
            confidences.append(min(80, 50 + int((0.5 - pr) * 15)))
            reasons.append(f"Presión venta fuerte {pr:.1f}x")
        elif pr < 0.65:
            signals.append(0)
            confidences.append(min(65, 40 + int((0.7 - pr) * 20)))
            reasons.append(f"Presión venta {pr:.1f}x")
        
        # 2. MOMENTUM - SEÑAL MEDIA
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
        
        # 3. BUY/SELL PRESSURE - SEÑAL DIRECTA
        if bp > 0.70:
            signals.append(1)
            confidences.append(70)
            reasons.append(f"Dominio compra {bp:.0%}")
        elif sp > 0.70:
            signals.append(0)
            confidences.append(70)
            reasons.append(f"Dominio venta {sp:.0%}")
        
        # 4. VOLATILIDAD + FASE MERCADO
        if vol > 6.0:
            if phase == "strong_trend" and abs(mom) > 1.5:
                direction = 1 if mom > 0 else 0
                signals.append(direction)
                confidences.append(min(80, 60 + int(vol)))
                reasons.append(f"Tendencia volátil {vol:.1f}pips")
        
        # DECISIÓN FINAL CON CONFIANZA EN ENTEROS
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
                
                if abs(buy_signals - sell_signals) <= 1:
                    final_confidence = max(35, final_confidence - 10)
                    reasons.append("Alta indecisión")
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
        
        if total_ticks < 8:
            final_confidence = max(30, final_confidence - 15)
            reasons.append(f"Pocos datos: {total_ticks} ticks")
        elif total_ticks > 25:
            final_confidence = min(95, final_confidence + 5)
        
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
        """FUSIÓN MEJORADA - CONFIANZA EN ENTEROS"""
        if not mlp_pred:
            return rules_pred
            
        vol = metrics.get("volatility", 0.0)
        phase = metrics.get("market_phase", "neutral")
        total_ticks = metrics.get("total_ticks", 0)
        
        base_mlp_weight = 0.6
        
        if phase == "consolidation":
            mlp_weight = 0.4
        elif phase == "strong_trend" and total_ticks > 20:
            mlp_weight = 0.7
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
        
        if total_ticks < 5:
            return {
                "direction": "N/A",
                "confidence": 0,
                "reason": f"solo_{total_ticks}_ticks",
                "timestamp": now_iso()
            }
        
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
            if total_ticks < 12 and mlp_pred:
                final_pred["reasons"].append("Fusión no disponible - pocos datos")
        
        self.last_prediction = final_pred.copy()
        return final_pred

# -------------- IQ CONNECTION COMPLETA --------------
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
        test_pairs = [
            "EURUSD",
            "EURUSD-OTC", 
            "EURUSD",
        ]
        
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
    """Loop de entrenamiento"""
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

# --------------- Main loop CON VALIDACIÓN ---------------
def professional_tick_analyzer():
    global current_prediction
    
    logging.info("🚀 Delowyss AI V3.8 iniciado - PREDICCIÓN 3-5s ANTES DEL CIERRE")
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
                
            # LÓGICA DE VELAS MEJORADA
            now = time.time()
            current_candle_start = now//TIMEFRAME*TIMEFRAME
            seconds_remaining = TIMEFRAME - (now % TIMEFRAME)
            
            # PREDICCIÓN ACTIVA SOLO EN ÚLTIMOS 3-5 SEGUNDOS
            if seconds_remaining <= PREDICTION_WINDOW and seconds_remaining > 1:
                if predictor.analyzer.tick_count >= 8:
                    if (time.time() - last_prediction_time) > 2:
                        pred = predictor.predict_next_candle(seconds_remaining_norm=seconds_remaining/TIMEFRAME)
                        current_prediction.update(pred)
                        last_prediction_time = time.time()
                        
                        logging.info("🎯 PREDICCIÓN VELA SIGUIENTE: %s | Confianza: %s%% | Ticks: %s", 
                                   pred.get("direction"), 
                                   pred.get("confidence"),
                                   pred.get("tick_count", 0))
            
            # CAMBIO DE VELA CON VALIDACIÓN
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
                
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(2)

# --------------- FastAPI COMPLETO CON VALIDACIÓN ---------------
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
    except:
        phase = "n/a"
        patterns = []
        
    direction = current_prediction.get("direction","N/A")
    color = "#00ff88" if direction=="ALZA" else ("#ff4444" if direction=="BAJA" else "#ffbb33")
    
    actual_pair = iq_connector.actual_pair if iq_connector and iq_connector.actual_pair else PAR
    
    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width'>
        <title>Delowyss AI V3.8</title>
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
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🤖 Delowyss Trading AI — V3.8</h1>
            <p>Par: <strong>{actual_pair}</strong> • UTC: <span id="current-time">{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</span>
            • Estado: <span id="connection-status" class="{'status-connected' if iq_connector.connected else 'status-disconnected'}">{'CONECTADO' if iq_connector.connected else 'DISCONNECTED'}</span>
            </p>
            
            <div class="countdown" id="countdown">--</div>
            
            <div class="direction-arrow" id="direction-arrow">
                {"⬆️" if direction == "ALZA" else "⬇️" if direction == "BAJA" else "⏸️"}
            </div>
            
            <div class="prediction-card">
                <h2 style="color:{color}; margin:0">{direction} — {current_prediction.get('confidence',0)}% de confianza</h2>
                <p>Modelo: {current_prediction.get('model_used','HYBRID')} • Precio: {current_prediction.get('price',0)}</p>
                <p>Fase: <strong>{phase}</strong> • Patrones: {', '.join(patterns[:3]) if patterns else 'ninguno'}</p>
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
                        
                        document.querySelector('.prediction-card h2').textContent = 
                            `${{direction}} — ${{confidence}}% de confianza`;
                        document.querySelector('.prediction-card p:nth-child(2)').innerHTML = 
                            `Modelo: ${{data.model_used || 'HYBRID'}} • Precio: ${{price.toFixed(5)}}`;
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
                            
                            document.getElementById('validation-result').innerHTML = `
                                <div style="color:${{color}}; font-weight:bold;">
                                    ${{icon}} Predicción: <strong>${{validation.predicted}}</strong> 
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
    """Endpoint para validaciones - MANTIENE ESTILO ORIGINAL"""
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
            "timestamp": now_iso()
        })
    except Exception as e:
        logging.error(f"Error en /api/validation: {e}")
        return JSONResponse({"error": "Error obteniendo validación"})

@app.get("/api/prediction_history")
def api_prediction_history():
    """Historial de predicciones - USA ARCHIVO EXISTENTE"""
    try:
        if os.path.exists(PERF_CSV):
            df = pd.read_csv(PERF_CSV)
            if not df.empty:
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp', ascending=False)
                recent = df.head(20).to_dict('records')
                
                if 'correct' in df.columns:
                    hist_accuracy = df['correct'].mean() * 100
                else:
                    hist_accuracy = 0.0
                    
                return JSONResponse({
                    "recent_predictions": recent,
                    "historical_accuracy": round(hist_accuracy, 1),
                    "total_historical": len(df)
                })
    except Exception as e:
        logging.error(f"Error reading prediction history: {e}")
    
    return JSONResponse({"recent_predictions": [], "historical_accuracy": 0.0})

@app.get("/api/status")
def api_status():
    connected = iq_connector.connected if iq_connector else False
    training_samples = len(pd.read_csv(TRAINING_CSV)) if os.path.exists(TRAINING_CSV) else 0
    perf_rows = len(pd.read_csv(PERF_CSV)) if os.path.exists(PERF_CSV) else 0
    actual_pair = iq_connector.actual_pair if iq_connector and iq_connector.actual_pair else PAR
    
    return JSONResponse({
        "status": "online",
        "connected": connected,
        "pair": actual_pair,
        "model_loaded": predictor.model is not None,
        "training_samples": training_samples,
        "perf_rows": perf_rows,
        "total_ticks_processed": predictor.analyzer.tick_count,
        "timestamp": now_iso()
    })

@app.get("/api/performance")
def api_performance():
    try:
        if os.path.exists(PERF_CSV):
            perf_df = pd.read_csv(PERF_CSV)
            total = len(perf_df)
            if total > 0 and "correct" in perf_df:
                accuracy = perf_df["correct"].mean() * 100
                recent_perf = perf_df.tail(min(20, total))
                recent_accuracy = recent_perf["correct"].mean() * 100 if len(recent_perf) > 0 else 0
                
                return JSONResponse({
                    "total_predictions": total,
                    "overall_accuracy": round(accuracy, 2),
                    "recent_accuracy": round(recent_accuracy, 2),
                    "confidence_avg": round(perf_df["confidence"].mean(), 2) if "confidence" in perf_df else 0
                })
    except Exception as e:
        logging.error("Error /api/performance: %s", e)
    
    return JSONResponse({"error": "No performance data available"})

@app.get("/api/patterns")
def api_patterns():
    try:
        metrics = predictor.analyzer.get_candle_metrics()
        if metrics:
            return JSONResponse({
                "market_phase": metrics.get("market_phase", "unknown"),
                "current_patterns": [p for (_, p) in predictor.analyzer.last_patterns],
                "volatility": metrics.get("volatility", 0),
                "momentum": metrics.get("momentum", 0),
                "price_history": predictor.analyzer.get_price_history(),
                "total_ticks": predictor.analyzer.tick_count
            })
    except Exception as e:
        logging.error("Error /api/patterns: %s", e)
    
    return JSONResponse({"error": "No pattern data available"})

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
