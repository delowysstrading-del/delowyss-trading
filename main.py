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
from fastapi.staticfiles import StaticFiles

from iqoptionapi.stable_api import IQ_Option
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD-OTC")
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

LOG_FILE = os.getenv("LOG_FILE", "delowyss_v38.log")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
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
                
        if self.tick_count % 10 == 0:
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
        if len(self.candle_ticks) < 5:
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

# ------------------ Predictor ------------------
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
            logging.info(f"💾 Sample guardado - label={label} conf={confidence:.1f}% buffer={len(self.partial_buffer)}")
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
                "confidence": pred.get("confidence",0.0), 
                "model_used": pred.get("model_used","HYBRID")
            }
            pd.DataFrame([rec]).to_csv(PERF_CSV, mode="a", header=False, index=False)
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['correct_predictions'] += int(correct)
            self.performance_stats['recent'].append(int(correct))
            logging.info(f"📊 Performance registrada - Correcto: {correct}")
        except Exception as e:
            logging.error(f"❌ Error recording performance: {e}")

    def predict_next_candle(self, seconds_remaining_norm=None):
        metrics = self.analyzer.get_candle_metrics(seconds_remaining_norm=seconds_remaining_norm)
        if not metrics:
            return {"direction":"N/A","confidence":0.0,"reason":"insufficient_ticks","timestamp":now_iso()}
        features = self.extract_features(metrics).reshape(1,-1)
        mlp_pred = None
        if self.model and self.scaler:
            try:
                Xs = self.scaler.transform(features)
                proba = self.model.predict_proba(Xs)[0]
                up_prob = float(proba[1]) if len(proba)>1 else float(proba[0])
                mlp_pred = {
                    "direction":"ALZA" if up_prob>=0.5 else "BAJA",
                    "prob_up":up_prob,
                    "confidence":round(max(up_prob,1-up_prob)*100,2),
                    "model_type":"MLP"
                }
            except Exception as e:
                logging.warning(f"⚠️ MLP predict error: {e}")
                mlp_pred = None
        rules = self._rule_based(metrics)
        final = self._fuse(mlp_pred, rules, metrics)
        self.last_prediction = final.copy()
        return final

    def _rule_based(self, metrics):
        signals=[]
        confs=[]
        pr=metrics.get("pressure_ratio",1.0)
        mom=metrics.get("momentum",0.0)
        bp=metrics.get("buy_pressure",0.5)
        vol=metrics.get("volatility",0.0)
        ts=metrics.get("tick_speed",0.0)
        
        if pr>1.8: 
            signals.append(1)
            confs.append(min(85,(pr-1)*25))
        elif pr<0.6: 
            signals.append(0)
            confs.append(min(85,(1-pr)*25))
        
        if mom>1.0: 
            signals.append(1)
            confs.append(min(75,abs(mom)*12))
        elif mom<-1.0: 
            signals.append(0)
            confs.append(min(75,abs(mom)*12))
        
        if bp>0.66: 
            signals.append(1)
            confs.append(65)
        elif bp<0.34: 
            signals.append(0)
            confs.append(65)
        
        if ts>6: 
            signals.append(1 if mom>0 else 0)
            confs.append(55)
            
        if signals:
            dir_ = 1 if sum(signals)/len(signals)>0.5 else 0
            avg = sum(confs)/len(confs)
            if len(set(signals))==1: 
                avg=min(95,avg*1.2)
        else:
            dir_=1 if metrics.get("price_change",0)>0 else 0
            avg=45
            
        reasons=[]
        if pr>1.5: 
            reasons.append(f"Buy pressure {pr:.2f}x")
        if pr<0.7: 
            reasons.append(f"Sell pressure {pr:.2f}x")
        if abs(mom)>1.0: 
            reasons.append("Momentum strong")
            
        return {
            "direction":"ALZA" if dir_==1 else "BAJA",
            "confidence":round(avg,2),
            "price":round(metrics.get("current_price",0.0),5),
            "tick_count":metrics.get("total_ticks",0),
            "reasons":reasons,
            "model_type":"RULES"
        }

    def _fuse(self, mlp_pred, rules_pred, metrics):
        if not mlp_pred:
            res = rules_pred.copy()
            res["model_used"]="RULES"
            return res
            
        vol = metrics.get("volatility",0.0)
        phase = metrics.get("market_phase","neutral")
        mlp_weight=0.75
        
        if phase=="consolidation": 
            mlp_weight=0.6
        elif phase=="strong_trend" and vol>2.0: 
            mlp_weight=0.8
            
        if vol>3.0: 
            mlp_weight=max(0.5, mlp_weight-0.1)
            
        rules_weight = 1.0-mlp_weight
        rules_up = 0.8 if rules_pred["direction"]=="ALZA" else 0.2
        combined_up = mlp_pred["prob_up"]*mlp_weight + rules_up*rules_weight
        
        direction = "ALZA" if combined_up>=0.5 else "BAJA"
        confidence = round(max(combined_up,1-combined_up)*100,2)
        
        reasons = [f"Fusion MLP({mlp_pred.get('prob_up'):.3f})+Rules"] + rules_pred.get("reasons",[])
        
        return {
            "direction":direction,
            "confidence":confidence,
            "price":round(metrics.get("current_price",0.0),5),
            "tick_count":metrics.get("total_ticks",0),
            "reasons":reasons,
            "model_used":"HYBRID",
            "mlp_confidence":mlp_pred.get("confidence"),
            "rules_confidence":rules_pred.get("confidence")
        }

# -------------- IQ CONNECTION CON PRECIOS REALES --------------
class IQOptionConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.last_tick_time = None
        self.tick_count = 0
        self.last_price = None
        
    def connect(self):
        """Conectar a IQ Option con múltiples intentos"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if not IQ_EMAIL or not IQ_PASSWORD:
                    logging.warning("❌ Credenciales IQ no configuradas en env")
                    return None
                    
                logging.info(f"🔗 Conectando a IQ Option (intento {attempt + 1}/{max_attempts})...")
                self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
                check, reason = self.iq.connect()
                
                if check:
                    self.iq.change_balance("PRACTICE")
                    self.connected = True
                    logging.info("✅ Conectado exitosamente a IQ Option")
                    
                    # Verificar que el par esté disponible
                    self._check_pair_availability()
                    return self.iq
                else:
                    logging.warning(f"⚠️ Conexión IQ fallida: {reason}")
                    
            except Exception as e:
                logging.error(f"❌ Error conexión IQ: {e}")
                
            time.sleep(2)
            
        return None

    def _check_pair_availability(self):
        """Verificar si el par está disponible"""
        try:
            # Obtener todos los activos disponibles
            all_assets = self.iq.get_all_open_time()
            
            # Buscar en diferentes categorías
            for asset_type in ["digital", "turbo", "binary", "forex", "crypto"]:
                if asset_type in all_assets and PAR in all_assets[asset_type]:
                    is_open = all_assets[asset_type][PAR]["open"]
                    logging.info(f"✅ Par {PAR} encontrado en {asset_type} - Abierto: {is_open}")
                    return True
            
            logging.warning(f"⚠️ Par {PAR} no encontrado en activos disponibles")
            return False
            
        except Exception as e:
            logging.error(f"❌ Error verificando par: {e}")
            return False

    def get_realtime_ticks(self):
        """Obtener ticks en tiempo real - MÉTODOS REALES"""
        try:
            if not self.connected or not self.iq:
                return None

            price = None
            
            # MÉTODO 1: get_candles (MÁS CONFIABLE)
            try:
                candles = self.iq.get_candles(PAR, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"Método candles falló: {e}")

            # MÉTODO 2: get_realtime_candles
            try:
                realtime_candles = self.iq.get_realtime_candles(PAR, TIMEFRAME)
                if realtime_candles:
                    candle_list = list(realtime_candles.values())
                    if candle_list:
                        latest_candle = candle_list[-1]
                        price_fields = ['close', 'max', 'min', 'open']
                        for field in price_fields:
                            if field in latest_candle and latest_candle[field]:
                                price = float(latest_candle[field])
                                if price > 0:
                                    self._record_tick(price)
                                    return price
            except Exception as e:
                logging.debug(f"Método realtime candles falló: {e}")

            # MÉTODO 3: get_digital_spot (PARA OTC)
            try:
                # Para OTC/digital
                spot_data = self.iq.get_digital_spot(PAR, 1)
                if spot_data and 'profit' in spot_data:
                    price = float(spot_data['profit'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"Método digital spot falló: {e}")

            # MÉTODO 4: get_api_option_init_all
            try:
                all_data = self.iq.get_api_option_init_all()
                if all_data and 'digital' in all_data and PAR in all_data['digital']:
                    pair_data = all_data['digital'][PAR]
                    if 'current_spot' in pair_data:
                        price = float(pair_data['current_spot'])
                        if price > 0:
                            self._record_tick(price)
                            return price
            except Exception as e:
                logging.debug(f"Método api option falló: {e}")

            # MÉTODO 5: Último precio conocido
            if self.last_price:
                logging.info(f"🔄 Usando último precio conocido: {self.last_price:.5f}")
                return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo ticks: {e}")
            
        return None

    def _record_tick(self, price):
        """Registrar tick recibido"""
        self.tick_count += 1
        self.last_tick_time = time.time()
        self.last_price = price
        
        # Log cada tick para debugging
        if self.tick_count <= 10 or self.tick_count % 20 == 0:
            logging.info(f"🎯 Tick #{self.tick_count} - Precio REAL: {price:.5f}")

    def check_connection(self):
        """Verificar si la conexión sigue activa"""
        try:
            if self.iq and hasattr(self.iq, 'check_connect'):
                return self.iq.check_connect()
            return False
        except:
            return False

# --------------- Adaptive Trainer Loop ---------------
def adaptive_trainer_loop(predictor: ProductionPredictor):
    """Loop de entrenamiento batch con validación y early stopping"""
    last_training_size = 0
    best_validation_accuracy = 0.55
    patience_counter = 0
    max_patience = 3
    
    while True:
        try:
            time.sleep(30)
            
            if not os.path.exists(TRAINING_CSV):
                continue
                
            df = pd.read_csv(TRAINING_CSV)
            current_size = len(df)
            
            if current_size >= BATCH_TRAIN_SIZE and current_size > last_training_size + 25:
                logging.info(f"🔁 Entrenamiento batch con {current_size} samples...")
                
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
                
                train_accuracy = model.score(X_train_scaled, y_train)
                val_accuracy = model.score(X_val_scaled, y_val)
                
                logging.info(f"📊 Train: {train_accuracy:.3f}, Val: {val_accuracy:.3f}")
                
                if val_accuracy >= best_validation_accuracy:
                    predictor.model = model
                    predictor.scaler = scaler
                    predictor._save_artifacts()
                    
                    last_training_size = current_size
                    best_validation_accuracy = max(best_validation_accuracy, val_accuracy)
                    patience_counter = 0
                    
                    logging.info(f"✅ Modelo actualizado (val_acc: {val_accuracy:.3f})")
                else:
                    patience_counter += 1
                    logging.warning(f"⚠️ Modelo no mejoró. Paciencia: {patience_counter}/{max_patience}")
                    
                    if patience_counter >= max_patience:
                        logging.info("🔄 Reiniciando paciencia...")
                        patience_counter = 0
                        best_validation_accuracy = max(0.50, best_validation_accuracy * 0.95)
                        
        except Exception as e:
            logging.error(f"❌ Error entrenamiento: {e}")
            time.sleep(60)

# --------------- Initialization ---------------
iq_connector = IQOptionConnector()
IQ = iq_connector.connect()

predictor = ProductionPredictor()
current_prediction = {
    "direction":"N/A",
    "confidence":0.0,
    "price":0.0,
    "tick_count":0,
    "reasons":[],
    "timestamp":now_iso(),
    "model_used":"INIT"
}

# --------------- Main loop CON PRECIOS REALES ---------------
def professional_tick_analyzer():
    logging.info("🚀 Delowyss AI V3.8 iniciado (assistant-only) - PRECIOS REALES")
    last_prediction_time = 0
    last_candle_start = time.time()//TIMEFRAME*TIMEFRAME
    consecutive_failures = 0
    max_consecutive_failures = 10

    while True:
        try:
            global IQ, iq_connector
            
            # Verificar conexión
            if not iq_connector.connected or not iq_connector.check_connection():
                logging.warning("🔌 Reconectando a IQ Option...")
                IQ = iq_connector.connect()
                if not IQ:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logging.error("❌ Máximo de reconexiones fallidas")
                        consecutive_failures = 0
                    time.sleep(5)
                    continue
                else:
                    consecutive_failures = 0
                    
            # Obtener tick REAL
            price = iq_connector.get_realtime_ticks()
            
            if price is not None and price > 0:
                tick_data = predictor.analyzer.add_tick(price)
                if tick_data:
                    # Log inicial de ticks
                    if predictor.analyzer.tick_count <= 5:
                        logging.info(f"💰 Tick REAL #{predictor.analyzer.tick_count}: {price:.5f}")
            else:
                time.sleep(1)
                continue
                
            # Lógica de velas y predicciones
            now = time.time()
            current_candle_start = now//TIMEFRAME*TIMEFRAME
            seconds_remaining = TIMEFRAME - (now % TIMEFRAME)
            
            # Cambio de vela
            if current_candle_start > last_candle_start:
                closed_metrics = predictor.analyzer.get_candle_metrics()
                if closed_metrics:
                    predictor.on_candle_closed(closed_metrics)
                predictor.analyzer.reset_candle()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela iniciada")
                
            # Realizar predicción con datos REALES
            if seconds_remaining <= PREDICTION_WINDOW and (time.time() - last_prediction_time) > (TIMEFRAME - 4):
                if predictor.analyzer.tick_count >= 5:  # Mínimo 5 ticks para predicción
                    pred = predictor.predict_next_candle(seconds_remaining_norm=seconds_remaining/TIMEFRAME)
                    global current_prediction
                    current_prediction = pred
                    last_prediction_time = time.time()
                    
                    logging.info("🎯 PREDICCIÓN REAL: %s | Confianza: %s%% | Precio: %s", 
                               pred.get("direction"), 
                               pred.get("confidence"),
                               pred.get("price", 0))
                           
            time.sleep(1)  # Revisar cada segundo
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(2)

# --------------- FastAPI ---------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    training_samples = len(pd.read_csv(TRAINING_CSV)) if os.path.exists(TRAINING_CSV) else 0
    perf_rows = len(pd.read_csv(PERF_CSV)) if os.path.exists(PERF_CSV) else 0
    perf_acc = 0.0
    try:
        if perf_rows>0:
            perf_df = pd.read_csv(PERF_CSV)
            if "correct" in perf_df:
                perf_acc = perf_df["correct"].mean()*100
    except Exception:
        perf_acc = 0.0
        
    phase = predictor.analyzer.get_candle_metrics().get("market_phase") if predictor.analyzer.get_candle_metrics() else "n/a"
    patterns = [p for (_,p) in predictor.analyzer.last_patterns] if predictor.analyzer.last_patterns else []
    direction = current_prediction.get("direction","N/A")
    color = "#00ff88" if direction=="ALZA" else ("#ff4444" if direction=="BAJA" else "#ffbb33")
    
    price_history = predictor.analyzer.get_price_history()
    price_history_json = json.dumps(price_history)
    
    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width'>
        <title>Delowyss AI V3.8</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            
            .chart-container {{
                background: rgba(255,255,255,0.02);
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            
            .status-connected {{ color: #00ff88; }}
            .status-disconnected {{ color: #ff4444; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🤖 Delowyss Trading AI — V3.8</h1>
            <p>Pair: {PAR} • UTC: <span id="current-time">{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</span>
            • Status: <span id="connection-status" class="{'status-connected' if iq_connector.connected else 'status-disconnected'}">{'CONNECTED' if iq_connector.connected else 'DISCONNECTED'}</span>
            </p>
            
            <div class="countdown" id="countdown">--</div>
            
            <div class="direction-arrow" id="direction-arrow">
                {"⬆️" if direction == "ALZA" else "⬇️" if direction == "BAJA" else "⏸️"}
            </div>
            
            <div class="prediction-card">
                <h2 style="color:{color}; margin:0">{direction} — {current_prediction.get('confidence',0)}% Confidence</h2>
                <p>Model: {current_prediction.get('model_used','HYBRID')} • Price: {current_prediction.get('price',0)}</p>
                <p>Phase: <strong>{phase}</strong> • Patterns: {', '.join(patterns[:3]) if patterns else 'none'}</p>
                <p>Ticks procesados: <strong>{predictor.analyzer.tick_count}</strong></p>
            </div>
            
            <div class="chart-container">
                <h3>📊 Live Price Movement</h3>
                <canvas id="priceChart" width="400" height="100"></canvas>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-cell">
                    <strong>Training Samples</strong>
                    <div>{training_samples}</div>
                </div>
                <div class="metric-cell">
                    <strong>Performance Rows</strong>
                    <div>{perf_rows}</div>
                </div>
                <div class="metric-cell">
                    <strong>Historical Accuracy</strong>
                    <div class="{'accuracy-high' if perf_acc > 60 else 'accuracy-medium' if perf_acc > 50 else 'accuracy-low'}">
                        {perf_acc:.1f}%
                    </div>
                </div>
                <div class="metric-cell">
                    <strong>Current Ticks</strong>
                    <div>{current_prediction.get('tick_count',0)}</div>
                </div>
            </div>
            
            <div class="metric-cell">
                <h3>Prediction Reasons</h3>
                <ul>
                    {"".join([f"<li>✅ {r}</li>" for r in current_prediction.get('reasons',[])]) if current_prediction.get('reasons') else "<li>🔄 Analyzing market data...</li>"}
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
            
            setInterval(updateCountdown, 1000);
            updateCountdown();
            
            const priceHistory = {price_history_json};
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            if (priceHistory.length > 1) {{
                const chart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: Array.from({{length: priceHistory.length}}, (_, i) => i),
                        datasets: [{{
                            label: 'Price',
                            data: priceHistory,
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{ enabled: false }}
                        }},
                        scales: {{
                            x: {{ display: false }},
                            y: {{ 
                                display: true,
                                grid: {{ color: 'rgba(255,255,255,0.1)' }},
                                ticks: {{ color: '#fff' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            setInterval(() => {{
                window.location.reload();
            }}, 3000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/status")
def api_status():
    connected = iq_connector.connected if iq_connector else False
    training_samples = len(pd.read_csv(TRAINING_CSV)) if os.path.exists(TRAINING_CSV) else 0
    perf_rows = len(pd.read_csv(PERF_CSV)) if os.path.exists(PERF_CSV) else 0
    
    return JSONResponse({
        "status": "online",
        "connected": connected,
        "pair": PAR,
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

# --------------- Start threads & server ---------------
if __name__ == "__main__":
    analyzer_thread = threading.Thread(target=professional_tick_analyzer, daemon=True)
    analyzer_thread.start()
    logging.info("📊 Tick analyzer thread started")

    trainer_thread = threading.Thread(target=adaptive_trainer_loop, args=(predictor,), daemon=True)
    trainer_thread.start()
    logging.info("🧠 Model trainer thread started")

    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    logging.info("🚀 Starting Delowyss AI V3.8 server on port %s", port)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except Exception as e:
        logging.error("Server error: %s", e)
    finally:
        logging.info("Server shutdown")
