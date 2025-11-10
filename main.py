# main.py - V5.7 COMPLETO Y FUNCIONAL
"""
Delowyss Trading AI — V5.7 ULTRA EFICIENTE COMPLETO
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
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ------------------ CONFIGURACIÓN ------------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = "EURUSD"
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 8
TICK_BUFFER_SIZE = 150
PORT = int(os.getenv("PORT", "10000"))

# Model paths
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
ONLINE_MODEL_PATH = os.path.join(MODEL_DIR, "online_sgd_ultra.pkl")
ONLINE_SCALER_PATH = os.path.join(MODEL_DIR, "online_scaler_ultra.pkl")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ CONEXIÓN SIMULADA ------------------
class ProfessionalIQConnector:
    def __init__(self):
        self.connected = True
        self.current_price = 1.08500
        self.price_trend = 0.00001
        self.volatility = 0.00005
        logging.info("✅ ProfessionalIQConnector inicializado")
    
    def get_realtime_price(self):
        import random
        price_change = random.uniform(-self.volatility, self.volatility) + self.price_trend
        self.current_price += price_change
        self.current_price = round(self.current_price, 5)
        
        if random.random() > 0.7:
            self.price_trend = random.uniform(-0.00002, 0.00002)
            
        return self.current_price

    def get_remaining_time(self):
        return TIMEFRAME - (int(time.time()) % TIMEFRAME)

# ------------------ IA ULTRA EFICIENTE ------------------
class UltraEfficientAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=80)
        self.last_candle_close = None
        
        self.velocity_metrics = deque(maxlen=30)
        self.acceleration_metrics = deque(maxlen=20)
        self.volume_profile = deque(maxlen=15)
        self.price_levels = deque(maxlen=10)
        
        self.candle_start_time = None
        self.analysis_phases = {
            'initial': {'ticks': 0, 'analysis': {}, 'weight': 0.2},
            'middle': {'ticks': 0, 'analysis': {}, 'weight': 0.3},
            'final': {'ticks': 0, 'analysis': {}, 'weight': 0.5}
        }
        self.phase_accuracy = {'initial': 0.6, 'middle': 0.7, 'final': 0.9}
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Análisis ultra eficiente activado")
            
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)
            self.current_candle_close = price
            
            tick_data = {
                'price': price,
                'timestamp': current_time,
                'volume': 1,
                'microtimestamp': current_time * 1000,
                'seconds_remaining': seconds_remaining,
                'candle_age': current_time - self.candle_start_time if self.candle_start_time else 0
            }
            
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            if self.tick_count % 3 == 0:
                self._calculate_ultra_metrics(tick_data)
                self._analyze_optimized_phases(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"❌ Error en add_tick: {e}")
            return None
    
    def _calculate_ultra_metrics(self, current_tick):
        if len(self.ticks) < 2:
            return
            
        try:
            current_price = current_tick['price']
            current_time = current_tick['timestamp']

            previous_tick = list(self.ticks)[-2]
            time_diff = current_time - previous_tick['timestamp']
            
            if time_diff > 0:
                price_diff = current_price - previous_tick['price']
                velocity = price_diff / time_diff
                
                self.velocity_metrics.append({
                    'velocity': velocity,
                    'timestamp': current_time,
                    'price_change': price_diff
                })
            
            if len(self.velocity_metrics) >= 2:
                current_velocity = self.velocity_metrics[-1]['velocity']
                previous_velocity = self.velocity_metrics[-2]['velocity']
                velocity_time_diff = current_time - self.velocity_metrics[-2]['timestamp']
                
                if velocity_time_diff > 0:
                    acceleration = (current_velocity - previous_velocity) / velocity_time_diff
                    self.acceleration_metrics.append({
                        'acceleration': acceleration,
                        'timestamp': current_time
                    })
            
            if len(self.price_memory) >= 12:
                prices = list(self.price_memory)
                resistance = max(prices[-12:])
                support = min(prices[-12:])
                self.price_levels.append({
                    'resistance': resistance,
                    'support': support,
                    'timestamp': current_time
                })
                
        except Exception as e:
            logging.debug(f"🔧 Error en cálculo de métricas: {e}")
    
    def _analyze_optimized_phases(self, tick_data):
        candle_age = tick_data['candle_age']
        
        if candle_age < 20:
            self.analysis_phases['initial']['ticks'] += 1
            if self.analysis_phases['initial']['ticks'] % 8 == 0:
                self.analysis_phases['initial']['analysis'] = self._get_phase_analysis_optimized('initial')
                
        elif candle_age < 40:
            self.analysis_phases['middle']['ticks'] += 1
            if self.analysis_phases['middle']['ticks'] % 6 == 0:
                self.analysis_phases['middle']['analysis'] = self._get_phase_analysis_optimized('middle')
                
        else:
            self.analysis_phases['final']['ticks'] += 1
            if self.analysis_phases['final']['ticks'] % 3 == 0:
                self.analysis_phases['final']['analysis'] = self._get_phase_analysis_optimized('final')
    
    def _get_phase_analysis_optimized(self, phase):
        try:
            ticks_list = list(self.ticks)
            if not ticks_list:
                return {}
                
            if phase == 'initial':
                ticks = ticks_list[:min(20, len(ticks_list))]
            elif phase == 'middle':
                if len(ticks_list) >= 40:
                    ticks = ticks_list[20:40]
                elif len(ticks_list) > 20:
                    ticks = ticks_list[20:]
                else:
                    ticks = []
            else:
                if len(ticks_list) >= 40:
                    ticks = ticks_list[40:]
                else:
                    ticks = []
            
            if not ticks:
                return {}
            
            prices = [tick['price'] for tick in ticks]
            
            volatility = (max(prices) - min(prices)) * 10000 if prices else 0
            avg_price = np.mean(prices) if prices else 0
            
            if len(prices) >= 5:
                window_prices = prices[-5:] if len(prices) >= 5 else prices
                x_values = np.arange(len(window_prices))
                recent_trend = np.polyfit(x_values, window_prices, 1)[0] * 10000
            else:
                recent_trend = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            if len(prices) >= 2:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                positive_changes = len([x for x in price_changes if x > 0])
                total_changes = len(price_changes)
                buy_pressure = positive_changes / total_changes if total_changes > 0 else 0.5
            else:
                buy_pressure = 0.5
            
            return {
                'avg_price': avg_price,
                'volatility': volatility,
                'trend': 'ALCISTA' if recent_trend > 0.1 else 'BAJISTA' if recent_trend < -0.1 else 'LATERAL',
                'trend_strength': abs(recent_trend),
                'buy_pressure': buy_pressure,
                'tick_count': len(ticks),
                'phase_accuracy': self.phase_accuracy[phase]
            }
        except Exception as e:
            logging.debug(f"🔧 Error en análisis de fase {phase}: {e}")
            return {}
    
    def _calculate_ultra_advanced_metrics(self):
        if len(self.price_memory) < 8:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            trend_metrics = []
            for window in [5, 10, 15]:
                if len(prices) >= window:
                    window_prices = prices[-window:]
                    x_values = np.arange(len(window_prices))
                    trend = np.polyfit(x_values, window_prices, 1)[0] * 10000
                    trend_metrics.append(trend)
            
            trend_strength = np.mean(trend_metrics) if trend_metrics else 0
            
            if len(prices) >= 5:
                momentum = (prices[-1] - prices[-5]) * 10000
            else:
                momentum = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            if len(prices) >= 10:
                recent_prices = prices[-10:]
                volatility = (np.max(recent_prices) - np.min(recent_prices)) * 10000
            else:
                volatility = (np.max(prices) - np.min(prices)) * 10000
            
            if len(self.ticks) >= 8:
                recent_ticks = list(self.ticks)[-8:]
                price_changes = []
                for i in range(1, len(recent_ticks)):
                    change = recent_ticks[i]['price'] - recent_ticks[i-1]['price']
                    price_changes.append(change)
                
                if price_changes:
                    positive = len([x for x in price_changes if x > 0])
                    total = len(price_changes)
                    buy_pressure = positive / total
                else:
                    buy_pressure = 0.5
            else:
                buy_pressure = 0.5
            
            avg_velocity = 0
            if self.velocity_metrics:
                velocities = [v['velocity'] for v in list(self.velocity_metrics)[-10:]]
                avg_velocity = np.mean(velocities) * 10000 if velocities else 0
            
            phase_analysis = self._combine_phase_analysis_optimized()
            
            market_phase = self._determine_market_phase_optimized(
                trend_strength, volatility, phase_analysis
            )
            
            data_quality = min(1.0, self.tick_count / 20.0)
            
            result = {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volatility': volatility,
                'buy_pressure': buy_pressure,
                'sell_pressure': 1 - buy_pressure,
                'pressure_ratio': buy_pressure / (1 - buy_pressure) if buy_pressure < 1 else 10.0,
                'market_phase': market_phase,
                'data_quality': data_quality,
                'velocity': avg_velocity,
                'phase_analysis': phase_analysis,
                'candle_progress': (time.time() - self.candle_start_time) / TIMEFRAME if self.candle_start_time else 0,
                'total_ticks': self.tick_count,
                'confidence_score': self._calculate_confidence_score()
            }
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error en cálculo de métricas avanzadas: {e}")
            return {}
    
    def _combine_phase_analysis_optimized(self):
        try:
            initial = self.analysis_phases['initial']['analysis']
            middle = self.analysis_phases['middle']['analysis']
            final = self.analysis_phases['final']['analysis']
            
            weights = {
                'initial': self.analysis_phases['initial']['weight'],
                'middle': self.analysis_phases['middle']['weight'], 
                'final': self.analysis_phases['final']['weight']
            }
            
            trends = []
            for phase, data in [('initial', initial), ('middle', middle), ('final', final)]:
                if data and data.get('trend'):
                    trends.append((data['trend'], weights[phase]))
            
            if trends:
                alcista_weight = sum(weight for trend, weight in trends if trend == 'ALCISTA')
                bajista_weight = sum(weight for trend, weight in trends if trend == 'BAJISTA')
                
                if alcista_weight > bajista_weight:
                    combined_trend = 'ALCISTA'
                elif bajista_weight > alcista_weight:
                    combined_trend = 'BAJISTA'
                else:
                    combined_trend = 'LATERAL'
            else:
                combined_trend = 'N/A'
            
            combined = {
                'trend': combined_trend,
                'momentum_shift': len(set(trend for trend, _ in trends)) > 1 if trends else False,
                'consistency_score': alcista_weight if combined_trend == 'ALCISTA' else bajista_weight if combined_trend == 'BAJISTA' else 0.5,
            }
            
            return combined
        except Exception as e:
            logging.debug(f"🔧 Error combinando análisis de fases: {e}")
            return {}
    
    def _determine_market_phase_optimized(self, trend_strength, volatility, phase_analysis):
        if volatility < 0.2 and abs(trend_strength) < 0.3:
            return "consolidation"
        elif abs(trend_strength) > 2.5:
            return "strong_trend"
        elif abs(trend_strength) > 1.2:
            return "trending" 
        elif volatility > 2.0:
            return "high_volatility"
        elif phase_analysis.get('momentum_shift', False):
            return "reversal_potential"
        else:
            return "normal"
    
    def _calculate_confidence_score(self):
        score = min(30, (self.tick_count / 25) * 30)
        
        if len(self.velocity_metrics) >= 10:
            score += 20
        
        score += self.analysis_phases['final']['weight'] * 30
        
        if len(self.price_memory) >= 10:
            prices = list(self.price_memory)[-10:]
            volatility = (max(prices) - min(prices)) * 10000
            if volatility < 1.0:
                score += 20
        
        return min(100, score)
    
    def get_ultra_analysis(self):
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'INSUFFICIENT_DATA', 
                'tick_count': self.tick_count,
                'message': f'Recolectando ticks: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION}',
                'confidence': self._calculate_confidence_score()
            }
        
        try:
            advanced_metrics = self._calculate_ultra_advanced_metrics()
            if not advanced_metrics:
                return {'status': 'ERROR', 'message': 'Error en métricas'}
            
            overall_confidence = min(95, advanced_metrics.get('confidence_score', 0) + 
                                   advanced_metrics.get('data_quality', 0) * 20)
            
            return {
                'status': 'SUCCESS',
                'tick_count': self.tick_count,
                'current_price': self.current_candle_close,
                'open_price': self.current_candle_open,
                'high_price': self.current_candle_high,
                'low_price': self.current_candle_low,
                'candle_range': (self.current_candle_high - self.current_candle_low) * 10000,
                'timestamp': time.time(),
                'candle_age': time.time() - self.candle_start_time if self.candle_start_time else 0,
                'overall_confidence': overall_confidence,
                **advanced_metrics
            }
            
        except Exception as e:
            logging.error(f"❌ Error en análisis completo: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        try:
            if self.current_candle_close is not None:
                self.last_candle_close = self.current_candle_close
                
            self.ticks.clear()
            self.current_candle_open = None
            self.current_candle_high = None
            self.current_candle_low = None
            self.current_candle_close = None
            self.tick_count = 0
            self.price_memory.clear()
            self.velocity_metrics.clear()
            self.acceleration_metrics.clear()
            self.volume_profile.clear()
            self.price_levels.clear()
            self.candle_start_time = None
            
            for phase in self.analysis_phases:
                self.analysis_phases[phase] = {'ticks': 0, 'analysis': {}, 'weight': self.analysis_phases[phase]['weight']}
                
        except Exception as e:
            logging.error(f"❌ Error en reset: {e}")

# ------------------ ADAPTIVE MARKET LEARNER ------------------
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class AdaptiveMarketLearner:
    def __init__(self, feature_size=18, classes=None, buffer_size=1000):
        self.feature_size = feature_size
        self.classes = np.array(['BAJA', 'LATERAL', 'ALZA']) if classes is None else np.array(classes)
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.model_path = ONLINE_MODEL_PATH
        self.scaler_path = ONLINE_SCALER_PATH
        self._ensure_dirs()
        self.scaler = self._load_scaler()
        self.model = self._load_model()
        self.training_count = 0
        self.last_training_result = {"trained": False, "reason": "not_initialized"}

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logging.info("✅ Modelo online cargado exitosamente")
                return model
            except Exception as e:
                logging.warning(f"⚠️ No se pudo cargar modelo online: {e}")
        
        model = SGDClassifier(
            loss='log_loss', 
            max_iter=1000,
            tol=1e-3,
            warm_start=True,
            learning_rate='optimal'
        )
        dummy_X = np.random.normal(0, 0.1, (3, self.feature_size))
        dummy_y = np.array(['BAJA', 'LATERAL', 'ALZA'])
        model.partial_fit(dummy_X, dummy_y, classes=self.classes)
        logging.info("🆕 Nuevo modelo online creado")
        return model

    def _load_scaler(self):
        if os.path.exists(self.scaler_path):
            try:
                scaler = joblib.load(self.scaler_path)
                logging.info("✅ Scaler online cargado exitosamente")
                return scaler
            except Exception:
                pass
        return StandardScaler()

    def persist(self):
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.training_count % 10 == 0:
                logging.info(f"💾 Modelo persistido (entrenamientos: {self.training_count})")
        except Exception as e:
            logging.error(f"❌ Error guardando modelo: {e}")

    def add_sample(self, features: np.ndarray, label: str):
        try:
            if features is not None and features.size > 0:
                if features.ndim == 1 and features.shape[0] == self.feature_size:
                    self.replay_buffer.append((features.astype(float), label))
                    return True
            return False
        except Exception as e:
            logging.error(f"❌ Error añadiendo muestra: {e}")
            return False

    def partial_train(self, batch_size=32):
        if len(self.replay_buffer) < 5:
            result = {"trained": False, "reason": "not_enough_samples", "buffer_size": len(self.replay_buffer)}
            self.last_training_result = result
            return result
        
        try:
            samples = list(self.replay_buffer)[-batch_size:]
            X = np.vstack([s[0] for s in samples])
            y = np.array([s[1] for s in samples])
            
            if len(self.replay_buffer) >= 10:
                self.scaler.partial_fit(X)
            
            Xs = self.scaler.transform(X) if hasattr(self.scaler, "mean_") else X
            
            self.model.partial_fit(Xs, y, classes=self.classes)
            self.training_count += 1
            
            if self.training_count % 5 == 0:
                self.persist()
                
            result = {
                "trained": True, 
                "n_samples": len(samples),
                "training_count": self.training_count,
                "buffer_size": len(self.replay_buffer)
            }
            self.last_training_result = result
            return result
            
        except Exception as e:
            logging.error(f"❌ Error en entrenamiento: {e}")
            result = {"trained": False, "reason": str(e)}
            self.last_training_result = result
            return result

    def predict(self, features: np.ndarray):
        try:
            X = np.atleast_2d(features.astype(float))
            
            if (hasattr(self.scaler, "mean_") and self.scaler.mean_ is not None and 
                self.training_count > 0 and len(self.replay_buffer) >= 5):
                
                Xs = self.scaler.transform(X)
                predicted = self.model.predict(Xs)[0]
                
                decision_scores = self.model.decision_function(Xs)[0]
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probs = exp_scores / np.sum(exp_scores)
                prob_dict = dict(zip(self.model.classes_, probs))
                
                confidence = max(prob_dict.values()) * 100
                
                return {
                    "predicted": predicted,
                    "proba": prob_dict,
                    "confidence": round(confidence, 2),
                    "training_count": self.training_count,
                    "status": "ML_ACTIVE"
                }
            else:
                return {
                    "predicted": "LATERAL",
                    "proba": dict(zip(self.classes, [1/3]*3)),
                    "confidence": 33.3,
                    "training_count": self.training_count,
                    "status": "ML_INITIALIZING"
                }
                
        except Exception as e:
            logging.error(f"❌ Error en predict: {e}")
            return {
                "predicted": "LATERAL",
                "proba": dict(zip(self.classes, [1/3]*3)),
                "confidence": 33.3,
                "training_count": self.training_count,
                "status": "ML_ERROR"
            }

# ------------------ SISTEMA DE VALIDACIÓN ------------------
class RealTimeValidator:
    def __init__(self):
        self.tracking_active = False
        self.prediction_data = None
        self.validation_history = deque(maxlen=50)
        
    def start_tracking(self, prediction, current_price):
        self.tracking_active = True
        self.prediction_data = {
            'start_time': time.time(),
            'initial_price': current_price,
            'predicted_direction': prediction['direction'],
            'prediction_confidence': prediction['confidence'],
            'validation_points': []
        }
        logging.info(f"🔍 INICIANDO TRACKING: {prediction['direction']} | Conf: {prediction['confidence']}%")
    
    def update_validation(self, current_price, current_tick_count):
        if not self.tracking_active or not self.prediction_data:
            return None
            
        current_time = time.time()
        if current_time - self.prediction_data['start_time'] < 3 and current_tick_count % 5 != 0:
            return None
            
        price_change = (current_price - self.prediction_data['initial_price']) * 10000
        expected_direction = self.prediction_data['predicted_direction']
        
        movement_threshold = max(0.2, abs(price_change) * 0.3)
        if abs(price_change) < movement_threshold:
            current_direction = "LATERAL"
            is_tracking = None
        else:
            current_direction = "ALZA" if price_change > 0 else "BAJA"
            is_tracking = (current_direction == expected_direction)
        
        tracking_score = self._calculate_tracking_score(price_change, is_tracking)
        
        validation_point = {
            'timestamp': current_time,
            'price_change': round(price_change, 2),
            'current_direction': current_direction,
            'is_tracking': is_tracking,
            'tracking_score': tracking_score,
            'tick_count': current_tick_count
        }
        
        self.prediction_data['validation_points'].append(validation_point)
        
        if len(self.prediction_data['validation_points']) % 5 == 0:
            status_icon = "✅" if is_tracking else "❌" if is_tracking is False else "⚪"
            logging.info(f"🔍 TRACKING: {status_icon} | Mov: {price_change:.1f}pips | Score: {tracking_score:.1f}%")
        
        return validation_point
    
    def _calculate_tracking_score(self, price_change, is_tracking):
        base_score = 50
        
        if is_tracking:
            base_score += 25
            movement_bonus = min(20, abs(price_change) * 2)
            base_score += movement_bonus
        elif is_tracking is False:
            base_score -= 25
        
        return max(0, min(100, base_score))
    
    def get_realtime_summary(self):
        if not self.tracking_active or not self.prediction_data:
            return None
            
        points = self.prediction_data['validation_points']
        if not points:
            return None
            
        recent_points = list(points)[-10:]
        tracking_points = [p for p in recent_points if p['is_tracking'] is True]
        
        tracking_ratio = len(tracking_points) / len(recent_points) if recent_points else 0
        avg_tracking_score = np.mean([p['tracking_score'] for p in recent_points]) if recent_points else 50
        
        return {
            'active': True,
            'predicted_direction': self.prediction_data['predicted_direction'],
            'tracking_ratio': round(tracking_ratio, 2),
            'current_score': round(avg_tracking_score, 1),
            'total_validation_points': len(points),
            'recent_tracking': len(tracking_points),
            'status': 'STRONG' if avg_tracking_score > 70 else 'WEAK' if avg_tracking_score < 40 else 'NEUTRAL'
        }
    
    def stop_tracking(self):
        if self.tracking_active:
            summary = self.get_realtime_summary()
            self.tracking_active = False
            if summary:
                self.validation_history.append(summary)
            return summary
        return None

# ------------------ FEATURE BUILDER ------------------
def build_advanced_features_from_analysis(analysis, seconds_remaining):
    try:
        if analysis.get('status') != 'SUCCESS':
            return np.zeros(18)
            
        current_price = analysis.get('current_price', 0)
        trend_strength = analysis.get('trend_strength', 0)
        momentum = analysis.get('momentum', 0)
        volatility = analysis.get('volatility', 0)
        buy_pressure = analysis.get('buy_pressure', 0.5)
        velocity = analysis.get('velocity', 0)
        candle_progress = analysis.get('candle_progress', 0)
        phase_analysis = analysis.get('phase_analysis', {})
        
        time_remaining = seconds_remaining / TIMEFRAME
        
        features = np.array([
            current_price,
            trend_strength,
            momentum,
            volatility,
            buy_pressure,
            1 - buy_pressure,
            buy_pressure / (1 - buy_pressure) if buy_pressure < 1 else 10.0,
            velocity,
            candle_progress,
            1.0 if phase_analysis.get('momentum_shift', False) else 0.0,
            phase_analysis.get('consistency_score', 0),
            time_remaining,
            analysis.get('tick_count', 0) / 100.0,
            analysis.get('data_quality', 0),
            analysis.get('candle_range', 0),
            min(1.0, analysis.get('tick_count', 0) / 50.0),
            np.log1p(abs(trend_strength)),
            np.sqrt(abs(momentum)) if momentum != 0 else 0
        ]).astype(float)
        
        if features.shape[0] < 18:
            features = np.pad(features, (0, 18 - features.shape[0]))
        elif features.shape[0] > 18:
            features = features[:18]
            
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logging.warning("⚠️ Características inválidas detectadas, usando valores por defecto")
            return np.zeros(18)
            
        return features
        
    except Exception as e:
        logging.error(f"❌ Error construyendo features: {e}")
        return np.zeros(18)

# ------------------ SISTEMA IA PROFESIONAL ------------------
class ComprehensiveAIPredictor:
    def __init__(self):
        self.analyzer = UltraEfficientAnalyzer()
        self.realtime_validator = RealTimeValidator()
        self.prediction_history = deque(maxlen=20)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': 0.0
        }
        self.last_prediction = None
        
    def process_tick(self, price: float, seconds_remaining: float = None):
        try:
            tick_data = self.analyzer.add_tick(price, seconds_remaining)
            
            if self.realtime_validator.tracking_active:
                self.realtime_validator.update_validation(price, self.analyzer.tick_count)
                
            return {
                "tick_count": self.analyzer.tick_count,
                "status": "PROCESSED"
            }
        except Exception as e:
            logging.error(f"❌ Error en process_tick: {e}")
            return None
    
    def _comprehensive_ai_analysis(self, analysis, ml_prediction=None):
        try:
            momentum = analysis['momentum']
            trend_strength = analysis['trend_strength']
            pressure_ratio = analysis['pressure_ratio']
            market_phase = analysis['market_phase']
            data_quality = analysis['data_quality']
            candle_progress = analysis.get('candle_progress', 0)
            
            buy_score = 0
            sell_score = 0
            reasons = []
            
            ml_boost = 0
            if ml_prediction and ml_prediction.get('confidence', 0) > 60:
                ml_direction = ml_prediction.get('predicted', 'LATERAL')
                ml_boost = (ml_prediction['confidence'] / 100.0) * 0.3
                reasons.append(f"🤖 ML: {ml_direction} ({ml_prediction['confidence']}%)")
            
            late_phase_weight = 1.0 if candle_progress > 0.8 else 0.7

            if abs(trend_strength) > 1.0:
                weight = 0.35 * late_phase_weight
                if trend_strength > 0:
                    buy_score += 8 * weight + ml_boost
                    reasons.append(f"📈 Tendencia alcista ({trend_strength:.1f})")
                else:
                    sell_score += 8 * weight + ml_boost
                    reasons.append(f"📉 Tendencia bajista ({trend_strength:.1f})")
            
            if abs(momentum) > 0.8:
                weight = 0.30 * late_phase_weight
                if momentum > 0:
                    buy_score += 7 * weight
                else:
                    sell_score += 7 * weight
            
            if pressure_ratio > 2.0:
                weight = 0.20 * late_phase_weight
                buy_score += 6 * weight
                reasons.append(f"💰 Presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.5:
                weight = 0.20 * late_phase_weight
                sell_score += 6 * weight
                reasons.append(f"💸 Presión vendedora ({pressure_ratio:.1f}x)")
            
            score_difference = buy_score - sell_score
            
            base_threshold = 0.4 - (0.1 * (1 - data_quality))
            
            if abs(score_difference) > base_threshold:
                if score_difference > 0:
                    direction = "ALZA"
                    base_confidence = 55 + (score_difference * 40)
                else:
                    direction = "BAJA" 
                    base_confidence = 55 + (abs(score_difference) * 40)
            else:
                direction = "LATERAL"
                base_confidence = 40
                reasons.append("⚡ Señales mixtas")
            
            confidence = min(90, base_confidence * data_quality)
            confidence = max(35, confidence)
            
            if analysis['tick_count'] > 40:
                confidence = min(90, confidence + 15)
                reasons.append("📊 Alta calidad de datos")

            return {
                'direction': direction,
                'confidence': int(confidence),
                'buy_score': round(buy_score, 2),
                'sell_score': round(sell_score, 2),
                'score_difference': round(score_difference, 2),
                'reasons': reasons,
                'market_phase': market_phase,
                'candle_progress': round(candle_progress, 2),
                'ml_boost': round(ml_boost, 2)
            }
        except Exception as e:
            logging.error(f"❌ Error en análisis IA: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 35,
                'reasons': ['🤖 Error en análisis'],
                'buy_score': 0,
                'sell_score': 0
            }
    
    def predict_next_candle(self, ml_prediction=None):
        try:
            analysis = self.analyzer.get_ultra_analysis()
            
            if analysis.get('status') != 'SUCCESS':
                return {
                    'direction': 'LATERAL',
                    'confidence': 0,
                    'reason': analysis.get('message', 'Analizando...'),
                    'timestamp': now_iso()
                }
            
            realtime_boost = 0
            if self.realtime_validator.tracking_active:
                tracking_summary = self.realtime_validator.get_realtime_summary()
                if tracking_summary:
                    if tracking_summary['status'] == 'STRONG':
                        realtime_boost = 0.15
                    elif tracking_summary['status'] == 'WEAK':
                        realtime_boost = -0.10
            
            prediction = self._comprehensive_ai_analysis(analysis, ml_prediction)
            
            if realtime_boost != 0:
                prediction['confidence'] = int(max(35, min(90, prediction['confidence'] + (realtime_boost * 100))))
                prediction['realtime_boost'] = realtime_boost
            
            if prediction['confidence'] < 45:
                prediction['direction'] = 'LATERAL'
                prediction['reasons'].append("🔍 Confianza insuficiente")
            
            prediction.update({
                'tick_count': analysis['tick_count'],
                'current_price': analysis['current_price'],
                'candle_range': analysis.get('candle_range', 0),
                'timestamp': now_iso(),
                'model_version': 'ULTRA_EFFICIENT_V5.7'
            })
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            if prediction['direction'] != 'LATERAL' and prediction['confidence'] > 50:
                self.realtime_validator.start_tracking(prediction, analysis['current_price'])
            
            if prediction['direction'] != 'LATERAL':
                logging.info(f"🎯 PREDICCIÓN: {prediction['direction']} | Conf: {prediction['confidence']}% | Ticks: {analysis['tick_count']}")
            
            return prediction
        except Exception as e:
            logging.error(f"❌ Error en predict_next_candle: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 0,
                'reason': 'Error en predicción',
                'timestamp': now_iso()
            }
    
    def validate_prediction(self, new_candle_open_price):
        try:
            if not self.last_prediction:
                return None
                
            realtime_summary = self.realtime_validator.stop_tracking()
                
            last_pred = self.last_prediction
            predicted_direction = last_pred.get('direction', 'N/A')
            
            previous_close = self.analyzer.last_candle_close
            current_open = new_candle_open_price
            
            if previous_close is None or current_open is None:
                return None
                
            price_change = (current_open - previous_close) * 10000
            
            candle_range = last_pred.get('candle_range', 0.5)
            minimal_change = max(0.15, candle_range * 0.2)
            
            if abs(price_change) < minimal_change:
                actual_direction = "LATERAL"
                is_correct = False
            else:
                actual_direction = "ALZA" if price_change > 0 else "BAJA"
                is_correct = (actual_direction == predicted_direction)
            
            if predicted_direction != "LATERAL":
                self.performance_stats['total_predictions'] += 1
                if is_correct:
                    self.performance_stats['correct_predictions'] += 1
            
            total = self.performance_stats['total_predictions']
            correct = self.performance_stats['correct_predictions']
            accuracy = (correct / total * 100) if total > 0 else 0
            self.performance_stats['recent_accuracy'] = accuracy
            
            status_icon = "✅" if is_correct else "❌"
            if actual_direction == "LATERAL":
                status_icon = "⚪"
            
            logging.info(f"🎯 VALIDACIÓN: {status_icon} {predicted_direction}→{actual_direction} | Conf: {last_pred.get('confidence', 0)}% | Cambio: {price_change:.1f}pips")
            
            validation_result = {
                'correct': is_correct,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': last_pred.get('confidence', 0),
                'price_change': round(price_change, 2),
                'accuracy': round(accuracy, 1),
                'total_predictions': total,
                'correct_predictions': correct,
                'status_icon': status_icon,
                'timestamp': now_iso()
            }
            
            return validation_result
            
        except Exception as e:
            logging.error(f"❌ Error en validación: {e}")
            return None
    
    def get_performance_stats(self):
        return self.performance_stats.copy()
    
    def reset(self):
        try:
            self.analyzer.reset()
        except Exception as e:
            logging.error(f"❌ Error en reset predictor: {e}")

# ------------------ FASTAPI APP ------------------
app = FastAPI(
    title="Delowyss Trading AI V5.7",
    description="Sistema de IA para trading algorítmico - EUR/USD",
    version="5.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- SISTEMA PRINCIPAL ---------------
iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner(feature_size=18)

# VARIABLES GLOBALES
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 Sistema inicializando..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING"
}

# Estado interno
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False
_last_prediction_time = 0
_last_price = None

def tick_processor(price, timestamp):
    global current_prediction
    try:
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        if predictor.analyzer.tick_count == 0:
            delay = current_time - _last_candle_start
            logging.info(f"🎯 PRIMER TICK: {price:.5f} | Retardo: {delay:.1f}s")
        
        if predictor.analyzer.tick_count < 15 and predictor.analyzer.tick_count % 5 == 0:
            logging.info(f"📊 Tick #{predictor.analyzer.tick_count + 1}: {price:.5f}")

        tick_data = predictor.process_tick(price, seconds_remaining)
        
        if tick_data:
            current_prediction.update({
                "current_price": float(price),
                "tick_count": predictor.analyzer.tick_count,
                "timestamp": now_iso(),
                "status": "ACTIVE"
            })
            
    except Exception as e:
        logging.error(f"❌ Error procesando tick: {e}")

def premium_main_loop_corregido():
    """🚀 LOOP PRINCIPAL CORREGIDO - PREDICCIÓN 55s-60s"""
    global current_prediction, _last_candle_start, _prediction_made_this_candle
    global _last_prediction_time, _last_price
    
    logging.info(f"🚀 DELOWYSS AI V5.7 ULTRA EFICIENTE COMPLETO")
    logging.info("🎯 ANÁLISIS 0s-55s + PREDICCIÓN 55s-60s ACTIVADO")
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # Obtener precio
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price
                tick_processor(price, current_time)

            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress

            # ✅ PREDICCIÓN EN VENTANA 55s-60s
            prediction_window_active = (seconds_remaining <= PREDICTION_WINDOW and 
                                      seconds_remaining > 0.5)
            
            if (prediction_window_active and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - _last_prediction_time) >= 2 and
                not _prediction_made_this_candle):

                logging.info(f"🎯 VENTANA PREDICCIÓN: {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                # ANÁLISIS CON DATOS COMPLETOS (55s)
                analysis = predictor.analyzer.get_ultra_analysis()
                if analysis.get('status') == 'SUCCESS':
                    features = build_advanced_features_from_analysis(analysis, seconds_remaining)
                    ml_prediction = online_learner.predict(features)
                    
                    # PREDECIR SIGUIENTE VELA
                    final_prediction = predictor.predict_next_candle(ml_prediction)
                    
                    current_prediction.update(final_prediction)
                    current_prediction.update({
                        "ai_model_predicted": ml_prediction['predicted'],
                        "ml_confidence": ml_prediction['confidence'],
                        "training_count": ml_prediction['training_count']
                    })

                    _last_prediction_time = time.time()
                    _prediction_made_this_candle = True
                    
                    logging.info(f"🚀 PREDICCIÓN COMPLETADA: {final_prediction['direction']} {final_prediction['confidence']}%")

            # DETECCIÓN NUEVA VELA
            if current_candle_start > _last_candle_start:
                if _last_price is not None:
                    validation = predictor.validate_prediction(_last_price)
                    if validation:
                        price_change = validation.get("price_change", 0)
                        actual_direction = validation.get("actual", "LATERAL")
                        
                        # Auto-learning
                        analysis = predictor.analyzer.get_ultra_analysis()
                        if analysis.get('status') == 'SUCCESS':
                            features = build_advanced_features_from_analysis(analysis, 0)
                            
                            if features is not None and features.size == 18:
                                online_learner.add_sample(features, actual_direction)
                                training_result = online_learner.partial_train(batch_size=16)
                                
                                if training_result.get('trained', False):
                                    logging.info(f"📚 AutoLearning: {actual_direction} | Cambio: {price_change:.1f}pips")

                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🕯️ NUEVA VELA - Sistema reiniciado")

            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(1)

# ------------------ ENDPOINTS SIMPLIFICADOS ------------------
@app.get("/")
async def root():
    return {
        "message": "Delowyss Trading AI V5.7 - Sistema Completo", 
        "status": "active",
        "version": "5.7.0",
        "pair": "EURUSD",
        "timestamp": now_iso()
    }

@app.get("/api/prediction")
async def get_prediction():
    return current_prediction

@app.get("/api/performance")
async def get_performance():
    stats = predictor.get_performance_stats()
    return {
        "performance": stats,
        "ml_training": online_learner.last_training_result,
        "system_status": "ACTIVE",
        "timestamp": now_iso()
    }

@app.get("/api/analysis")
async def get_analysis():
    analysis = predictor.analyzer.get_ultra_analysis()
    return {
        "analysis": analysis,
        "timestamp": now_iso()
    }

@app.get("/api/status")
async def get_status():
    return {
        "status": "operational",
        "version": "5.7.0",
        "pair": "EURUSD",
        "timeframe": "1min",
        "iq_connected": iq_connector.connected,
        "current_price": iq_connector.current_price,
        "timestamp": now_iso()
    }

# ------------------ INICIALIZACIÓN ------------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop_corregido, daemon=True)
        thread.start()
        logging.info(f"⭐ DELOWYSS AI V5.7 INICIADA - SISTEMA COMPLETO")
        logging.info("🎯 FLUJO: Análisis 0s-55s → Predicción 55s-60s → Validación")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

# ✅ INICIAR SISTEMA
if __name__ == "__main__":
    start_system()
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=False
    )
else:
    start_system()
