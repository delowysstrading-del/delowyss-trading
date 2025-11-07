# 🚀 **DELOWYSS AI PREMIUM V5.4 - VERSIÓN COMPLETA OPTIMIZADA**

```python
# main.py - V5.4 PREMIUM COMPLETA (IA Avanzada + AutoLearning + Interfaz Original)
"""
Delowyss Trading AI — V5.4 PREMIUM COMPLETA CON AUTOLEARNING
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
import atexit
import signal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Gestión elegante de dependencias opcionales
try:
    from iqoptionapi.stable_api import IQ_Option
    IQ_OPTION_AVAILABLE = True
except ImportError:
    IQ_Option = None
    IQ_OPTION_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN PREMIUM ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = int(os.getenv("TIMEFRAME", "60"))
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "5"))
MIN_TICKS_FOR_PREDICTION = int(os.getenv("MIN_TICKS_FOR_PREDICTION", "20"))
TICK_BUFFER_SIZE = int(os.getenv("TICK_BUFFER_SIZE", "500"))
PORT = int(os.getenv("PORT", "10000"))

# Model paths
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
ONLINE_MODEL_PATH = os.path.join(MODEL_DIR, "online_sgd.pkl")
ONLINE_SCALER_PATH = os.path.join(MODEL_DIR, "online_scaler.pkl")

# ---------------- VARIABLES GLOBALES DE ESTADO ----------------
SYSTEM_READY = False
SYSTEM_RESTART_COUNT = 0
MAX_RESTARTS = 3

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ---------------- MANEJADOR DE SHUTDOWN GRACEFUL ----------------
def handle_shutdown(signum, frame):
    """Maneja el cierre graceful del sistema"""
    logging.info("🔄 Señal de apagado recibida - Guardando estado...")
    try:
        online_learner.persist()
        logging.info("💾 Modelo guardado antes del apagado")
    except Exception as e:
        logging.error(f"❌ Error guardando modelo: {e}")
    exit(0)

# Registrar manejadores de señales
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# ------------------ IA AVANZADA COMPLETA (ORIGINAL MEJORADA) ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=100)
        self.last_candle_close = None
        
        # Métricas avanzadas ORIGINALES
        self.velocity_metrics = deque(maxlen=50)
        self.acceleration_metrics = deque(maxlen=30)
        self.volume_profile = deque(maxlen=20)
        self.price_levels = deque(maxlen=15)
        
        # Estados del análisis ORIGINAL
        self.candle_start_time = None
        self.analysis_phases = {
            'initial': {'ticks': 0, 'analysis': {}},
            'middle': {'ticks': 0, 'analysis': {}},
            'final': {'ticks': 0, 'analysis': {}}
        }
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            # Inicializar vela si es el primer tick
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Comenzando análisis tick-by-tick")
            
            # Actualizar precios extremos
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
            
            # Almacenar tick
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            # Calcular métricas en tiempo real ORIGINAL
            self._calculate_comprehensive_metrics(tick_data)
            
            # Análisis por fases de la vela ORIGINAL
            self._analyze_candle_phase(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"Error en add_tick: {e}")
            return None
    
    def _calculate_comprehensive_metrics(self, current_tick):
        """Métricas avanzadas ORIGINALES"""
        if len(self.ticks) < 2:
            return
            
        try:
            current_price = current_tick['price']
            current_time = current_tick['timestamp']
            
            # Velocidad del precio
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
            
            # Aceleración
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
            
            # Perfil de volumen por niveles
            if len(self.ticks) >= 10:
                recent_ticks = list(self.ticks)[-10:]
                price_changes = [tick['price'] for tick in recent_ticks]
                if price_changes:
                    avg_price = np.mean(price_changes)
                    self.volume_profile.append({
                        'avg_price': avg_price,
                        'tick_count': len(recent_ticks),
                        'timestamp': current_time
                    })
            
            # Identificar niveles de precio importantes
            if len(self.price_memory) >= 15:
                prices = list(self.price_memory)
                resistance = max(prices[-15:])
                support = min(prices[-15:])
                self.price_levels.append({
                    'resistance': resistance,
                    'support': support,
                    'timestamp': current_time
                })
                
        except Exception as e:
            logging.debug(f"Error en cálculo de métricas: {e}")
    
    def _analyze_candle_phase(self, tick_data):
        """Análisis por fases TEMPORALES ORIGINAL"""
        candle_age = tick_data['candle_age']
        
        if candle_age < 20:  # Primera fase: 0-20 segundos
            self.analysis_phases['initial']['ticks'] += 1
            if self.analysis_phases['initial']['ticks'] % 10 == 0:
                self.analysis_phases['initial']['analysis'] = self._get_phase_analysis('initial')
                
        elif candle_age < 40:  # Segunda fase: 20-40 segundos
            self.analysis_phases['middle']['ticks'] += 1
            if self.analysis_phases['middle']['ticks'] % 10 == 0:
                self.analysis_phases['middle']['analysis'] = self._get_phase_analysis('middle')
                
        else:  # Fase final: 40-60 segundos
            self.analysis_phases['final']['ticks'] += 1
            if self.analysis_phases['final']['ticks'] % 5 == 0:
                self.analysis_phases['final']['analysis'] = self._get_phase_analysis('final')
    
    def _get_phase_analysis(self, phase):
        """Análisis específico por fase ORIGINAL"""
        try:
            if phase == 'initial':
                ticks = list(self.ticks)[:20] if len(self.ticks) >= 20 else list(self.ticks)
            elif phase == 'middle':
                ticks = list(self.ticks)[20:40] if len(self.ticks) >= 40 else list(self.ticks)[20:]
            else:  # final
                ticks = list(self.ticks)[40:] if len(self.ticks) >= 40 else []
            
            if not ticks:
                return {}
            
            prices = [tick['price'] for tick in ticks]
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            return {
                'avg_price': np.mean(prices),
                'volatility': max(prices) - min(prices) if prices else 0,
                'trend': 'ALCISTA' if prices[-1] > prices[0] else 'BAJISTA' if prices[-1] < prices[0] else 'LATERAL',
                'buy_pressure': len([x for x in price_changes if x > 0]) / len(price_changes) if price_changes else 0.5,
                'tick_count': len(ticks)
            }
        except Exception as e:
            logging.debug(f"Error en análisis de fase {phase}: {e}")
            return {}
    
    def _calculate_advanced_metrics(self):
        """Métricas avanzadas ORIGINALES COMPLETAS"""
        if len(self.price_memory) < 10:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            # Análisis de tendencia completo
            if len(prices) >= 30:
                short_trend = np.polyfit(range(10), prices[-10:], 1)[0]
                medium_trend = np.polyfit(range(20), prices[-20:], 1)[0]
                full_trend = np.polyfit(range(min(30, len(prices))), prices[-min(30, len(prices)):], 1)[0]
                trend_strength = (short_trend * 0.4 + medium_trend * 0.3 + full_trend * 0.3) * 10000
            else:
                trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            # Momentum multi-temporal
            momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
            momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else 0
            momentum_20 = (prices[-1] - prices[-20]) * 10000 if len(prices) >= 20 else 0
            momentum = (momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2)
            
            # Volatilidad segmentada
            if len(prices) >= 20:
                early_volatility = (max(prices[:10]) - min(prices[:10])) * 10000
                late_volatility = (max(prices[-10:]) - min(prices[-10:])) * 10000
                volatility = (early_volatility * 0.3 + late_volatility * 0.7)
            else:
                volatility = (max(prices) - min(prices)) * 10000
            
            # Presión de compra/venta basada en toda la vela
            if len(self.ticks) > 10:
                price_changes = []
                for i in range(1, len(self.ticks)):
                    change = self.ticks[i]['price'] - self.ticks[i-1]['price']
                    price_changes.append(change)
                
                if price_changes:
                    positive = len([x for x in price_changes if x > 0])
                    negative = len([x for x in price_changes if x < 0])
                    total = len(price_changes)
                    
                    buy_pressure = positive / total
                    sell_pressure = negative / total
                    
                    if sell_pressure > 0.05:
                        pressure_ratio = buy_pressure / sell_pressure
                    else:
                        pressure_ratio = 10 if buy_pressure > 0 else 1
                else:
                    buy_pressure = sell_pressure = pressure_ratio = 0.5
            else:
                buy_pressure = sell_pressure = pressure_ratio = 0.5
            
            # Velocidad promedio
            avg_velocity = 0
            if self.velocity_metrics:
                velocities = [v['velocity'] for v in self.velocity_metrics]
                avg_velocity = np.mean(velocities) * 10000
            
            # Análisis de fases combinado
            phase_analysis = self._combine_phase_analysis()
            
            # Determinar fase de mercado con análisis completo
            if volatility < 0.3 and abs(trend_strength) < 0.5:
                market_phase = "consolidation"
            elif abs(trend_strength) > 2.0:
                market_phase = "strong_trend"
            elif abs(trend_strength) > 1.0:
                market_phase = "trending"
            elif volatility > 1.5:
                market_phase = "high_volatility"
            elif phase_analysis.get('momentum_shift', False):
                market_phase = "reversal_potential"
            else:
                market_phase = "normal"
            
            return {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volatility': volatility,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'pressure_ratio': pressure_ratio,
                'market_phase': market_phase,
                'data_quality': min(1.0, self.tick_count / 25.0),
                'velocity': avg_velocity,
                'phase_analysis': phase_analysis,
                'candle_progress': (time.time() - self.candle_start_time) / TIMEFRAME if self.candle_start_time else 0,
                'total_ticks': self.tick_count
            }
        except Exception as e:
            logging.error(f"Error en cálculo de métricas avanzadas: {e}")
            return {}
    
    def _combine_phase_analysis(self):
        """Combina análisis de todas las fases de la vela ORIGINAL"""
        try:
            initial = self.analysis_phases['initial']['analysis']
            middle = self.analysis_phases['middle']['analysis']
            final = self.analysis_phases['final']['analysis']
            
            combined = {
                'initial_trend': initial.get('trend', 'N/A'),
                'middle_trend': middle.get('trend', 'N/A'),
                'final_trend': final.get('trend', 'N/A'),
                'momentum_shift': False,
                'consistency_score': 0
            }
            
            # Detectar cambios de momentum
            trends = [initial.get('trend'), middle.get('trend'), final.get('trend')]
            if len(set(trends)) > 1:  # Si hay diferentes tendencias
                combined['momentum_shift'] = True
            
            # Calcular consistencia
            same_trend_count = sum(1 for i in range(len(trends)-1) if trends[i] == trends[i+1])
            combined['consistency_score'] = same_trend_count / max(1, len(trends)-1)
            
            return combined
        except Exception as e:
            logging.debug(f"Error combinando análisis de fases: {e}")
            return {}
    
    def get_comprehensive_analysis(self):
        """Análisis completo ORIGINAL MEJORADO"""
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'INSUFFICIENT_DATA', 
                'tick_count': self.tick_count,
                'message': f'Recolectando ticks: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION}'
            }
        
        try:
            advanced_metrics = self._calculate_advanced_metrics()
            if not advanced_metrics:
                return {'status': 'ERROR', 'message': 'Error en métricas'}
            
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
                **advanced_metrics
            }
            
        except Exception as e:
            logging.error(f"Error en análisis completo: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_recent_ticks(self, n=60):
        """Para compatibilidad con AutoLearning"""
        return [tick['price'] for tick in list(self.ticks)[-n:]]
    
    def reset(self):
        """Reinicia el análisis para nueva vela ORIGINAL"""
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
            
            # Reiniciar análisis de fases
            for phase in self.analysis_phases:
                self.analysis_phases[phase] = {'ticks': 0, 'analysis': {}}
                
        except Exception as e:
            logging.error(f"Error en reset: {e}")

# ------------------ ADAPTIVE MARKET LEARNER (NUEVO - MEJORADO) ------------------
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class AdaptiveMarketLearner:
    """
    Aprendizaje incremental MEJORADO con métricas avanzadas
    """
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
        # Inicialización con datos dummy
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
        """Persiste modelo y scaler"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.training_count % 10 == 0:
                logging.info(f"💾 Modelo persistido (entrenamientos: {self.training_count})")
        except Exception as e:
            logging.error(f"❌ Error guardando modelo: {e}")

    def add_sample(self, features: np.ndarray, label: str):
        """Añade muestra al buffer de entrenamiento"""
        if features.shape[0] == self.feature_size:
            self.replay_buffer.append((features.astype(float), label))

    def partial_train(self, batch_size=32):
        """Entrenamiento incremental MEJORADO"""
        if len(self.replay_buffer) < 10:
            return {"trained": False, "reason": "not_enough_samples", "buffer_size": len(self.replay_buffer)}
        
        # Tomar muestras más recientes
        samples = list(self.replay_buffer)[-batch_size:]
        X = np.vstack([s[0] for s in samples])
        y = np.array([s[1] for s in samples])
        
        try:
            # Entrenar scaler
            if hasattr(self.scaler, "partial_fit"):
                self.scaler.partial_fit(X)
            else:
                self.scaler.fit(X)
            Xs = self.scaler.transform(X)
            
            # Entrenar modelo
            self.model.partial_fit(Xs, y, classes=self.classes)
            self.training_count += 1
            
            # Persistir periódicamente
            if self.training_count % 5 == 0:
                self.persist()
                
            return {
                "trained": True, 
                "n_samples": len(samples),
                "training_count": self.training_count,
                "buffer_size": len(self.replay_buffer)
            }
        except Exception as e:
            logging.error(f"❌ Error en entrenamiento: {e}")
            return {"trained": False, "reason": str(e)}

    def predict_proba(self, features: np.ndarray):
        """Predicción de probabilidades MEJORADA"""
        X = np.atleast_2d(features.astype(float))
        try:
            Xs = self.scaler.transform(X)
            probs = self.model.predict_proba(Xs)[0]
            return dict(zip(self.model.classes_, probs))
        except Exception as e:
            logging.warning(f"⚠️ Fallback en predict_proba: {e}")
            return dict(zip(self.classes, np.ones(len(self.classes)) / len(self.classes)))

    def predict(self, features: np.ndarray):
        """Predicción completa MEJORADA"""
        try:
            X = np.atleast_2d(features.astype(float))
            Xs = self.scaler.transform(X)
            predicted = self.model.predict(Xs)[0]
            proba = self.predict_proba(features)
            confidence = max(proba.values()) * 100
            
            return {
                "predicted": predicted,
                "proba": proba,
                "confidence": round(confidence, 2),
                "training_count": self.training_count
            }
        except Exception as e:
            logging.error(f"❌ Error en predict: {e}")
            return {
                "predicted": "LATERAL",
                "proba": dict(zip(self.classes, [1/3]*3)),
                "confidence": 33.3,
                "training_count": self.training_count
            }

# ------------------ FEATURE BUILDER MEJORADO ------------------
def build_advanced_features_from_analysis(analysis, seconds_remaining, tick_window=30):
    """
    Construye features AVANZADOS combinando análisis tradicional + métricas ML
    """
    try:
        # Features básicos de precio
        if analysis.get('status') != 'SUCCESS':
            return np.zeros(18)
            
        current_price = analysis.get('current_price', 0)
        tick_count = analysis.get('tick_count', 0)
        
        # Features de tendencia y momentum
        trend_strength = analysis.get('trend_strength', 0)
        momentum = analysis.get('momentum', 0)
        volatility = analysis.get('volatility', 0)
        
        # Features de presión de mercado
        buy_pressure = analysis.get('buy_pressure', 0.5)
        sell_pressure = analysis.get('sell_pressure', 0.5)
        pressure_ratio = analysis.get('pressure_ratio', 1.0)
        
        # Features de velocidad y fase
        velocity = analysis.get('velocity', 0)
        candle_progress = analysis.get('candle_progress', 0)
        
        # Features de análisis de fases
        phase_analysis = analysis.get('phase_analysis', {})
        momentum_shift = 1.0 if phase_analysis.get('momentum_shift', False) else 0.0
        consistency_score = phase_analysis.get('consistency_score', 0)
        
        # Features de tiempo
        time_remaining = seconds_remaining / TIMEFRAME
        
        # Construir vector de features
        features = np.array([
            current_price,
            trend_strength,
            momentum,
            volatility,
            buy_pressure,
            sell_pressure,
            pressure_ratio,
            velocity,
            candle_progress,
            momentum_shift,
            consistency_score,
            time_remaining,
            tick_count / 100.0,  # Normalizado
            analysis.get('data_quality', 0),
            # Features adicionales para robustez
            analysis.get('candle_range', 0),
            min(1.0, tick_count / 50.0),  # Saturation feature
            np.log1p(abs(trend_strength)),  # Log feature
            np.sqrt(abs(momentum))  # Sqrt feature
        ]).astype(float)
        
        # Asegurar tamaño consistente
        if features.shape[0] < 18:
            features = np.pad(features, (0, 18 - features.shape[0]))
        elif features.shape[0] > 18:
            features = features[:18]
            
        return features
        
    except Exception as e:
        logging.error(f"❌ Error construyendo features avanzados: {e}")
        return np.zeros(18)

# ------------------ SISTEMA IA PROFESIONAL COMPLETO ------------------
class ComprehensiveAIPredictor:
    def __init__(self):
        self.analyzer = PremiumAIAnalyzer()
        self.prediction_history = deque(maxlen=20)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': 0.0,
            'phase_analysis_count': 0
        }
        self.last_prediction = None
        self.last_validation_result = None
        
    def process_tick(self, price: float, seconds_remaining: float = None):
        try:
            tick_data = self.analyzer.add_tick(price, seconds_remaining)
            return {
                "tick_count": self.analyzer.tick_count,
                "status": "PROCESSED"
            }
        except Exception as e:
            logging.error(f"Error en process_tick: {e}")
            return None
    
    def _comprehensive_ai_analysis(self, analysis, ml_prediction=None):
        """Análisis de IA ORIGINAL MEJORADO con ML"""
        try:
            # Métricas tradicionales
            momentum = analysis['momentum']
            trend_strength = analysis['trend_strength']
            pressure_ratio = analysis['pressure_ratio']
            volatility = analysis['volatility']
            market_phase = analysis['market_phase']
            data_quality = analysis['data_quality']
            phase_analysis = analysis.get('phase_analysis', {})
            candle_progress = analysis.get('candle_progress', 0)
            
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # Integrar predicción ML si está disponible
            ml_boost = 0
            if ml_prediction and ml_prediction.get('confidence', 0) > 60:
                ml_direction = ml_prediction.get('predicted', 'LATERAL')
                ml_confidence = ml_prediction.get('confidence', 0) / 100.0
                ml_boost = ml_confidence * 0.3  # 30% de boost por ML confiable
                reasons.append(f"🤖 ML confirma {ml_direction} ({ml_prediction['confidence']}%)")
            
            # Peso basado en progreso de la vela ORIGINAL
            late_phase_weight = 1.0 if candle_progress > 0.8 else 0.7
            
            # Tendencia (35% de peso)
            trend_weight = 0.35 * late_phase_weight
            if abs(trend_strength) > 1.0:
                if trend_strength > 0:
                    buy_score += 8 * trend_weight + ml_boost
                    reasons.append(f"📈 Tendencia alcista fuerte ({trend_strength:.1f})")
                else:
                    sell_score += 8 * trend_weight + ml_boost
                    reasons.append(f"📉 Tendencia bajista fuerte ({trend_strength:.1f})")
            
            # Momentum (30% de peso)
            momentum_weight = 0.30 * late_phase_weight
            if abs(momentum) > 0.8:
                if momentum > 0:
                    buy_score += 7 * momentum_weight
                    reasons.append(f"🚀 Momentum alcista fuerte ({momentum:.1f}pips)")
                else:
                    sell_score += 7 * momentum_weight
                    reasons.append(f"🔻 Momentum bajista fuerte ({momentum:.1f}pips)")
            
            # Análisis de fases (15% de peso)
            phase_weight = 0.15 * late_phase_weight
            if phase_analysis.get('momentum_shift', False):
                current_trend = phase_analysis.get('final_trend', 'N/A')
                if current_trend == 'ALCISTA':
                    buy_score += 4 * phase_weight
                    reasons.append("🔄 Cambio de momentum a alcista")
                elif current_trend == 'BAJISTA':
                    sell_score += 4 * phase_weight
                    reasons.append("🔄 Cambio de momentum a bajista")
            
            # Presión de compra/venta (20% de peso)
            pressure_weight = 0.20 * late_phase_weight
            if pressure_ratio > 2.0:
                buy_score += 6 * pressure_weight
                reasons.append(f"💰 Fuerte presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.5:
                sell_score += 6 * pressure_weight
                reasons.append(f"💸 Fuerte presión vendedora ({pressure_ratio:.1f}x)")
            
            score_difference = buy_score - sell_score
            
            # Umbral dinámico con boost de ML
            base_threshold = 0.4
            if ml_prediction and ml_prediction.get('confidence', 0) > 70:
                base_threshold = 0.3  # Más sensible con ML confiable
            
            confidence_threshold = base_threshold - (0.1 * (1 - data_quality))
            
            if abs(score_difference) > confidence_threshold:
                if score_difference > 0:
                    direction = "ALZA"
                    base_confidence = 55 + (score_difference * 40)
                else:
                    direction = "BAJA" 
                    base_confidence = 55 + (abs(score_difference) * 40)
            else:
                direction = "LATERAL"
                base_confidence = 40
                reasons.append("⚡ Señales mixtas o insuficientes")
            
            # Ajustar confianza con ML
            confidence = base_confidence
            confidence *= data_quality
            
            # Bonus por alta calidad de datos
            if analysis['tick_count'] > 40:
                confidence = min(90, confidence + 15)
                reasons.append("📊 Alta calidad de datos (muchos ticks)")
            
            confidence = max(35, min(90, confidence))
            
            return {
                'direction': direction,
                'confidence': int(confidence),
                'buy_score': round(buy_score, 2),
                'sell_score': round(sell_score, 2),
                'score_difference': round(score_difference, 2),
                'reasons': reasons,
                'market_phase': market_phase,
                'candle_progress': round(candle_progress, 2),
                'phase_analysis': phase_analysis,
                'ml_boost': round(ml_boost, 2)
            }
        except Exception as e:
            logging.error(f"Error en análisis IA comprehensivo: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 35,
                'reasons': ['🤖 Error en análisis comprehensivo'],
                'buy_score': 0,
                'sell_score': 0,
                'score_difference': 0
            }
    
    def predict_next_candle(self, ml_prediction=None):
        try:
            analysis = self.analyzer.get_comprehensive_analysis()
            
            if analysis.get('status') != 'SUCCESS':
                return {
                    'direction': 'LATERAL',
                    'confidence': 0,
                    'reason': analysis.get('message', 'Analizando...'),
                    'timestamp': now_iso()
                }
            
            prediction = self._comprehensive_ai_analysis(analysis, ml_prediction)
            
            # Solo emitir predicción si tenemos suficiente confianza
            if prediction['confidence'] < 45:
                prediction['direction'] = 'LATERAL'
                prediction['reasons'].append("🔍 Confianza insuficiente para predicción direccional")
            
            prediction.update({
                'tick_count': analysis['tick_count'],
                'current_price': analysis['current_price'],
                'candle_range': analysis.get('candle_range', 0),
                'timestamp': now_iso(),
                'model_version': 'COMPREHENSIVE_AI_V5.4_HYBRID'
            })
            
            # Actualizar historial y estadísticas
            self.last_prediction = prediction
            self.performance_stats['total_predictions'] += 1
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error en predict_next_candle: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 0,
                'reason': f'Error: {str(e)}',
                'timestamp': now_iso()
            }
    
    def reset_for_new_candle(self):
        """Reinicia el análisis para nueva vela"""
        try:
            self.analyzer.reset()
            logging.info("🔄 Sistema reiniciado para nueva vela")
        except Exception as e:
            logging.error(f"Error en reset_for_new_candle: {e}")

# ------------------ IQ OPTION MANAGER MEJORADO ------------------
class IQOptionManager:
    def __init__(self, email, password, pair, timeframe):
        self.email = email
        self.password = password
        self.pair = pair
        self.timeframe = timeframe
        self.api = None
        self.connected = False
        self.last_candle_time = None
        self.candle_callbacks = []
        self.tick_callbacks = []
        
    def connect(self):
        """Conecta a IQ Option"""
        if not IQ_OPTION_AVAILABLE:
            logging.error("❌ IQ Option API no disponible")
            return False
            
        try:
            self.api = IQ_Option(self.email, self.password)
            check, reason = self.api.connect()
            
            if check:
                self.connected = True
                logging.info(f"✅ Conectado a IQ Option - Par: {self.pair}")
                return True
            else:
                logging.error(f"❌ Error de conexión: {reason}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error en conexión: {e}")
            return False
    
    def subscribe_to_candles(self):
        """Suscribe a velas"""
        if not self.connected:
            return False
            
        try:
            self.api.start_candles_stream(self.pair, self.timeframe)
            logging.info(f"📊 Suscrito a velas de {self.pair} - Timeframe: {self.timeframe}")
            return True
        except Exception as e:
            logging.error(f"❌ Error suscribiendo a velas: {e}")
            return False
    
    def subscribe_to_ticks(self):
        """Suscribe a ticks"""
        if not self.connected:
            return False
            
        try:
            self.api.start_mood_stream(self.pair)
            logging.info(f"🔔 Suscrito a ticks de {self.pair}")
            return True
        except Exception as e:
            logging.error(f"❌ Error suscribiendo a ticks: {e}")
            return False
    
    def get_candles(self):
        """Obtiene velas históricas"""
        if not self.connected:
            return None
            
        try:
            candles = self.api.get_candles(self.pair, self.timeframe, 100, time.time())
            return candles
        except Exception as e:
            logging.error(f"❌ Error obteniendo velas: {e}")
            return None
    
    def add_candle_callback(self, callback):
        """Añade callback para nuevas velas"""
        self.candle_callbacks.append(callback)
    
    def add_tick_callback(self, callback):
        """Añade callback para nuevos ticks"""
        self.tick_callbacks.append(callback)
    
    def start_streaming(self):
        """Inicia streaming en hilo separado"""
        def stream_worker():
            while self.connected:
                try:
                    # Procesar ticks
                    ticks = self.api.get_realtime_candles(self.pair, self.timeframe)
                    if ticks:
                        for tick in ticks:
                            for callback in self.tick_callbacks:
                                callback(tick)
                    
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"❌ Error en streaming: {e}")
                    time.sleep(1)
        
        threading.Thread(target=stream_worker, daemon=True).start()
        logging.info("🎯 Streaming iniciado")

# ------------------ FASTAPI SERVER MEJORADO ------------------
app = FastAPI(title="Delowyss AI Premium V5.4", version="5.4.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias globales
ai_predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner()
iq_manager = None

# Estado del sistema
system_status = {
    "status": "initializing",
    "version": "V5.4 PREMIUM COMPLETA",
    "start_time": now_iso(),
    "features": {
        "premium_analyzer": True,
        "adaptive_learning": True,
        "iq_option_integration": IQ_OPTION_AVAILABLE,
        "real_time_predictions": True
    }
}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal"""
    return f"""
    <html>
        <head>
            <title>Delowyss AI Premium V5.4</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #0f0f23; color: #00ff00; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; padding: 20px; background: #1a1a2e; border-radius: 10px; }}
                .status {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .ready {{ background: #1a472a; }}
                .warning {{ background: #5d4037; }}
                .error {{ background: #4a235a; }}
                .endpoints {{ margin-top: 30px; }}
                .endpoint {{ padding: 10px; margin: 5px 0; background: #162447; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Delowyss AI Premium V5.4</h1>
                    <p>Sistema de Trading con IA Avanzada + AutoLearning</p>
                    <p>CEO: Eduardo Solis — © 2025</p>
                </div>
                
                <div class="status {system_status['status']}">
                    <h3>Estado del Sistema: {system_status['status'].upper()}</h3>
                    <p>Versión: {system_status['version']}</p>
                    <p>Inicio: {system_status['start_time']}</p>
                </div>
                
                <div class="endpoints">
                    <h3>Endpoints Disponibles:</h3>
                    <div class="endpoint"><strong>GET /status</strong> - Estado del sistema</div>
                    <div class="endpoint"><strong>GET /analysis</strong> - Análisis actual</div>
                    <div class="endpoint"><strong>GET /prediction</strong> - Predicción actual</div>
                    <div class="endpoint"><strong>GET /history</strong> - Historial de predicciones</div>
                    <div class="endpoint"><strong>POST /tick</strong> - Enviar tick manual</div>
                    <div class="endpoint"><strong>GET /ml/status</strong> - Estado del AutoLearning</div>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/status")
async def get_status():
    """Estado del sistema"""
    return {
        **system_status,
        "ai_predictor": {
            "tick_count": ai_predictor.analyzer.tick_count,
            "total_predictions": ai_predictor.performance_stats['total_predictions'],
            "last_prediction_time": ai_predictor.last_prediction.get('timestamp') if ai_predictor.last_prediction else None
        },
        "online_learner": {
            "training_count": online_learner.training_count,
            "buffer_size": len(online_learner.replay_buffer),
            "model_loaded": os.path.exists(online_learner.model_path)
        },
        "iq_option": {
            "connected": iq_manager.connected if iq_manager else False,
            "pair": PAR,
            "timeframe": TIMEFRAME
        }
    }

@app.get("/analysis")
async def get_analysis():
    """Análisis actual del mercado"""
    analysis = ai_predictor.analyzer.get_comprehensive_analysis()
    return analysis

@app.get("/prediction")
async def get_prediction():
    """Obtener predicción actual"""
    # Obtener análisis actual
    analysis = ai_predictor.analyzer.get_comprehensive_analysis()
    
    # Generar predicción ML si hay datos suficientes
    ml_prediction = None
    if analysis.get('status') == 'SUCCESS':
        features = build_advanced_features_from_analysis(
            analysis, 
            seconds_remaining=0  # Se puede ajustar según el contexto
        )
        ml_prediction = online_learner.predict(features)
    
    # Generar predicción comprehensiva
    prediction = ai_predictor.predict_next_candle(ml_prediction)
    
    return {
        "prediction": prediction,
        "ml_prediction": ml_prediction,
        "analysis_status": analysis.get('status'),
        "timestamp": now_iso()
    }

@app.get("/history")
async def get_history():
    """Historial de predicciones"""
    return {
        "performance": ai_predictor.performance_stats,
        "recent_predictions": list(ai_predictor.prediction_history),
        "last_prediction": ai_predictor.last_prediction
    }

@app.post("/tick")
async def add_tick(tick_data: dict):
    """Añadir tick manualmente"""
    try:
        price = float(tick_data.get('price', 0))
        seconds_remaining = tick_data.get('seconds_remaining')
        
        result = ai_predictor.process_tick(price, seconds_remaining)
        
        if result:
            return {"status": "success", "tick_count": result['tick_count']}
        else:
            return {"status": "error", "message": "Error procesando tick"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/ml/status")
async def get_ml_status():
    """Estado del AutoLearning"""
    return {
        "training_count": online_learner.training_count,
        "buffer_size": len(online_learner.replay_buffer),
        "model_path": online_learner.model_path,
        "last_training": online_learner.partial_train(),
        "features_size": online_learner.feature_size
    }

@app.post("/ml/train")
async def train_ml():
    """Forzar entrenamiento del modelo ML"""
    result = online_learner.partial_train()
    return result

# ------------------ INICIALIZACIÓN DEL SISTEMA ------------------
def initialize_system():
    """Inicializa el sistema completo"""
    global SYSTEM_READY, iq_manager
    
    logging.info("🚀 Iniciando Delowyss AI Premium V5.4...")
    
    # Inicializar IQ Option Manager si hay credenciales
    if IQ_EMAIL and IQ_PASSWORD and IQ_OPTION_AVAILABLE:
        iq_manager = IQOptionManager(IQ_EMAIL, IQ_PASSWORD, PAR, TIMEFRAME)
        if iq_manager.connect():
            iq_manager.subscribe_to_ticks()
            
            # Configurar callback para ticks
            def handle_tick(tick_data):
                try:
                    price = tick_data.get('close', tick_data.get('price'))
                    if price:
                        ai_predictor.process_tick(float(price))
                except Exception as e:
                    logging.error(f"Error procesando tick: {e}")
            
            iq_manager.add_tick_callback(handle_tick)
            iq_manager.start_streaming()
    
    # Configurar shutdown graceful
    atexit.register(handle_shutdown, None, None)
    
    SYSTEM_READY = True
    system_status["status"] = "ready"
    logging.info("✅ Sistema Delowyss AI Premium V5.4 inicializado correctamente")

# Inicializar al importar
if __name__ == "main":
    initialize_system()

# Inicializar cuando se ejecuta directamente
if __name__ == "__main__":
    import uvicorn
    initialize_system()
    logging.info(f"🌐 Servidor iniciado en puerto {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
