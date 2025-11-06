# main.py - V5.5 PREMIUM COMPLETA (IA Avanzada + AutoLearning + Interfaz Original + Mejoras Críticas)
"""
Delowyss Trading AI — V5.5 PREMIUM COMPLETA CON AUTOLEARNING MEJORADO
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

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ NUEVO: CANDLE BUFFER PARA EVITAR FUGA DE DATOS ------------------
class CandleBufferManager:
    """Gestiona el buffer para evitar fugas de datos en el entrenamiento ML"""
    def __init__(self):
        self.pending_features = None
        self.pending_candle_close = None
        self.pending_timestamp = None
        self.last_direction = None
        
    def store_candle_features(self, features, close_price):
        """Almacena features de vela cerrada para entrenamiento futuro"""
        self.pending_features = features.copy() if features is not None else None
        self.pending_candle_close = close_price
        self.pending_timestamp = time.time()
        logging.info(f"📦 Features de vela almacenados para entrenamiento futuro")
        
    def train_with_new_direction(self, new_candle_direction):
        """Entrena con features antiguos y dirección nueva (sin fuga de datos)"""
        if self.pending_features is not None and new_candle_direction is not None:
            online_learner.add_sample(self.pending_features, new_candle_direction)
            training_result = online_learner.partial_train(batch_size=32)
            self.last_direction = new_candle_direction
            logging.info(f"🎓 AutoLearning: {new_candle_direction} | {training_result}")
            self.pending_features = None  # Limpiar después del entrenamiento
            return training_result
        return None

# ------------------ NUEVO: VALIDADOR TIEMPO REAL ------------------
class RealTimePerformanceValidator:
    """Valida y monitoriza performance en tiempo real sin backtesting"""
    def __init__(self, max_history=100):
        self.predictions_history = deque(maxlen=max_history)
        self.performance_metrics = {
            'accuracy_5': 0.0, 'accuracy_10': 0.0, 'accuracy_20': 0.0,
            'accuracy_50': 0.0, 'total_predictions': 0, 'correct_predictions': 0,
            'recent_trend': 'NEUTRAL', 'avg_confidence_correct': 0.0,
            'avg_confidence_wrong': 0.0, 'best_patterns': {}, 'worst_patterns': {}
        }
        self.confidence_correct = []
        self.confidence_wrong = []
        
    def add_prediction_result(self, prediction, actual_direction, market_phase):
        """Añade resultado de predicción para análisis en tiempo real"""
        is_correct = prediction['direction'] == actual_direction
        
        self.predictions_history.append({
            'predicted': prediction['direction'],
            'actual': actual_direction,
            'confidence': prediction['confidence'],
            'correct': is_correct,
            'market_phase': market_phase,
            'timestamp': time.time(),
            'reasons': prediction.get('reasons', [])
        })
        
        # Actualizar métricas básicas
        self.performance_metrics['total_predictions'] += 1
        if is_correct:
            self.performance_metrics['correct_predictions'] += 1
            self.confidence_correct.append(prediction['confidence'])
        else:
            self.confidence_wrong.append(prediction['confidence'])
            
        # Calcular precisiones rolling
        self._update_rolling_accuracies()
        
        # Calcular confianzas promedio
        if self.confidence_correct:
            self.performance_metrics['avg_confidence_correct'] = np.mean(self.confidence_correct[-20:])
        if self.confidence_wrong:
            self.performance_metrics['avg_confidence_wrong'] = np.mean(self.confidence_wrong[-20:])
            
        return is_correct
    
    def _update_rolling_accuracies(self):
        """Actualiza precisiones en ventanas temporales"""
        history = list(self.predictions_history)
        
        for window in [5, 10, 20, 50]:
            if len(history) >= window:
                recent = history[-window:]
                correct = sum(1 for p in recent if p['correct'])
                self.performance_metrics[f'accuracy_{window}'] = (correct / window) * 100
    
    def get_performance_insights(self):
        """Obtiene insights de performance para la interfaz"""
        insights = self.performance_metrics.copy()
        
        # Calcular tendencia reciente
        if len(self.predictions_history) >= 10:
            recent = list(self.predictions_history)[-10:]
            recent_accuracy = sum(1 for p in recent if p['correct']) / len(recent)
            if recent_accuracy > 0.7:
                insights['recent_trend'] = 'MEJORANDO'
            elif recent_accuracy < 0.4:
                insights['recent_trend'] = 'BAJANDO'
            else:
                insights['recent_trend'] = 'ESTABLE'
                
        return insights

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
            
            # NUEVO: Métricas mejoradas para predicción intra-vela
            enhanced_metrics = self._calculate_enhanced_intra_candle_metrics(phase_analysis)
            
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
                'total_ticks': self.tick_count,
                **enhanced_metrics  # NUEVO: Métricas mejoradas
            }
        except Exception as e:
            logging.error(f"Error en cálculo de métricas avanzadas: {e}")
            return {}
    
    def _calculate_enhanced_intra_candle_metrics(self, phase_analysis):
        """NUEVO: Métricas mejoradas para predicción intra-vela"""
        try:
            # Fuerza de tendencia intra-vela
            early_strength = self.analysis_phases['initial']['analysis'].get('trend_strength', 0)
            late_strength = self.analysis_phases['final']['analysis'].get('trend_strength', 0)
            trend_persistence = late_strength - early_strength
            
            # Aceleración en últimos ticks
            acceleration_trend = 0
            if len(self.acceleration_metrics) >= 3:
                recent_accel = [m['acceleration'] for m in list(self.acceleration_metrics)[-3:]]
                acceleration_trend = np.mean(recent_accel) * 10000
            
            # Presión neta en fase final
            final_phase = self.analysis_phases['final']['analysis']
            buy_pressure_final = final_phase.get('buy_pressure', 0.5)
            
            # Consistencia de momentum
            initial_trend = self.analysis_phases['initial']['analysis'].get('trend', 'LATERAL')
            final_trend = self.analysis_phases['final']['analysis'].get('trend', 'LATERAL')
            momentum_consistency = 1.0 if initial_trend == final_trend else 0.0
            
            return {
                'trend_persistence': trend_persistence,
                'acceleration_trend': acceleration_trend,
                'final_buy_pressure': buy_pressure_final,
                'momentum_consistency': momentum_consistency,
                'intra_candle_quality': min(1.0, self.tick_count / 40.0)  # Calidad del análisis intra-vela
            }
        except Exception as e:
            logging.debug(f"Error en métricas mejoradas: {e}")
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

# ------------------ ADAPTIVE MARKET LEARNER (MEJORADO SIN FUGAS) ------------------
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class AdaptiveMarketLearner:
    """
    Aprendizaje incremental MEJORADO sin fugas de datos
    """
    def __init__(self, feature_size=22, classes=None, buffer_size=1000):  # Aumentado por nuevas métricas
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
        """Añade muestra al buffer de entrenamiento SIN FUGAS"""
        if features.shape[0] == self.feature_size:
            self.replay_buffer.append((features.astype(float), label))
            logging.debug(f"📚 Muestra añadida: {label} (Total: {len(self.replay_buffer)})")

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
                "buffer_size": len(self.replay_buffer),
                "accuracy_self": self._calculate_self_accuracy(samples)
            }
        except Exception as e:
            logging.error(f"❌ Error en entrenamiento: {e}")
            return {"trained": False, "reason": str(e)}
    
    def _calculate_self_accuracy(self, samples):
        """Calcula precisión en muestras de entrenamiento"""
        try:
            X = np.vstack([s[0] for s in samples])
            y = np.array([s[1] for s in samples])
            Xs = self.scaler.transform(X)
            predictions = self.model.predict(Xs)
            accuracy = np.mean(predictions == y)
            return round(accuracy * 100, 1)
        except:
            return 0.0

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
                "training_count": self.training_count,
                "buffer_size": len(self.replay_buffer)
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
    INCLUYE nuevas métricas intra-vela
    """
    try:
        # Features básicos de precio
        if analysis.get('status') != 'SUCCESS':
            return np.zeros(22)  # Aumentado por nuevas métricas
            
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
        
        # NUEVOS: Features intra-vela mejorados
        trend_persistence = analysis.get('trend_persistence', 0)
        acceleration_trend = analysis.get('acceleration_trend', 0)
        final_buy_pressure = analysis.get('final_buy_pressure', 0.5)
        momentum_consistency = analysis.get('momentum_consistency', 0)
        intra_candle_quality = analysis.get('intra_candle_quality', 0)
        
        # Features de tiempo
        time_remaining = seconds_remaining / TIMEFRAME
        
        # Construir vector de features MEJORADO
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
            np.sqrt(abs(momentum)),  # Sqrt feature
            # NUEVOS: Features intra-vela
            trend_persistence,
            acceleration_trend,
            final_buy_pressure,
            momentum_consistency
        ]).astype(float)
        
        # Asegurar tamaño consistente
        if features.shape[0] < 22:
            features = np.pad(features, (0, 22 - features.shape[0]))
        elif features.shape[0] > 22:
            features = features[:22]
            
        return features
        
    except Exception as e:
        logging.error(f"❌ Error construyendo features avanzados: {e}")
        return np.zeros(22)

# ------------------ SISTEMA IA PROFESIONAL COMPLETO MEJORADO ------------------
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
        """Análisis de IA ORIGINAL MEJORADO con ML y métricas intra-vela"""
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
            
            # NUEVAS: Métricas intra-vela mejoradas
            trend_persistence = analysis.get('trend_persistence', 0)
            acceleration_trend = analysis.get('acceleration_trend', 0)
            final_buy_pressure = analysis.get('final_buy_pressure', 0.5)
            momentum_consistency = analysis.get('momentum_consistency', 0)
            intra_candle_quality = analysis.get('intra_candle_quality', 0)
            
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
            
            # NUEVO: Peso basado en calidad intra-vela
            quality_weight = intra_candle_quality
            
            # Tendencia (25% de peso)
            trend_weight = 0.25 * late_phase_weight * quality_weight
            if abs(trend_strength) > 1.0:
                if trend_strength > 0:
                    buy_score += 8 * trend_weight + ml_boost
                    reasons.append(f"📈 Tendencia alcista fuerte ({trend_strength:.1f})")
                else:
                    sell_score += 8 * trend_weight + ml_boost
                    reasons.append(f"📉 Tendencia bajista fuerte ({trend_strength:.1f})")
            
            # Persistencia de tendencia intra-vela (NUEVO - 15% peso)
            persistence_weight = 0.15 * late_phase_weight
            if abs(trend_persistence) > 0.5:
                if trend_persistence > 0:
                    buy_score += 6 * persistence_weight
                    reasons.append(f"🔄 Persistencia alcista ({trend_persistence:.1f})")
                else:
                    sell_score += 6 * persistence_weight
                    reasons.append(f"🔄 Persistencia bajista ({trend_persistence:.1f})")
            
            # Momentum (20% de peso)
            momentum_weight = 0.20 * late_phase_weight * quality_weight
            if abs(momentum) > 0.8:
                if momentum > 0:
                    buy_score += 7 * momentum_weight
                    reasons.append(f"🚀 Momentum alcista fuerte ({momentum:.1f}pips)")
                else:
                    sell_score += 7 * momentum_weight
                    reasons.append(f"🔻 Momentum bajista fuerte ({momentum:.1f}pips)")
            
            # Aceleración intra-vela (NUEVO - 10% peso)
            acceleration_weight = 0.10 * late_phase_weight
            if abs(acceleration_trend) > 0.2:
                if acceleration_trend > 0:
                    buy_score += 4 * acceleration_weight
                    reasons.append(f"⚡ Aceleración alcista ({acceleration_trend:.1f})")
                else:
                    sell_score += 4 * acceleration_weight
                    reasons.append(f"⚡ Aceleración bajista ({acceleration_trend:.1f})")
            
            # Análisis de fases (10% de peso)
            phase_weight = 0.10 * late_phase_weight
            if phase_analysis.get('momentum_shift', False):
                current_trend = phase_analysis.get('final_trend', 'N/A')
                if current_trend == 'ALCISTA':
                    buy_score += 4 * phase_weight
                    reasons.append("🔄 Cambio de momentum a alcista")
                elif current_trend == 'BAJISTA':
                    sell_score += 4 * phase_weight
                    reasons.append("🔄 Cambio de momentum a bajista")
            
            # Consistencia de momentum (NUEVO - 5% peso)
            consistency_weight = 0.05 * late_phase_weight
            if momentum_consistency > 0.8:
                current_trend = phase_analysis.get('final_trend', 'N/A')
                if current_trend == 'ALCISTA':
                    buy_score += 3 * consistency_weight
                    reasons.append("🎯 Momentum consistente alcista")
                elif current_trend == 'BAJISTA':
                    sell_score += 3 * consistency_weight
                    reasons.append("🎯 Momentum consistente bajista")
            
            # Presión de compra/venta (15% de peso)
            pressure_weight = 0.15 * late_phase_weight * quality_weight
            if pressure_ratio > 2.0:
                buy_score += 6 * pressure_weight
                reasons.append(f"💰 Fuerte presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.5:
                sell_score += 6 * pressure_weight
                reasons.append(f"💸 Fuerte presión vendedora ({pressure_ratio:.1f}x)")
            
            score_difference = buy_score - sell_score
            
            # Umbral dinámico con boost de ML y calidad intra-vela
            base_threshold = 0.4
            if ml_prediction and ml_prediction.get('confidence', 0) > 70:
                base_threshold = 0.3  # Más sensible con ML confiable
            
            # Ajustar umbral por calidad de análisis intra-vela
            quality_adjusted_threshold = base_threshold - (0.1 * (1 - intra_candle_quality))
            confidence_threshold = quality_adjusted_threshold - (0.1 * (1 - data_quality))
            
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
            
            # Ajustar confianza con calidad de datos e intra-vela
            confidence = base_confidence
            confidence *= data_quality
            confidence *= (0.7 + 0.3 * intra_candle_quality)  # Boost por calidad intra-vela
            
            # Bonus por alta calidad de datos
            if analysis['tick_count'] > 40:
                confidence = min(90, confidence + 15)
                reasons.append("📊 Alta calidad de datos (muchos ticks)")
            
            confidence = max(35, min(90, confidence))
            
            return {
                'direction': direction,
                'confidence': int(confidence),
                'buy_score': round(buy_score * 100, 2),  # Convertir a porcentaje
                'sell_score': round(sell_score * 100, 2),  # Convertir a porcentaje
                'score_difference': round(score_difference, 2),
                'reasons': reasons,
                'market_phase': market_phase,
                'candle_progress': round(candle_progress, 2),
                'phase_analysis': phase_analysis,
                'ml_boost': round(ml_boost, 2),
                'intra_candle_quality': round(intra_candle_quality, 2)  # NUEVO
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
                'model_version': 'COMPREHENSIVE_AI_V5.5_HYBRID_ENHANCED'
            })
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            # Log de predicción detallado
            if prediction['direction'] != 'LATERAL':
                ml_info = f" | ML Boost: {prediction.get('ml_boost', 0):.2f}" if ml_prediction else ""
                quality_info = f" | Calidad: {prediction.get('intra_candle_quality', 0):.2f}"
                logging.info(f"🎯 PREDICCIÓN HÍBRIDA MEJORADA: {prediction['direction']} | "
                           f"Conf: {prediction['confidence']}%{ml_info}{quality_info} | "
                           f"Ticks: {analysis['tick_count']}")
            
            return prediction
        except Exception as e:
            logging.error(f"Error en predict_next_candle: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 0,
                'reason': 'Error en predicción',
                'timestamp': now_iso()
            }
    
    def validate_prediction(self, new_candle_open_price):
        """Validación mejorada ORIGINAL"""
        try:
            if not self.last_prediction:
                return None
                
            last_pred = self.last_prediction
            predicted_direction = last_pred.get('direction', 'N/A')
            
            previous_close = self.analyzer.last_candle_close
            current_open = new_candle_open_price
            
            if previous_close is None or current_open is None:
                return None
                
            price_change = (current_open - previous_close) * 10000
            
            # Umbral dinámico basado en el rango de la vela anterior
            candle_range = last_pred.get('candle_range', 0.5)
            minimal_change = max(0.15, candle_range * 0.2)
            
            if abs(price_change) < minimal_change:
                actual_direction = "LATERAL"
                is_correct = False
            else:
                if price_change > 0:
                    actual_direction = "ALZA"
                else:
                    actual_direction = "BAJA"
                
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
            
            logging.info(f"🎯 VALIDACIÓN: {status_icon} {predicted_direction}→{actual_direction} | "
                        f"Conf: {last_pred.get('confidence', 0)}% | "
                        f"Cambio: {price_change:.1f}pips | "
                        f"Rango vela: {candle_range:.1f}pips")
            
            if total > 0 and total % 5 == 0:
                logging.info(f"📊 PRECISIÓN ACUMULADA: {accuracy:.1f}% (Total: {total})")
            
            self.last_validation_result = {
                'correct': is_correct,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': last_pred.get('confidence', 0),
                'price_change': round(price_change, 2),
                'candle_range': round(candle_range, 2),
                'accuracy': round(accuracy, 1),
                'total_predictions': total,
                'correct_predictions': correct,
                'status_icon': status_icon,
                'timestamp': now_iso()
            }
            
            return self.last_validation_result
            
        except Exception as e:
            logging.error(f"Error en validación: {e}")
            return None
    
    def get_performance_stats(self):
        return self.performance_stats.copy()
    
    def get_last_validation(self):
        return self.last_validation_result
    
    def reset(self):
        try:
            self.analyzer.reset()
        except Exception as e:
            logging.error(f"Error en reset predictor: {e}")

# ------------------ CONEXIÓN PROFESIONAL MEJORADA ------------------
class ProfessionalIQConnector:
    def __init__(self):
        self.connected = False
        self.tick_listeners = []
        self.last_price = 1.10000
        self.tick_count = 0
        self.simulation_mode = not IQ_OPTION_AVAILABLE
        
    def connect(self):
        if self.simulation_mode:
            logging.info("🔧 MODO SIMULACIÓN ACTIVADO - IQ Option no disponible")
            self.connected = True
            # Iniciar simulador de ticks
            thread = threading.Thread(target=self._simulate_ticks, daemon=True)
            thread.start()
            return True
            
        try:
            logging.info("🌐 Conectando a IQ Option...")
            self.api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.api.connect()
            
            if check:
                self.api.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conexión IQ Option establecida")
                return True
            else:
                logging.warning(f"⚠️ Conexión IQ Option fallida: {reason}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error de conexión IQ Option: {e}")
            return False

    def _simulate_ticks(self):
        """Simulador de ticks realista"""
        base_price = 1.10000
        volatility = 0.0001
        
        while True:
            # Random walk con reversión a la media
            change = np.random.normal(0, volatility)
            base_price += change
            # Suavizar movimientos
            base_price = base_price * 0.999 + 1.10000 * 0.001
            
            self.last_price = base_price
            self.tick_count += 1
            
            # Notificar listeners
            timestamp = time.time()
            for listener in self.tick_listeners:
                try:
                    listener(self.last_price, timestamp)
                except Exception as e:
                    logging.error(f"Error en listener: {e}")
            
            time.sleep(0.1)  # 10 ticks por segundo

    def add_tick_listener(self, listener):
        self.tick_listeners.append(listener)

    def get_realtime_price(self):
        if self.simulation_mode:
            return float(self.last_price)
            
        try:
            if hasattr(self, 'api') and self.connected:
                candles = self.api.get_candles(PAR, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        return price
            return float(self.last_price)
        except Exception as e:
            logging.error(f"Error obteniendo precio real: {e}")
            return float(self.last_price)

# --------------- SISTEMA PRINCIPAL MEJORADO ---------------
iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner(feature_size=22)

# NUEVOS: Sistemas de gestión mejorados
candle_buffer = CandleBufferManager()
performance_validator = RealTimePerformanceValidator()

# VARIABLES GLOBALES
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 Sistema inicializando..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING",
    "candle_progress": 0,
    "market_phase": "N/A",
    "buy_score": 0,
    "sell_score": 0,
    "ai_model_predicted": "N/A",
    "ml_confidence": 0,
    "training_count": 0,
    "intra_candle_quality": 0,  # NUEVO
    "performance_metrics": {}   # NUEVO
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None
}

# Estado interno
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False
_last_prediction_time = 0
_last_price = None
_new_candle_direction = None

def tick_processor(price, timestamp):
    """Procesador de ticks MEJORADO con ML integrado"""
    global current_prediction
    try:
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        # Procesar tick en el predictor
        tick_data = predictor.process_tick(price, seconds_remaining)
        
        if tick_data:
            # Obtener análisis completo para features ML
            analysis = predictor.analyzer.get_comprehensive_analysis()
            
            if analysis.get('status') == 'SUCCESS':
                # Construir features avanzados para ML
                features = build_advanced_features_from_analysis(analysis, seconds_remaining)
                
                # Obtener predicción ML
                ml_prediction = online_learner.predict(features)
                
                # Actualizar predicción global
                current_prediction.update({
                    "current_price": float(price),
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE",
                    "buy_score": round(ml_prediction['proba'].get('ALZA', 0) * 100, 2),
                    "sell_score": round(ml_prediction['proba'].get('BAJA', 0) * 100, 2),
                    "ai_model_predicted": ml_prediction['predicted'],
                    "ml_confidence": ml_prediction['confidence'],
                    "training_count": ml_prediction['training_count'],
                    "intra_candle_quality": analysis.get('intra_candle_quality', 0)  # NUEVO
                })
                
    except Exception as e:
        logging.error(f"Error procesando tick: {e}")

def determine_new_candle_direction(initial_price, current_price, time_elapsed):
    """Determina la dirección de la nueva vela después de 10 segundos"""
    if time_elapsed < 10:  # Esperar al menos 10 segundos
        return None
        
    price_change = (current_price - initial_price) * 10000
    minimal_change = 0.3  # 0.3 pips mínimo para considerar dirección
    
    if abs(price_change) < minimal_change:
        return "LATERAL"
    elif price_change > 0:
        return "ALZA"
    else:
        return "BAJA"

def premium_main_loop():
    """Loop principal MEJORADO con AutoLearning integrado SIN FUGAS"""
    global current_prediction, performance_stats, _last_candle_start
    global _prediction_made_this_candle, _last_prediction_time, _last_price
    global _new_candle_direction
    
    logging.info(f"🚀 DELOWYSS AI V5.5 PREMIUM INICIADA EN PUERTO {PORT}")
    logging.info("🎯 Sistema HÍBRIDO MEJORADO: IA Avanzada + AutoLearning Sin Fugas + Interfaz Original")
    
    iq_connector.connect()
    iq_connector.add_tick_listener(tick_processor)

    # Variables para nueva vela
    new_candle_initial_price = None
    new_candle_start_time = None

    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price

            # Actualizar progreso de vela
            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress

            # Lógica de predicción en últimos 5 segundos
            if (seconds_remaining <= PREDICTION_WINDOW and
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - _last_prediction_time) >= 2 and
                not _prediction_made_this_candle):

                logging.info(f"🎯 VENTANA DE PREDICCIÓN: {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                # Obtener análisis completo
                analysis = predictor.analyzer.get_comprehensive_analysis()
                if analysis.get('status') == 'SUCCESS':
                    # Construir features para ML
                    features = build_advanced_features_from_analysis(analysis, seconds_remaining)
                    
                    # Obtener predicción ML
                    ml_prediction = online_learner.predict(features)
                    
                    # Generar predicción híbrida (IA tradicional + ML)
                    hybrid_prediction = predictor.predict_next_candle(ml_prediction)
                    
                    # Actualizar predicción global
                    current_prediction.update(hybrid_prediction)
                    current_prediction.update({
                        "ai_model_predicted": ml_prediction['predicted'],
                        "ml_confidence": ml_prediction['confidence'],
                        "training_count": ml_prediction['training_count']
                    })

                    _last_prediction_time = time.time()
                    _prediction_made_this_candle = True

            # Detectar nueva vela (entrenamiento AutoLearning MEJORADO)
            if current_candle_start > _last_candle_start:
                logging.info("🕯️ DETECTADA NUEVA VELA - Procesando...")
                
                # 1. GUARDAR FEATURES de la vela que acaba de cerrar (sin fugar datos)
                if _last_price is not None:
                    # Obtener análisis final de la vela cerrada
                    closed_analysis = predictor.analyzer.get_comprehensive_analysis()
                    if closed_analysis.get('status') == 'SUCCESS':
                        closed_features = build_advanced_features_from_analysis(closed_analysis, 0)
                        candle_buffer.store_candle_features(closed_features, _last_price)
                        
                        # Validar predicción anterior
                        validation = predictor.validate_prediction(_last_price)
                        if validation:
                            # Actualizar validador en tiempo real
                            is_correct = performance_validator.add_prediction_result(
                                current_prediction, 
                                validation['actual'],
                                closed_analysis.get('market_phase', 'N/A')
                            )
                            
                            # Actualizar estadísticas globales
                            performance_stats['last_validation'] = validation
                            current_prediction['performance_metrics'] = performance_validator.get_performance_insights()
                
                # 2. INICIALIZAR SEGUIMIENTO de nueva vela
                new_candle_initial_price = _last_price
                new_candle_start_time = current_time
                _new_candle_direction = None
                
                # 3. Reiniciar para nueva vela
                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🔄 Análisis completo reiniciado para nueva vela")

            # 4. DETERMINAR DIRECCIÓN de nueva vela después de 10 segundos (entrenamiento SIN FUGAS)
            if (new_candle_initial_price is not None and 
                new_candle_start_time is not None and
                _new_candle_direction is None):
                
                time_elapsed = current_time - new_candle_start_time
                new_candle_direction = determine_new_candle_direction(
                    new_candle_initial_price, _last_price, time_elapsed
                )
                
                if new_candle_direction is not None:
                    _new_candle_direction = new_candle_direction
                    # ENTRENAR con features antiguos y dirección nueva (SIN FUGAS)
                    training_result = candle_buffer.train_with_new_direction(new_candle_direction)
                    logging.info(f"🎓 AutoLearning completado: {new_candle_direction} | Tiempo: {time_elapsed:.1f}s")

            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(0.5)

# ------------------ INTERFAZ WEB COMPLETA ORIGINAL MEJORADA ------------------
app = FastAPI(
    title="Delowyss AI Premium V5.5",
    version="5.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=generate_html_interface(), status_code=200)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/validation")
def api_validation():
    last_val = predictor.get_last_validation()
    performance_insights = performance_validator.get_performance_insights()
    return JSONResponse({
        "last_validation": last_val,
        "performance": performance_stats,
        "performance_insights": performance_insights,
        "timestamp": now_iso()
    })

@app.get("/api/health")
def api_health():
    return JSONResponse({
        "status": "healthy",
        "timestamp": now_iso(),
        "version": "5.5.0-hybrid-enhanced",
        "port": PORT,
        "features": [
            "full_candle_analysis", 
            "phase_analysis", 
            "tick_by_tick", 
            "online_learning",
            "hybrid_ai_ml",
            "responsive_interface",
            "no_data_leaks",  # NUEVO
            "enhanced_intra_candle",  # NUEVO
            "realtime_validation"  # NUEVO
        ]
    })

@app.get("/api/system-info")
def api_system_info():
    return JSONResponse({
        "status": "running",
        "pair": PAR,
        "timeframe": TIMEFRAME,
        "prediction_window": PREDICTION_WINDOW,
        "current_ticks": predictor.analyzer.tick_count,
        "ml_training_count": online_learner.training_count,
        "ml_buffer_size": len(online_learner.replay_buffer),
        "performance_metrics": performance_validator.get_performance_insights(),
        "timestamp": now_iso()
    })

def generate_html_interface():
    """Interfaz HTML COMPLETA ORIGINAL MEJORADA"""
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    current_price = current_prediction.get("current_price", 0)
    tick_count = current_prediction.get("tick_count", 0)
    candle_progress = current_prediction.get("candle_progress", 0)
    market_phase = current_prediction.get("market_phase", "N/A")
    ml_predicted = current_prediction.get("ai_model_predicted", "N/A")
    ml_confidence = current_prediction.get("ml_confidence", 0)
    training_count = current_prediction.get("training_count", 0)
    intra_candle_quality = current_prediction.get("intra_candle_quality", 0)
    
    # Obtener métricas de performance
    performance_insights = current_prediction.get('performance_metrics', {})
    accuracy_10 = performance_insights.get('accuracy_10', 0)
    accuracy_20 = performance_insights.get('accuracy_20', 0)
    recent_trend = performance_insights.get('recent_trend', 'NEUTRAL')
    avg_confidence_correct = performance_insights.get('avg_confidence_correct', 0)
    avg_confidence_wrong = performance_insights.get('avg_confidence_wrong', 0)
    
    # Colores dinámicos ORIGINALES
    if direction == "ALZA":
        primary_color = "#00ff88"
        gradient = "linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)"
        status_emoji = "📈"
    elif direction == "BAJA":
        primary_color = "#ff4444"
        gradient = "linear-gradient(135deg, #ff4444 0%, #cc3636 100%)"
        status_emoji = "📉"
    else:
        primary_color = "#ffbb33"
        gradient = "linear-gradient(135deg, #ffbb33 0%, #cc9929 100%)"
        status_emoji = "⚡"
    
    # Calcular nivel de confianza ORIGINAL
    confidence_level = "ALTA" if confidence > 70 else "MEDIA" if confidence > 50 else "BAJA"
    confidence_color = "#00ff88" if confidence > 70 else "#ffbb33" if confidence > 50 else "#ff4444"
    
    # Calcular calidad intra-vela
    quality_level = "EXCELENTE" if intra_candle_quality > 0.8 else "BUENA" if intra_candle_quality > 0.6 else "REGULAR"
    quality_color = "#00ff88" if intra_candle_quality > 0.8 else "#ffbb33" if intra_candle_quality > 0.6 else "#ff4444"
    
    # Generar HTML de razones
    reasons_html = ""
    reasons_list = current_prediction.get('reasons', ['Analizando mercado...'])
    for reason in reasons_list:
        reasons_html += f'<li class="reason-item">{reason}</li>'
    
    # Calcular tiempo hasta siguiente vela
    current_time = time.time()
    seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
    progress_percentage = min(100, max(0, (1 - seconds_remaining/TIMEFRAME) * 100))
    
    # HTML COMPLETO ORIGINAL MEJORADO
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.5</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
                min-height: 100vh;
                padding: 20px;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            /* HEADER ORIGINAL MEJORADO */
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 25px 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }}
            
            .logo {{
                font-size: clamp(2rem, 4vw, 2.8rem);
                font-weight: 700;
                margin-bottom: 10px;
                background: {gradient};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                line-height: 1.2;
            }}
            
            .subtitle {{
                color: #94a3b8;
                font-size: clamp(0.9rem, 2vw, 1.1rem);
                margin-bottom: 15px;
            }}
            
            .version {{
                background: rgba({primary_color.replace('#', '')}, 0.1);
                color: {primary_color};
                padding: 6px 12px;
                border-radius: 15px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
            }}
            
            /* DASHBOARD RESPONSIVE ORIGINAL */
            .dashboard {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            @media (max-width: 768px) {{
                .dashboard {{
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
            }}
            
            /* CARDS ORIGINALES MEJORADAS */
            .card {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }}
            
            .prediction-card {{
                grid-column: 1 / -1;
                text-align: center;
                border-left: 5px solid {primary_color};
                position: relative;
                overflow: hidden;
            }}
            
            .prediction-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: {gradient};
            }}
            
            .direction {{
                font-size: clamp(2.5rem, 6vw, 4rem);
                font-weight: 700;
                color: {primary_color};
                margin: 20px 0;
                text-shadow: 0 0 20px rgba({primary_color.replace('#', '')}, 0.3);
            }}
            
            .confidence {{
                font-size: clamp(1.1rem, 2vw, 1.3rem);
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }}
            
            .confidence-badge {{
                background: {confidence_color};
                color: #0f172a;
                padding: 6px 15px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9em;
                white-space: nowrap;
            }}
            
            /* NUEVO: CALIDAD INTRA-VELA */
            .quality-badge {{
                background: {quality_color};
                color: #0f172a;
                padding: 4px 12px;
                border-radius: 15px;
                font-weight: 600;
                font-size: 0.8em;
                margin-left: 10px;
            }}
            
            /* PROGRESS BAR ORIGINAL */
            .candle-progress {{
                margin: 20px 0;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                overflow: hidden;
                height: 8px;
            }}
            
            .progress-bar {{
                height: 100%;
                background: {gradient};
                width: {progress_percentage}%;
                transition: width 0.5s ease;
                border-radius: 10px;
            }}
            
            .progress-info {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                color: #94a3b8;
                margin-top: 8px;
            }}
            
            /* COUNTDOWN ORIGINAL MEJORADO */
            .countdown {{
                background: rgba(0, 0, 0, 0.3);
                padding: 20px;
                border-radius: 15px;
                margin: 25px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .countdown-number {{
                font-size: clamp(2rem, 5vw, 3rem);
                font-weight: 700;
                color: {primary_color};
                font-family: 'Courier New', monospace;
                margin: 10px 0;
            }}
            
            /* METRICS GRID RESPONSIVE ORIGINAL */
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
                margin: 20px 0;
            }}
            
            @media (max-width: 480px) {{
                .metrics-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
            
            .metric {{
                background: rgba(255, 255, 255, 0.03);
                padding: 15px 10px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.05);
                transition: all 0.2s ease;
            }}
            
            .metric:hover {{
                background: rgba(255, 255, 255, 0.06);
                transform: scale(1.02);
            }}
            
            .metric-value {{
                font-size: clamp(1.2rem, 3vw, 1.5rem);
                font-weight: 700;
                color: {primary_color};
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #94a3b8;
                font-size: 0.75rem;
                font-weight: 500;
            }}
            
            /* REASONS LIST MEJORADA ORIGINAL */
            .reasons-list {{
                list-style: none;
                margin-top: 15px;
            }}
            
            .reason-item {{
                background: rgba(255, 255, 255, 0.03);
                margin: 8px 0;
                padding: 12px 15px;
                border-radius: 10px;
                border-left: 3px solid {primary_color};
                transition: all 0.2s ease;
            }}
            
            .reason-item:hover {{
                background: rgba(255, 255, 255, 0.06);
                transform: translateX(5px);
            }}
            
            /* PERFORMANCE SECTION ORIGINAL */
            .performance {{
                margin-top: 25px;
                padding-top: 20px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .validation-result {{
                background: rgba(255, 255, 255, 0.03);
                padding: 18px;
                border-radius: 12px;
                margin: 12px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.2s ease;
            }}
            
            .validation-result:hover {{
                background: rgba(255, 255, 255, 0.06);
            }}
            
            /* SCORE DISPLAY ORIGINAL */
            .score-display {{
                background: rgba(255, 255, 255, 0.03);
                padding: 15px;
                border-radius: 12px;
                margin: 15px 0;
            }}
            
            .score-bar-container {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                height: 8px;
                margin: 10px 0;
                overflow: hidden;
            }}
            
            .score-bar {{
                height: 100%;
                border-radius: 10px;
                transition: width 0.5s ease;
            }}
            
            .buy-bar {{
                background: linear-gradient(90deg, #00ff88, #00cc6a);
                width: {current_prediction.get('buy_score', 0)}%;
            }}
            
            .sell-bar {{
                background: linear-gradient(90deg, #ff4444, #cc3636);
                width: {current_prediction.get('sell_score', 0)}%;
            }}
            
            /* INFO GRID RESPONSIVE ORIGINAL */
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            
            .info-item {{
                background: rgba({primary_color.replace('#', '')}, 0.1);
                padding: 20px 15px;
                border-radius: 12px;
                border-left: 3px solid {primary_color};
                transition: all 0.2s ease;
            }}
            
            .info-item:hover {{
                transform: translateY(-2px);
                background: rgba({primary_color.replace('#', '')}, 0.15);
            }}
            
            /* ML INFO STYLES */
            .ml-info {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 12px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            
            .ml-badge {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
                padding: 4px 8px;
                border-radius: 8px;
                font-size: 0.8rem;
                font-weight: 600;
            }}
            
            /* NUEVO: PERFORMANCE INSIGHTS */
            .performance-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 15px 0;
            }}
            
            .performance-metric {{
                background: rgba(255, 255, 255, 0.03);
                padding: 12px;
                border-radius: 10px;
                text-align: center;
            }}
            
            .performance-value {{
                font-size: 1.2rem;
                font-weight: 700;
                color: {primary_color};
            }}
            
            .performance-label {{
                font-size: 0.7rem;
                color: #94a3b8;
                margin-top: 5px;
            }}
            
            /* RESPONSIVE ADJUSTMENTS ORIGINAL */
            @media (max-width: 480px) {{
                body {{
                    padding: 15px;
                }}
                
                .card {{
                    padding: 20px 15px;
                    border-radius: 15px;
                }}
                
                .header {{
                    padding: 20px 15px;
                    margin-bottom: 20px;
                }}
                
                .metrics-grid {{
                    gap: 8px;
                }}
                
                .metric {{
                    padding: 12px 8px;
                }}
                
                .reason-item {{
                    padding: 10px 12px;
                    font-size: 0.9rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- HEADER MEJORADO ORIGINAL -->
            <div class="header">
                <div class="logo">🤖 DELOWYSS AI PREMIUM V5.5</div>
                <div class="subtitle">Sistema HÍBRIDO MEJORADO: IA Avanzada + AutoLearning Sin Fugas + Análisis Intra-Vela</div>
                <div class="version">VERSION 5.5 HYBRID ENHANCED - SIN FUGAS DE DATOS</div>
            </div>
            
            <!-- DASHBOARD PRINCIPAL ORIGINAL -->
            <div class="dashboard">
                <!-- PREDICCIÓN PRINCIPAL ORIGINAL -->
                <div class="card prediction-card">
                    <h2>🎯 PREDICCIÓN ACTUAL HÍBRIDA MEJORADA</h2>
                    <div class="direction" id="direction">{direction} {status_emoji}</div>
                    <div class="confidence">
                        CONFIANZA: {confidence}%
                        <span class="confidence-badge">{confidence_level}</span>
                        <span class="quality-badge">Calidad: {quality_level}</span>
                    </div>
                    
                    <!-- INFORMACIÓN ML -->
                    <div class="ml-info">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <div>
                                <strong>🤖 AutoLearning Sin Fugas:</strong> {ml_predicted} 
                                <span class="ml-badge">{ml_confidence}% conf</span>
                            </div>
                            <div style="color: #e2e8f0; font-size: 0.9rem;">
                                Entrenamientos: {training_count} | Calidad: {(intra_candle_quality * 100):.1f}%
                            </div>
                        </div>
                    </div>
                    
                    <!-- BARRA DE PROGRESO DE VELA ORIGINAL -->
                    <div class="candle-progress">
                        <div class="progress-bar"></div>
                    </div>
                    <div class="progress-info">
                        <span>Progreso de vela: {progress_percentage:.1f}%</span>
                        <span>Fase: {market_phase}</span>
                    </div>
                    
                    <!-- COUNTDOWN ORIGINAL -->
                    <div class="countdown">
                        <div style="color: #94a3b8; margin-bottom: 10px; font-size: 0.9rem;">
                            SIGUIENTE PREDICCIÓN EN:
                        </div>
                        <div class="countdown-number" id="countdown">{int(seconds_remaining)}s</div>
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 5px;">
                            Análisis completo de {tick_count} ticks | Calidad intra-vela: {(intra_candle_quality * 100):.1f}%
                        </div>
                    </div>
                    
                    <!-- MÉTRICAS RÁPIDAS ORIGINALES -->
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" id="tick-count">{tick_count}</div>
                            <div class="metric-label">TICKS ANALIZADOS</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{current_price:.5f}</div>
                            <div class="metric-label">PRECIO ACTUAL</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="accuracy">{accuracy_10:.1f}%</div>
                            <div class="metric-label">PRECISIÓN (10)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{int(candle_progress * 100)}%</div>
                            <div class="metric-label">PROGRESO VELA</div>
                        </div>
                    </div>
                    
                    <!-- NUEVO: PERFORMANCE INSIGHTS -->
                    <div class="performance-grid">
                        <div class="performance-metric">
                            <div class="performance-value" id="accuracy-10">{accuracy_10:.1f}%</div>
                            <div class="performance-label">PREC. 10 VELAS</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-value" id="accuracy-20">{accuracy_20:.1f}%</div>
                            <div class="performance-label">PREC. 20 VELAS</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-value" id="conf-correct">{avg_confidence_correct:.1f}%</div>
                            <div class="performance-label">CONF. ACIERTOS</div>
                        </div>
                        <div class="performance-metric">
                            <div class="performance-value" id="conf-wrong">{avg_confidence_wrong:.1f}%</div>
                            <div class="performance-label">CONF. FALLOS</div>
                        </div>
                    </div>
                </div>
                
                <!-- ANÁLISIS DE IA ORIGINAL -->
                <div class="card">
                    <h3>🧠 ANÁLISIS DE IA AVANZADO MEJORADO</h3>
                    
                    <!-- BARRAS DE SCORE ORIGINALES -->
                    <div class="score-display">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #00ff88; font-weight: 600;">
                                COMPRA: <span id="buy-score">{current_prediction.get('buy_score', 0)}%</span>
                            </span>
                            <span style="color: #ff4444; font-weight: 600;">
                                VENTA: <span id="sell-score">{current_prediction.get('sell_score', 0)}%</span>
                            </span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar buy-bar"></div>
                        </div>
                        <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 5px;">
                            Diferencia: <span id="score-diff">{current_prediction.get('score_difference', 0)}</span>
                            | ML Boost: <span id="ml-boost">{current_prediction.get('ml_boost', 0)}</span>
                        </div>
                    </div>
                    
                    <h4 style="margin: 20px 0 10px 0; color: #e2e8f0;">📊 FACTORES DE DECISIÓN MEJORADOS:</h4>
                    <ul class="reasons-list" id="reasons-list">
                        {reasons_html}
                    </ul>
                </div>
                
                <!-- RENDIMIENTO Y VALIDACIÓN ORIGINAL -->
                <div class="card">
                    <h3>📈 RENDIMIENTO EN TIEMPO REAL</h3>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" style="color: #00ff88;">{accuracy_10:.1f}%</div>
                            <div class="metric-label">PRECISIÓN (10)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="total-pred">{performance_insights.get('total_predictions', 0)}</div>
                            <div class="metric-label">TOTAL PREDICCIONES</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #00ff88;">{performance_insights.get('correct_predictions', 0)}</div>
                            <div class="metric-label">CORRECTAS</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #ffbb33;">{training_count}</div>
                            <div class="metric-label">ENTRENAMIENTOS ML</div>
                        </div>
                    </div>
                    
                    <div class="performance">
                        <h4 style="margin-bottom: 15px;">✅ ÚLTIMA VALIDACIÓN</h4>
                        <div class="validation-result" id="validation-result">
                            <div style="color: #94a3b8; text-align: center; padding: 20px;">
                                <div class="loading">⏳ Esperando validación...</div>
                            </div>
                        </div>
                        
                        <!-- NUEVO: TENDENCIA RECIENTE -->
                        <div style="margin-top: 15px; padding: 12px; background: rgba(255, 255, 255, 0.03); border-radius: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #94a3b8;">Tendencia reciente:</span>
                                <span style="color: { '#00ff88' if recent_trend == 'MEJORANDO' else '#ff4444' if recent_trend == 'BAJANDO' else '#ffbb33' }; font-weight: 600;">
                                    {recent_trend}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- INFORMACIÓN DEL SISTEMA MEJORADA ORIGINAL -->
            <div class="card">
                <h3>⚙️ SISTEMA HÍBRIDO AVANZADO MEJORADO</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div style="font-weight: 600; color: #00ff88; font-size: 1.1rem;">🤖 IA AVANZADA</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                            Análisis tick-by-tick completo + métricas intra-vela
                        </div>
                    </div>
                    <div class="info-item">
                        <div style="font-weight: 600; color: #667eea; font-size: 1.1rem;">🧠 AUTOLEARNING SIN FUGAS</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                            ML mejorado sin uso de datos futuros
                        </div>
                    </div>
                    <div class="info-item">
                        <div style="font-weight: 600; color: #ff4444; font-size: 1.1rem;">🎯 PREDICCIÓN HÍBRIDA</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                            IA tradicional + Machine Learning integrado
                        </div>
                    </div>
                    <div class="info-item">
                        <div style="font-weight: 600; color: #00ff88; font-size: 1.1rem;">🚀 V5.5 MEJORADA</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                            Métricas intra-vela + Validación tiempo real
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Actualizar datos en tiempo real ORIGINAL MEJORADO
            function updateData() {{
                fetch('/api/prediction')
                    .then(response => response.json())
                    .then(data => {{
                        updatePrediction(data);
                    }})
                    .catch(error => {{
                        console.log('Error fetching prediction:', error);
                    }});
                    
                fetch('/api/validation')
                    .then(response => response.json())
                    .then(data => {{
                        updateValidation(data);
                    }})
                    .catch(error => {{
                        console.log('Error fetching validation:', error);
                    }});
            }}
            
            function updatePrediction(data) {{
                // Actualizar dirección
                const directionEl = document.getElementById('direction');
                if (directionEl) {{
                    let emoji = '⚡';
                    if (data.direction === 'ALZA') emoji = '📈';
                    if (data.direction === 'BAJA') emoji = '📉';
                    directionEl.textContent = data.direction + ' ' + emoji;
                }}
                
                // Actualizar colores según dirección
                let color = '#ffbb33';
                let gradient = 'linear-gradient(135deg, #ffbb33 0%, #cc9929 100%)';
                if (data.direction === 'ALZA') {{
                    color = '#00ff88';
                    gradient = 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)';
                }} else if (data.direction === 'BAJA') {{
                    color = '#ff4444';
                    gradient = 'linear-gradient(135deg, #ff4444 0%, #cc3636 100%)';
                }}
                
                // Actualizar confianza
                const confidence = data.confidence || 0;
                const confidenceEl = document.querySelector('.confidence');
                if (confidenceEl) {{
                    const confidenceLevel = confidence > 70 ? 'ALTA' : confidence > 50 ? 'MEDIA' : 'BAJA';
                    const confidenceColor = confidence > 70 ? '#00ff88' : confidence > 50 ? '#ffbb33' : '#ff4444';
                    
                    const quality = data.intra_candle_quality || 0;
                    const qualityLevel = quality > 0.8 ? 'EXCELENTE' : quality > 0.6 ? 'BUENA' : 'REGULAR';
                    const qualityColor = quality > 0.8 ? '#00ff88' : quality > 0.6 ? '#ffbb33' : '#ff4444';
                    
                    confidenceEl.innerHTML = `CONFIANZA: ${{confidence}}% <span class="confidence-badge" style="background: ${{confidenceColor}}">${{confidenceLevel}}</span> <span class="quality-badge" style="background: ${{qualityColor}}">Calidad: ${{qualityLevel}}</span>`;
                }}
                
                // Actualizar métricas
                updateMetric('tick-count', data.tick_count || 0);
                updateMetric('buy-score', data.buy_score || 0);
                updateMetric('sell-score', data.sell_score || 0);
                updateMetric('score-diff', (data.score_difference || 0).toFixed(2));
                updateMetric('ml-boost', (data.ml_boost || 0).toFixed(2));
                
                // Actualizar barras de score
                updateScoreBars(data.buy_score || 0, data.sell_score || 0);
                
                // Actualizar razones
                const reasons = data.reasons || ['Analizando mercado...'];
                const reasonsList = document.getElementById('reasons-list');
                if (reasonsList) {{
                    reasonsList.innerHTML = reasons.map(reason => 
                        `<li class="reason-item">${{reason}}</li>`
                    ).join('');
                }}
                
                // Actualizar métricas de performance
                const perf = data.performance_metrics || {{}};
                updateMetric('accuracy-10', perf.accuracy_10 || 0);
                updateMetric('accuracy-20', perf.accuracy_20 || 0);
                updateMetric('conf-correct', perf.avg_confidence_correct || 0);
                updateMetric('conf-wrong', perf.avg_confidence_wrong || 0);
                updateMetric('total-pred', perf.total_predictions || 0);
            }}
            
            function updateScoreBars(buyScore, sellScore) {{
                const buyBar = document.querySelector('.buy-bar');
                const sellBar = document.querySelector('.sell-bar');
                if (buyBar) buyBar.style.width = (buyScore || 0) + '%';
                if (sellBar) sellBar.style.width = (sellScore || 0) + '%';
            }}
            
            function updateMetric(id, value) {{
                const element = document.getElementById(id);
                if (element) {{
                    if (typeof value === 'number') {{
                        element.textContent = value.toFixed(value % 1 === 0 ? 0 : 2);
                    }} else {{
                        element.textContent = value;
                    }}
                }}
            }}
            
            function updateValidation(data) {{
                if (data.performance_insights) {{
                    updateMetric('accuracy', data.performance_insights.accuracy_10);
                }}
                
                if (data.last_validation) {{
                    const val = data.last_validation;
                    const color = val.correct ? '#00ff88' : '#ff4444';
                    const icon = val.correct ? '✅' : '❌';
                    const bgColor = val.correct ? 'rgba(0, 255, 136, 0.1)' : 'rgba(255, 68, 68, 0.1)';
                    
                    const validationEl = document.getElementById('validation-result');
                    if (validationEl) {{
                        validationEl.innerHTML = `
                            <div style="color: ${{color}}; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px;">
                                ${{icon}} ${{val.predicted}} → ${{val.actual}}
                            </div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">
                                Confianza: ${{val.confidence}}% | Cambio: ${{val.price_change}}pips
                            </div>
                            <div style="color: #64748b; font-size: 0.8rem; margin-top: 5px;">
                                Precisión actual: ${{val.accuracy}}% | Total: ${{val.total_predictions}}
                            </div>
                        `;
                        validationEl.style.borderLeftColor = color;
                        validationEl.style.background = bgColor;
                    }}
                }}
            }}
            
            // Actualizar countdown ORIGINAL
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                const countdownEl = document.getElementById('countdown');
                if (countdownEl) {{
                    countdownEl.textContent = remaining + 's';
                }}
            }}
            
            // Inicializar ORIGINAL
            setInterval(updateCountdown, 1000);
            setInterval(updateData, 2000);
            updateData();
            updateCountdown();
        </script>
    </body>
    </html>
    """
    return html_content

# --------------- INICIALIZACIÓN MEJORADA ---------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info(f"⭐ DELOWYSS AI V5.5 INICIADA EN PUERTO {PORT}")
        logging.info("🎯 SISTEMA HÍBRIDO MEJORADO: IA Avanzada + AutoLearning Sin Fugas + Interfaz Original")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

start_system()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True
    )
