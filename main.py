# main.py - V5.8 ANÁLISIS PROFUNDO TICK POR TICK
"""
Delowyss Trading AI — V5.8 ANÁLISIS PROFUNDO EN TIEMPO REAL
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
IQ_EMAIL = os.getenv("IQ_EMAIL", "vozhechacancion1@gmail.com")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "tu_password_real")
PAR = "EURUSD"
TIMEFRAME = 60
PREDICTION_WINDOW = 5  # Predecir cuando falten 5 segundos
MIN_TICKS_FOR_PREDICTION = 30  # Mínimo de ticks para análisis confiable
TICK_BUFFER_SIZE = 200
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

# ------------------ CONEXIÓN REAL IQ OPTION ------------------
try:
    from iqoptionapi.stable_api import IQ_Option
    IQ_OPTION_AVAILABLE = True
    logging.info("✅ iqoptionapi disponible")
except ImportError as e:
    logging.error(f"❌ iqoptionapi no disponible: {e}")
    IQ_OPTION_AVAILABLE = False
    class IQ_Option:
        def __init__(self, email, password): pass
        def connect(self): return False, "Biblioteca no disponible"
        def change_balance(self, balance_type): pass

class RealIQOptionConnector:
    def __init__(self, email, password, pair="EURUSD"):
        self.email = email
        self.password = password
        self.pair = pair
        self.api = None
        self.connected = False
        self.current_price = None
        self.connection_attempts = 0
        self.max_attempts = 5
        
    def connect(self):
        if not IQ_OPTION_AVAILABLE:
            logging.error("❌ iqoptionapi no disponible")
            return False
            
        try:
            logging.info(f"🔗 Conectando a IQ Option: {self.email}")
            self.api = IQ_Option(self.email, self.password)
            
            while self.connection_attempts < self.max_attempts:
                check, reason = self.api.connect()
                if check:
                    self.connected = True
                    logging.info("✅ Conexión exitosa a IQ Option")
                    
                    try:
                        self.api.change_balance("REAL")
                        logging.info("💰 Modo: Cuenta REAL")
                    except:
                        logging.info("💰 Modo: Cuenta PRACTICE")
                    
                    self.api.start_candles_stream(self.pair, TIMEFRAME, 1)
                    logging.info(f"📊 Stream iniciado para {self.pair}")
                    
                    time.sleep(2)
                    self._get_initial_price()
                    return True
                else:
                    self.connection_attempts += 1
                    logging.warning(f"⚠️ Intento {self.connection_attempts} fallado: {reason}")
                    time.sleep(3)
                    
            logging.error("❌ No se pudo conectar después de varios intentos")
            return False
            
        except Exception as e:
            logging.error(f"❌ Error en conexión: {e}")
            return False
    
    def _get_initial_price(self):
        try:
            candles = self.api.get_realtime_candles(self.pair, TIMEFRAME)
            if candles:
                for candle_id in candles:
                    candle = candles[candle_id]
                    if 'close' in candle:
                        self.current_price = candle['close']
                        logging.info(f"💰 Precio inicial {self.pair}: {self.current_price}")
                        break
            return self.current_price
        except Exception as e:
            logging.error(f"❌ Error obteniendo precio inicial: {e}")
            return None
    
    def get_realtime_price(self):
        if not self.connected or not self.api:
            logging.error("🔌 No conectado a IQ Option")
            return None
            
        try:
            candles = self.api.get_realtime_candles(self.pair, TIMEFRAME)
            if candles:
                for candle_id in candles:
                    candle = candles[candle_id]
                    if 'close' in candle:
                        new_price = candle['close']
                        if new_price and new_price > 0:
                            self.current_price = new_price
                            return self.current_price
            
            return self.current_price
            
        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            if self.connection_attempts < self.max_attempts:
                logging.info("🔄 Intentando reconectar...")
                self.connect()
            return self.current_price
    
    def get_remaining_time(self):
        if not self.connected:
            return TIMEFRAME - (int(time.time()) % TIMEFRAME)
            
        try:
            server_time = self.api.get_server_timestamp()
            if server_time:
                current_second = server_time % TIMEFRAME
                remaining = TIMEFRAME - current_second
                return max(0, min(TIMEFRAME, remaining))
            return TIMEFRAME - (int(time.time()) % TIMEFRAME)
        except Exception as e:
            logging.debug(f"🔧 Error obteniendo tiempo restante: {e}")
            return TIMEFRAME - (int(time.time()) % TIMEFRAME)

# ------------------ IA ULTRA EFICIENTE MEJORADA ------------------
class UltraEfficientAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=100)
        self.last_candle_close = None
        
        # Métricas mejoradas
        self.velocity_metrics = deque(maxlen=50)
        self.acceleration_metrics = deque(maxlen=40)
        self.volume_profile = deque(maxlen=20)
        self.price_levels = deque(maxlen=15)
        self.micro_trends = deque(maxlen=25)
        
        self.candle_start_time = None
        self.analysis_phases = {
            'initial': {'ticks': 0, 'analysis': {}, 'weight': 0.2},
            'middle': {'ticks': 0, 'analysis': {}, 'weight': 0.3},
            'final': {'ticks': 0, 'analysis': {}, 'weight': 0.5}
        }
        self.phase_accuracy = {'initial': 0.6, 'middle': 0.7, 'final': 0.9}
        
        # Análisis en tiempo real
        self.last_analysis_time = 0
        self.analysis_interval = 2  # Analizar cada 2 segundos
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Análisis profundo activado")
            
            # Actualizar precios de vela
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
            
            # ANÁLISIS PROFUNDO CADA TICK (optimizado)
            self._calculate_advanced_metrics(tick_data)
            
            # Análisis por fases cada 3 ticks para optimizar
            if self.tick_count % 3 == 0:
                self._analyze_market_phases(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"❌ Error en add_tick: {e}")
            return None
    
    def _calculate_advanced_metrics(self, current_tick):
        """Cálculo profundo de métricas por cada tick"""
        try:
            current_price = current_tick['price']
            current_time = current_tick['timestamp']

            if len(self.ticks) >= 2:
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
                    
                    # Calcular micro-tendencias
                    if len(self.velocity_metrics) >= 3:
                        recent_velocities = [v['velocity'] for v in list(self.velocity_metrics)[-3:]]
                        micro_trend = np.mean(recent_velocities)
                        self.micro_trends.append({
                            'trend': micro_trend,
                            'timestamp': current_time
                        })

            # Calcular aceleración
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
            
            # Actualizar niveles de precio dinámicos
            if len(self.price_memory) >= 10:
                prices = list(self.price_memory)
                recent_prices = prices[-10:]
                resistance = max(recent_prices)
                support = min(recent_prices)
                self.price_levels.append({
                    'resistance': resistance,
                    'support': support,
                    'timestamp': current_time
                })
                
        except Exception as e:
            logging.debug(f"🔧 Error en cálculo avanzado: {e}")
    
    def _analyze_market_phases(self, tick_data):
        """Análisis por fases de la vela"""
        candle_age = tick_data['candle_age']
        
        if candle_age < 20:
            self.analysis_phases['initial']['ticks'] += 1
            if self.analysis_phases['initial']['ticks'] % 5 == 0:
                self.analysis_phases['initial']['analysis'] = self._get_phase_analysis('initial')
                
        elif candle_age < 40:
            self.analysis_phases['middle']['ticks'] += 1
            if self.analysis_phases['middle']['ticks'] % 4 == 0:
                self.analysis_phases['middle']['analysis'] = self._get_phase_analysis('middle')
                
        else:
            self.analysis_phases['final']['ticks'] += 1
            if self.analysis_phases['final']['ticks'] % 2 == 0:  # Más frecuente al final
                self.analysis_phases['final']['analysis'] = self._get_phase_analysis('final')
    
    def _get_phase_analysis(self, phase):
        """Análisis detallado por fase"""
        try:
            ticks_list = list(self.ticks)
            if not ticks_list:
                return {}
                
            if phase == 'initial':
                ticks = ticks_list[:min(25, len(ticks_list))]
            elif phase == 'middle':
                if len(ticks_list) >= 50:
                    ticks = ticks_list[25:50]
                elif len(ticks_list) > 25:
                    ticks = ticks_list[25:]
                else:
                    ticks = []
            else:
                if len(ticks_list) >= 50:
                    ticks = ticks_list[50:]
                else:
                    ticks = []
            
            if not ticks:
                return {}
            
            prices = [tick['price'] for tick in ticks]
            
            # Cálculos más precisos
            volatility = (max(prices) - min(prices)) * 10000 if prices else 0
            avg_price = np.mean(prices) if prices else 0
            
            # Trend calculation mejorado
            if len(prices) >= 8:
                window_prices = prices[-8:]
                x_values = np.arange(len(window_prices))
                try:
                    recent_trend = np.polyfit(x_values, window_prices, 1)[0] * 10000
                except:
                    recent_trend = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            else:
                recent_trend = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            # Pressure calculation mejorado
            if len(prices) >= 3:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                positive_changes = len([x for x in price_changes if x > 0])
                total_changes = len(price_changes)
                buy_pressure = positive_changes / total_changes if total_changes > 0 else 0.5
            else:
                buy_pressure = 0.5
            
            # Momentum adicional
            momentum = 0
            if len(prices) >= 5:
                momentum = (prices[-1] - prices[-5]) * 10000
            
            return {
                'avg_price': avg_price,
                'volatility': volatility,
                'trend': 'ALCISTA' if recent_trend > 0.15 else 'BAJISTA' if recent_trend < -0.15 else 'LATERAL',
                'trend_strength': abs(recent_trend),
                'buy_pressure': buy_pressure,
                'momentum': momentum,
                'tick_count': len(ticks),
                'phase_accuracy': self.phase_accuracy[phase]
            }
        except Exception as e:
            logging.debug(f"🔧 Error en análisis de fase {phase}: {e}")
            return {}
    
    def get_deep_analysis(self):
        """Análisis profundo en tiempo real"""
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'INSUFFICIENT_DATA', 
                'tick_count': self.tick_count,
                'message': f'Analizando ticks: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION}',
                'confidence': self._calculate_confidence_score()
            }
        
        try:
            # Métricas avanzadas
            advanced_metrics = self._calculate_advanced_analysis()
            if not advanced_metrics:
                return {'status': 'ERROR', 'message': 'Error en análisis avanzado'}
            
            # Combinar análisis de fases
            phase_analysis = self._combine_phase_analysis()
            
            # Calcular confianza general
            overall_confidence = self._calculate_overall_confidence(advanced_metrics, phase_analysis)
            
            result = {
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
                **advanced_metrics,
                'phase_analysis': phase_analysis
            }
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error en análisis profundo: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _calculate_advanced_analysis(self):
        """Cálculo de métricas avanzadas"""
        if len(self.price_memory) < 10:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            # Múltiples ventanas de tendencia
            trend_windows = [5, 8, 12, 15]
            trend_metrics = []
            
            for window in trend_windows:
                if len(prices) >= window:
                    window_prices = prices[-window:]
                    x_values = np.arange(len(window_prices))
                    try:
                        trend = np.polyfit(x_values, window_prices, 1)[0] * 10000
                        trend_metrics.append(trend)
                    except:
                        continue
            
            trend_strength = np.mean(trend_metrics) if trend_metrics else 0
            
            # Momentum en múltiples timeframes
            momentums = []
            for period in [3, 5, 8]:
                if len(prices) >= period:
                    momentum = (prices[-1] - prices[-period]) * 10000
                    momentums.append(momentum)
            
            momentum = np.mean(momentums) if momentums else 0
            
            # Volatilidad dinámica
            if len(prices) >= 15:
                recent_prices = prices[-15:]
                volatility = (np.max(recent_prices) - np.min(recent_prices)) * 10000
            else:
                volatility = (np.max(prices) - np.min(prices)) * 10000
            
            # Presión de compra/venta mejorada
            if len(self.ticks) >= 10:
                recent_ticks = list(self.ticks)[-10:]
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
            
            # Velocidad y aceleración
            avg_velocity = 0
            if self.velocity_metrics:
                velocities = [v['velocity'] for v in list(self.velocity_metrics)[-15:]]
                avg_velocity = np.mean(velocities) * 10000 if velocities else 0
            
            avg_acceleration = 0
            if self.acceleration_metrics:
                accelerations = [a['acceleration'] for a in list(self.acceleration_metrics)[-10:]]
                avg_acceleration = np.mean(accelerations) * 10000 if accelerations else 0
            
            # Micro-tendencias
            micro_trend_strength = 0
            if self.micro_trends:
                recent_micro_trends = [t['trend'] for t in list(self.micro_trends)[-8:]]
                micro_trend_strength = np.mean(recent_micro_trends) * 10000 if recent_micro_trends else 0
            
            return {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volatility': volatility,
                'buy_pressure': buy_pressure,
                'sell_pressure': 1 - buy_pressure,
                'pressure_ratio': buy_pressure / (1 - buy_pressure) if buy_pressure < 1 else 10.0,
                'velocity': avg_velocity,
                'acceleration': avg_acceleration,
                'micro_trend': micro_trend_strength,
                'market_phase': self._determine_market_phase(trend_strength, volatility, momentum),
                'data_quality': min(1.0, self.tick_count / 40.0)
            }
            
        except Exception as e:
            logging.error(f"❌ Error en análisis avanzado: {e}")
            return {}
    
    def _combine_phase_analysis(self):
        """Combinar análisis de todas las fases"""
        try:
            initial = self.analysis_phases['initial']['analysis']
            middle = self.analysis_phases['middle']['analysis']
            final = self.analysis_phases['final']['analysis']
            
            weights = {
                'initial': self.analysis_phases['initial']['weight'],
                'middle': self.analysis_phases['middle']['weight'], 
                'final': self.analysis_phases['final']['weight']
            }
            
            # Combinar tendencias
            trends = []
            trend_strengths = []
            
            for phase, data in [('initial', initial), ('middle', middle), ('final', final)]:
                if data and data.get('trend'):
                    trends.append((data['trend'], weights[phase]))
                    trend_strengths.append(data.get('trend_strength', 0) * weights[phase])
            
            if trends:
                alcista_weight = sum(weight for trend, weight in trends if trend == 'ALCISTA')
                bajista_weight = sum(weight for trend, weight in trends if trend == 'BAJISTA')
                
                if alcista_weight > bajista_weight:
                    combined_trend = 'ALCISTA'
                elif bajista_weight > alcista_weight:
                    combined_trend = 'BAJISTA'
                else:
                    combined_trend = 'LATERAL'
                    
                combined_trend_strength = np.mean(trend_strengths) if trend_strengths else 0
            else:
                combined_trend = 'N/A'
                combined_trend_strength = 0
            
            return {
                'trend': combined_trend,
                'trend_strength': combined_trend_strength,
                'momentum_shift': len(set(trend for trend, _ in trends)) > 1 if trends else False,
                'consistency_score': alcista_weight if combined_trend == 'ALCISTA' else bajista_weight if combined_trend == 'BAJISTA' else 0.5,
            }
        except Exception as e:
            logging.debug(f"🔧 Error combinando análisis de fases: {e}")
            return {}
    
    def _determine_market_phase(self, trend_strength, volatility, momentum):
        """Determinar fase del mercado"""
        if volatility < 0.3 and abs(trend_strength) < 0.4:
            return "CONSOLIDACIÓN"
        elif abs(trend_strength) > 2.0:
            return "TENDENCIA_FUERTE"
        elif abs(trend_strength) > 1.0:
            return "TENDENCIA"
        elif volatility > 2.5:
            return "ALTA_VOLATILIDAD"
        elif abs(momentum) > 1.5:
            return "MOMENTUM"
        else:
            return "NORMAL"
    
    def _calculate_confidence_score(self):
        """Calcular score de confianza"""
        score = min(40, (self.tick_count / 30) * 40)
        
        if len(self.velocity_metrics) >= 15:
            score += 20
        
        if len(self.acceleration_metrics) >= 10:
            score += 15
        
        score += self.analysis_phases['final']['weight'] * 25
        
        if len(self.price_memory) >= 20:
            prices = list(self.price_memory)[-20:]
            volatility = (max(prices) - min(prices)) * 10000
            if 0.5 < volatility < 3.0:  # Volatilidad óptima
                score += 20
        
        return min(100, score)
    
    def _calculate_overall_confidence(self, advanced_metrics, phase_analysis):
        """Calcular confianza general"""
        base_confidence = self._calculate_confidence_score()
        
        # Ajustar por calidad de datos
        data_quality = advanced_metrics.get('data_quality', 0)
        adjusted_confidence = base_confidence * data_quality
        
        # Bonus por consistencia en fases
        consistency = phase_analysis.get('consistency_score', 0.5)
        consistency_bonus = consistency * 20
        
        return min(95, adjusted_confidence + consistency_bonus)
    
    def reset(self):
        """Reiniciar análisis para nueva vela"""
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
            self.micro_trends.clear()
            self.candle_start_time = None
            
            for phase in self.analysis_phases:
                self.analysis_phases[phase] = {'ticks': 0, 'analysis': {}, 'weight': self.analysis_phases[phase]['weight']}
                
        except Exception as e:
            logging.error(f"❌ Error en reset: {e}")

# ------------------ RESTANTE DEL CÓDIGO (AdaptiveMarketLearner, ComprehensiveAIPredictor, etc.) ------------------
# [Mantener las mismas clases AdaptiveMarketLearner, ComprehensiveAIPredictor, RealTimeValidator]
# [Mantener las mismas funciones build_advanced_features_from_analysis, etc.]

# ------------------ SISTEMA PRINCIPAL MEJORADO ------------------
iq_connector = RealIQOptionConnector(IQ_EMAIL, IQ_PASSWORD, PAR)
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
_last_analysis_time = 0

def tick_processor(price, timestamp, seconds_remaining):
    global current_prediction, _last_analysis_time
    try:
        current_time = time.time()
        
        # Procesar tick
        tick_data = predictor.process_tick(price, seconds_remaining)
        
        # Análisis profundo cada 2 segundos
        if current_time - _last_analysis_time >= 2:
            analysis = predictor.analyzer.get_deep_analysis()
            if analysis.get('status') == 'SUCCESS':
                # Actualizar información de análisis en tiempo real
                current_prediction.update({
                    "current_price": float(price),
                    "tick_count": predictor.analyzer.tick_count,
                    "analysis_quality": analysis.get('data_quality', 0),
                    "market_phase": analysis.get('market_phase', 'N/A'),
                    "timestamp": now_iso(),
                    "status": "ANALYZING"
                })
                _last_analysis_time = current_time
            
        if tick_data:
            current_prediction.update({
                "current_price": float(price),
                "tick_count": predictor.analyzer.tick_count,
                "timestamp": now_iso(),
                "status": "ACTIVE"
            })
            
    except Exception as e:
        logging.error(f"❌ Error procesando tick: {e}")

def premium_main_loop_deep_analysis():
    """🚀 LOOP PRINCIPAL CON ANÁLISIS PROFUNDO"""
    global current_prediction, _last_candle_start, _prediction_made_this_candle
    global _last_prediction_time, _last_price
    
    logging.info(f"🚀 DELOWYSS AI V5.8 - ANÁLISIS PROFUNDO ACTIVADO")
    logging.info("🎯 PREDICCIÓN A 5 SEGUNDOS - ANÁLISIS TICK POR TICK")
    
    # Conectar a IQ Option
    if not iq_connector.connect():
        logging.error("❌ No se pudo conectar a IQ Option")
        return
    
    logging.info(f"✅ CONECTADO A IQ OPTION | Predicción a {PREDICTION_WINDOW}s")
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = iq_connector.get_remaining_time()
            
            # Obtener precio REAL
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price
                tick_processor(price, current_time, seconds_remaining)

            # ✅ PREDICCIÓN EXACTA A 5 SEGUNDOS
            prediction_time = (seconds_remaining <= PREDICTION_WINDOW and 
                             seconds_remaining > 0.5)
            
            if (prediction_time and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                not _prediction_made_this_candle):

                logging.info(f"🎯 PREDICCIÓN A {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                # ANÁLISIS PROFUNDO CON DATOS COMPLETOS
                analysis = predictor.analyzer.get_deep_analysis()
                if analysis.get('status') == 'SUCCESS':
                    features = build_advanced_features_from_analysis(analysis, seconds_remaining)
                    ml_prediction = online_learner.predict(features)
                    
                    # PREDECIR SIGUIENTE VELA
                    final_prediction = predictor.predict_next_candle(ml_prediction)
                    
                    current_prediction.update(final_prediction)
                    current_prediction.update({
                        "ai_model_predicted": ml_prediction['predicted'],
                        "ml_confidence": ml_prediction['confidence'],
                        "training_count": ml_prediction['training_count'],
                        "prediction_time": seconds_remaining
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
                        
                        # Auto-learning mejorado
                        analysis = predictor.analyzer.get_deep_analysis()
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
                logging.info("🕯️ NUEVA VELA - Análisis profundo reiniciado")

            time.sleep(0.05)  # Loop más rápido para mejor respuesta
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(1)

# ------------------ FASTAPI APP (mantener igual) ------------------
app = FastAPI(
    title="Delowyss Trading AI V5.8 - Análisis Profundo",
    description="Sistema de IA con análisis profundo tick por tick",
    version="5.8.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ ENDPOINTS (mantener igual) ------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_INTERFACE

@app.get("/api/prediction")
async def get_prediction():
    return current_prediction

@app.get("/api/performance")
async def get_performance():
    stats = predictor.get_performance_stats()
    return {
        "performance": stats,
        "ml_training": online_learner.last_training_result,
        "system_status": "DEEP_ANALYSIS_ACTIVE",
        "timestamp": now_iso()
    }

@app.get("/api/analysis")
async def get_analysis():
    analysis = predictor.analyzer.get_deep_analysis()
    return {
        "analysis": analysis,
        "timestamp": now_iso()
    }

@app.get("/api/status")
async def get_status():
    return {
        "status": "operational",
        "version": "5.8.0",
        "pair": PAR,
        "timeframe": "1min",
        "iq_connected": iq_connector.connected,
        "current_price": iq_connector.current_price,
        "prediction_window": f"{PREDICTION_WINDOW}s",
        "timestamp": now_iso()
    }

# ------------------ INICIALIZACIÓN ------------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop_deep_analysis, daemon=True)
        thread.start()
        logging.info(f"⭐ DELOWYSS AI V5.8 INICIADA - ANÁLISIS PROFUNDO")
        logging.info("🎯 PREDICCIÓN A 5s - ANÁLISIS TICK POR TICK")
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
