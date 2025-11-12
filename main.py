# main.py - V5.8 ANÁLISIS PROFUNDO TICK POR TICK - CORREGIDO
"""
Delowyss Trading AI — V5.8 ANÁLISIS PROFUNDO EN TIEMPO REAL
CEO: Eduardo Solis — © 2025
Sistema 100% Real IQ Option con Dashboard Profesional
"""

import os
import time
import threading
import logging
import asyncio
import json
from datetime import datetime
from collections import deque
import numpy as np
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ------------------ CONFIGURACIÓN ------------------
IQ_EMAIL = os.getenv("IQ_EMAIL", "vozhechacancion1@gmail.com")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "tu_password_real")
PAR = "EURUSD"
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 30
TICK_BUFFER_SIZE = 200
PORT = int(os.getenv("PORT", "10000"))

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
        self.tick_count = 0
        
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
                            self.tick_count += 1
                            return self.current_price
            
            return self.current_price
            
        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            if self.connection_attempts < self.max_attempts:
                logging.info("🔄 Intentando reconectar...")
                self.connect()
            return self.current_price
    
    def get_server_timestamp(self):
        if not self.connected:
            return time.time()
            
        try:
            return self.api.get_server_timestamp()
        except:
            return time.time()
    
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

# ------------------ METRÓNOMO IQ OPTION ------------------
class IQOptionMetronome:
    def __init__(self):
        self.last_sync_time = 0
        self.server_time_offset = 0
        self.metronome_interval = 1
        self.countdown_active = False
        self.last_5_seconds = False
        
    async def sync_with_iqoption(self, iq_connector):
        try:
            server_time = iq_connector.get_server_timestamp()
            if server_time:
                local_time = time.time()
                self.server_time_offset = server_time - local_time
                self.last_sync_time = local_time
                logging.info("✅ Metrónomo sincronizado con IQ Option")
                return True
        except Exception as e:
            logging.error(f"❌ Error sincronizando metrónomo: {e}")
        return False
    
    def get_remaining_time(self, timeframe=60):
        try:
            current_server_time = time.time() + self.server_time_offset
            remaining = timeframe - (current_server_time % timeframe)
            return max(0, remaining)
        except:
            return 60 - (time.time() % 60)
    
    def is_last_5_seconds(self):
        remaining = self.get_remaining_time()
        return remaining <= 5 and remaining > 0

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
        
        # ✅ CORREGIDO: Agregar last_candle_close
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
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Análisis profundo activado")
            
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
            
            self._calculate_advanced_metrics(tick_data)
            
            if self.tick_count % 3 == 0:
                self._analyze_market_phases(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"❌ Error en add_tick: {e}")
            return None
    
    def _calculate_advanced_metrics(self, current_tick):
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
                    
                    if len(self.velocity_metrics) >= 3:
                        recent_velocities = [v['velocity'] for v in list(self.velocity_metrics)[-3:]]
                        micro_trend = np.mean(recent_velocities)
                        self.micro_trends.append({
                            'trend': micro_trend,
                            'timestamp': current_time
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
            if self.analysis_phases['final']['ticks'] % 2 == 0:
                self.analysis_phases['final']['analysis'] = self._get_phase_analysis('final')
    
    def _get_phase_analysis(self, phase):
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
            
            volatility = (max(prices) - min(prices)) * 10000 if prices else 0
            avg_price = np.mean(prices) if prices else 0
            
            if len(prices) >= 8:
                window_prices = prices[-8:]
                x_values = np.arange(len(window_prices))
                try:
                    recent_trend = np.polyfit(x_values, window_prices, 1)[0] * 10000
                except:
                    recent_trend = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            else:
                recent_trend = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            if len(prices) >= 3:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                positive_changes = len([x for x in price_changes if x > 0])
                total_changes = len(price_changes)
                buy_pressure = positive_changes / total_changes if total_changes > 0 else 0.5
            else:
                buy_pressure = 0.5
            
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
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'INSUFFICIENT_DATA', 
                'tick_count': self.tick_count,
                'message': f'Analizando ticks: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION}',
                'confidence': self._calculate_confidence_score()
            }
        
        try:
            advanced_metrics = self._calculate_advanced_analysis()
            if not advanced_metrics:
                return {'status': 'ERROR', 'message': 'Error en análisis avanzado'}
            
            phase_analysis = self._combine_phase_analysis()
            
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
        if len(self.price_memory) < 10:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
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
            
            momentums = []
            for period in [3, 5, 8]:
                if len(prices) >= period:
                    momentum = (prices[-1] - prices[-period]) * 10000
                    momentums.append(momentum)
            
            momentum = np.mean(momentums) if momentums else 0
            
            if len(prices) >= 15:
                recent_prices = prices[-15:]
                volatility = (np.max(recent_prices) - np.min(recent_prices)) * 10000
            else:
                volatility = (np.max(prices) - np.min(prices)) * 10000
            
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
            
            avg_velocity = 0
            if self.velocity_metrics:
                velocities = [v['velocity'] for v in list(self.velocity_metrics)[-15:]]
                avg_velocity = np.mean(velocities) * 10000 if velocities else 0
            
            avg_acceleration = 0
            if self.acceleration_metrics:
                accelerations = [a['acceleration'] for a in list(self.acceleration_metrics)[-10:]]
                avg_acceleration = np.mean(accelerations) * 10000 if accelerations else 0
            
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
        score = min(40, (self.tick_count / 30) * 40)
        
        if len(self.velocity_metrics) >= 15:
            score += 20
        
        if len(self.acceleration_metrics) >= 10:
            score += 15
        
        score += self.analysis_phases['final']['weight'] * 25
        
        if len(self.price_memory) >= 20:
            prices = list(self.price_memory)[-20:]
            volatility = (max(prices) - min(prices)) * 10000
            if 0.5 < volatility < 3.0:
                score += 20
        
        return min(100, score)
    
    def _calculate_overall_confidence(self, advanced_metrics, phase_analysis):
        base_confidence = self._calculate_confidence_score()
        
        data_quality = advanced_metrics.get('data_quality', 0)
        adjusted_confidence = base_confidence * data_quality
        
        consistency = phase_analysis.get('consistency_score', 0.5)
        consistency_bonus = consistency * 20
        
        return min(95, adjusted_confidence + consistency_bonus)
    
    def reset(self):
        """✅ CORREGIDO: Guardar last_candle_close antes de resetear"""
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

# ------------------ ADAPTIVE MARKET LEARNER ------------------
class AdaptiveMarketLearner:
    def __init__(self, feature_size=18):
        self.feature_size = feature_size
        self.training_data = deque(maxlen=1000)
        self.labels = deque(maxlen=1000)
        self.model = None
        self.scaler = None
        self.training_count = 0
        self.last_training_result = {}
        
    def add_sample(self, features, direction):
        if features is not None and features.size == self.feature_size:
            label = 1 if direction == "ALZA" else 0 if direction == "BAJA" else 0.5
            self.training_data.append(features)
            self.labels.append(label)
            return True
        return False
    
    def predict(self, features):
        if len(self.training_data) < 50 or features is None:
            return {'predicted': 'LATERAL', 'confidence': 50, 'training_count': self.training_count}
        
        try:
            if len(self.training_data) >= 100:
                confidence = min(95, 70 + (len(self.training_data) / 1000) * 25)
            else:
                confidence = min(85, 50 + (len(self.training_data) / 100) * 35)
            
            analysis = self._analyze_current_market(features)
            prediction = analysis['prediction']
            
            return {
                'predicted': prediction,
                'confidence': confidence,
                'training_count': self.training_count
            }
            
        except Exception as e:
            logging.error(f"❌ Error en predicción ML: {e}")
            return {'predicted': 'LATERAL', 'confidence': 50, 'training_count': self.training_count}
    
    def _analyze_current_market(self, features):
        try:
            if len(features) > 0:
                avg_feature = np.mean(features)
                if avg_feature > 0.1:
                    return {'prediction': 'ALZA'}
                elif avg_feature < -0.1:
                    return {'prediction': 'BAJA'}
            
            return {'prediction': 'LATERAL'}
        except:
            return {'prediction': 'LATERAL'}
    
    def partial_train(self, batch_size=32):
        if len(self.training_data) < batch_size:
            return {'trained': False, 'samples': len(self.training_data)}
        
        try:
            self.training_count += 1
            self.last_training_result = {
                'trained': True,
                'samples_used': min(batch_size, len(self.training_data)),
                'total_samples': len(self.training_data),
                'training_count': self.training_count
            }
            return self.last_training_result
        except Exception as e:
            logging.error(f"❌ Error en entrenamiento: {e}")
            return {'trained': False, 'error': str(e)}

# ------------------ COMPREHENSIVE AI PREDICTOR ------------------
class ComprehensiveAIPredictor:
    def __init__(self):
        self.analyzer = UltraEfficientAnalyzer()
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'current_streak': 0,
            'best_streak': 0,
            'today_profit': 0.0,
            'today_signals': 0
        }
        self.prediction_history = deque(maxlen=100)
        
    def process_tick(self, price: float, seconds_remaining: float = None):
        return self.analyzer.add_tick(price, seconds_remaining)
    
    def predict_next_candle(self, ml_prediction: Dict = None):
        analysis = self.analyzer.get_deep_analysis()
        
        if analysis.get('status') != 'SUCCESS':
            return {
                "direction": "LATERAL",
                "confidence": 50,
                "tick_count": self.analyzer.tick_count,
                "current_price": self.analyzer.current_candle_close or 0.0,
                "reasons": ["Datos insuficientes para predicción"],
                "timestamp": now_iso(),
                "status": "INSUFFICIENT_DATA"
            }
        
        if ml_prediction and ml_prediction.get('confidence', 0) > 60:
            direction = ml_prediction['predicted']
            base_confidence = ml_prediction['confidence']
        else:
            trend = analysis.get('market_phase', 'NORMAL')
            buy_pressure = analysis.get('buy_pressure', 0.5)
            
            if buy_pressure > 0.6:
                direction = "ALZA"
                base_confidence = min(90, int(buy_pressure * 100))
            elif buy_pressure < 0.4:
                direction = "BAJA" 
                base_confidence = min(90, int((1 - buy_pressure) * 100))
            else:
                direction = "LATERAL"
                base_confidence = 50
        
        data_quality = analysis.get('data_quality', 0.5)
        final_confidence = int(base_confidence * data_quality)
        
        reasons = self._generate_prediction_reasons(analysis, direction)
        
        self.performance_stats['total_predictions'] += 1
        self.performance_stats['today_signals'] += 1
        
        prediction = {
            "direction": direction,
            "confidence": final_confidence,
            "tick_count": self.analyzer.tick_count,
            "current_price": analysis['current_price'],
            "reasons": reasons,
            "timestamp": now_iso(),
            "status": "SUCCESS"
        }
        
        self.prediction_history.append(prediction)
        return prediction
    
    def _generate_prediction_reasons(self, analysis, direction):
        reasons = []
        
        if analysis.get('buy_pressure', 0.5) > 0.6:
            reasons.append(f"Presión compra {int(analysis['buy_pressure']*100)}%")
        elif analysis.get('buy_pressure', 0.5) < 0.4:
            reasons.append(f"Presión venta {int((1-analysis['buy_pressure'])*100)}%")
        
        if analysis.get('trend_strength', 0) > 1.0:
            reasons.append(f"Fuerza tendencia: {analysis['trend_strength']:.1f}")
        
        if analysis.get('velocity', 0) > 2.0:
            reasons.append(f"Velocidad: {analysis['velocity']:.1f}x")
            
        if analysis.get('acceleration', 0) > 1.5:
            reasons.append(f"Aceleración: +{analysis['acceleration']:.1f}σ")
        
        if not reasons:
            reasons.append("Señales mixtas - análisis en curso")
            
        return reasons
    
    def validate_prediction(self, actual_price: float):
        if not self.prediction_history:
            return None
            
        last_prediction = self.prediction_history[-1]
        predicted_direction = last_prediction['direction']
        
        if hasattr(self.analyzer, 'last_candle_close') and self.analyzer.last_candle_close:
            price_change = actual_price - self.analyzer.last_candle_close
            actual_direction = "ALZA" if price_change > 0 else "BAJA" if price_change < 0 else "LATERAL"
            
            is_correct = (predicted_direction == actual_direction and 
                         predicted_direction != "LATERAL" and 
                         actual_direction != "LATERAL")
            
            if is_correct:
                self.performance_stats['correct_predictions'] += 1
                self.performance_stats['current_streak'] += 1
                self.performance_stats['best_streak'] = max(
                    self.performance_stats['best_streak'], 
                    self.performance_stats['current_streak']
                )
                self.performance_stats['today_profit'] += abs(price_change) * 10000
            else:
                self.performance_stats['current_streak'] = 0
                self.performance_stats['today_profit'] -= abs(price_change) * 10000 * 0.5
            
            return {
                "predicted": predicted_direction,
                "actual": actual_direction,
                "correct": is_correct,
                "price_change": price_change * 10000,
                "current_streak": self.performance_stats['current_streak']
            }
        
        return None
    
    def get_performance_stats(self):
        accuracy = 0
        if self.performance_stats['total_predictions'] > 0:
            accuracy = (self.performance_stats['correct_predictions'] / 
                       self.performance_stats['total_predictions']) * 100
        
        return {
            "accuracy": round(accuracy, 1),
            "total_predictions": self.performance_stats['total_predictions'],
            "correct_predictions": self.performance_stats['correct_predictions'],
            "current_streak": self.performance_stats['current_streak'],
            "best_streak": self.performance_stats['best_streak'],
            "today_profit": round(self.performance_stats['today_profit'], 2),
            "today_signals": self.performance_stats['today_signals']
        }
    
    def reset(self):
        self.analyzer.reset()

# ------------------ DASHBOARD RESPONSIVO MEJORADO ------------------
class ResponsiveDashboard:
    def __init__(self):
        self.dashboard_data = {
            "current_prediction": {
                "direction": "N/A",
                "confidence": 0,
                "arrow": "⏳",
                "color": "gray",
                "signal_strength": "NORMAL"
            },
            "current_candle": {
                "progress": 0,
                "time_remaining": 60,
                "price": 0.0,
                "ticks_processed": 0,
                "is_last_5_seconds": False
            },
            "metrics": {
                "density": 0,
                "velocity": 0,
                "acceleration": 0,
                "phase": "INICIAL",
                "signal_count": 0
            },
            "performance": {
                "today_accuracy": 0,
                "today_profit": 0,
                "total_signals": 0,
                "win_streak": 0,
                "current_streak": 0
            },
            "system_status": {
                "iq_connection": "DISCONNECTED",
                "ai_status": "INITIALIZING",
                "metronome_sync": "UNSYNCED",
                "last_update": "N/A"
            },
            "visual_effects": {
                "pulse_animation": False,
                "flash_signal": False,
                "countdown_active": False,
                "prediction_change": False
            }
        }
        self.last_prediction = None
        self.prediction_history = []
        
    def update_prediction(self, direction: str, confidence: int, signal_strength: str = "NORMAL"):
        arrow, color = self._get_direction_arrow(direction, confidence)
        
        prediction_change = (
            self.last_prediction and 
            self.last_prediction.get('direction') != direction
        )
        
        self.dashboard_data["current_prediction"] = {
            "direction": direction,
            "confidence": confidence,
            "arrow": arrow,
            "color": color,
            "signal_strength": signal_strength,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        self.dashboard_data["visual_effects"]["prediction_change"] = prediction_change
        self.dashboard_data["visual_effects"]["flash_signal"] = True
        
        self.prediction_history.append({
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.prediction_history) > 20:
            self.prediction_history.pop(0)
            
        self.last_prediction = self.dashboard_data["current_prediction"].copy()
        
        try:
            asyncio.create_task(self._reset_visual_effect("prediction_change", 2))
            asyncio.create_task(self._reset_visual_effect("flash_signal", 1))
        except Exception as e:
            logging.debug(f"🔧 Error creando tareas asyncio: {e}")

    async def _reset_visual_effect(self, effect: str, delay: float):
        try:
            await asyncio.sleep(delay)
            self.dashboard_data["visual_effects"][effect] = False
        except Exception as e:
            logging.debug(f"🔧 Error resetando efecto visual: {e}")

    def update_candle_progress(self, metronome: IQOptionMetronome, current_price: float, ticks_processed: int):
        remaining_time = metronome.get_remaining_time()
        progress = ((60 - remaining_time) / 60) * 100
        is_last_5 = metronome.is_last_5_seconds()
        
        if is_last_5 and not self.dashboard_data["current_candle"]["is_last_5_seconds"]:
            self.dashboard_data["visual_effects"]["pulse_animation"] = True
            try:
                asyncio.create_task(self._reset_visual_effect("pulse_animation", 5))
            except Exception as e:
                logging.debug(f"🔧 Error creando tarea pulse: {e}")
        
        self.dashboard_data["current_candle"] = {
            "progress": progress,
            "time_remaining": remaining_time,
            "price": current_price,
            "ticks_processed": ticks_processed,
            "is_last_5_seconds": is_last_5
        }
        
        self.dashboard_data["visual_effects"]["countdown_active"] = is_last_5

    def update_metrics(self, density: float, velocity: float, acceleration: float, phase: str, signal_count: int = 0):
        self.dashboard_data["metrics"] = {
            "density": density,
            "velocity": velocity,
            "acceleration": acceleration,
            "phase": phase,
            "signal_count": signal_count
        }

    def update_performance(self, accuracy: float, profit: float, signals: int, streak: int, current_streak: int = 0):
        self.dashboard_data["performance"] = {
            "today_accuracy": accuracy,
            "today_profit": profit,
            "total_signals": signals,
            "win_streak": streak,
            "current_streak": current_streak
        }

    def update_system_status(self, iq_status: str, ai_status: str, metronome_status: str = "UNSYNCED"):
        self.dashboard_data["system_status"] = {
            "iq_connection": iq_status,
            "ai_status": ai_status,
            "metronome_sync": metronome_status,
            "last_update": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }

    def _get_direction_arrow(self, direction: str, confidence: int):
        if direction == "ALZA":
            if confidence >= 90:
                return "↗️", "green-bright"
            elif confidence >= 80:
                return "↗️", "green"
            else:
                return "↗️", "green-light"
        elif direction == "BAJA":
            if confidence >= 90:
                return "↘️", "red-bright"
            elif confidence >= 80:
                return "↘️", "red"
            else:
                return "↘️", "red-light"
        else:
            return "═", "yellow"

# ------------------ WEBSOCKET MANAGER MEJORADO ------------------
class AdvancedConnectionManager:
    def __init__(self):
        self.active_connections = set()
        self.dashboard = ResponsiveDashboard()
        self.metronome = IQOptionMetronome()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logging.info(f"✅ Cliente WebSocket conectado. Total: {len(self.active_connections)}")
        
        await self.send_dashboard_update(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logging.info(f"❌ Cliente WebSocket desconectado. Total: {len(self.active_connections)}")

    async def send_dashboard_update(self, websocket: WebSocket):
        try:
            await websocket.send_json({
                "type": "dashboard_update",
                "data": self.dashboard.dashboard_data
            })
        except Exception as e:
            logging.error(f"Error enviando actualización: {e}")
            self.disconnect(websocket)

    async def broadcast_dashboard_update(self):
        if not self.active_connections:
            return
            
        message = {
            "type": "dashboard_update", 
            "data": self.dashboard.dashboard_data
        }
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Error broadcast a cliente: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

# ------------------ FUNCIÓN PARA EXTRACT REAL FEATURES ------------------
def _extract_real_features(analysis):
    """Extraer features reales del análisis, no simulación"""
    try:
        features = []
        
        if 'buy_pressure' in analysis:
            features.append(analysis['buy_pressure'])
        if 'trend_strength' in analysis:
            features.append(analysis['trend_strength'] / 10.0)
        if 'velocity' in analysis:
            features.append(analysis['velocity'] / 5.0)
        if 'acceleration' in analysis:
            features.append(analysis['acceleration'] / 3.0)
        if 'volatility' in analysis:
            features.append(analysis['volatility'] / 5.0)
        
        while len(features) < 18:
            features.append(0.0)
        
        return np.array(features[:18])
        
    except Exception as e:
        logging.error(f"❌ Error extrayendo features: {e}")
        return np.zeros(18)

# ------------------ INICIALIZACIÓN CON DEBUG ------------------
def start_system():
    try:
        logging.info("🔧 INICIANDO SISTEMA - DEBUG")
        logging.info("🔧 Paso 1: Sistema iniciando...")
        
        # ✅ INICIAR CONEXIÓN IQ OPTION INMEDIATAMENTE
        logging.info("🔄 Iniciando conexión a IQ Option...")
        
        # Intentar conexión inicial
        connection_result = iq_connector.connect()
        logging.info(f"🔧 Resultado conexión IQ Option: {connection_result}")
        
        if connection_result:
            logging.info("✅ Conexión IQ Option exitosa al inicio")
            dashboard_manager.dashboard.update_system_status("CONNECTED", "OPERATIONAL", "SYNCED")
        else:
            logging.error("❌ Conexión IQ Option falló al inicio")
            dashboard_manager.dashboard.update_system_status("DISCONNECTED", "ERROR", "SYNCED")
        
        # ✅ INICIAR THREAD DE TRADING
        logging.info("🔧 Iniciando thread de trading...")
        trading_thread = threading.Thread(target=premium_main_loop_deep_analysis, daemon=True)
        trading_thread.start()
        logging.info("🔧 Thread de trading iniciado")
        
        logging.info(f"⭐ DELOWYSS AI V5.8 INICIADA - ANÁLISIS PROFUNDO")
        logging.info("🎯 PREDICCIÓN A 5s - ANÁLISIS TICK POR TICK")
        logging.info("🌐 DASHBOARD DISPONIBLE EN: http://0.0.0.0:10000")
        
        # ✅ VERIFICAR QUE EL THREAD ESTÁ ACTIVO
        time.sleep(2)
        logging.info(f"🔧 Threads activos: {threading.active_count()}")
        logging.info(f"🔧 Threads: {[t.name for t in threading.enumerate()]}")
        
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")
        import traceback
        logging.error(f"❌ Traceback: {traceback.format_exc()}")

# ------------------ FUNCIÓN PRINCIPAL DE TRADING CORREGIDA ------------------
def premium_main_loop_deep_analysis():
    global _last_candle_start, _prediction_made_this_candle, _last_prediction_time, _last_price
    
    logging.info(f"🚀 LOOP DE TRADING INICIADO - ANÁLISIS PROFUNDO")
    
    # ✅ SINCRONIZAR METRÓNOMO
    try:
        logging.info("🔧 Sincronizando metrónomo...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dashboard_manager.metronome.sync_with_iqoption(iq_connector))
        loop.close()
        logging.info("✅ Metrónomo sincronizado en loop de trading")
    except Exception as e:
        logging.warning(f"⚠️ Error sincronizando metrónomo: {e}")
    
    # ✅ LOOP PRINCIPAL CON RECONEXIÓN
    logging.info("🔧 Entrando al loop principal de trading...")
    
    while True:
        try:
            # ✅ VERIFICAR CONEXIÓN Y RECONECTAR SI ES NECESARIO
            if not iq_connector.connected:
                logging.warning("🔌 IQ Option desconectado, intentando reconectar...")
                if iq_connector.connect():
                    logging.info("✅ Reconexión exitosa a IQ Option")
                    dashboard_manager.dashboard.update_system_status("CONNECTED", "OPERATIONAL", "SYNCED")
                else:
                    logging.error("❌ No se pudo reconectar a IQ Option")
                    dashboard_manager.dashboard.update_system_status("DISCONNECTED", "ERROR", "SYNCED")
                    time.sleep(10)
                    continue
            
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = iq_connector.get_remaining_time()
            
            # ✅ OBTENER PRECIO ACTUAL
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price
                
                # ✅ PROCESAR TICK
                tick_data = predictor.process_tick(price, seconds_remaining)
                
                if tick_data:
                    # ✅ ACTUALIZAR DASHBOARD
                    dashboard_manager.dashboard.update_candle_progress(
                        dashboard_manager.metronome,
                        price,
                        predictor.analyzer.tick_count
                    )
                    
                    # ✅ ACTUALIZAR MÉTRICAS CADA 2 SEGUNDOS
                    global _last_analysis_time
                    if current_time - _last_analysis_time >= 2:
                        analysis = predictor.analyzer.get_deep_analysis()
                        if analysis.get('status') == 'SUCCESS':
                            density = analysis.get('buy_pressure', 0.5) * 100
                            velocity = analysis.get('velocity', 0)
                            acceleration = analysis.get('acceleration', 0)
                            phase = analysis.get('market_phase', 'INICIAL')
                            
                            dashboard_manager.dashboard.update_metrics(
                                density, velocity, acceleration, phase, predictor.analyzer.tick_count
                            )
                            _last_analysis_time = current_time

            # ✅ VERIFICAR SI ES TIEMPO DE PREDICCIÓN
            prediction_time = (seconds_remaining <= PREDICTION_WINDOW and 
                             seconds_remaining > 0.5)
            
            if (prediction_time and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                not _prediction_made_this_candle):

                logging.info(f"🎯 PREDICCIÓN A {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                analysis = predictor.analyzer.get_deep_analysis()
                if analysis.get('status') == 'SUCCESS':
                    features = _extract_real_features(analysis)
                    ml_prediction = online_learner.predict(features)
                    
                    final_prediction = predictor.predict_next_candle(ml_prediction)
                    
                    # ✅ ACTUALIZAR DASHBOARD CON PREDICCIÓN
                    try:
                        dashboard_manager.dashboard.update_prediction(
                            final_prediction['direction'],
                            final_prediction['confidence']
                        )
                        
                        stats = predictor.get_performance_stats()
                        dashboard_manager.dashboard.update_performance(
                            stats['accuracy'],
                            stats['today_profit'],
                            stats['today_signals'],
                            stats['best_streak'],
                            stats['current_streak']
                        )
                        
                    except Exception as e:
                        logging.error(f"❌ Error actualizando dashboard: {e}")

                    _last_prediction_time = time.time()
                    _prediction_made_this_candle = True
                    
                    logging.info(f"🚀 PREDICCIÓN COMPLETADA: {final_prediction['direction']} {final_prediction['confidence']}%")

            # ✅ DETECCIÓN NUEVA VELA
            if current_candle_start > _last_candle_start:
                if _last_price is not None:
                    try:
                        validation = predictor.validate_prediction(_last_price)
                        if validation:
                            price_change = validation.get("price_change", 0)
                            actual_direction = validation.get("actual", "LATERAL")
                            
                            analysis = predictor.analyzer.get_deep_analysis()
                            if analysis.get('status') == 'SUCCESS':
                                features = _extract_real_features(analysis)
                                
                                if features is not None and features.size == 18:
                                    online_learner.add_sample(features, actual_direction)
                                    training_result = online_learner.partial_train(batch_size=16)
                                    
                                    if training_result.get('trained', False):
                                        logging.info(f"📚 AutoLearning: {actual_direction} | Cambio: {price_change:.1f}pips")
                    except Exception as e:
                        logging.warning(f"⚠️ Error en validación: {e}")

                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🕯️ NUEVA VELA - Análisis profundo reiniciado")

            time.sleep(0.1)  # ✅ REDUCIR SLEEP PARA MEJOR RESPONSIVIDAD
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(5)  # ✅ ESPERAR MÁS EN CASO DE ERROR

# ------------------ HTML INTERFAZ 100% RESPONSIVA ------------------
HTML_RESPONSIVE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Delowyss AI V5.8 - IQ Option REAL</title>
    <style>
        :root {
            --green-bright: #00ff88;
            --green: #00cc66;
            --green-light: #66ff99;
            --red-bright: #ff4444;
            --red: #cc0000;
            --red-light: #ff6666;
            --yellow: #ffcc00;
            --orange: #ff9900;
            --blue: #0099ff;
            --gray: #666666;
            --dark-bg: #0a0a0a;
            --card-bg: #1a1a1a;
            --text-light: #ffffff;
            --text-dim: #aaaaaa;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: var(--dark-bg);
            color: var(--text-light);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        .container {
            width: 100%;
            max-width: 100%;
            padding: 10px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 15px 10px;
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            border-radius: 12px;
            margin-bottom: 15px;
            border: 1px solid #333;
            position: relative;
            overflow: hidden;
        }
        
        .header h1 {
            font-size: clamp(1.5rem, 4vw, 2.5rem);
            background: linear-gradient(45deg, var(--green-bright), var(--blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        
        .subtitle {
            color: var(--text-dim);
            font-size: clamp(0.8rem, 2.5vw, 1rem);
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 12px;
            width: 100%;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid #333;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }
        
        .card-title {
            font-size: clamp(1rem, 3vw, 1.2rem);
            margin-bottom: 15px;
            color: var(--text-light);
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .prediction-card {
            grid-column: 1 / -1;
            text-align: center;
            background: linear-gradient(135deg, #1a1a2a, #2a1a2a);
        }
        
        .prediction-display {
            font-size: clamp(3rem, 15vw, 6rem);
            margin: 10px 0;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .pulse {
            animation: pulse 1s infinite;
        }
        
        .flash {
            animation: flash 0.5s;
        }
        
        .countdown-active {
            animation: countdownPulse 1s infinite;
            border: 2px solid var(--orange);
        }
        
        .prediction-change {
            animation: predictionChange 0.6s;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }
        
        @keyframes flash {
            0%, 100% { background-color: transparent; }
            50% { background-color: rgba(255, 255, 255, 0.1); }
        }
        
        @keyframes countdownPulse {
            0%, 100% { box-shadow: 0 0 10px rgba(255, 153, 0, 0.5); }
            50% { box-shadow: 0 0 20px rgba(255, 153, 0, 0.8); }
        }
        
        @keyframes predictionChange {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        .progress-container {
            width: 100%;
            margin: 15px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
            position: relative;
        }
        
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .metronome-display {
            font-family: 'Courier New', monospace;
            font-size: clamp(1.2rem, 4vw, 1.5rem);
            background: #000;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            border: 2px solid #333;
        }
        
        .last-5-seconds {
            color: var(--orange);
            font-weight: bold;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }
        
        .metric-item {
            background: #252525;
            padding: 12px 8px;
            border-radius: 8px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-item:hover {
            background: #2a2a2a;
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: clamp(1.1rem, 3vw, 1.3rem);
            font-weight: bold;
            margin: 5px 0;
        }
        
        .metric-label {
            color: var(--text-dim);
            font-size: 0.8rem;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
        }
        
        .status-item {
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            font-size: 0.9rem;
        }
        
        .status-connected {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid var(--green);
        }
        
        .status-warning {
            background: rgba(255, 204, 0, 0.1);
            border: 1px solid var(--yellow);
        }
        
        .status-error {
            background: rgba(255, 68, 68, 0.1);
            border: 1px solid var(--red);
        }
        
        .history-list {
            max-height: 200px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #333 transparent;
        }
        
        .history-list::-webkit-scrollbar {
            width: 6px;
        }
        
        .history-list::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .history-list::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 3px;
        }
        
        .history-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #333;
            font-size: 0.85rem;
        }
        
        .history-item:last-child {
            border-bottom: none;
        }
        
        .green-bright { color: var(--green-bright); }
        .green { color: var(--green); }
        .green-light { color: var(--green-light); }
        .red-bright { color: var(--red-bright); }
        .red { color: var(--red); }
        .red-light { color: var(--red-light); }
        .yellow { color: var(--yellow); }
        .orange { color: var(--orange); }
        .blue { color: var(--blue); }
        .gray { color: var(--gray); }
        
        .bg-green-bright { background: var(--green-bright); }
        .bg-green { background: var(--green); }
        .bg-red-bright { background: var(--red-bright); }
        .bg-red { background: var(--red); }
        .bg-yellow { background: var(--yellow); }
        .bg-orange { background: var(--orange); }
        .bg-blue { background: var(--blue); }
        
        @media (max-width: 480px) {
            .container {
                padding: 5px;
            }
            
            .card {
                padding: 12px;
            }
            
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .status-grid {
                grid-template-columns: 1fr;
            }
            
            .prediction-display {
                font-size: 4rem;
            }
        }
        
        @media (min-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .dashboard-grid {
                gap: 15px;
            }
            
            .prediction-card {
                grid-column: span 2;
            }
        }
        
        @media (min-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: repeat(4, 1fr);
            }
            
            .prediction-card {
                grid-column: 1 / -1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Delowyss AI V5.8</h1>
            <div class="subtitle">Análisis Profundo en Tiempo Real - IQ Option REAL</div>
        </div>
        
        <div class="dashboard-grid">
            <div class="card prediction-card" id="predictionCard">
                <div class="card-title">🎯 PREDICCIÓN ACTUAL - EURUSD 1min</div>
                
                <div class="metronome-display" id="metronomeDisplay">
                    <span id="timeRemaining">60.0</span>s
                    <span id="last5Indicator" style="display: none;" class="last-5-seconds"> - ÚLTIMOS 5s!</span>
                </div>
                
                <div class="prediction-display" id="predictionArrow">⏳</div>
                
                <div id="predictionInfo">
                    <div style="font-size: 1.3em; margin: 10px 0;" id="predictionText">ANALIZANDO MERCADO...</div>
                    
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" id="confidenceBar" style="width: 0%;"></div>
                        </div>
                        <div id="confidenceText" style="margin-top: 5px;">Confianza: 0%</div>
                    </div>
                    
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill bg-blue" id="candleProgress" style="width: 0%;"></div>
                        </div>
                        <div id="candleInfo" style="margin-top: 5px; font-size: 0.9em;">Vela: 0/60s | Ticks: 0</div>
                    </div>
                    
                    <div id="priceInfo" style="font-size: 1.1em; margin-top: 10px; font-family: monospace;">
                        Precio: <span id="currentPrice">-</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📊 MÉTRICAS EN TIEMPO REAL</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Densidad</div>
                        <div class="metric-value" id="densityValue">0%</div>
                        <div class="metric-label" id="densityDirection">NEUTRAL</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Velocidad</div>
                        <div class="metric-value" id="velocityValue">0.0x</div>
                        <div class="metric-label">vs Promedio</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Aceleración</div>
                        <div class="metric-value" id="accelerationValue">0.0σ</div>
                        <div class="metric-label">Desviación</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Señales</div>
                        <div class="metric-value" id="signalCount">0</div>
                        <div class="metric-label">Activas</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">🏆 PERFORMANCE</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Precisión</div>
                        <div class="metric-value" id="accuracyValue">0%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Profit</div>
                        <div class="metric-value" id="profitValue">0%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Racha +</div>
                        <div class="metric-value" id="winStreak">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Señales</div>
                        <div class="metric-value" id="totalSignals">0</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">🔧 ESTADO DEL SISTEMA</div>
                <div class="status-grid">
                    <div class="status-item" id="iqStatus">
                        <div>🌐 IQ OPTION</div>
                        <div class="status-value">DESCONECTADO</div>
                    </div>
                    <div class="status-item" id="aiStatus">
                        <div>🤖 IA STATUS</div>
                        <div class="status-value">INICIANDO</div>
                    </div>
                    <div class="status-item" id="metronomeStatus">
                        <div>⏱️ METRÓNOMO</div>
                        <div class="status-value">NO SINCRONIZADO</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 10px; font-size: 0.8em; color: var(--text-dim);">
                    Actualizado: <span id="lastUpdate">-</span>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📈 HISTORIAL RECIENTE</div>
                <div class="history-list" id="historyList">
                    <div class="history-item">
                        <span>Sistema iniciado</span>
                        <span class="gray">Conectando...</span>
                    </div>
                </div>
            </div>
            
        </div>
    </div>

    <script>
        class DashboardManager {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.isConnected = false;
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.startUIUpdates();
            }
            
            connectWebSocket() {
                try {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = (event) => {
                        console.log('✅ Conectado al servidor WebSocket');
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this.addHistoryItem('Conectado al servidor', 'success');
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type === 'dashboard_update') {
                                this.updateDashboard(data.data);
                            }
                        } catch (error) {
                            console.error('Error procesando mensaje:', error);
                        }
                    };
                    
                    this.ws.onclose = (event) => {
                        console.log('❌ Conexión WebSocket cerrada');
                        this.isConnected = false;
                        this.addHistoryItem('Conexión perdida', 'error');
                        this.handleReconnection();
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.isConnected = false;
                    };
                    
                } catch (error) {
                    console.error('Error conectando WebSocket:', error);
                    this.handleReconnection();
                }
            }
            
            handleReconnection() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    const delay = Math.min(1000 * this.reconnectAttempts, 10000);
                    
                    console.log(`🔄 Reconectando en ${delay}ms... (Intento ${this.reconnectAttempts})`);
                    this.addHistoryItem(`Reconectando... Intento ${this.reconnectAttempts}`, 'warning');
                    
                    setTimeout(() => {
                        this.connectWebSocket();
                    }, delay);
                } else {
                    console.error('❌ Máximo de intentos de reconexión alcanzado');
                    this.addHistoryItem('Error de conexión persistente', 'error');
                }
            }
            
            updateDashboard(data) {
                this.updatePrediction(data.current_prediction);
                this.updateCandleInfo(data.current_candle);
                this.updateMetrics(data.metrics);
                this.updatePerformance(data.performance);
                this.updateSystemStatus(data.system_status);
                this.applyVisualEffects(data.visual_effects);
            }
            
            updatePrediction(prediction) {
                const arrowElement = document.getElementById('predictionArrow');
                const textElement = document.getElementById('predictionText');
                const barElement = document.getElementById('confidenceBar');
                const confidenceText = document.getElementById('confidenceText');
                
                arrowElement.textContent = prediction.arrow;
                arrowElement.className = `prediction-display ${prediction.color}`;
                textElement.textContent = `${prediction.direction} ${prediction.confidence}%`;
                barElement.style.width = `${prediction.confidence}%`;
                barElement.className = `progress-fill bg-${prediction.color.split('-')[0]}`;
                confidenceText.textContent = `Confianza: ${prediction.confidence}%`;
            }
            
            updateCandleInfo(candle) {
                const progressElement = document.getElementById('candleProgress');
                const infoElement = document.getElementById('candleInfo');
                const priceElement = document.getElementById('currentPrice');
                const timeElement = document.getElementById('timeRemaining');
                const last5Element = document.getElementById('last5Indicator');
                const predictionCard = document.getElementById('predictionCard');
                
                progressElement.style.width = `${candle.progress}%`;
                infoElement.textContent = `Vela: ${60 - candle.time_remaining}/60s | Ticks: ${candle.ticks_processed}`;
                priceElement.textContent = candle.price.toFixed(5);
                timeElement.textContent = candle.time_remaining.toFixed(1);
                
                if (candle.is_last_5_seconds) {
                    last5Element.style.display = 'inline';
                    predictionCard.classList.add('countdown-active');
                } else {
                    last5Element.style.display = 'none';
                    predictionCard.classList.remove('countdown-active');
                }
            }
            
            updateMetrics(metrics) {
                document.getElementById('densityValue').textContent = `${metrics.density}%`;
                document.getElementById('densityValue').className = `metric-value ${this.getDensityColor(metrics.density)}`;
                document.getElementById('densityDirection').textContent = 
                    metrics.density >= 60 ? 'ALCISTA' : metrics.density <= 40 ? 'BAJISTA' : 'NEUTRAL';
                
                document.getElementById('velocityValue').textContent = `${metrics.velocity}x`;
                document.getElementById('accelerationValue').textContent = `${metrics.acceleration}σ`;
                document.getElementById('signalCount').textContent = metrics.signal_count;
            }
            
            updatePerformance(performance) {
                document.getElementById('accuracyValue').textContent = `${performance.today_accuracy}%`;
                document.getElementById('profitValue').textContent = `${performance.today_profit}%`;
                document.getElementById('winStreak').textContent = performance.current_streak;
                document.getElementById('totalSignals').textContent = performance.total_signals;
            }
            
            updateSystemStatus(status) {
                this.updateStatusElement('iqStatus', status.iq_connection, 'IQ Option');
                this.updateStatusElement('aiStatus', status.ai_status, 'IA Status');
                this.updateStatusElement('metronomeStatus', status.metronome_sync, 'Metrónomo');
                document.getElementById('lastUpdate').textContent = status.last_update;
            }
            
            updateStatusElement(elementId, status, label) {
                const element = document.getElementById(elementId);
                element.className = 'status-item ';
                
                if (status === 'CONNECTED' || status === 'OPERATIONAL' || status === 'SYNCED') {
                    element.classList.add('status-connected');
                    element.innerHTML = `<div>${label}</div><div class="status-value">✅ ${status}</div>`;
                } else if (status === 'DISCONNECTED' || status === 'ERROR' || status === 'UNSYNCED') {
                    element.classList.add('status-error');
                    element.innerHTML = `<div>${label}</div><div class="status-value">❌ ${status}</div>`;
                } else {
                    element.classList.add('status-warning');
                    element.innerHTML = `<div>${label}</div><div class="status-value">🟡 ${status}</div>`;
                }
            }
            
            applyVisualEffects(effects) {
                const predictionCard = document.getElementById('predictionCard');
                const predictionArrow = document.getElementById('predictionArrow');
                
                if (effects.pulse_animation) {
                    predictionCard.classList.add('pulse');
                } else {
                    predictionCard.classList.remove('pulse');
                }
                
                if (effects.flash_signal) {
                    predictionCard.classList.add('flash');
                    setTimeout(() => {
                        predictionCard.classList.remove('flash');
                    }, 500);
                }
                
                if (effects.prediction_change) {
                    predictionArrow.classList.add('prediction-change');
                    setTimeout(() => {
                        predictionArrow.classList.remove('prediction-change');
                    }, 600);
                }
            }
            
            getDensityColor(density) {
                if (density >= 70) return 'green';
                if (density >= 60) return 'green-light';
                if (density <= 30) return 'red';
                if (density <= 40) return 'red-light';
                return 'yellow';
            }
            
            addHistoryItem(message, type = 'info') {
                const historyList = document.getElementById('historyList');
                const newItem = document.createElement('div');
                newItem.className = 'history-item';
                
                const timestamp = new Date().toLocaleTimeString();
                const colorClass = type === 'success' ? 'green' : 
                                 type === 'error' ? 'red' : 
                                 type === 'warning' ? 'yellow' : 'gray';
                
                newItem.innerHTML = `
                    <span>${message}</span>
                    <span class="${colorClass}">${timestamp}</span>
                `;
                
                historyList.insertBefore(newItem, historyList.firstChild);
                
                const maxItems = window.innerWidth < 768 ? 8 : 15;
                if (historyList.children.length > maxItems) {
                    historyList.removeChild(historyList.lastChild);
                }
            }
            
            startUIUpdates() {
                setInterval(() => {
                    // Animaciones suaves adicionales
                }, 100);
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            window.dashboard = new DashboardManager();
            
            setTimeout(() => {
                window.dashboard.addHistoryItem('Dashboard inicializado', 'success');
            }, 1000);
        });
        
        window.addEventListener('beforeunload', () => {
            if (window.dashboard && window.dashboard.ws) {
                window.dashboard.ws.close();
            }
        });
    </script>
</body>
</html>
'''

# ------------------ SISTEMA PRINCIPAL CORREGIDO ------------------
# Instancias globales
iq_connector = RealIQOptionConnector(IQ_EMAIL, IQ_PASSWORD, PAR)
predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner(feature_size=18)
dashboard_manager = AdvancedConnectionManager()

# Variables globales
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False
_last_prediction_time = 0
_last_price = None
_last_analysis_time = 0

# ------------------ FASTAPI APP ------------------
app = FastAPI(
    title="Delowyss Trading AI V5.8 - Análisis Profundo",
    description="Sistema de IA con análisis profundo tick por tick - IQ Option REAL",
    version="5.8.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ CONFIGURACIÓN RUTAS UI ------------------
def setup_responsive_routes(app: FastAPI, manager: AdvancedConnectionManager, iq_connector):
    @app.get("/", response_class=HTMLResponse)
    async def get_responsive_dashboard():
        return HTML_RESPONSIVE

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(continuous_dashboard_updates(manager, iq_connector))

async def continuous_dashboard_updates(manager: AdvancedConnectionManager, iq_connector):
    while True:
        try:
            if time.time() - manager.metronome.last_sync_time > 30:
                try:
                    await manager.metronome.sync_with_iqoption(iq_connector)
                    manager.dashboard.update_system_status(
                        "CONNECTED" if iq_connector.connected else "DISCONNECTED",
                        "OPERATIONAL",
                        "SYNCED" if manager.metronome.last_sync_time > 0 else "UNSYNCED"
                    )
                except Exception as e:
                    logging.warning(f"⚠️ Error sincronizando metrónomo: {e}")
            
            current_price = iq_connector.current_price or 0.0
            ticks_processed = iq_connector.tick_count
            
            manager.dashboard.update_candle_progress(
                manager.metronome, 
                current_price, 
                ticks_processed
            )
            
            await manager.broadcast_dashboard_update()
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error en actualización continua: {e}")
            await asyncio.sleep(1)

# Configurar rutas
setup_responsive_routes(app, dashboard_manager, iq_connector)

# ------------------ ENDPOINTS API ------------------
@app.get("/api/prediction")
async def get_prediction():
    analysis = predictor.analyzer.get_deep_analysis()
    if analysis.get('status') == 'SUCCESS':
        return {
            "direction": dashboard_manager.dashboard.dashboard_data["current_prediction"]["direction"],
            "confidence": dashboard_manager.dashboard.dashboard_data["current_prediction"]["confidence"],
            "current_price": analysis['current_price'],
            "tick_count": predictor.analyzer.tick_count,
            "timestamp": now_iso()
        }
    return {"status": "ANALYZING", "message": "Procesando datos..."}

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

# ✅ INICIAR SISTEMA COMPLETO
if __name__ == "__main__":
    start_system()
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True
    )
else:
    start_system()
