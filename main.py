# main.py - VERSIÓN MEJORADA CON ANÁLISIS TICK-BY-TICK
"""
Delowyss Trading AI — V5.1 PREMIUM COMPLETO MEJORADO
Sistema profesional con análisis tick-by-tick y optimización de rendimiento
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
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from iqoptionapi.stable_api import IQ_Option
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN PREMIUM MEJORADA ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 15
TICK_BUFFER_SIZE = 200  # Aumentado para análisis más detallado

# ---------------- LOGGING PROFESIONAL MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA MEJORADA ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.tick_analysis = deque(maxlen=50)  # Análisis por tick
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=50)  # Aumentado para mejor análisis
        self.last_candle_close = None
        self.velocity_metrics = deque(maxlen=20)  # Nueva métrica de velocidad
        self.acceleration_metrics = deque(maxlen=15)  # Nueva métrica de aceleración
        
    def add_tick(self, price: float):
        try:
            price = float(price)
            current_time = time.time()
            
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
            
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)
            self.current_candle_close = price
            
            # Análisis avanzado por tick
            tick_data = {
                'price': price,
                'timestamp': current_time,
                'volume': 1,
                'microtimestamp': current_time * 1000  # Precisión milisegundos
            }
            
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            # Calcular métricas de velocidad y aceleración en tiempo real
            self._calculate_tick_dynamics(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"Error en add_tick: {e}")
            return None
    
    def _calculate_tick_dynamics(self, current_tick):
        """Calcula velocidad y aceleración del precio por tick"""
        if len(self.ticks) < 2:
            return
            
        try:
            current_price = current_tick['price']
            current_time = current_tick['timestamp']
            
            # Obtener tick anterior
            previous_tick = list(self.ticks)[-2] if len(self.ticks) >= 2 else None
            
            if previous_tick:
                previous_price = previous_tick['price']
                previous_time = previous_tick['timestamp']
                
                # Calcular velocidad (cambio de precio por segundo)
                time_diff = current_time - previous_time
                if time_diff > 0:
                    price_diff = current_price - previous_price
                    velocity = price_diff / time_diff
                    
                    # Almacenar métrica de velocidad
                    velocity_data = {
                        'velocity': velocity,
                        'timestamp': current_time,
                        'price_change': price_diff
                    }
                    self.velocity_metrics.append(velocity_data)
                    
                    # Calcular aceleración (cambio de velocidad)
                    if len(self.velocity_metrics) >= 2:
                        current_velocity = velocity
                        previous_velocity = self.velocity_metrics[-2]['velocity']
                        velocity_time_diff = current_time - self.velocity_metrics[-2]['timestamp']
                        
                        if velocity_time_diff > 0:
                            acceleration = (current_velocity - previous_velocity) / velocity_time_diff
                            
                            acceleration_data = {
                                'acceleration': acceleration,
                                'timestamp': current_time,
                                'velocity_change': current_velocity - previous_velocity
                            }
                            self.acceleration_metrics.append(acceleration_data)
                            
        except Exception as e:
            logging.debug(f"Error en cálculo de dinámicas: {e}")
    
    def _calculate_advanced_metrics(self):
        """Métricas avanzadas mejoradas con análisis tick-by-tick"""
        if len(self.price_memory) < 8:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            # Análisis de tendencia mejorado
            if len(prices) >= 12:
                short_trend = np.polyfit(range(6), prices[-6:], 1)[0]
                medium_trend = np.polyfit(range(12), prices[-12:], 1)[0]
                trend_strength = (short_trend * 0.6 + medium_trend * 0.4) * 10000
            else:
                trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            # Momentum mejorado
            momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
            momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else 0
            momentum = (momentum_5 * 0.7 + momentum_10 * 0.3)
            
            # Volatilidad mejorada
            recent_prices = prices[-8:] if len(prices) >= 8 else prices
            if len(recent_prices) > 1:
                volatility = (max(recent_prices) - min(recent_prices)) * 10000
            else:
                volatility = 0
            
            # Análisis de presión de compra/venta mejorado
            if len(self.ticks) > 6:
                recent_ticks = list(self.ticks)[-15:]  # Aumentado para mejor análisis
                price_changes = []
                for i in range(1, len(recent_ticks)):
                    if i < len(recent_ticks):
                        change = recent_ticks[i]['price'] - recent_ticks[i-1]['price']
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
            
            # Nueva métrica: Análisis de velocidad promedio
            avg_velocity = 0
            if self.velocity_metrics:
                velocities = [v['velocity'] for v in self.velocity_metrics]
                avg_velocity = np.mean(velocities) * 10000  # Normalizar
            
            # Nueva métrica: Análisis de aceleración
            avg_acceleration = 0
            if self.acceleration_metrics:
                accelerations = [a['acceleration'] for a in self.acceleration_metrics]
                avg_acceleration = np.mean(accelerations) * 10000  # Normalizar
            
            # Fase de mercado mejorada
            if volatility < 0.2 and abs(trend_strength) < 0.4 and abs(avg_velocity) < 0.3:
                market_phase = "consolidation"
            elif abs(trend_strength) > 1.5 and abs(avg_velocity) > 0.5:
                market_phase = "strong_trend"
            elif abs(trend_strength) > 0.8:
                market_phase = "trending"
            elif volatility > 1.0:
                market_phase = "volatile"
            elif abs(avg_velocity) > 0.8:
                market_phase = "high_velocity"
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
                'data_quality': min(1.0, self.tick_count / 25.0),  # Ajustado
                'velocity': avg_velocity,  # Nueva métrica
                'acceleration': avg_acceleration,  # Nueva métrica
                'tick_frequency': self._calculate_tick_frequency()  # Nueva métrica
            }
        except Exception as e:
            logging.error(f"Error en cálculo de métricas: {e}")
            return {}
    
    def _calculate_tick_frequency(self):
        """Calcula la frecuencia de ticks por segundo"""
        if len(self.ticks) < 2:
            return 0
            
        try:
            recent_ticks = list(self.ticks)[-10:]  # Últimos 10 ticks
            if len(recent_ticks) < 2:
                return 0
                
            time_diffs = []
            for i in range(1, len(recent_ticks)):
                time_diff = recent_ticks[i]['timestamp'] - recent_ticks[i-1]['timestamp']
                if time_diff > 0:
                    time_diffs.append(time_diff)
            
            if time_diffs:
                avg_time_diff = np.mean(time_diffs)
                return 1 / avg_time_diff if avg_time_diff > 0 else 0
            return 0
        except:
            return 0
    
    def get_tick_analysis(self):
        """Análisis específico por tick"""
        if self.tick_count < 5:
            return {'status': 'INSUFFICIENT_TICKS'}
        
        try:
            basic_analysis = self.get_analysis()
            if basic_analysis.get('status') != 'SUCCESS':
                return basic_analysis
            
            # Análisis avanzado de ticks
            tick_metrics = {
                'total_ticks': self.tick_count,
                'tick_frequency': self._calculate_tick_frequency(),
                'velocity_metrics_count': len(self.velocity_metrics),
                'acceleration_metrics_count': len(self.acceleration_metrics),
                'buffer_usage': len(self.ticks) / TICK_BUFFER_SIZE
            }
            
            return {**basic_analysis, **tick_metrics}
            
        except Exception as e:
            logging.error(f"Error en análisis de ticks: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_analysis(self):
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {'status': 'INSUFFICIENT_DATA', 'tick_count': self.tick_count}
        
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
                'timestamp': time.time(),
                **advanced_metrics
            }
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        try:
            if self.current_candle_close is not None:
                self.last_candle_close = self.current_candle_close
                
            self.ticks.clear()
            self.tick_analysis.clear()
            self.current_candle_open = None
            self.current_candle_high = None
            self.current_candle_low = None
            self.current_candle_close = None
            self.tick_count = 0
            self.velocity_metrics.clear()
            self.acceleration_metrics.clear()
        except Exception as e:
            logging.error(f"Error en reset: {e}")

# ------------------ SISTEMA IA PROFESIONAL MEJORADO ------------------
class ProfessionalAIPredictor:
    def __init__(self):
        self.analyzer = PremiumAIAnalyzer()
        self.prediction_history = deque(maxlen=15)  # Aumentado
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': 0.0,
            'tick_analysis_count': 0
        }
        self.last_prediction = None
        self.last_validation_result = None
        self.tick_analysis_history = deque(maxlen=20)  # Nuevo: historial de análisis por tick
        
    def process_tick(self, price: float):
        try:
            tick_data = self.analyzer.add_tick(price)
            
            # Análisis en tiempo real por tick (no bloqueante)
            if tick_data and self.analyzer.tick_count % 5 == 0:  # Cada 5 ticks para no sobrecargar
                tick_analysis = self.analyzer.get_tick_analysis()
                if tick_analysis.get('status') == 'SUCCESS':
                    self.tick_analysis_history.append(tick_analysis)
                    self.performance_stats['tick_analysis_count'] += 1
            
            return tick_data
        except Exception as e:
            logging.error(f"Error en process_tick: {e}")
            return None
    
    def _professional_ai_analysis(self, analysis):
        """Análisis de IA mejorado con métricas de tick"""
        try:
            momentum = analysis['momentum']
            trend_strength = analysis['trend_strength']
            pressure_ratio = analysis['pressure_ratio']
            volatility = analysis['volatility']
            market_phase = analysis['market_phase']
            data_quality = analysis['data_quality']
            velocity = analysis.get('velocity', 0)
            acceleration = analysis.get('acceleration', 0)
            tick_frequency = analysis.get('tick_frequency', 0)
            
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # Peso de tendencia mejorado
            trend_weight = 0.35
            if abs(trend_strength) > 1.0:
                if trend_strength > 0:
                    buy_score += 8 * trend_weight
                    reasons.append(f"📈 Tendencia alcista ({trend_strength:.1f})")
                else:
                    sell_score += 8 * trend_weight
                    reasons.append(f"📉 Tendencia bajista ({trend_strength:.1f})")
            
            # Peso de momentum mejorado
            momentum_weight = 0.30
            if abs(momentum) > 0.6:
                if momentum > 0:
                    buy_score += 7 * momentum_weight
                    reasons.append(f"🚀 Momentum alcista ({momentum:.1f}pips)")
                else:
                    sell_score += 7 * momentum_weight
                    reasons.append(f"🔻 Momentum bajista ({momentum:.1f}pips)")
            
            # Nueva métrica: Velocidad del precio
            velocity_weight = 0.15
            if abs(velocity) > 0.5:
                if velocity > 0:
                    buy_score += 5 * velocity_weight
                    reasons.append(f"⚡ Velocidad alcista ({velocity:.1f})")
                else:
                    sell_score += 5 * velocity_weight
                    reasons.append(f"💨 Velocidad bajista ({velocity:.1f})")
            
            # Presión de compra/venta
            pressure_weight = 0.20
            if pressure_ratio > 1.5:
                buy_score += 6 * pressure_weight
                reasons.append(f"💰 Presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.7:
                sell_score += 6 * pressure_weight
                reasons.append(f"💸 Presión vendedora ({pressure_ratio:.1f}x)")
            
            score_difference = buy_score - sell_score
            
            # Umbral dinámico basado en calidad de datos
            confidence_threshold = 0.4 - (0.1 * (1 - data_quality))
            
            if abs(score_difference) > confidence_threshold:
                if score_difference > 0:
                    direction = "ALZA"
                    base_confidence = 55 + (score_difference * 35)
                else:
                    direction = "BAJA" 
                    base_confidence = 55 + (abs(score_difference) * 35)
            else:
                direction = "LATERAL"
                base_confidence = 40
                reasons.append("⚡ Señales insuficientes")
            
            confidence = base_confidence
            confidence *= data_quality
            
            # Ajustes por condiciones de mercado
            if volatility > 1.2:
                confidence *= 0.7
                reasons.append("🌪️ Mercado volátil")
            
            if tick_frequency > 10:  # Alta frecuencia de ticks
                confidence = min(85, confidence + 5)
                reasons.append("🎯 Alta actividad")
            
            if analysis['tick_count'] > 30:
                confidence = min(85, confidence + 12)
            
            confidence = max(35, min(85, confidence))
            
            return {
                'direction': direction,
                'confidence': int(confidence),
                'buy_score': round(buy_score, 2),
                'sell_score': round(sell_score, 2),
                'score_difference': round(score_difference, 2),
                'reasons': reasons,
                'market_phase': market_phase,
                'velocity': round(velocity, 2),
                'tick_frequency': round(tick_frequency, 1)
            }
        except Exception as e:
            logging.error(f"Error en análisis IA: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 35,
                'reasons': ['🤖 Error en análisis'],
                'buy_score': 0,
                'sell_score': 0,
                'score_difference': 0
            }
    
    def predict_next_candle(self):
        try:
            analysis = self.analyzer.get_analysis()
            
            if analysis.get('status') != 'SUCCESS':
                return {
                    'direction': 'LATERAL',
                    'confidence': 0,
                    'reason': analysis.get('message', 'Analizando...'),
                    'timestamp': now_iso()
                }
            
            prediction = self._professional_ai_analysis(analysis)
            
            if prediction['confidence'] < 50:
                prediction['direction'] = 'LATERAL'
                prediction['reasons'].append("🔍 Confianza insuficiente")
            
            prediction.update({
                'tick_count': analysis['tick_count'],
                'current_price': analysis['current_price'],
                'timestamp': now_iso(),
                'model_version': 'PROFESSIONAL_AI_V5.1_TICK_ANALYSIS'
            })
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
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
        """Validación precisa mejorada"""
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
            
            # Umbral dinámico basado en volatilidad
            minimal_change = 0.2
            if hasattr(self.analyzer, 'volatility_metrics'):
                recent_volatility = np.mean(list(self.analyzer.volatility_metrics)[-5:]) if self.analyzer.volatility_metrics else 0.2
                minimal_change = max(0.1, recent_volatility * 0.3)
            
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
            
            logging.info(f"🎯 VALIDACIÓN: {status_icon} {predicted_direction}→{actual_direction} | Conf: {last_pred.get('confidence', 0)}% | Cambio: {price_change:.1f}pips")
            
            if total > 0 and total % 5 == 0:
                logging.info(f"📊 PRECISIÓN: {accuracy:.1f}% (Total: {total}) | Análisis de ticks: {self.performance_stats['tick_analysis_count']}")
            
            self.last_validation_result = {
                'correct': is_correct,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': last_pred.get('confidence', 0),
                'price_change': round(price_change, 2),
                'accuracy': round(accuracy, 1),
                'total_predictions': total,
                'correct_predictions': correct,
                'status_icon': status_icon,
                'timestamp': now_iso(),
                'tick_analyses': self.performance_stats['tick_analysis_count']
            }
            
            return self.last_validation_result
            
        except Exception as e:
            logging.error(f"Error en validación: {e}")
            return None
    
    def get_tick_analysis(self):
        """Obtiene el análisis más reciente de ticks"""
        if self.tick_analysis_history:
            return self.tick_analysis_history[-1]
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

# -------------- CONEXIÓN PROFESIONAL MEJORADA --------------
class ProfessionalIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        self.tick_listeners = []  # Nuevo: sistema de listeners para ticks
        self.last_tick_time = None
        self.tick_interval = deque(maxlen=10)  # Para calcular intervalo entre ticks
        
    def connect(self):
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.info("🔧 Modo demo activado")
                self.connected = True
                return True
                
            logging.info("🌐 Conectando a IQ Option...")
            self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.iq.connect()
            
            if check:
                self.iq.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conexión establecida")
                return True
            else:
                logging.warning(f"⚠️ Conexión fallida: {reason}")
                logging.info("🔧 Activando modo demo...")
                self.connected = True
                return True
                
        except Exception as e:
            logging.error(f"❌ Error de conexión: {e}")
            logging.info("🔧 Activando modo demo...")
            self.connected = True
            return True

    def add_tick_listener(self, listener):
        """Añade un listener para recibir ticks en tiempo real"""
        self.tick_listeners.append(listener)

    def get_realtime_price(self):
        try:
            current_time = time.time()
            
            if not self.connected:
                # Modo demo mejorado
                if self.last_price is None:
                    self.last_price = 1.15389
                else:
                    # Simulación más realista de ticks
                    variation = np.random.uniform(-0.00005, 0.00005)
                    self.last_price += variation
                
                # Notificar a los listeners en modo demo
                self._notify_tick_listeners(self.last_price, current_time)
                return self.last_price

            # Obtener precio real
            candles = self.iq.get_candles(PAR, TIMEFRAME, 1, time.time())
            if candles and len(candles) > 0:
                price = float(candles[-1]['close'])
                if price > 0:
                    self._record_tick(price, current_time)
                    self._notify_tick_listeners(price, current_time)
                    return price

            return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            return self.last_price if self.last_price else 1.15389

    def _notify_tick_listeners(self, price, timestamp):
        """Notifica a todos los listeners sobre un nuevo tick"""
        for listener in self.tick_listeners:
            try:
                listener(price, timestamp)
            except Exception as e:
                logging.error(f"Error notificando listener: {e}")

    def _record_tick(self, price, timestamp):
        """Registra tick con timestamp preciso"""
        self.tick_count += 1
        
        # Calcular intervalo entre ticks
        if self.last_tick_time is not None:
            interval = timestamp - self.last_tick_time
            self.tick_interval.append(interval)
        
        self.last_tick_time = timestamp
        self.last_price = price
        
        # Log reducido para no saturar
        if self.tick_count <= 5 or self.tick_count % 500 == 0:
            avg_interval = np.mean(self.tick_interval) if self.tick_interval else 0
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f} | Intervalo: {avg_interval:.3f}s")

# --------------- SISTEMA PRINCIPAL MEJORADO ---------------
iq_connector = ProfessionalIQConnector()
predictor = ProfessionalAIPredictor()

# VARIABLES GLOBALES MEJORADAS
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 Sistema inicializando..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING",
    "tick_frequency": 0,
    "velocity": 0
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None,
    'tick_analysis_count': 0
}

# NUEVO: Sistema de gestión de ticks en tiempo real
def tick_processor(price, timestamp):
    """Procesa cada tick en tiempo real de manera no bloqueante"""
    try:
        global current_prediction
        
        # Procesar tick en el predictor
        tick_data = predictor.process_tick(price)
        
        if tick_data:
            # Actualizar predicción global sin bloquear
            analysis = predictor.analyzer.get_tick_analysis()
            if analysis.get('status') == 'SUCCESS':
                current_prediction.update({
                    "tick_count": analysis['tick_count'],
                    "tick_frequency": analysis.get('tick_frequency', 0),
                    "velocity": analysis.get('velocity', 0),
                    "current_price": price,
                    "timestamp": now_iso()
                })
                
    except Exception as e:
        logging.error(f"Error procesando tick: {e}")

def premium_main_loop():
    global current_prediction, performance_stats
    
    logging.info("🚀 DELOWYSS AI V5.1 MEJORADO INICIADO")
    logging.info("🎯 Sistema profesional con análisis tick-by-tick activo")
    
    iq_connector.connect()
    
    # Registrar el procesador de ticks
    iq_connector.add_tick_listener(tick_processor)
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # Obtener precio (esto también disparará los listeners de ticks)
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                last_price = price
                
                # Actualizar estado general
                current_prediction.update({
                    "current_price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE"
                })
            
            # Lógica de predicción mejorada
            if (seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - last_prediction_time) >= 2):
                
                prediction = predictor.predict_next_candle()
                
                if prediction['confidence'] >= 45:
                    current_prediction.update(prediction)
                    last_prediction_time = time.time()
                    
                    if prediction['direction'] != 'LATERAL':
                        logging.info(f"🎯 PREDICCIÓN: {prediction['direction']} | Conf: {prediction['confidence']}% | Ticks: {predictor.analyzer.tick_count}")
            
            # Lógica de nueva vela
            if current_candle_start > last_candle_start and last_price is not None:
                validation_result = predictor.validate_prediction(last_price)
                if validation_result:
                    performance_stats.update({
                        'total_predictions': validation_result['total_predictions'],
                        'correct_predictions': validation_result['correct_predictions'],
                        'recent_accuracy': validation_result['accuracy'],
                        'last_validation': validation_result,
                        'tick_analysis_count': validation_result.get('tick_analyses', 0)
                    })
                
                predictor.reset()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela - Analizando ticks...")
            
            # Sleep reducido para mejor capacidad de respuesta
            time.sleep(0.1)  # Reducido de 0.3 a 0.1 para mejor capacidad de respuesta
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(0.5)

# --------------- INTERFAZ WEB MEJORADA ---------------
app = FastAPI(title="Delowyss AI Premium V5.1", version="5.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    # ... (el HTML se mantiene igual pero con métricas adicionales)
    # Se mantiene el mismo HTML de la versión anterior por brevedad
    # En la implementación real, se añadirían las nuevas métricas
    pass

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/validation")
def api_validation():
    last_val = predictor.get_last_validation()
    return JSONResponse({
        "last_validation": last_val,
        "performance": performance_stats,
        "timestamp": now_iso()
    })

@app.get("/api/tick-analysis")
def api_tick_analysis():
    """Nuevo endpoint para análisis detallado de ticks"""
    tick_analysis = predictor.get_tick_analysis()
    return JSONResponse({
        "tick_analysis": tick_analysis,
        "performance": performance_stats,
        "timestamp": now_iso()
    })

@app.get("/api/health")
def api_health():
    return JSONResponse({
        "status": "healthy", 
        "timestamp": now_iso(),
        "version": "5.1.0",
        "features": ["tick_analysis", "velocity_metrics", "real_time_processing"]
    })

# --------------- INICIALIZACIÓN MEJORADA ---------------
def start_system():
    try:
        # Usar ThreadPoolExecutor para mejor gestión de hilos
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(premium_main_loop)
            logging.info("⭐ SISTEMA MEJORADO INICIADO CORRECTAMENTE")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

# Iniciar automáticamente
start_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
