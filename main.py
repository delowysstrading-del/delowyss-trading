# main.py - VERSIÓN CON ANÁLISIS COMPLETO DE VELA ACTUAL
"""
Delowyss Trading AI — V5.3 PREMIUM CON ANÁLISIS COMPLETO DE VELA
Sistema con análisis tick-by-tick desde inicio de vela hasta últimos 5 segundos
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
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN PARA ANÁLISIS COMPLETO DE VELA ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = 60
PREDICTION_WINDOW = 5  # Últimos 5 segundos para predicción
MIN_TICKS_FOR_PREDICTION = 20  # Suficiente para análisis completo
TICK_BUFFER_SIZE = 500  # Buffer amplio para toda la vela

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA CON ANÁLISIS COMPLETO DE VELA ------------------
class CompleteCandleAIAnalyzer:
    def __init__(self):
        # Buffers principales para toda la vela
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=100)
        self.last_candle_close = None
        
        # Métricas avanzadas para análisis completo
        self.velocity_metrics = deque(maxlen=50)
        self.acceleration_metrics = deque(maxlen=30)
        self.volume_profile = deque(maxlen=20)  # Perfil de volumen por segmentos
        self.price_levels = deque(maxlen=15)    # Niveles de precio importantes
        
        # Estados del análisis
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
            
            # Calcular métricas en tiempo real
            self._calculate_comprehensive_metrics(tick_data)
            
            # Análisis por fases de la vela
            self._analyze_candle_phase(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"Error en add_tick: {e}")
            return None
    
    def _calculate_comprehensive_metrics(self, current_tick):
        """Calcula métricas completas basadas en todos los ticks de la vela"""
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
        """Analiza la vela por fases temporales"""
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
        """Obtiene análisis específico por fase"""
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
        """Métricas avanzadas basadas en análisis completo de la vela"""
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
        """Combina análisis de todas las fases de la vela"""
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
        """Análisis completo basado en todos los ticks de la vela actual"""
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
    
    def reset(self):
        """Reinicia el análisis para nueva vela"""
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

# ------------------ SISTEMA IA CON ANÁLISIS COMPLETO ------------------
class ComprehensiveAIPredictor:
    def __init__(self):
        self.analyzer = CompleteCandleAIAnalyzer()
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
            # Procesar cada tick con información del tiempo restante
            tick_data = self.analyzer.add_tick(price, seconds_remaining)
            return tick_data
        except Exception as e:
            logging.error(f"Error en process_tick: {e}")
            return None
    
    def _comprehensive_ai_analysis(self, analysis):
        """Análisis de IA mejorado con datos completos de la vela"""
        try:
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
            
            # Peso basado en progreso de la vela
            late_phase_weight = 1.0 if candle_progress > 0.8 else 0.7  # Más peso en fase final
            
            # Tendencia (35% de peso)
            trend_weight = 0.35 * late_phase_weight
            if abs(trend_strength) > 1.0:
                if trend_strength > 0:
                    buy_score += 8 * trend_weight
                    reasons.append(f"📈 Tendencia alcista fuerte ({trend_strength:.1f})")
                else:
                    sell_score += 8 * trend_weight
                    reasons.append(f"📉 Tendencia bajista fuerte ({trend_strength:.1f})")
            elif abs(trend_strength) > 0.3:
                if trend_strength > 0:
                    buy_score += 5 * trend_weight
                    reasons.append(f"↗️ Tendencia alcista moderada ({trend_strength:.1f})")
                else:
                    sell_score += 5 * trend_weight
                    reasons.append(f"↘️ Tendencia bajista moderada ({trend_strength:.1f})")
            
            # Momentum (30% de peso)
            momentum_weight = 0.30 * late_phase_weight
            if abs(momentum) > 0.8:
                if momentum > 0:
                    buy_score += 7 * momentum_weight
                    reasons.append(f"🚀 Momentum alcista fuerte ({momentum:.1f}pips)")
                else:
                    sell_score += 7 * momentum_weight
                    reasons.append(f"🔻 Momentum bajista fuerte ({momentum:.1f}pips)")
            elif abs(momentum) > 0.4:
                if momentum > 0:
                    buy_score += 4 * momentum_weight
                    reasons.append(f"↗️ Momentum alcista moderado ({momentum:.1f}pips)")
                else:
                    sell_score += 4 * momentum_weight
                    reasons.append(f"↘️ Momentum bajista moderado ({momentum:.1f}pips)")
            
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
            
            if phase_analysis.get('consistency_score', 0) > 0.7:
                consistent_trend = phase_analysis.get('final_trend', 'N/A')
                if consistent_trend == 'ALCISTA':
                    buy_score += 3 * phase_weight
                    reasons.append("✅ Tendencia consistente alcista")
                elif consistent_trend == 'BAJISTA':
                    sell_score += 3 * phase_weight
                    reasons.append("✅ Tendencia consistente bajista")
            
            # Presión de compra/venta (20% de peso)
            pressure_weight = 0.20 * late_phase_weight
            if pressure_ratio > 2.0:
                buy_score += 6 * pressure_weight
                reasons.append(f"💰 Fuerte presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio > 1.3:
                buy_score += 4 * pressure_weight
                reasons.append(f"📊 Presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.5:
                sell_score += 6 * pressure_weight
                reasons.append(f"💸 Fuerte presión vendedora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.8:
                sell_score += 4 * pressure_weight
                reasons.append(f"📉 Presión vendedora ({pressure_ratio:.1f}x)")
            
            score_difference = buy_score - sell_score
            
            # Umbral dinámico basado en calidad de datos y fase
            base_threshold = 0.4
            if candle_progress > 0.9:  # Últimos segundos
                base_threshold = 0.3  # Más sensible al final
            
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
            
            # Ajustar confianza basada en análisis completo
            confidence = base_confidence
            confidence *= data_quality
            
            # Bonus por alta calidad de datos
            if analysis['tick_count'] > 40:
                confidence = min(90, confidence + 15)
                reasons.append("📊 Alta calidad de datos (muchos ticks)")
            elif analysis['tick_count'] > 25:
                confidence = min(85, confidence + 10)
                reasons.append("📈 Buena calidad de datos")
            
            # Ajustes por condiciones de mercado
            if volatility > 2.0:
                confidence *= 0.8
                reasons.append("🌪️ Alta volatilidad - confianza reducida")
            elif volatility < 0.2:
                confidence *= 0.9
                reasons.append("😴 Baja volatilidad")
            
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
                'phase_analysis': phase_analysis
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
    
    def predict_next_candle(self):
        try:
            analysis = self.analyzer.get_comprehensive_analysis()
            
            if analysis.get('status') != 'SUCCESS':
                return {
                    'direction': 'LATERAL',
                    'confidence': 0,
                    'reason': analysis.get('message', 'Analizando...'),
                    'timestamp': now_iso()
                }
            
            prediction = self._comprehensive_ai_analysis(analysis)
            
            # Solo emitir predicción si tenemos suficiente confianza
            if prediction['confidence'] < 45:
                prediction['direction'] = 'LATERAL'
                prediction['reasons'].append("🔍 Confianza insuficiente para predicción direccional")
            
            prediction.update({
                'tick_count': analysis['tick_count'],
                'current_price': analysis['current_price'],
                'candle_range': analysis.get('candle_range', 0),
                'timestamp': now_iso(),
                'model_version': 'COMPREHENSIVE_AI_V5.3_FULL_CANDLE'
            })
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            # Log de predicción detallado
            if prediction['direction'] != 'LATERAL':
                logging.info(f"🎯 PREDICCIÓN COMPLETA: {prediction['direction']} | "
                           f"Conf: {prediction['confidence']}% | "
                           f"Ticks: {analysis['tick_count']} | "
                           f"Progreso: {analysis.get('candle_progress', 0):.1%}")
            
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
        """Validación mejorada con análisis completo"""
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
            minimal_change = max(0.15, candle_range * 0.2)  # 20% del rango como mínimo significativo
            
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

# -------------- CONEXIÓN PROFESIONAL --------------
class ProfessionalIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        self.tick_listeners = []
        
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
        self.tick_listeners.append(listener)

    def get_realtime_price(self):
        try:
            current_time = time.time()
            
            if not self.connected:
                if self.last_price is None:
                    self.last_price = 1.15389
                else:
                    variation = np.random.uniform(-0.00005, 0.00005)
                    self.last_price += variation
                
                self._notify_tick_listeners(self.last_price, current_time)
                return self.last_price

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
        for listener in self.tick_listeners:
            try:
                listener(price, timestamp)
            except Exception as e:
                logging.error(f"Error notificando listener: {e}")

    def _record_tick(self, price, timestamp):
        self.tick_count += 1
        self.last_price = price
        
        if self.tick_count <= 5 or self.tick_count % 200 == 0:
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL MEJORADO ---------------
iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()

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
    "market_phase": "N/A"
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None
}

def tick_processor(price, timestamp):
    """Procesa cada tick con información del tiempo restante"""
    try:
        global current_prediction
        
        # Calcular segundos restantes en la vela actual
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        # Procesar tick en el predictor
        tick_data = predictor.process_tick(price, seconds_remaining)
        
        if tick_data:
            # Actualizar información global
            current_prediction.update({
                "current_price": price,
                "tick_count": predictor.analyzer.tick_count,
                "timestamp": now_iso(),
                "status": "ACTIVE"
            })
                
    except Exception as e:
        logging.error(f"Error procesando tick: {e}")

def premium_main_loop():
    global current_prediction, performance_stats
    
    logging.info("🚀 DELOWYSS AI V5.3 CON ANÁLISIS COMPLETO INICIADO")
    logging.info("🎯 Analizando ticks desde inicio de vela hasta últimos 5 segundos")
    
    iq_connector.connect()
    iq_connector.add_tick_listener(tick_processor)
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    prediction_made_this_candle = False
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # Obtener precio (dispara listeners de ticks)
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                last_price = price
            
            # ACTUALIZAR PROGRESO DE VELA
            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress
            
            # LÓGICA DE PREDICCIÓN EN ÚLTIMOS 5 SEGUNDOS
            if (seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - last_prediction_time) >= 2 and
                not prediction_made_this_candle):
                
                logging.info(f"🎯 VENTANA DE PREDICCIÓN: {seconds_remaining:.1f}s restantes | "
                           f"Ticks analizados: {predictor.analyzer.tick_count}")
                
                prediction = predictor.predict_next_candle()
                
                if prediction['confidence'] >= 45:
                    current_prediction.update(prediction)
                    last_prediction_time = time.time()
                    prediction_made_this_candle = True
                    
                    if prediction['direction'] != 'LATERAL':
                        logging.info(f"🎯 PREDICCIÓN EMITIDA: {prediction['direction']} | "
                                   f"Confianza: {prediction['confidence']}% | "
                                   f"Base de ticks: {predictor.analyzer.tick_count}")
            
            # DETECTAR NUEVA VELA
            if current_candle_start > last_candle_start:
                if last_price is not None:
                    validation_result = predictor.validate_prediction(last_price)
                    if validation_result:
                        performance_stats.update({
                            'total_predictions': validation_result['total_predictions'],
                            'correct_predictions': validation_result['correct_predictions'],
                            'recent_accuracy': validation_result['accuracy'],
                            'last_validation': validation_result
                        })
                
                # Reiniciar para nueva vela
                predictor.reset()
                last_candle_start = current_candle_start
                prediction_made_this_candle = False
                
                logging.info("🕯️ NUEVA VELA INICIADA - Comenzando análisis tick-by-tick completo")
            
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(0.5)

# ... (EL RESTO DEL CÓDIGO DE FASTAPI SE MANTIENE IGUAL)
# Los endpoints /api/prediction, /api/validation, etc. permanecen iguales

app = FastAPI(title="Delowyss AI Premium V5.3", version="5.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    # ... (implementación de interfaz HTML igual que antes)
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

@app.get("/api/health")
def api_health():
    return JSONResponse({
        "status": "healthy", 
        "timestamp": now_iso(),
        "version": "5.3.0",
        "features": ["full_candle_analysis", "phase_analysis", "tick_by_tick"]
    })

def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info("⭐ SISTEMA DE ANÁLISIS COMPLETO INICIADO")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

start_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
