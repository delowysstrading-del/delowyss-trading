# main.py - DELOWYSS AI V4.6-PRO PROFESSIONAL TICK ANALYZER
"""
Delowyss Trading AI — V4.6-PRO PROFESSIONAL
Sistema profesional de análisis tick-by-tick con aprendizaje continuo avanzado
CEO: Eduardo Solis — © 2025
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from iqoptionapi.stable_api import IQ_Option
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN PROFESIONAL ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = int(os.getenv("TIMEFRAME", "60"))
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "5"))

MODEL_VERSION = os.getenv("MODEL_VERSION", "v46_pro_tick_analyzer")
TRAINING_CSV = os.getenv("TRAINING_CSV", f"training_data_{MODEL_VERSION}.csv")
PERF_CSV = os.getenv("PERF_CSV", f"performance_{MODEL_VERSION}.csv")

# PARÁMETROS AVANZADOS DE ANÁLISIS
MIN_TICKS_FOR_ANALYSIS = int(os.getenv("MIN_TICKS_FOR_ANALYSIS", "15"))
IDEAL_TICKS_FOR_PREDICTION = int(os.getenv("IDEAL_TICKS_FOR_PREDICTION", "40"))
TICK_ANALYSIS_WINDOW = int(os.getenv("TICK_ANALYSIS_WINDOW", "30"))

# APRENDIZAJE CONTINUO
CONTINUOUS_LEARNING = os.getenv("CONTINUOUS_LEARNING", "true").lower() == "true"
LEARNING_UPDATE_INTERVAL = int(os.getenv("LEARNING_UPDATE_INTERVAL", "30"))

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] 🎯 %(message)s',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# ------------------ ANALIZADOR PROFESIONAL DE TICKS ------------------
class ProfessionalTickAnalyzer:
    def __init__(self):
        self.current_candle_ticks = deque(maxlen=300)
        self.tick_sequences = deque(maxlen=TICK_ANALYSIS_WINDOW)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.last_analysis_time = 0
        self.analysis_interval = 1.0
        
        # MÉTRICAS AVANZADAS EN TIEMPO REAL
        self.real_time_metrics = {
            'price_momentum': 0.0,
            'tick_intensity': 0.0,
            'volatility_trend': 0.0,
            'buy_sell_imbalance': 0.0,
            'price_acceleration': 0.0,
            'trend_consistency': 0.0,
            'pressure_strength': 0.0,
            'market_noise': 0.0,
            'liquidity_flow': 0.0,
            'order_imbalance': 0.0,
            'price_efficiency': 0.0,
            'momentum_quality': 0.0,
            'trend_persistence': 0.0
        }
        
        # HISTORIAL PARA ANÁLISIS TEMPORAL
        self.momentum_history = deque(maxlen=20)
        self.volatility_history = deque(maxlen=25)
        self.imbalance_history = deque(maxlen=15)
        self.price_changes = deque(maxlen=50)
        
    def add_tick(self, price: float, timestamp: float, volume: float = 1.0):
        """Procesamiento profesional de cada tick"""
        price = float(price)
        
        # VALIDACIÓN AVANZADA DE TICKS
        if not self._validate_tick(price, timestamp):
            return None
        
        # INICIALIZACIÓN DE VELA
        if self.current_candle_open is None:
            self.current_candle_open = self.current_candle_high = self.current_candle_low = price
            logging.info(f"🕯️ Vela profesional iniciada - Apertura: {price:.5f}")
        
        # ACTUALIZACIÓN HIGH/LOW
        self.current_candle_high = max(self.current_candle_high, price)
        self.current_candle_low = min(self.current_candle_low, price)
        self.current_candle_close = price
        
        # ALMACENAMIENTO DE DATOS DEL TICK
        tick_data = {
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'time_since_open': timestamp - (timestamp // TIMEFRAME * TIMEFRAME),
            'price_change': 0.0
        }
        
        # CALCULAR CAMBIO DE PRECIO
        if len(self.current_candle_ticks) > 0:
            last_tick = self.current_candle_ticks[-1]
            tick_data['price_change'] = (price - last_tick['price']) * 10000
            self.price_changes.append(tick_data['price_change'])
        
        self.current_candle_ticks.append(tick_data)
        self.tick_sequences.append(price)
        self.tick_count += 1
        
        # ANÁLISIS EN TIEMPO REAL MEJORADO
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self._comprehensive_real_time_analysis()
            self.last_analysis_time = current_time
            
        return tick_data
    
    def _validate_tick(self, price: float, timestamp: float) -> bool:
        """Validación avanzada de ticks"""
        if price <= 0:
            return False
            
        # Detección de outliers estadísticos
        if len(self.current_candle_ticks) >= 10:
            recent_prices = [t['price'] for t in list(self.current_candle_ticks)[-10:]]
            price_std = np.std(recent_prices)
            price_mean = np.mean(recent_prices)
            
            if abs(price - price_mean) > 3 * price_std:
                logging.warning(f"🚨 Outlier detectado: {price:.5f} (mean: {price_mean:.5f})")
                return False
        
        # Validación de timestamp
        if len(self.current_candle_ticks) > 0:
            last_timestamp = self.current_candle_ticks[-1]['timestamp']
            if timestamp < last_timestamp:
                logging.warning("⚠️ Timestamp inconsistente")
                return False
                
        return True
    
    def _comprehensive_real_time_analysis(self):
        """Análisis en tiempo real profesional"""
        if len(self.current_candle_ticks) < 8:
            return
            
        try:
            recent_prices = [t['price'] for t in list(self.current_candle_ticks)[-25:]]
            recent_changes = [t['price_change'] for t in list(self.current_candle_ticks)[-20:] if 'price_change' in t]
            
            if len(recent_prices) >= 8:
                # 1. ANÁLISIS DE MOMENTUM AVANZADO
                self.real_time_metrics['price_momentum'] = self._calculate_advanced_momentum(recent_prices)
                self.momentum_history.append(self.real_time_metrics['price_momentum'])
                
                # 2. INTENSIDAD DE TICKS MEJORADA
                self.real_time_metrics['tick_intensity'] = self._calculate_professional_tick_intensity()
                
                # 3. ANÁLISIS DE VOLATILIDAD PROFESIONAL
                self.real_time_metrics['volatility_trend'] = self._calculate_advanced_volatility(recent_prices)
                self.volatility_history.append(self.real_time_metrics['volatility_trend'])
                
                # 4. DESEQUILIBRIO COMPRA/VENTA AVANZADO
                imbalance = self._calculate_professional_imbalance(recent_prices, recent_changes)
                self.real_time_metrics['buy_sell_imbalance'] = imbalance
                self.imbalance_history.append(imbalance)
                
                # 5. ACELERACIÓN DE PRECIO MEJORADA
                self.real_time_metrics['price_acceleration'] = self._calculate_advanced_acceleration(recent_prices)
                
                # 6. NUEVAS MÉTRICAS PROFESIONALES
                self.real_time_metrics['market_noise'] = self._calculate_market_noise(recent_prices)
                self.real_time_metrics['liquidity_flow'] = self._calculate_liquidity_flow()
                self.real_time_metrics['order_imbalance'] = self._calculate_order_imbalance(recent_changes)
                self.real_time_metrics['price_efficiency'] = self._calculate_price_efficiency(recent_prices)
                self.real_time_metrics['momentum_quality'] = self._calculate_momentum_quality()
                self.real_time_metrics['trend_persistence'] = self._calculate_trend_persistence()
                
        except Exception as e:
            logging.debug(f"Error en análisis profesional: {e}")
    
    def _calculate_advanced_momentum(self, prices):
        """Cálculo de momentum profesional multi-timeframe"""
        if len(prices) < 15:
            return 0.0
            
        momentums = []
        weights = []
        
        # Momentum ultra-corto (últimos 3 ticks)
        if len(prices) >= 3:
            ultra_short = (prices[-1] - prices[-3]) / prices[-3] * 10000
            momentums.append(ultra_short)
            weights.append(0.25)
        
        # Momentum corto (últimos 8 ticks)
        if len(prices) >= 8:
            short_term = (prices[-1] - prices[-8]) / prices[-8] * 10000
            momentums.append(short_term)
            weights.append(0.35)
            
        # Momentum medio (últimos 15 ticks)
        if len(prices) >= 15:
            mid_term = (prices[-1] - prices[-15]) / prices[-15] * 10000
            momentums.append(mid_term)
            weights.append(0.40)
        
        # Aplicar promedio ponderado
        if len(momentums) >= 2:
            smoothed_momentum = np.average(momentums, weights=weights[:len(momentums)])
            return smoothed_momentum
            
        return momentums[0] if momentums else 0.0
    
    def _calculate_professional_tick_intensity(self):
        """Cálculo profesional de intensidad de ticks"""
        if len(self.current_candle_ticks) < 5:
            return 0.0
            
        recent_ticks = list(self.current_candle_ticks)[-20:]
        if len(recent_ticks) < 3:
            return 0.0
            
        # Calcular intensidad basada en frecuencia y volatilidad
        time_span = recent_ticks[-1]['timestamp'] - recent_ticks[0]['timestamp']
        if time_span > 0:
            base_intensity = len(recent_ticks) / time_span
            
            # Ajustar por volatilidad de los ticks
            price_changes = [abs(tick['price_change']) for tick in recent_ticks if 'price_change' in tick]
            if price_changes:
                volatility_factor = np.mean(price_changes) / 0.5
                adjusted_intensity = base_intensity * (1 + min(2.0, volatility_factor))
                return min(20.0, adjusted_intensity)
                
        return 0.0
    
    def _calculate_advanced_volatility(self, prices):
        """Análisis profesional de volatilidad"""
        if len(prices) < 12:
            return 0.0
            
        volatilities = []
        
        # Volatilidad de diferentes ventanas temporales
        windows = [5, 8, 12]
        for window in windows:
            if len(prices) >= window:
                window_prices = prices[-window:]
                vol = (max(window_prices) - min(window_prices)) * 10000
                volatilities.append(vol)
        
        if volatilities:
            # Tendencia de volatilidad
            recent_vol = volatilities[-1] if len(volatilities) > 0 else 0
            if len(volatilities) > 1:
                vol_trend = (recent_vol - np.mean(volatilities[:-1])) / (np.mean(volatilities[:-1]) + 0.1)
                return vol_trend
        
        return 0.0
    
    def _calculate_professional_imbalance(self, prices, changes):
        """Desequilibrio profesional compra/venta"""
        if len(prices) < 6 or len(changes) < 5:
            return 0.0
            
        # Análisis multi-nivel del desequilibrio
        imbalances = []
        
        # 1. Dirección simple de ticks
        directions = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                directions.append(1)
            elif prices[i] < prices[i-1]:
                directions.append(-1)
            else:
                directions.append(0)
        
        if directions:
            simple_imbalance = np.mean(directions)
            imbalances.append(simple_imbalance)
        
        # 2. Fuerza de los movimientos
        if changes:
            strength_imbalance = np.mean([c / (abs(c) + 0.1) for c in changes])
            imbalances.append(strength_imbalance * 0.8)
        
        # 3. Persistencia de dirección
        if len(directions) >= 8:
            recent_directions = directions[-8:]
            persistence = sum(1 for i in range(1, len(recent_directions)) 
                            if recent_directions[i] == recent_directions[i-1] != 0)
            persistence_ratio = persistence / len(recent_directions)
            imbalances.append(persistence_ratio * np.sign(simple_imbalance))
        
        return np.mean(imbalances) if imbalances else 0.0
    
    def _calculate_advanced_acceleration(self, prices):
        """Aceleración de precio profesional"""
        if len(prices) < 10:
            return 0.0
            
        # Calcular velocidades en múltiples timeframe
        velocities = []
        
        # Velocidad ultra-rápida (2 ticks)
        if len(prices) >= 3:
            v_fast = (prices[-1] - prices[-3]) / 2
            velocities.append(v_fast)
        
        # Velocidad rápida (5 ticks)
        if len(prices) >= 6:
            v_medium = (prices[-1] - prices[-6]) / 5
            velocities.append(v_medium)
            
        # Velocidad normal (8 ticks)
        if len(prices) >= 9:
            v_slow = (prices[-1] - prices[-9]) / 8
            velocities.append(v_slow)
        
        if len(velocities) >= 2:
            # Aceleración como cambio en velocidad
            acceleration = (velocities[0] - velocities[-1]) * 10000
            return acceleration
            
        return 0.0
    
    def _calculate_market_noise(self, prices):
        """Calcular ruido de mercado"""
        if len(prices) < 10:
            return 0.0
            
        # Ruido como ratio de movimientos laterales vs direccionales
        directional_moves = 0
        total_moves = len(prices) - 1
        
        for i in range(1, len(prices)):
            change_pct = abs(prices[i] - prices[i-1]) / prices[i-1] * 10000
            if change_pct > 0.1:
                directional_moves += 1
        
        noise_ratio = 1 - (directional_moves / total_moves) if total_moves > 0 else 0.5
        return noise_ratio
    
    def _calculate_liquidity_flow(self):
        """Calcular flujo de liquidez"""
        if len(self.current_candle_ticks) < 10:
            return 0.0
            
        # Basado en consistencia de ticks
        recent_ticks = list(self.current_candle_ticks)[-15:]
        time_gaps = [recent_ticks[i]['timestamp'] - recent_ticks[i-1]['timestamp'] 
                    for i in range(1, len(recent_ticks))]
        
        if time_gaps:
            gap_consistency = 1.0 / (np.std(time_gaps) + 0.1)
            return min(1.0, gap_consistency * 0.3)
            
        return 0.0
    
    def _calculate_order_imbalance(self, changes):
        """Desequilibrio de órdenes profesional"""
        if not changes:
            return 0.0
            
        # Analizar distribución de cambios de precio
        positive_changes = [c for c in changes if c > 0.05]
        negative_changes = [c for c in changes if c < -0.05]
        
        total_significant = len(positive_changes) + len(negative_changes)
        if total_significant > 0:
            imbalance = (len(positive_changes) - len(negative_changes)) / total_significant
            return imbalance
            
        return 0.0
    
    def _calculate_price_efficiency(self, prices):
        """Eficiencia del precio (ratio de tendencia vs ruido)"""
        if len(prices) < 15:
            return 0.0
            
        # Calcular qué porcentaje del movimiento es direccional
        total_movement = abs(prices[-1] - prices[0]) * 10000
        oscillating_movement = sum(abs(prices[i] - prices[i-1]) * 10000 
                                 for i in range(1, len(prices)))
        
        if oscillating_movement > 0:
            efficiency = total_movement / oscillating_movement
            return min(1.0, efficiency)
            
        return 0.0
    
    def _calculate_momentum_quality(self):
        """Calidad del momentum (consistencia y fuerza)"""
        if len(self.momentum_history) < 5:
            return 0.0
            
        recent_momentum = list(self.momentum_history)[-5:]
        
        # Consistencia de dirección
        positive_momentum = sum(1 for m in recent_momentum if m > 0.1)
        negative_momentum = sum(1 for m in recent_momentum if m < -0.1)
        consistency = max(positive_momentum, negative_momentum) / len(recent_momentum)
        
        # Fuerza del momentum
        strength = min(1.0, np.mean([abs(m) for m in recent_momentum]) / 3.0)
        
        return (consistency + strength) / 2
    
    def _calculate_trend_persistence(self):
        """Persistencia de la tendencia actual"""
        if len(self.momentum_history) < 8:
            return 0.0
            
        # Verificar cuánto tiempo se mantiene la tendencia
        recent_momentum = list(self.momentum_history)[-8:]
        same_direction = sum(1 for i in range(1, len(recent_momentum)) 
                           if recent_momentum[i] * recent_momentum[i-1] > 0)
        
        persistence = same_direction / (len(recent_momentum) - 1)
        return persistence
    
    def get_professional_analysis(self):
        """Análisis profesional completo de ticks"""
        if self.tick_count < MIN_TICKS_FOR_ANALYSIS:
            return {
                'status': 'INSUFFICIENT_DATA',
                'tick_count': self.tick_count,
                'message': f'Mínimo {MIN_TICKS_FOR_ANALYSIS} ticks requeridos',
                'data_quality': 0.0,
                'analysis_quality': 0.0
            }
        
        try:
            prices = [tick['price'] for tick in self.current_candle_ticks]
            
            # CALIDAD DE DATOS PROFESIONAL
            data_quality = self._calculate_professional_data_quality()
            analysis_quality = self._calculate_analysis_quality()
            
            # ANÁLISIS COMPLETO
            analysis = {
                'tick_count': self.tick_count,
                'current_price': self.current_candle_close,
                'open_price': self.current_candle_open,
                'high_price': self.current_candle_high,
                'low_price': self.current_candle_low,
                'price_range': (self.current_candle_high - self.current_candle_low) * 10000,
                'current_change': (self.current_candle_close - self.current_candle_open) * 10000,
                
                # MÉTRICAS AVANZADAS
                **self.real_time_metrics,
                
                # ANÁLISIS TÉCNICO PROFESIONAL
                'detected_patterns': self._detect_professional_patterns(prices),
                'trend_strength': self._calculate_professional_trend_strength(prices),
                'support_resistance': self._find_advanced_support_resistance(prices),
                'market_phase': self._determine_professional_market_phase(),
                'volume_analysis': self._analyze_volume_profile(),
                'risk_metrics': self._calculate_risk_metrics(),
                
                # CALIDAD Y CONFIANZA
                'data_quality': data_quality,
                'analysis_quality': analysis_quality,
                'prediction_confidence': self._calculate_prediction_confidence(),
                
                'timestamp': time.time()
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"❌ Error en análisis profesional: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _calculate_professional_data_quality(self):
        """Calcular calidad profesional de los datos"""
        quality_factors = []
        
        # Factor 1: Cantidad y frecuencia de ticks
        tick_density = min(1.0, self.tick_count / 50.0)
        quality_factors.append(tick_density * 0.25)
        
        # Factor 2: Consistencia temporal
        if len(self.current_candle_ticks) > 10:
            timestamps = [t['timestamp'] for t in self.current_candle_ticks]
            time_gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            gap_std = np.std(time_gaps) if time_gaps else 1.0
            consistency = 1.0 / (gap_std + 0.1)
            quality_factors.append(min(1.0, consistency) * 0.20)
        
        # Factor 3: Calidad de precio
        if len(self.current_candle_ticks) > 15:
            prices = [t['price'] for t in self.current_candle_ticks]
            price_volatility = (max(prices) - min(prices)) * 10000
            volatility_factor = 1.0 - min(1.0, abs(price_volatility - 1.5) / 3.0)
            quality_factors.append(volatility_factor * 0.20)
        
        # Factor 4: Complejidad del análisis
        analysis_complexity = min(1.0, len(self.momentum_history) / 10.0)
        quality_factors.append(analysis_complexity * 0.35)
        
        return sum(quality_factors)
    
    def _calculate_analysis_quality(self):
        """Calcular calidad del análisis realizado"""
        if self.tick_count < MIN_TICKS_FOR_ANALYSIS:
            return 0.0
            
        quality_metrics = []
        
        # Métrica 1: Estabilidad de métricas
        if len(self.momentum_history) >= 5:
            momentum_stability = 1.0 - (np.std(list(self.momentum_history)) / 2.0)
            quality_metrics.append(max(0.0, momentum_stability))
        
        # Métrica 2: Consistencia de señales
        if len(self.imbalance_history) >= 5:
            imbalance_consistency = 1.0 - (np.std(list(self.imbalance_history)) / 0.5)
            quality_metrics.append(max(0.0, imbalance_consistency))
        
        # Métrica 3: Profundidad del análisis
        depth_factor = min(1.0, self.tick_count / IDEAL_TICKS_FOR_PREDICTION)
        quality_metrics.append(depth_factor)
        
        return np.mean(quality_metrics) if quality_metrics else 0.5
    
    def _calculate_prediction_confidence(self):
        """Calcular confianza base para predicción"""
        base_confidence = 0.0
        
        # Factor 1: Calidad de datos
        base_confidence += self._calculate_professional_data_quality() * 40
        
        # Factor 2: Fuerza de señales
        signal_strength = (abs(self.real_time_metrics['price_momentum']) / 3.0 + 
                          abs(self.real_time_metrics['buy_sell_imbalance']) / 0.6)
        base_confidence += min(30, signal_strength * 15)
        
        # Factor 3: Calidad de análisis
        base_confidence += self._calculate_analysis_quality() * 30
        
        return min(85, base_confidence)
    
    def _detect_professional_patterns(self, prices):
        """Detección profesional de patrones"""
        patterns = []
        
        if len(prices) < 12:
            return patterns
        
        # 1. PATRÓN: ACUMULACIÓN PROFESIONAL
        accumulation_score = self._detect_accumulation_pattern(prices)
        if accumulation_score > 0.7:
            patterns.append('strong_accumulation')
        elif accumulation_score > 0.5:
            patterns.append('accumulation')
        
        # 2. PATRÓN: BREAKOUT AVANZADO
        breakout_score = self._detect_breakout_pattern(prices)
        if breakout_score > 0.8:
            patterns.append('very_strong_breakout')
        elif breakout_score > 0.6:
            patterns.append('strong_breakout')
        elif breakout_score > 0.4:
            patterns.append('breakout')
        
        # 3. PATRÓN: TENDENCIA PROFESIONAL
        trend_analysis = self._analyze_trend_quality(prices)
        if trend_analysis['quality'] > 0.8:
            patterns.append(f'excellent_trend_{trend_analysis["direction"]}')
        elif trend_analysis['quality'] > 0.6:
            patterns.append(f'good_trend_{trend_analysis["direction"]}')
        elif trend_analysis['quality'] > 0.4:
            patterns.append(f'weak_trend_{trend_analysis["direction"]}')
        
        # 4. PATRÓN: REVERSIÓN AVANZADA
        if self._detect_professional_reversal(prices):
            patterns.append('professional_reversal_signal')
        
        # 5. PATRÓN: CONSISTENCIA DE MERCADO
        if self.real_time_metrics['market_noise'] < 0.3:
            patterns.append('low_noise_environment')
        elif self.real_time_metrics['market_noise'] > 0.7:
            patterns.append('high_noise_environment')
            
        return patterns
    
    def _detect_accumulation_pattern(self, prices):
        """Detectar patrón de acumulación profesional"""
        if len(prices) < 15:
            return 0.0
            
        # Rango de precio estrecho
        recent_range = max(prices[-10:]) - min(prices[-10:])
        range_score = 1.0 - min(1.0, recent_range * 10000 / 0.8)
        
        # Volumen de ticks (intensidad)
        intensity_score = min(1.0, self.real_time_metrics['tick_intensity'] / 8.0)
        
        # Baja volatilidad
        volatility_score = 1.0 - min(1.0, abs(self.real_time_metrics['volatility_trend']))
        
        return (range_score + intensity_score + volatility_score) / 3.0
    
    def _detect_breakout_pattern(self, prices):
        """Detectar patrón de breakout profesional"""
        if len(prices) < 20:
            return 0.0
            
        # Comparar rango reciente vs rango anterior
        current_range = max(prices[-5:]) - min(prices[-5:])
        previous_range = max(prices[-15:-5]) - min(prices[-15:-5])
        
        if previous_range > 0:
            range_ratio = current_range / previous_range
            range_score = min(1.0, (range_ratio - 1.0) / 2.0)
        else:
            range_score = 0.0
        
        # Momentum acompañante
        momentum_score = min(1.0, abs(self.real_time_metrics['price_momentum']) / 4.0)
        
        # Fuerza de presión
        pressure_score = min(1.0, abs(self.real_time_metrics['pressure_strength']) / 0.5)
        
        return (range_score + momentum_score + pressure_score) / 3.0
    
    def _analyze_trend_quality(self, prices):
        """Análisis profesional de calidad de tendencia"""
        if len(prices) < 15:
            return {'direction': 'neutral', 'quality': 0.0}
        
        # Dirección de la tendencia
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        direction = "up" if slope > 0 else "down"
        
        # Calidad de la tendencia (R-cuadrado)
        y_pred = np.polyval([slope, prices[0]], x)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Consistencia del momentum
        momentum_consistency = self.real_time_metrics['trend_consistency']
        
        quality = (r_squared + abs(momentum_consistency)) / 2
        
        return {'direction': direction, 'quality': quality}
    
    def _detect_professional_reversal(self, prices):
        """Detección profesional de reversión"""
        if len(prices) < 20:
            return False
            
        signals = 0
        
        # Señal 1: Divergencia de momentum
        if len(self.momentum_history) >= 5:
            recent_momentum = list(self.momentum_history)[-5:]
            price_trend = (prices[-1] - prices[-5]) / prices[-5] * 10000
            
            if (price_trend > 0 and np.mean(recent_momentum) < -0.5) or \
               (price_trend < 0 and np.mean(recent_momentum) > 0.5):
                signals += 1
        
        # Señal 2: Patrón de precio de reversión
        if self._detect_price_reversal_pattern(prices):
            signals += 1
            
        # Señal 3: Cambio en la microestructura
        if (self.real_time_metrics['order_imbalance'] * self.real_time_metrics['price_momentum'] < -0.2):
            signals += 1
            
        return signals >= 2
    
    def _detect_price_reversal_pattern(self, prices):
        """Detectar patrones de precio de reversión"""
        if len(prices) < 15:
            return False
            
        # Doble techo/suelo en ticks
        recent_high = max(prices[-8:])
        recent_low = min(prices[-8:])
        current = prices[-1]
        
        # Verificar si el precio está en extremos con pérdida de momentum
        near_high = abs(current - recent_high) / recent_high < 0.0002
        near_low = abs(current - recent_low) / recent_low < 0.0002
        
        if (near_high and self.real_time_metrics['price_momentum'] < -0.8) or \
           (near_low and self.real_time_metrics['price_momentum'] > 0.8):
            return True
            
        return False
    
    def _calculate_professional_trend_strength(self, prices):
        """Calcular fuerza de tendencia profesional"""
        if len(prices) < 15:
            return 0.0
            
        strengths = []
        
        # 1. Análisis de regresión
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        regression_strength = (slope * len(prices)) / np.mean(prices) * 10000
        strengths.append(regression_strength)
        
        # 2. Análisis de diferencias
        first_third = prices[:len(prices)//3]
        last_third = prices[-(len(prices)//3):]
        if first_third and last_third:
            diff_strength = (np.mean(last_third) - np.mean(first_third)) / np.mean(first_third) * 10000
            strengths.append(diff_strength)
        
        # 3. Análisis de consistencia direccional
        directions = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                directions.append(1)
            elif prices[i] < prices[i-1]:
                directions.append(-1)
        
        if directions:
            directional_strength = np.mean(directions) * 3.0
            strengths.append(directional_strength)
        
        # 4. Incorporar métricas de tiempo real
        strengths.append(self.real_time_metrics['price_momentum'])
        strengths.append(self.real_time_metrics['trend_consistency'] * 2.0)
        
        return np.mean(strengths)
    
    def _find_advanced_support_resistance(self, prices):
        """Encontrar soportes y resistencias avanzados"""
        if len(prices) < 20:
            return {'support': None, 'resistance': None, 'quality': 0.0}
        
        # Usar método de pivotes para identificar niveles
        support_levels = []
        resistance_levels = []
        
        # Buscar mínimos locales (soportes)
        for i in range(4, len(prices)-4):
            if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and 
                prices[i] < prices[i+1] and prices[i] < prices[i+2] and
                prices[i] < prices[i-3] and prices[i] < prices[i+3] and
                prices[i] < prices[i-4] and prices[i] < prices[i+4]):
                support_levels.append(prices[i])
        
        # Buscar máximos locales (resistencias)
        for i in range(4, len(prices)-4):
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and 
                prices[i] > prices[i+1] and prices[i] > prices[i+2] and
                prices[i] > prices[i-3] and prices[i] > prices[i+3] and
                prices[i] > prices[i-4] and prices[i] > prices[i+4]):
                resistance_levels.append(prices[i])
        
        current_price = prices[-1]
        
        # Encontrar niveles más cercanos
        closest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else None
        closest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else None
        
        # Calcular calidad de niveles
        quality = min(1.0, (len(support_levels) + len(resistance_levels)) / 15.0)
        
        return {
            'support': round(closest_support, 5) if closest_support else None,
            'resistance': round(closest_resistance, 5) if closest_resistance else None,
            'distance_to_support': round((current_price - closest_support) * 10000, 2) if closest_support else None,
            'distance_to_resistance': round((closest_resistance - current_price) * 10000, 2) if closest_resistance else None,
            'quality': quality,
            'support_count': len(support_levels),
            'resistance_count': len(resistance_levels)
        }
    
    def _determine_professional_market_phase(self):
        """Determinar fase del mercado profesional"""
        momentum = abs(self.real_time_metrics['price_momentum'])
        volatility = abs(self.real_time_metrics['volatility_trend'])
        imbalance = abs(self.real_time_metrics['buy_sell_imbalance'])
        consistency = abs(self.real_time_metrics['trend_consistency'])
        pressure = abs(self.real_time_metrics['pressure_strength'])
        noise = self.real_time_metrics['market_noise']
        efficiency = self.real_time_metrics['price_efficiency']
        
        # ANÁLISIS PROFESIONAL DE FASE
        if momentum > 3.0 and consistency > 0.8 and efficiency > 0.7:
            return "very_strong_trend"
        elif momentum > 2.0 and consistency > 0.7 and pressure > 0.4:
            return "strong_trend"
        elif momentum > 1.2 and imbalance > 0.4:
            return "moderate_trend"
        elif noise < 0.2 and momentum < 0.5:
            return "tight_consolidation"
        elif noise < 0.4 and momentum < 0.8:
            return "consolidation"
        elif volatility > 0.5 and noise > 0.6:
            return "high_volatility"
        elif pressure > 0.5 and efficiency < 0.4:
            return "building_pressure"
        elif efficiency > 0.8 and consistency > 0.6:
            return "efficient_movement"
        else:
            return "neutral"
    
    def _analyze_volume_profile(self):
        """Análisis de perfil de volumen (basado en ticks)"""
        if len(self.current_candle_ticks) < 10:
            return {'volume_trend': 0.0, 'volume_consistency': 0.0}
        
        # Análisis de intensidad de ticks como proxy de volumen
        recent_intensity = self.real_time_metrics['tick_intensity']
        base_intensity = 5.0
        
        volume_trend = min(2.0, recent_intensity / base_intensity) - 1.0
        
        # Consistencia del volumen
        if len(self.current_candle_ticks) > 20:
            intensities = []
            for i in range(0, len(self.current_candle_ticks)-5, 5):
                window_ticks = list(self.current_candle_ticks)[i:i+5]
                if len(window_ticks) >= 3:
                    time_span = window_ticks[-1]['timestamp'] - window_ticks[0]['timestamp']
                    if time_span > 0:
                        intensity = len(window_ticks) / time_span
                        intensities.append(intensity)
            
            if intensities:
                volume_consistency = 1.0 - (np.std(intensities) / (np.mean(intensities) + 0.1))
            else:
                volume_consistency = 0.5
        else:
            volume_consistency = 0.5
        
        return {
            'volume_trend': volume_trend,
            'volume_consistency': volume_consistency
        }
    
    def _calculate_risk_metrics(self):
        """Calcular métricas de riesgo profesionales"""
        risk_metrics = {
            'volatility_risk': 0.0,
            'liquidity_risk': 0.0,
            'noise_risk': 0.0,
            'overall_risk': 0.0
        }
        
        # Riesgo por volatilidad
        volatility = abs(self.real_time_metrics['volatility_trend'])
        risk_metrics['volatility_risk'] = min(1.0, volatility / 0.8)
        
        # Riesgo por liquidez
        liquidity = self.real_time_metrics['liquidity_flow']
        risk_metrics['liquidity_risk'] = 1.0 - liquidity
        
        # Riesgo por ruido
        noise = self.real_time_metrics['market_noise']
        risk_metrics['noise_risk'] = noise
        
        # Riesgo general (promedio ponderado)
        risk_metrics['overall_risk'] = (
            risk_metrics['volatility_risk'] * 0.4 +
            risk_metrics['liquidity_risk'] * 0.3 +
            risk_metrics['noise_risk'] * 0.3
        )
        
        return risk_metrics
    
    def reset_analysis(self):
        """Reiniciar análisis profesional"""
        self.current_candle_ticks.clear()
        self.tick_sequences.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        
        # Limpiar historiales
        self.momentum_history.clear()
        self.volatility_history.clear()
        self.imbalance_history.clear()
        self.price_changes.clear()
        
        # Resetear métricas
        for key in self.real_time_metrics:
            self.real_time_metrics[key] = 0.0
            
        logging.info("🔄 Análisis profesional reiniciado para nueva vela")

# ------------------ SISTEMA DE APRENDIZAJE CONTINUO ------------------
class ContinuousLearningSystem:
    def __init__(self):
        self.learning_enabled = CONTINUOUS_LEARNING
        self.performance_history = deque(maxlen=100)
        self.feature_importance = defaultdict(float)
        self.pattern_effectiveness = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.market_condition_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.last_learning_update = time.time()
        
    def record_prediction_result(self, prediction, actual_direction, analysis):
        """Registrar resultado para aprendizaje"""
        if not self.learning_enabled:
            return
            
        correct = prediction['direction'] == actual_direction
        result = {
            'timestamp': now_iso(),
            'prediction': prediction,
            'actual': actual_direction,
            'correct': correct,
            'analysis': analysis,
            'confidence': prediction.get('confidence', 0)
        }
        
        self.performance_history.append(result)
        
        # Aprendizaje inmediato
        self._immediate_learning(result)
        
        # Aprendizaje periódico
        if time.time() - self.last_learning_update > LEARNING_UPDATE_INTERVAL:
            self._periodic_learning_update()
            self.last_learning_update = time.time()
    
    def _immediate_learning(self, result):
        """Aprendizaje inmediato después de cada predicción"""
        analysis = result['analysis']
        
        # Aprender efectividad de patrones
        patterns = analysis.get('detected_patterns', [])
        for pattern in patterns:
            self.pattern_effectiveness[pattern]['total'] += 1
            if result['correct']:
                self.pattern_effectiveness[pattern]['correct'] += 1
        
        # Aprender performance por condición de mercado
        market_phase = analysis.get('market_phase', 'unknown')
        self.market_condition_performance[market_phase]['total'] += 1
        if result['correct']:
            self.market_condition_performance[market_phase]['correct'] += 1
        
        # Aprender importancia de características
        if result['correct']:
            self._update_feature_importance(analysis, positive=True)
        else:
            self._update_feature_importance(analysis, positive=False)
    
    def _update_feature_importance(self, analysis, positive=True):
        """Actualizar importancia de características"""
        features = [
            'price_momentum', 'buy_sell_imbalance', 'volatility_trend',
            'price_acceleration', 'trend_consistency', 'pressure_strength'
        ]
        
        for feature in features:
            value = abs(analysis.get(feature, 0))
            if positive:
                self.feature_importance[feature] += value * 0.1
            else:
                self.feature_importance[feature] -= value * 0.05
            
            # Limitar rango
            self.feature_importance[feature] = max(0.0, min(1.0, self.feature_importance[feature]))
    
    def _periodic_learning_update(self):
        """Actualización periódica del sistema de aprendizaje"""
        if len(self.performance_history) < 10:
            return
            
        logging.info("🧠 Actualizando sistema de aprendizaje...")
        
        # Calcular accuracy reciente
        recent_results = list(self.performance_history)[-20:]
        if recent_results:
            recent_accuracy = sum(1 for r in recent_results if r['correct']) / len(recent_results)
            logging.info(f"📊 Accuracy reciente: {recent_accuracy:.1%}")
        
        # Log de patrones más efectivos
        effective_patterns = []
        for pattern, stats in self.pattern_effectiveness.items():
            if stats['total'] >= 5:
                accuracy = stats['correct'] / stats['total']
                if accuracy > 0.6:
                    effective_patterns.append((pattern, accuracy))
        
        if effective_patterns:
            effective_patterns.sort(key=lambda x: x[1], reverse=True)
            logging.info(f"🎯 Patrones más efectivos: {effective_patterns[:3]}")
    
    def get_learning_insights(self):
        """Obtener insights del aprendizaje"""
        insights = {
            'recent_accuracy': 0.0,
            'total_predictions': len(self.performance_history),
            'effective_patterns': [],
            'market_condition_insights': [],
            'feature_importance': dict(self.feature_importance)
        }
        
        # Calcular accuracy reciente
        if self.performance_history:
            recent_correct = sum(1 for r in list(self.performance_history)[-20:] if r['correct'])
            insights['recent_accuracy'] = recent_correct / min(20, len(self.performance_history))
        
        # Patrones efectivos
        for pattern, stats in self.pattern_effectiveness.items():
            if stats['total'] >= 5:
                accuracy = stats['correct'] / stats['total']
                if accuracy > 0.55:
                    insights['effective_patterns'].append({
                        'pattern': pattern,
                        'accuracy': accuracy,
                        'occurrences': stats['total']
                    })
        
        # Insights por condición de mercado
        for condition, stats in self.market_condition_performance.items():
            if stats['total'] >= 5:
                accuracy = stats['correct'] / stats['total']
                insights['market_condition_insights'].append({
                    'condition': condition,
                    'accuracy': accuracy,
                    'occurrences': stats['total']
                })
        
        return insights
    
    def get_confidence_boost(self, analysis):
        """Obtener boost de confianza basado en aprendizaje"""
        if not self.learning_enabled:
            return 0
        
        boost = 0
        
        # Boost por patrones efectivos
        patterns = analysis.get('detected_patterns', [])
        for pattern in patterns:
            stats = self.pattern_effectiveness[pattern]
            if stats['total'] >= 3:
                accuracy = stats['correct'] / stats['total']
                if accuracy > 0.6:
                    boost += (accuracy - 0.5) * 10
        
        # Boost por condición de mercado favorable
        market_phase = analysis.get('market_phase', '')
        stats = self.market_condition_performance[market_phase]
        if stats['total'] >= 5:
            accuracy = stats['correct'] / stats['total']
            if accuracy > 0.6:
                boost += (accuracy - 0.5) * 8
        
        return min(15, boost)

# ------------------ PREDICTOR PROFESIONAL CON APRENDIZAJE ------------------
class ProfessionalLearningPredictor:
    def __init__(self):
        self.tick_analyzer = ProfessionalTickAnalyzer()
        self.learning_system = ContinuousLearningSystem()
        self.last_prediction = None
        self.prediction_history = deque(maxlen=30)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'high_confidence_predictions': 0,
            'high_confidence_correct': 0
        }
        
    def process_tick(self, price: float):
        """Procesar cada tick"""
        timestamp = time.time()
        tick_data = self.tick_analyzer.add_tick(price, timestamp)
        return tick_data
    
    def predict_next_candle(self, seconds_remaining: float):
        """Predicción profesional con aprendizaje"""
        analysis = self.tick_analyzer.get_professional_analysis()
        
        if analysis.get('status') == 'INSUFFICIENT_DATA':
            return {
                'direction': 'N/A',
                'confidence': 0,
                'reason': analysis['message'],
                'tick_count': analysis['tick_count'],
                'data_quality': analysis.get('data_quality', 0),
                'timestamp': now_iso()
            }
        
        # PREDICCIÓN PROFESIONAL
        prediction = self._professional_prediction_engine(analysis, seconds_remaining)
        
        # APLICAR APRENDIZAJE A LA CONFIANZA
        if self.learning_system.learning_enabled:
            confidence_boost = self.learning_system.get_confidence_boost(analysis)
            prediction['confidence'] = min(95, prediction['confidence'] + confidence_boost)
            prediction['learning_boost'] = confidence_boost
        
        self.last_prediction = prediction
        self.prediction_history.append(prediction)
        
        # LOG PROFESIONAL
        if prediction['direction'] != 'N/A':
            confidence_level = "ALTA" if prediction['confidence'] >= 75 else "MEDIA" if prediction['confidence'] >= 60 else "BAJA"
            learning_info = f" (+{prediction.get('learning_boost', 0)})" if prediction.get('learning_boost', 0) > 0 else ""
            logging.info(f"🎯 PREDICCIÓN PRO: {prediction['direction']} | Conf: {prediction['confidence']}%{learning_info} | Ticks: {prediction['tick_count']} | Calidad: {analysis['data_quality']:.0%}")
        
        return prediction
    
    def _professional_prediction_engine(self, analysis, seconds_remaining):
        """Motor de predicción profesional"""
        # FACTORES CLAVE MEJORADOS
        momentum = analysis['price_momentum']
        imbalance = analysis['buy_sell_imbalance']
        acceleration = analysis['price_acceleration']
        patterns = analysis['detected_patterns']
        trend_strength = analysis['trend_strength']
        market_phase = analysis['market_phase']
        pressure_strength = analysis['pressure_strength']
        trend_consistency = analysis['trend_consistency']
        data_quality = analysis['data_quality']
        
        # SISTEMA DE PUNTUACIÓN PROFESIONAL
        buy_signals = 0
        sell_signals = 0
        confidence_factors = []
        reasons = []
        
        # 1. SEÑAL: MOMENTUM Y ACELERACIÓN
        momentum_score = self._evaluate_momentum_signal(momentum, acceleration, trend_consistency)
        if momentum_score > 0:
            buy_signals += abs(momentum_score)
            reasons.append(f"Momentum alcista profesional ({momentum:.1f}pips)")
        elif momentum_score < 0:
            sell_signals += abs(momentum_score)
            reasons.append(f"Momentum bajista profesional ({momentum:.1f}pips)")
        confidence_factors.append(min(0.8, abs(momentum_score) / 12))
        
        # 2. SEÑAL: PRESIÓN DE MERCADO
        pressure_score = self._evaluate_pressure_signal(imbalance, pressure_strength)
        if pressure_score > 0:
            buy_signals += pressure_score
            reasons.append(f"Presión compradora fuerte ({imbalance:.2f})")
        elif pressure_score < 0:
            sell_signals += abs(pressure_score)
            reasons.append(f"Presión vendedora fuerte ({imbalance:.2f})")
        confidence_factors.append(min(0.7, abs(pressure_score) / 10))
        
        # 3. SEÑAL: PATRONES DETECTADOS
        pattern_score = self._evaluate_pattern_signals(patterns, trend_strength)
        buy_signals += max(0, pattern_score)
        sell_signals += max(0, -pattern_score)
        if abs(pattern_score) > 2:
            confidence_factors.append(0.6)
            reasons.append("Patrones técnicos favorables")
        
        # 4. SEÑAL: FASE DE MERCADO
        phase_score = self._evaluate_market_phase_signal(market_phase, trend_strength)
        buy_signals += max(0, phase_score)
        sell_signals += max(0, -phase_score)
        confidence_factors.append(0.5 if abs(phase_score) > 1 else 0.3)
        
        # 5. SEÑAL: SOPORTE/RESISTENCIA
        sr_score = self._evaluate_support_resistance_signal(analysis['support_resistance'], analysis['current_price'])
        buy_signals += max(0, sr_score)
        sell_signals += max(0, -sr_score)
        if abs(sr_score) > 1:
            confidence_factors.append(0.5)
            reasons.append("Niveles clave detectados")
        
        # DECISIÓN FINAL PROFESIONAL
        if buy_signals > sell_signals:
            direction = "ALZA"
            signal_advantage = (buy_signals - sell_signals) / max(buy_signals, 1)
        elif sell_signals > buy_signals:
            direction = "BAJA"
            signal_advantage = (sell_signals - buy_signals) / max(sell_signals, 1)
        else:
            direction = "ALZA" if momentum > 0 else "BAJA"
            signal_advantage = 0.1
            reasons.append("Señales equilibradas - decisión por momentum")
        
        # CÁLCULO DE CONFIANZA PROFESIONAL
        if confidence_factors:
            base_confidence = np.mean(confidence_factors) * 100
        else:
            base_confidence = 40
            
        # AJUSTES PROFESIONALES
        confidence = base_confidence + (signal_advantage * 25)
        confidence = confidence * (0.6 + 0.4 * data_quality)  # Ajuste por calidad
        
        # Factor de tiempo (más ticks = más confianza)
        tick_factor = min(1.0, analysis['tick_count'] / IDEAL_TICKS_FOR_PREDICTION)
        confidence = confidence * (0.7 + 0.3 * tick_factor)
        
        # Límites profesionales
        confidence = max(25, min(90, confidence))
        
        return {
            'direction': direction,
            'confidence': int(confidence),
            'tick_count': analysis['tick_count'],
            'current_price': analysis['current_price'],
            'reasons': reasons,
            'market_phase': market_phase,
            'momentum': round(momentum, 2),
            'imbalance': round(imbalance, 2),
            'pressure_strength': round(pressure_strength, 2),
            'data_quality': round(data_quality, 2),
            'analysis_quality': round(analysis['analysis_quality'], 2),
            'timestamp': now_iso(),
            'model_used': 'PROFESSIONAL_V46'
        }
    
    def _evaluate_momentum_signal(self, momentum, acceleration, consistency):
        """Evaluar señal de momentum profesional"""
        score = 0
        
        # Momentum fuerte
        if abs(momentum) > 2.5:
            score += 6 * np.sign(momentum)
        elif abs(momentum) > 1.5:
            score += 4 * np.sign(momentum)
        elif abs(momentum) > 0.8:
            score += 2 * np.sign(momentum)
        
        # Aceleración favorable
        if momentum * acceleration > 0.2:
            score += 3 * np.sign(momentum)
        elif momentum * acceleration < -0.3:
            score -= 3 * np.sign(momentum)
            
        # Consistencia
        if consistency > 0.7:
            score += 2 * np.sign(momentum)
        elif consistency < -0.7:
            score -= 2 * np.sign(momentum)
            
        return score
    
    def _evaluate_pressure_signal(self, imbalance, pressure_strength):
        """Evaluar señal de presión de mercado"""
        score = 0
        
        # Desequilibrio fuerte
        if abs(imbalance) > 0.5:
            score += 5 * np.sign(imbalance)
        elif abs(imbalance) > 0.3:
            score += 3 * np.sign(imbalance)
        elif abs(imbalance) > 0.15:
            score += 1 * np.sign(imbalance)
            
        # Fuerza de presión
        if abs(pressure_strength) > 0.4:
            score += 3 * np.sign(pressure_strength)
        elif abs(pressure_strength) > 0.2:
            score += 1 * np.sign(pressure_strength)
            
        return score
    
    def _evaluate_pattern_signals(self, patterns, trend_strength):
        """Evaluar señales de patrones"""
        score = 0
        
        for pattern in patterns:
            if 'very_strong_trend_up' in pattern or 'excellent_trend_up' in pattern:
                score += 8
            elif 'very_strong_trend_down' in pattern or 'excellent_trend_down' in pattern:
                score -= 8
            elif 'strong_trend_up' in pattern or 'good_trend_up' in pattern:
                score += 6
            elif 'strong_trend_down' in pattern or 'good_trend_down' in pattern:
                score -= 6
            elif 'very_strong_breakout' in pattern:
                score += 5 if trend_strength > 0 else -5
            elif 'strong_breakout' in pattern:
                score += 4 if trend_strength > 0 else -4
            elif 'strong_accumulation' in pattern:
                score += 3
            elif 'professional_reversal_signal' in pattern:
                score -= 4 * np.sign(trend_strength) if abs(trend_strength) > 1 else 0
            elif 'low_noise_environment' in pattern:
                score += 2
            elif 'high_noise_environment' in pattern:
                score -= 2
                
        return score
    
    def _evaluate_market_phase_signal(self, market_phase, trend_strength):
        """Evaluar señal de fase de mercado"""
        phase_scores = {
            'very_strong_trend': 8 * np.sign(trend_strength),
            'strong_trend': 6 * np.sign(trend_strength),
            'moderate_trend': 4 * np.sign(trend_strength),
            'efficient_movement': 3 * np.sign(trend_strength),
            'building_pressure': 2 * np.sign(trend_strength) if abs(trend_strength) > 0.5 else 0,
            'tight_consolidation': 0,
            'consolidation': 0,
            'high_volatility': -3,
            'neutral': 0
        }
        
        return phase_scores.get(market_phase, 0)
    
    def _evaluate_support_resistance_signal(self, sr_analysis, current_price):
        """Evaluar señal de soporte/resistencia"""
        if not sr_analysis or not sr_analysis['support'] or not sr_analysis['resistance']:
            return 0
            
        score = 0
        distance_to_support = sr_analysis.get('distance_to_support')
        distance_to_resistance = sr_analysis.get('distance_to_resistance')
        quality = sr_analysis.get('quality', 0)
        
        # Señal fuerte cerca de soporte
        if distance_to_support and distance_to_support < 0.2:
            score += 4 * quality
            
        # Señal moderada cerca de soporte
        elif distance_to_support and distance_to_support < 0.4:
            score += 2 * quality
            
        # Señal fuerte cerca de resistencia
        if distance_to_resistance and distance_to_resistance < 0.2:
            score -= 4 * quality
            
        # Señal moderada cerca de resistencia
        elif distance_to_resistance and distance_to_resistance < 0.4:
            score -= 2 * quality
            
        return score
    
    def get_current_analysis(self):
        """Obtener análisis actual"""
        return self.tick_analyzer.get_professional_analysis()
    
    def reset_for_new_candle(self):
        """Preparar para nueva vela"""
        self.tick_analyzer.reset_analysis()
    
    def record_prediction_result(self, actual_direction: str):
        """Registrar resultado de predicción"""
        if self.last_prediction:
            correct = self.last_prediction['direction'] == actual_direction
            
            # Actualizar estadísticas
            self.performance_stats['total_predictions'] += 1
            if correct:
                self.performance_stats['correct_predictions'] += 1
            
            # Registrar en sistema de aprendizaje
            analysis = self.get_current_analysis()
            if analysis.get('status') != 'INSUFFICIENT_DATA':
                self.learning_system.record_prediction_result(
                    self.last_prediction, actual_direction, analysis
                )
            
            # Log de performance
            if self.performance_stats['total_predictions'] % 10 == 0:
                total = self.performance_stats['total_predictions']
                correct = self.performance_stats['correct_predictions']
                accuracy = (correct / total * 100) if total > 0 else 0
                logging.info(f"📈 PERFORMANCE: {accuracy:.1f}% de precisión ({correct}/{total})")

# -------------- IQ CONNECTION PROFESIONAL --------------
class ProfessionalIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        self.actual_pair = None
        self.connection_attempts = 0
        self.max_attempts = 3
        
    def connect(self):
        """Conectar a IQ Option de manera profesional"""
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
                self.connection_attempts = 0
                logging.info("✅ Conectado exitosamente a IQ Option")
                self._find_working_pair()
                return self.iq
            else:
                self.connection_attempts += 1
                logging.warning(f"⚠️ Conexión fallida ({self.connection_attempts}/{self.max_attempts}): {reason}")
                
                if self.connection_attempts < self.max_attempts:
                    time.sleep(2)
                    return self.connect()
                else:
                    logging.error("❌ Máximo de intentos de conexión alcanzado")
                    return None
                
        except Exception as e:
            logging.error(f"❌ Error conexión: {e}")
            return None

    def _find_working_pair(self):
        """Encontrar un par que funcione profesionalmente"""
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

    def get_realtime_price(self):
        """Obtener precio en tiempo real profesional"""
        try:
            if not self.connected or not self.iq:
                if self.connection_attempts < self.max_attempts:
                    self.connect()
                return None

            working_pair = self.actual_pair if self.actual_pair else "EURUSD"
            
            # Método principal: candles en tiempo real
            try:
                candles = self.iq.get_candles(working_pair, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"get_candles falló: {e}")

            # Método alternativo
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

            # Usar último precio conocido
            if self.last_price:
                return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            if self.connection_attempts < self.max_attempts:
                self.connect()
            
        return None

    def _record_tick(self, price):
        """Registrar tick recibido profesionalmente"""
        self.tick_count += 1
        self.last_price = price
        
        # Log informativo controlado
        if self.tick_count <= 5 or self.tick_count % 25 == 0:
            pair_info = f" ({self.actual_pair})" if self.actual_pair else ""
            logging.info(f"💰 Tick #{self.tick_count}{pair_info}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL PROFESIONAL ---------------
professional_connector = ProfessionalIQConnector()
professional_predictor = ProfessionalLearningPredictor()

# Estado global profesional
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": [],
    "timestamp": now_iso(),
    "data_quality": 0.0,
    "market_phase": "N/A",
    "model_used": "PROFESSIONAL_V46"
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_10': deque(maxlen=10),
    'last_validation': None
}

def professional_tick_analyzer():
    """Loop principal profesional de análisis tick-by-tick"""
    global current_prediction
    
    logging.info("🎯 DELOWYSS AI V4.6-PRO PROFESSIONAL iniciado")
    logging.info("📊 Sistema profesional de análisis tick-by-tick con aprendizaje continuo")
    
    professional_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_tick_time = 0
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # 1. OBTENER TICK ACTUAL
            price = professional_connector.get_realtime_price()
            
            if price is not None and price > 0:
                # 2. PROCESAR TICK CON CONTROL DE FRECUENCIA
                if current_time - last_tick_time >= 0.08:
                    professional_predictor.process_tick(price)
                    last_tick_time = current_time
                
                # Actualizar estado actual
                current_analysis = professional_predictor.get_current_analysis()
                current_prediction.update({
                    "current_price": price,
                    "tick_count": professional_predictor.tick_analyzer.tick_count,
                    "data_quality": current_analysis.get('data_quality', 0) if current_analysis else 0,
                    "market_phase": current_analysis.get('market_phase', 'N/A') if current_analysis else 'N/A',
                    "timestamp": now_iso()
                })
            
            # 3. PREDICCIÓN EN VENTANA OPTIMIZADA
            if seconds_remaining <= PREDICTION_WINDOW and seconds_remaining > 1:
                if professional_predictor.tick_analyzer.tick_count >= MIN_TICKS_FOR_ANALYSIS:
                    if current_time - last_prediction_time >= 1.2:
                        prediction = professional_predictor.predict_next_candle(seconds_remaining)
                        current_prediction.update(prediction)
                        last_prediction_time = current_time
            
            # 4. CAMBIO DE VELA - VALIDACIÓN Y RESET
            if current_candle_start > last_candle_start:
                # Validar predicción anterior
                validate_previous_prediction()
                
                # Reset para nueva vela
                professional_predictor.reset_for_new_candle()
                last_candle_start = current_candle_start
                
                logging.info(f"🕯️ Nueva vela iniciada - Analizando ticks profesionalmente...")
            
            time.sleep(0.03)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal profesional: {e}")
            time.sleep(1)

def validate_previous_prediction():
    """Validar predicción anterior"""
    global current_prediction, performance_stats
    
    if not professional_predictor.last_prediction:
        return
        
    try:
        current_analysis = professional_predictor.get_current_analysis()
        if current_analysis.get('status') == 'INSUFFICIENT_DATA':
            return
            
        # Precio de cierre de la vela actual
        close_price = current_analysis['current_price']
        
        # Recuperar predicción anterior
        prev_prediction = professional_predictor.last_prediction
        if not prev_prediction or 'current_price' not in prev_prediction:
            return
            
        # Precio de referencia
        reference_price = prev_prediction['current_price']
        
        # Determinar dirección real
        actual_direction = "ALZA" if close_price > reference_price else "BAJA"
        predicted_direction = prev_prediction.get('direction', 'N/A')
        
        correct = (actual_direction == predicted_direction)
        change_pips = (close_price - reference_price) * 10000
        
        validation_result = {
            "timestamp": now_iso(),
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": correct,
            "confidence": prev_prediction.get('confidence', 0),
            "price_change_pips": round(change_pips, 2),
            "tick_count": prev_prediction.get('tick_count', 0),
            "data_quality": prev_prediction.get('data_quality', 0),
            "model_used": prev_prediction.get('model_used', 'UNKNOWN')
        }
        
        # Actualizar estadísticas
        performance_stats['total_predictions'] += 1
        performance_stats['correct_predictions'] += 1 if correct else 0
        performance_stats['last_10'].append(1 if correct else 0)
        
        # Calcular precisión reciente
        if performance_stats['last_10']:
            recent_correct = sum(performance_stats['last_10'])
            performance_stats['recent_accuracy'] = (recent_correct / len(performance_stats['last_10'])) * 100
            
        performance_stats['last_validation'] = validation_result
        
        # Registrar resultado en el predictor
        professional_predictor.record_prediction_result(actual_direction)
        
        status = "✅ CORRECTA" if correct else "❌ ERRÓNEA"
        logging.info(f"📊 VALIDACIÓN: {status} | Pred: {predicted_direction} | Real: {actual_direction} | Conf: {prev_prediction.get('confidence', 0)}% | Change: {change_pips:.1f}pips")
        
        # Log de rendimiento periódico
        if performance_stats['total_predictions'] % 10 == 0:
            total = performance_stats['total_predictions']
            correct_total = performance_stats['correct_predictions']
            overall_accuracy = (correct_total / total * 100) if total > 0 else 0
            
            logging.info(f"📈 RENDIMIENTO SISTEMA: Global: {overall_accuracy:.1f}% | Reciente: {performance_stats['recent_accuracy']:.1f}%")
        
    except Exception as e:
        logging.error(f"❌ Error en validación profesional: {e}")

# --------------- FASTAPI PROFESIONAL ---------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    analysis = professional_predictor.get_current_analysis()
    
    direction = current_prediction.get("direction", "N/A")
    color = "#00ff88" if direction == "ALZA" else ("#ff4444" if direction == "BAJA" else "#ffbb33")
    
    # Calcular precisión profesional
    total = performance_stats.get('total_predictions', 0)
    correct = performance_stats.get('correct_predictions', 0)
    accuracy = (correct / total * 100) if total > 0 else 0
    recent_accuracy = performance_stats.get('recent_accuracy', 0)
    
    # Insights de aprendizaje
    learning_insights = professional_predictor.learning_system.get_learning_insights()
    
    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8'>
        <meta name='viewport' content='width=device-width'>
        <title>Delowyss AI V4.6-PRO Professional</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background: #0f172a;
                color: #fff;
                padding: 20px;
                margin: 0;
            }}
            .card {{
                background: rgba(255,255,255,0.03);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
            }}
            .prediction-card {{
                border-left: 6px solid {color};
                padding: 20px;
                background: rgba(255,255,255,0.05);
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 15px 0;
            }}
            .metric {{
                background: rgba(255,255,255,0.05);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .countdown {{
                font-size: 2.5em;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
            }}
            .quality-bar {{
                height: 6px;
                background: #333;
                border-radius: 3px;
                margin: 5px 0;
                overflow: hidden;
            }}
            .quality-fill {{
                height: 100%;
                background: #00ff88;
                transition: width 0.3s ease;
            }}
            .phase-indicator {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                margin-left: 8px;
            }}
            .phase-strong {{ background: #00ff88; color: #000; }}
            .phase-moderate {{ background: #ffbb33; color: #000; }}
            .phase-weak {{ background: #ff4444; color: #fff; }}
            .phase-neutral {{ background: #666; color: #fff; }}
            .learning-badge {{
                background: linear-gradient(45deg, #ff0080, #00ff88);
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 0.7em;
                margin-left: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🎯 Delowyss AI V4.6-PRO Professional</h1>
            <p>Sistema profesional de análisis tick-by-tick con aprendizaje continuo</p>
            
            <div class="countdown" id="countdown">--</div>
            
            <div class="prediction-card">
                <h2 style="color: {color}; margin: 0 0 10px 0;">
                    {direction} — {current_prediction.get('confidence', 0)}% confianza
                    {'<span class="learning-badge">AI+'</span>' if current_prediction.get('learning_boost', 0) > 0 else ''}
                </h2>
                <p>Precio: {current_prediction.get('current_price', 0):.5f} • Ticks: {current_prediction.get('tick_count', 0)}</p>
                <p>
                    Fase: {analysis.get('market_phase', 'N/A') if analysis else 'N/A'}
                    <span class="phase-indicator {{
                        'phase-strong' if analysis and 'strong' in analysis.get('market_phase', '') else 
                        'phase-moderate' if analysis and 'moderate' in analysis.get('market_phase', '') else 
                        'phase-weak' if analysis and 'weak' in analysis.get('market_phase', '') else 
                        'phase-neutral'
                    }}">
                        {analysis.get('market_phase', 'N/A') if analysis else 'N/A'}
                    </span>
                </p>
                <div class="quality-bar">
                    <div class="quality-fill" style="width: {current_prediction.get('data_quality', 0) * 100}%"></div>
                </div>
                <p>Calidad de datos: {(current_prediction.get('data_quality', 0) * 100):.1f}%</p>
            </div>

            <div class="card">
                <h3>📊 Análisis en Tiempo Real Profesional</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div>Momentum</div>
                        <div>{analysis.get('price_momentum', 0):.1f}pips</div>
                    </div>
                    <div class="metric">
                        <div>Desequilibrio</div>
                        <div>{analysis.get('buy_sell_imbalance', 0):.2f}</div>
                    </div>
                    <div class="metric">
                        <div>Presión</div>
                        <div>{analysis.get('pressure_strength', 0):.2f}</div>
                    </div>
                    <div class="metric">
                        <div>Precisión Global</div>
                        <div>{accuracy:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div>Precisión Reciente</div>
                        <div>{recent_accuracy:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div>Total Predicciones</div>
                        <div>{total}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🎯 Razones de Predicción Profesional</h3>
                <ul id="reasons">
                    {"".join([f"<li>📈 {r}</li>" for r in current_prediction.get('reasons', [])])}
                </ul>
            </div>

            <div class="card">
                <h3>🧠 Sistema de Aprendizaje</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div>Predicciones Aprendidas</div>
                        <div>{learning_insights.get('total_predictions', 0)}</div>
                    </div>
                    <div class="metric">
                        <div>Accuracy Reciente</div>
                        <div>{(learning_insights.get('recent_accuracy', 0) * 100):.1f}%</div>
                    </div>
                    <div class="metric">
                        <div>Patrones Efectivos</div>
                        <div>{len(learning_insights.get('effective_patterns', []))}</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                document.getElementById('countdown').textContent = remaining + 's';
            }}
            
            function updateData() {{
                fetch('/api/prediction')
                    .then(response => response.json())
                    .then(data => {{
                        const direction = data.direction || 'N/A';
                        const confidence = data.confidence || 0;
                        const reasons = data.reasons || [];
                        const learningBoost = data.learning_boost || 0;
                        
                        document.querySelector('.prediction-card h2').innerHTML = 
                            `${{direction}} — ${{confidence}}% confianza ${{learningBoost > 0 ? '<span class="learning-badge">AI+</span>' : ''}}`;
                            
                        document.getElementById('reasons').innerHTML = 
                            reasons.map(r => `<li>📈 ${{r}}</li>`).join('');
                    }})
                    .catch(error => console.error('Error:', error));
            }}
            
            setInterval(updateCountdown, 1000);
            setInterval(updateData, 2000);
            updateCountdown();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/analysis")
def api_analysis():
    analysis = professional_predictor.get_current_analysis()
    return JSONResponse(analysis)

@app.get("/api/performance")
def api_performance():
    total = performance_stats.get('total_predictions', 0)
    correct = performance_stats.get('correct_predictions', 0)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    learning_insights = professional_predictor.learning_system.get_learning_insights()
    
    return JSONResponse({
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 1),
        "recent_accuracy": performance_stats.get('recent_accuracy', 0),
        "last_validation": performance_stats.get('last_validation'),
        "learning_insights": learning_insights
    })

@app.get("/api/learning")
def api_learning():
    insights = professional_predictor.learning_system.get_learning_insights()
    return JSONResponse(insights)

@app.get("/api/status")
def api_status():
    return JSONResponse({
        "status": "online",
        "version": "V4.6-PRO-PROFESSIONAL",
        "connected": professional_connector.connected,
        "continuous_learning": CONTINUOUS_LEARNING,
        "total_ticks_processed": professional_predictor.tick_analyzer.tick_count,
        "timestamp": now_iso()
    })

# --------------- INICIALIZACIÓN PROFESIONAL ---------------
def start_background_tasks():
    """Iniciar análisis en background profesional"""
    analyzer_thread = threading.Thread(target=professional_tick_analyzer, daemon=True)
    analyzer_thread.start()
    logging.info("📊 Sistema profesional de análisis tick-by-tick iniciado")

start_background_tasks()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
