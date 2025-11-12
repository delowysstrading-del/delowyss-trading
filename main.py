# main.py - V5.10 ANÁLISIS COMPLETO DE VELA + PREDICCIÓN
"""
Delowyss Trading AI — V5.10 ANÁLISIS COMPLETO DE VELA CON PREDICCIÓN
CEO: Eduardo Solis — © 2025
Sistema de análisis completo de vela actual para predecir siguiente vela
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
PREDICTION_WINDOW = 5  # Predecir a 5 segundos del final
MIN_TICKS_FOR_PREDICTION = 20
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
    
    def is_prediction_time(self):
        """Determina si es momento de hacer predicción (últimos 5 segundos)"""
        remaining = self.get_remaining_time()
        return remaining <= PREDICTION_WINDOW and remaining > 0

# ------------------ ANALIZADOR DE VELA COMPLETA ------------------
class CompleteCandleAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=100)
        
        # Almacenar datos de vela anterior para comparación
        self.previous_candle = {
            'open': None,
            'high': None, 
            'low': None,
            'close': None,
            'direction': None,
            'body_size': None
        }
        
        # Métricas de análisis de vela completa
        self.candle_phases = {
            'first_15s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'next_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'middle_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'final_5s': {'ticks': 0, 'analysis': {}, 'completed': False}
        }
        
        # Análisis de comportamiento por segmentos de tiempo
        self.time_segments = {
            '0-15s': {'price_action': [], 'volatility': 0, 'direction': None},
            '15-35s': {'price_action': [], 'volatility': 0, 'direction': None},
            '35-55s': {'price_action': [], 'volatility': 0, 'direction': None},
            '55-60s': {'price_action': [], 'volatility': 0, 'direction': None}
        }
        
        # Indicadores técnicos para la vela actual
        self.velocity_metrics = deque(maxlen=50)
        self.pressure_zones = deque(maxlen=30)
        self.momentum_indicators = deque(maxlen=20)
        self.support_resistance = deque(maxlen=15)
        self.volume_profile = deque(maxlen=25)
        
        self.candle_start_time = None
        self.prediction_ready = False
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            # Inicializar vela si es necesario
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Análisis completo activado")
                self._reset_candle_analysis()
            
            # Actualizar precios extremos
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)
            self.current_candle_close = price
            
            tick_data = {
                'price': price,
                'timestamp': current_time,
                'seconds_remaining': seconds_remaining,
                'candle_age': current_time - self.candle_start_time if self.candle_start_time else 0,
                'segment': self._get_time_segment(current_time - self.candle_start_time)
            }
            
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            # Análisis en tiempo real según el segmento
            self._analyze_time_segment(tick_data)
            self._calculate_advanced_metrics(tick_data)
            self._analyze_pressure_zones(tick_data)
            
            # Verificar si se completaron fases
            self._check_phase_completion(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"❌ Error en add_tick: {e}")
            return None
    
    def _get_time_segment(self, candle_age):
        """Determina el segmento de tiempo actual de la vela"""
        if candle_age < 15:
            return '0-15s'
        elif candle_age < 35:
            return '15-35s'
        elif candle_age < 55:
            return '35-55s'
        else:
            return '55-60s'
    
    def _reset_candle_analysis(self):
        """Reinicia el análisis para nueva vela"""
        self.candle_phases = {
            'first_15s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'next_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'middle_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'final_5s': {'ticks': 0, 'analysis': {}, 'completed': False}
        }
        self.time_segments = {
            '0-15s': {'price_action': [], 'volatility': 0, 'direction': None},
            '15-35s': {'price_action': [], 'volatility': 0, 'direction': None},
            '35-55s': {'price_action': [], 'volatility': 0, 'direction': None},
            '55-60s': {'price_action': [], 'volatility': 0, 'direction': None}
        }
        self.prediction_ready = False
    
    def _analyze_time_segment(self, tick_data):
        """Analiza el comportamiento del precio en cada segmento de tiempo"""
        segment = tick_data['segment']
        price = tick_data['price']
        
        # Agregar precio al segmento actual
        self.time_segments[segment]['price_action'].append(price)
        
        # Calcular volatilidad del segmento
        if len(self.time_segments[segment]['price_action']) >= 5:
            prices = self.time_segments[segment]['price_action']
            volatility = (max(prices) - min(prices)) * 10000
            self.time_segments[segment]['volatility'] = volatility
            
            # Determinar dirección del segmento
            if len(prices) >= 3:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                positive_changes = len([x for x in price_changes if x > 0])
                buy_ratio = positive_changes / len(price_changes)
                
                if buy_ratio > 0.6:
                    self.time_segments[segment]['direction'] = 'ALCISTA'
                elif buy_ratio < 0.4:
                    self.time_segments[segment]['direction'] = 'BAJISTA'
                else:
                    self.time_segments[segment]['direction'] = 'LATERAL'
    
    def _check_phase_completion(self, tick_data):
        """Verifica y actualiza el estado de completitud de las fases"""
        candle_age = tick_data['candle_age']
        segment = tick_data['segment']
        
        # Primera fase: 0-15 segundos
        if candle_age >= 15 and not self.candle_phases['first_15s']['completed']:
            self.candle_phases['first_15s']['completed'] = True
            self.candle_phases['first_15s']['analysis'] = self._analyze_phase('first_15s')
            logging.info("📊 Fase 0-15s completada - Análisis inicial listo")
        
        # Segunda fase: 15-35 segundos  
        elif candle_age >= 35 and not self.candle_phases['next_20s']['completed']:
            self.candle_phases['next_20s']['completed'] = True
            self.candle_phases['next_20s']['analysis'] = self._analyze_phase('next_20s')
            logging.info("📊 Fase 15-35s completada - Tendencia definiéndose")
        
        # Tercera fase: 35-55 segundos
        elif candle_age >= 55 and not self.candle_phases['middle_20s']['completed']:
            self.candle_phases['middle_20s']['completed'] = True
            self.candle_phases['middle_20s']['analysis'] = self._analyze_phase('middle_20s')
            logging.info("📊 Fase 35-55s completada - Comportamiento establecido")
        
        # Cuarta fase: Últimos 5 segundos (para predicción)
        elif segment == '55-60s' and not self.candle_phases['final_5s']['completed']:
            self.candle_phases['final_5s']['completed'] = True
            self.candle_phases['final_5s']['analysis'] = self._analyze_phase('final_5s')
            self.prediction_ready = True
            logging.info("🎯 Fase 55-60s - Predicción habilitada")
    
    def _analyze_phase(self, phase):
        """Analiza una fase específica de la vela"""
        try:
            if phase == 'first_15s':
                segment_data = self.time_segments['0-15s']
            elif phase == 'next_20s':
                segment_data = self.time_segments['15-35s']
            elif phase == 'middle_20s':
                segment_data = self.time_segments['35-55s']
            else:  # final_5s
                segment_data = self.time_segments['55-60s']
            
            if not segment_data['price_action']:
                return {}
            
            prices = segment_data['price_action']
            
            # Análisis básico
            high = max(prices)
            low = min(prices)
            open_price = prices[0] if prices else 0
            close_price = prices[-1] if prices else 0
            
            volatility = (high - low) * 10000
            body_size = abs(close_price - open_price) * 10000
            body_direction = 'ALCISTA' if close_price > open_price else 'BAJISTA' if close_price < open_price else 'LATERAL'
            
            # Análisis de tendencia intra-fase
            if len(prices) >= 5:
                x_values = np.arange(len(prices))
                try:
                    trend_coeff = np.polyfit(x_values, prices, 1)[0]
                    trend_strength = abs(trend_coeff) * 10000
                    trend_direction = 'ALCISTA' if trend_coeff > 0 else 'BAJISTA' if trend_coeff < 0 else 'LATERAL'
                except:
                    trend_strength = 0
                    trend_direction = 'LATERAL'
            else:
                trend_strength = 0
                trend_direction = segment_data.get('direction', 'LATERAL')
            
            # Presión de compra/venta
            if len(prices) >= 3:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes)
            else:
                buy_pressure = 0.5
            
            return {
                'prices_analyzed': len(prices),
                'volatility': volatility,
                'body_size': body_size,
                'body_direction': body_direction,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'buy_pressure': buy_pressure,
                'high': high,
                'low': low,
                'open': open_price,
                'close': close_price
            }
        except Exception as e:
            logging.debug(f"🔧 Error analizando fase {phase}: {e}")
            return {}
    
    def _calculate_advanced_metrics(self, tick_data):
        """Calcula métricas avanzadas en tiempo real"""
        try:
            current_price = tick_data['price']
            current_time = tick_data['timestamp']

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
        except Exception as e:
            logging.debug(f"🔧 Error en cálculo avanzado: {e}")
    
    def _analyze_pressure_zones(self, tick_data):
        """Analiza zonas de presión de compra/venta"""
        try:
            if len(self.ticks) < 6:
                return
                
            recent_ticks = list(self.ticks)[-6:]
            price_changes = []
            
            for i in range(1, len(recent_ticks)):
                change = recent_ticks[i]['price'] - recent_ticks[i-1]['price']
                price_changes.append(change)
            
            if price_changes:
                buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes)
                
                self.pressure_zones.append({
                    'buy_pressure': buy_pressure,
                    'sell_pressure': 1 - buy_pressure,
                    'timestamp': tick_data['timestamp'],
                    'strength': abs(buy_pressure - 0.5) * 2
                })
                
        except Exception as e:
            logging.debug(f"🔧 Error analizando presión: {e}")
    
    def get_candle_analysis(self):
        """Obtiene el análisis completo de la vela actual"""
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'ANALYZING',
                'tick_count': self.tick_count,
                'message': f'Analizando vela: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION} ticks',
                'current_progress': self._get_candle_progress()
            }
        
        try:
            # Análisis de fases completadas
            phase_analysis = {}
            for phase, data in self.candle_phases.items():
                if data['completed']:
                    phase_analysis[phase] = data['analysis']
            
            # Análisis de segmentos de tiempo
            segment_analysis = {}
            for segment, data in self.time_segments.items():
                if data['price_action']:
                    segment_analysis[segment] = {
                        'direction': data['direction'],
                        'volatility': data['volatility'],
                        'samples': len(data['price_action'])
                    }
            
            # Análisis general de la vela
            general_analysis = self._analyze_complete_candle()
            
            # Preparar predicción si está lista
            prediction_readiness = self._assess_prediction_readiness()
            
            result = {
                'status': 'COMPLETE_ANALYSIS',
                'tick_count': self.tick_count,
                'current_price': self.current_candle_close,
                'candle_progress': self._get_candle_progress(),
                'candle_stats': {
                    'open': self.current_candle_open,
                    'high': self.current_candle_high,
                    'low': self.current_candle_low,
                    'close': self.current_candle_close,
                    'range': (self.current_candle_high - self.current_candle_low) * 10000,
                    'body_size': abs(self.current_candle_close - self.current_candle_open) * 10000,
                    'direction': 'ALCISTA' if self.current_candle_close > self.current_candle_open else 'BAJISTA' if self.current_candle_close < self.current_candle_open else 'LATERAL'
                },
                'phase_analysis': phase_analysis,
                'segment_analysis': segment_analysis,
                'general_analysis': general_analysis,
                'prediction_readiness': prediction_readiness,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error en análisis de vela: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _get_candle_progress(self):
        """Calcula el progreso actual de la vela"""
        if not self.candle_start_time:
            return 0
        candle_age = time.time() - self.candle_start_time
        return min(100, (candle_age / TIMEFRAME) * 100)
    
    def _analyze_complete_candle(self):
        """Analiza el comportamiento completo de la vela"""
        try:
            if not self.ticks:
                return {}
            
            # Comportamiento por cuartos
            total_ticks = len(self.ticks)
            quarter_size = max(1, total_ticks // 4)
            
            quarters = []
            for i in range(4):
                start_idx = i * quarter_size
                end_idx = min((i + 1) * quarter_size, total_ticks)
                quarter_ticks = list(self.ticks)[start_idx:end_idx]
                
                if quarter_ticks:
                    quarter_prices = [t['price'] for t in quarter_ticks]
                    quarter_direction = 'ALCISTA' if quarter_prices[-1] > quarter_prices[0] else 'BAJISTA' if quarter_prices[-1] < quarter_prices[0] else 'LATERAL'
                    quarter_volatility = (max(quarter_prices) - min(quarter_prices)) * 10000
                    
                    quarters.append({
                        'quarter': i + 1,
                        'direction': quarter_direction,
                        'volatility': quarter_volatility,
                        'ticks': len(quarter_ticks)
                    })
            
            # Consistencia de la vela
            consistency_score = self._calculate_consistency()
            
            # Fuerza de la tendencia
            trend_strength = self._calculate_trend_strength()
            
            return {
                'quarters_analysis': quarters,
                'consistency_score': consistency_score,
                'trend_strength': trend_strength,
                'total_volatility': (self.current_candle_high - self.current_candle_low) * 10000,
                'current_momentum': self._calculate_current_momentum(),
                'pressure_balance': self._calculate_pressure_balance()
            }
        except Exception as e:
            logging.debug(f"🔧 Error en análisis completo: {e}")
            return {}
    
    def _calculate_consistency(self):
        """Calcula la consistencia del movimiento de la vela"""
        try:
            if len(self.ticks) < 10:
                return 50
                
            directions = []
            for segment in self.time_segments.values():
                if segment['direction']:
                    dir_map = {'ALCISTA': 1, 'BAJISTA': -1, 'LATERAL': 0}
                    directions.append(dir_map[segment['direction']])
            
            if directions:
                consistency = 1 - (np.std(directions) / 2)  # Normalizar a 0-1
                return min(100, consistency * 100)
            return 50
        except:
            return 50
    
    def _calculate_trend_strength(self):
        """Calcula la fuerza de la tendencia general"""
        try:
            if len(self.price_memory) < 8:
                return 0
                
            prices = list(self.price_memory)
            x_values = np.arange(len(prices))
            trend_coeff = np.polyfit(x_values, prices, 1)[0]
            return abs(trend_coeff) * 10000
        except:
            return 0
    
    def _calculate_current_momentum(self):
        """Calcula el momentum actual"""
        try:
            if len(self.price_memory) < 5:
                return 0
            prices = list(self.price_memory)[-5:]
            return (prices[-1] - prices[0]) * 10000
        except:
            return 0
    
    def _calculate_pressure_balance(self):
        """Calcula el balance de presión"""
        try:
            if not self.pressure_zones:
                return 0.5
            recent_pressure = list(self.pressure_zones)[-5:]
            avg_buy_pressure = np.mean([p['buy_pressure'] for p in recent_pressure])
            return avg_buy_pressure
        except:
            return 0.5
    
    def _assess_prediction_readiness(self):
        """Evalúa si el sistema está listo para predecir"""
        try:
            readiness = {
                'ready': self.prediction_ready,
                'phases_completed': sum(1 for phase in self.candle_phases.values() if phase['completed']),
                'total_phases': len(self.candle_phases),
                'data_sufficiency': min(100, (self.tick_count / 30) * 100),
                'analysis_quality': self._calculate_analysis_quality()
            }
            
            # Solo considerar listo si tenemos al menos 3 fases completas y suficientes ticks
            readiness['ready'] = (
                readiness['phases_completed'] >= 3 and 
                readiness['data_sufficiency'] >= 70 and
                self.prediction_ready
            )
            
            return readiness
        except:
            return {'ready': False, 'phases_completed': 0, 'data_sufficiency': 0}
    
    def _calculate_analysis_quality(self):
        """Calcula la calidad general del análisis"""
        quality = 0
        
        # Por cada fase completada
        quality += sum(20 for phase in self.candle_phases.values() if phase['completed'])
        
        # Por ticks suficientes
        quality += min(30, (self.tick_count / 40) * 30)
        
        # Por consistencia
        quality += min(20, self._calculate_consistency() / 5)
        
        return min(100, quality)
    
    def is_ready_for_prediction(self):
        """Verifica si el sistema está listo para hacer predicción"""
        readiness = self._assess_prediction_readiness()
        return readiness['ready']
    
    def reset(self):
        """Prepara el analyzer para la siguiente vela"""
        try:
            # Guardar vela actual como anterior
            if all([self.current_candle_open, self.current_candle_high, 
                   self.current_candle_low, self.current_candle_close]):
                self.previous_candle = {
                    'open': self.current_candle_open,
                    'high': self.current_candle_high,
                    'low': self.current_candle_low,
                    'close': self.current_candle_close,
                    'direction': 'ALCISTA' if self.current_candle_close > self.current_candle_open else 'BAJISTA',
                    'body_size': abs(self.current_candle_close - self.current_candle_open) * 10000
                }
            
            # Reiniciar para nueva vela
            self.ticks.clear()
            self.current_candle_open = None
            self.current_candle_high = None
            self.current_candle_low = None
            self.current_candle_close = None
            self.tick_count = 0
            self.price_memory.clear()
            self.velocity_metrics.clear()
            self.pressure_zones.clear()
            self.momentum_indicators.clear()
            self.candle_start_time = None
            self.prediction_ready = False
            
            # Mantener el reset de fases en _reset_candle_analysis
                
        except Exception as e:
            logging.error(f"❌ Error en reset: {e}")

# ------------------ PREDICTOR DE SIGUIENTE VELA ------------------
class NextCandlePredictor:
    def __init__(self):
        self.analyzer = CompleteCandleAnalyzer()
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'current_streak': 0,
            'best_streak': 0,
            'today_signals': 0
        }
        self.prediction_history = deque(maxlen=50)
        
    def process_tick(self, price: float, seconds_remaining: float = None):
        return self.analyzer.add_tick(price, seconds_remaining)
    
    def predict_next_candle(self):
        """Predice la dirección de la siguiente vela basado en el análisis completo"""
        analysis = self.analyzer.get_candle_analysis()
        
        if analysis.get('status') != 'COMPLETE_ANALYSIS':
            return {
                "direction": "LATERAL",
                "confidence": 50,
                "tick_count": self.analyzer.tick_count,
                "current_price": self.analyzer.current_candle_close or 0.0,
                "reasons": ["Análisis de vela en curso"],
                "timestamp": now_iso(),
                "status": "ANALYZING"
            }
        
        if not self.analyzer.is_ready_for_prediction():
            return {
                "direction": "LATERAL", 
                "confidence": 50,
                "tick_count": self.analyzer.tick_count,
                "current_price": self.analyzer.current_candle_close or 0.0,
                "reasons": ["Esperando análisis completo de vela"],
                "timestamp": now_iso(),
                "status": "WAITING"
            }
        
        try:
            # Obtener análisis detallado
            phase_analysis = analysis.get('phase_analysis', {})
            segment_analysis = analysis.get('segment_analysis', {})
            general_analysis = analysis.get('general_analysis', {})
            candle_stats = analysis.get('candle_stats', {})
            
            # 1. Análisis de tendencia por fases
            phase_trends = self._analyze_phase_trends(phase_analysis)
            
            # 2. Análisis de momentum y presión
            momentum_analysis = self._analyze_momentum_pressure(general_analysis)
            
            # 3. Análisis de comportamiento por segmentos
            segment_prediction = self._analyze_segment_behavior(segment_analysis)
            
            # 4. Análisis de vela completa
            candle_pattern = self._analyze_candle_pattern(candle_stats, general_analysis)
            
            # Combinar todas las predicciones
            final_prediction = self._combine_predictions(
                phase_trends, momentum_analysis, segment_prediction, candle_pattern
            )
            
            # Generar razones detalladas
            reasons = self._generate_prediction_reasons(
                phase_trends, momentum_analysis, segment_prediction, candle_pattern
            )
            
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['today_signals'] += 1
            
            prediction = {
                "direction": final_prediction['direction'],
                "confidence": final_prediction['confidence'],
                "tick_count": self.analyzer.tick_count,
                "current_price": analysis['current_price'],
                "reasons": reasons,
                "timestamp": now_iso(),
                "status": "PREDICTION_READY",
                "analysis_breakdown": {
                    "phase_analysis": phase_trends,
                    "momentum_analysis": momentum_analysis,
                    "segment_analysis": segment_prediction,
                    "candle_pattern": candle_pattern
                }
            }
            
            self.prediction_history.append(prediction)
            return prediction
            
        except Exception as e:
            logging.error(f"❌ Error en predicción: {e}")
            return {
                "direction": "LATERAL",
                "confidence": 50,
                "tick_count": self.analyzer.tick_count,
                "reasons": [f"Error en análisis: {str(e)}"],
                "timestamp": now_iso(),
                "status": "ERROR"
            }
    
    def _analyze_phase_trends(self, phase_analysis):
        """Analiza las tendencias por fases de la vela"""
        trends = []
        strengths = []
        
        for phase, analysis in phase_analysis.items():
            if analysis.get('trend_direction') and analysis.get('trend_strength', 0) > 0.5:
                trends.append(analysis['trend_direction'])
                strengths.append(analysis['trend_strength'])
        
        if trends:
            # Ponderar más las fases finales
            weights = [0.1, 0.2, 0.3, 0.4]  # Pesos para cada fase
            weighted_trends = {}
            
            for i, trend in enumerate(trends):
                weight = weights[i] if i < len(weights) else 0.1
                if trend in weighted_trends:
                    weighted_trends[trend] += weight
                else:
                    weighted_trends[trend] = weight
            
            dominant_trend = max(weighted_trends, key=weighted_trends.get)
            avg_strength = np.mean(strengths) if strengths else 0
            
            return {
                'direction': dominant_trend,
                'strength': min(100, avg_strength * 10),
                'consistency': len(set(trends)) == 1  # True si todas las fases coinciden
            }
        
        return {'direction': 'LATERAL', 'strength': 0, 'consistency': False}
    
    def _analyze_momentum_pressure(self, general_analysis):
        """Analiza momentum y presión de la vela"""
        momentum = general_analysis.get('current_momentum', 0)
        pressure = general_analysis.get('pressure_balance', 0.5)
        
        direction = "ALZA" if momentum > 1.0 else "BAJA" if momentum < -1.0 else "LATERAL"
        strength = min(100, abs(momentum) * 20)
        
        pressure_signal = "ALZA" if pressure > 0.6 else "BAJA" if pressure < 0.4 else "LATERAL"
        pressure_strength = abs(pressure - 0.5) * 200
        
        return {
            'momentum_direction': direction,
            'momentum_strength': strength,
            'pressure_direction': pressure_signal,
            'pressure_strength': pressure_strength,
            'alignment': direction == pressure_signal
        }
    
    def _analyze_segment_behavior(self, segment_analysis):
        """Analiza el comportamiento por segmentos de tiempo"""
        segments = list(segment_analysis.keys())
        directions = []
        
        for segment in segments:
            if segment_analysis[segment].get('direction'):
                directions.append(segment_analysis[segment]['direction'])
        
        if directions:
            # Los segmentos finales tienen más peso
            recent_directions = directions[-2:] if len(directions) >= 2 else directions
            alcista_count = recent_directions.count('ALCISTA')
            bajista_count = recent_directions.count('BAJISTA')
            
            if alcista_count > bajista_count:
                direction = "ALZA"
                confidence = (alcista_count / len(recent_directions)) * 80
            elif bajista_count > alcista_count:
                direction = "BAJA"
                confidence = (bajista_count / len(recent_directions)) * 80
            else:
                direction = "LATERAL"
                confidence = 50
                
            return {
                'direction': direction,
                'confidence': confidence,
                'recent_alignment': alcista_count == len(recent_directions) or bajista_count == len(recent_directions)
            }
        
        return {'direction': 'LATERAL', 'confidence': 50, 'recent_alignment': False}
    
    def _analyze_candle_pattern(self, candle_stats, general_analysis):
        """Analiza el patrón de la vela actual"""
        direction = candle_stats.get('direction', 'LATERAL')
        body_size = candle_stats.get('body_size', 0)
        range_size = candle_stats.get('range', 0)
        
        # Vela con cuerpo grande → continuación probable
        if body_size > range_size * 0.7:  # Cuerpo > 70% del rango
            pattern_strength = 80
            pattern_type = "FUERTE"
        elif body_size > range_size * 0.4:  # Cuerpo > 40% del rango
            pattern_strength = 65
            pattern_type = "MODERADO"
        else:
            pattern_strength = 50
            pattern_type = "LIGERO"
        
        consistency = general_analysis.get('consistency_score', 50)
        
        return {
            'direction': direction,
            'strength': pattern_strength,
            'type': pattern_type,
            'consistency': consistency,
            'continuation_bias': pattern_strength > 60  # Sesgo hacia continuación
        }
    
    def _combine_predictions(self, phase_trends, momentum_analysis, segment_prediction, candle_pattern):
        """Combina todas las predicciones en una final"""
        predictions = [
            (phase_trends['direction'], phase_trends['strength'], 0.30),
            (momentum_analysis['momentum_direction'], momentum_analysis['momentum_strength'], 0.25),
            (segment_prediction['direction'], segment_prediction['confidence'], 0.25),
            (candle_pattern['direction'], candle_pattern['strength'], 0.20)
        ]
        
        direction_scores = {"ALZA": 0, "BAJA": 0, "LATERAL": 0}
        total_confidence = 0
        
        for direction, confidence, weight in predictions:
            direction_scores[direction] += confidence * weight
            total_confidence += confidence * weight
        
        final_direction = max(direction_scores, key=direction_scores.get)
        
        # Ajustar confianza basado en consistencia
        base_confidence = min(90, int(total_confidence))
        
        # Bonus por consistencia
        consistency_bonus = 0
        if phase_trends.get('consistency', False):
            consistency_bonus += 10
        if momentum_analysis.get('alignment', False):
            consistency_bonus += 8
        if segment_prediction.get('recent_alignment', False):
            consistency_bonus += 7
        if candle_pattern.get('continuation_bias', False):
            consistency_bonus += 5
        
        final_confidence = min(95, base_confidence + consistency_bonus)
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'base_confidence': base_confidence,
            'consistency_bonus': consistency_bonus
        }
    
    def _generate_prediction_reasons(self, phase_trends, momentum_analysis, segment_prediction, candle_pattern):
        """Genera razones detalladas para la predicción"""
        reasons = []
        
        # Razones de fase
        if phase_trends['strength'] > 60:
            reasons.append(f"Tendencia {phase_trends['direction']} en fases ({phase_trends['strength']:.0f}%)")
        
        # Razones de momentum
        if momentum_analysis['momentum_strength'] > 50:
            reasons.append(f"Momentum {momentum_analysis['momentum_direction']} fuerte")
        
        if momentum_analysis['pressure_strength'] > 60:
            reasons.append(f"Presión {momentum_analysis['pressure_direction']} dominante")
        
        # Razones de segmentos
        if segment_prediction['recent_alignment']:
            reasons.append("Alineación consistente en segmentos finales")
        
        # Razones de patrón
        if candle_pattern['type'] != "LIGERO":
            reasons.append(f"Patrón {candle_pattern['type']} {candle_pattern['direction']}")
        
        # Razón de consistencia
        if len([r for r in reasons if 'consist' in r.lower() or 'aline' in r.lower()]) >= 2:
            reasons.append("Alta consistencia en señales")
        
        if not reasons:
            reasons.append("Señales equilibradas - análisis conservador")
        
        return reasons
    
    def validate_prediction(self, actual_direction: str):
        """Valida la predicción contra el resultado real"""
        if not self.prediction_history:
            return None
            
        last_prediction = self.prediction_history[-1]
        predicted_direction = last_prediction['direction']
        
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
        else:
            self.performance_stats['current_streak'] = 0
        
        return {
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": is_correct,
            "current_streak": self.performance_stats['current_streak']
        }
    
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
            "today_signals": self.performance_stats['today_signals']
        }
    
    def reset(self):
        self.analyzer.reset()

# ------------------ DASHBOARD RESPONSIVO ------------------
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
                "is_last_5_seconds": False,
                "current_phase": "INICIAL"
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
        
        # Determinar fase actual
        if remaining_time > 45:
            current_phase = "FASE 1 (0-15s)"
        elif remaining_time > 25:
            current_phase = "FASE 2 (15-35s)"
        elif remaining_time > 5:
            current_phase = "FASE 3 (35-55s)"
        else:
            current_phase = "PREDICCIÓN (55-60s)"
        
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
            "is_last_5_seconds": is_last_5,
            "current_phase": current_phase
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

# ------------------ WEBSOCKET MANAGER ------------------
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

# ------------------ HTML INTERFAZ COMPLETA ------------------
HTML_RESPONSIVE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Delowyss AI V5.10 - Análisis Completo de Vela</title>
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
            <h1>🚀 Delowyss AI V5.10</h1>
            <div class="subtitle">Análisis Completo de Vela - Predicción en Tiempo Real</div>
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
                    <div style="font-size: 1.3em; margin: 10px 0;" id="predictionText">ANALIZANDO VELA ACTUAL...</div>
                    
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
                        <div id="phaseInfo" style="font-size: 0.8em; color: var(--text-dim);">Fase: INICIAL</div>
                    </div>
                    
                    <div id="priceInfo" style="font-size: 1.1em; margin-top: 10px; font-family: monospace;">
                        Precio: <span id="currentPrice">-</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📊 ANÁLISIS DE VELA</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Fase Actual</div>
                        <div class="metric-value" id="currentPhase">INICIAL</div>
                        <div class="metric-label" id="phaseProgress">0%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Ticks</div>
                        <div class="metric-value" id="tickCount">0</div>
                        <div class="metric-label">Procesados</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Volatilidad</div>
                        <div class="metric-value" id="volatilityValue">0.0</div>
                        <div class="metric-label">pips</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Presión</div>
                        <div class="metric-value" id="pressureValue">50%</div>
                        <div class="metric-label">Compra/Venta</div>
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
                        <div class="metric-label">Racha +</div>
                        <div class="metric-value" id="winStreak">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Señales</div>
                        <div class="metric-value" id="totalSignals">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Éxito</div>
                        <div class="metric-value" id="successRate">0%</div>
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
                <div class="card-title">📈 HISTORIAL DE PREDICCIONES</div>
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
                
                // Agregar al historial si es nueva predicción
                if (prediction.direction !== 'N/A' && prediction.confidence > 0) {
                    this.addHistoryItem(
                        `Predicción: ${prediction.direction} ${prediction.confidence}%`,
                        prediction.direction === 'ALZA' ? 'success' : prediction.direction === 'BAJA' ? 'error' : 'warning'
                    );
                }
            }
            
            updateCandleInfo(candle) {
                const progressElement = document.getElementById('candleProgress');
                const infoElement = document.getElementById('candleInfo');
                const phaseElement = document.getElementById('phaseInfo');
                const priceElement = document.getElementById('currentPrice');
                const timeElement = document.getElementById('timeRemaining');
                const last5Element = document.getElementById('last5Indicator');
                const currentPhaseElement = document.getElementById('currentPhase');
                const phaseProgressElement = document.getElementById('phaseProgress');
                const tickCountElement = document.getElementById('tickCount');
                const predictionCard = document.getElementById('predictionCard');
                
                progressElement.style.width = `${candle.progress}%`;
                infoElement.textContent = `Vela: ${60 - candle.time_remaining}/60s | Ticks: ${candle.ticks_processed}`;
                phaseElement.textContent = `Fase: ${candle.current_phase}`;
                priceElement.textContent = candle.price ? candle.price.toFixed(5) : '-';
                timeElement.textContent = candle.time_remaining.toFixed(1);
                currentPhaseElement.textContent = candle.current_phase;
                phaseProgressElement.textContent = `${candle.progress.toFixed(0)}%`;
                tickCountElement.textContent = candle.ticks_processed;
                
                if (candle.is_last_5_seconds) {
                    last5Element.style.display = 'inline';
                    predictionCard.classList.add('countdown-active');
                } else {
                    last5Element.style.display = 'none';
                    predictionCard.classList.remove('countdown-active');
                }
            }
            
            updateMetrics(metrics) {
                document.getElementById('pressureValue').textContent = `${metrics.density}%`;
                document.getElementById('pressureValue').className = `metric-value ${this.getPressureColor(metrics.density)}`;
                document.getElementById('volatilityValue').textContent = `${metrics.velocity}`;
            }
            
            updatePerformance(performance) {
                document.getElementById('accuracyValue').textContent = `${performance.today_accuracy}%`;
                document.getElementById('winStreak').textContent = performance.current_streak;
                document.getElementById('totalSignals').textContent = performance.total_signals;
                
                const successRate = performance.today_accuracy > 0 ? performance.today_accuracy : 0;
                document.getElementById('successRate').textContent = `${successRate}%`;
                document.getElementById('successRate').className = `metric-value ${this.getSuccessColor(successRate)}`;
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
            
            getPressureColor(pressure) {
                if (pressure >= 70) return 'green';
                if (pressure >= 60) return 'green-light';
                if (pressure <= 30) return 'red';
                if (pressure <= 40) return 'red-light';
                return 'yellow';
            }
            
            getSuccessColor(successRate) {
                if (successRate >= 70) return 'green';
                if (successRate >= 60) return 'green-light';
                if (successRate <= 40) return 'red';
                if (successRate <= 50) return 'red-light';
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

# ------------------ FASTAPI APP ------------------
app = FastAPI(
    title="Delowyss Trading AI V5.10 - Análisis Completo de Vela",
    description="Sistema de IA con análisis completo de vela actual para predecir siguiente vela",
    version="5.10.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ CONFIGURACIÓN RUTAS ------------------
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
    analysis = predictor.analyzer.get_candle_analysis()
    if analysis.get('status') == 'COMPLETE_ANALYSIS':
        prediction = predictor.predict_next_candle()
        return prediction
    return {"status": "ANALYZING", "message": "Analizando vela actual..."}

@app.get("/api/performance")
async def get_performance():
    stats = predictor.get_performance_stats()
    return {
        "performance": stats,
        "system_status": "CANDLE_ANALYSIS_ACTIVE",
        "timestamp": now_iso()
    }

@app.get("/api/analysis")
async def get_analysis():
    analysis = predictor.analyzer.get_candle_analysis()
    return {
        "analysis": analysis,
        "timestamp": now_iso()
    }

@app.get("/api/status")
async def get_status():
    return {
        "status": "operational",
        "version": "5.10.0",
        "pair": PAR,
        "timeframe": "1min",
        "iq_connected": iq_connector.connected,
        "current_price": iq_connector.current_price,
        "prediction_window": f"{PREDICTION_WINDOW}s",
        "timestamp": now_iso()
    }

# ------------------ SISTEMA PRINCIPAL ------------------
# Instancias globales
iq_connector = RealIQOptionConnector(IQ_EMAIL, IQ_PASSWORD, PAR)
predictor = NextCandlePredictor()
dashboard_manager = AdvancedConnectionManager()

# Variables globales
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False
_last_price = None

# ------------------ INICIALIZACIÓN ------------------
def start_system():
    try:
        logging.info("🔧 INICIANDO SISTEMA V5.10 - ANÁLISIS COMPLETO DE VELA")
        logging.info("🎯 SISTEMA DE PREDICCIÓN BASADO EN ANÁLISIS COMPLETO")
        
        # ✅ INICIAR CONEXIÓN IQ OPTION
        logging.info("🔄 Iniciando conexión a IQ Option...")
        connection_result = iq_connector.connect()
        logging.info(f"🔧 Resultado conexión IQ Option: {connection_result}")
        
        if connection_result:
            logging.info("✅ Conexión IQ Option exitosa al inicio")
            dashboard_manager.dashboard.update_system_status("CONNECTED", "OPERATIONAL", "SYNCED")
        else:
            logging.error("❌ Conexión IQ Option falló al inicio")
            dashboard_manager.dashboard.update_system_status("DISCONNECTED", "ERROR", "SYNCED")
        
        # ✅ INICIAR THREAD DE ANÁLISIS
        logging.info("🔧 Iniciando thread de análisis de vela...")
        trading_thread = threading.Thread(target=premium_candle_analysis_loop, daemon=True)
        trading_thread.start()
        logging.info("🔧 Thread de análisis de vela iniciado")
        
        logging.info(f"⭐ DELOWYSS AI V5.10 INICIADA - ANÁLISIS COMPLETO DE VELA")
        logging.info("🎯 PREDICCIÓN A 5s - ANÁLISIS COMPLETO DESDE INICIO DE VELA")
        logging.info("🌐 DASHBOARD DISPONIBLE EN: http://0.0.0.0:10000")
        
        time.sleep(2)
        logging.info(f"🔧 Threads activos: {threading.active_count()}")
        
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")
        import traceback
        logging.error(f"❌ Traceback: {traceback.format_exc()}")

# ------------------ LOOP PRINCIPAL ------------------
def premium_candle_analysis_loop():
    global _last_candle_start, _prediction_made_this_candle, _last_price
    
    logging.info(f"🚀 LOOP DE ANÁLISIS DE VELA COMPLETA INICIADO")
    
    # ✅ SINCRONIZAR METRÓNOMO
    try:
        logging.info("🔧 Sincronizando metrónomo...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(dashboard_manager.metronome.sync_with_iqoption(iq_connector))
        loop.close()
        logging.info("✅ Metrónomo sincronizado en loop de análisis")
    except Exception as e:
        logging.warning(f"⚠️ Error sincronizando metrónomo: {e}")
    
    while True:
        try:
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
            
            # Obtener precio actual
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price
                
                # Procesar tick para análisis
                predictor.process_tick(price, seconds_remaining)
                
                # Hacer predicción en los últimos 5 segundos si no se ha hecho
                if (dashboard_manager.metronome.is_prediction_time() and 
                    not _prediction_made_this_candle and
                    predictor.analyzer.is_ready_for_prediction()):
                    
                    logging.info(f"🎯 HACIENDO PREDICCIÓN A {seconds_remaining:.1f}s")
                    
                    prediction = predictor.predict_next_candle()
                    
                    # Actualizar dashboard
                    dashboard_manager.dashboard.update_prediction(
                        prediction['direction'],
                        prediction['confidence']
                    )
                    
                    # Actualizar métricas de performance
                    stats = predictor.get_performance_stats()
                    dashboard_manager.dashboard.update_performance(
                        stats['accuracy'],
                        0,  # profit no aplica en este sistema
                        stats['today_signals'],
                        stats['best_streak'],
                        stats['current_streak']
                    )
                    
                    _prediction_made_this_candle = True
                    logging.info(f"✅ PREDICCIÓN: {prediction['direction']} {prediction['confidence']}%")
            
            # Detectar nueva vela
            if current_candle_start > _last_candle_start:
                if _last_price is not None:
                    # Determinar dirección real de la vela que acaba de cerrar
                    if (predictor.analyzer.current_candle_close is not None and 
                        predictor.analyzer.current_candle_open is not None):
                        
                        price_change = predictor.analyzer.current_candle_close - predictor.analyzer.current_candle_open
                        actual_direction = "ALZA" if price_change > 0.00001 else "BAJA" if price_change < -0.00001 else "LATERAL"
                        
                        # Validar predicción
                        validation = predictor.validate_prediction(actual_direction)
                        if validation:
                            result_icon = '✅' if validation['correct'] else '❌'
                            logging.info(f"📊 VALIDACIÓN: Predicho {validation['predicted']} vs Real {validation['actual']} - {result_icon}")
                
                # Reiniciar para nueva vela
                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🕯️ NUEVA VELA - Análisis completo reiniciado")
            
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(1)

# ------------------ EJECUCIÓN PRINCIPAL ------------------
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
