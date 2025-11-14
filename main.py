# main.py - V7.0 ANÁLISIS COMPLETO DE VELA + PREDICCIÓN + AUTOAPRENDIZAJE OPTIMIZADO
"""
Delowyss Trading AI — V7.0 SISTEMA OPTIMIZADO CON ARQUITECTURA MEJORADA
CEO: Eduardo Solis — © 2025
Sistema de trading con IA avanzada y arquitectura optimizada
"""

import os
import time
import threading
import logging
import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ------------------ CONFIGURACIÓN INICIAL ------------------
# Configuración mejorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_ai.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("DelowyssAI")

# Importaciones condicionales
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn no disponible, usando modo básico")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI no disponible")

# ------------------ CONFIGURACIÓN CENTRALIZADA ------------------
class Config:
    """Configuración centralizada del sistema"""
    # Credenciales IQ Option
    IQ_EMAIL = os.getenv("IQ_EMAIL", "vozhechacancion1@gmail.com")
    IQ_PASSWORD = os.getenv("IQ_PASSWORD", "tu_password_real")
    
    # Parámetros de trading
    PAR = "EURUSD"
    TIMEFRAME = 60
    PREDICTION_WINDOW = 5
    MIN_TICKS_FOR_PREDICTION = 15
    TICK_BUFFER_SIZE = 200
    
    # Servidor
    PORT = int(os.getenv("PORT", "10000"))
    HOST = "0.0.0.0"
    
    # ML Configuration
    ML_TRAINING_INTERVAL = 300  # 5 minutos
    MIN_TRAINING_SAMPLES = 20
    MAX_TRAINING_SAMPLES = 2000
    
    # Risk Management
    MAX_CONSECUTIVE_LOSSES = 5
    MIN_CONFIDENCE_THRESHOLD = 60

# ------------------ ESTRUCTURAS DE DATOS ------------------
class Direction(Enum):
    """Direcciones de mercado estandarizadas"""
    UP = "ALZA"
    DOWN = "BAJA" 
    SIDEWAYS = "LATERAL"

@dataclass
class CandleData:
    """Estructura para datos de vela"""
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
    volume: int = 0

@dataclass
class PredictionResult:
    """Resultado de predicción estructurado"""
    direction: Direction
    confidence: float
    method: str
    features: Dict[str, float]
    timestamp: datetime
    debug_info: List[str]

@dataclass
class MarketContext:
    """Contexto de mercado para análisis"""
    volatility: float
    trend_strength: float
    market_regime: str
    volume_profile: Dict[str, float]
    support_resistance: Dict[str, float]

# ------------------ UTILIDADES MEJORADAS ------------------
class DateTimeUtils:
    """Utilidades para manejo de tiempo"""
    
    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat() + 'Z'
    
    @staticmethod
    def get_timestamp() -> float:
        return time.time()
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Formatea segundos a string MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

class DataValidator:
    """Validador de datos de mercado"""
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Valida que el precio sea razonable"""
        if price <= 0 or price > 1000:
            return False
        if np.isnan(price) or np.isinf(price):
            return False
        return True
    
    @staticmethod
    def validate_candle(candle: CandleData) -> bool:
        """Valida la integridad de una vela"""
        if candle.high < candle.low:
            return False
        if candle.open <= 0 or candle.close <= 0:
            return False
        if not (candle.low <= candle.open <= candle.high):
            return False
        if not (candle.low <= candle.close <= candle.high):
            return False
        return True

# ------------------ METRÓNOMO OPTIMIZADO ------------------
class EnhancedMetronome:
    """Metrónomo mejorado con sincronización precisa"""
    
    def __init__(self, timeframe: int = 60):
        self.timeframe = timeframe
        self.server_offset = 0
        self.last_sync = 0
        self.sync_interval = 30  # Sincronizar cada 30 segundos
        
    def sync_with_server(self, server_time_func) -> bool:
        """Sincroniza con el servidor de IQ Option"""
        try:
            server_time = server_time_func()
            if server_time:
                local_time = time.time()
                self.server_offset = server_time - local_time
                self.last_sync = local_time
                logger.info(f"Metrónomo sincronizado. Offset: {self.server_offset:.2f}s")
                return True
        except Exception as e:
            logger.error(f"Error sincronizando metrónomo: {e}")
        return False
    
    def get_remaining_time(self) -> float:
        """Calcula tiempo restante para la siguiente vela"""
        try:
            current_time = time.time() + self.server_offset
            remaining = self.timeframe - (current_time % self.timeframe)
            return max(0, min(self.timeframe, remaining))
        except:
            return self.timeframe - (time.time() % self.timeframe)
    
    def get_candle_progress(self) -> float:
        """Obtiene progreso de la vela actual en porcentaje"""
        remaining = self.get_remaining_time()
        return ((self.timeframe - remaining) / self.timeframe) * 100
    
    def is_prediction_time(self, prediction_window: int = 5) -> bool:
        """Determina si es momento de predecir"""
        remaining = self.get_remaining_time()
        return remaining <= prediction_window and remaining > 0

# ------------------ ANALIZADOR DE VELA OPTIMIZADO ------------------
class OptimizedCandleAnalyzer:
    """Analizador de vela optimizado con mejor rendimiento"""
    
    def __init__(self, buffer_size: int = 200):
        self.buffer_size = buffer_size
        self.ticks = deque(maxlen=buffer_size)
        self.current_candle = None
        self.previous_candle = None
        self.candle_start_time = None
        self.tick_count = 0
        
        # Métricas en tiempo real
        self.metrics = {
            'velocity': deque(maxlen=50),
            'acceleration': deque(maxlen=30),
            'volatility': deque(maxlen=20),
            'volume_profile': deque(maxlen=25)
        }
        
        # Análisis por fases temporales
        self.time_phases = {
            'initial': (0, 15),      # 0-15 segundos
            'development': (15, 35),  # 15-35 segundos  
            'consolidation': (35, 55), # 35-55 segundos
            'prediction': (55, 60)    # 55-60 segundos
        }
        
        self.phase_data = {phase: {} for phase in self.time_phases}
        
    def start_new_candle(self, initial_price: float):
        """Inicia una nueva vela"""
        self.current_candle = CandleData(
            open=initial_price,
            high=initial_price,
            low=initial_price,
            close=initial_price,
            timestamp=datetime.utcnow()
        )
        self.candle_start_time = time.time()
        self.tick_count = 0
        self.ticks.clear()
        self.phase_data = {phase: {} for phase in self.time_phases}
        logger.info("Nueva vela iniciada")
    
    def add_tick(self, price: float, timestamp: float = None) -> bool:
        """Procesa un nuevo tick"""
        if not DataValidator.validate_price(price):
            return False
            
        if self.current_candle is None:
            self.start_new_candle(price)
            return True
        
        # Actualizar datos de vela actual
        self.current_candle.high = max(self.current_candle.high, price)
        self.current_candle.low = min(self.current_candle.low, price)
        self.current_candle.close = price
        
        # Agregar tick al buffer
        tick_data = {
            'price': price,
            'timestamp': timestamp or time.time(),
            'candle_age': time.time() - self.candle_start_time
        }
        self.ticks.append(tick_data)
        self.tick_count += 1
        
        # Actualizar métricas en tiempo real
        self._update_realtime_metrics(tick_data)
        
        # Analizar fase actual
        self._analyze_current_phase(tick_data)
        
        return True
    
    def _update_realtime_metrics(self, tick_data: Dict):
        """Actualiza métricas en tiempo real"""
        if len(self.ticks) >= 2:
            # Calcular velocidad (cambio de precio por segundo)
            prev_tick = list(self.ticks)[-2]
            time_diff = tick_data['timestamp'] - prev_tick['timestamp']
            if time_diff > 0:
                price_diff = tick_data['price'] - prev_tick['price']
                velocity = price_diff / time_diff
                self.metrics['velocity'].append(velocity)
                
                # Calcular aceleración
                if len(self.metrics['velocity']) >= 2:
                    velocity_diff = self.metrics['velocity'][-1] - self.metrics['velocity'][-2]
                    acceleration = velocity_diff / time_diff if time_diff > 0 else 0
                    self.metrics['acceleration'].append(acceleration)
    
    def _analyze_current_phase(self, tick_data: Dict):
        """Analiza la fase temporal actual"""
        candle_age = tick_data['candle_age']
        
        for phase, (start, end) in self.time_phases.items():
            if start <= candle_age < end:
                if phase not in self.phase_data or not self.phase_data[phase]:
                    self.phase_data[phase] = {
                        'start_time': tick_data['timestamp'],
                        'ticks': [],
                        'price_range': (tick_data['price'], tick_data['price']),
                        'analysis': {}
                    }
                
                # Actualizar datos de la fase
                phase_info = self.phase_data[phase]
                phase_info['ticks'].append(tick_data)
                phase_info['price_range'] = (
                    min(phase_info['price_range'][0], tick_data['price']),
                    max(phase_info['price_range'][1], tick_data['price'])
                )
                
                # Realizar análisis de la fase cuando tenga suficientes datos
                if len(phase_info['ticks']) >= 5:
                    self._analyze_phase(phase)
                break
    
    def _analyze_phase(self, phase: str):
        """Analiza una fase específica"""
        phase_info = self.phase_data[phase]
        ticks = phase_info['ticks']
        
        if not ticks:
            return
            
        prices = [t['price'] for t in ticks]
        
        analysis = {
            'price_range': phase_info['price_range'],
            'volatility': (max(prices) - min(prices)) * 10000,
            'trend': self._calculate_trend(prices),
            'volume': len(ticks),
            'price_action': self._analyze_price_action(prices)
        }
        
        phase_info['analysis'] = analysis
    
    def _calculate_trend(self, prices: List[float]) -> Dict[str, Any]:
        """Calcula la tendencia de una serie de precios"""
        if len(prices) < 2:
            return {'direction': 'LATERAL', 'strength': 0}
        
        try:
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            if slope > 0.00001:
                direction = 'ALCISTA'
            elif slope < -0.00001:
                direction = 'BAJISTA'
            else:
                direction = 'LATERAL'
                
            strength = abs(slope) * 10000
            return {'direction': direction, 'strength': min(strength, 100)}
            
        except:
            return {'direction': 'LATERAL', 'strength': 0}
    
    def _analyze_price_action(self, prices: List[float]) -> Dict[str, Any]:
        """Analiza la acción del precio"""
        if len(prices) < 3:
            return {'momentum': 0, 'volatility': 0, 'pattern': 'INSUFFICIENT_DATA'}
        
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        positive_changes = sum(1 for change in price_changes if change > 0)
        
        return {
            'momentum': (positive_changes / len(price_changes)) * 100,
            'volatility': np.std(prices) * 10000 if len(prices) > 1 else 0,
            'pattern': self._identify_pattern(prices)
        }
    
    def _identify_pattern(self, prices: List[float]) -> str:
        """Identifica patrones simples de precio"""
        if len(prices) < 5:
            return 'INSUFFICIENT_DATA'
            
        first_half = prices[:len(prices)//2]
        second_half = prices[len(prices)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if second_avg > first_avg * 1.001:
            return 'UPTREND'
        elif second_avg < first_avg * 0.999:
            return 'DOWNTREND'
        else:
            return 'RANGING'
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Obtiene análisis completo de la vela actual"""
        if self.current_candle is None:
            return {'status': 'NO_CANDLE', 'message': 'No hay vela activa'}
        
        analysis = {
            'status': 'COMPLETE',
            'candle': {
                'open': self.current_candle.open,
                'high': self.current_candle.high,
                'low': self.current_candle.low,
                'close': self.current_candle.close,
                'body_size': abs(self.current_candle.close - self.current_candle.open) * 10000,
                'total_range': (self.current_candle.high - self.current_candle.low) * 10000,
                'direction': 'ALCISTA' if self.current_candle.close > self.current_candle.open else 'BAJISTA'
            },
            'metrics': {
                'tick_count': self.tick_count,
                'current_velocity': np.mean(list(self.metrics['velocity'])[-5:]) if self.metrics['velocity'] else 0,
                'current_acceleration': np.mean(list(self.metrics['acceleration'])[-5:]) if self.metrics['acceleration'] else 0,
            },
            'phases': self.phase_data,
            'timestamp': DateTimeUtils.now_iso()
        }
        
        return analysis
    
    def is_ready_for_prediction(self) -> bool:
        """Determina si el analizador está listo para predecir"""
        if self.tick_count < Config.MIN_TICKS_FOR_PREDICTION:
            return False
            
        # Verificar que tenemos análisis de al menos 3 fases
        analyzed_phases = sum(1 for phase in self.phase_data.values() if phase.get('analysis'))
        return analyzed_phases >= 3

# ------------------ SISTEMA ML OPTIMIZADO ------------------
class OptimizedMLSystem:
    """Sistema de machine learning optimizado"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self.training_data = deque(maxlen=Config.MAX_TRAINING_SAMPLES)
        self.feature_names = []
        self.last_training = 0
        self.performance_history = deque(maxlen=100)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo de ML"""
        if not SKLEARN_AVAILABLE:
            logger.warning("ML no disponible - modo básico activado")
            return
            
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1  # Usar todos los cores
            )
            logger.info("Modelo ML inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando modelo ML: {e}")
            self.model = None
    
    def prepare_features(self, candle_analysis: Dict, market_context: MarketContext) -> Dict[str, float]:
        """Prepara características para el modelo"""
        features = {}
        
        try:
            # Características de la vela
            candle = candle_analysis.get('candle', {})
            features['body_size'] = candle.get('body_size', 0)
            features['range_size'] = candle.get('total_range', 0)
            features['body_ratio'] = features['body_size'] / max(features['range_size'], 0.0001)
            
            # Características de fases
            phases = candle_analysis.get('phases', {})
            phase_strengths = []
            phase_directions = []
            
            for phase_name, phase_data in phases.items():
                analysis = phase_data.get('analysis', {})
                trend = analysis.get('trend', {})
                phase_strengths.append(trend.get('strength', 0))
                phase_directions.append(1 if trend.get('direction') == 'ALCISTA' else -1 if trend.get('direction') == 'BAJISTA' else 0)
            
            features['avg_trend_strength'] = np.mean(phase_strengths) if phase_strengths else 0
            features['trend_consistency'] = 1 - (np.std(phase_directions) / 2) if phase_directions else 0
            
            # Características de mercado
            features['market_volatility'] = market_context.volatility
            features['trend_strength'] = market_context.trend_strength
            
            # Características temporales
            current_time = datetime.utcnow()
            features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
            
            self.feature_names = list(features.keys())
            return features
            
        except Exception as e:
            logger.error(f"Error preparando características: {e}")
            return {}
    
    def add_training_sample(self, features: Dict, actual_direction: Direction, confidence: float):
        """Agrega muestra de entrenamiento"""
        if not features:
            return
            
        sample = {
            'features': list(features.values()),
            'label': 0 if actual_direction == Direction.UP else 1 if actual_direction == Direction.DOWN else 2,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        self.training_data.append(sample)
        logger.debug(f"Muestra de entrenamiento agregada. Total: {len(self.training_data)}")
    
    def train(self) -> bool:
        """Entrena el modelo"""
        if not SKLEARN_AVAILABLE or self.model is None:
            return False
            
        if len(self.training_data) < Config.MIN_TRAINING_SAMPLES:
            logger.warning(f"Muestras insuficientes para entrenar: {len(self.training_data)}")
            return False
        
        try:
            # Preparar datos
            X = np.array([sample['features'] for sample in self.training_data])
            y = np.array([sample['label'] for sample in self.training_data])
            
            # Validar datos
            if len(np.unique(y)) < 2:
                logger.warning("Se necesitan al menos 2 clases para entrenar")
                return False
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluar
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.performance_history.append(self.accuracy)
            
            self.is_trained = True
            self.last_training = time.time()
            
            logger.info(f"Modelo entrenado - Accuracy: {self.accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            return False
    
    def predict(self, features: Dict) -> Tuple[Optional[Direction], float]:
        """Realiza predicción"""
        if not self.is_trained or not features:
            return None, 0.0
            
        try:
            X = np.array([list(features.values())])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = np.max(probabilities) * 100
            
            direction_map = {
                0: Direction.UP,
                1: Direction.DOWN, 
                2: Direction.SIDEWAYS
            }
            
            predicted_direction = direction_map.get(prediction, Direction.SIDEWAYS)
            return predicted_direction, confidence
            
        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return None, 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema ML"""
        return {
            'is_trained': self.is_trained,
            'accuracy': self.accuracy,
            'training_samples': len(self.training_data),
            'last_training': datetime.fromtimestamp(self.last_training).isoformat() if self.last_training > 0 else 'Nunca',
            'performance_trend': list(self.performance_history)[-10:] if self.performance_history else []
        }

# ------------------ PREDICTOR PRINCIPAL OPTIMIZADO ------------------
class AdvancedPredictor:
    """Predictor principal con lógica mejorada"""
    
    def __init__(self):
        self.analyzer = OptimizedCandleAnalyzer()
        self.ml_system = OptimizedMLSystem()
        self.metronome = EnhancedMetronome(Config.TIMEFRAME)
        self.performance_tracker = PerformanceTracker()
        
        # Estado del predictor
        self.last_prediction = None
        self.prediction_history = deque(maxlen=100)
        self.market_context = MarketContext(0, 0, "NORMAL", {}, {})
        
        logger.info("Predictor avanzado inicializado")
    
    def process_tick(self, price: float) -> bool:
        """Procesa un nuevo tick de precio"""
        return self.analyzer.add_tick(price)
    
    def predict(self) -> PredictionResult:
        """Genera predicción para la siguiente vela"""
        analysis = self.analyzer.get_comprehensive_analysis()
        
        if not self.analyzer.is_ready_for_prediction():
            return self._get_default_prediction("ANALYSIS_INCOMPLETE")
        
        try:
            # Actualizar contexto de mercado
            self._update_market_context(analysis)
            
            # Preparar características
            features = self.ml_system.prepare_features(analysis, self.market_context)
            
            # Generar predicción con ML si está disponible
            ml_direction, ml_confidence = self.ml_system.predict(features)
            
            if ml_direction and ml_confidence >= Config.MIN_CONFIDENCE_THRESHOLD:
                prediction = PredictionResult(
                    direction=ml_direction,
                    confidence=ml_confidence,
                    method="ML_MODEL",
                    features=features,
                    timestamp=datetime.utcnow(),
                    debug_info=[f"ML Prediction - Confidence: {ml_confidence:.1f}%"]
                )
            else:
                # Usar análisis técnico tradicional
                prediction = self._get_technical_prediction(analysis, features)
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            logger.info(f"Predicción generada: {prediction.direction.value} - {prediction.confidence:.1f}%")
            return prediction
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return self._get_default_prediction(f"ERROR: {str(e)}")
    
    def _get_technical_prediction(self, analysis: Dict, features: Dict) -> PredictionResult:
        """Predicción basada en análisis técnico"""
        candle = analysis['candle']
        phases = analysis['phases']
        
        # Análisis de tendencia por fases
        phase_directions = []
        for phase_name, phase_data in phases.items():
            trend = phase_data.get('analysis', {}).get('trend', {})
            if trend:
                direction = trend.get('direction', 'LATERAL')
                phase_directions.append(1 if direction == 'ALCISTA' else -1 if direction == 'BAJISTA' else 0)
        
        if phase_directions:
            avg_direction = np.mean(phase_directions)
            if avg_direction > 0.1:
                direction = Direction.UP
                confidence = min(80, abs(avg_direction) * 100)
            elif avg_direction < -0.1:
                direction = Direction.DOWN
                confidence = min(80, abs(avg_direction) * 100)
            else:
                direction = Direction.SIDEWAYS
                confidence = 50
        else:
            direction = Direction.SIDEWAYS
            confidence = 50
        
        return PredictionResult(
            direction=direction,
            confidence=confidence,
            method="TECHNICAL_ANALYSIS",
            features=features,
            timestamp=datetime.utcnow(),
            debug_info=[f"Technical Analysis - Phase Directions: {phase_directions}"]
        )
    
    def _get_default_prediction(self, reason: str) -> PredictionResult:
        """Predicción por defecto cuando no hay datos suficientes"""
        return PredictionResult(
            direction=Direction.SIDEWAYS,
            confidence=50.0,
            method="DEFAULT",
            features={},
            timestamp=datetime.utcnow(),
            debug_info=[f"Default prediction: {reason}"]
        )
    
    def _update_market_context(self, analysis: Dict):
        """Actualiza el contexto de mercado"""
        try:
            candle = analysis['candle']
            volatility = candle['total_range']
            
            # Calcular fuerza de tendencia basada en fases
            trend_strengths = []
            for phase_data in analysis['phases'].values():
                trend = phase_data.get('analysis', {}).get('trend', {})
                trend_strengths.append(trend.get('strength', 0))
            
            avg_trend_strength = np.mean(trend_strengths) if trend_strengths else 0
            
            # Determinar régimen de mercado
            if volatility > 20:
                regime = "HIGH_VOLATILITY"
            elif volatility < 5:
                regime = "LOW_VOLATILITY"
            else:
                regime = "NORMAL"
            
            self.market_context = MarketContext(
                volatility=volatility,
                trend_strength=avg_trend_strength,
                market_regime=regime,
                volume_profile={},
                support_resistance={}
            )
            
        except Exception as e:
            logger.error(f"Error actualizando contexto de mercado: {e}")
    
    def validate_prediction(self, actual_direction: Direction) -> bool:
        """Valida la última predicción contra el resultado real"""
        if not self.last_prediction:
            return False
            
        is_correct = (self.last_prediction.direction == actual_direction and
                     actual_direction != Direction.SIDEWAYS)
        
        self.performance_tracker.record_prediction(
            self.last_prediction.direction,
            actual_direction,
            self.last_prediction.confidence,
            is_correct
        )
        
        # Agregar muestra de entrenamiento si tenemos características
        if self.last_prediction.features:
            self.ml_system.add_training_sample(
                self.last_prediction.features,
                actual_direction,
                self.last_prediction.confidence
            )
        
        logger.info(f"Validación: {'✅' if is_correct else '❌'} "
                   f"({self.last_prediction.direction.value} vs {actual_direction.value})")
        
        return is_correct
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento"""
        return self.performance_tracker.get_stats()
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema ML"""
        return self.ml_system.get_status()

# ------------------ SEGUIMIENTO DE RENDIMIENTO ------------------
class PerformanceTracker:
    """Seguimiento de rendimiento del predictor"""
    
    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        self.current_streak = 0
        self.best_streak = 0
        self.prediction_history = deque(maxlen=500)
        self.daily_stats = {}
        
    def record_prediction(self, predicted: Direction, actual: Direction, 
                         confidence: float, is_correct: bool):
        """Registra una predicción"""
        self.total_predictions += 1
        
        if is_correct:
            self.correct_predictions += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0
        
        record = {
            'timestamp': datetime.utcnow(),
            'predicted': predicted.value,
            'actual': actual.value,
            'confidence': confidence,
            'correct': is_correct,
            'streak': self.current_streak
        }
        
        self.prediction_history.append(record)
        
        # Actualizar estadísticas diarias
        today = datetime.utcnow().date().isoformat()
        if today not in self.daily_stats:
            self.daily_stats[today] = {'total': 0, 'correct': 0}
        
        self.daily_stats[today]['total'] += 1
        if is_correct:
            self.daily_stats[today]['correct'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento"""
        accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        # Estadísticas de hoy
        today = datetime.utcnow().date().isoformat()
        today_stats = self.daily_stats.get(today, {'total': 0, 'correct': 0})
        today_accuracy = (today_stats['correct'] / today_stats['total'] * 100) if today_stats['total'] > 0 else 0
        
        return {
            'overall_accuracy': round(accuracy, 1),
            'today_accuracy': round(today_accuracy, 1),
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'current_streak': self.current_streak,
            'best_streak': self.best_streak,
            'today_signals': today_stats['total'],
            'total_signals': self.total_predictions
        }

# ------------------ CONEXIÓN IQ OPTION MEJORADA ------------------
class EnhancedIQConnector:
    """Conector mejorado para IQ Option"""
    
    def __init__(self, email: str, password: str, pair: str = "EURUSD"):
        self.email = email
        self.password = password
        self.pair = pair
        self.api = None
        self.connected = False
        self.current_price = None
        self.connection_attempts = 0
        self.max_attempts = 3
        
    def connect(self) -> bool:
        """Establece conexión con IQ Option"""
        try:
            from iqoptionapi.stable_api import IQ_Option
            
            self.api = IQ_Option(self.email, self.password)
            check, reason = self.api.connect()
            
            if check:
                self.connected = True
                self.api.change_balance("PRACTICE")
                self.api.start_candles_stream(self.pair, Config.TIMEFRAME, 1)
                
                # Obtener precio inicial
                self._get_initial_price()
                
                logger.info(f"Conexión IQ Option exitosa - Par: {self.pair}")
                return True
            else:
                logger.error(f"Error conectando a IQ Option: {reason}")
                return False
                
        except Exception as e:
            logger.error(f"Excepción en conexión IQ Option: {e}")
            return False
    
    def _get_initial_price(self):
        """Obtiene el precio inicial"""
        try:
            candles = self.api.get_realtime_candles(self.pair, Config.TIMEFRAME)
            if candles:
                for candle_id in candles:
                    candle = candles[candle_id]
                    if 'close' in candle:
                        self.current_price = candle['close']
                        logger.info(f"Precio inicial {self.pair}: {self.current_price}")
                        break
        except Exception as e:
            logger.error(f"Error obteniendo precio inicial: {e}")
    
    def get_price(self) -> Optional[float]:
        """Obtiene el precio actual"""
        if not self.connected:
            return None
            
        try:
            candles = self.api.get_realtime_candles(self.pair, Config.TIMEFRAME)
            if candles:
                for candle_id in candles:
                    candle = candles[candle_id]
                    if 'close' in candle:
                        price = candle['close']
                        if DataValidator.validate_price(price):
                            self.current_price = price
                            return price
            return self.current_price
            
        except Exception as e:
            logger.error(f"Error obteniendo precio: {e}")
            return self.current_price
    
    def get_server_time(self) -> Optional[float]:
        """Obtiene tiempo del servidor"""
        if not self.connected:
            return None
            
        try:
            return self.api.get_server_timestamp()
        except:
            return None

# ------------------ SISTEMA PRINCIPAL OPTIMIZADO ------------------
class TradingAISystem:
    """Sistema principal de Trading AI"""
    
    def __init__(self):
        self.iq_connector = EnhancedIQConnector(Config.IQ_EMAIL, Config.IQ_PASSWORD, Config.PAR)
        self.predictor = AdvancedPredictor()
        self.metronome = EnhancedMetronome(Config.TIMEFRAME)
        self.is_running = False
        
        # Estado del sistema
        self.last_candle_start = 0
        self.prediction_made = False
        
    def start(self):
        """Inicia el sistema de trading"""
        logger.info("Iniciando Delowyss Trading AI V7.0")
        
        # Conectar a IQ Option
        if not self.iq_connector.connect():
            logger.error("No se pudo conectar a IQ Option. Sistema en modo simulación.")
        
        # Sincronizar metrónomo
        self.metronome.sync_with_server(self.iq_connector.get_server_time)
        
        self.is_running = True
        
        # Iniciar hilos
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()
        
        ml_thread = threading.Thread(target=self._ml_training_loop, daemon=True)
        ml_thread.start()
        
        logger.info("Sistema de trading iniciado correctamente")
    
    def _trading_loop(self):
        """Loop principal de trading"""
        logger.info("Loop de trading iniciado")
        
        while self.is_running:
            try:
                # Obtener precio actual
                price = self.iq_connector.get_price()
                if price is None:
                    time.sleep(1)
                    continue
                
                # Procesar tick
                self.predictor.process_tick(price)
                
                # Verificar si es momento de predecir
                current_time = time.time()
                current_candle_start = int(current_time // Config.TIMEFRAME * Config.TIMEFRAME)
                
                # Nueva vela detectada
                if current_candle_start > self.last_candle_start:
                    self._handle_new_candle()
                    self.last_candle_start = current_candle_start
                    self.prediction_made = False
                
                # Momento de predicción
                if (self.metronome.is_prediction_time(Config.PREDICTION_WINDOW) and 
                    not self.prediction_made and
                    self.predictor.analyzer.is_ready_for_prediction()):
                    
                    prediction = self.predictor.predict()
                    self.prediction_made = True
                    
                    logger.info(f"🎯 PREDICCIÓN: {prediction.direction.value} "
                               f"{prediction.confidence:.1f}% - {prediction.method}")
                
                time.sleep(0.1)  # Control de frecuencia
                
            except Exception as e:
                logger.error(f"Error en loop de trading: {e}")
                time.sleep(1)
    
    def _handle_new_candle(self):
        """Maneja el inicio de una nueva vela"""
        logger.info("🕯️ Nueva vela detectada")
        
        # Validar predicción anterior si existe
        if (self.predictor.last_prediction and 
            self.predictor.analyzer.current_candle is not None):
            
            actual_direction = Direction.UP if (
                self.predictor.analyzer.current_candle.close > 
                self.predictor.analyzer.current_candle.open
            ) else Direction.DOWN if (
                self.predictor.analyzer.current_candle.close < 
                self.predictor.analyzer.current_candle.open
            ) else Direction.SIDEWAYS
            
            self.predictor.validate_prediction(actual_direction)
    
    def _ml_training_loop(self):
        """Loop de entrenamiento ML automático"""
        logger.info("Loop de entrenamiento ML iniciado")
        
        while self.is_running:
            try:
                # Entrenar cada 5 minutos si hay suficientes muestras
                ml_status = self.predictor.ml_system.get_status()
                if (ml_status['training_samples'] >= Config.MIN_TRAINING_SAMPLES and
                    time.time() - self.predictor.ml_system.last_training > Config.ML_TRAINING_INTERVAL):
                    
                    logger.info("Iniciando entrenamiento automático ML...")
                    self.predictor.ml_system.train()
                
                time.sleep(60)  # Revisar cada minuto
                
            except Exception as e:
                logger.error(f"Error en loop de entrenamiento: {e}")
                time.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema"""
        return {
            'version': '7.0.0',
            'status': 'RUNNING' if self.is_running else 'STOPPED',
            'iq_connected': self.iq_connector.connected,
            'current_price': self.iq_connector.current_price,
            'performance': self.predictor.get_performance_stats(),
            'ml_status': self.predictor.get_ml_status(),
            'candle_progress': self.metronome.get_candle_progress(),
            'time_remaining': self.metronome.get_remaining_time(),
            'timestamp': DateTimeUtils.now_iso()
        }

# ------------------ CREACIÓN DE LA APLICACIÓN FASTAPI ------------------
# Crear la aplicación FastAPI
app = FastAPI(
    title="Delowyss Trading AI V7.0",
    description="Sistema de trading con IA avanzada - Versión Optimizada",
    version="7.0.0"
)

# Configuración CORS para Render.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del sistema de trading
trading_system = TradingAISystem()

# ------------------ RUTAS DE LA API ------------------
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Dashboard principal"""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss Trading AI V7.0</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                color: #ffffff;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.1);
            }
            .header h1 {
                color: #00ff88;
                margin: 0;
                font-size: 2.5em;
            }
            .header .subtitle {
                color: #888;
                font-size: 1.1em;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,255,136,0.1);
            }
            .card h2 {
                color: #00ff88;
                margin-top: 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                padding-bottom: 10px;
            }
            .stat {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }
            .stat .value {
                font-weight: bold;
                color: #00ff88;
            }
            .prediction-card {
                background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
                color: #000;
                grid-column: 1 / -1;
            }
            .prediction-card h2 {
                color: #000;
            }
            .prediction-card .value {
                color: #000;
                font-weight: 800;
            }
            .countdown {
                font-size: 2em;
                text-align: center;
                font-weight: bold;
                margin: 20px 0;
                color: #000;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-connected { background: #00ff88; }
            .status-disconnected { background: #ff4444; }
            .status-synced { background: #00ff88; }
            .status-unsynced { background: #ffaa00; }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            .pulse {
                animation: pulse 2s infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Delowyss Trading AI V7.0</h1>
                <div class="subtitle">Sistema de Trading con IA Avanzada - Versión Optimizada</div>
            </div>
            
            <div class="grid">
                <div class="card prediction-card" id="predictionCard">
                    <h2>🎯 PREDICCIÓN ACTUAL</h2>
                    <div class="countdown" id="countdown">--</div>
                    <div class="stat">
                        <span>Dirección:</span>
                        <span class="value" id="predictionDirection">ANALIZANDO...</span>
                    </div>
                    <div class="stat">
                        <span>Confianza:</span>
                        <span class="value" id="predictionConfidence">0%</span>
                    </div>
                    <div class="stat">
                        <span>Método:</span>
                        <span class="value" id="predictionMethod">TRADICIONAL</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📊 ESTADO DEL SISTEMA</h2>
                    <div class="stat">
                        <span>IQ Option:</span>
                        <span class="value">
                            <span class="status-indicator" id="iqStatus"></span>
                            <span id="iqStatusText">CONECTANDO...</span>
                        </span>
                    </div>
                    <div class="stat">
                        <span>Precio Actual:</span>
                        <span class="value" id="currentPrice">0.00000</span>
                    </div>
                    <div class="stat">
                        <span>Progreso Vela:</span>
                        <span class="value" id="candleProgress">0%</span>
                    </div>
                    <div class="stat">
                        <span>Tiempo Restante:</span>
                        <span class="value" id="timeRemaining">60s</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🏆 RENDIMIENTO</h2>
                    <div class="stat">
                        <span>Precisión Total:</span>
                        <span class="value" id="overallAccuracy">0%</span>
                    </div>
                    <div class="stat">
                        <span>Precisión Hoy:</span>
                        <span class="value" id="todayAccuracy">0%</span>
                    </div>
                    <div class="stat">
                        <span>Racha Actual:</span>
                        <span class="value" id="currentStreak">0</span>
                    </div>
                    <div class="stat">
                        <span>Mejor Racha:</span>
                        <span class="value" id="bestStreak">0</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🧠 AUTOAPRENDIZAJE</h2>
                    <div class="stat">
                        <span>Modelo Entrenado:</span>
                        <span class="value" id="modelTrained">NO</span>
                    </div>
                    <div class="stat">
                        <span>Accuracy Modelo:</span>
                        <span class="value" id="modelAccuracy">0%</span>
                    </div>
                    <div class="stat">
                        <span>Muestras:</span>
                        <span class="value" id="trainingSamples">0</span>
                    </div>
                    <div class="stat">
                        <span>Último Entrenamiento:</span>
                        <span class="value" id="lastTraining">NUNCA</span>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function updateDashboard() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    // Actualizar predicción
                    document.getElementById('predictionDirection').textContent = 
                        data.last_prediction?.direction || 'ANALIZANDO';
                    document.getElementById('predictionConfidence').textContent = 
                        data.last_prediction ? data.last_prediction.confidence + '%' : '0%';
                    document.getElementById('predictionMethod').textContent = 
                        data.last_prediction?.method || 'TRADICIONAL';
                    
                    // Actualizar estado del sistema
                    document.getElementById('iqStatus').className = 
                        `status-indicator ${data.iq_connected ? 'status-connected' : 'status-disconnected'}`;
                    document.getElementById('iqStatusText').textContent = 
                        data.iq_connected ? 'CONECTADO' : 'DESCONECTADO';
                    document.getElementById('currentPrice').textContent = 
                        data.current_price ? data.current_price.toFixed(5) : '0.00000';
                    document.getElementById('candleProgress').textContent = 
                        data.candle_progress ? data.candle_progress.toFixed(1) + '%' : '0%';
                    document.getElementById('timeRemaining').textContent = 
                        data.time_remaining ? Math.round(data.time_remaining) + 's' : '60s';
                    
                    // Actualizar rendimiento
                    document.getElementById('overallAccuracy').textContent = 
                        data.performance.overall_accuracy + '%';
                    document.getElementById('todayAccuracy').textContent = 
                        data.performance.today_accuracy + '%';
                    document.getElementById('currentStreak').textContent = 
                        data.performance.current_streak;
                    document.getElementById('bestStreak').textContent = 
                        data.performance.best_streak;
                    
                    // Actualizar autoaprendizaje
                    document.getElementById('modelTrained').textContent = 
                        data.ml_status.is_trained ? 'SI' : 'NO';
                    document.getElementById('modelAccuracy').textContent = 
                        (data.ml_status.accuracy * 100).toFixed(1) + '%';
                    document.getElementById('trainingSamples').textContent = 
                        data.ml_status.training_samples;
                    document.getElementById('lastTraining').textContent = 
                        data.ml_status.last_training;
                    
                    // Efectos visuales
                    const predictionCard = document.getElementById('predictionCard');
                    if (data.last_prediction?.confidence > 70) {
                        predictionCard.classList.add('pulse');
                    } else {
                        predictionCard.classList.remove('pulse');
                    }
                    
                } catch (error) {
                    console.error('Error actualizando dashboard:', error);
                }
            }
            
            // Actualizar cada 2 segundos
            setInterval(updateDashboard, 2000);
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_status():
    """Obtiene el estado completo del sistema"""
    status = trading_system.get_system_status()
    
    # Agregar última predicción si existe
    if trading_system.predictor.last_prediction:
        status['last_prediction'] = {
            'direction': trading_system.predictor.last_prediction.direction.value,
            'confidence': trading_system.predictor.last_prediction.confidence,
            'method': trading_system.predictor.last_prediction.method,
            'timestamp': trading_system.predictor.last_prediction.timestamp.isoformat()
        }
    
    return JSONResponse(status)

@app.get("/api/prediction")
async def get_prediction():
    """Obtiene una nueva predicción"""
    prediction = trading_system.predictor.predict()
    return JSONResponse({
        'direction': prediction.direction.value,
        'confidence': prediction.confidence,
        'method': prediction.method,
        'timestamp': prediction.timestamp.isoformat()
    })

@app.get("/api/performance")
async def get_performance():
    """Obtiene estadísticas de rendimiento"""
    return JSONResponse(trading_system.predictor.get_performance_stats())

@app.get("/api/ml-status")
async def get_ml_status():
    """Obtiene estado del sistema ML"""
    return JSONResponse(trading_system.predictor.get_ml_status())

@app.post("/api/retrain")
async def retrain_model():
    """Fuerza el reentrenamiento del modelo ML"""
    success = trading_system.predictor.ml_system.train()
    return JSONResponse({
        'success': success,
        'message': 'Modelo reentrenado exitosamente' if success else 'Error en el reentrenamiento'
    })

@app.get("/api/analysis")
async def get_analysis():
    """Obtiene análisis completo de la vela actual"""
    analysis = trading_system.predictor.analyzer.get_comprehensive_analysis()
    return JSONResponse(analysis)

# ------------------ INICIALIZACIÓN DEL SISTEMA ------------------
@app.on_event("startup")
async def startup_event():
    """Inicia el sistema cuando la aplicación FastAPI arranca"""
    logger.info("Iniciando Delowyss Trading AI V7.0 en el evento startup...")
    trading_system.start()

# ------------------ EJECUCIÓN PRINCIPAL ------------------
if __name__ == "__main__":
    import uvicorn
    
    # Iniciar el servidor
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,  # Desactivar reload en producción
        log_level="info"
    )
