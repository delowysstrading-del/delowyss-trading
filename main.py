# main.py - V5.4 PREMIUM ULTRA RESPONSIVE
"""
Delowyss Trading AI — V5.4 PREMIUM ULTRA RESPONSIVE
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Gestión elegante de dependencias opcionales
try:
    from iqoptionapi.stable_api import IQ_Option
    IQ_OPTION_AVAILABLE = True
except ImportError:
    IQ_Option = None
    IQ_OPTION_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN PREMIUM OPTIMIZADA ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = "EURUSD"
TIMEFRAME = int(os.getenv("TIMEFRAME", "60"))
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "5"))
MIN_TICKS_FOR_PREDICTION = int(os.getenv("MIN_TICKS_FOR_PREDICTION", "15"))  # ⬇ Reducido
TICK_BUFFER_SIZE = int(os.getenv("TICK_BUFFER_SIZE", "300"))  # ⬇ Optimizado
PORT = int(os.getenv("PORT", "10000"))

# Model paths
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
ONLINE_MODEL_PATH = os.path.join(MODEL_DIR, "online_sgd.pkl")
ONLINE_SCALER_PATH = os.path.join(MODEL_DIR, "online_scaler.pkl")

# ---------------- THREAD POOL PARA PROCESAMIENTO PARALELO ----------------
ML_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml_processor")
TRAINING_THREAD_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")

# ---------------- LOGGING OPTIMIZADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA COMPLETA OPTIMIZADA ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=80)  # ⬇ Optimizado
        self.last_candle_close = None
        
        # Métricas avanzadas OPTIMIZADAS
        self.velocity_metrics = deque(maxlen=30)  # ⬇ Reducido
        self.acceleration_metrics = deque(maxlen=20)
        self.volume_profile = deque(maxlen=15)
        self.price_levels = deque(maxlen=10)
        
        # Estados del análisis ORIGINAL pero optimizado
        self.candle_start_time = None
        self.analysis_phases = {
            'initial': {'ticks': 0, 'analysis': {}},
            'middle': {'ticks': 0, 'analysis': {}},
            'final': {'ticks': 0, 'analysis': {}}
        }
        
        # Cache para cálculos frecuentes
        self._cached_analysis = None
        self._last_analysis_time = 0
        self._analysis_cache_ttl = 0.5  # Cache de 500ms
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            # Inicializar vela si es el primer tick
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                # Log solo en primer tick para evitar spam
                if self.tick_count == 0:
                    logging.info("🕯️ Nueva vela iniciada - Análisis tick-by-tick activo")
            
            # Actualizar precios extremos (OPERACIÓN RÁPIDA)
            if price > self.current_candle_high:
                self.current_candle_high = price
            if price < self.current_candle_low:
                self.current_candle_low = price
            self.current_candle_close = price
            
            tick_data = {
                'price': price,
                'timestamp': current_time,
                'volume': 1,
                'microtimestamp': current_time * 1000,
                'seconds_remaining': seconds_remaining,
                'candle_age': current_time - self.candle_start_time if self.candle_start_time else 0
            }
            
            # Almacenar tick (OPERACIÓN ATÓMICA)
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            # Calcular métricas en tiempo real (NO BLOQUEANTE)
            self._calculate_comprehensive_metrics(tick_data)
            
            # Análisis por fases (OPTIMIZADO)
            self._analyze_candle_phase(tick_data)
            
            # Invalidar cache
            self._cached_analysis = None
            
            return tick_data
        except Exception as e:
            logging.error(f"Error en add_tick: {e}")
            return None
    
    def _calculate_comprehensive_metrics(self, current_tick):
        """Métricas avanzadas OPTIMIZADAS para velocidad"""
        if len(self.ticks) < 2:
            return
            
        try:
            current_price = current_tick['price']
            current_time = current_tick['timestamp']
            
            # Velocidad del precio (CÁLCULO RÁPIDO)
            previous_tick = self.ticks[-2]  # Acceso directo
            time_diff = current_time - previous_tick['timestamp']
            if time_diff > 0:
                price_diff = current_price - previous_tick['price']
                velocity = price_diff / time_diff
                
                self.velocity_metrics.append({
                    'velocity': velocity,
                    'timestamp': current_time,
                    'price_change': price_diff
                })
            
            # Aceleración (CÁLCULO CONDICIONAL)
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
            
            # Perfil de volumen (CÁLCULO CADA 5 TICKS)
            if len(self.ticks) >= 10 and len(self.ticks) % 5 == 0:
                recent_ticks = list(self.ticks)[-10:]
                price_changes = [tick['price'] for tick in recent_ticks]
                if price_changes:
                    avg_price = np.mean(price_changes)
                    self.volume_profile.append({
                        'avg_price': avg_price,
                        'tick_count': len(recent_ticks),
                        'timestamp': current_time
                    })
            
            # Identificar niveles de precio (CÁLCULO CADA 10 TICKS)
            if len(self.price_memory) >= 15 and len(self.price_memory) % 10 == 0:
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
        """Análisis por fases OPTIMIZADO"""
        candle_age = tick_data['candle_age']
        
        # Solo analizar cada 5 ticks para reducir carga
        if self.tick_count % 5 != 0:
            return
            
        if candle_age < 20:
            self.analysis_phases['initial']['ticks'] += 1
            if self.analysis_phases['initial']['ticks'] % 15 == 0:  # ⬆ Menos frecuente
                self.analysis_phases['initial']['analysis'] = self._get_phase_analysis('initial')
                
        elif candle_age < 40:
            self.analysis_phases['middle']['ticks'] += 1
            if self.analysis_phases['middle']['ticks'] % 15 == 0:
                self.analysis_phases['middle']['analysis'] = self._get_phase_analysis('middle')
                
        else:
            self.analysis_phases['final']['ticks'] += 1
            if self.analysis_phases['final']['ticks'] % 8 == 0:  # ⬆ Optimizado
                self.analysis_phases['final']['analysis'] = self._get_phase_analysis('final')
    
    def _get_phase_analysis(self, phase):
        """Análisis específico por fase OPTIMIZADO"""
        try:
            if phase == 'initial':
                ticks = list(self.ticks)[:20] if len(self.ticks) >= 20 else list(self.ticks)
            elif phase == 'middle':
                ticks = list(self.ticks)[20:40] if len(self.ticks) >= 40 else list(self.ticks)[20:]
            else:
                ticks = list(self.ticks)[40:] if len(self.ticks) >= 40 else []
            
            if not ticks:
                return {}
            
            # CÁLCULOS VECTORIZADOS RÁPIDOS
            prices = np.array([tick['price'] for tick in ticks])
            price_changes = np.diff(prices)
            
            return {
                'avg_price': float(np.mean(prices)),
                'volatility': float(np.max(prices) - np.min(prices)),
                'trend': 'ALCISTA' if prices[-1] > prices[0] else 'BAJISTA' if prices[-1] < prices[0] else 'LATERAL',
                'buy_pressure': float(np.sum(price_changes > 0) / len(price_changes)) if len(price_changes) > 0 else 0.5,
                'tick_count': len(ticks)
            }
        except Exception as e:
            logging.debug(f"Error en análisis de fase {phase}: {e}")
            return {}
    
    def _calculate_advanced_metrics(self):
        """Métricas avanzadas OPTIMIZADAS con cache"""
        current_time = time.time()
        
        # Usar cache si está disponible y es reciente
        if (self._cached_analysis and 
            current_time - self._last_analysis_time < self._analysis_cache_ttl):
            return self._cached_analysis
            
        if len(self.price_memory) < 8:  # ⬇ Umbral más bajo
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            # Análisis de tendencia OPTIMIZADO
            trend_strength = self._calculate_trend_strength(prices)
            
            # Momentum multi-temporal OPTIMIZADO
            momentum = self._calculate_momentum(prices)
            
            # Volatilidad segmentada RÁPIDA
            volatility = self._calculate_volatility(prices)
            
            # Presión de compra/venta OPTIMIZADA
            buy_pressure, sell_pressure, pressure_ratio = self._calculate_pressure()
            
            # Velocidad promedio RÁPIDA
            avg_velocity = self._calculate_velocity()
            
            # Análisis de fases combinado
            phase_analysis = self._combine_phase_analysis()
            
            # Determinar fase de mercado
            market_phase = self._determine_market_phase(volatility, trend_strength, phase_analysis)
            
            result = {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volatility': volatility,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'pressure_ratio': pressure_ratio,
                'market_phase': market_phase,
                'data_quality': min(1.0, self.tick_count / 20.0),  # ⬆ Optimizado
                'velocity': avg_velocity,
                'phase_analysis': phase_analysis,
                'candle_progress': (time.time() - self.candle_start_time) / TIMEFRAME if self.candle_start_time else 0,
                'total_ticks': self.tick_count
            }
            
            # Actualizar cache
            self._cached_analysis = result
            self._last_analysis_time = current_time
            
            return result
            
        except Exception as e:
            logging.error(f"Error en cálculo de métricas avanzadas: {e}")
            return {}
    
    def _calculate_trend_strength(self, prices):
        """Cálculo OPTIMIZADO de tendencia"""
        n = min(len(prices), 25)  # ⬇ Máximo 25 puntos
        if n < 5:
            return (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
        
        # Usar menos puntos para mejor performance
        x = np.arange(n)
        try:
            coefficients = np.polyfit(x, prices[-n:], 1)
            return coefficients[0] * 10000
        except:
            return (prices[-1] - prices[0]) * 10000
    
    def _calculate_momentum(self, prices):
        """Cálculo RÁPIDO de momentum"""
        momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
        momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else 0
        return (momentum_5 * 0.6 + momentum_10 * 0.4)  # ⬆ Pesos simplificados
    
    def _calculate_volatility(self, prices):
        """Cálculo RÁPIDO de volatilidad"""
        if len(prices) >= 15:
            recent_prices = prices[-15:]
            return (np.max(recent_prices) - np.min(recent_prices)) * 10000
        return (np.max(prices) - np.min(prices)) * 10000
    
    def _calculate_pressure(self):
        """Cálculo OPTIMIZADO de presión"""
        if len(self.ticks) < 8:  # ⬇ Umbral más bajo
            return 0.5, 0.5, 1.0
        
        # Usar solo últimos ticks para mejor performance
        recent_ticks = list(self.ticks)[-20:]
        price_changes = []
        
        for i in range(1, len(recent_ticks)):
            change = recent_ticks[i]['price'] - recent_ticks[i-1]['price']
            price_changes.append(change)
        
        if not price_changes:
            return 0.5, 0.5, 1.0
            
        positive = sum(1 for x in price_changes if x > 0)
        negative = sum(1 for x in price_changes if x < 0)
        total = len(price_changes)
        
        buy_pressure = positive / total
        sell_pressure = negative / total
        
        pressure_ratio = buy_pressure / max(0.01, sell_pressure)
        
        return buy_pressure, sell_pressure, pressure_ratio
    
    def _calculate_velocity(self):
        """Cálculo RÁPIDO de velocidad"""
        if not self.velocity_metrics:
            return 0
        # Usar numpy para cálculo rápido
        velocities = np.array([v['velocity'] for v in list(self.velocity_metrics)[-10:]])  # ⬇ Solo últimos 10
        return float(np.mean(velocities)) * 10000
    
    def _determine_market_phase(self, volatility, trend_strength, phase_analysis):
        """Determinación RÁPIDA de fase de mercado"""
        if volatility < 0.3 and abs(trend_strength) < 0.5:
            return "consolidation"
        elif abs(trend_strength) > 2.0:
            return "strong_trend"
        elif abs(trend_strength) > 1.0:
            return "trending"
        elif volatility > 1.5:
            return "high_volatility"
        elif phase_analysis.get('momentum_shift', False):
            return "reversal_potential"
        else:
            return "normal"
    
    def get_comprehensive_analysis(self):
        """Análisis completo con CACHE para máxima velocidad"""
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'INSUFFICIENT_DATA', 
                'tick_count': self.tick_count,
                'message': f'Recolectando ticks: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION}'
            }
        
        try:
            # Usar cached analysis si está disponible
            advanced_metrics = self._calculate_advanced_metrics()
            if not advanced_metrics:
                return {'status': 'ERROR', 'message': 'Error en métricas'}
            
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
                **advanced_metrics
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error en análisis completo: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        """Reinicio OPTIMIZADO"""
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
            self._cached_analysis = None
            
            # Reiniciar análisis de fases
            for phase in self.analysis_phases:
                self.analysis_phases[phase] = {'ticks': 0, 'analysis': {}}
                
        except Exception as e:
            logging.error(f"Error en reset: {e}")

# ------------------ ADAPTIVE MARKET LEARNER OPTIMIZADO ------------------
class AdaptiveMarketLearner:
    """
    Aprendizaje incremental OPTIMIZADO para no-bloqueo
    """
    def __init__(self, feature_size=18, classes=None, buffer_size=800):  # ⬇ Buffer reducido
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
        self._prediction_cache = {}
        self._cache_ttl = 2.0  # Cache de 2 segundos para predicciones

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
            max_iter=800,  # ⬇ Iteraciones reducidas
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
        """Persistencia NO BLOQUEANTE en hilo separado"""
        def async_persist():
            try:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                if self.training_count % 10 == 0:
                    logging.info(f"💾 Modelo persistido (entrenamientos: {self.training_count})")
            except Exception as e:
                logging.error(f"❌ Error guardando modelo: {e}")
        
        # Ejecutar en background
        ML_THREAD_POOL.submit(async_persist)

    def add_sample(self, features: np.ndarray, label: str):
        """Añadir muestra de forma NO BLOQUEANTE"""
        if features.shape[0] == self.feature_size:
            self.replay_buffer.append((features.astype(float), label))

    def partial_train(self, batch_size=24):  # ⬇ Batch size reducido
        """Entrenamiento ASINCRÓNICO en hilo separado"""
        if len(self.replay_buffer) < 8:  # ⬇ Umbral más bajo
            return {"trained": False, "reason": "not_enough_samples", "buffer_size": len(self.replay_buffer)}
        
        def async_training():
            try:
                # Tomar muestras más recientes
                samples = list(self.replay_buffer)[-batch_size:]
                X = np.vstack([s[0] for s in samples])
                y = np.array([s[1] for s in samples])
                
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
                    
                logging.info(f"🧠 Entrenamiento completado: {len(samples)} muestras")
                
            except Exception as e:
                logging.error(f"❌ Error en entrenamiento: {e}")
        
        # Ejecutar entrenamiento en background
        TRAINING_THREAD_POOL.submit(async_training)
        
        return {
            "trained": True, 
            "n_samples": min(batch_size, len(self.replay_buffer)),
            "training_count": self.training_count,
            "buffer_size": len(self.replay_buffer)
        }

    def predict(self, features: np.ndarray):
        """Predicción con CACHE para máxima velocidad"""
        # Generar hash para cache
        features_hash = hash(features.tobytes())
        current_time = time.time()
        
        # Verificar cache
        if (features_hash in self._prediction_cache and 
            current_time - self._prediction_cache[features_hash]['timestamp'] < self._cache_ttl):
            return self._prediction_cache[features_hash]['result']
        
        try:
            X = np.atleast_2d(features.astype(float))
            Xs = self.scaler.transform(X)
            predicted = self.model.predict(Xs)[0]
            proba = self.model.predict_proba(Xs)[0]
            confidence = max(proba) * 100
            
            result = {
                "predicted": predicted,
                "proba": dict(zip(self.model.classes_, proba)),
                "confidence": round(confidence, 2),
                "training_count": self.training_count
            }
            
            # Actualizar cache
            self._prediction_cache[features_hash] = {
                'result': result,
                'timestamp': current_time
            }
            
            # Limpiar cache viejo
            self._clean_old_cache()
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error en predict: {e}")
            return {
                "predicted": "LATERAL",
                "proba": dict(zip(self.classes, [1/3]*3)),
                "confidence": 33.3,
                "training_count": self.training_count
            }
    
    def _clean_old_cache(self):
        """Limpieza de cache viejo"""
        current_time = time.time()
        expired_keys = [
            key for key, value in self._prediction_cache.items()
            if current_time - value['timestamp'] > self._cache_ttl * 2
        ]
        for key in expired_keys:
            del self._prediction_cache[key]

# ------------------ SISTEMA PRINCIPAL ULTRA RESPONSIVE ------------------
# [El resto del código se mantiene igual pero usando las clases optimizadas]

iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner(feature_size=18)

# VARIABLES GLOBALES (iguales)
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
    "training_count": 0
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

def tick_processor(price, timestamp):
    """Procesador de ticks ULTRA RÁPIDO"""
    global current_prediction
    try:
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        # Log MUY selectivo para máximo rendimiento
        if predictor.analyzer.tick_count == 0:
            logging.info(f"🎯 PRIMER TICK PROCESADO: {price:.5f}")
        elif predictor.analyzer.tick_count % 25 == 0:  # ⬆ Menos frecuente
            logging.info(f"📊 Tick #{predictor.analyzer.tick_count}: {price:.5f}")
        
        # Procesar tick (OPERACIÓN RÁPIDA)
        tick_data = predictor.process_tick(price, seconds_remaining)
        
        if tick_data:
            # Actualizar información en tiempo real (OPERACIÓN ATÓMICA)
            current_prediction.update({
                "current_price": float(price),
                "tick_count": predictor.analyzer.tick_count,
                "timestamp": now_iso(),
                "status": "ACTIVE",
                "candle_progress": (current_time % TIMEFRAME) / TIMEFRAME
            })
            
    except Exception as e:
        logging.error(f"❌ Error procesando tick: {e}")

def premium_main_loop():
    """Loop principal ULTRA OPTIMIZADO"""
    global current_prediction, performance_stats, _last_candle_start
    global _prediction_made_this_candle, _last_prediction_time, _last_price
    
    logging.info(f"🚀 DELOWYSS AI V5.4 ULTRA RESPONSIVE INICIADA")
    logging.info("🎯 SISTEMA OPTIMIZADO: Máxima velocidad + Cero congelaciones")
    
    # Conectar a IQ Option
    iq_connected = iq_connector.connect()
    
    if not iq_connected:
        logging.warning("⚠️ No se pudo conectar a IQ Option - Activando modo simulación")
        start_tick_simulation()
    else:
        iq_connector.add_tick_listener(tick_processor)
        logging.info("✅ Sistema principal configurado - Máxima responsividad activada")
    
    # Bucle principal ULTRA OPTIMIZADO
    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # Obtener precio actual (OPERACIÓN RÁPIDA)
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price

            # Actualizar progreso de vela
            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress

            # Log de estado MUY selectivo
            if int(current_time) % 30 == 0:  # ⬆ Cada 30 segundos
                logging.info(f"📈 ESTADO: Ticks={predictor.analyzer.tick_count}, Precio={price:.5f}")

            # Lógica de predicción OPTIMIZADA
            if (seconds_remaining <= PREDICTION_WINDOW and
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - _last_prediction_time) >= 2 and
                not _prediction_made_this_candle):

                logging.info(f"🎯 PREDICCIÓN: {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                # Generar predicción híbrida (CON CACHE Y OPTIMIZACIONES)
                analysis = predictor.analyzer.get_comprehensive_analysis()
                if analysis.get('status') == 'SUCCESS':
                    features = build_advanced_features_from_analysis(analysis, seconds_remaining)
                    ml_prediction = online_learner.predict(features)  # ← CON CACHE
                    hybrid_prediction = predictor.predict_next_candle(ml_prediction)
                    
                    current_prediction.update(hybrid_prediction)
                    current_prediction.update({
                        "ai_model_predicted": ml_prediction['predicted'],
                        "ml_confidence": ml_prediction['confidence'],
                        "training_count": ml_prediction['training_count']
                    })

                    _last_prediction_time = time.time()
                    _prediction_made_this_candle = True

            # Detectar nueva vela para reinicio y aprendizaje
            if current_candle_start > _last_candle_start:
                if _last_price is not None:
                    validation = predictor.validate_prediction(_last_price)
                    if validation:
                        # AutoLearning ASINCRÓNICO
                        price_change = validation.get("price_change", 0)
                        label = "LATERAL"
                        if price_change > 0.5:
                            label = "ALZA"
                        elif price_change < -0.5:
                            label = "BAJA"
                        
                        analysis = predictor.analyzer.get_comprehensive_analysis()
                        if analysis.get('status') == 'SUCCESS':
                            features = build_advanced_features_from_analysis(analysis, 0)
                            online_learner.add_sample(features, label)
                            training_result = online_learner.partial_train(batch_size=24)  # ← ASINCRÓNICO
                            
                            if training_result["trained"]:
                                logging.info(f"📚 AutoLearning: {label} | {training_result}")
                            
                            performance_stats['last_validation'] = validation

                # Reiniciar análisis para nueva vela
                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🕯️ Nueva vela - Sistema reiniciado")

            time.sleep(0.05)  # ⬆ Loop más rápido
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(0.5)  # ⬇ Recuperación más rápida

# [El resto del código (FastAPI, endpoints, etc.) se mantiene IGUAL]

# ------------------ INICIALIZACIÓN MEJORADA ------------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info(f"⭐ DELOWYSS AI V5.4 ULTRA RESPONSIVE INICIADA EN PUERTO {PORT}")
        logging.info("🎯 OPTIMIZACIONES: Cache + ThreadPool + Cálculos vectorizados")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

start_system()

# ------------------ CLEANUP AL SALIR ------------------
import atexit

@atexit.register
def cleanup():
    """Limpieza graceful al cerrar la aplicación"""
    logging.info("🛑 Cerrando Delowyss AI...")
    ML_THREAD_POOL.shutdown(wait=False)
    TRAINING_THREAD_POOL.shutdown(wait=False)
    logging.info("✅ Recursos liberados")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=False  # ⬇ Deshabilitar logs de acceso para mejor performance
    )
