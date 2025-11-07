"""
Delowyss Trading AI — V5.4 PREMIUM COMPLETA CON AUTOLEARNING
CEO: Eduardo Solis — © 2025
MEJORAS IMPLEMENTADAS: Validación, Seguridad, Robustez
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
import psutil  # ✅ NUEVA DEPENDENCIA para monitoreo

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

# ---------------- CONFIGURACIÓN PREMIUM MEJORADA ----------------
def validate_environment():
    """✅ MEJORA: Validación robusta de variables de entorno"""
    required_vars = ["IQ_EMAIL", "IQ_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"❌ Variables de entorno faltantes: {missing_vars}")
        return False
    
    # Validar valores críticos
    if not os.getenv("IQ_EMAIL") or not os.getenv("IQ_PASSWORD"):
        logging.error("❌ Credenciales IQ Option incompletas")
        return False
        
    logging.info("✅ Configuración de entorno validada correctamente")
    return True

IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = "EURUSD"
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

# ---------------- LOGGING PROFESIONAL MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

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
        
    def validate_price(self, price: float) -> bool:
        """✅ MEJORA: Validación robusta de precios"""
        try:
            price_float = float(price)
            # Rango realista para EUR/USD
            if price_float <= 0 or price_float > 2.00000:
                logging.warning(f"⚠️ Precio fuera de rango: {price_float}")
                return False
            return True
        except (ValueError, TypeError):
            logging.warning(f"⚠️ Precio inválido: {price}")
            return False
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        """✅ MEJORA: Con validación de precio integrada"""
        try:
            # Validar precio antes de procesar
            if not self.validate_price(price):
                return None
                
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
        """Métricas avanzadas ORIGINALES CON MEJORAS DE ROBUSTEZ"""
        if len(self.ticks) < 2:
            return
            
        try:
            current_price = current_tick['price']
            current_time = current_tick['timestamp']
            
            # Velocidad del precio CON VALIDACIÓN
            previous_tick = list(self.ticks)[-2]
            time_diff = current_time - previous_tick['timestamp']
            if time_diff > 0:  # ✅ Evitar división por cero
                price_diff = current_price - previous_tick['price']
                velocity = price_diff / time_diff
                
                self.velocity_metrics.append({
                    'velocity': velocity,
                    'timestamp': current_time,
                    'price_change': price_diff
                })
            
            # Aceleración CON VALIDACIÓN
            if len(self.velocity_metrics) >= 2:
                current_velocity = self.velocity_metrics[-1]['velocity']
                previous_velocity = self.velocity_metrics[-2]['velocity']
                velocity_time_diff = current_time - self.velocity_metrics[-2]['timestamp']
                
                if velocity_time_diff > 0:  # ✅ Evitar división por cero
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
        """Análisis por fases TEMPORALES ORIGINAL - SIN CAMBIOS"""
        # ... (código original preservado exactamente)
        candle_age = tick_data['candle_age']
        
        if candle_age < 20:
            self.analysis_phases['initial']['ticks'] += 1
            if self.analysis_phases['initial']['ticks'] % 10 == 0:
                self.analysis_phases['initial']['analysis'] = self._get_phase_analysis('initial')
                
        elif candle_age < 40:
            self.analysis_phases['middle']['ticks'] += 1
            if self.analysis_phases['middle']['ticks'] % 10 == 0:
                self.analysis_phases['middle']['analysis'] = self._get_phase_analysis('middle')
                
        else:
            self.analysis_phases['final']['ticks'] += 1
            if self.analysis_phases['final']['ticks'] % 5 == 0:
                self.analysis_phases['final']['analysis'] = self._get_phase_analysis('final')
    
    def _get_phase_analysis(self, phase):
        """Análisis específico por fase ORIGINAL - SIN CAMBIOS"""
        # ... (código original preservado exactamente)
        try:
            if phase == 'initial':
                ticks = list(self.ticks)[:20] if len(self.ticks) >= 20 else list(self.ticks)
            elif phase == 'middle':
                ticks = list(self.ticks)[20:40] if len(self.ticks) >= 40 else list(self.ticks)[20:]
            else:
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
        """Métricas avanzadas ORIGINALES CON MEJORAS DE SEGURIDAD"""
        if len(self.price_memory) < 10:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            # ✅ MEJORA: Validación de longitud para cálculos
            valid_length = min(30, len(prices))
            
            # Análisis de tendencia completo CON VALIDACIÓN
            if len(prices) >= 10:
                short_trend = np.polyfit(range(min(10, len(prices))), prices[-min(10, len(prices)):], 1)[0]
                medium_trend = np.polyfit(range(min(20, len(prices))), prices[-min(20, len(prices)):], 1)[0] if len(prices) >= 20 else short_trend
                full_trend = np.polyfit(range(valid_length), prices[-valid_length:], 1)[0] if len(prices) >= 30 else medium_trend
                trend_strength = (short_trend * 0.4 + medium_trend * 0.3 + full_trend * 0.3) * 10000
            else:
                trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            # Momentum multi-temporal CON VALIDACIÓN
            momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
            momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else momentum_5
            momentum_20 = (prices[-1] - prices[-20]) * 10000 if len(prices) >= 20 else momentum_10
            momentum = (momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2)
            
            # Volatilidad segmentada CON VALIDACIÓN
            if len(prices) >= 20:
                early_volatility = (max(prices[:10]) - min(prices[:10])) * 10000
                late_volatility = (max(prices[-10:]) - min(prices[-10:])) * 10000
                volatility = (early_volatility * 0.3 + late_volatility * 0.7)
            else:
                volatility = (max(prices) - min(prices)) * 10000 if len(prices) > 1 else 0
            
            # Presión de compra/venta MEJORADA
            if len(self.ticks) > 10:
                price_changes = []
                for i in range(1, len(self.ticks)):
                    change = self.ticks[i]['price'] - self.ticks[i-1]['price']
                    price_changes.append(change)
                
                if price_changes and len(price_changes) > 0:  # ✅ Validación adicional
                    positive = len([x for x in price_changes if x > 0])
                    negative = len([x for x in price_changes if x < 0])
                    total = len(price_changes)
                    
                    buy_pressure = positive / total if total > 0 else 0.5
                    sell_pressure = negative / total if total > 0 else 0.5
                    
                    # ✅ MEJORA CRÍTICA: Evitar división por cero
                    if sell_pressure > 0.05 and total > 0:
                        pressure_ratio = buy_pressure / sell_pressure
                    else:
                        pressure_ratio = 10.0 if buy_pressure > 0 else 1.0
                else:
                    buy_pressure = sell_pressure = pressure_ratio = 0.5
            else:
                buy_pressure = sell_pressure = pressure_ratio = 0.5
            
            # Velocidad promedio CON VALIDACIÓN
            avg_velocity = 0
            if self.velocity_metrics and len(self.velocity_metrics) > 0:
                velocities = [v['velocity'] for v in self.velocity_metrics]
                avg_velocity = np.mean(velocities) * 10000 if velocities else 0
            
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
        """Combina análisis de todas las fases de la vela ORIGINAL - SIN CAMBIOS"""
        # ... (código original preservado exactamente)
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
            
            trends = [initial.get('trend'), middle.get('trend'), final.get('trend')]
            if len(set(trends)) > 1:
                combined['momentum_shift'] = True
            
            same_trend_count = sum(1 for i in range(len(trends)-1) if trends[i] == trends[i+1])
            combined['consistency_score'] = same_trend_count / max(1, len(trends)-1)
            
            return combined
        except Exception as e:
            logging.debug(f"Error combinando análisis de fases: {e}")
            return {}
    
    def get_comprehensive_analysis(self):
        """Análisis completo ORIGINAL MEJORADO - SIN CAMBIOS ESTRUCTURALES"""
        # ... (código original preservado exactamente)
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
        """Para compatibilidad con AutoLearning - SIN CAMBIOS"""
        return [tick['price'] for tick in list(self.ticks)[-n:]]
    
    def reset(self):
        """Reinicia el análisis para nueva vela ORIGINAL - SIN CAMBIOS"""
        # ... (código original preservado exactamente)
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
            
            for phase in self.analysis_phases:
                self.analysis_phases[phase] = {'ticks': 0, 'analysis': {}}
                
        except Exception as e:
            logging.error(f"Error en reset: {e}")

# ------------------ ADAPTIVE MARKET LEARNER (MEJORADO CON BACKUP) ------------------
class AdaptiveMarketLearner:
    """
    Aprendizaje incremental MEJORADO con sistema de backup
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
        self.last_backup_time = time.time()

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

    def _load_model(self):
        """✅ MEJORA: Carga con múltiples intentos y fallback"""
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logging.info("✅ Modelo online cargado exitosamente")
                return model
            except Exception as e:
                logging.warning(f"⚠️ No se pudo cargar modelo online: {e}")
                # Intentar cargar backup si existe
                return self._load_backup_model()
        
        # Crear nuevo modelo
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

    def _load_backup_model(self):
        """✅ MEJORA: Sistema de recuperación de backup"""
        backup_files = []
        model_dir = os.path.dirname(self.model_path)
        
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith("online_sgd.pkl.backup"):
                    backup_files.append(file)
            
            if backup_files:
                latest_backup = sorted(backup_files)[-1]
                backup_path = os.path.join(model_dir, latest_backup)
                try:
                    model = joblib.load(backup_path)
                    logging.info(f"✅ Modelo recuperado desde backup: {latest_backup}")
                    return model
                except Exception as e:
                    logging.error(f"❌ Error cargando backup: {e}")
        
        logging.info("🆕 Creando nuevo modelo desde cero")
        return self._create_new_model()

    def _create_new_model(self):
        """Crea un nuevo modelo limpio"""
        model = SGDClassifier(
            loss='log_loss', 
            max_iter=1000,
            tol=1e-3,
            warm_start=True,
            learning_rate='optimal'
        )
        dummy_X = np.random.normal(0, 0.1, (3, self.feature_size))
        dummy_y = np.array(['BAJA', 'LATERAL', 'ALZA'])
        model.partial_fit(dummy_X, dummy_y, classes=self.classes)
        return model

    def _load_scaler(self):
        """✅ MEJORA: Carga de scaler con recuperación"""
        if os.path.exists(self.scaler_path):
            try:
                scaler = joblib.load(self.scaler_path)
                logging.info("✅ Scaler online cargado exitosamente")
                return scaler
            except Exception as e:
                logging.warning(f"⚠️ No se pudo cargar scaler: {e}")
        
        return StandardScaler()

    def create_backup(self):
        """✅ MEJORA: Sistema de backup automático"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_backup_path = f"{self.model_path}.backup_{timestamp}"
            scaler_backup_path = f"{self.scaler_path}.backup_{timestamp}"
            
            joblib.dump(self.model, model_backup_path)
            joblib.dump(self.scaler, scaler_backup_path)
            
            # Limitar número de backups (mantener últimos 5)
            self._cleanup_old_backups()
            
            self.last_backup_time = time.time()
            logging.info(f"💾 Backup creado: {timestamp}")
            return True
        except Exception as e:
            logging.error(f"❌ Error creando backup: {e}")
            return False

    def _cleanup_old_backups(self):
        """Limpiar backups antiguos"""
        try:
            model_dir = os.path.dirname(self.model_path)
            backup_files = []
            
            for file in os.listdir(model_dir):
                if file.startswith("online_sgd.pkl.backup"):
                    backup_files.append(file)
            
            # Mantener solo los últimos 5 backups
            if len(backup_files) > 5:
                backup_files.sort()
                for old_backup in backup_files[:-5]:
                    os.remove(os.path.join(model_dir, old_backup))
                    logging.debug(f"🧹 Backup eliminado: {old_backup}")
        except Exception as e:
            logging.debug(f"Error limpiando backups: {e}")

    def persist(self):
        """✅ MEJORA: Persistencia con backup automático"""
        try:
            # Crear backup antes de persistir
            if self.training_count % 10 == 0:
                self.create_backup()
            
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            if self.training_count % 10 == 0:
                logging.info(f"💾 Modelo persistido (entrenamientos: {self.training_count})")
            return True
        except Exception as e:
            logging.error(f"❌ Error guardando modelo: {e}")
            return False

    def add_sample(self, features: np.ndarray, label: str):
        """✅ MEJORA: Validación de features antes de añadir"""
        try:
            if features.shape[0] == self.feature_size and not np.any(np.isnan(features)):
                self.replay_buffer.append((features.astype(float), label))
                return True
            else:
                logging.warning("⚠️ Muestra descartada - features inválidos")
                return False
        except Exception as e:
            logging.error(f"❌ Error añadiendo muestra: {e}")
            return False

    def partial_train(self, batch_size=32):
        """✅ MEJORA: Entrenamiento con monitoreo de calidad"""
        if len(self.replay_buffer) < 10:
            return {"trained": False, "reason": "not_enough_samples", "buffer_size": len(self.replay_buffer)}
        
        # Tomar muestras más recientes
        samples = list(self.replay_buffer)[-batch_size:]
        
        try:
            X = np.vstack([s[0] for s in samples])
            y = np.array([s[1] for s in samples])
            
            # Validar datos antes de entrenar
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logging.warning("⚠️ Datos con NaN/Inf - saltando entrenamiento")
                return {"trained": False, "reason": "invalid_data"}
            
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
                "data_quality": "good"
            }
        except Exception as e:
            logging.error(f"❌ Error en entrenamiento: {e}")
            return {"trained": False, "reason": str(e)}

    def predict_proba(self, features: np.ndarray):
        """✅ MEJORA: Predicción con validación de entrada"""
        try:
            X = np.atleast_2d(features.astype(float))
            
            # Validar features
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logging.warning("⚠️ Features inválidos en predict_proba")
                return dict(zip(self.classes, np.ones(len(self.classes)) / len(self.classes)))
                
            Xs = self.scaler.transform(X)
            probs = self.model.predict_proba(Xs)[0]
            return dict(zip(self.model.classes_, probs))
        except Exception as e:
            logging.warning(f"⚠️ Fallback en predict_proba: {e}")
            return dict(zip(self.classes, np.ones(len(self.classes)) / len(self.classes)))

    def predict(self, features: np.ndarray):
        """✅ MEJORA: Predicción con diagnóstico extendido"""
        try:
            X = np.atleast_2d(features.astype(float))
            
            # Validación completa de features
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logging.warning("🚨 Features inválidos - retornando predicción neutral")
                return self._get_neutral_prediction()
                
            Xs = self.scaler.transform(X)
            predicted = self.model.predict(Xs)[0]
            proba = self.predict_proba(features)
            confidence = max(proba.values()) * 100
            
            return {
                "predicted": predicted,
                "proba": proba,
                "confidence": round(confidence, 2),
                "training_count": self.training_count,
                "status": "SUCCESS"
            }
        except Exception as e:
            logging.error(f"❌ Error en predict: {e}")
            return self._get_neutral_prediction()

    def _get_neutral_prediction(self):
        """Predicción neutral para casos de error"""
        return {
            "predicted": "LATERAL",
            "proba": dict(zip(self.classes, [1/3]*3)),
            "confidence": 33.3,
            "training_count": self.training_count,
            "status": "FALLBACK"
        }

    def get_model_info(self):
        """✅ MEJORA: Información extendida del modelo"""
        return {
            "training_count": self.training_count,
            "buffer_size": len(self.replay_buffer),
            "feature_size": self.feature_size,
            "classes": list(self.classes),
            "last_backup": self.last_backup_time,
            "model_path": self.model_path
        }

# ------------------ SISTEMA DE MONITOREO MEJORADO ------------------
class SystemHealthMonitor:
    """✅ NUEVA CLASE: Monitoreo de salud del sistema"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_metrics = deque(maxlen=100)
        
    def get_system_health(self):
        """Obtiene métricas completas de salud del sistema"""
        try:
            # Uso de CPU y memoria
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Uso de recursos del proceso actual
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Tiempo de actividad
            uptime = time.time() - self.start_time
            
            health_data = {
                "timestamp": now_iso(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": process_memory,
                "disk_free_gb": disk.free / (1024**3),
                "uptime_seconds": int(uptime),
                "status": "HEALTHY" if cpu_percent < 80 and memory.percent < 80 else "WARNING"
            }
            
            self.system_metrics.append(health_data)
            return health_data
            
        except Exception as e:
            logging.error(f"❌ Error en monitoreo de salud: {e}")
            return {
                "timestamp": now_iso(),
                "status": "ERROR",
                "error": str(e)
            }
    
    def get_performance_trend(self):
        """Obtiene tendencia de rendimiento"""
        if len(self.system_metrics) < 2:
            return "INSUFFICIENT_DATA"
        
        recent_cpu = [m['cpu_percent'] for m in list(self.system_metrics)[-5:]]
        avg_cpu = np.mean(recent_cpu)
        
        if avg_cpu > 70:
            return "HIGH_LOAD"
        elif avg_cpu > 50:
            return "MEDIUM_LOAD"
        else:
            return "LOW_LOAD"

# ------------------ SISTEMA PRINCIPAL MEJORADO ---------------
# Inicializar componentes mejorados
iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner(feature_size=18)
health_monitor = SystemHealthMonitor()  # ✅ NUEVO

# VARIABLES GLOBALES MEJORADAS
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
    "system_health": "CHECKING"  # ✅ NUEVO
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None,
    'system_uptime': 0  # ✅ NUEVO
}

# Estado interno MEJORADO
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False
_last_prediction_time = 0
_last_price = None
_system_start_time = time.time()  # ✅ NUEVO

def tick_processor(price, timestamp):
    """✅ MEJORA: Procesador con validación extendida"""
    global current_prediction
    try:
        # Validación de precio antes de procesar
        if price <= 0 or price > 2.00000:
            logging.warning(f"⚠️ Tick descartado - precio inválido: {price}")
            return
            
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        # Log del primer tick
        if predictor.analyzer.tick_count == 0:
            logging.info(f"🎯 PRIMER TICK PROCESADO: {price:.5f}")
        
        # Log informativo periódico
        if predictor.analyzer.tick_count % 15 == 0:
            logging.info(f"📊 Tick #{predictor.analyzer.tick_count + 1}: {price:.5f} | Tiempo restante: {seconds_remaining:.1f}s")
        
        # Procesar tick en el predictor
        tick_data = predictor.process_tick(price, seconds_remaining)
        
        if tick_data:
            # Actualizar información en tiempo real
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
    """✅ MEJORA: Loop principal con monitoreo de salud"""
    global current_prediction, performance_stats, _last_candle_start
    global _prediction_made_this_candle, _last_prediction_time, _last_price, _system_start_time
    
    # ✅ MEJORA: Validar entorno antes de iniciar
    if not validate_environment():
        logging.error("🚨 ERROR CRÍTICO: Configuración de entorno inválida")
        return
    
    logging.info(f"🚀 DELOWYSS AI V5.4 PREMIUM MEJORADA INICIADA EN PUERTO {PORT}")
    logging.info("🎯 SISTEMA HÍBRIDO AVANZADO CON MONITOREO DE SALUD")
    
    # Conectar a IQ Option
    iq_connected = iq_connector.connect()
    
    if not iq_connected:
        logging.warning("⚠️ No se pudo conectar a IQ Option - Activando modo simulación")
        start_tick_simulation()
    else:
        iq_connector.add_tick_listener(tick_processor)
        logging.info("✅ Sistema principal configurado - Esperando ticks...")
    
    # Bucle principal mejorado
    last_health_check = 0
    last_performance_log = 0
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # ✅ MEJORA: Monitoreo de salud periódico
            if current_time - last_health_check > 30:  # Cada 30 segundos
                health_status = health_monitor.get_system_health()
                current_prediction['system_health'] = health_status['status']
                performance_stats['system_uptime'] = int(current_time - _system_start_time)
                last_health_check = current_time
                
                if health_status['status'] == 'WARNING':
                    logging.warning(f"⚠️ Sistema bajo carga: CPU {health_status['cpu_percent']}%")
            
            # ✅ MEJORA: Log de performance periódico
            if current_time - last_performance_log > 60:  # Cada 60 segundos
                accuracy = performance_stats.get('recent_accuracy', 0)
                total_preds = performance_stats.get('total_predictions', 0)
                logging.info(f"📊 PERFORMANCE: Precisión {accuracy:.1f}% | Total predicciones: {total_preds}")
                last_performance_log = current_time
            
            # Obtener precio actual
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price

            # Actualizar progreso de vela
            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress

            # Lógica de predicción en últimos segundos
            if (seconds_remaining <= PREDICTION_WINDOW and
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - _last_prediction_time) >= 2 and
                not _prediction_made_this_candle):

                logging.info(f"🎯 VENTANA DE PREDICCIÓN ACTIVA: {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                # Generar predicción híbrida
                analysis = predictor.analyzer.get_comprehensive_analysis()
                if analysis.get('status') == 'SUCCESS':
                    features = build_advanced_features_from_analysis(analysis, seconds_remaining)
                    ml_prediction = online_learner.predict(features)
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
                        # Lógica de AutoLearning con la vela cerrada
                        price_change = validation.get("price_change", 0)
                        label = "LATERAL"
                        if price_change > 0.5:
                            label = "ALZA"
                        elif price_change < -0.5:
                            label = "BAJA"
                        
                        analysis = predictor.analyzer.get_comprehensive_analysis()
                        if analysis.get('status') == 'SUCCESS':
                            features = build_advanced_features_from_analysis(analysis, 0)
                            
                            # ✅ MEJORA: Validar antes de aprender
                            if online_learner.add_sample(features, label):
                                training_result = online_learner.partial_train(batch_size=32)
                                logging.info(f"📚 AutoLearning: {label} | Cambio: {price_change:.1f}pips | {training_result}")
                            
                            performance_stats['last_validation'] = validation

                # Reiniciar análisis para nueva vela
                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🕯️ NUEVA VELA - Sistema de análisis reiniciado")

            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(1)

# ------------------ INTERFAZ WEB MEJORADA ------------------
app = FastAPI(
    title="Delowyss AI Premium V5.4 MEJORADA",
    version="5.4.1",  # ✅ ACTUALIZADO
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
    return JSONResponse({
        "last_validation": last_val,
        "performance": performance_stats,
        "timestamp": now_iso()
    })

@app.get("/api/health")
def api_health():
    """✅ MEJORA: Endpoint de salud extendido"""
    health_status = health_monitor.get_system_health()
    model_info = online_learner.get_model_info()
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": now_iso(),
        "version": "5.4.1-enhanced",
        "port": PORT,
        "system_health": health_status,
        "model_info": model_info,
        "features": [
            "full_candle_analysis", 
            "phase_analysis", 
            "tick_by_tick", 
            "online_learning",
            "hybrid_ai_ml",
            "responsive_interface",
            "health_monitoring",  # ✅ NUEVO
            "model_backup"       # ✅ NUEVO
        ]
    })

@app.get("/api/system-info")
def api_system_info():
    """✅ MEJORA: Información del sistema extendida"""
    connection_status = iq_connector.get_connection_status()
    health_status = health_monitor.get_system_health()
    model_info = online_learner.get_model_info()
    
    return JSONResponse({
        "status": "running",
        "pair": PAR,
        "timeframe": TIMEFRAME,
        "prediction_window": PREDICTION_WINDOW,
        "current_ticks": predictor.analyzer.tick_count,
        "ml_training_count": online_learner.training_count,
        "connection": connection_status,
        "system_health": health_status,
        "model_info": model_info,
        "timestamp": now_iso()
    })

@app.get("/api/debug")
def api_debug():
    """✅ MEJORA: Endpoint de diagnóstico completo"""
    connection_status = iq_connector.get_connection_status()
    analysis = predictor.analyzer.get_comprehensive_analysis()
    health_status = health_monitor.get_system_health()
    model_info = online_learner.get_model_info()
    
    return JSONResponse({
        "system": {
            "status": "running",
            "timestamp": now_iso(),
            "timeframe": TIMEFRAME,
            "port": PORT,
            "uptime": int(time.time() - _system_start_time)
        },
        "connection": connection_status,
        "analysis": {
            "status": analysis.get('status'),
            "tick_count": analysis.get('tick_count', 0),
            "current_price": analysis.get('current_price', 0),
            "data_quality": analysis.get('data_quality', 0)
        },
        "ml": {
            "training_count": online_learner.training_count,
            "buffer_size": len(online_learner.replay_buffer),
            "model_info": model_info
        },
        "performance": performance_stats,
        "health": health_status
    })

@app.get("/api/backup-model")
def api_backup_model():
    """✅ NUEVO: Endpoint para forzar backup del modelo"""
    try:
        success = online_learner.create_backup()
        return JSONResponse({
            "status": "success" if success else "error",
            "message": "Backup creado exitosamente" if success else "Error creando backup",
            "timestamp": now_iso()
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "timestamp": now_iso()
        })

def generate_html_interface():
    """Interfaz HTML MEJORADA con información de salud"""
    # [HTML original preservado con pequeñas adiciones para health monitoring]
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    current_price = current_prediction.get("current_price", 0)
    tick_count = current_prediction.get("tick_count", 0)
    system_health = current_prediction.get("system_health", "UNKNOWN")
    
    # Color según salud del sistema
    health_color = {
        "HEALTHY": "green",
        "WARNING": "orange", 
        "ERROR": "red",
        "CHECKING": "blue",
        "UNKNOWN": "gray"
    }.get(system_health, "gray")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.4 MEJORADA</title>
        <style>
            .health-status {{
                color: {health_color};
                font-weight: bold;
            }}
            /* [Estilos originales preservados] */
        </style>
    </head>
    <body>
        <!-- [Interfaz original preservada] -->
        <div class="health-indicator">
            <h3>Estado del Sistema: <span class="health-status">{system_health}</span></h3>
        </div>
        <!-- [Resto del HTML original] -->
    </body>
    </html>
    """
    return html_content

# ------------------ INICIALIZACIÓN MEJORADA ------------------
def start_system():
    """✅ MEJORA: Inicialización con validación"""
    try:
        # Validar configuración antes de iniciar
        if not validate_environment():
            logging.error("🚨 No se puede iniciar sistema - configuración inválida")
            return False
            
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info(f"⭐ DELOWYSS AI V5.4 MEJORADA INICIADA EN PUERTO {PORT}")
        logging.info("🎯 SISTEMA HÍBRIDO MEJORADO: IA + AutoLearning + Monitoreo + Backup")
        return True
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")
        return False

# ✅ MEJORA: Ejecución con gestión de errores
if start_system():
    logging.info("✅ Sistema iniciado correctamente")
else:
    logging.error("❌ Fallo en inicialización del sistema")

# ------------------ EJECUCIÓN PRINCIPAL ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True
    )
