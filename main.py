"""
Delowyss Trading AI — V5.4 PREMIUM COMPLETA CON AUTOLEARNING
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
PAR = "EURUSD"  # ✅ EUR/USD REAL - MERCADO PRINCIPAL
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
        
    def validate_eurusd_price(self, price: float) -> bool:
        """✅ MEJORA: Validación específica para EUR/USD"""
        try:
            price_float = float(price)
            # Rango realista específico para EUR/USD
            if price_float <= 0.8000 or price_float > 1.5000:
                logging.warning(f"⚠️ Precio EUR/USD fuera de rango: {price_float}")
                return False
            return True
        except (ValueError, TypeError):
            logging.warning(f"⚠️ Precio EUR/USD inválido: {price}")
            return False
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        """✅ MEJORA: Con validación EUR/USD integrada"""
        try:
            # Validar precio EUR/USD antes de procesar
            if not self.validate_eurusd_price(price):
                return None
                
            price = float(price)
            current_time = time.time()
            
            # Inicializar vela si es el primer tick
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela EUR/USD iniciada - Comenzando análisis tick-by-tick")
            
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
            logging.error(f"Error en add_tick EUR/USD: {e}")
            return None
    
    # ... (el resto de la clase PremiumAIAnalyzer se mantiene ORIGINAL) ...

# ------------------ ADAPTIVE MARKET LEARNER (ORIGINAL) ------------------
class AdaptiveMarketLearner:
    # ... (clase completa se mantiene ORIGINAL) ...

# ------------------ FEATURE BUILDER MEJORADO ------------------
def build_advanced_features_from_analysis(analysis, seconds_remaining, tick_window=30):
    # ... (función se mantiene ORIGINAL) ...

# ------------------ SISTEMA IA PROFESIONAL COMPLETO ------------------
class ComprehensiveAIPredictor:
    # ... (clase completa se mantiene ORIGINAL) ...

# ------------------ CONEXIÓN PROFESIONAL MEJORADA (EUR/USD OPTIMIZADA) ------------------
class ProfessionalIQConnector:
    def __init__(self):
        self.connected = False
        self.tick_listeners = []
        self.last_price = 1.10000  # ✅ Precio inicial típico EUR/USD
        self.tick_count = 0
        self.api = None
        self.asset_name = "EURUSD"
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.eurusd_variants = ["EURUSD", "EURUSD-OTC", "EURUSD"]  # ✅ Variantes EUR/USD
        
    def validate_iq_credentials(self):
        """✅ MEJORA: Validación específica de credenciales IQ Option"""
        if not IQ_EMAIL or not IQ_PASSWORD:
            logging.error("❌ CREDENCIALES IQ OPTION NO CONFIGURADAS")
            logging.info("💡 Configure las variables de entorno:")
            logging.info("   - IQ_EMAIL=su_email@ejemplo.com")
            logging.info("   - IQ_PASSWORD=su_contraseña")
            return False
        
        logging.info("✅ Credenciales IQ Option validadas")
        return True

    def connect(self):
        """✅ MEJORA: Conexión optimizada para EUR/USD"""
        if not IQ_OPTION_AVAILABLE:
            logging.error("❌ IQ Option API no disponible - Instala: pip install iqoptionapi")
            return False
            
        # Validar credenciales primero
        if not self.validate_iq_credentials():
            return False
            
        try:
            self.connection_attempts += 1
            logging.info(f"🌐 Conectando a IQ Option para EUR/USD (Intento {self.connection_attempts}/{self.max_connection_attempts})...")
            
            self.api = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.api.connect()
            
            if check:
                result = self.api.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conexión IQ Option establecida - MODO PRÁCTICA")
                
                # Buscar EUR/USD específicamente
                self._setup_eurusd_stream()
                return True
            else:
                logging.error(f"❌ Conexión IQ Option fallida: {reason}")
                if self.connection_attempts < self.max_connection_attempts:
                    time.sleep(2)
                    return self.connect()
                return False
                
        except Exception as e:
            logging.error(f"❌ Error de conexión IQ Option: {e}")
            if self.connection_attempts < self.max_connection_attempts:
                time.sleep(2)
                return self.connect()
            return False

    def _setup_eurusd_stream(self):
        """✅ MEJORA: Configuración específica para EUR/USD"""
        try:
            # Probar diferentes variantes de EUR/USD
            eurusd_found = False
            
            for asset in self.eurusd_variants:
                if self._test_eurusd_asset(asset):
                    self.asset_name = asset
                    eurusd_found = True
                    logging.info(f"✅ EUR/USD configurado: {asset}")
                    break
            
            if not eurusd_found:
                logging.error("❌ No se encontró EUR/USD en IQ Option")
                self.connected = False
                return
            
            # Iniciar stream de velas para EUR/USD
            self.api.start_candles_stream(self.asset_name, TIMEFRAME, 10)
            
            # Iniciar listener específico para EUR/USD
            thread = threading.Thread(target=self._eurusd_tick_listener, daemon=True)
            thread.start()
            
            logging.info(f"📡 Stream de EUR/USD iniciado: {self.asset_name}")
            
        except Exception as e:
            logging.error(f"❌ Error configurando stream EUR/USD: {e}")

    def _test_eurusd_asset(self, asset_name):
        """✅ MEJORA: Test específico para EUR/USD"""
        try:
            self.api.start_candles_stream(asset_name, TIMEFRAME, 5)
            time.sleep(1)
            candles = self.api.get_realtime_candles(asset_name, TIMEFRAME)
            
            if not candles:
                return False
                
            # Verificar que sea realmente EUR/USD por el precio
            candle_list = list(candles.values())
            if candle_list:
                latest_candle = candle_list[-1]
                price = latest_candle.get('close', 0)
                
                # Rango de precio típico EUR/USD
                if 0.8000 < price < 1.5000:
                    logging.info(f"✅ EUR/USD válido detectado: {asset_name} @ {price}")
                    return True
                else:
                    logging.warning(f"⚠️ Precio no corresponde a EUR/USD: {price}")
                    return False
            
            return False
            
        except Exception as e:
            logging.debug(f"❌ Variante {asset_name} no disponible: {e}")
            return False

    def _eurusd_tick_listener(self):
        """✅ MEJORA: Listener optimizado para EUR/USD"""
        consecutive_failures = 0
        max_failures = 10
        
        logging.info(f"🎯 Iniciando monitor de ticks EUR/USD: {self.asset_name}")
        
        while self.connected and consecutive_failures < max_failures:
            try:
                candles = self.api.get_realtime_candles(self.asset_name, TIMEFRAME)
                
                if candles:
                    consecutive_failures = 0
                    candle_list = list(candles.values())
                    
                    if candle_list:
                        latest_candle = candle_list[-1]
                        price = latest_candle.get('close')
                        
                        if price and price > 0:
                            price_float = float(price)
                            
                            # Validación adicional de precio EUR/USD
                            if 0.8000 < price_float < 1.5000:
                                
                                if price_float != self.last_price:
                                    self.last_price = price_float
                                    self.tick_count += 1
                                    
                                    # Log informativo periódico
                                    if self.tick_count <= 10 or self.tick_count % 25 == 0:
                                        logging.info(f"🎯 TICK EUR/USD #{self.tick_count}: {self.last_price:.5f}")
                                    
                                    # Notificar listeners
                                    timestamp = time.time()
                                    for listener in self.tick_listeners[:]:
                                        try:
                                            listener(self.last_price, timestamp)
                                        except Exception as e:
                                            logging.error(f"❌ Error en listener EUR/USD: {e}")
                            else:
                                logging.warning(f"⚠️ Tick EUR/USD descartado - precio inválido: {price_float}")
                else:
                    consecutive_failures += 1
                    if consecutive_failures % 5 == 0:
                        logging.warning(f"⚠️ Sin datos de EUR/USD (fallos consecutivos: {consecutive_failures})")
                
                time.sleep(0.5)
                
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"❌ Error en listener EUR/USD (fallo {consecutive_failures}): {e}")
                time.sleep(2)
        
        if consecutive_failures >= max_failures:
            logging.error("🚨 MÁXIMOS FALLOS CONSECUTIVOS EUR/USD - Deteniendo listener")
            self.connected = False

    def add_tick_listener(self, listener):
        """Añadir listener para procesamiento de ticks EUR/USD"""
        self.tick_listeners.append(listener)
        logging.info(f"✅ Listener de EUR/USD añadido (total: {len(self.tick_listeners)})")

    def get_realtime_price(self):
        """Obtener precio EUR/USD en tiempo real"""
        return float(self.last_price)

    def get_connection_status(self):
        """Estado completo de la conexión EUR/USD"""
        return {
            "connected": self.connected,
            "tick_count": self.tick_count,
            "last_price": self.last_price,
            "asset": self.asset_name,
            "listeners": len(self.tick_listeners),
            "attempts": self.connection_attempts,
            "pair": "EUR/USD"  # ✅ Específico para EUR/USD
        }

# ------------------ SISTEMA DE SIMULACIÓN DE TICKS MEJORADO ------------------
def start_tick_simulation():
    """✅ MEJORA: Simulación específica para EUR/USD"""
    logging.info("🔧 INICIANDO SISTEMA DE SIMULACIÓN EUR/USD")
    
    def simulated_eurusd_tick_generator():
        price = 1.10000  # ✅ Precio inicial típico EUR/USD
        tick_num = 0
        
        while True:
            try:
                # Generar movimiento de precio realista para EUR/USD
                change = np.random.normal(0, 0.00008)  # ✅ Volatilidad realista EUR/USD
                price += change
                
                # Mantener en rango realista EUR/USD
                price = max(1.08000, min(1.12000, price))
                
                tick_num += 1
                
                # Procesar tick simulado EUR/USD
                tick_processor(price, time.time())
                
                # Log periódico
                if tick_num <= 5 or tick_num % 30 == 0:
                    logging.info(f"🔧 TICK SIMULADO EUR/USD #{tick_num}: {price:.5f}")
                
                time.sleep(1)  # 1 tick por segundo
                
            except Exception as e:
                logging.error(f"❌ Error en simulación EUR/USD: {e}")
                time.sleep(5)
    
    # Iniciar en hilo separado
    sim_thread = threading.Thread(target=simulated_eurusd_tick_generator, daemon=True)
    sim_thread.start()

# --------------- SISTEMA PRINCIPAL MEJORADO (EUR/USD) ---------------
iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()
online_learner = AdaptiveMarketLearner(feature_size=18)

# VARIABLES GLOBALES
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 Sistema EUR/USD inicializando..."],  # ✅ Específico EUR/USD
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
    """Procesador de ticks MEJORADO para EUR/USD"""
    global current_prediction
    try:
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        # Log del primer tick EUR/USD
        if predictor.analyzer.tick_count == 0:
            logging.info(f"🎯 PRIMER TICK EUR/USD PROCESADO: {price:.5f}")
        
        # Log informativo periódico
        if predictor.analyzer.tick_count % 15 == 0:
            logging.info(f"📊 Tick EUR/USD #{predictor.analyzer.tick_count + 1}: {price:.5f} | Tiempo restante: {seconds_remaining:.1f}s")
        
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
        logging.error(f"❌ Error procesando tick EUR/USD: {e}")

def premium_main_loop():
    """Loop principal MEJORADO para EUR/USD"""
    global current_prediction, performance_stats, _last_candle_start
    global _prediction_made_this_candle, _last_prediction_time, _last_price
    
    logging.info(f"🚀 DELOWYSS AI V5.4 PREMIUM EUR/USD INICIADA EN PUERTO {PORT}")
    logging.info("🎯 SISTEMA HÍBRIDO AVANZADO ESPECIALIZADO EN EUR/USD")
    
    # Conectar a IQ Option para EUR/USD
    iq_connected = iq_connector.connect()
    
    if not iq_connected:
        logging.warning("⚠️ No se pudo conectar a IQ Option - Activando modo simulación EUR/USD")
        start_tick_simulation()
    else:
        iq_connector.add_tick_listener(tick_processor)
        logging.info("✅ Sistema EUR/USD configurado - Esperando ticks...")
    
    # Bucle principal mejorado
    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # Obtener precio actual EUR/USD
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price

            # Actualizar progreso de vela
            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress

            # Log de estado cada 20 segundos
            if int(current_time) % 20 == 0:
                tick_info = iq_connector.get_connection_status()
                logging.info(f"📈 ESTADO EUR/USD: Ticks={predictor.analyzer.tick_count}, Precio={price:.5f}, Vela={candle_progress:.1%}")

            # Lógica de predicción en últimos segundos
            if (seconds_remaining <= PREDICTION_WINDOW and
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - _last_prediction_time) >= 2 and
                not _prediction_made_this_candle):

                logging.info(f"🎯 VENTANA DE PREDICCIÓN EUR/USD ACTIVA: {seconds_remaining:.1f}s | Ticks: {predictor.analyzer.tick_count}")
                
                # Generar predicción híbrida para EUR/USD
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
                        # Lógica de AutoLearning con la vela cerrada EUR/USD
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
                            training_result = online_learner.partial_train(batch_size=32)
                            
                            logging.info(f"📚 AutoLearning EUR/USD: {label} | Cambio: {price_change:.1f}pips | {training_result}")
                            
                            performance_stats['last_validation'] = validation

                # Reiniciar análisis para nueva vela EUR/USD
                predictor.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                logging.info("🕯️ NUEVA VELA EUR/USD - Sistema de análisis reiniciado")

            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal EUR/USD: {e}")
            time.sleep(1)

# ------------------ INTERFAZ WEB COMPLETA ORIGINAL ------------------
# ... (TODO el código de FastAPI se mantiene EXACTAMENTE IGUAL) ...

@app.get("/api/system-info")
def api_system_info():
    connection_status = iq_connector.get_connection_status()
    return JSONResponse({
        "status": "running",
        "pair": "EUR/USD",  # ✅ Específico para EUR/USD
        "timeframe": TIMEFRAME,
        "prediction_window": PREDICTION_WINDOW,
        "current_ticks": predictor.analyzer.tick_count,
        "ml_training_count": online_learner.training_count,
        "connection": connection_status,
        "timestamp": now_iso()
    })

# ... (resto del código FastAPI ORIGINAL) ...

def generate_html_interface():
    """Interfaz HTML COMPLETA ORIGINAL con mención EUR/USD"""
    # [HTML original preservado con pequeña mención EUR/USD]
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    current_price = current_prediction.get("current_price", 0)
    tick_count = current_prediction.get("tick_count", 0)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.4 - EUR/USD</title>
        <!-- ESTILOS ORIGINALES COMPLETOS PRESERVADOS -->
    </head>
    <body>
        <!-- INTERFAZ ORIGINAL COMPLETA -->
        <div class="header">
            <h1>Delowyss AI Premium V5.4 - EUR/USD</h1>
        </div>
        <!-- [Resto del HTML original preservado exactamente] -->
    </body>
    </html>
    """
    return html_content

# ------------------ INICIALIZACIÓN MEJORADA ------------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info(f"⭐ DELOWYSS AI V5.4 EUR/USD INICIADA EN PUERTO {PORT}")
        logging.info("🎯 SISTEMA HÍBRIDO ESPECIALIZADO EN EUR/USD: IA + AutoLearning + Interfaz Original")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema EUR/USD: {e}")

start_system()

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
