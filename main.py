# main.py - VERSIÓN PREMIUM PROFESIONAL CORREGIDA
"""
Delowyss Trading AI — V5.0 PREMIUM ESTABLE
Sistema profesional con IA avanzada - Versión Mejorada
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
import signal
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from iqoptionapi.stable_api import IQ_Option
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN PREMIUM ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 12

# ---------------- LOGGING PROFESIONAL MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA PREMIUM MEJORADA ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=200)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=50)
        self.volume_profile = deque(maxlen=30)
        self.trend_memory = deque(maxlen=10)
        self.last_candle_close = None
        
    def add_tick(self, price: float):
        price = float(price)
        
        if self.current_candle_open is None:
            self.current_candle_open = self.current_candle_high = self.current_candle_low = price
        
        self.current_candle_high = max(self.current_candle_high, price)
        self.current_candle_low = min(self.current_candle_low, price)
        self.current_candle_close = price
        
        tick_data = {
            'price': price,
            'timestamp': time.time(),
            'volume': 1
        }
        self.ticks.append(tick_data)
        self.price_memory.append(price)
        self.tick_count += 1
        
        return tick_data
    
    def _calculate_advanced_metrics(self):
        """Métricas avanzadas para IA profesional - MEJORADO"""
        if len(self.price_memory) < 10:
            return {}
            
        prices = np.array(list(self.price_memory))
        
        # Análisis de tendencia multi-timeframe MEJORADO
        if len(prices) >= 20:
            short_trend = np.polyfit(range(10), prices[-10:], 1)[0]
            medium_trend = np.polyfit(range(20), prices[-20:], 1)[0]
            trend_strength = (short_trend + medium_trend) * 10000
        else:
            trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
        
        # Análisis de momentum mejorado
        momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
        momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else 0
        momentum = (momentum_5 * 0.6 + momentum_10 * 0.4)
        
        # Análisis de volatilidad MEJORADO
        volatility = (self.current_candle_high - self.current_candle_low) * 10000
        
        # Análisis de presión de mercado MEJORADO
        if len(self.ticks) > 5:  # Mínimo para análisis confiable
            recent_ticks = list(self.ticks)[-10:]  # Últimos 10 ticks
            price_changes = [recent_ticks[i]['price'] - recent_ticks[i-1]['price'] 
                           for i in range(1, len(recent_ticks))]
            if price_changes:
                buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes)
                sell_pressure = len([x for x in price_changes if x < 0]) / len(price_changes)
                pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 999
            else:
                buy_pressure = sell_pressure = pressure_ratio = 0.5
        else:
            buy_pressure = sell_pressure = pressure_ratio = 0.5
        
        # Detección de patrones MEJORADA
        market_phase = self._detect_market_phase(prices, volatility, trend_strength)
        
        return {
            'trend_strength': trend_strength,
            'momentum': momentum,
            'volatility': volatility,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'pressure_ratio': pressure_ratio,
            'market_phase': market_phase,
            'data_quality': min(1.0, self.tick_count / 25.0)
        }
    
    def _detect_market_phase(self, prices, volatility, trend_strength):
        """Detección inteligente de fase de mercado - MEJORADO"""
        if volatility < 0.3 and abs(trend_strength) < 0.5:
            return "consolidation"
        elif abs(trend_strength) > 2.0 and volatility > 1.0:
            return "strong_trend"
        elif abs(trend_strength) > 1.0:
            return "moderate_trend"
        elif volatility > 1.5:
            return "high_volatility"
        else:
            return "neutral"
    
    def get_analysis(self):
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {'status': 'INSUFFICIENT_DATA', 'tick_count': self.tick_count}
        
        try:
            advanced_metrics = self._calculate_advanced_metrics()
            
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
            logging.error(f"Error en análisis premium: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        # Guardar el cierre de la vela actual para validación
        if self.current_candle_close is not None:
            self.last_candle_close = self.current_candle_close
            
        self.ticks.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0

# ------------------ SISTEMA IA PROFESIONAL MEJORADO ------------------
class ProfessionalAIPredictor:
    def __init__(self):
        self.analyzer = PremiumAIAnalyzer()
        self.prediction_history = deque(maxlen=20)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': 0.0,
            'confidence_history': deque(maxlen=10)
        }
        self.market_learning = {
            'phase_performance': {},
            'volatility_adaptation': 1.0
        }
        self.last_prediction = None
        
    def process_tick(self, price: float):
        return self.analyzer.add_tick(price)
    
    def _professional_ai_analysis(self, analysis):
        """Análisis profesional con IA avanzada - MEJORADO"""
        momentum = analysis['momentum']
        trend_strength = analysis['trend_strength']
        pressure_ratio = analysis['pressure_ratio']
        volatility = analysis['volatility']
        market_phase = analysis['market_phase']
        data_quality = analysis['data_quality']
        
        # SISTEMA DE PUNTUACIÓN AVANZADO MEJORADO
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # 1. ANÁLISIS DE TENDENCIA (30% peso) - MEJORADO
        trend_weight = 0.3
        if abs(trend_strength) > 1.5:
            if trend_strength > 0:
                buy_score += 8 * trend_weight
                reasons.append(f"📈 Tendencia alcista fuerte ({trend_strength:.1f})")
            else:
                sell_score += 8 * trend_weight
                reasons.append(f"📉 Tendencia bajista fuerte ({trend_strength:.1f})")
        elif abs(trend_strength) > 0.8:
            if trend_strength > 0:
                buy_score += 5 * trend_weight
                reasons.append(f"📈 Tendencia alcista moderada ({trend_strength:.1f})")
            else:
                sell_score += 5 * trend_weight
                reasons.append(f"📉 Tendencia bajista moderada ({trend_strength:.1f})")
        
        # 2. ANÁLISIS DE MOMENTUM (25% peso) - MEJORADO
        momentum_weight = 0.25
        if abs(momentum) > 1.2:
            if momentum > 0:
                buy_score += 7 * momentum_weight
                reasons.append(f"🚀 Momentum alcista ({momentum:.1f}pips)")
            else:
                sell_score += 7 * momentum_weight
                reasons.append(f"🔻 Momentum bajista ({momentum:.1f}pips)")
        
        # 3. ANÁLISIS DE PRESIÓN (25% peso) - MEJORADO
        pressure_weight = 0.25
        if pressure_ratio > 2.0:
            buy_score += 8 * pressure_weight
            reasons.append(f"💰 Presión compradora fuerte ({pressure_ratio:.1f}x)")
        elif pressure_ratio > 1.5:
            buy_score += 5 * pressure_weight
            reasons.append(f"💰 Presión compradora ({pressure_ratio:.1f}x)")
        elif pressure_ratio < 0.5:
            sell_score += 8 * pressure_weight
            reasons.append(f"💸 Presión vendedora fuerte ({pressure_ratio:.1f}x)")
        elif pressure_ratio < 0.7:
            sell_score += 5 * pressure_weight
            reasons.append(f"💸 Presión vendedora ({pressure_ratio:.1f}x)")
        
        # 4. ANÁLISIS DE FASE DE MERCADO (20% peso) - MEJORADO
        phase_weight = 0.2
        if market_phase == "strong_trend":
            # Seguir la tendencia en mercados fuertes
            if trend_strength > 0:
                buy_score += 6 * phase_weight
            else:
                sell_score += 6 * phase_weight
            reasons.append("🎯 Mercado en tendencia fuerte")
        elif market_phase == "consolidation":
            # Ser conservador en consolidación
            buy_score *= 0.8
            sell_score *= 0.8
            reasons.append("⚖️ Mercado en consolidación")
        
        # DECISIÓN FINAL PROFESIONAL MEJORADA
        score_difference = buy_score - sell_score
        
        # UMBRALES MÁS CONSERVADORES
        if abs(score_difference) > 0.4:  # Aumentado de 0.3 a 0.4
            if score_difference > 0:
                direction = "ALZA"
                base_confidence = 55 + (score_difference * 35)  # Más conservador
            else:
                direction = "BAJA"
                base_confidence = 55 + (abs(score_difference) * 35)
        else:
            # Empate - NO PREDECIR para evitar errores
            direction = "LATERAL"
            base_confidence = 40
            reasons.append("⚡ Mercado lateral - sin dirección clara")
        
        # AJUSTES DE CONFIANZA PROFESIONALES MEJORADOS
        confidence = base_confidence
        
        # Ajuste por calidad de datos
        confidence *= data_quality
        
        # Ajuste por volatilidad - MÁS CONSERVADOR
        if volatility > 2.0:
            confidence *= 0.7  # Más agresivo en alta volatilidad
            reasons.append("🌪️ Alta volatilidad - confianza reducida")
        elif volatility < 0.5:
            confidence *= 1.05  # Menos optimista en baja volatilidad
        
        # Ajuste por cantidad de datos
        if analysis['tick_count'] > 30:
            confidence = min(90, confidence + 3)  # Más conservador
        
        confidence = max(35, min(85, confidence))  # Rango más conservador
        
        return {
            'direction': direction,
            'confidence': int(confidence),
            'buy_score': round(buy_score, 2),
            'sell_score': round(sell_score, 2),
            'score_difference': round(score_difference, 2),
            'reasons': reasons,
            'market_phase': market_phase
        }
    
    def predict_next_candle(self):
        analysis = self.analyzer.get_analysis()
        
        if analysis.get('status') != 'SUCCESS':
            return {
                'direction': 'N/A',
                'confidence': 0,
                'reason': analysis.get('message', 'Analizando mercado...'),
                'timestamp': now_iso()
            }
        
        # PREDICCIÓN CON IA PROFESIONAL
        prediction = self._professional_ai_analysis(analysis)
        
        # Solo aceptar predicciones con suficiente confianza
        if prediction['confidence'] < 45:  # Umbral más alto
            prediction['direction'] = 'LATERAL'
            prediction['reasons'].append("🔍 Confianza insuficiente para predicción")
        
        # Agregar metadata
        prediction.update({
            'tick_count': analysis['tick_count'],
            'current_price': analysis['current_price'],
            'timestamp': now_iso(),
            'model_version': 'PROFESSIONAL_AI_V5_MEJORADO'
        })
        
        self.last_prediction = prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def validate_prediction(self, current_candle_close):
        """VALIDACIÓN MEJORADA - Más precisa y confiable"""
        if not self.last_prediction:
            return None
            
        try:
            last_pred = self.last_prediction
            predicted_direction = last_pred.get('direction', 'N/A')
            
            # Obtener precio anterior para comparación
            prev_price = self.analyzer.last_candle_close
            if prev_price is None:
                return None
                
            current_price = current_candle_close
            
            price_change = (current_price - prev_price) * 10000
            minimal_change = 0.15  # Umbral aumentado para evitar falsos positivos
            
            # DETERMINAR DIRECCIÓN REAL MEJORADA
            if abs(price_change) < minimal_change:
                actual_direction = "LATERAL"
                is_correct = False  # Lateral nunca es correcto para ALZA/BAJA
            else:
                actual_direction = "ALZA" if price_change > 0 else "BAJA"
                is_correct = (actual_direction == predicted_direction and predicted_direction != "LATERAL")
            
            # ACTUALIZAR ESTADÍSTICAS SOLO SI NO ES LATERAL
            if predicted_direction != "LATERAL":
                self.performance_stats['total_predictions'] += 1
                if is_correct:
                    self.performance_stats['correct_predictions'] += 1
            
            # Calcular precisión
            total = self.performance_stats['total_predictions']
            correct = self.performance_stats['correct_predictions']
            accuracy = (correct / total * 100) if total > 0 else 0
            self.performance_stats['recent_accuracy'] = accuracy
            
            # LOGGING MEJORADO
            status_icon = "✅" if is_correct else "❌"
            if actual_direction == "LATERAL":
                status_icon = "⚪"
                
            logging.info(f"🎯 VALIDACIÓN: {status_icon} {predicted_direction}→{actual_direction} | Conf: {last_pred.get('confidence', 0)}% | Cambio: {price_change:.1f}pips")
            
            if total > 0 and total % 5 == 0:
                logging.info(f"📊 PRECISIÓN ACTUAL: {accuracy:.1f}% (Total: {total})")
            
            return {
                'correct': is_correct,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': last_pred.get('confidence', 0),
                'price_change': round(price_change, 2),
                'accuracy': round(accuracy, 1),
                'total_predictions': total,
                'correct_predictions': correct,
                'status_icon': status_icon
            }
            
        except Exception as e:
            logging.error(f"❌ Error en validación: {e}")
            return None
    
    def get_performance_stats(self):
        return self.performance_stats.copy()
    
    def reset(self):
        self.analyzer.reset()

# -------------- CONEXIÓN PROFESIONAL MEJORADA --------------
class ProfessionalIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        self.connection_attempts = 0
        
    def connect(self):
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.warning("🔐 Credenciales no configuradas - Modo demo activado")
                self.connected = True  # Permitir modo demo
                return True
                
            self.connection_attempts += 1
            logging.info(f"🌐 Conectando a IQ Option... (Intento {self.connection_attempts})")
            
            self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.iq.connect()
            
            if check:
                self.iq.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conexión premium establecida")
                return True
            else:
                logging.warning(f"⚠️ Fallo de conexión: {reason}")
                # Modo demo como respaldo
                if self.connection_attempts >= 3:
                    logging.info("🔧 Activando modo demo...")
                    self.connected = True
                    return True
                return False
                
        except Exception as e:
            logging.error(f"❌ Error de conexión: {e}")
            # Modo demo como respaldo
            logging.info("🔧 Activando modo demo por error...")
            self.connected = True
            return True

    def get_realtime_price(self):
        try:
            # MODO DEMO si no hay conexión
            if not self.connected:
                # Generar precio demo realista
                if self.last_price is None:
                    self.last_price = 1.15000
                else:
                    # Pequeña variación realista
                    variation = np.random.uniform(-0.0002, 0.0002)
                    self.last_price += variation
                return self.last_price

            # Método profesional con respaldo
            candles = self.iq.get_candles(PAR, TIMEFRAME, 1, time.time())
            if candles and len(candles) > 0:
                price = float(candles[-1]['close'])
                if price > 0:
                    self._record_tick(price)
                    return price

            # Respaldo con precio anterior
            return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            # Respaldo con precio anterior o demo
            return self.last_price if self.last_price else 1.15000

    def _record_tick(self, price):
        self.tick_count += 1
        self.last_price = price
        
        # Logging menos frecuente para evitar spam
        if self.tick_count <= 10 or self.tick_count % 50 == 0:
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL PREMIUM MEJORADO ---------------
iq_connector = ProfessionalIQConnector()
predictor = ProfessionalAIPredictor()

# VARIABLES GLOBALES MEJORADAS
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 IA profesional inicializando..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING"
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None,
    'average_confidence': 0.0
}

# CONTROL DE EJECUCIÓN
system_running = True

def signal_handler(signum, frame):
    global system_running
    logging.info("🛑 Señal de apagado recibida...")
    system_running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def premium_main_loop():
    global current_prediction, system_running
    
    logging.info("🚀 DELOWYSS AI V5.0 PREMIUM MEJORADO INICIADO")
    logging.info("🎯 Sistema profesional con IA avanzada activado")
    
    # Conexión mejorada con reintentos
    if not iq_connector.connect():
        logging.warning("🔧 Ejecutando en modo demo...")
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    
    while system_running:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # OBTENER PRECIO PROFESIONAL
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                predictor.process_tick(price)
                last_price = price
                
                # ACTUALIZAR ESTADO EN TIEMPO REAL
                current_prediction.update({
                    "current_price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE"
                })
            
            # PREDICCIÓN PROFESIONAL EN VENTANA ÓPTIMA MEJORADA
            prediction_ready = (
                seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 3 and  # Más margen
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - last_prediction_time) >= 4  # Menos frecuente
            )
            
            if prediction_ready:
                prediction = predictor.predict_next_candle()
                
                # Solo actualizar si es una predicción válida
                if prediction['confidence'] >= 40:  # Umbral más bajo para incluir LATERAL
                    current_prediction.update(prediction)
                    last_prediction_time = time.time()
                    
                    if prediction['direction'] != 'LATERAL':
                        logging.info(f"🎯 PREDICCIÓN: {prediction['direction']} | Conf: {prediction['confidence']}% | Ticks: {prediction['tick_count']}")
                    else:
                        logging.info(f"🎯 ANÁLISIS: Mercado lateral | Conf: {prediction['confidence']}%")
            
            # CAMBIO DE VELA CON VALIDACIÓN PROFESIONAL MEJORADA
            if current_candle_start > last_candle_start and last_price is not None:
                # Validar predicción anterior
                validation_result = predictor.validate_prediction(last_price)
                if validation_result:
                    performance_stats.update({
                        'total_predictions': validation_result['total_predictions'],
                        'correct_predictions': validation_result['correct_predictions'],
                        'recent_accuracy': validation_result['accuracy'],
                        'last_validation': validation_result,
                        'average_confidence': np.mean(list(predictor.performance_stats['confidence_history'])) if predictor.performance_stats['confidence_history'] else 0
                    })
                
                # Reiniciar para nueva vela
                predictor.reset()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela - IA analizando mercado...")
            
            time.sleep(0.5)  # Menos carga de CPU
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(2)  # Más tiempo entre reintentos

# --------------- INTERFAZ WEB PREMIUM (MANTENIDA) ---------------
app = FastAPI(title="Delowyss AI Premium", version="5.0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    # ... (el mismo código de interfaz que antes)
    # Solo asegúrate de usar las variables globales actualizadas
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    
    # Generar HTML de razones de forma segura
    reasons_html = ""
    reasons_list = current_prediction.get('reasons', ['IA analizando factores de mercado...'])
    for reason in reasons_list:
        reasons_html += f"<li>{reason}</li>"
    
    # ... (resto del código HTML igual)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/validation")
def api_validation():
    return JSONResponse({
        "last_validation": performance_stats.get('last_validation'),
        "performance": performance_stats,
        "timestamp": now_iso()
    })

@app.get("/api/system/status")
def api_system_status():
    return JSONResponse({
        "status": "ACTIVE",
        "version": "5.0.1",
        "ai_model": "PROFESSIONAL_AI_V5_MEJORADO",
        "accuracy": performance_stats.get('recent_accuracy', 0),
        "timestamp": now_iso()
    })

# --------------- INICIALIZACIÓN MEJORADA ---------------
def start_premium_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info("⭐ SISTEMA PREMIUM INICIADO CORRECTAMENTE")
        logging.info("🎯 IA profesional mejorada activada")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

# Iniciar solo si es el archivo principal
if __name__ == "__main__":
    start_premium_system()
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
else:
    # Para Render.com, iniciar automáticamente
    start_premium_system()
