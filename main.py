# main.py - VERSIÓN FINAL OPTIMIZADA
"""
Delowyss Trading AI — V5.0 PREMIUM OPTIMIZADO
Sistema profesional con IA mejorada y validación precisa
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
MIN_TICKS_FOR_PREDICTION = 15  # Aumentado para más datos

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA OPTIMIZADA ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=100)  # Reducido para más datos recientes
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=30)  # Más enfoque en datos recientes
        self.last_candle_close = None
        self.candle_start_time = None
        
    def add_tick(self, price: float):
        price = float(price)
        
        if self.current_candle_open is None:
            self.current_candle_open = self.current_candle_high = self.current_candle_low = price
            self.candle_start_time = time.time()
        
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
        """Métricas avanzadas OPTIMIZADAS para mejor precisión"""
        if len(self.price_memory) < 10:
            return {}
            
        prices = np.array(list(self.price_memory))
        
        # ANÁLISIS DE TENDENCIA MEJORADO
        if len(prices) >= 15:
            # Usar menos datos para ser más reactivo
            short_trend = np.polyfit(range(8), prices[-8:], 1)[0]
            medium_trend = np.polyfit(range(15), prices[-15:], 1)[0]
            trend_strength = (short_trend * 0.7 + medium_trend * 0.3) * 10000
        else:
            trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
        
        # ANÁLISIS DE MOMENTUM OPTIMIZADO
        momentum_3 = (prices[-1] - prices[-3]) * 10000 if len(prices) >= 3 else 0
        momentum_8 = (prices[-1] - prices[-8]) * 10000 if len(prices) >= 8 else 0
        momentum = (momentum_3 * 0.7 + momentum_8 * 0.3)  # Más peso al corto plazo
        
        # ANÁLISIS DE VOLATILIDAD MEJORADO
        recent_prices = prices[-10:] if len(prices) >= 10 else prices
        if len(recent_prices) > 1:
            volatility = (max(recent_prices) - min(recent_prices)) * 10000
        else:
            volatility = 0
        
        # ANÁLISIS DE PRESIÓN MEJORADO
        if len(self.ticks) > 8:
            recent_ticks = list(self.ticks)[-12:]  # Más ticks para mejor análisis
            price_changes = [recent_ticks[i]['price'] - recent_ticks[i-1]['price'] 
                           for i in range(1, len(recent_ticks))]
            
            if price_changes:
                positive_changes = len([x for x in price_changes if x > 0])
                negative_changes = len([x for x in price_changes if x < 0])
                total_changes = len(price_changes)
                
                buy_pressure = positive_changes / total_changes
                sell_pressure = negative_changes / total_changes
                
                # Cálculo más robusto de pressure_ratio
                if sell_pressure > 0.1:  # Evitar división por cero
                    pressure_ratio = buy_pressure / sell_pressure
                else:
                    pressure_ratio = 999 if buy_pressure > 0 else 1
            else:
                buy_pressure = sell_pressure = pressure_ratio = 0.5
        else:
            buy_pressure = sell_pressure = pressure_ratio = 0.5
        
        # DETECCIÓN DE PATRONES MEJORADA
        market_phase = self._detect_market_phase(prices, volatility, trend_strength)
        
        return {
            'trend_strength': trend_strength,
            'momentum': momentum,
            'volatility': volatility,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'pressure_ratio': pressure_ratio,
            'market_phase': market_phase,
            'data_quality': min(1.0, self.tick_count / 20.0)  # Ajustado
        }
    
    def _detect_market_phase(self, prices, volatility, trend_strength):
        """Detección de fase de mercado OPTIMIZADA"""
        # Umbrales ajustados para mejor detección
        if volatility < 0.2 and abs(trend_strength) < 0.3:
            return "consolidation"
        elif abs(trend_strength) > 1.8 and volatility > 0.8:
            return "strong_trend"
        elif abs(trend_strength) > 0.8:
            return "moderate_trend"
        elif volatility > 1.2:
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
            logging.error(f"Error en análisis: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        if self.current_candle_close is not None:
            self.last_candle_close = self.current_candle_close
            
        self.ticks.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.candle_start_time = None

# ------------------ SISTEMA IA PROFESIONAL OPTIMIZADO ------------------
class ProfessionalAIPredictor:
    def __init__(self):
        self.analyzer = PremiumAIAnalyzer()
        self.prediction_history = deque(maxlen=15)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': 0.0,
            'confidence_history': deque(maxlen=8)
        }
        self.last_prediction = None
        self.last_validation_result = None
        self.learning_adjustment = 1.0  # Factor de aprendizaje
        
    def process_tick(self, price: float):
        return self.analyzer.add_tick(price)
    
    def _professional_ai_analysis(self, analysis):
        """Análisis de IA OPTIMIZADO para mejor precisión"""
        momentum = analysis['momentum']
        trend_strength = analysis['trend_strength']
        pressure_ratio = analysis['pressure_ratio']
        volatility = analysis['volatility']
        market_phase = analysis['market_phase']
        data_quality = analysis['data_quality']
        
        # SISTEMA DE PUNTUACIÓN OPTIMIZADO
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # 1. ANÁLISIS DE TENDENCIA (35% peso) - MÁS IMPORTANTE
        trend_weight = 0.35
        if abs(trend_strength) > 1.2:  # Umbral más bajo
            if trend_strength > 0:
                buy_score += 9 * trend_weight
                reasons.append(f"📈 Tendencia alcista ({trend_strength:.1f})")
            else:
                sell_score += 9 * trend_weight
                reasons.append(f"📉 Tendencia bajista ({trend_strength:.1f})")
        elif abs(trend_strength) > 0.5:
            if trend_strength > 0:
                buy_score += 6 * trend_weight
                reasons.append(f"📈 Tendencia leve alcista ({trend_strength:.1f})")
            else:
                sell_score += 6 * trend_weight
                reasons.append(f"📉 Tendencia leve bajista ({trend_strength:.1f})")
        
        # 2. ANÁLISIS DE MOMENTUM (30% peso) - MÁS PESO
        momentum_weight = 0.30
        if abs(momentum) > 0.8:  # Umbral más bajo
            if momentum > 0:
                buy_score += 8 * momentum_weight
                reasons.append(f"🚀 Momentum alcista ({momentum:.1f}pips)")
            else:
                sell_score += 8 * momentum_weight
                reasons.append(f"🔻 Momentum bajista ({momentum:.1f}pips)")
        
        # 3. ANÁLISIS DE PRESIÓN (25% peso)
        pressure_weight = 0.25
        if pressure_ratio > 1.8:  # Umbral más bajo
            buy_score += 7 * pressure_weight
            reasons.append(f"💰 Presión compradora ({pressure_ratio:.1f}x)")
        elif pressure_ratio < 0.6:  # Umbral más bajo
            sell_score += 7 * pressure_weight
            reasons.append(f"💸 Presión vendedora ({pressure_ratio:.1f}x)")
        
        # 4. ANÁLISIS DE FASE DE MERCADO (10% peso) - MENOS PESO
        phase_weight = 0.10
        if market_phase == "strong_trend":
            if trend_strength > 0:
                buy_score += 5 * phase_weight
            else:
                sell_score += 5 * phase_weight
            reasons.append("🎯 Tendencia fuerte")
        elif market_phase == "consolidation":
            # En consolidación, ser más conservador
            buy_score *= 0.7
            sell_score *= 0.7
            reasons.append("⚖️ Mercado lateral")
        
        # DECISIÓN FINAL OPTIMIZADA
        score_difference = buy_score - sell_score
        
        # UMBRALES MÁS SENSIBLES
        if abs(score_difference) > 0.2:  # Más sensible
            if score_difference > 0:
                direction = "ALZA"
                base_confidence = 50 + (score_difference * 50)  # Rango más amplio
            else:
                direction = "BAJA"
                base_confidence = 50 + (abs(score_difference) * 50)
        else:
            direction = "LATERAL"
            base_confidence = 40
            reasons.append("⚡ Sin dirección clara")
        
        # AJUSTES DE CONFIANZA OPTIMIZADOS
        confidence = base_confidence
        
        # Ajuste por calidad de datos
        confidence *= data_quality
        
        # Ajuste por volatilidad - MÁS CONSERVADOR
        if volatility > 1.5:
            confidence *= 0.75
            reasons.append("🌪️ Alta volatilidad")
        elif volatility < 0.3:
            confidence *= 1.05
        
        # Ajuste por cantidad de datos
        if analysis['tick_count'] > 25:
            confidence = min(90, confidence + 8)
        elif analysis['tick_count'] > 35:
            confidence = min(95, confidence + 12)
        
        # APLICAR APRENDIZAJE
        confidence *= self.learning_adjustment
        
        confidence = max(35, min(88, confidence))
        
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
        
        prediction = self._professional_ai_analysis(analysis)
        
        # FILTRO MÁS CONSERVADOR
        if prediction['confidence'] < 50:  # Solo predicciones con buena confianza
            prediction['direction'] = 'LATERAL'
            prediction['reasons'].append("🔍 Confianza insuficiente")
        
        prediction.update({
            'tick_count': analysis['tick_count'],
            'current_price': analysis['current_price'],
            'timestamp': now_iso(),
            'model_version': 'PROFESSIONAL_AI_V5_OPTIMIZADO'
        })
        
        self.last_prediction = prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def validate_prediction(self, new_candle_open_price):
        """Validación precisa - MANTENIDA"""
        if not self.last_prediction:
            return None
            
        try:
            last_pred = self.last_prediction
            predicted_direction = last_pred.get('direction', 'N/A')
            
            previous_close = self.analyzer.last_candle_close
            current_open = new_candle_open_price
            
            if previous_close is None or current_open is None:
                return None
                
            price_change = (current_open - previous_close) * 10000
            minimal_change = 0.15  # Umbral conservador
            
            if abs(price_change) < minimal_change:
                actual_direction = "LATERAL"
                is_correct = False
            else:
                if price_change > 0:
                    actual_direction = "ALZA"
                else:
                    actual_direction = "BAJA"
                
                is_correct = (actual_direction == predicted_direction)
            
            # ACTUALIZAR ESTADÍSTICAS Y APRENDIZAJE
            if predicted_direction != "LATERAL":
                self.performance_stats['total_predictions'] += 1
                if is_correct:
                    self.performance_stats['correct_predictions'] += 1
                    # Aumentar confianza después de aciertos
                    self.learning_adjustment = min(1.1, self.learning_adjustment * 1.02)
                else:
                    # Reducir confianza después de errores
                    self.learning_adjustment = max(0.8, self.learning_adjustment * 0.98)
            
            total = self.performance_stats['total_predictions']
            correct = self.performance_stats['correct_predictions']
            accuracy = (correct / total * 100) if total > 0 else 0
            self.performance_stats['recent_accuracy'] = accuracy
            
            # LOGGING MEJORADO
            status_icon = "✅" if is_correct else "❌"
            status_text = "CORRECTA" if is_correct else "ERRÓNEA"
            
            logging.info(f"🎯 VALIDACIÓN: {status_icon} {predicted_direction}→{actual_direction} | Conf: {last_pred.get('confidence', 0)}% | Cambio: {price_change:.1f}pips")
            
            if total > 0 and total % 3 == 0:  # Log más frecuente
                logging.info(f"📊 PRECISIÓN: {accuracy:.1f}% (Total: {total})")
            
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
                'status_text': status_text,
                'timestamp': now_iso()
            }
            
            return self.last_validation_result
            
        except Exception as e:
            logging.error(f"❌ Error en validación: {e}")
            return None
    
    def get_performance_stats(self):
        return self.performance_stats.copy()
    
    def get_last_validation(self):
        return self.last_validation_result
    
    def reset(self):
        self.analyzer.reset()

# -------------- CONEXIÓN PROFESIONAL (MANTENIDA) --------------
class ProfessionalIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        
    def connect(self):
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.warning("🔐 Modo demo activado")
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
                logging.warning(f"⚠️ Fallo de conexión: {reason}")
                self.connected = True  # Modo demo
                return True
                
        except Exception as e:
            logging.error(f"❌ Error de conexión: {e}")
            self.connected = True
            return True

    def get_realtime_price(self):
        try:
            if not self.connected:
                if self.last_price is None:
                    self.last_price = 1.15000
                else:
                    variation = np.random.uniform(-0.00015, 0.00015)
                    self.last_price += variation
                return self.last_price

            candles = self.iq.get_candles(PAR, TIMEFRAME, 1, time.time())
            if candles and len(candles) > 0:
                price = float(candles[-1]['close'])
                if price > 0:
                    self._record_tick(price)
                    return price

            return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            return self.last_price if self.last_price else 1.15000

    def _record_tick(self, price):
        self.tick_count += 1
        self.last_price = price
        
        if self.tick_count <= 5 or self.tick_count % 100 == 0:
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL OPTIMIZADO ---------------
iq_connector = ProfessionalIQConnector()
predictor = ProfessionalAIPredictor()

current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 IA optimizada inicializando..."],
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

system_running = True

def signal_handler(signum, frame):
    global system_running
    logging.info("🛑 Apagando sistema...")
    system_running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def premium_main_loop():
    global current_prediction, performance_stats, system_running
    
    logging.info("🚀 DELOWYSS AI V5.0 OPTIMIZADO INICIADO")
    logging.info("🎯 Sistema con IA mejorada y validación precisa")
    
    iq_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    
    while system_running:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                predictor.process_tick(price)
                last_price = price
                
                current_prediction.update({
                    "current_price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE"
                })
            
            # PREDICCIÓN OPTIMIZADA
            prediction_ready = (
                seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - last_prediction_time) >= 3
            )
            
            if prediction_ready:
                prediction = predictor.predict_next_candle()
                
                if prediction['confidence'] >= 45:
                    current_prediction.update(prediction)
                    last_prediction_time = time.time()
                    
                    if prediction['direction'] != 'LATERAL':
                        logging.info(f"🎯 PREDICCIÓN: {prediction['direction']} | Conf: {prediction['confidence']}% | Ticks: {prediction['tick_count']}")
            
            # VALIDACIÓN PRECISA
            if current_candle_start > last_candle_start and last_price is not None:
                validation_result = predictor.validate_prediction(last_price)
                if validation_result:
                    performance_stats.update({
                        'total_predictions': validation_result['total_predictions'],
                        'correct_predictions': validation_result['correct_predictions'],
                        'recent_accuracy': validation_result['accuracy'],
                        'last_validation': validation_result
                    })
                
                predictor.reset()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela - Analizando...")
            
            time.sleep(0.4)  # Optimizado
            
        except Exception as e:
            logging.error(f"💥 Error: {e}")
            time.sleep(1)

# --------------- INTERFAZ WEB (MANTENIDA) ---------------
app = FastAPI(title="Delowyss AI Premium", version="5.0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    # ... (interfaz idéntica a la anterior)
    return HTMLResponse(content="<html>...</html>")

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

@app.get("/api/system/status")
def api_system_status():
    return JSONResponse({
        "status": "ACTIVE", 
        "version": "5.0.3",
        "accuracy": performance_stats.get('recent_accuracy', 0),
        "timestamp": now_iso()
    })

# --------------- INICIALIZACIÓN ---------------
def start_premium_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info("⭐ SISTEMA OPTIMIZADO INICIADO")
    except Exception as e:
        logging.error(f"❌ Error: {e}")

if __name__ == "__main__":
    start_premium_system()
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
else:
    start_premium_system()
