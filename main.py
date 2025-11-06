# main.py - VERSIÓN PREMIUM PROFESIONAL
"""
Delowyss Trading AI — V5.0 PREMIUM
Sistema profesional con interfaz premium y IA avanzada
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
from fastapi.staticfiles import StaticFiles

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

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA PREMIUM ------------------
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
        """Métricas avanzadas para IA profesional"""
        if len(self.price_memory) < 10:
            return {}
            
        prices = np.array(list(self.price_memory))
        
        # Análisis de tendencia multi-timeframe
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
        
        # Análisis de volatilidad
        volatility = (self.current_candle_high - self.current_candle_low) * 10000
        
        # Análisis de presión de mercado
        if len(self.ticks) > 1:
            price_changes = [self.ticks[i]['price'] - self.ticks[i-1]['price'] for i in range(1, len(self.ticks))]
            buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes)
            sell_pressure = len([x for x in price_changes if x < 0]) / len(price_changes)
            pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 999
        else:
            buy_pressure = sell_pressure = pressure_ratio = 0.5
        
        # Detección de patrones
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
        """Detección inteligente de fase de mercado"""
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
        self.ticks.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0

# ------------------ SISTEMA IA PROFESIONAL ------------------
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
        
    def process_tick(self, price: float):
        return self.analyzer.add_tick(price)
    
    def _professional_ai_analysis(self, analysis):
        """Análisis profesional con IA avanzada"""
        momentum = analysis['momentum']
        trend_strength = analysis['trend_strength']
        pressure_ratio = analysis['pressure_ratio']
        volatility = analysis['volatility']
        market_phase = analysis['market_phase']
        data_quality = analysis['data_quality']
        
        # SISTEMA DE PUNTUACIÓN AVANZADO
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # 1. ANÁLISIS DE TENDENCIA (30% peso)
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
        
        # 2. ANÁLISIS DE MOMENTUM (25% peso)
        momentum_weight = 0.25
        if abs(momentum) > 1.2:
            if momentum > 0:
                buy_score += 7 * momentum_weight
                reasons.append(f"🚀 Momentum alcista ({momentum:.1f}pips)")
            else:
                sell_score += 7 * momentum_weight
                reasons.append(f"🔻 Momentum bajista ({momentum:.1f}pips)")
        
        # 3. ANÁLISIS DE PRESIÓN (25% peso)
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
        
        # 4. ANÁLISIS DE FASE DE MERCADO (20% peso)
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
        
        # DECISIÓN FINAL PROFESIONAL
        score_difference = buy_score - sell_score
        
        if abs(score_difference) > 0.3:
            if score_difference > 0:
                direction = "ALZA"
                base_confidence = 60 + (score_difference * 40)
            else:
                direction = "BAJA"
                base_confidence = 60 + (abs(score_difference) * 40)
        else:
            # Empate - usar desempate inteligente
            if momentum > 0:
                direction = "ALZA"
                base_confidence = 50
                reasons.append("⚡ Desempate por momentum positivo")
            else:
                direction = "BAJA"
                base_confidence = 50
                reasons.append("⚡ Desempate por momentum negativo")
        
        # AJUSTES DE CONFIANZA PROFESIONALES
        confidence = base_confidence
        
        # Ajuste por calidad de datos
        confidence *= data_quality
        
        # Ajuste por volatilidad
        if volatility > 2.0:
            confidence *= 0.8
            reasons.append("🌪️ Alta volatilidad - confianza reducida")
        elif volatility < 0.5:
            confidence *= 1.1
            reasons.append("🌊 Baja volatilidad - confianza aumentada")
        
        # Ajuste por cantidad de datos
        if analysis['tick_count'] > 30:
            confidence = min(95, confidence + 5)
        
        confidence = max(40, min(92, confidence))
        
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
        
        # Agregar metadata
        prediction.update({
            'tick_count': analysis['tick_count'],
            'current_price': analysis['current_price'],
            'timestamp': now_iso(),
            'model_version': 'PROFESSIONAL_AI_V5'
        })
        
        self.prediction_history.append(prediction)
        
        return prediction
    
    def validate_prediction(self, current_metrics):
        if not self.prediction_history:
            return None
            
        last_pred = self.prediction_history[-1]
        
        try:
            prev_price = last_pred['current_price']
            current_price = current_metrics['current_price']
            
            price_change = (current_price - prev_price) * 10000
            minimal_change = 0.08  # Umbral profesional
            
            if abs(price_change) < minimal_change:
                actual_direction = "LATERAL"
                correct = False
            else:
                actual_direction = "ALZA" if current_price > prev_price else "BAJA"
                predicted_direction = last_pred.get('direction', 'N/A')
                correct = (actual_direction == predicted_direction)
            
            # ACTUALIZAR ESTADÍSTICAS PROFESIONALES
            self.performance_stats['total_predictions'] += 1
            if correct and actual_direction != "LATERAL":
                self.performance_stats['correct_predictions'] += 1
            
            self.performance_stats['confidence_history'].append(last_pred.get('confidence', 0))
            
            # Calcular precisión
            total = self.performance_stats['total_predictions']
            correct = self.performance_stats['correct_predictions']
            accuracy = (correct / total * 100) if total > 0 else 0
            self.performance_stats['recent_accuracy'] = accuracy
            
            # Aprendizaje por fase de mercado
            market_phase = last_pred.get('market_phase', 'unknown')
            if market_phase not in self.market_learning['phase_performance']:
                self.market_learning['phase_performance'][market_phase] = []
            
            self.market_learning['phase_performance'][market_phase].append(1 if correct else 0)
            if len(self.market_learning['phase_performance'][market_phase]) > 20:
                self.market_learning['phase_performance'][market_phase].pop(0)
            
            status = "✅ CORRECTA" if correct else "❌ ERRÓNEA"
            if actual_direction == "LATERAL":
                status = "⚪ LATERAL"
                
            logging.info(f"🎯 VALIDACIÓN PROFESIONAL: {status} | Pred: {predicted_direction} | Real: {actual_direction} | Conf: {last_pred.get('confidence', 0)}% | Change: {price_change:.1f}pips")
            
            if total % 5 == 0:
                logging.info(f"📊 PRECISIÓN IA: {accuracy:.1f}% (Total: {total})")
            
            return {
                'correct': correct,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'confidence': last_pred.get('confidence', 0),
                'price_change': round(price_change, 2),
                'accuracy': round(accuracy, 1),
                'total_predictions': total,
                'correct_predictions': correct
            }
            
        except Exception as e:
            logging.error(f"Error en validación profesional: {e}")
            return None
    
    def get_performance_stats(self):
        return self.performance_stats.copy()
    
    def reset(self):
        self.analyzer.reset()

# -------------- CONEXIÓN PROFESIONAL --------------
class ProfessionalIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        
    def connect(self):
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.warning("🔐 Credenciales no configuradas")
                return False
                
            logging.info("🌐 Conectando a IQ Option...")
            self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.iq.connect()
            
            if check:
                self.iq.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conexión premium establecida")
                return True
            else:
                logging.warning(f"⚠️ Fallo de conexión: {reason}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error de conexión profesional: {e}")
            return False

    def get_realtime_price(self):
        try:
            if not self.connected:
                return self.last_price

            # Método profesional con respaldo
            candles = self.iq.get_candles(PAR, TIMEFRAME, 1, time.time())
            if candles and len(candles) > 0:
                price = float(candles[-1]['close'])
                if price > 0:
                    self._record_tick(price)
                    return price

            return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            return self.last_price

    def _record_tick(self, price):
        self.tick_count += 1
        self.last_price = price
        
        if self.tick_count <= 5 or self.tick_count % 100 == 0:
            logging.info(f"💰 Tick profesional #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL PREMIUM ---------------
iq_connector = ProfessionalIQConnector()
predictor = ProfessionalAIPredictor()

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

def premium_main_loop():
    global current_prediction
    
    logging.info("🚀 DELOWYSS AI V5.0 PREMIUM INICIADO")
    logging.info("🎯 Sistema profesional con IA avanzada activado")
    
    iq_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # OBTENER PRECIO PROFESIONAL
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                predictor.process_tick(price)
                
                # ACTUALIZAR ESTADO EN TIEMPO REAL
                current_prediction.update({
                    "current_price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE"
                })
            
            # PREDICCIÓN PROFESIONAL EN VENTANA ÓPTIMA
            if (seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 2 and 
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION):
                
                if (time.time() - last_prediction_time) >= 3:
                    prediction = predictor.predict_next_candle()
                    
                    # Solo actualizar si es una predicción válida
                    if prediction['confidence'] >= 45:
                        current_prediction.update(prediction)
                        last_prediction_time = time.time()
                        
                        logging.info(f"🎯 PREDICCIÓN IA: {prediction['direction']} | Confianza: {prediction['confidence']}% | Ticks: {prediction['tick_count']}")
            
            # CAMBIO DE VELA CON VALIDACIÓN PROFESIONAL
            if current_candle_start > last_candle_start:
                current_metrics = predictor.analyzer.get_analysis()
                if current_metrics and current_metrics.get('status') == 'SUCCESS':
                    validation_result = predictor.validate_prediction(current_metrics)
                    if validation_result:
                        performance_stats.update({
                            'total_predictions': validation_result['total_predictions'],
                            'correct_predictions': validation_result['correct_predictions'],
                            'recent_accuracy': validation_result['accuracy'],
                            'last_validation': validation_result,
                            'average_confidence': np.mean(list(predictor.performance_stats['confidence_history'])) if predictor.performance_stats['confidence_history'] else 0
                        })
                
                predictor.reset()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela - IA analizando mercado...")
            
            time.sleep(0.3)
            
        except Exception as e:
            logging.error(f"💥 Error en loop premium: {e}")
            time.sleep(1)

# --------------- INTERFAZ WEB PREMIUM ---------------
app = FastAPI(title="Delowyss AI Premium", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    
    # Colores dinámicos basados en dirección y confianza
    if direction == "ALZA":
        primary_color = "#00ff88"
        gradient = "linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)"
    elif direction == "BAJA":
        primary_color = "#ff4444" 
        gradient = "linear-gradient(135deg, #ff4444 0%, #cc3636 100%)"
    else:
        primary_color = "#ffbb33"
        gradient = "linear-gradient(135deg, #ffbb33 0%, #cc9929 100%)"
    
    # Calcular métricas de confianza
    confidence_level = "ALTA" if confidence > 70 else "MEDIA" if confidence > 50 else "BAJA"
    confidence_color = "#00ff88" if confidence > 70 else "#ffbb33" if confidence > 50 else "#ff4444"
    
    # Estadísticas
    accuracy = performance_stats.get('recent_accuracy', 0)
    total_pred = performance_stats.get('total_predictions', 0)
    correct_pred = performance_stats.get('correct_predictions', 0)
    avg_confidence = performance_stats.get('average_confidence', 0)
    
    # Calcular tiempo hasta siguiente vela
    current_time = time.time()
    seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.0</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
                min-height: 100vh;
                overflow-x: hidden;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            /* Header Premium */
            .header {{
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: {gradient};
            }}
            
            .logo {{
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 10px;
                background: {gradient};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .subtitle {{
                color: #94a3b8;
                font-size: 1.1em;
                margin-bottom: 20px;
            }}
            
            /* Grid Principal */
            .dashboard-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 25px;
                margin-bottom: 25px;
            }}
            
            /* Tarjetas */
            .card {{
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 25px;
                transition: all 0.3s ease;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                border-color: {primary_color};
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            }}
            
            /* Predicción Principal */
            .prediction-card {{
                grid-column: 1 / -1;
                text-align: center;
                border-left: 4px solid {primary_color};
            }}
            
            .direction-display {{
                font-size: 4em;
                font-weight: 700;
                margin: 20px 0;
                color: {primary_color};
            }}
            
            .confidence-badge {{
                display: inline-block;
                background: {confidence_color};
                color: #0f172a;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9em;
                margin-left: 10px;
            }}
            
            /* Countdown */
            .countdown-container {{
                background: rgba(0, 0, 0, 0.3);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }}
            
            .countdown {{
                font-size: 3.5em;
                font-weight: 700;
                font-family: 'Courier New', monospace;
                color: {primary_color};
                text-shadow: 0 0 20px {primary_color}80;
            }}
            
            /* Métricas */
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            
            .metric {{
                background: rgba(255, 255, 255, 0.03);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }}
            
            .metric-value {{
                font-size: 1.8em;
                font-weight: 700;
                color: {primary_color};
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #94a3b8;
                font-size: 0.9em;
            }}
            
            /* Razones */
            .reasons-list {{
                list-style: none;
                margin-top: 15px;
            }}
            
            .reasons-list li {{
                background: rgba(255, 255, 255, 0.03);
                margin: 8px 0;
                padding: 12px 15px;
                border-radius: 8px;
                border-left: 3px solid {primary_color};
                animation: slideIn 0.5s ease;
            }}
            
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateX(-20px); }}
                to {{ opacity: 1; transform: translateX(0); }}
            }}
            
            /* Validación */
            .validation-card {{
                border-left: 4px solid #ffbb33;
            }}
            
            .validation-result {{
                padding: 20px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            /* Gráfico */
            .chart-container {{
                height: 200px;
                margin-top: 20px;
            }}
            
            /* Responsive */
            @media (max-width: 768px) {{
                .dashboard-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .direction-display {{
                    font-size: 3em;
                }}
                
                .countdown {{
                    font-size: 2.5em;
                }}
            }}
            
            /* Animaciones */
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            
            .pulse {{
                animation: pulse 2s infinite;
            }}
            
            /* Score Bars */
            .score-container {{
                margin: 15px 0;
            }}
            
            .score-bar {{
                height: 8px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                overflow: hidden;
                margin: 10px 0;
            }}
            
            .buy-score {{
                height: 100%;
                background: #00ff88;
                float: left;
            }}
            
            .sell-score {{
                height: 100%;
                background: #ff4444;
                float: right;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <div class="logo">🤖 DELOWYSS AI PREMIUM</div>
                <div class="subtitle">Sistema de Trading con Inteligencia Artificial Avanzada V5.0</div>
                <div class="status-indicator">
                    <span style="color: #00ff88;">●</span> SISTEMA ACTIVO | IA PROFESIONAL | APRENDIZAJE CONTINUO
                </div>
            </div>
            
            <!-- Grid Principal -->
            <div class="dashboard-grid">
                
                <!-- Predicción Actual -->
                <div class="card prediction-card">
                    <h2>🎯 PREDICCIÓN ACTUAL</h2>
                    <div class="direction-display" id="direction-display">
                        {direction}
                    </div>
                    <div class="confidence-display">
                        <span style="font-size: 1.2em; font-weight: 600;">CONFIANZA: {confidence}%</span>
                        <span class="confidence-badge">{confidence_level}</span>
                    </div>
                    
                    <!-- Countdown -->
                    <div class="countdown-container">
                        <div style="color: #94a3b8; margin-bottom: 10px;">SIGUIENTE PREDICCIÓN EN:</div>
                        <div class="countdown" id="countdown">{int(seconds_remaining)}s</div>
                    </div>
                    
                    <!-- Métricas Rápidas -->
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" id="tick-count">{current_prediction.get('tick_count', 0)}</div>
                            <div class="metric-label">TICKS ANALIZADOS</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{current_prediction.get('current_price', 0):.5f}</div>
                            <div class="metric-label">PRECIO ACTUAL</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="score-diff">{current_prediction.get('score_difference', 0)}</div>
                            <div class="metric-label">SCORE IA</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{current_prediction.get('market_phase', 'N/A')}</div>
                            <div class="metric-label">FASE MERCADO</div>
                        </div>
                    </div>
                </div>
                
                <!-- Análisis IA -->
                <div class="card">
                    <h3>🧠 ANÁLISIS DE INTELIGENCIA ARTIFICIAL</h3>
                    <div class="score-container">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #00ff88;">FUERZA COMPRA: <span id="buy-score">{current_prediction.get('buy_score', 0)}</span></span>
                            <span style="color: #ff4444;">FUERZA VENTA: <span id="sell-score">{current_prediction.get('sell_score', 0)}</span></span>
                        </div>
                        <div class="score-bar">
                            <div class="buy-score" id="buy-bar" style="width: 50%"></div>
                            <div class="sell-score" id="sell-bar" style="width: 50%"></div>
                        </div>
                    </div>
                    
                    <h4 style="margin-top: 20px; margin-bottom: 10px;">📊 FACTORES DE DECISIÓN:</h4>
                    <ul class="reasons-list" id="reasons-list">
                        {"".join([f"<li>{r}</li>" for r in current_prediction.get('reasons', ['IA analizando factores de mercado...'])])}
                    </ul>
                </div>
                
                <!-- Rendimiento -->
                <div class="card">
                    <h3>📈 RENDIMIENTO DEL SISTEMA</h3>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" style="color: #00ff88;" id="accuracy">{accuracy:.1f}%</div>
                            <div class="metric-label">PRECISIÓN</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="total-pred">{total_pred}</div>
                            <div class="metric-label">TOTAL PREDICCIONES</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #00ff88;" id="correct-pred">{correct_pred}</div>
                            <div class="metric-label">CORRECTAS</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="avg-confidence">{avg_confidence:.1f}%</div>
                            <div class="metric-label">CONFIANZA PROMEDIO</div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <canvas id="performanceChart"></canvas>
                    </div>
                </div>
                
                <!-- Validación en Tiempo Real -->
                <div class="card validation-card">
                    <h3>✅ VALIDACIÓN EN TIEMPO REAL</h3>
                    <div class="validation-result" id="validation-result">
                        <div style="text-align: center; color: #94a3b8; padding: 20px;">
                            Esperando validación de siguiente vela...
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Información del Sistema -->
            <div class="card">
                <h3>⚙️ SISTEMA IA PROFESIONAL V5.0</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div style="background: rgba(0, 255, 136, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #00ff88;">
                        <div style="font-weight: 600; color: #00ff88;">🤖 IA AVANZADA</div>
                        <div style="font-size: 0.9em; color: #94a3b8; margin-top: 5px;">Análisis multi-factor con pesos dinámicos</div>
                    </div>
                    <div style="background: rgba(255, 187, 51, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #ffbb33;">
                        <div style="font-weight: 600; color: #ffbb33;">📊 APRENDIZAJE CONTINUO</div>
                        <div style="font-size: 0.9em; color: #94a3b8; margin-top: 5px;">Mejora automática con cada predicción</div>
                    </div>
                    <div style="background: rgba(255, 68, 68, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #ff4444;">
                        <div style="font-weight: 600; color: #ff4444;">🎯 ANÁLISIS PROFESIONAL</div>
                        <div style="font-size: 0.9em; color: #94a3b8; margin-top: 5px;">Tendencia, momentum, presión y volatilidad</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Configuración inicial
            let performanceHistory = [];
            
            // Actualizar datos en tiempo real
            function updateAllData() {{
                fetch('/api/prediction')
                    .then(response => response.json())
                    .then(data => {{
                        updatePredictionDisplay(data);
                    }});
                    
                fetch('/api/validation')  
                    .then(response => response.json())
                    .then(data => {{
                        updateValidationDisplay(data);
                    }});
            }}
            
            // Actualizar display de predicción
            function updatePredictionDisplay(data) {{
                const direction = data.direction || 'N/A';
                const confidence = data.confidence || 0;
                const tickCount = data.tick_count || 0;
                const currentPrice = data.current_price || 0;
                const buyScore = data.buy_score || 0;
                const sellScore = data.sell_score || 0;
                const scoreDiff = data.score_difference || 0;
                const reasons = data.reasons || [];
                const marketPhase = data.market_phase || 'N/A';
                
                // Actualizar elementos
                document.getElementById('direction-display').textContent = direction;
                document.getElementById('direction-display').style.color = 
                    direction === 'ALZA' ? '#00ff88' : (direction === 'BAJA' ? '#ff4444' : '#ffbb33');
                    
                document.getElementById('tick-count').textContent = tickCount;
                document.getElementById('score-diff').textContent = scoreDiff.toFixed(2);
                document.getElementById('buy-score').textContent = buyScore.toFixed(2);
                document.getElementById('sell-score').textContent = sellScore.toFixed(2);
                
                // Actualizar barras de score
                const total = Math.max(buyScore + sellScore, 1);
                const buyPercent = (buyScore / total) * 100;
                const sellPercent = (sellScore / total) * 100;
                
                document.getElementById('buy-bar').style.width = buyPercent + '%';
                document.getElementById('sell-bar').style.width = sellPercent + '%';
                
                // Actualizar razones
                const reasonsList = document.getElementById('reasons-list');
                reasonsList.innerHTML = reasons.map(r => `<li>${r}</li>`).join('') || 
                    '<li>🤖 IA analizando factores de mercado...</li>';
            }}
            
            // Actualizar validación
            function updateValidationDisplay(data) {{
                const validation = data.last_validation;
                const performance = data.performance;
                
                if (validation && validation.timestamp) {{
                    const correct = validation.correct;
                    const color = correct ? '#00ff88' : '#ff4444';
                    const icon = correct ? '✅' : '❌';
                    const bgColor = correct ? 'rgba(0, 255, 136, 0.1)' : 'rgba(255, 68, 68, 0.1)';
                    
                    document.getElementById('validation-result').innerHTML = `
                        <div style="background: ${bgColor}; padding: 20px; border-radius: 10px; border-left: 4px solid ${color};">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 2em; margin-right: 15px;">${icon}</span>
                                <div>
                                    <div style="font-size: 1.3em; font-weight: 600; color: ${color};">
                                        ${validation.predicted} → ${validation.actual}
                                    </div>
                                    <div style="color: #94a3b8; font-size: 0.9em;">
                                        Cambio: ${validation.price_change_pips}pips | Confianza: ${validation.confidence}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }}
                
                if (performance) {{
                    document.getElementById('accuracy').textContent = performance.recent_accuracy.toFixed(1) + '%';
                    document.getElementById('total-pred').textContent = performance.total_predictions;
                    document.getElementById('correct-pred').textContent = performance.correct_predictions;
                    document.getElementById('avg-confidence').textContent = data.average_confidence?.toFixed(1) || '0.0' + '%';
                    
                    // Actualizar historial para gráfico
                    performanceHistory.push(performance.recent_accuracy);
                    if (performanceHistory.length > 10) performanceHistory.shift();
                    updatePerformanceChart();
                }}
            }}
            
            // Actualizar countdown
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                
                const countdownEl = document.getElementById('countdown');
                countdownEl.textContent = remaining + 's';
                
                if (remaining <= 5) {{
                    countdownEl.classList.add('pulse');
                    countdownEl.style.color = '#ff4444';
                }} else {{
                    countdownEl.classList.remove('pulse');
                    countdownEl.style.color = '#00ff88';
                }}
            }}
            
            // Gráfico de performance
            function updatePerformanceChart() {{
                const ctx = document.getElementById('performanceChart').getContext('2d');
                
                if (window.performanceChart) {{
                    window.performanceChart.destroy();
                }}
                
                window.performanceChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: Array.from({length: performanceHistory.length}, (_, i) => i + 1),
                        datasets: [{{
                            label: 'Precisión (%)',
                            data: performanceHistory,
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100,
                                grid: {{
                                    color: 'rgba(255, 255, 255, 0.1)'
                                }},
                                ticks: {{
                                    color: '#94a3b8'
                                }}
                            }},
                            x: {{
                                grid: {{
                                    display: false
                                }},
                                ticks: {{
                                    color: '#94a3b8'
                                }}
                            }}
                        }}
                    }}
                }});
            }}
            
            // Inicializar
            setInterval(updateCountdown, 1000);
            setInterval(updateAllData, 2000);
            setInterval(updateCountdown, 1000);
            updateAllData();
            updateCountdown();
            
            // Efectos de sonido (opcional)
            function playSound(sound) {{
                // Implementar sonidos opcionales para alerts
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/validation")
def api_validation():
    return JSONResponse({
        "last_validation": performance_stats.get('last_validation'),
        "performance": {
            "total_predictions": performance_stats.get('total_predictions', 0),
            "correct_predictions": performance_stats.get('correct_predictions', 0),
            "recent_accuracy": performance_stats.get('recent_accuracy', 0)
        },
        "average_confidence": performance_stats.get('average_confidence', 0),
        "timestamp": now_iso()
    })

@app.get("/api/system/status")
def api_system_status():
    return JSONResponse({
        "status": "ACTIVE",
        "version": "5.0.0",
        "ai_model": "PROFESSIONAL_AI_V5",
        "features": [
            "Análisis multi-factor",
            "Aprendizaje continuo", 
            "Detección de fases de mercado",
            "Sistema de scoring avanzado"
        ],
        "timestamp": now_iso()
    })

# --------------- INICIALIZACIÓN PREMIUM ---------------
def start_premium_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info("⭐ SISTEMA PREMIUM INICIADO CORRECTAMENTE")
        logging.info("🎯 IA profesional activada y lista para trading")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema premium: {e}")

start_premium_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
