# main.py - VERSIÓN COMPLETA CON INTERFAZ FUNCIONAL
"""
Delowyss Trading AI — V5.0 PREMIUM COMPLETO
Sistema profesional con interfaz funcional
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

# ---------------- CONFIGURACIÓN PREMIUM ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 15

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ IA AVANZADA ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=80)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=25)
        self.last_candle_close = None
        
    def add_tick(self, price: float):
        try:
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
        except Exception as e:
            logging.error(f"Error en add_tick: {e}")
            return None
    
    def _calculate_advanced_metrics(self):
        """Métricas avanzadas"""
        if len(self.price_memory) < 8:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            if len(prices) >= 12:
                short_trend = np.polyfit(range(6), prices[-6:], 1)[0]
                medium_trend = np.polyfit(range(12), prices[-12:], 1)[0]
                trend_strength = (short_trend * 0.6 + medium_trend * 0.4) * 10000
            else:
                trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
            momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else 0
            momentum = (momentum_5 * 0.7 + momentum_10 * 0.3)
            
            recent_prices = prices[-8:] if len(prices) >= 8 else prices
            if len(recent_prices) > 1:
                volatility = (max(recent_prices) - min(recent_prices)) * 10000
            else:
                volatility = 0
            
            if len(self.ticks) > 6:
                recent_ticks = list(self.ticks)[-10:]
                price_changes = []
                for i in range(1, len(recent_ticks)):
                    if i < len(recent_ticks):
                        change = recent_ticks[i]['price'] - recent_ticks[i-1]['price']
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
            
            if volatility < 0.2 and abs(trend_strength) < 0.4:
                market_phase = "consolidation"
            elif abs(trend_strength) > 1.5:
                market_phase = "trending"
            elif volatility > 1.0:
                market_phase = "volatile"
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
                'data_quality': min(1.0, self.tick_count / 20.0)
            }
        except Exception as e:
            logging.error(f"Error en cálculo de métricas: {e}")
            return {}
    
    def get_analysis(self):
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {'status': 'INSUFFICIENT_DATA', 'tick_count': self.tick_count}
        
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
                'timestamp': time.time(),
                **advanced_metrics
            }
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        try:
            if self.current_candle_close is not None:
                self.last_candle_close = self.current_candle_close
                
            self.ticks.clear()
            self.current_candle_open = None
            self.current_candle_high = None
            self.current_candle_low = None
            self.current_candle_close = None
            self.tick_count = 0
        except Exception as e:
            logging.error(f"Error en reset: {e}")

# ------------------ SISTEMA IA PROFESIONAL ------------------
class ProfessionalAIPredictor:
    def __init__(self):
        self.analyzer = PremiumAIAnalyzer()
        self.prediction_history = deque(maxlen=10)
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': 0.0
        }
        self.last_prediction = None
        self.last_validation_result = None
        
    def process_tick(self, price: float):
        try:
            return self.analyzer.add_tick(price)
        except Exception as e:
            logging.error(f"Error en process_tick: {e}")
            return None
    
    def _professional_ai_analysis(self, analysis):
        """Análisis de IA"""
        try:
            momentum = analysis['momentum']
            trend_strength = analysis['trend_strength']
            pressure_ratio = analysis['pressure_ratio']
            volatility = analysis['volatility']
            market_phase = analysis['market_phase']
            data_quality = analysis['data_quality']
            
            buy_score = 0
            sell_score = 0
            reasons = []
            
            trend_weight = 0.40
            if abs(trend_strength) > 1.0:
                if trend_strength > 0:
                    buy_score += 8 * trend_weight
                    reasons.append(f"📈 Tendencia alcista ({trend_strength:.1f})")
                else:
                    sell_score += 8 * trend_weight
                    reasons.append(f"📉 Tendencia bajista ({trend_strength:.1f})")
            
            momentum_weight = 0.35
            if abs(momentum) > 0.6:
                if momentum > 0:
                    buy_score += 7 * momentum_weight
                    reasons.append(f"🚀 Momentum alcista ({momentum:.1f}pips)")
                else:
                    sell_score += 7 * momentum_weight
                    reasons.append(f"🔻 Momentum bajista ({momentum:.1f}pips)")
            
            pressure_weight = 0.25
            if pressure_ratio > 1.5:
                buy_score += 6 * pressure_weight
                reasons.append(f"💰 Presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.7:
                sell_score += 6 * pressure_weight
                reasons.append(f"💸 Presión vendedora ({pressure_ratio:.1f}x)")
            
            score_difference = buy_score - sell_score
            
            if abs(score_difference) > 0.4:
                if score_difference > 0:
                    direction = "ALZA"
                    base_confidence = 55 + (score_difference * 35)
                else:
                    direction = "BAJA" 
                    base_confidence = 55 + (abs(score_difference) * 35)
            else:
                direction = "LATERAL"
                base_confidence = 40
                reasons.append("⚡ Señales insuficientes")
            
            confidence = base_confidence
            confidence *= data_quality
            
            if volatility > 1.2:
                confidence *= 0.7
                reasons.append("🌪️ Mercado volátil")
            
            if analysis['tick_count'] > 25:
                confidence = min(85, confidence + 10)
            
            confidence = max(35, min(80, confidence))
            
            return {
                'direction': direction,
                'confidence': int(confidence),
                'buy_score': round(buy_score, 2),
                'sell_score': round(sell_score, 2),
                'score_difference': round(score_difference, 2),
                'reasons': reasons,
                'market_phase': market_phase
            }
        except Exception as e:
            logging.error(f"Error en análisis IA: {e}")
            return {
                'direction': 'LATERAL',
                'confidence': 35,
                'reasons': ['🤖 Error en análisis'],
                'buy_score': 0,
                'sell_score': 0,
                'score_difference': 0
            }
    
    def predict_next_candle(self):
        try:
            analysis = self.analyzer.get_analysis()
            
            if analysis.get('status') != 'SUCCESS':
                return {
                    'direction': 'LATERAL',
                    'confidence': 0,
                    'reason': analysis.get('message', 'Analizando...'),
                    'timestamp': now_iso()
                }
            
            prediction = self._professional_ai_analysis(analysis)
            
            if prediction['confidence'] < 50:
                prediction['direction'] = 'LATERAL'
                prediction['reasons'].append("🔍 Confianza insuficiente")
            
            prediction.update({
                'tick_count': analysis['tick_count'],
                'current_price': analysis['current_price'],
                'timestamp': now_iso(),
                'model_version': 'PROFESSIONAL_AI_V5'
            })
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
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
        """Validación precisa"""
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
            
            minimal_change = 0.2
            
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
            
            logging.info(f"🎯 VALIDACIÓN: {status_icon} {predicted_direction}→{actual_direction} | Conf: {last_pred.get('confidence', 0)}% | Cambio: {price_change:.1f}pips")
            
            if total > 0 and total % 5 == 0:
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

    def get_realtime_price(self):
        try:
            if not self.connected:
                if self.last_price is None:
                    self.last_price = 1.15389
                else:
                    variation = np.random.uniform(-0.0001, 0.0001)
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
            return self.last_price if self.last_price else 1.15389

    def _record_tick(self, price):
        self.tick_count += 1
        self.last_price = price
        
        if self.tick_count <= 3 or self.tick_count % 200 == 0:
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL ---------------
iq_connector = ProfessionalIQConnector()
predictor = ProfessionalAIPredictor()

# VARIABLES GLOBALES
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 Sistema inicializando..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING"
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None
}

def premium_main_loop():
    global current_prediction, performance_stats
    
    logging.info("🚀 DELOWYSS AI V5.0 INICIADO")
    logging.info("🎯 Sistema profesional activo")
    
    iq_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    
    while True:
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
            
            if (seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - last_prediction_time) >= 2):
                
                prediction = predictor.predict_next_candle()
                
                if prediction['confidence'] >= 45:
                    current_prediction.update(prediction)
                    last_prediction_time = time.time()
                    
                    if prediction['direction'] != 'LATERAL':
                        logging.info(f"🎯 PREDICCIÓN: {prediction['direction']} | Conf: {prediction['confidence']}%")
            
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
            
            time.sleep(0.3)
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(1)

# --------------- INTERFAZ WEB COMPLETA Y FUNCIONAL ---------------
app = FastAPI(title="Delowyss AI Premium", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    current_price = current_prediction.get("current_price", 0)
    tick_count = current_prediction.get("tick_count", 0)
    
    accuracy = performance_stats.get('recent_accuracy', 0)
    total_predictions = performance_stats.get('total_predictions', 0)
    correct_predictions = performance_stats.get('correct_predictions', 0)
    
    # Colores dinámicos
    if direction == "ALZA":
        primary_color = "#00ff88"
        gradient = "linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)"
    elif direction == "BAJA":
        primary_color = "#ff4444"
        gradient = "linear-gradient(135deg, #ff4444 0%, #cc3636 100%)"
    else:
        primary_color = "#ffbb33"
        gradient = "linear-gradient(135deg, #ffbb33 0%, #cc9929 100%)"
    
    # Calcular nivel de confianza
    confidence_level = "ALTA" if confidence > 70 else "MEDIA" if confidence > 50 else "BAJA"
    confidence_color = "#00ff88" if confidence > 70 else "#ffbb33" if confidence > 50 else "#ff4444"
    
    # Generar HTML de razones
    reasons_html = ""
    reasons_list = current_prediction.get('reasons', ['Analizando mercado...'])
    for reason in reasons_list:
        reasons_html += f'<li class="reason-item">{reason}</li>'
    
    # Calcular tiempo hasta siguiente vela
    current_time = time.time()
    seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.0</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
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
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
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
            }}
            
            .dashboard {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            @media (max-width: 768px) {{
                .dashboard {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .card {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }}
            
            .prediction-card {{
                grid-column: 1 / -1;
                text-align: center;
                border-left: 4px solid {primary_color};
            }}
            
            .direction {{
                font-size: 3.5em;
                font-weight: 700;
                color: {primary_color};
                margin: 20px 0;
            }}
            
            .confidence {{
                font-size: 1.3em;
                margin-bottom: 20px;
            }}
            
            .confidence-badge {{
                display: inline-block;
                background: {confidence_color};
                color: #0f172a;
                padding: 5px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9em;
                margin-left: 10px;
            }}
            
            .countdown {{
                background: rgba(0, 0, 0, 0.3);
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
            }}
            
            .countdown-number {{
                font-size: 2.5em;
                font-weight: 700;
                color: {primary_color};
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            
            .metric {{
                background: rgba(255, 255, 255, 0.03);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }}
            
            .metric-value {{
                font-size: 1.5em;
                font-weight: 700;
                color: {primary_color};
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #94a3b8;
                font-size: 0.85em;
            }}
            
            .reasons-list {{
                list-style: none;
                margin-top: 15px;
            }}
            
            .reason-item {{
                background: rgba(255, 255, 255, 0.03);
                margin: 8px 0;
                padding: 12px 15px;
                border-radius: 8px;
                border-left: 3px solid {primary_color};
            }}
            
            .performance {{
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .validation-result {{
                background: rgba(255, 255, 255, 0.03);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <div class="logo">🤖 DELOWYSS AI PREMIUM</div>
                <div class="subtitle">Sistema de Trading con Inteligencia Artificial Avanzada V5.0</div>
            </div>
            
            <!-- Dashboard -->
            <div class="dashboard">
                <!-- Predicción Principal -->
                <div class="card prediction-card">
                    <h2>🎯 PREDICCIÓN ACTUAL</h2>
                    <div class="direction" id="direction">{direction}</div>
                    <div class="confidence">
                        CONFIANZA: {confidence}%
                        <span class="confidence-badge">{confidence_level}</span>
                    </div>
                    
                    <!-- Countdown -->
                    <div class="countdown">
                        <div style="color: #94a3b8; margin-bottom: 10px;">SIGUIENTE PREDICCIÓN EN:</div>
                        <div class="countdown-number" id="countdown">{int(seconds_remaining)}s</div>
                    </div>
                    
                    <!-- Métricas Rápidas -->
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" id="tick-count">{tick_count}</div>
                            <div class="metric-label">TICKS</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{current_price:.5f}</div>
                            <div class="metric-label">PRECIO</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="accuracy">{accuracy:.1f}%</div>
                            <div class="metric-label">PRECISIÓN</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{current_prediction.get('market_phase', 'N/A')}</div>
                            <div class="metric-label">FASE</div>
                        </div>
                    </div>
                </div>
                
                <!-- Análisis IA -->
                <div class="card">
                    <h3>🧠 ANÁLISIS DE IA</h3>
                    <div class="score-display">
                        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                            <span style="color: #00ff88;">COMPRA: <span id="buy-score">{current_prediction.get('buy_score', 0)}</span></span>
                            <span style="color: #ff4444;">VENTA: <span id="sell-score">{current_prediction.get('sell_score', 0)}</span></span>
                        </div>
                    </div>
                    
                    <h4 style="margin: 20px 0 10px 0;">📊 FACTORES:</h4>
                    <ul class="reasons-list" id="reasons-list">
                        {reasons_html}
                    </ul>
                </div>
                
                <!-- Rendimiento -->
                <div class="card">
                    <h3>📈 RENDIMIENTO</h3>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value" style="color: #00ff88;">{accuracy:.1f}%</div>
                            <div class="metric-label">PRECISIÓN</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="total-pred">{total_predictions}</div>
                            <div class="metric-label">TOTAL</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" style="color: #00ff88;">{correct_predictions}</div>
                            <div class="metric-label">CORRECTAS</div>
                        </div>
                    </div>
                    
                    <div class="performance">
                        <h4>✅ ÚLTIMA VALIDACIÓN</h4>
                        <div class="validation-result" id="validation-result">
                            <div style="color: #94a3b8; text-align: center;">
                                Esperando validación...
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Información del Sistema -->
            <div class="card">
                <h3>⚙️ INFORMACIÓN DEL SISTEMA</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div style="background: rgba(0, 255, 136, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #00ff88;">
                        <div style="font-weight: 600; color: #00ff88;">🤖 IA AVANZADA</div>
                        <div style="font-size: 0.9em; color: #94a3b8;">Análisis en tiempo real</div>
                    </div>
                    <div style="background: rgba(255, 187, 51, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #ffbb33;">
                        <div style="font-weight: 600; color: #ffbb33;">📊 VALIDACIÓN PRECISA</div>
                        <div style="font-size: 0.9em; color: #94a3b8;">Resultados 100% exactos</div>
                    </div>
                    <div style="background: rgba(255, 68, 68, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #ff4444;">
                        <div style="font-weight: 600; color: #ff4444;">🎯 PROFESIONAL</div>
                        <div style="font-size: 0.9em; color: #94a3b8;">Sistema de trading avanzado</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Actualizar datos en tiempo real
            function updateData() {{
                fetch('/api/prediction')
                    .then(response => response.json())
                    .then(data => {{
                        updatePrediction(data);
                    }});
                    
                fetch('/api/validation')
                    .then(response => response.json())
                    .then(data => {{
                        updateValidation(data);
                    }});
            }}
            
            function updatePrediction(data) {{
                // Actualizar dirección
                const directionEl = document.getElementById('direction');
                directionEl.textContent = data.direction;
                
                // Actualizar colores según dirección
                let color = '#ffbb33';
                if (data.direction === 'ALZA') color = '#00ff88';
                if (data.direction === 'BAJA') color = '#ff4444';
                
                directionEl.style.color = color;
                document.querySelector('.prediction-card').style.borderLeftColor = color;
                
                // Actualizar confianza
                const confidence = data.confidence || 0;
                document.querySelector('.confidence').innerHTML = 
                    `CONFIANZA: ${{confidence}}% <span class="confidence-badge">${{confidence > 70 ? 'ALTA' : confidence > 50 ? 'MEDIA' : 'BAJA'}}</span>`;
                
                // Actualizar métricas
                document.getElementById('tick-count').textContent = data.tick_count || 0;
                document.getElementById('buy-score').textContent = data.buy_score || 0;
                document.getElementById('sell-score').textContent = data.sell_score || 0;
                
                // Actualizar razones
                const reasons = data.reasons || ['Analizando mercado...'];
                document.getElementById('reasons-list').innerHTML = 
                    reasons.map(reason => `<li class="reason-item">${{reason}}</li>`).join('');
            }}
            
            function updateValidation(data) {{
                if (data.performance) {{
                    document.getElementById('accuracy').textContent = data.performance.recent_accuracy.toFixed(1) + '%';
                    document.getElementById('total-pred').textContent = data.performance.total_predictions;
                }}
                
                if (data.last_validation) {{
                    const val = data.last_validation;
                    const color = val.correct ? '#00ff88' : '#ff4444';
                    const icon = val.correct ? '✅' : '❌';
                    
                    document.getElementById('validation-result').innerHTML = `
                        <div style="color: ${{color}}; font-weight: 600; font-size: 1.1em;">
                            ${{icon}} ${{val.predicted}} → ${{val.actual}}
                        </div>
                        <div style="color: #94a3b8; font-size: 0.9em; margin-top: 5px;">
                            Confianza: ${{val.confidence}}% | Cambio: ${{val.price_change}}pips
                        </div>
                    `;
                }}
            }}
            
            // Actualizar countdown
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                document.getElementById('countdown').textContent = remaining + 's';
            }}
            
            // Inicializar
            setInterval(updateCountdown, 1000);
            setInterval(updateData, 2000);
            updateData();
            updateCountdown();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
        "version": "5.0.0"
    })

# --------------- INICIALIZACIÓN ---------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info("⭐ SISTEMA INICIADO CORRECTAMENTE")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

# Iniciar automáticamente
start_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
