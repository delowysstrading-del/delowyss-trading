# main.py - VERSIÓN FINAL ESTABILIZADA
"""
Delowyss Trading AI — V5.0 PREMIUM ESTABLE
Sistema profesional con IA mejorada y máxima estabilidad
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

# ------------------ IA AVANZADA ESTABILIZADA ------------------
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
        """Métricas avanzadas ESTABILIZADAS"""
        if len(self.price_memory) < 8:
            return {}
            
        try:
            prices = np.array(list(self.price_memory))
            
            # ANÁLISIS SIMPLIFICADO PERO EFECTIVO
            if len(prices) >= 12:
                short_trend = np.polyfit(range(6), prices[-6:], 1)[0]
                medium_trend = np.polyfit(range(12), prices[-12:], 1)[0]
                trend_strength = (short_trend * 0.6 + medium_trend * 0.4) * 10000
            else:
                trend_strength = (prices[-1] - prices[0]) * 10000 if len(prices) > 1 else 0
            
            # MOMENTUM MEJORADO
            momentum_5 = (prices[-1] - prices[-5]) * 10000 if len(prices) >= 5 else 0
            momentum_10 = (prices[-1] - prices[-10]) * 10000 if len(prices) >= 10 else 0
            momentum = (momentum_5 * 0.7 + momentum_10 * 0.3)
            
            # VOLATILIDAD REALISTA
            recent_prices = prices[-8:] if len(prices) >= 8 else prices
            if len(recent_prices) > 1:
                volatility = (max(recent_prices) - min(recent_prices)) * 10000
            else:
                volatility = 0
            
            # PRESIÓN DE MERCADO MEJORADA
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
            
            # FASE DE MERCADO SIMPLIFICADA
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

# ------------------ SISTEMA IA PROFESIONAL ESTABILIZADO ------------------
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
        """Análisis de IA ESTABILIZADO y MÁS PRECISO"""
        try:
            momentum = analysis['momentum']
            trend_strength = analysis['trend_strength']
            pressure_ratio = analysis['pressure_ratio']
            volatility = analysis['volatility']
            market_phase = analysis['market_phase']
            data_quality = analysis['data_quality']
            
            # SISTEMA DE PUNTUACIÓN MÁS CONSERVADOR
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # 1. TENDENCIA (40% peso) - MÁS IMPORTANTE
            trend_weight = 0.40
            if abs(trend_strength) > 1.0:
                if trend_strength > 0:
                    buy_score += 8 * trend_weight
                    reasons.append(f"📈 Tendencia alcista ({trend_strength:.1f})")
                else:
                    sell_score += 8 * trend_weight
                    reasons.append(f"📉 Tendencia bajista ({trend_strength:.1f})")
            
            # 2. MOMENTUM (35% peso)
            momentum_weight = 0.35
            if abs(momentum) > 0.6:
                if momentum > 0:
                    buy_score += 7 * momentum_weight
                    reasons.append(f"🚀 Momentum alcista ({momentum:.1f}pips)")
                else:
                    sell_score += 7 * momentum_weight
                    reasons.append(f"🔻 Momentum bajista ({momentum:.1f}pips)")
            
            # 3. PRESIÓN (25% peso)
            pressure_weight = 0.25
            if pressure_ratio > 1.5:
                buy_score += 6 * pressure_weight
                reasons.append(f"💰 Presión compradora ({pressure_ratio:.1f}x)")
            elif pressure_ratio < 0.7:
                sell_score += 6 * pressure_weight
                reasons.append(f"💸 Presión vendedora ({pressure_ratio:.1f}x)")
            
            # DECISIÓN FINAL MÁS CONSERVADORA
            score_difference = buy_score - sell_score
            
            # SOLO PREDECIR CON SEÑALES FUERTES
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
            
            # AJUSTES DE CONFIANZA
            confidence = base_confidence
            confidence *= data_quality
            
            # SER MÁS CONSERVADOR EN ALTA VOLATILIDAD
            if volatility > 1.2:
                confidence *= 0.7
                reasons.append("🌪️ Mercado volátil")
            
            # MÁS DATOS = MÁS CONFIANZA
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
                'reasons': ['🤖 Error en análisis - modo seguro'],
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
            
            # SOLO PREDICCIONES CON BUENA CONFIANZA
            if prediction['confidence'] < 50:
                prediction['direction'] = 'LATERAL'
                prediction['reasons'].append("🔍 Confianza insuficiente")
            
            prediction.update({
                'tick_count': analysis['tick_count'],
                'current_price': analysis['current_price'],
                'timestamp': now_iso(),
                'model_version': 'PROFESSIONAL_AI_V5_ESTABLE'
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
        """Validación ESTABILIZADA"""
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
            
            # UMBRAL MÁS REALISTA
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
            
            # ACTUALIZAR ESTADÍSTICAS
            if predicted_direction != "LATERAL":
                self.performance_stats['total_predictions'] += 1
                if is_correct:
                    self.performance_stats['correct_predictions'] += 1
            
            total = self.performance_stats['total_predictions']
            correct = self.performance_stats['correct_predictions']
            accuracy = (correct / total * 100) if total > 0 else 0
            self.performance_stats['recent_accuracy'] = accuracy
            
            # LOGGING CLARO
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

# -------------- CONEXIÓN PROFESIONAL ESTABILIZADA --------------
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
                # Precio demo realista
                if self.last_price is None:
                    self.last_price = 1.15389  # Precio inicial realista
                else:
                    # Variación más realista
                    variation = np.random.uniform(-0.0001, 0.0001)
                    self.last_price += variation
                return self.last_price

            # Intentar obtener precio real
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
        
        # Logging menos frecuente
        if self.tick_count <= 3 or self.tick_count % 200 == 0:
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL ESTABILIZADO ---------------
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
    
    logging.info("🚀 DELOWYSS AI V5.0 ESTABLE INICIADO")
    logging.info("🎯 Sistema profesional con máxima estabilidad")
    
    # Conexión simple
    iq_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # OBTENER PRECIO
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                predictor.process_tick(price)
                last_price = price
                
                # ACTUALIZAR ESTADO
                current_prediction.update({
                    "current_price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso(),
                    "status": "ACTIVE"
                })
            
            # PREDICCIÓN CON TIMING MEJORADO
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
            
            # CAMBIO DE VELA
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
            
            time.sleep(0.3)  # Loop más rápido pero estable
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(1)  # Recuperación más rápida

# --------------- INTERFAZ WEB SIMPLIFICADA ---------------
app = FastAPI(title="Delowyss AI Premium", version="5.0.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def premium_interface():
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    accuracy = performance_stats.get('recent_accuracy', 0)
    
    # Colores dinámicos
    if direction == "ALZA":
        color = "#00ff88"
    elif direction == "BAJA":
        color = "#ff4444"
    else:
        color = "#ffbb33"
    
    # Generar HTML seguro
    reasons_html = ""
    reasons_list = current_prediction.get('reasons', ['Analizando mercado...'])
    for reason in reasons_list:
        reasons_html += f"<li>{reason}</li>"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Delowyss AI Premium V5.0</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: white;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
            }}
            .card {{
                background: rgba(255,255,255,0.05);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                border-left: 4px solid {color};
            }}
            .prediction {{
                font-size: 2em;
                font-weight: bold;
                color: {color};
                text-align: center;
            }}
            .confidence {{
                text-align: center;
                font-size: 1.2em;
                margin: 10px 0;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin: 15px 0;
            }}
            .metric {{
                background: rgba(255,255,255,0.1);
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }}
            .reasons {{
                list-style: none;
                padding: 0;
            }}
            .reasons li {{
                background: rgba(255,255,255,0.05);
                margin: 5px 0;
                padding: 8px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 DELOWYSS AI PREMIUM V5.0</h1>
            
            <div class="card">
                <div class="prediction" id="direction">{direction}</div>
                <div class="confidence" id="confidence">Confianza: {confidence}%</div>
                
                <div class="metrics">
                    <div class="metric">
                        <div id="tick-count">{current_prediction.get('tick_count', 0)}</div>
                        <small>Ticks</small>
                    </div>
                    <div class="metric">
                        <div>{current_prediction.get('current_price', 0):.5f}</div>
                        <small>Precio</small>
                    </div>
                    <div class="metric">
                        <div id="accuracy">{accuracy:.1f}%</div>
                        <small>Precisión</small>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Análisis IA</h3>
                <ul class="reasons" id="reasons">
                    {reasons_html}
                </ul>
            </div>
        </div>

        <script>
            function updateData() {{
                fetch('/api/prediction')
                    .then(r => r.json())
                    .then(data => {{
                        document.getElementById('direction').textContent = data.direction;
                        document.getElementById('confidence').textContent = 'Confianza: ' + data.confidence + '%';
                        document.getElementById('tick-count').textContent = data.tick_count;
                        
                        // Actualizar razones
                        const reasons = data.reasons || ['Analizando...'];
                        document.getElementById('reasons').innerHTML = reasons.map(r => `<li>${{r}}</li>`).join('');
                    }});
                    
                fetch('/api/validation')
                    .then(r => r.json())
                    .then(data => {{
                        if(data.performance) {{
                            document.getElementById('accuracy').textContent = data.performance.recent_accuracy.toFixed(1) + '%';
                        }}
                    }});
            }}
            
            setInterval(updateData, 2000);
            updateData();
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
        "version": "5.0.4"
    })

# --------------- INICIALIZACIÓN SEGURA ---------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info("⭐ SISTEMA ESTABLE INICIADO")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

# Iniciar automáticamente en Render
start_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
