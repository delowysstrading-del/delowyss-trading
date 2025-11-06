# main.py - VERSIÓN CORREGIDA Y OPTIMIZADA
"""
Delowyss Trading AI — V4.6-PRO CORREGIDO
Sistema profesional con errores corregidos y mejoras de precisión
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from iqoptionapi.stable_api import IQ_Option
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = 60
PREDICTION_WINDOW = 5

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat()

# ------------------ ANALIZADOR MEJORADO ------------------
class ImprovedTickAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=200)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        
    def add_tick(self, price: float):
        price = float(price)
        
        if self.current_candle_open is None:
            self.current_candle_open = self.current_candle_high = self.current_candle_low = price
        
        self.current_candle_high = max(self.current_candle_high, price)
        self.current_candle_low = min(self.current_candle_low, price)
        self.current_candle_close = price
        
        tick_data = {
            'price': price,
            'timestamp': time.time()
        }
        self.ticks.append(tick_data)
        self.tick_count += 1
        
        return tick_data
    
    def get_analysis(self):
        if self.tick_count < 15:
            return {'status': 'INSUFFICIENT_DATA', 'tick_count': self.tick_count}
        
        try:
            prices = [t['price'] for t in self.ticks]
            
            # ANÁLISIS MEJORADO
            current_price = self.current_candle_close
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Momentum mejorado
            if len(prices) >= 10:
                short_momentum = (prices[-1] - prices[-5]) / prices[-5] * 10000
                medium_momentum = (prices[-1] - prices[-10]) / prices[-10] * 10000
                momentum = (short_momentum * 0.6 + medium_momentum * 0.4)
            else:
                momentum = (current_price - self.current_candle_open) / self.current_candle_open * 10000
            
            # Presión compra/venta mejorada
            if price_changes:
                up_ticks = sum(1 for change in price_changes if change > 0)
                down_ticks = sum(1 for change in price_changes if change < 0)
                total_ticks = len(price_changes)
                
                if total_ticks > 0:
                    buy_pressure = up_ticks / total_ticks
                    sell_pressure = down_ticks / total_ticks
                    imbalance = (buy_pressure - sell_pressure) * 2
                else:
                    imbalance = 0
            else:
                imbalance = 0
            
            # Detección de tendencia mejorada
            if len(prices) >= 15:
                x = np.arange(len(prices))
                slope, _ = np.polyfit(x, prices, 1)
                trend_strength = (slope * len(prices)) / np.mean(prices) * 10000
            else:
                trend_strength = momentum
            
            # Fase del mercado mejorada
            volatility = (self.current_candle_high - self.current_candle_low) * 10000
            if abs(momentum) > 2.0 and abs(imbalance) > 0.3:
                market_phase = "strong_trend"
            elif abs(momentum) > 1.0:
                market_phase = "moderate_trend"
            elif volatility < 0.5:
                market_phase = "consolidation"
            else:
                market_phase = "neutral"
            
            return {
                'tick_count': self.tick_count,
                'current_price': current_price,
                'open_price': self.current_candle_open,
                'high_price': self.current_candle_high,
                'low_price': self.current_candle_low,
                'price_momentum': momentum,
                'buy_sell_imbalance': imbalance,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'market_phase': market_phase,
                'data_quality': min(1.0, self.tick_count / 40.0),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        self.ticks.clear()
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0

# ------------------ PREDICTOR MEJORADO ------------------
class ImprovedPredictor:
    def __init__(self):
        self.analyzer = ImprovedTickAnalyzer()
        self.last_prediction = None
        self.performance = {'total': 0, 'correct': 0}
        
    def process_tick(self, price: float):
        return self.analyzer.add_tick(price)
    
    def predict_next_candle(self):
        analysis = self.analyzer.get_analysis()
        
        if analysis.get('status') != 'INSUFFICIENT_DATA':
            # PREDICCIÓN MEJORADA CON MÁS FILTROS
            prediction = self._improved_prediction(analysis)
            self.last_prediction = prediction
            return prediction
        else:
            return {
                'direction': 'N/A',
                'confidence': 0,
                'reason': analysis['message'],
                'tick_count': analysis['tick_count'],
                'timestamp': now_iso()
            }
    
    def _improved_prediction(self, analysis):
        momentum = analysis['price_momentum']
        imbalance = analysis['buy_sell_imbalance']
        trend_strength = analysis['trend_strength']
        market_phase = analysis['market_phase']
        data_quality = analysis['data_quality']
        
        # SISTEMA DE DECISIÓN MEJORADO
        buy_signals = 0
        sell_signals = 0
        reasons = []
        
        # 1. SEÑAL DE MOMENTO (MEJORADA)
        if abs(momentum) > 1.5:
            if momentum > 0:
                buy_signals += 2
                reasons.append(f"Momentum alcista fuerte ({momentum:.1f}pips)")
            else:
                sell_signals += 2
                reasons.append(f"Momentum bajista fuerte ({momentum:.1f}pips)")
        elif abs(momentum) > 0.8:
            if momentum > 0:
                buy_signals += 1
                reasons.append(f"Momentum alcista ({momentum:.1f}pips)")
            else:
                sell_signals += 1
                reasons.append(f"Momentum bajista ({momentum:.1f}pips)")
        
        # 2. SEÑAL DE DESEQUILIBRIO (MEJORADA)
        if abs(imbalance) > 0.4:
            if imbalance > 0:
                buy_signals += 2
                reasons.append(f"Fuerte presión compradora ({imbalance:.2f})")
            else:
                sell_signals += 2
                reasons.append(f"Fuerte presión vendedora ({imbalance:.2f})")
        elif abs(imbalance) > 0.2:
            if imbalance > 0:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # 3. SEÑAL DE TENDENCIA (MEJORADA)
        if abs(trend_strength) > 1.0:
            if trend_strength > 0:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # 4. FILTRO POR FASE DE MERCADO (MEJORADO)
        if market_phase == "strong_trend":
            # En tendencia fuerte, confiar más en las señales
            buy_signals = int(buy_signals * 1.2)
            sell_signals = int(sell_signals * 1.2)
        elif market_phase == "consolidation":
            # En consolidación, ser más conservador
            buy_signals = int(buy_signals * 0.8)
            sell_signals = int(sell_signals * 0.8)
            reasons.append("Mercado en consolidación")
        
        # DECISIÓN FINAL MEJORADA
        if buy_signals > sell_signals:
            direction = "ALZA"
            signal_strength = buy_signals - sell_signals
        elif sell_signals > buy_signals:
            direction = "BAJA"
            signal_strength = sell_signals - buy_signals
        else:
            # Empate - usar momentum como desempate
            direction = "ALZA" if momentum > 0 else "BAJA"
            signal_strength = 0
            reasons.append("Señales equilibradas - usando momentum")
        
        # CÁLCULO DE CONFIANZA MEJORADO
        base_confidence = 40 + (signal_strength * 8)
        
        # Ajustar por calidad de datos
        confidence = base_confidence * (0.6 + 0.4 * data_quality)
        
        # Ajustar por cantidad de ticks
        tick_factor = min(1.0, analysis['tick_count'] / 40.0)
        confidence = confidence * (0.7 + 0.3 * tick_factor)
        
        # Límites
        confidence = max(25, min(85, confidence))
        
        return {
            'direction': direction,
            'confidence': int(confidence),
            'tick_count': analysis['tick_count'],
            'current_price': analysis['current_price'],
            'reasons': reasons,
            'market_phase': market_phase,
            'timestamp': now_iso()
        }
    
    def get_analysis(self):
        return self.analyzer.get_analysis()
    
    def reset(self):
        self.analyzer.reset()
    
    def record_result(self, correct: bool):
        self.performance['total'] += 1
        if correct:
            self.performance['correct'] += 1
        
        accuracy = (self.performance['correct'] / self.performance['total'] * 100) if self.performance['total'] > 0 else 0
        if self.performance['total'] % 5 == 0:
            logging.info(f"📊 Precisión actual: {accuracy:.1f}%")

# -------------- IQ CONNECTION MEJORADO --------------
class ImprovedIQConnector:
    def __init__(self):
        self.iq = None
        self.connected = False
        self.tick_count = 0
        self.last_price = None
        self.actual_pair = "EURUSD"
        
    def connect(self):
        try:
            if not IQ_EMAIL or not IQ_PASSWORD:
                logging.warning("❌ Credenciales no configuradas")
                return None
                
            logging.info("🔗 Conectando a IQ Option...")
            self.iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
            check, reason = self.iq.connect()
            
            if check:
                self.iq.change_balance("PRACTICE")
                self.connected = True
                logging.info("✅ Conectado exitosamente")
                return self.iq
            else:
                logging.warning(f"⚠️ Conexión fallida: {reason}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Error de conexión: {e}")
            return None

    def get_realtime_price(self):
        try:
            if not self.connected:
                return None

            # Múltiples métodos para obtener precio
            try:
                candles = self.iq.get_candles(self.actual_pair, TIMEFRAME, 1, time.time())
                if candles and len(candles) > 0:
                    price = float(candles[-1]['close'])
                    if price > 0:
                        self._record_tick(price)
                        return price
            except Exception as e:
                logging.debug(f"get_candles falló: {e}")

            try:
                realtime = self.iq.get_realtime_candles(self.actual_pair, TIMEFRAME)
                if realtime:
                    candle_list = list(realtime.values())
                    if candle_list:
                        price = float(candle_list[-1].get('close', 0))
                        if price > 0:
                            self._record_tick(price)
                            return price
            except Exception as e:
                logging.debug(f"get_realtime_candles falló: {e}")

            return self.last_price

        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            return None

    def _record_tick(self, price):
        self.tick_count += 1
        self.last_price = price
        
        if self.tick_count <= 10 or self.tick_count % 25 == 0:
            logging.info(f"💰 Tick #{self.tick_count}: {price:.5f}")

# --------------- SISTEMA PRINCIPAL MEJORADO ---------------
iq_connector = ImprovedIQConnector()
predictor = ImprovedPredictor()

current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": [],
    "timestamp": now_iso()
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'last_validation': None
}

def main_loop():
    global current_prediction
    
    logging.info("🚀 Delowyss AI V4.6-CORREGIDO iniciado")
    
    iq_connector.connect()
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # OBTENER PRECIO
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                predictor.process_tick(price)
                
                # ACTUALIZAR ESTADO
                current_prediction.update({
                    "current_price": price,
                    "tick_count": predictor.analyzer.tick_count,
                    "timestamp": now_iso()
                })
            
            # PREDICCIÓN EN ÚLTIMOS SEGUNDOS
            if seconds_remaining <= PREDICTION_WINDOW and seconds_remaining > 1:
                if predictor.analyzer.tick_count >= 15:
                    if current_time - last_prediction_time >= 2:
                        prediction = predictor.predict_next_candle()
                        current_prediction.update(prediction)
                        last_prediction_time = current_time
            
            # CAMBIO DE VELA
            if current_candle_start > last_candle_start:
                validate_prediction()
                predictor.reset()
                last_candle_start = current_candle_start
                logging.info("🕯️ Nueva vela iniciada")
            
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(1)

def validate_prediction():
    global performance_stats
    
    if not predictor.last_prediction:
        return
        
    try:
        analysis = predictor.get_analysis()
        if analysis.get('status') == 'INSUFFICIENT_DATA':
            return
            
        close_price = analysis['current_price']
        prev_prediction = predictor.last_prediction
        
        if not prev_prediction or 'current_price' not in prev_prediction:
            return
            
        reference_price = prev_prediction['current_price']
        actual_direction = "ALZA" if close_price > reference_price else "BAJA"
        predicted_direction = prev_prediction.get('direction', 'N/A')
        
        correct = (actual_direction == predicted_direction)
        change_pips = (close_price - reference_price) * 10000
        
        # ACTUALIZAR ESTADÍSTICAS
        performance_stats['total_predictions'] += 1
        if correct:
            performance_stats['correct_predictions'] += 1
        
        performance_stats['last_validation'] = {
            "timestamp": now_iso(),
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": correct,
            "confidence": prev_prediction.get('confidence', 0),
            "price_change_pips": round(change_pips, 2)
        }
        
        # REGISTRAR RESULTADO
        predictor.record_result(correct)
        
        status = "✅ CORRECTA" if correct else "❌ ERRÓNEA"
        logging.info(f"📊 VALIDACIÓN: {status} | Pred: {predicted_direction} | Real: {actual_direction} | Conf: {prev_prediction.get('confidence', 0)}% | Change: {change_pips:.1f}pips")
        
    except Exception as e:
        logging.error(f"❌ Error en validación: {e}")

# --------------- FASTAPI CORREGIDO ---------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    analysis = predictor.get_analysis()
    
    direction = current_prediction.get("direction", "N/A")
    color = "#00ff88" if direction == "ALZA" else ("#ff4444" if direction == "BAJA" else "#ffbb33")
    
    total = performance_stats.get('total_predictions', 0)
    correct = performance_stats.get('correct_predictions', 0)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # HTML CORREGIDO - SIN ERRORES DE SINTAXIS
    html = f"""
    <!doctype html>
    <html>
    <head>
        <title>Delowyss AI V4.6-CORREGIDO</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #fff;
                padding: 20px;
            }}
            .card {{
                background: rgba(255,255,255,0.03);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
            }}
            .prediction-card {{
                border-left: 6px solid {color};
                padding: 20px;
                background: rgba(255,255,255,0.05);
            }}
            .countdown {{
                font-size: 2.5em;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🎯 Delowyss AI V4.6-CORREGIDO</h1>
            <p>Sistema mejorado con errores corregidos</p>
            
            <div class="countdown" id="countdown">--</div>
            
            <div class="prediction-card">
                <h2 style="color: {color}; margin: 0 0 10px 0;">
                    {direction} — {current_prediction.get('confidence', 0)}% confianza
                </h2>
                <p>Precio: {current_prediction.get('current_price', 0):.5f} • Ticks: {current_prediction.get('tick_count', 0)}</p>
                <p>Fase: {analysis.get('market_phase', 'N/A') if analysis else 'N/A'}</p>
            </div>

            <div class="card">
                <h3>📊 Rendimiento del Sistema</h3>
                <p>Precisión: {accuracy:.1f}% ({correct}/{total} predicciones)</p>
            </div>

            <div class="card">
                <h3>🎯 Razones de Predicción</h3>
                <ul>
                    {"".join([f"<li>📈 {r}</li>" for r in current_prediction.get('reasons', [])])}
                </ul>
            </div>
        </div>

        <script>
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                document.getElementById('countdown').textContent = remaining + 's';
            }}
            setInterval(updateCountdown, 1000);
            updateCountdown();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/prediction")
def api_prediction():
    return JSONResponse(current_prediction)

@app.get("/api/performance")
def api_performance():
    total = performance_stats.get('total_predictions', 0)
    correct = performance_stats.get('correct_predictions', 0)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    return JSONResponse({
        "accuracy": round(accuracy, 1),
        "total_predictions": total,
        "correct_predictions": correct,
        "last_validation": performance_stats.get('last_validation')
    })

# --------------- INICIALIZACIÓN ---------------
def start_system():
    thread = threading.Thread(target=main_loop, daemon=True)
    thread.start()
    logging.info("📊 Sistema principal iniciado")

start_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
