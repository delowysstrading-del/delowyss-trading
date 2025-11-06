# main.py - V5.5.1 PREMIUM (Fix Conexión + Simulación Mejorada)
"""
Delowyss Trading AI — V5.5.1 PREMIUM CONEXIÓN MEJORADA
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
IQ_EMAIL = os.getenv("IQ_EMAIL", "demo@example.com")  # Default para simulación
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "demo123")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = int(os.getenv("TIMEFRAME", "60"))
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "5"))
MIN_TICKS_FOR_PREDICTION = int(os.getenv("MIN_TICKS_FOR_PREDICTION", "10"))  # Reducido para testing
TICK_BUFFER_SIZE = int(os.getenv("TICK_BUFFER_SIZE", "200"))
PORT = int(os.getenv("PORT", "10000"))

# Model paths
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOGGING PROFESIONAL ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ SISTEMA DE SIMULACIÓN MEJORADO ------------------
class AdvancedTickSimulator:
    """Simulador de ticks realista con patrones de mercado"""
    def __init__(self, initial_price=1.10000):
        self.price = initial_price
        self.volatility = 0.0002
        self.trend_bias = 0
        self.last_update = time.time()
        self.tick_count = 0
        self.patterns = [
            'trending_up', 'trending_down', 'ranging', 'breakout'
        ]
        self.current_pattern = 'ranging'
        self.pattern_start = time.time()
        
    def generate_tick(self):
        """Genera tick realista con patrones de mercado"""
        current_time = time.time()
        time_diff = current_time - self.last_update
        
        # Cambiar patrón cada 30-60 segundos
        if current_time - self.pattern_start > np.random.randint(30, 60):
            self.current_pattern = np.random.choice(self.patterns)
            self.pattern_start = current_time
            logging.info(f"🔄 Patrón de mercado cambiado: {self.current_pattern}")
        
        # Aplicar patrón actual
        if self.current_pattern == 'trending_up':
            trend_strength = 0.0001
            volatility = self.volatility * 0.8
        elif self.current_pattern == 'trending_down':
            trend_strength = -0.0001
            volatility = self.volatility * 0.8
        elif self.current_pattern == 'breakout':
            trend_strength = np.random.choice([-0.0002, 0.0002])
            volatility = self.volatility * 1.5
        else:  # ranging
            trend_strength = 0
            volatility = self.volatility
        
        # Generar movimiento de precio
        random_walk = np.random.normal(trend_strength, volatility)
        self.price += random_walk
        
        # Suavizar y mantener en rango realista
        self.price = max(1.08000, min(1.12000, self.price))
        
        self.last_update = current_time
        self.tick_count += 1
        
        return float(self.price)

# ------------------ CONEXIÓN PROFESIONAL MEJORADA ------------------
class ProfessionalIQConnector:
    def __init__(self):
        self.connected = False
        self.tick_listeners = []
        self.last_price = 1.10000
        self.tick_count = 0
        self.simulation_mode = True  # Forzado para Render
        self.simulator = AdvancedTickSimulator()
        
    def connect(self):
        """Siempre usa simulación en Render para garantizar funcionamiento"""
        logging.info("🎯 MODO SIMULACIÓN AVANZADO ACTIVADO - Garantizado en Render")
        self.connected = True
        
        # Iniciar simulador de ticks MEJORADO
        thread = threading.Thread(target=self._simulate_ticks, daemon=True)
        thread.start()
        return True

    def _simulate_ticks(self):
        """Simulador de ticks MEJORADO con patrones realistas"""
        logging.info("🚀 Iniciando simulador de ticks avanzado...")
        
        while True:
            try:
                # Generar tick realista
                new_price = self.simulator.generate_tick()
                self.last_price = new_price
                self.tick_count += 1
                
                # Notificar a todos los listeners
                timestamp = time.time()
                for listener in self.tick_listeners:
                    try:
                        listener(new_price, timestamp)
                    except Exception as e:
                        logging.error(f"Error en listener: {e}")
                
                # Frecuencia de ticks realista (5-20 ticks por segundo)
                tick_interval = np.random.uniform(0.05, 0.2)
                time.sleep(tick_interval)
                
            except Exception as e:
                logging.error(f"Error en simulador: {e}")
                time.sleep(1)

    def add_tick_listener(self, listener):
        self.tick_listeners.append(listener)
        logging.info(f"✅ Listener añadido. Total: {len(self.tick_listeners)}")

    def get_realtime_price(self):
        return float(self.last_price)

# ------------------ IA AVANZADA COMPLETA (VERSIÓN SIMPLIFICADA PARA TEST) ------------------
class PremiumAIAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=50)
        self.last_candle_close = None
        self.candle_start_time = None
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            # Inicializar vela si es el primer tick
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Primer tick recibido")
            
            # Actualizar precios extremos
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)
            self.current_candle_close = price
            
            tick_data = {
                'price': price,
                'timestamp': current_time,
                'seconds_remaining': seconds_remaining,
                'candle_age': current_time - self.candle_start_time if self.candle_start_time else 0
            }
            
            # Almacenar tick
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            # Log cada 10 ticks
            if self.tick_count % 10 == 0:
                logging.info(f"📊 Tick {self.tick_count} - Precio: {price:.5f}")
            
            return tick_data
        except Exception as e:
            logging.error(f"Error en add_tick: {e}")
            return None
    
    def get_comprehensive_analysis(self):
        """Análisis simplificado para testing"""
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'INSUFFICIENT_DATA', 
                'tick_count': self.tick_count,
                'message': f'Recolectando ticks: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION}'
            }
        
        try:
            # Métricas básicas para testing
            prices = list(self.price_memory)
            current_price = self.current_candle_close
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes) if price_changes else 0.5
            volatility = (max(prices) - min(prices)) * 10000 if prices else 0
            
            # Análisis de tendencia simple
            if len(prices) >= 10:
                trend = (prices[-1] - prices[0]) * 10000
            else:
                trend = 0
            
            return {
                'status': 'SUCCESS',
                'tick_count': self.tick_count,
                'current_price': current_price,
                'open_price': self.current_candle_open,
                'high_price': self.current_candle_high,
                'low_price': self.current_candle_low,
                'candle_range': (self.current_candle_high - self.current_candle_low) * 10000,
                'trend_strength': trend,
                'volatility': volatility,
                'buy_pressure': buy_pressure,
                'market_phase': 'normal',
                'data_quality': min(1.0, self.tick_count / 20.0),
                'candle_progress': (time.time() - self.candle_start_time) / TIMEFRAME if self.candle_start_time else 0,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def reset(self):
        """Reinicia el análisis para nueva vela"""
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
            self.candle_start_time = None
            
            logging.info("🔄 Analyzer reiniciado para nueva vela")
                
        except Exception as e:
            logging.error(f"Error en reset: {e}")

# ------------------ SISTEMA PRINCIPAL SIMPLIFICADO ------------------
iq_connector = ProfessionalIQConnector()
analyzer = PremiumAIAnalyzer()

# VARIABLES GLOBALES MEJORADAS
current_prediction = {
    "direction": "ANALIZANDO",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🎯 Iniciando análisis de mercado..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING",
    "candle_progress": 0,
    "market_phase": "N/A",
    "buy_score": 0,
    "sell_score": 0
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0
}

# Estado interno
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False

def tick_processor(price, timestamp):
    """Procesador de ticks MEJORADO"""
    global current_prediction
    try:
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        # Procesar tick en el analyzer
        tick_data = analyzer.add_tick(price, seconds_remaining)
        
        if tick_data:
            # Obtener análisis actualizado
            analysis = analyzer.get_comprehensive_analysis()
            
            # Actualizar predicción global
            current_prediction.update({
                "current_price": float(price),
                "tick_count": analyzer.tick_count,
                "timestamp": now_iso(),
                "status": "ACTIVE",
                "candle_progress": (current_time - _last_candle_start) / TIMEFRAME
            })
            
            # Generar predicción básica cuando tengamos datos
            if analysis.get('status') == 'SUCCESS':
                generate_basic_prediction(analysis, seconds_remaining)
                
    except Exception as e:
        logging.error(f"Error procesando tick: {e}")

def generate_basic_prediction(analysis, seconds_remaining):
    """Genera predicción básica basada en análisis simple"""
    global current_prediction
    
    try:
        trend = analysis.get('trend_strength', 0)
        buy_pressure = analysis.get('buy_pressure', 0.5)
        volatility = analysis.get('volatility', 0)
        
        # Lógica de predicción simple
        if abs(trend) > 0.5:
            if trend > 0:
                direction = "ALZA"
                confidence = min(80, 50 + abs(trend) * 10)
                reasons = [f"📈 Tendencia alcista ({trend:.1f}pips)"]
            else:
                direction = "BAJA"
                confidence = min(80, 50 + abs(trend) * 10)
                reasons = [f"📉 Tendencia bajista ({trend:.1f}pips)"]
        elif buy_pressure > 0.6:
            direction = "ALZA"
            confidence = 60
            reasons = [f"💰 Presión compradora ({buy_pressure:.1%})"]
        elif buy_pressure < 0.4:
            direction = "BAJA"
            confidence = 60
            reasons = [f"💸 Presión vendedora ({buy_pressure:.1%})"]
        else:
            direction = "LATERAL"
            confidence = 40
            reasons = ["⚡ Mercado en equilibrio"]
        
        # Ajustar confianza por volatilidad
        if volatility > 1.0:
            confidence = max(30, confidence - 10)
            reasons.append(f"🌊 Alta volatilidad ({volatility:.1f}pips)")
        
        # Solo predecir en últimos 10 segundos
        if seconds_remaining > 10:
            direction = "ANALIZANDO"
            confidence = 0
            reasons = [f"⏳ Analizando... {seconds_remaining:.0f}s restantes"]
        
        current_prediction.update({
            "direction": direction,
            "confidence": confidence,
            "buy_score": round(buy_pressure * 100, 1),
            "sell_score": round((1 - buy_pressure) * 100, 1),
            "reasons": reasons,
            "market_phase": analysis.get('market_phase', 'N/A')
        })
        
        # Log de predicción
        if direction != "ANALIZANDO" and seconds_remaining <= 10:
            logging.info(f"🎯 PREDICCIÓN: {direction} | Conf: {confidence}% | Ticks: {analysis['tick_count']}")
            
    except Exception as e:
        logging.error(f"Error generando predicción: {e}")

def premium_main_loop():
    """Loop principal SIMPLIFICADO y ROBUSTO"""
    global current_prediction, _last_candle_start, _prediction_made_this_candle
    
    logging.info(f"🚀 DELOWYSS AI V5.5.1 INICIADA EN PUERTO {PORT}")
    logging.info("🎯 SISTEMA SIMPLIFICADO - Garantizado para Render")
    
    # Conectar y iniciar
    iq_connector.connect()
    iq_connector.add_tick_listener(tick_processor)
    
    logging.info("✅ Sistema listo - Esperando ticks...")

    while True:
        try:
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            # Detectar nueva vela
            if current_candle_start > _last_candle_start:
                logging.info("🕯️ DETECTADA NUEVA VELA - Reiniciando análisis")
                
                # Reiniciar analyzer
                analyzer.reset()
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                
                # Actualizar estado
                current_prediction.update({
                    "direction": "ANALIZANDO",
                    "confidence": 0,
                    "reasons": ["🔄 Nueva vela - Iniciando análisis..."],
                    "candle_progress": 0
                })

            # Actualizar progreso de vela
            current_prediction['candle_progress'] = (current_time - _last_candle_start) / TIMEFRAME

            time.sleep(0.5)  # Loop más eficiente
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal: {e}")
            time.sleep(1)

# ------------------ INTERFAZ WEB MEJORADA ------------------
app = FastAPI(
    title="Delowyss AI Premium V5.5.1",
    version="5.5.1",
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

@app.get("/api/health")
def api_health():
    return JSONResponse({
        "status": "healthy",
        "timestamp": now_iso(),
        "version": "5.5.1-simplified",
        "ticks_processed": analyzer.tick_count,
        "simulation_mode": True,
        "message": "Sistema operativo con simulación avanzada"
    })

@app.get("/api/system-info")
def api_system_info():
    return JSONResponse({
        "status": "running",
        "pair": PAR,
        "timeframe": TIMEFRAME,
        "current_ticks": analyzer.tick_count,
        "current_price": current_prediction.get("current_price", 0),
        "current_direction": current_prediction.get("direction", "N/A"),
        "timestamp": now_iso()
    })

def generate_html_interface():
    """Interfaz HTML SIMPLIFICADA pero FUNCIONAL"""
    direction = current_prediction.get("direction", "ANALIZANDO")
    confidence = current_prediction.get("confidence", 0)
    current_price = current_prediction.get("current_price", 0)
    tick_count = current_prediction.get("tick_count", 0)
    candle_progress = current_prediction.get("candle_progress", 0)
    
    # Colores dinámicos
    if direction == "ALZA":
        primary_color = "#00ff88"
        status_emoji = "📈"
    elif direction == "BAJA":
        primary_color = "#ff4444" 
        status_emoji = "📉"
    elif direction == "LATERAL":
        primary_color = "#ffbb33"
        status_emoji = "⚡"
    else:
        primary_color = "#667eea"
        status_emoji = "🔍"
    
    # Calcular tiempo hasta siguiente vela
    current_time = time.time()
    seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
    progress_percentage = min(100, max(0, (1 - seconds_remaining/TIMEFRAME) * 100))
    
    # Generar HTML de razones
    reasons_html = ""
    reasons_list = current_prediction.get('reasons', ['Iniciando sistema...'])
    for reason in reasons_list:
        reasons_html += f'<li class="reason-item">{reason}</li>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.5.1</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
                min-height: 100vh;
                padding: 20px;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 800px;
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
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 10px;
                color: {primary_color};
            }}
            
            .prediction-card {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                border-left: 5px solid {primary_color};
                text-align: center;
            }}
            
            .direction {{
                font-size: 3rem;
                font-weight: bold;
                color: {primary_color};
                margin: 20px 0;
            }}
            
            .confidence {{
                font-size: 1.2rem;
                margin-bottom: 20px;
            }}
            
            .metrics {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin: 20px 0;
            }}
            
            .metric {{
                background: rgba(255, 255, 255, 0.03);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }}
            
            .metric-value {{
                font-size: 1.5rem;
                font-weight: bold;
                color: {primary_color};
            }}
            
            .progress-bar {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                height: 8px;
                margin: 20px 0;
                overflow: hidden;
            }}
            
            .progress {{
                height: 100%;
                background: {primary_color};
                width: {progress_percentage}%;
                transition: width 0.5s ease;
            }}
            
            .reasons-list {{
                list-style: none;
                margin-top: 20px;
            }}
            
            .reason-item {{
                background: rgba(255, 255, 255, 0.03);
                margin: 10px 0;
                padding: 12px;
                border-radius: 8px;
                border-left: 3px solid {primary_color};
            }}
            
            .countdown {{
                background: rgba(0, 0, 0, 0.3);
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            }}
            
            .countdown-number {{
                font-size: 2rem;
                font-weight: bold;
                color: {primary_color};
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">🤖 DELOWYSS AI V5.5.1</div>
                <div>Sistema de Trading con Simulación Avanzada</div>
            </div>
            
            <div class="prediction-card">
                <h2>🎯 PREDICCIÓN ACTUAL</h2>
                <div class="direction">{direction} {status_emoji}</div>
                <div class="confidence">CONFIANZA: {confidence}%</div>
                
                <div class="progress-bar">
                    <div class="progress"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #94a3b8;">
                    <span>Progreso: {progress_percentage:.1f}%</span>
                    <span>Ticks: {tick_count}</span>
                </div>
                
                <div class="countdown">
                    <div>SIGUIENTE ACTUALIZACIÓN</div>
                    <div class="countdown-number">{int(seconds_remaining)}s</div>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{tick_count}</div>
                        <div>TICKS</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{current_price:.5f}</div>
                        <div>PRECIO</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{current_prediction.get('buy_score', 0)}%</div>
                        <div>COMPRA</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{current_prediction.get('sell_score', 0)}%</div>
                        <div>VENTA</div>
                    </div>
                </div>
                
                <h3 style="margin-top: 20px;">📊 ANÁLISIS:</h3>
                <ul class="reasons-list">
                    {reasons_html}
                </ul>
            </div>
            
            <div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 20px;">
                ✅ Sistema operativo con simulación avanzada | V5.5.1
            </div>
        </div>

        <script>
            function updateData() {{
                fetch('/api/prediction')
                    .then(response => response.json())
                    .then(data => {{
                        updateDisplay(data);
                    }});
            }}
            
            function updateDisplay(data) {{
                // Actualizar dirección y confianza
                document.querySelector('.direction').textContent = data.direction + ' ' + getEmoji(data.direction);
                document.querySelector('.confidence').textContent = 'CONFIANZA: ' + data.confidence + '%';
                
                // Actualizar métricas
                document.querySelector('.metric-value:nth-child(1)').textContent = data.tick_count;
                document.querySelector('.metric-value:nth-child(2)').textContent = data.current_price.toFixed(5);
                document.querySelector('.metric-value:nth-child(3)').textContent = data.buy_score + '%';
                document.querySelector('.metric-value:nth-child(4)').textContent = data.sell_score + '%';
                
                // Actualizar razones
                const reasonsList = document.querySelector('.reasons-list');
                reasonsList.innerHTML = data.reasons.map(reason => 
                    `<li class="reason-item">${{reason}}</li>`
                ).join('');
            }}
            
            function getEmoji(direction) {{
                if (direction === 'ALZA') return '📈';
                if (direction === 'BAJA') return '📉';
                if (direction === 'LATERAL') return '⚡';
                return '🔍';
            }}
            
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                document.querySelector('.countdown-number').textContent = remaining + 's';
            }}
            
            // Actualizar cada 2 segundos
            setInterval(updateData, 2000);
            setInterval(updateCountdown, 1000);
            updateData();
        </script>
    </body>
    </html>
    """
    return html_content

# --------------- INICIALIZACIÓN ---------------
def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info(f"✅ DELOWYSS AI V5.5.1 INICIADA EN PUERTO {PORT}")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

start_system()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
