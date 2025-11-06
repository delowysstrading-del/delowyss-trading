# main.py - VERSIÓN CORREGIDA PARA RENDER
"""
Delowyss Trading AI — V5.3 PREMIUM CORREGIDO PARA RENDER
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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from iqoptionapi.stable_api import IQ_Option
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURACIÓN CORREGIDA PARA RENDER ----------------
IQ_EMAIL = os.getenv("IQ_EMAIL")
IQ_PASSWORD = os.getenv("IQ_PASSWORD")
PAR = os.getenv("PAIR", "EURUSD")
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 20
TICK_BUFFER_SIZE = 500

# ---------------- CONFIGURACIÓN DE PUERTO PARA RENDER ----------------
PORT = int(os.getenv("PORT", 10000))  # Render proporciona el puerto via env var

# ---------------- LOGGING MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ... (TODO EL CÓDIGO DE LAS CLASES SE MANTIENE IGUAL)
# CompleteCandleAIAnalyzer, ComprehensiveAIPredictor, ProfessionalIQConnector
# ...

# --------------- INICIALIZACIÓN DEL SISTEMA ---------------
iq_connector = ProfessionalIQConnector()
predictor = ComprehensiveAIPredictor()

# VARIABLES GLOBALES
current_prediction = {
    "direction": "N/A",
    "confidence": 0,
    "tick_count": 0,
    "current_price": 0.0,
    "reasons": ["🤖 Sistema inicializando..."],
    "timestamp": now_iso(),
    "status": "INITIALIZING",
    "candle_progress": 0,
    "market_phase": "N/A"
}

performance_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'recent_accuracy': 0.0,
    'last_validation': None
}

def tick_processor(price, timestamp):
    """Procesa cada tick con información del tiempo restante"""
    try:
        global current_prediction
        
        current_time = time.time()
        seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
        
        tick_data = predictor.process_tick(price, seconds_remaining)
        
        if tick_data:
            current_prediction.update({
                "current_price": price,
                "tick_count": predictor.analyzer.tick_count,
                "timestamp": now_iso(),
                "status": "ACTIVE"
            })
                
    except Exception as e:
        logging.error(f"Error procesando tick: {e}")

def premium_main_loop():
    global current_prediction, performance_stats
    
    logging.info(f"🚀 DELOWYSS AI V5.3 INICIADO EN PUERTO {PORT}")
    logging.info("🎯 Analizando ticks desde inicio de vela hasta últimos 5 segundos")
    
    iq_connector.connect()
    iq_connector.add_tick_listener(tick_processor)
    
    last_prediction_time = 0
    last_candle_start = time.time() // TIMEFRAME * TIMEFRAME
    last_price = None
    prediction_made_this_candle = False
    
    while True:
        try:
            current_time = time.time()
            current_candle_start = current_time // TIMEFRAME * TIMEFRAME
            seconds_remaining = TIMEFRAME - (current_time % TIMEFRAME)
            
            price = iq_connector.get_realtime_price()
            
            if price and price > 0:
                last_price = price
            
            # Actualizar progreso de vela
            candle_progress = (current_time - current_candle_start) / TIMEFRAME
            current_prediction['candle_progress'] = candle_progress
            
            # Lógica de predicción en últimos 5 segundos
            if (seconds_remaining <= PREDICTION_WINDOW and 
                seconds_remaining > 2 and
                predictor.analyzer.tick_count >= MIN_TICKS_FOR_PREDICTION and
                (time.time() - last_prediction_time) >= 2 and
                not prediction_made_this_candle):
                
                logging.info(f"🎯 VENTANA DE PREDICCIÓN: {seconds_remaining:.1f}s restantes | "
                           f"Ticks analizados: {predictor.analyzer.tick_count}")
                
                prediction = predictor.predict_next_candle()
                
                if prediction['confidence'] >= 45:
                    current_prediction.update(prediction)
                    last_prediction_time = time.time()
                    prediction_made_this_candle = True
                    
                    if prediction['direction'] != 'LATERAL':
                        logging.info(f"🎯 PREDICCIÓN EMITIDA: {prediction['direction']} | "
                                   f"Confianza: {prediction['confidence']}% | "
                                   f"Base de ticks: {predictor.analyzer.tick_count}")
            
            # Detectar nueva vela
            if current_candle_start > last_candle_start:
                if last_price is not None:
                    validation_result = predictor.validate_prediction(last_price)
                    if validation_result:
                        performance_stats.update({
                            'total_predictions': validation_result['total_predictions'],
                            'correct_predictions': validation_result['correct_predictions'],
                            'recent_accuracy': validation_result['accuracy'],
                            'last_validation': validation_result
                        })
                
                # Reiniciar para nueva vela
                predictor.reset()
                last_candle_start = current_candle_start
                prediction_made_this_candle = False
                
                logging.info("🕯️ NUEVA VELA INICIADA - Comenzando análisis tick-by-tick completo")
            
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop: {e}")
            time.sleep(0.5)

# --------------- FASTAPI APP CON CONFIGURACIÓN PARA RENDER ---------------
app = FastAPI(
    title="Delowyss AI Premium V5.3",
    version="5.3.0",
    docs_url="/docs",  # Habilitar docs en Render
    redoc_url="/redoc"  # Habilitar redoc
)

# CORS para Render
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
    return JSONResponse({
        "status": "healthy", 
        "timestamp": now_iso(),
        "version": "5.3.0",
        "port": PORT,
        "features": ["full_candle_analysis", "phase_analysis", "tick_by_tick"]
    })

@app.get("/api/system-info")
def api_system_info():
    """Endpoint adicional para información del sistema"""
    return JSONResponse({
        "status": "running",
        "pair": PAR,
        "timeframe": TIMEFRAME,
        "prediction_window": PREDICTION_WINDOW,
        "current_ticks": predictor.analyzer.tick_count,
        "timestamp": now_iso()
    })

def generate_html_interface():
    """Genera la interfaz HTML responsive"""
    direction = current_prediction.get("direction", "N/A")
    confidence = current_prediction.get("confidence", 0)
    current_price = current_prediction.get("current_price", 0)
    tick_count = current_prediction.get("tick_count", 0)
    candle_progress = current_prediction.get("candle_progress", 0)
    market_phase = current_prediction.get("market_phase", "N/A")
    
    accuracy = performance_stats.get('recent_accuracy', 0)
    total_predictions = performance_stats.get('total_predictions', 0)
    correct_predictions = performance_stats.get('correct_predictions', 0)
    
    # Colores dinámicos
    if direction == "ALZA":
        primary_color = "#00ff88"
        gradient = "linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)"
        status_emoji = "📈"
    elif direction == "BAJA":
        primary_color = "#ff4444"
        gradient = "linear-gradient(135deg, #ff4444 0%, #cc3636 100%)"
        status_emoji = "📉"
    else:
        primary_color = "#ffbb33"
        gradient = "linear-gradient(135deg, #ffbb33 0%, #cc9929 100%)"
        status_emoji = "⚡"
    
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
    progress_percentage = min(100, max(0, (1 - seconds_remaining/TIMEFRAME) * 100))
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Delowyss AI Premium V5.3</title>
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
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 25px 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }}
            
            .logo {{
                font-size: clamp(2rem, 4vw, 2.8rem);
                font-weight: 700;
                margin-bottom: 10px;
                background: {gradient};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                line-height: 1.2;
            }}
            
            .subtitle {{
                color: #94a3b8;
                font-size: clamp(0.9rem, 2vw, 1.1rem);
                margin-bottom: 15px;
            }}
            
            .version {{
                background: rgba({primary_color.replace('#', '')}, 0.1);
                color: {primary_color};
                padding: 6px 12px;
                border-radius: 15px;
                font-size: 0.8rem;
                font-weight: 600;
                display: inline-block;
            }}
            
            .dashboard {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            @media (max-width: 768px) {{
                .dashboard {{
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
            }}
            
            .card {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }}
            
            .prediction-card {{
                grid-column: 1 / -1;
                text-align: center;
                border-left: 5px solid {primary_color};
            }}
            
            .direction {{
                font-size: clamp(2.5rem, 6vw, 4rem);
                font-weight: 700;
                color: {primary_color};
                margin: 20px 0;
            }}
            
            .confidence {{
                font-size: clamp(1.1rem, 2vw, 1.3rem);
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }}
            
            .confidence-badge {{
                background: {confidence_color};
                color: #0f172a;
                padding: 6px 15px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9em;
            }}
            
            .candle-progress {{
                margin: 20px 0;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                overflow: hidden;
                height: 8px;
            }}
            
            .progress-bar {{
                height: 100%;
                background: {gradient};
                width: {progress_percentage}%;
                transition: width 0.5s ease;
                border-radius: 10px;
            }}
            
            .progress-info {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                color: #94a3b8;
                margin-top: 8px;
            }}
            
            .countdown {{
                background: rgba(0, 0, 0, 0.3);
                padding: 20px;
                border-radius: 15px;
                margin: 25px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .countdown-number {{
                font-size: clamp(2rem, 5vw, 3rem);
                font-weight: 700;
                color: {primary_color};
                font-family: 'Courier New', monospace;
                margin: 10px 0;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
                margin: 20px 0;
            }}
            
            @media (max-width: 480px) {{
                .metrics-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
            
            .metric {{
                background: rgba(255, 255, 255, 0.03);
                padding: 15px 10px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }}
            
            .metric-value {{
                font-size: clamp(1.2rem, 3vw, 1.5rem);
                font-weight: 700;
                color: {primary_color};
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                color: #94a3b8;
                font-size: 0.75rem;
                font-weight: 500;
            }}
            
            .reasons-list {{
                list-style: none;
                margin-top: 15px;
            }}
            
            .reason-item {{
                background: rgba(255, 255, 255, 0.03);
                margin: 8px 0;
                padding: 12px 15px;
                border-radius: 10px;
                border-left: 3px solid {primary_color};
            }}
            
            .performance {{
                margin-top: 25px;
                padding-top: 20px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .validation-result {{
                background: rgba(255, 255, 255, 0.03);
                padding: 18px;
                border-radius: 12px;
                margin: 12px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            
            .info-item {{
                background: rgba({primary_color.replace('#', '')}, 0.1);
                padding: 20px 15px;
                border-radius: 12px;
                border-left: 3px solid {primary_color};
            }}
            
            @media (max-width: 480px) {{
                body {{
                    padding: 15px;
                }}
                
                .card {{
                    padding: 20px 15px;
                }}
                
                .header {{
                    padding: 20px 15px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">🤖 DELOWYSS AI PREMIUM</div>
                <div class="subtitle">Sistema de Trading con Inteligencia Artificial Avanzada</div>
                <div class="version">VERSION 5.3 - RENDER DEPLOY</div>
            </div>
            
            <div class="dashboard">
                <div class="card prediction-card">
                    <h2>🎯 PREDICCIÓN ACTUAL</h2>
                    <div class="direction" id="direction">{direction} {status_emoji}</div>
                    <div class="confidence">
                        CONFIANZA: {confidence}%
                        <span class="confidence-badge">{confidence_level}</span>
                    </div>
                    
                    <div class="candle-progress">
                        <div class="progress-bar"></div>
                    </div>
                    <div class="progress-info">
                        <span>Progreso: {progress_percentage:.1f}%</span>
                        <span>Fase: {market_phase}</span>
                    </div>
                    
                    <div class="countdown">
                        <div style="color: #94a3b8; margin-bottom: 10px;">
                            SIGUIENTE PREDICCIÓN EN:
                        </div>
                        <div class="countdown-number" id="countdown">{int(seconds_remaining)}s</div>
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 5px;">
                            Análisis de {tick_count} ticks
                        </div>
                    </div>
                    
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
                            <div class="metric-value">{int(candle_progress * 100)}%</div>
                            <div class="metric-label">PROGRESO</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🧠 ANÁLISIS DE IA</h3>
                    <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                        <span style="color: #00ff88;">COMPRA: <span id="buy-score">{current_prediction.get('buy_score', 0)}</span></span>
                        <span style="color: #ff4444;">VENTA: <span id="sell-score">{current_prediction.get('sell_score', 0)}</span></span>
                    </div>
                    
                    <h4 style="margin: 20px 0 10px 0;">📊 FACTORES:</h4>
                    <ul class="reasons-list" id="reasons-list">
                        {reasons_html}
                    </ul>
                </div>
                
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
            
            <div class="card">
                <h3>⚙️ INFORMACIÓN DEL SISTEMA</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div style="font-weight: 600; color: #00ff88;">🤖 IA AVANZADA</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">
                            Análisis completo de vela
                        </div>
                    </div>
                    <div class="info-item">
                        <div style="font-weight: 600; color: #ffbb33;">📊 VALIDACIÓN PRECISA</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">
                            Resultados en tiempo real
                        </div>
                    </div>
                    <div class="info-item">
                        <div style="font-weight: 600; color: #ff4444;">🎯 PROFESIONAL</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">
                            Sistema optimizado para Render
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
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
                const directionEl = document.getElementById('direction');
                if (directionEl) {{
                    let emoji = '⚡';
                    if (data.direction === 'ALZA') emoji = '📈';
                    if (data.direction === 'BAJA') emoji = '📉';
                    directionEl.textContent = data.direction + ' ' + emoji;
                }}
                
                const confidence = data.confidence || 0;
                const confidenceEl = document.querySelector('.confidence');
                if (confidenceEl) {{
                    confidenceEl.innerHTML = `CONFIANZA: ${{confidence}}% <span class="confidence-badge">${{confidence > 70 ? 'ALTA' : confidence > 50 ? 'MEDIA' : 'BAJA'}}</span>`;
                }}
                
                updateMetric('tick-count', data.tick_count || 0);
                updateMetric('buy-score', data.buy_score || 0);
                updateMetric('sell-score', data.sell_score || 0);
                
                const reasons = data.reasons || ['Analizando mercado...'];
                const reasonsList = document.getElementById('reasons-list');
                if (reasonsList) {{
                    reasonsList.innerHTML = reasons.map(reason => 
                        `<li class="reason-item">${{reason}}</li>`
                    ).join('');
                }}
            }}
            
            function updateMetric(id, value) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.textContent = value;
                }}
            }}
            
            function updateValidation(data) {{
                if (data.performance) {{
                    updateMetric('accuracy', data.performance.recent_accuracy.toFixed(1));
                    updateMetric('total-pred', data.performance.total_predictions);
                }}
                
                if (data.last_validation) {{
                    const val = data.last_validation;
                    const color = val.correct ? '#00ff88' : '#ff4444';
                    const icon = val.correct ? '✅' : '❌';
                    
                    const validationEl = document.getElementById('validation-result');
                    if (validationEl) {{
                        validationEl.innerHTML = `
                            <div style="color: ${{color}}; font-weight: 600; font-size: 1.1rem;">
                                ${{icon}} ${{val.predicted}} → ${{val.actual}}
                            </div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">
                                Confianza: ${{val.confidence}}% | Cambio: ${{val.price_change}}pips
                            </div>
                        `;
                    }}
                }}
            }}
            
            function updateCountdown() {{
                const now = new Date();
                const seconds = now.getSeconds();
                const remaining = 60 - seconds;
                const countdownEl = document.getElementById('countdown');
                if (countdownEl) {{
                    countdownEl.textContent = remaining + 's';
                }}
            }}
            
            setInterval(updateCountdown, 1000);
            setInterval(updateData, 2000);
            updateData();
        </script>
    </body>
    </html>
    """
    return html_content

def start_system():
    try:
        thread = threading.Thread(target=premium_main_loop, daemon=True)
        thread.start()
        logging.info(f"⭐ SISTEMA INICIADO EN PUERTO {PORT}")
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema: {e}")

# Iniciar el sistema
start_system()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,  # Usar el puerto de Render
        log_level="info",
        access_log=True
    )
