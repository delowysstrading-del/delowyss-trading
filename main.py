# main.py - V6.3 ANÁLISIS COMPLETO DE VELA + PREDICCIÓN + AUTOAPRENDIZAJE MEJORADO - VERSIÓN DEFINITIVA
"""
Delowyss Trading AI — V6.3 ANÁLISIS COMPLETO DE VELA CON PREDICCIÓN + AUTOAPRENDIZAJE MEJORADO
CEO: Eduardo Solis — © 2025
Sistema de análisis completo con IA avanzada y autoaprendizaje MEJORADO - VERSIÓN DEFINITIVA
"""

import os
import time
import threading
import logging
import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, List, Optional

# Importaciones de FastAPI primero
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Luego las demás importaciones
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("⚠️ scikit-learn no disponible, usando modo básico")

# ------------------ CONFIGURACIÓN MEJORADA ------------------
IQ_EMAIL = os.getenv("IQ_EMAIL", "vozhechacancion1@gmail.com")
IQ_PASSWORD = os.getenv("IQ_PASSWORD", "tu_password_real")
PAR = "EURUSD"
TIMEFRAME = 60
PREDICTION_WINDOW = 5
MIN_TICKS_FOR_PREDICTION = 15
TICK_BUFFER_SIZE = 200
PORT = int(os.getenv("PORT", "10000"))

# ---------------- LOGGING MEJORADO ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

# ------------------ HTML RESPONSIVE DASHBOARD SIMPLIFICADO ------------------
HTML_RESPONSIVE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delowyss Trading AI V6.3</title>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: #0a0a0a;
            color: #ffffff; 
            min-height: 100vh;
            padding: 0;
            overflow-x: hidden;
        }
        
        .container { 
            max-width: 100%;
            margin: 0 auto;
            padding: 10px;
        }
        
        .header { 
            text-align: center; 
            margin-bottom: 15px; 
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .header h1 { 
            color: #00ff88; 
            margin-bottom: 5px;
            font-size: 1.4em;
            font-weight: 600;
        }
        
        .header .subtitle {
            color: #888;
            font-size: 0.9em;
        }
        
        .grid { 
            display: grid; 
            grid-template-columns: 1fr;
            gap: 12px; 
            margin-bottom: 15px;
        }
        
        @media (min-width: 768px) {
            .grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,255,136,0.1);
        }
        
        .card h2 {
            color: #00ff88;
            margin-bottom: 12px;
            font-size: 1.1em;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 8px;
            font-weight: 600;
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
            color: #000;
            grid-column: 1 / -1;
        }
        
        .prediction-card h2 {
            color: #000;
            border-bottom-color: rgba(0,0,0,0.2);
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
            font-size: 0.9em;
        }
        
        .metric .label {
            flex: 1;
            color: #ccc;
        }
        
        .metric .value {
            font-weight: 600;
            color: #00ff88;
        }
        
        .prediction-card .value {
            color: #000;
        }
        
        .confidence-bar {
            height: 16px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin: 8px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: #000;
            border-radius: 8px;
            transition: width 0.3s ease;
        }
        
        .countdown {
            font-size: 1.3em;
            font-weight: 700;
            text-align: center;
            color: #00ff88;
            margin: 8px 0;
        }
        
        .prediction-arrow {
            font-size: 2em;
            text-align: center;
            margin: 5px 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        
        .status-connected { background: #00ff88; }
        .status-disconnected { background: #ff4444; }
        .status-synced { background: #00ff88; }
        .status-unsynced { background: #ffaa00; }
        
        .pulse {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .compact-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }
        
        .compact-metric {
            font-size: 0.8em;
            padding: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            text-align: center;
        }
        
        .compact-value {
            font-weight: 700;
            color: #00ff88;
            display: block;
            font-size: 1.1em;
        }
        
        .debug-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 6px;
            padding: 10px;
            margin-top: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.75em;
            max-height: 80px;
            overflow-y: auto;
        }
        
        .debug-item {
            margin: 1px 0;
            color: #ccc;
            line-height: 1.2;
        }
        
        .debug-warning { color: #ffaa00; }
        .debug-error { color: #ff4444; }
        .debug-success { color: #00ff88; }
        
        .signal-badge {
            background: rgba(0,255,136,0.2);
            color: #00ff88;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Delowyss AI V6.3</h1>
            <div class="subtitle">Análisis Completo + Predicción + AutoML</div>
        </div>
        
        <div class="grid">
            <div class="card prediction-card" id="predictionCard">
                <h2>🎯 PREDICCIÓN ACTUAL <span class="signal-badge" id="predictionStatus">ANALIZANDO</span></h2>
                <div class="countdown" id="countdown">--</div>
                <div class="prediction-arrow" id="predictionArrow">⏳</div>
                <div style="text-align: center; font-size: 1.1em; font-weight: 700;" id="predictionDirection">ANALIZANDO...</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceBar" style="width: 0%"></div>
                </div>
                <div style="text-align: center; font-size: 0.9em;" id="confidenceText">Confianza: 0%</div>
                <div style="text-align: center; font-size: 0.8em; margin-top: 5px;" id="predictionMethod">Método: Tradicional</div>
                <div class="debug-panel" id="predictionDebug">
                    <div class="debug-item">Esperando datos...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 VELA ACTUAL</h2>
                <div class="compact-grid">
                    <div class="compact-metric">
                        <span class="label">Progreso</span>
                        <span class="compact-value" id="candleProgress">0%</span>
                    </div>
                    <div class="compact-metric">
                        <span class="label">Tiempo</span>
                        <span class="compact-value" id="timeRemaining">60s</span>
                    </div>
                    <div class="compact-metric">
                        <span class="label">Precio</span>
                        <span class="compact-value" id="currentPrice">0.00000</span>
                    </div>
                    <div class="compact-metric">
                        <span class="label">Ticks</span>
                        <span class="compact-value" id="ticksProcessed">0</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🧠 AUTOAPRENDIZAJE</h2>
                <div class="metric">
                    <span class="label">Accuracy Modelo:</span>
                    <span class="value" id="modelAccuracy">0%</span>
                </div>
                <div class="metric">
                    <span class="label">Muestras:</span>
                    <span class="value" id="trainingSamples">0</span>
                </div>
                <div class="metric">
                    <span class="label">Estado:</span>
                    <span class="value" id="learningState">ACTIVO</span>
                </div>
                <div class="debug-panel" id="learningDebug">
                    <div class="debug-item">Inicializando aprendizaje...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 MÉTRICAS</h2>
                <div class="compact-grid">
                    <div class="compact-metric">
                        <span class="label">Densidad</span>
                        <span class="compact-value" id="density">0</span>
                    </div>
                    <div class="compact-metric">
                        <span class="label">Velocidad</span>
                        <span class="compact-value" id="velocity">0</span>
                    </div>
                    <div class="compact-metric">
                        <span class="label">Aceleración</span>
                        <span class="compact-value" id="acceleration">0</span>
                    </div>
                    <div class="compact-metric">
                        <span class="label">Fase</span>
                        <span class="compact-value" id="phase">INICIAL</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🏆 PERFORMANCE</h2>
                <div class="metric">
                    <span class="label">Precisión Hoy:</span>
                    <span class="value" id="todayAccuracy">0%</span>
                </div>
                <div class="metric">
                    <span class="label">Señales:</span>
                    <span class="value" id="totalSignals">0</span>
                </div>
                <div class="metric">
                    <span class="label">Racha Actual:</span>
                    <span class="value" id="currentStreak">0</span>
                </div>
                <div class="debug-panel" id="performanceDebug">
                    <div class="debug-item">Esperando datos...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>🔧 SISTEMA</h2>
                <div class="metric">
                    <span class="label">IQ Option:</span>
                    <span class="value">
                        <span class="status-indicator" id="iqStatus"></span>
                        <span id="iqStatusText">CONECTANDO...</span>
                    </span>
                </div>
                <div class="metric">
                    <span class="label">Estado IA:</span>
                    <span class="value" id="aiStatus">INICIANDO</span>
                </div>
                <div class="metric">
                    <span class="label">Última Actualización:</span>
                    <span class="value" id="lastUpdate">--:--:--</span>
                </div>
                <div class="debug-panel" id="systemDebug">
                    <div class="debug-item">Inicializando sistema...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectTimeout = null;
        
        function getWebSocketUrl() {
            if (window.location.protocol === 'https:') {
                return `wss://${window.location.host}/ws`;
            } else {
                return `ws://${window.location.host}/ws`;
            }
        }
        
        function connectWebSocket() {
            const wsUrl = getWebSocketUrl();
            
            try {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('✅ WebSocket conectado');
                    updateStatus('CONECTADO', '#00ff88');
                    clearTimeout(reconnectTimeout);
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'dashboard_update') {
                        updateDashboard(data.data);
                    }
                };
                
                ws.onclose = function(event) {
                    console.log('🔌 WebSocket cerrado');
                    updateStatus('RECONECTANDO...', '#ffaa00');
                    scheduleReconnect();
                };
                
                ws.onerror = function(error) {
                    console.error('❌ Error WebSocket:', error);
                    updateStatus('ERROR', '#ff4444');
                };
                
            } catch (error) {
                console.error('❌ Error creando WebSocket:', error)
                updateStatus('ERROR', '#ff4444');
                scheduleReconnect();
            }
        }
        
        function updateStatus(status, color) {
            const element = document.getElementById('aiStatus');
            if (element) {
                element.textContent = status;
                element.style.color = color;
            }
        }
        
        function scheduleReconnect() {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = setTimeout(() => {
                console.log('🔄 Reconectando...');
                connectWebSocket();
            }, 2000);
        }
        
        function updateDashboard(data) {
            // Predicción Actual
            const pred = data.current_prediction;
            document.getElementById('predictionDirection').textContent = pred.direction;
            document.getElementById('predictionArrow').textContent = pred.arrow;
            document.getElementById('confidenceText').textContent = `Confianza: ${pred.confidence}%`;
            document.getElementById('confidenceBar').style.width = `${pred.confidence}%`;
            document.getElementById('countdown').textContent = `${Math.round(data.current_candle.time_remaining)}s`;
            document.getElementById('predictionMethod').textContent = `Método: ${pred.method || 'Tradicional'}`;
            document.getElementById('predictionStatus').textContent = pred.status || 'ANALIZANDO';
            
            updateDebugPanel('predictionDebug', pred.debug_info || []);
            
            // Sistema de Aprendizaje
            const learning = data.learning_stats || {};
            document.getElementById('modelAccuracy').textContent = `${learning.model_accuracy || 0}%`;
            document.getElementById('trainingSamples').textContent = learning.training_samples || 0;
            document.getElementById('learningState').textContent = learning.learning_status || 'ACTIVO';
            updateDebugPanel('learningDebug', learning.debug_info || []);
            
            // Efectos visuales
            if (data.visual_effects.pulse_animation) {
                document.getElementById('predictionCard').classList.add('pulse');
            } else {
                document.getElementById('predictionCard').classList.remove('pulse');
            }
            
            // Vela Actual
            const candle = data.current_candle;
            document.getElementById('candleProgress').textContent = `${Math.round(candle.progress)}%`;
            document.getElementById('timeRemaining').textContent = `${Math.round(candle.time_remaining)}s`;
            document.getElementById('currentPrice').textContent = candle.price.toFixed(5);
            document.getElementById('ticksProcessed').textContent = candle.ticks_processed;
            
            // Métricas
            const metrics = data.metrics;
            document.getElementById('density').textContent = Math.round(metrics.density);
            document.getElementById('velocity').textContent = metrics.velocity.toFixed(2);
            document.getElementById('acceleration').textContent = metrics.acceleration.toFixed(2);
            document.getElementById('phase').textContent = metrics.phase;
            
            // Performance
            const perf = data.performance;
            document.getElementById('todayAccuracy').textContent = `${perf.today_accuracy}%`;
            document.getElementById('totalSignals').textContent = perf.total_signals;
            document.getElementById('currentStreak').textContent = perf.current_streak;
            updateDebugPanel('performanceDebug', perf.debug_info || []);
            
            // Estado del Sistema
            const status = data.system_status;
            document.getElementById('iqStatus').className = `status-indicator ${status.iq_connection === 'CONNECTED' ? 'status-connected' : 'status-disconnected'}`;
            document.getElementById('iqStatusText').textContent = status.iq_connection;
            document.getElementById('aiStatus').textContent = status.ai_status;
            document.getElementById('lastUpdate').textContent = status.last_update;
            updateDebugPanel('systemDebug', status.debug_info || []);
        }
        
        function updateDebugPanel(panelId, debugItems) {
            const container = document.getElementById(panelId);
            if (!debugItems || debugItems.length === 0) {
                container.innerHTML = '<div class="debug-item">Esperando datos...</div>';
                return;
            }
            
            let html = '';
            debugItems.forEach(item => {
                const level = item.level || 'info';
                const timestamp = item.timestamp || '';
                const message = item.message || '';
                const cssClass = `debug-item debug-${level}`;
                
                html += `<div class="${cssClass}">${timestamp} ${message}</div>`;
            });
            container.innerHTML = html;
            container.scrollTop = container.scrollHeight;
        }
        
        // Inicializar cuando cargue la página
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 Iniciando Delowyss Trading AI V6.3');
            connectWebSocket();
            
            // Verificar conexión periódicamente
            setInterval(() => {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                }
            }, 10000);
        });
    </script>
</body>
</html>
"""

# ------------------ CONEXIÓN REAL IQ OPTION ------------------
try:
    from iqoptionapi.stable_api import IQ_Option
    IQ_OPTION_AVAILABLE = True
    logging.info("✅ iqoptionapi disponible")
except ImportError as e:
    logging.error(f"❌ iqoptionapi no disponible: {e}")
    IQ_OPTION_AVAILABLE = False
    class IQ_Option:
        def __init__(self, email, password): pass
        def connect(self): return False, "Biblioteca no disponible"
        def change_balance(self, balance_type): pass

class RealIQOptionConnector:
    def __init__(self, email, password, pair="EURUSD"):
        self.email = email
        self.password = password
        self.pair = pair
        self.api = None
        self.connected = False
        self.current_price = None
        self.connection_attempts = 0
        self.max_attempts = 5
        self.tick_count = 0
        
    def connect(self):
        if not IQ_OPTION_AVAILABLE:
            logging.error("❌ iqoptionapi no disponible")
            return False
            
        try:
            logging.info(f"🔗 Conectando a IQ Option: {self.email}")
            self.api = IQ_Option(self.email, self.password)
            
            while self.connection_attempts < self.max_attempts:
                check, reason = self.api.connect()
                if check:
                    self.connected = True
                    logging.info("✅ Conexión exitosa a IQ Option")
                    
                    try:
                        self.api.change_balance("PRACTICE")
                        logging.info("💰 Modo: Cuenta PRACTICE")
                    except:
                        logging.info("💰 Modo: Cuenta PRACTICE")
                    
                    self.api.start_candles_stream(self.pair, TIMEFRAME, 1)
                    logging.info(f"📊 Stream iniciado para {self.pair}")
                    
                    time.sleep(2)
                    self._get_initial_price()
                    return True
                else:
                    self.connection_attempts += 1
                    logging.warning(f"⚠️ Intento {self.connection_attempts} fallado: {reason}")
                    time.sleep(3)
                    
            logging.error("❌ No se pudo conectar después de varios intentos")
            return False
            
        except Exception as e:
            logging.error(f"❌ Error en conexión: {e}")
            return False
    
    def _get_initial_price(self):
        try:
            candles = self.api.get_realtime_candles(self.pair, TIMEFRAME)
            if candles:
                for candle_id in candles:
                    candle = candles[candle_id]
                    if 'close' in candle:
                        self.current_price = candle['close']
                        logging.info(f"💰 Precio inicial {self.pair}: {self.current_price}")
                        break
            return self.current_price
        except Exception as e:
            logging.error(f"❌ Error obteniendo precio inicial: {e}")
            return None
    
    def get_realtime_price(self):
        if not self.connected or not self.api:
            logging.error("🔌 No conectado a IQ Option")
            return None
            
        try:
            candles = self.api.get_realtime_candles(self.pair, TIMEFRAME)
            if candles:
                for candle_id in candles:
                    candle = candles[candle_id]
                    if 'close' in candle:
                        new_price = candle['close']
                        if new_price and new_price > 0:
                            self.current_price = new_price
                            self.tick_count += 1
                            return self.current_price
            
            return self.current_price
            
        except Exception as e:
            logging.error(f"❌ Error obteniendo precio: {e}")
            if self.connection_attempts < self.max_attempts:
                logging.info("🔄 Intentando reconectar...")
                self.connect()
            return self.current_price
    
    def get_server_timestamp(self):
        if not self.connected:
            return time.time()
            
        try:
            return self.api.get_server_timestamp()
        except:
            return time.time()
    
    def get_remaining_time(self):
        if not self.connected:
            return TIMEFRAME - (int(time.time()) % TIMEFRAME)
            
        try:
            server_time = self.api.get_server_timestamp()
            if server_time:
                current_second = server_time % TIMEFRAME
                remaining = TIMEFRAME - current_second
                return max(0, min(TIMEFRAME, remaining))
            return TIMEFRAME - (int(time.time()) % TIMEFRAME)
        except Exception as e:
            logging.debug(f"🔧 Error obteniendo tiempo restante: {e}")
            return TIMEFRAME - (int(time.time()) % TIMEFRAME)

# ------------------ METRÓNOMO IQ OPTION ------------------
class IQOptionMetronome:
    def __init__(self):
        self.last_sync_time = 0
        self.server_time_offset = 0
        self.metronome_interval = 1
        self.countdown_active = False
        self.last_5_seconds = False
        
    async def sync_with_iqoption(self, iq_connector):
        try:
            server_time = iq_connector.get_server_timestamp()
            if server_time:
                local_time = time.time()
                self.server_time_offset = server_time - local_time
                self.last_sync_time = local_time
                logging.info("✅ Metrónomo sincronizado con IQ Option")
                return True
        except Exception as e:
            logging.error(f"❌ Error sincronizando metrónomo: {e}")
        return False
    
    def get_remaining_time(self, timeframe=60):
        try:
            current_server_time = time.time() + self.server_time_offset
            remaining = timeframe - (current_server_time % timeframe)
            return max(0, remaining)
        except:
            return 60 - (time.time() % 60)
    
    def is_last_5_seconds(self):
        remaining = self.get_remaining_time()
        return remaining <= 5 and remaining > 0
    
    def is_prediction_time(self):
        """Determina si es momento de hacer predicción (últimos 5 segundos)"""
        remaining = self.get_remaining_time()
        return remaining <= PREDICTION_WINDOW and remaining > 0

# ------------------ ANALIZADOR DE VELA COMPLETA ------------------
class CompleteCandleAnalyzer:
    def __init__(self):
        self.ticks = deque(maxlen=TICK_BUFFER_SIZE)
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.tick_count = 0
        self.price_memory = deque(maxlen=100)
        
        # Almacenar datos de vela anterior para comparación
        self.previous_candle = {
            'open': None,
            'high': None, 
            'low': None,
            'close': None,
            'direction': None,
            'body_size': None
        }
        
        # Métricas de análisis de vela completa
        self.candle_phases = {
            'first_15s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'next_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'middle_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'final_5s': {'ticks': 0, 'analysis': {}, 'completed': False}
        }
        
        # Análisis de comportamiento por segmentos de tiempo
        self.time_segments = {
            '0-15s': {'price_action': [], 'volatility': 0, 'direction': None},
            '15-35s': {'price_action': [], 'volatility': 0, 'direction': None},
            '35-55s': {'price_action': [], 'volatility': 0, 'direction': None},
            '55-60s': {'price_action': [], 'volatility': 0, 'direction': None}
        }
        
        # Indicadores técnicos para la vela actual
        self.velocity_metrics = deque(maxlen=50)
        self.pressure_zones = deque(maxlen=30)
        self.momentum_indicators = deque(maxlen=20)
        self.support_resistance = deque(maxlen=15)
        self.volume_profile = deque(maxlen=25)
        
        self.candle_start_time = None
        self.prediction_ready = False
        
    def add_tick(self, price: float, seconds_remaining: float = None):
        try:
            price = float(price)
            current_time = time.time()
            
            # Inicializar vela si es necesario
            if self.current_candle_open is None:
                self.current_candle_open = self.current_candle_high = self.current_candle_low = price
                self.candle_start_time = current_time
                logging.info("🕯️ Nueva vela iniciada - Análisis completo activado")
                self._reset_candle_analysis()
            
            # Actualizar precios extremos
            self.current_candle_high = max(self.current_candle_high, price)
            self.current_candle_low = min(self.current_candle_low, price)
            self.current_candle_close = price
            
            tick_data = {
                'price': price,
                'timestamp': current_time,
                'seconds_remaining': seconds_remaining,
                'candle_age': current_time - self.candle_start_time if self.candle_start_time else 0,
                'segment': self._get_time_segment(current_time - self.candle_start_time)
            }
            
            self.ticks.append(tick_data)
            self.price_memory.append(price)
            self.tick_count += 1
            
            # Análisis en tiempo real según el segmento
            self._analyze_time_segment(tick_data)
            self._calculate_advanced_metrics(tick_data)
            self._analyze_pressure_zones(tick_data)
            
            # Verificar si se completaron fases
            self._check_phase_completion(tick_data)
            
            return tick_data
        except Exception as e:
            logging.error(f"❌ Error en add_tick: {e}")
            return None
    
    def _get_time_segment(self, candle_age):
        """Determina el segmento de tiempo actual de la vela"""
        if candle_age < 15:
            return '0-15s'
        elif candle_age < 35:
            return '15-35s'
        elif candle_age < 55:
            return '35-55s'
        else:
            return '55-60s'
    
    def _reset_candle_analysis(self):
        """Reinicia el análisis para nueva vela"""
        self.candle_phases = {
            'first_15s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'next_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'middle_20s': {'ticks': 0, 'analysis': {}, 'completed': False},
            'final_5s': {'ticks': 0, 'analysis': {}, 'completed': False}
        }
        self.time_segments = {
            '0-15s': {'price_action': [], 'volatility': 0, 'direction': None},
            '15-35s': {'price_action': [], 'volatility': 0, 'direction': None},
            '35-55s': {'price_action': [], 'volatility': 0, 'direction': None},
            '55-60s': {'price_action': [], 'volatility': 0, 'direction': None}
        }
        self.prediction_ready = False
    
    def _analyze_time_segment(self, tick_data):
        """Analiza el comportamiento del precio en cada segmento de tiempo"""
        segment = tick_data['segment']
        price = tick_data['price']
        
        # Agregar precio al segmento actual
        self.time_segments[segment]['price_action'].append(price)
        
        # Calcular volatilidad del segmento
        if len(self.time_segments[segment]['price_action']) >= 5:
            prices = self.time_segments[segment]['price_action']
            volatility = (max(prices) - min(prices)) * 10000
            self.time_segments[segment]['volatility'] = volatility
            
            # Determinar dirección del segmento
            if len(prices) >= 3:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                positive_changes = len([x for x in price_changes if x > 0])
                buy_ratio = positive_changes / len(price_changes)
                
                if buy_ratio > 0.6:
                    self.time_segments[segment]['direction'] = 'ALCISTA'
                elif buy_ratio < 0.4:
                    self.time_segments[segment]['direction'] = 'BAJISTA'
                else:
                    self.time_segments[segment]['direction'] = 'LATERAL'
    
    def _check_phase_completion(self, tick_data):
        """Verifica y actualiza el estado de completitud de las fases"""
        candle_age = tick_data['candle_age']
        segment = tick_data['segment']
        
        # Primera fase: 0-15 segundos
        if candle_age >= 15 and not self.candle_phases['first_15s']['completed']:
            self.candle_phases['first_15s']['completed'] = True
            self.candle_phases['first_15s']['analysis'] = self._analyze_phase('first_15s')
            logging.info("📊 Fase 0-15s completada - Análisis inicial listo")
        
        # Segunda fase: 15-35 segundos  
        elif candle_age >= 35 and not self.candle_phases['next_20s']['completed']:
            self.candle_phases['next_20s']['completed'] = True
            self.candle_phases['next_20s']['analysis'] = self._analyze_phase('next_20s')
            logging.info("📊 Fase 15-35s completada - Tendencia definiéndose")
        
        # Tercera fase: 35-55 segundos
        elif candle_age >= 55 and not self.candle_phases['middle_20s']['completed']:
            self.candle_phases['middle_20s']['completed'] = True
            self.candle_phases['middle_20s']['analysis'] = self._analyze_phase('middle_20s')
            logging.info("📊 Fase 35-55s completada - Comportamiento establecido")
        
        # Cuarta fase: Últimos 5 segundos (para predicción)
        elif segment == '55-60s' and not self.candle_phases['final_5s']['completed']:
            self.candle_phases['final_5s']['completed'] = True
            self.candle_phases['final_5s']['analysis'] = self._analyze_phase('final_5s')
            self.prediction_ready = True
            logging.info("🎯 Fase 55-60s - Predicción habilitada")
    
    def _analyze_phase(self, phase):
        """Analiza una fase específica de la vela"""
        try:
            if phase == 'first_15s':
                segment_data = self.time_segments['0-15s']
            elif phase == 'next_20s':
                segment_data = self.time_segments['15-35s']
            elif phase == 'middle_20s':
                segment_data = self.time_segments['35-55s']
            else:  # final_5s
                segment_data = self.time_segments['55-60s']
            
            if not segment_data['price_action']:
                return {}
            
            prices = segment_data['price_action']
            
            # Análisis básico
            high = max(prices)
            low = min(prices)
            open_price = prices[0] if prices else 0
            close_price = prices[-1] if prices else 0
            
            volatility = (high - low) * 10000
            body_size = abs(close_price - open_price) * 10000
            body_direction = 'ALCISTA' if close_price > open_price else 'BAJISTA' if close_price < open_price else 'LATERAL'
            
            # Análisis de tendencia intra-fase
            if len(prices) >= 5:
                x_values = np.arange(len(prices))
                try:
                    trend_coeff = np.polyfit(x_values, prices, 1)[0]
                    trend_strength = abs(trend_coeff) * 10000
                    trend_direction = 'ALCISTA' if trend_coeff > 0 else 'BAJISTA' if trend_coeff < 0 else 'LATERAL'
                except:
                    trend_strength = 0
                    trend_direction = 'LATERAL'
            else:
                trend_strength = 0
                trend_direction = segment_data.get('direction', 'LATERAL')
            
            # Presión de compra/venta
            if len(prices) >= 3:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes)
            else:
                buy_pressure = 0.5
            
            return {
                'prices_analyzed': len(prices),
                'volatility': volatility,
                'body_size': body_size,
                'body_direction': body_direction,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'buy_pressure': buy_pressure,
                'high': high,
                'low': low,
                'open': open_price,
                'close': close_price
            }
        except Exception as e:
            logging.debug(f"🔧 Error analizando fase {phase}: {e}")
            return {}
    
    def _calculate_advanced_metrics(self, tick_data):
        """Calcula métricas avanzadas en tiempo real"""
        try:
            current_price = tick_data['price']
            current_time = tick_data['timestamp']

            if len(self.ticks) >= 2:
                previous_tick = list(self.ticks)[-2]
                time_diff = current_time - previous_tick['timestamp']
                
                if time_diff > 0:
                    price_diff = current_price - previous_tick['price']
                    velocity = price_diff / time_diff
                    
                    self.velocity_metrics.append({
                        'velocity': velocity,
                        'timestamp': current_time,
                        'price_change': price_diff
                    })
        except Exception as e:
            logging.debug(f"🔧 Error en cálculo avanzado: {e}")
    
    def _analyze_pressure_zones(self, tick_data):
        """Analiza zonas de presión de compra/venta"""
        try:
            if len(self.ticks) < 6:
                return
                
            recent_ticks = list(self.ticks)[-6:]
            price_changes = []
            
            for i in range(1, len(recent_ticks)):
                change = recent_ticks[i]['price'] - recent_ticks[i-1]['price']
                price_changes.append(change)
            
            if price_changes:
                buy_pressure = len([x for x in price_changes if x > 0]) / len(price_changes)
                
                self.pressure_zones.append({
                    'buy_pressure': buy_pressure,
                    'sell_pressure': 1 - buy_pressure,
                    'timestamp': tick_data['timestamp'],
                    'strength': abs(buy_pressure - 0.5) * 2
                })
                
        except Exception as e:
            logging.debug(f"🔧 Error analizando presión: {e}")
    
    def get_candle_analysis(self):
        """Obtiene el análisis completo de la vela actual"""
        if self.tick_count < MIN_TICKS_FOR_PREDICTION:
            return {
                'status': 'ANALYZING',
                'tick_count': self.tick_count,
                'message': f'Analizando vela: {self.tick_count}/{MIN_TICKS_FOR_PREDICTION} ticks',
                'current_progress': self._get_candle_progress()
            }
        
        try:
            # Análisis de fases completadas
            phase_analysis = {}
            for phase, data in self.candle_phases.items():
                if data['completed']:
                    phase_analysis[phase] = data['analysis']
            
            # Análisis de segmentos de tiempo
            segment_analysis = {}
            for segment, data in self.time_segments.items():
                if data['price_action']:
                    segment_analysis[segment] = {
                        'direction': data['direction'],
                        'volatility': data['volatility'],
                        'samples': len(data['price_action'])
                    }
            
            # Análisis general de la vela
            general_analysis = self._analyze_complete_candle()
            
            # Preparar predicción si está lista
            prediction_readiness = self._assess_prediction_readiness()
            
            result = {
                'status': 'COMPLETE_ANALYSIS',
                'tick_count': self.tick_count,
                'current_price': self.current_candle_close,
                'candle_progress': self._get_candle_progress(),
                'candle_stats': {
                    'open': self.current_candle_open,
                    'high': self.current_candle_high,
                    'low': self.current_candle_low,
                    'close': self.current_candle_close,
                    'range': (self.current_candle_high - self.current_candle_low) * 10000,
                    'body_size': abs(self.current_candle_close - self.current_candle_open) * 10000,
                    'direction': 'ALCISTA' if self.current_candle_close > self.current_candle_open else 'BAJISTA' if self.current_candle_close < self.current_candle_open else 'LATERAL'
                },
                'phase_analysis': phase_analysis,
                'segment_analysis': segment_analysis,
                'general_analysis': general_analysis,
                'prediction_readiness': prediction_readiness,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Error en análisis de vela: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def _get_candle_progress(self):
        """Calcula el progreso actual de la vela"""
        if not self.candle_start_time:
            return 0
        candle_age = time.time() - self.candle_start_time
        return min(100, (candle_age / TIMEFRAME) * 100)
    
    def _analyze_complete_candle(self):
        """Analiza el comportamiento completo de la vela"""
        try:
            if not self.ticks:
                return {}
            
            # Comportamiento por cuartos
            total_ticks = len(self.ticks)
            quarter_size = max(1, total_ticks // 4)
            
            quarters = []
            for i in range(4):
                start_idx = i * quarter_size
                end_idx = min((i + 1) * quarter_size, total_ticks)
                quarter_ticks = list(self.ticks)[start_idx:end_idx]
                
                if quarter_ticks:
                    quarter_prices = [t['price'] for t in quarter_ticks]
                    quarter_direction = 'ALCISTA' if quarter_prices[-1] > quarter_prices[0] else 'BAJISTA' if quarter_prices[-1] < quarter_prices[0] else 'LATERAL'
                    quarter_volatility = (max(quarter_prices) - min(quarter_prices)) * 10000
                    
                    quarters.append({
                        'quarter': i + 1,
                        'direction': quarter_direction,
                        'volatility': quarter_volatility,
                        'ticks': len(quarter_ticks)
                    })
            
            # Consistencia de la vela
            consistency_score = self._calculate_consistency()
            
            # Fuerza de la tendencia
            trend_strength = self._calculate_trend_strength()
            
            return {
                'quarters_analysis': quarters,
                'consistency_score': consistency_score,
                'trend_strength': trend_strength,
                'total_volatility': (self.current_candle_high - self.current_candle_low) * 10000,
                'current_momentum': self._calculate_current_momentum(),
                'pressure_balance': self._calculate_pressure_balance()
            }
        except Exception as e:
            logging.debug(f"🔧 Error en análisis completo: {e}")
            return {}
    
    def _calculate_consistency(self):
        """Calcula la consistencia del movimiento de la vela"""
        try:
            if len(self.ticks) < 10:
                return 50
                
            directions = []
            for segment in self.time_segments.values():
                if segment['direction']:
                    dir_map = {'ALCISTA': 1, 'BAJISTA': -1, 'LATERAL': 0}
                    directions.append(dir_map[segment['direction']])
            
            if directions:
                consistency = 1 - (np.std(directions) / 2)  # Normalizar a 0-1
                return min(100, consistency * 100)
            return 50
        except:
            return 50
    
    def _calculate_trend_strength(self):
        """Calcula la fuerza de la tendencia general"""
        try:
            if len(self.price_memory) < 8:
                return 0
                
            prices = list(self.price_memory)
            x_values = np.arange(len(prices))
            trend_coeff = np.polyfit(x_values, prices, 1)[0]
            return abs(trend_coeff) * 10000
        except:
            return 0
    
    def _calculate_current_momentum(self):
        """Calcula el momentum actual"""
        try:
            if len(self.price_memory) < 5:
                return 0
            prices = list(self.price_memory)[-5:]
            return (prices[-1] - prices[0]) * 10000
        except:
            return 0
    
    def _calculate_pressure_balance(self):
        """Calcula el balance de presión"""
        try:
            if not self.pressure_zones:
                return 0.5
            recent_pressure = list(self.pressure_zones)[-5:]
            avg_buy_pressure = np.mean([p['buy_pressure'] for p in recent_pressure])
            return avg_buy_pressure
        except:
            return 0.5
    
    def _assess_prediction_readiness(self):
        """Evalúa si el sistema está listo para predecir"""
        try:
            readiness = {
                'ready': self.prediction_ready,
                'phases_completed': sum(1 for phase in self.candle_phases.values() if phase['completed']),
                'total_phases': len(self.candle_phases),
                'data_sufficiency': min(100, (self.tick_count / 30) * 100),
                'analysis_quality': self._calculate_analysis_quality()
            }
            
            # Solo considerar listo si tenemos al menos 3 fases completas y suficientes ticks
            readiness['ready'] = (
                readiness['phases_completed'] >= 3 and 
                readiness['data_sufficiency'] >= 70 and
                self.prediction_ready
            )
            
            return readiness
        except:
            return {'ready': False, 'phases_completed': 0, 'data_sufficiency': 0}
    
    def _calculate_analysis_quality(self):
        """Calcula la calidad general del análisis"""
        quality = 0
        
        # Por cada fase completada
        quality += sum(20 for phase in self.candle_phases.values() if phase['completed'])
        
        # Por ticks suficientes
        quality += min(30, (self.tick_count / 40) * 30)
        
        # Por consistencia
        quality += min(20, self._calculate_consistency() / 5)
        
        return min(100, quality)
    
    def is_ready_for_prediction(self):
        """Verifica si el sistema está listo para hacer predicción"""
        readiness = self._assess_prediction_readiness()
        return readiness['ready']
    
    def reset(self):
        """Prepara el analyzer para la siguiente vela"""
        try:
            # Guardar vela actual como anterior
            if all([self.current_candle_open, self.current_candle_high, 
                   self.current_candle_low, self.current_candle_close]):
                self.previous_candle = {
                    'open': self.current_candle_open,
                    'high': self.current_candle_high,
                    'low': self.current_candle_low,
                    'close': self.current_candle_close,
                    'direction': 'ALCISTA' if self.current_candle_close > self.current_candle_open else 'BAJISTA',
                    'body_size': abs(self.current_candle_close - self.current_candle_open) * 10000
                }
            
            # Reiniciar para nueva vela
            self.ticks.clear()
            self.current_candle_open = None
            self.current_candle_high = None
            self.current_candle_low = None
            self.current_candle_close = None
            self.tick_count = 0
            self.price_memory.clear()
            self.velocity_metrics.clear()
            self.pressure_zones.clear()
            self.momentum_indicators.clear()
            self.candle_start_time = None
            self.prediction_ready = False
            
            # Mantener el reset de fases en _reset_candle_analysis
                
        except Exception as e:
            logging.error(f"❌ Error en reset: {e}")

# ------------------ SISTEMA DE AUTOAPRENDIZAJE AVANZADO MEJORADO - VERSIÓN DEFINITIVA ------------------
class AdvancedLearningSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_data = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=500)
        self.model_accuracy_history = deque(maxlen=100)
        self.last_training_time = 0
        self.training_interval = 300
        self.min_training_samples = 10  # 🚀 AUMENTADO para tener suficientes datos
        self.model_accuracy = 0.0
        self.feature_importance = {}
        self.debug_logs = deque(maxlen=50)
        
        # Cargar modelo existente si existe
        self.load_model()
        
        self._add_debug_log("info", "Sistema de aprendizaje inicializado - VERSIÓN DEFINITIVA")
        
        # 🚀 INICIALIZACIÓN MEJORADA - SIN ENTRENAMIENTO INMEDIATO
        self.initialize_ml_system()

    def _add_debug_log(self, level: str, message: str):
        """Agrega log de debug"""
        log_entry = {
            'level': level,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': message
        }
        self.debug_logs.append(log_entry)
        logging.info(f"🧠 [LEARNING_DEBUG] {message}")

    def initialize_ml_system(self):
        """Inicializa el sistema ML SIN entrenamiento inmediato - VERSIÓN DEFINITIVA"""
        try:
            # 🚀 SOLO crear datos sintéticos, NO entrenar inmediatamente
            if len(self.training_data) < 10:
                self._add_debug_log("info", "Preparando datos de entrenamiento iniciales...")
                
                # Datos sintéticos balanceados PERO NO entrenar hasta tener suficientes
                synthetic_samples = [
                    # Muestras ALCISTAS (4 muestras)
                    {'features': [5.0, 8.0, 0.7, 0.8, 0.6, 0.3, 0.9, 0.7, 0.2, 0.5, 0.4, 0.6, 0.8, 0.3, 0.7, 0.5, 0.6, 0.4, 0.5, 0.6], 'label': 0, 'confidence': 70},
                    {'features': [4.0, 7.0, 0.6, 0.7, 0.5, 0.4, 0.8, 0.6, 0.3, 0.6, 0.5, 0.7, 0.7, 0.4, 0.6, 0.6, 0.7, 0.5, 0.6, 0.7], 'label': 0, 'confidence': 65},
                    {'features': [6.0, 9.0, 0.8, 0.9, 0.7, 0.2, 0.95, 0.8, 0.1, 0.4, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.8, 0.3, 0.4, 0.8], 'label': 0, 'confidence': 75},
                    {'features': [5.5, 8.5, 0.75, 0.85, 0.65, 0.25, 0.85, 0.75, 0.15, 0.45, 0.35, 0.65, 0.85, 0.25, 0.75, 0.45, 0.75, 0.35, 0.45, 0.75], 'label': 0, 'confidence': 68},
                    # Muestras BAJISTAS (4 muestras)  
                    {'features': [3.0, 5.0, 0.3, 0.2, 0.4, 0.7, 0.1, 0.3, 0.8, 0.5, 0.6, 0.4, 0.2, 0.7, 0.3, 0.5, 0.4, 0.6, 0.5, 0.4], 'label': 1, 'confidence': 65},
                    {'features': [2.0, 4.0, 0.4, 0.3, 0.3, 0.6, 0.2, 0.4, 0.7, 0.4, 0.7, 0.3, 0.3, 0.6, 0.4, 0.4, 0.3, 0.7, 0.4, 0.3], 'label': 1, 'confidence': 60},
                    {'features': [2.5, 4.5, 0.35, 0.25, 0.35, 0.65, 0.15, 0.35, 0.75, 0.45, 0.65, 0.35, 0.25, 0.65, 0.35, 0.45, 0.35, 0.65, 0.45, 0.35], 'label': 1, 'confidence': 62},
                    {'features': [3.5, 5.5, 0.25, 0.15, 0.45, 0.75, 0.05, 0.25, 0.85, 0.55, 0.55, 0.45, 0.15, 0.75, 0.25, 0.55, 0.45, 0.55, 0.55, 0.45], 'label': 1, 'confidence': 67},
                    # Muestras LATERALES (4 muestras)
                    {'features': [2.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 'label': 2, 'confidence': 50},
                    {'features': [3.0, 4.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 'label': 2, 'confidence': 55},
                    {'features': [4.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 'label': 2, 'confidence': 52},
                    {'features': [3.5, 4.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 'label': 2, 'confidence': 53},
                ]
                
                for sample in synthetic_samples:
                    self.training_data.append(sample)
                
                self._add_debug_log("success", f"Datos sintéticos preparados: {len(self.training_data)} muestras")
                # 🚀 NO entrenar aquí - esperar a tener datos reales
                    
        except Exception as e:
            self._add_debug_log("error", f"Error inicializando ML: {e}")

    def extract_advanced_features(self, candle_analysis, current_price, market_context):
        """Extrae características avanzadas para el modelo de ML"""
        features = {}
        
        try:
            # 1. Características de la vela actual
            candle_stats = candle_analysis.get('candle_stats', {})
            features['body_size'] = candle_stats.get('body_size', 0)
            features['range_size'] = candle_stats.get('range', 0)
            features['body_ratio'] = features['body_size'] / max(features['range_size'], 0.0001)
            features['is_doji'] = 1 if features['body_ratio'] < 0.1 else 0
            
            # 2. Características de tendencia por fases
            phase_analysis = candle_analysis.get('phase_analysis', {})
            phase_directions = []
            phase_strengths = []
            
            for phase, analysis in phase_analysis.items():
                if analysis.get('trend_direction'):
                    dir_map = {'ALCISTA': 1, 'BAJISTA': -1, 'LATERAL': 0}
                    phase_directions.append(dir_map.get(analysis['trend_direction'], 0))
                    phase_strengths.append(analysis.get('trend_strength', 0))
            
            features['phase_direction_std'] = np.std(phase_directions) if phase_directions else 0
            features['phase_strength_avg'] = np.mean(phase_strengths) if phase_strengths else 0
            features['phase_consistency'] = 1 - features['phase_direction_std']  # 1 = máxima consistencia
            
            # 3. Características de momentum y presión
            general_analysis = candle_analysis.get('general_analysis', {})
            features['momentum'] = general_analysis.get('current_momentum', 0)
            features['pressure_balance'] = general_analysis.get('pressure_balance', 0.5)
            features['consistency_score'] = general_analysis.get('consistency_score', 50) / 100.0
            
            # 4. Características de segmentos de tiempo
            segment_analysis = candle_analysis.get('segment_analysis', {})
            segment_directions = []
            recent_segment_strength = 0
            
            for segment, data in segment_analysis.items():
                if data.get('direction'):
                    dir_map = {'ALCISTA': 1, 'BAJISTA': -1, 'LATERAL': 0}
                    segment_directions.append(dir_map.get(data['direction'], 0))
                    
                # Dar más peso a segmentos recientes
                if segment in ['35-55s', '55-60s']:
                    recent_segment_strength += data.get('volatility', 0)
            
            features['segment_trend'] = np.mean(segment_directions) if segment_directions else 0
            features['recent_volatility'] = recent_segment_strength
            
            # 5. Características de comportamiento del mercado
            features['price_acceleration'] = market_context.get('price_acceleration', 0)
            features['volatility_ratio'] = market_context.get('volatility_ratio', 1.0)
            features['market_regime'] = market_context.get('market_regime', 0)
            
            # 6. Características temporales
            current_time = datetime.utcnow()
            features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
            features['minute'] = current_time.minute / 60.0
            
            # 7. Características de patrones históricos
            features['previous_candle_alignment'] = market_context.get('previous_candle_alignment', 0)
            features['trend_continuation'] = market_context.get('trend_continuation', 0)
            
            self.feature_names = list(features.keys())
            
            self._add_debug_log("info", f"Características extraídas: {len(features)}")
            return features
            
        except Exception as e:
            self._add_debug_log("error", f"Error extrayendo características: {e}")
            return {}
    
    def add_training_sample(self, features, actual_direction, prediction_confidence):
        """Agrega una muestra de entrenamiento al dataset"""
        try:
            if not features or actual_direction is None:
                self._add_debug_log("warning", "Datos insuficientes para muestra de entrenamiento")
                return
            
            # Mapear dirección a label numérico
            direction_map = {'ALZA': 0, 'BAJA': 1, 'LATERAL': 2}
            label = direction_map.get(actual_direction, 2)
            
            sample = {
                'features': list(features.values()),
                'label': label,
                'confidence': prediction_confidence,
                'timestamp': time.time()
            }
            
            self.training_data.append(sample)
            
            self._add_debug_log("info", f"Muestra agregada. Total: {len(self.training_data)}. Dirección: {actual_direction}")
            
        except Exception as e:
            self._add_debug_log("error", f"Error agregando muestra: {e}")
    
    def train_model(self):
        """Entrena el modelo de machine learning - VERSIÓN DEFINITIVA CORREGIDA"""
        if not SKLEARN_AVAILABLE:
            self._add_debug_log("warning", "scikit-learn no disponible")
            return False
            
        try:
            if len(self.training_data) < self.min_training_samples:
                self._add_debug_log("warning", 
                    f"Muestras insuficientes: {len(self.training_data)}/{self.min_training_samples}")
                return False
            
            self._add_debug_log("info", f"Iniciando entrenamiento con {len(self.training_data)} muestras")
            
            # Preparar datos
            X = np.array([sample['features'] for sample in self.training_data])
            y = np.array([sample['label'] for sample in self.training_data])
            
            # 🚀 VERIFICACIÓN CRÍTICA MEJORADA
            unique_classes, counts = np.unique(y, return_counts=True)
            self._add_debug_log("info", f"Clases: {unique_classes}, Conteos: {counts}")
            
            # Verificar que tenemos al menos 2 muestras por clase
            if len(unique_classes) < 2:
                self._add_debug_log("warning", f"Se necesitan al menos 2 clases. Clases disponibles: {len(unique_classes)}")
                return False
                
            if np.any(counts < 2):
                self._add_debug_log("warning", f"Clases con menos de 2 muestras: {dict(zip(unique_classes, counts))}")
                return False
            
            # 🚀 AJUSTE DINÁMICO DEL test_size
            n_samples = len(y)
            n_classes = len(unique_classes)
            
            # Calcular test_size que funcione para nuestras clases
            min_test_size = n_classes * 2  # Mínimo 2 por clase en test
            if n_samples < min_test_size + 5:  # Si no hay suficientes datos
                test_size = 0.1  # Usar 10% para test
            else:
                test_size = 0.2  # Usar 20% para test
                
            self._add_debug_log("info", f"n_samples: {n_samples}, n_classes: {n_classes}, test_size: {test_size}")
            
            # Dividir datos con manejo de errores
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError as e:
                self._add_debug_log("warning", f"Error en train_test_split: {e}. Usando división manual.")
                # División manual simple
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Verificar que tenemos datos en train y test
            if len(X_train) == 0 or len(X_test) == 0:
                self._add_debug_log("warning", "No hay suficientes datos para entrenamiento y prueba")
                return False
            
            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 🚀 ENTRENAR SOLO UN MODELO SIMPLE INICIALMENTE
            if len(self.training_data) < 20:  # Pocos datos → modelo simple
                self.model = RandomForestClassifier(
                    n_estimators=20,  # Menos estimadores
                    max_depth=5,      # Menos profundidad
                    min_samples_split=3,
                    random_state=42
                )
                self._add_debug_log("info", "Usando modelo simple (pocos datos)")
            else:
                # Muchos datos → modelo más complejo
                self.model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_split=5,
                    random_state=42
                )
                self._add_debug_log("info", "Usando modelo complejo (suficientes datos)")
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluar modelo
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_accuracy = accuracy
            self._add_debug_log("success", f"Modelo entrenado - Accuracy: {accuracy:.3f}")
            
            # Guardar importancia de características
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                self._add_debug_log("info", f"Top features: {top_features}")
            
            self.model_accuracy_history.append(self.model_accuracy)
            self.last_training_time = time.time()
            
            # Guardar modelo
            self.save_model()
            
            return True
            
        except Exception as e:
            self._add_debug_log("error", f"Error entrenando modelo: {e}")
            import traceback
            self._add_debug_log("error", f"Traceback: {traceback.format_exc()}")
            return False
    
    def predict_with_ml(self, features):
        """Realiza predicción usando el modelo de ML"""
        try:
            if self.model is None:
                self._add_debug_log("warning", "Modelo ML no disponible")
                return None, 0.0
            
            if not features:
                self._add_debug_log("warning", "Características vacías para predicción ML")
                return None, 0.0
            
            X = np.array([list(features.values())])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            probability = np.max(self.model.predict_proba(X_scaled))
            
            # Mapear de vuelta a dirección
            direction_map = {0: 'ALZA', 1: 'BAJA', 2: 'LATERAL'}
            predicted_direction = direction_map.get(prediction, 'LATERAL')
            
            self._add_debug_log("info", 
                f"Predicción ML: {predicted_direction} ({probability*100:.1f}%)")
            
            return predicted_direction, probability * 100
            
        except Exception as e:
            self._add_debug_log("error", f"Error en predicción ML: {e}")
            return None, 0.0
    
    def get_adaptive_confidence(self, ml_confidence, traditional_confidence, features):
        """Calcula confianza adaptativa basada en condiciones del mercado"""
        try:
            base_confidence = (ml_confidence * 0.6 + traditional_confidence * 0.4)
            
            # Ajustar confianza basado en consistencia de características
            consistency = features.get('phase_consistency', 0.5)
            volatility = features.get('recent_volatility', 0)
            
            # Reducir confianza en condiciones volátiles o inconsistentes
            if consistency < 0.3:
                base_confidence *= 0.7
            elif volatility > 50:
                base_confidence *= 0.8
            elif consistency > 0.8:
                base_confidence *= 1.1
            
            final_confidence = min(95, max(50, base_confidence))
            
            self._add_debug_log("info", 
                f"Confianza adaptativa: {final_confidence:.1f}% (ML: {ml_confidence:.1f}%, Trad: {traditional_confidence:.1f}%)")
            
            return final_confidence
            
        except Exception as e:
            self._add_debug_log("error", f"Error calculando confianza adaptativa: {e}")
            return traditional_confidence
    
    def save_model(self):
        """Guarda el modelo entrenado"""
        try:
            if self.model is not None:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'accuracy': self.model_accuracy,
                    'feature_importance': self.feature_importance,
                    'last_trained': self.last_training_time
                }
                
                with open('advanced_ml_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                
                self._add_debug_log("success", "Modelo guardado exitosamente")
                
        except Exception as e:
            self._add_debug_log("error", f"Error guardando modelo: {e}")
    
    def load_model(self):
        """Carga un modelo previamente entrenado"""
        try:
            if os.path.exists('advanced_ml_model.pkl'):
                with open('advanced_ml_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.model_accuracy = model_data.get('accuracy', 0.0)
                self.feature_importance = model_data.get('feature_importance', {})
                self.last_training_time = model_data.get('last_trained', 0)
                
                self._add_debug_log("success", 
                    f"Modelo cargado - Accuracy: {self.model_accuracy:.3f}")
                return True
                
        except Exception as e:
            self._add_debug_log("error", f"Error cargando modelo: {e}")
        
        return False
    
    def get_learning_stats(self):
        """Obtiene estadísticas del sistema de aprendizaje"""
        top_features = []
        if self.feature_importance:
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            top_features = [{'name': name, 'importance': importance * 100} for name, importance in sorted_features]
        
        return {
            'model_accuracy': round(self.model_accuracy * 100, 2),
            'training_samples': len(self.training_data),
            'feature_importance': self.feature_importance,
            'top_features': top_features,
            'last_training': datetime.fromtimestamp(self.last_training_time).isoformat() if self.last_training_time > 0 else 'Nunca',
            'accuracy_trend': list(self.model_accuracy_history)[-5:] if self.model_accuracy_history else [],
            'debug_info': list(self.debug_logs)[-5:]
        }

# ------------------ PREDICTOR MEJORADO CON AUTOAPRENDIZAJE MEJORADO - VERSIÓN DEFINITIVA ------------------
class EnhancedNextCandlePredictor:
    def __init__(self):
        self.analyzer = CompleteCandleAnalyzer()
        self.learning_system = AdvancedLearningSystem()
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'current_streak': 0,
            'best_streak': 0,
            'today_signals': 0,
            'total_signals': 0,
            'prediction_history': [],
            'bias_tracking': {
                'alza_count': 1,  # 🚀 CORRECCIÓN: Inicializado en 1 para evitar división por cero
                'baja_count': 1,  # 🚀 CORRECCIÓN: Inicializado en 1 para evitar división por cero  
                'lateral_count': 1  # 🚀 CORRECCIÓN: Inicializado en 1 para evitar división por cero
            }
        }
        self.prediction_history = deque(maxlen=50)
        self.market_context = {
            'previous_candle_alignment': 0,
            'trend_continuation': 0,
            'price_acceleration': 0,
            'volatility_ratio': 1.0,
            'market_regime': 0
        }
        self.last_prediction_features = None
        self.auto_learning_active = True
        self.debug_logs = deque(maxlen=50)
        
        self._add_debug_log("info", "Predictor mejorado inicializado - VERSIÓN DEFINITIVA")
    
    def _add_debug_log(self, level: str, message: str):
        """Agrega log de debug"""
        log_entry = {
            'level': level,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': message
        }
        self.debug_logs.append(log_entry)
        logging.info(f"🎯 [PREDICTION_DEBUG] {message}")
    
    def process_tick(self, price: float, seconds_remaining: float = None):
        return self.analyzer.add_tick(price, seconds_remaining)
    
    def update_market_context(self, candle_analysis):
        """Actualiza el contexto del mercado para el aprendizaje"""
        try:
            # Calcular alineación con vela anterior
            if hasattr(self.analyzer, 'previous_candle'):
                prev_candle = self.analyzer.previous_candle
                current_direction = candle_analysis.get('candle_stats', {}).get('direction', 'LATERAL')
                
                if prev_candle.get('direction') and current_direction:
                    dir_map = {'ALCISTA': 1, 'BAJISTA': -1, 'LATERAL': 0}
                    prev_dir = dir_map.get(prev_candle['direction'], 0)
                    curr_dir = dir_map.get(current_direction, 0)
                    
                    self.market_context['previous_candle_alignment'] = 1 if prev_dir == curr_dir else -1
            
            # Calcular aceleración del precio
            if len(self.analyzer.velocity_metrics) >= 3:
                recent_velocities = [v['velocity'] for v in list(self.analyzer.velocity_metrics)[-3:]]
                if len(recent_velocities) >= 2:
                    self.market_context['price_acceleration'] = recent_velocities[-1] - recent_velocities[0]
            
            # Determinar régimen de mercado basado en volatilidad
            volatility = candle_analysis.get('candle_stats', {}).get('range', 0)
            if volatility > 15:
                self.market_context['market_regime'] = 1  # Volátil
            elif volatility < 5:
                self.market_context['market_regime'] = 2  # Tranquilo
            else:
                self.market_context['market_regime'] = 0  # Normal
                
        except Exception as e:
            self._add_debug_log("error", f"Error actualizando contexto: {e}")
    
    def predict_next_candle(self):
        """Predice usando el sistema combinado tradicional + ML - VERSIÓN DEFINITIVA"""
        analysis = self.analyzer.get_candle_analysis()
        
        if analysis.get('status') != 'COMPLETE_ANALYSIS':
            self._add_debug_log("warning", "Análisis incompleto, usando predicción base")
            return self._get_base_prediction(analysis)
        
        if not self.analyzer.is_ready_for_prediction():
            self._add_debug_log("warning", "Sistema no listo para predicción")
            return self._get_base_prediction(analysis)
        
        try:
            self._add_debug_log("info", "Iniciando predicción mejorada")
            
            # 1. Obtener predicción tradicional
            traditional_prediction = self._get_traditional_prediction(analysis)
            self._add_debug_log("info", 
                f"Predicción tradicional: {traditional_prediction['direction']} {traditional_prediction['confidence']}%")
            
            # 2. Actualizar contexto de mercado
            self.update_market_context(analysis)
            
            # 3. Extraer características para ML
            features = self.learning_system.extract_advanced_features(
                analysis, 
                self.analyzer.current_candle_close, 
                self.market_context
            )
            
            if features:
                feature_summary = {k: round(v, 4) for k, v in features.items()}
                self._add_debug_log("info", f"Características ML: {len(features)} extraídas")
                self.last_prediction_features = features
            else:
                self._add_debug_log("warning", "No se pudieron extraer características para ML")
                traditional_prediction['method'] = 'TRADICIONAL'
                traditional_prediction['debug_info'] = list(self.debug_logs)[-3:]
                return traditional_prediction
            
            # 4. Obtener predicción de ML
            ml_direction, ml_confidence = self.learning_system.predict_with_ml(features)
            
            # VERIFICAR ACTIVACIÓN DE ML - VERSIÓN DEFINITIVA
            if ml_direction and ml_confidence > 55 and self.learning_system.model is not None:
                self._add_debug_log("success", 
                    f"ML ACTIVADO: {ml_direction} {ml_confidence:.1f}%")
                
                # 5. Combinar predicciones de manera inteligente
                combined_confidence = self.learning_system.get_adaptive_confidence(
                    ml_confidence, 
                    traditional_prediction['confidence'],
                    features
                )
                
                # Decidir dirección final (favorecer ML si es consistente)
                final_direction = self._combine_predictions(
                    traditional_prediction['direction'], 
                    ml_direction, 
                    traditional_prediction['confidence'],
                    ml_confidence
                )
                
                # Actualizar predicción con resultados combinados
                traditional_prediction['direction'] = final_direction
                traditional_prediction['confidence'] = combined_confidence
                traditional_prediction['ml_confidence'] = ml_confidence
                traditional_prediction['traditional_confidence'] = traditional_prediction['confidence']
                traditional_prediction['method'] = 'HÍBRIDO_ML'
                
                self._add_debug_log("success", 
                    f"Predicción combinada: {final_direction} {combined_confidence:.1f}%")
                
            else:
                traditional_prediction['method'] = 'TRADICIONAL'
                if ml_direction:
                    self._add_debug_log("warning", 
                        f"ML no activado: confianza {ml_confidence:.1f}% < 55% o modelo no disponible")
                else:
                    self._add_debug_log("warning", "ML no disponible para esta predicción")
            
            # ACTUALIZAR SEGUIMIENTO DE SESGO
            self._update_bias_tracking(traditional_prediction['direction'])
            
            # 6. Entrenamiento automático periódico MEJORADO - VERSIÓN DEFINITIVA
            if self.auto_learning_active:
                current_time = time.time()
                training_interval = self.learning_system.training_interval
                
                # 🚀 SOLO entrenar si tenemos suficientes datos NUEVOS
                current_samples = len(self.learning_system.training_data)
                has_new_data = current_samples > 15  # Solo si tenemos datos significativos
                
                if (has_new_data and 
                    current_time - self.learning_system.last_training_time > training_interval):
                    
                    self._add_debug_log("info", f"Ejecutando entrenamiento automático con {current_samples} muestras...")
                    if self.learning_system.train_model():
                        self._add_debug_log("success", "✅ Entrenamiento automático completado")
                    else:
                        self._add_debug_log("warning", "⚠️ Entrenamiento automático falló - continuando en modo tradicional")
            
            # Actualizar contadores
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['today_signals'] += 1
            self.performance_stats['total_signals'] += 1
            
            # AGREGAR DEBUG INFO A LA PREDICCIÓN
            traditional_prediction['debug_info'] = list(self.debug_logs)[-3:]
            traditional_prediction['status'] = 'PREDICTION_READY'
            
            return traditional_prediction
            
        except Exception as e:
            self._add_debug_log("error", f"Error en predicción mejorada: {e}")
            return self._get_base_prediction(analysis)
    
    def _update_bias_tracking(self, direction: str):
        """Actualiza el seguimiento de sesgo predictivo"""
        if direction == 'ALZA':
            self.performance_stats['bias_tracking']['alza_count'] += 1
        elif direction == 'BAJA':
            self.performance_stats['bias_tracking']['baja_count'] += 1
        else:
            self.performance_stats['bias_tracking']['lateral_count'] += 1
        
        total = sum(self.performance_stats['bias_tracking'].values())
        if total >= 15:  # 🚀 CORRECCIÓN: Aumentado a 15 para análisis más robusto
            alza_pct = (self.performance_stats['bias_tracking']['alza_count'] / total) * 100
            baja_pct = (self.performance_stats['bias_tracking']['baja_count'] / total) * 100
            
            if alza_pct > 90:  # 🚀 CORRECCIÓN: Umbral aumentado a 90%
                self._add_debug_log("warning", f"🚨 SESGO ALCISTA DETECTADO: {alza_pct:.1f}%")
            elif baja_pct > 90:  # 🚀 CORRECCIÓN: Umbral aumentado a 90%
                self._add_debug_log("warning", f"🚨 SESGO BAJISTA DETECTADO: {baja_pct:.1f}%")
    
    def _get_base_prediction(self, analysis):
        """Predicción base cuando el análisis no está completo"""
        return {
            "direction": "LATERAL",
            "confidence": 50,
            "tick_count": self.analyzer.tick_count,
            "current_price": self.analyzer.current_candle_close or 0.0,
            "reasons": ["Análisis de vela en curso"],
            "timestamp": now_iso(),
            "status": "ANALYZING",
            "method": "TRADICIONAL",
            "debug_info": list(self.debug_logs)[-2:] if self.debug_logs else []
        }
    
    def _get_traditional_prediction(self, analysis):
        """Predicción tradicional basada en análisis técnico - VERSIÓN DEFINITIVA"""
        try:
            self._add_debug_log("info", "Calculando predicción tradicional")
            
            # Análisis de fases
            phase_analysis = analysis.get('phase_analysis', {})
            phase_trends = self._analyze_phase_trends(phase_analysis)
            
            # Análisis de momentum
            general_analysis = analysis.get('general_analysis', {})
            momentum_analysis = self._analyze_momentum_pressure(general_analysis)
            
            # Análisis de segmentos
            segment_analysis = analysis.get('segment_analysis', {})
            segment_prediction = self._analyze_segment_behavior(segment_analysis)
            
            # Análisis de patrón de vela
            candle_stats = analysis.get('candle_stats', {})
            candle_pattern = self._analyze_candle_pattern(candle_stats, general_analysis)
            
            # DETECCIÓN MEJORADA DE SESGO - VERSIÓN DEFINITIVA
            self._detect_prediction_bias(phase_trends, momentum_analysis, segment_prediction, candle_pattern)
            
            # Combinar predicciones
            final_prediction = self._combine_traditional_predictions(
                phase_trends, momentum_analysis, segment_prediction, candle_pattern
            )
            
            reasons = self._generate_prediction_reasons(
                phase_trends, momentum_analysis, segment_prediction, candle_pattern
            )
            
            self._add_debug_log("info", 
                f"Predicción tradicional final: {final_prediction['direction']} {final_prediction['confidence']}%")
            
            return {
                "direction": final_prediction['direction'],
                "confidence": final_prediction['confidence'],
                "tick_count": self.analyzer.tick_count,
                "current_price": analysis['current_price'],
                "reasons": reasons,
                "timestamp": now_iso(),
                "status": "PREDICTION_READY",
                "method": "TRADICIONAL"
            }
            
        except Exception as e:
            self._add_debug_log("error", f"Error en predicción tradicional: {e}")
            return self._get_base_prediction(analysis)
    
    def _detect_prediction_bias(self, phase_trends, momentum_analysis, segment_prediction, candle_pattern):
        """Detecta y corrige sesgos en las predicciones - VERSIÓN DEFINITIVA"""
        try:
            # Contar direcciones
            directions = [
                phase_trends['direction'],
                momentum_analysis['momentum_direction'],
                segment_prediction['direction'],
                candle_pattern['direction']
            ]
            
            baja_count = directions.count('BAJA')
            alza_count = directions.count('ALZA')
            
            # CORRECCIÓN MEJORADA DE SESGO - Solo aplicar si hay sesgo extremo
            if baja_count >= 3 and alza_count == 0:
                self._add_debug_log("warning", "🚨 SESGO BAJISTA DETECTADO en análisis tradicional")
                # Aplicar corrección suave: reducir confianza en dirección bajista
                if phase_trends['direction'] == 'BAJA':
                    phase_trends['strength'] *= 0.95  # 🚀 CORRECCIÓN: Penalización mínima
            elif alza_count >= 3 and baja_count == 0:
                self._add_debug_log("warning", "🚨 SESGO ALCISTA DETECTADO en análisis tradicional")
                # Aplicar corrección suave: reducir confianza en dirección alcista
                if phase_trends['direction'] == 'ALZA':
                    phase_trends['strength'] *= 0.95  # 🚀 CORRECCIÓN: Penalización mínima
                    
        except Exception as e:
            self._add_debug_log("error", f"Error detectando sesgo: {e}")
    
    def _analyze_phase_trends(self, phase_analysis):
        """Analiza las tendencias por fases de la vela"""
        trends = []
        strengths = []
        
        for phase, analysis in phase_analysis.items():
            if analysis.get('trend_direction'):
                original_direction = analysis['trend_direction']
                if original_direction == 'ALCISTA':
                    mapped_direction = 'ALZA'
                elif original_direction == 'BAJISTA':
                    mapped_direction = 'BAJA'
                else:
                    mapped_direction = 'LATERAL'
                
                if analysis.get('trend_strength', 0) > 0.5:
                    trends.append(mapped_direction)
                    strengths.append(analysis['trend_strength'])
        
        if not trends:
            return {'direction': 'LATERAL', 'strength': 0, 'consistency': False}
        
        # Ponderar más las fases finales
        weights = [0.1, 0.2, 0.3, 0.4]
        weighted_trends = {}
        
        for i, trend in enumerate(trends):
            weight = weights[i] if i < len(weights) else 0.1
            if trend in weighted_trends:
                weighted_trends[trend] += weight
            else:
                weighted_trends[trend] = weight
        
        dominant_trend = max(weighted_trends, key=weighted_trends.get)
        avg_strength = np.mean(strengths) if strengths else 0
        
        return {
            'direction': dominant_trend,
            'strength': min(100, avg_strength * 10),
            'consistency': len(set(trends)) == 1
        }
    
    def _analyze_momentum_pressure(self, general_analysis):
        """Analiza momentum y presión de la vela"""
        momentum = general_analysis.get('current_momentum', 0)
        pressure = general_analysis.get('pressure_balance', 0.5)
        
        direction = "ALZA" if momentum > 1.0 else "BAJA" if momentum < -1.0 else "LATERAL"
        strength = min(100, abs(momentum) * 20)
        
        pressure_signal = "ALZA" if pressure > 0.6 else "BAJA" if pressure < 0.4 else "LATERAL"
        pressure_strength = abs(pressure - 0.5) * 200
        
        return {
            'momentum_direction': direction,
            'momentum_strength': strength,
            'pressure_direction': pressure_signal,
            'pressure_strength': pressure_strength,
            'alignment': direction == pressure_signal
        }
    
    def _analyze_segment_behavior(self, segment_analysis):
        """Analiza el comportamiento por segmentos de tiempo"""
        segments = list(segment_analysis.keys())
        directions = []
        
        for segment in segments:
            if segment_analysis[segment].get('direction'):
                original_direction = segment_analysis[segment]['direction']
                if original_direction == 'ALCISTA':
                    mapped_direction = 'ALZA'
                elif original_direction == 'BAJISTA':
                    mapped_direction = 'BAJA'
                else:
                    mapped_direction = 'LATERAL'
                directions.append(mapped_direction)
        
        if not directions:
            return {'direction': 'LATERAL', 'confidence': 50, 'recent_alignment': False}
        
        # Los segmentos finales tienen más peso
        recent_directions = directions[-2:] if len(directions) >= 2 else directions
        alcista_count = recent_directions.count('ALZA')
        bajista_count = recent_directions.count('BAJA')
        
        if alcista_count > bajista_count:
            direction = "ALZA"
            confidence = (alcista_count / len(recent_directions)) * 80
        elif bajista_count > alcista_count:
            direction = "BAJA"
            confidence = (bajista_count / len(recent_directions)) * 80
        else:
            direction = "LATERAL"
            confidence = 50
            
        return {
            'direction': direction,
            'confidence': confidence,
            'recent_alignment': alcista_count == len(recent_directions) or bajista_count == len(recent_directions)
        }
    
    def _analyze_candle_pattern(self, candle_stats, general_analysis):
        """Analiza el patrón de la vela actual"""
        direction = candle_stats.get('direction', 'LATERAL')
        body_size = candle_stats.get('body_size', 0)
        range_size = candle_stats.get('range', 0)
        
        # Mapear dirección de la vela
        if direction == 'ALCISTA':
            mapped_direction = 'ALZA'
        elif direction == 'BAJISTA':
            mapped_direction = 'BAJA'
        else:
            mapped_direction = 'LATERAL'
        
        # Vela con cuerpo grande → continuación probable
        if body_size > range_size * 0.7:
            pattern_strength = 80
            pattern_type = "FUERTE"
        elif body_size > range_size * 0.4:
            pattern_strength = 65
            pattern_type = "MODERADO"
        else:
            pattern_strength = 50
            pattern_type = "LIGERO"
        
        consistency = general_analysis.get('consistency_score', 50)
        
        return {
            'direction': mapped_direction,
            'strength': pattern_strength,
            'type': pattern_type,
            'consistency': consistency,
            'continuation_bias': pattern_strength > 60
        }
    
    def _combine_traditional_predictions(self, phase_trends, momentum_analysis, segment_prediction, candle_pattern):
        """Combina todas las predicciones tradicionales en una final - VERSIÓN ANTI-SESGO"""
        predictions = [
            (phase_trends['direction'], phase_trends['strength'], 0.30),
            (momentum_analysis['momentum_direction'], momentum_analysis['momentum_strength'], 0.25),
            (segment_prediction['direction'], segment_prediction['confidence'], 0.25),
            (candle_pattern['direction'], candle_pattern['strength'], 0.20)
        ]
        
        direction_scores = {"ALZA": 0, "BAJA": 0, "LATERAL": 0}
        total_confidence = 0
        
        for direction, confidence, weight in predictions:
            direction_scores[direction] += confidence * weight
            total_confidence += confidence * weight
        
        final_direction = max(direction_scores, key=direction_scores.get)
        
        # 🚀 CÁLCULO MEJORADO SIN PENALIZACIONES EXCESIVAS
        base_confidence = min(90, int(total_confidence))
        
        # Bonus por consistencia (más conservador)
        consistency_bonus = 0
        if phase_trends.get('consistency', False):
            consistency_bonus += 5  # Reducido
        if momentum_analysis.get('alignment', False):
            consistency_bonus += 4  # Reducido
        if segment_prediction.get('recent_alignment', False):
            consistency_bonus += 3  # Reducido
        
        # 🚀 CORRECCIÓN DEFINITIVA DE SESGO - MÁS CONSERVADORA
        bias_correction = 0
        total_predictions = self.performance_stats['total_predictions']
        
        if total_predictions > 15:  # 🚀 Solo después de suficientes predicciones
            bias_stats = self.performance_stats['bias_tracking']
            total_recent = sum(bias_stats.values())
            
            if total_recent > 0:
                recent_alza_pct = bias_stats['alza_count'] / total_recent
                recent_baja_pct = bias_stats['baja_count'] / total_recent
                
                # 🚀 CORRECCIÓN: Solo aplicar corrección si sesgo > 90% Y dirección coincide
                if final_direction == 'ALZA' and recent_alza_pct > 0.90:
                    bias_correction = -3  # 🚀 Mínima penalización
                    self._add_debug_log("warning", f"Corrección mínima por sesgo alcista: {recent_alza_pct:.1%}")
                elif final_direction == 'BAJA' and recent_baja_pct > 0.90:
                    bias_correction = -3  # 🚀 Mínima penalización  
                    self._add_debug_log("warning", f"Corrección mínima por sesgo bajista: {recent_baja_pct:.1%}")
                else:
                    # 🚀 BONUS por diversificación si no hay sesgo extremo
                    if final_direction == 'ALZA' and recent_alza_pct < 0.5:
                        bias_correction = +2
                    elif final_direction == 'BAJA' and recent_baja_pct < 0.5:
                        bias_correction = +2
        
        final_confidence = min(95, max(45, base_confidence + consistency_bonus + bias_correction))
        
        # 🚀 FORZAR DIVERSIFICACIÓN si confianza es muy baja
        if final_confidence < 50 and total_predictions > 10:
            alternative_directions = [d for d in direction_scores.keys() if d != final_direction]
            if alternative_directions:
                # Elegir la segunda mejor dirección
                final_direction = max(alternative_directions, key=lambda d: direction_scores[d])
                final_confidence = 55  # Confianza base para dirección alternativa
                self._add_debug_log("info", f"Forzando diversificación: {final_direction}")
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'base_confidence': base_confidence,
            'consistency_bonus': consistency_bonus,
            'bias_correction': bias_correction
        }
    
    def _combine_predictions(self, trad_direction, ml_direction, trad_confidence, ml_confidence):
        """Combina predicciones tradicionales y de ML"""
        # Si las direcciones coinciden, usar esa dirección
        if trad_direction == ml_direction:
            self._add_debug_log("info", "Coincidencia ML-Tradicional")
            return trad_direction
        
        # Si ML tiene alta confianza y tradicional baja, favorecer ML
        if ml_confidence > 70 and trad_confidence < 60:
            self._add_debug_log("info", "Favoreciendo ML (alta confianza)")
            return ml_direction
        
        # Si tradicional tiene alta confianza y ML baja, favorecer tradicional
        if trad_confidence > 70 and ml_confidence < 60:
            self._add_debug_log("info", "Favoreciendo Tradicional (alta confianza)")
            return trad_direction
        
        # En caso de empate, usar tradicional (más conservador)
        self._add_debug_log("info", "Usando Tradicional (conservador)")
        return trad_direction
    
    def _generate_prediction_reasons(self, phase_trends, momentum_analysis, segment_prediction, candle_pattern):
        """Genera razones detalladas para la predicción"""
        reasons = []
        
        # Razones de fase
        if phase_trends['strength'] > 60:
            reasons.append(f"Tendencia {phase_trends['direction']} en fases ({phase_trends['strength']:.0f}%)")
        
        # Razones de momentum
        if momentum_analysis['momentum_strength'] > 50:
            reasons.append(f"Momentum {momentum_analysis['momentum_direction']} fuerte")
        
        if momentum_analysis['pressure_strength'] > 60:
            reasons.append(f"Presión {momentum_analysis['pressure_direction']} dominante")
        
        # Razones de segmentos
        if segment_prediction['recent_alignment']:
            reasons.append("Alineación consistente en segmentos finales")
        
        # Razones de patrón
        if candle_pattern['type'] != "LIGERO":
            reasons.append(f"Patrón {candle_pattern['type']} {candle_pattern['direction']}")
        
        # Razón de consistencia
        if len([r for r in reasons if 'consist' in r.lower() or 'aline' in r.lower()]) >= 2:
            reasons.append("Alta consistencia en señales")
        
        if not reasons:
            reasons.append("Señales equilibradas - análisis conservador")
        
        return reasons
    
    def validate_prediction(self, actual_direction: str):
        """Valida la predicción contra el resultado real"""
        if not self.prediction_history:
            return None
            
        last_prediction = self.prediction_history[-1]
        predicted_direction = last_prediction['direction']
        
        # Mapear direcciones de forma consistente
        if actual_direction == 'ALCISTA':
            actual_mapped = 'ALZA'
        elif actual_direction == 'BAJISTA':
            actual_mapped = 'BAJA'
        else:
            actual_mapped = 'LATERAL'
        
        is_correct = (predicted_direction == actual_mapped and 
                     predicted_direction != "LATERAL" and 
                     actual_mapped != "LATERAL")
        
        # REGISTRO DETALLADO DE VALIDACIÓN
        validation_entry = {
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted_direction,
            'actual': actual_mapped,
            'correct': is_correct,
            'confidence': last_prediction.get('confidence', 0),
            'method': last_prediction.get('method', 'TRADICIONAL')
        }
        self.performance_stats['prediction_history'].append(validation_entry)
        
        # Aprendizaje automático MEJORADO - VERSIÓN DEFINITIVA
        if (self.auto_learning_active and 
            self.last_prediction_features is not None and 
            actual_direction is not None):
            
            # Siempre agregar muestra para aprendizaje
            self.learning_system.add_training_sample(
                self.last_prediction_features,
                actual_mapped,
                last_prediction.get('confidence', 50)
            )
            
            # Entrenamiento incremental más frecuente
            current_samples = len(self.learning_system.training_data)
            min_samples = self.learning_system.min_training_samples
            
            if current_samples >= min_samples and current_samples % 5 == 0:  # 🚀 Cada 5 muestras nuevas
                self._add_debug_log("info", f"Entrenamiento incremental ({current_samples} muestras)")
                self.learning_system.train_model()
        
        if is_correct:
            self.performance_stats['correct_predictions'] += 1
            self.performance_stats['current_streak'] += 1
            self.performance_stats['best_streak'] = max(
                self.performance_stats['best_streak'], 
                self.performance_stats['current_streak']
            )
            self._add_debug_log("success", 
                f"PREDICCIÓN CORRECTA: {predicted_direction} vs {actual_mapped}")
        else:
            self.performance_stats['current_streak'] = 0
            self._add_debug_log("error", 
                f"PREDICCIÓN INCORRECTA: {predicted_direction} vs {actual_mapped}")
        
        return {
            "predicted": predicted_direction,
            "actual": actual_mapped,
            "correct": is_correct,
            "current_streak": self.performance_stats['current_streak'],
            "validation_entry": validation_entry
        }
    
    def get_performance_stats(self):
        """Obtiene estadísticas de rendimiento - VERSIÓN DEFINITIVA"""
        try:
            # Calcular accuracy de forma segura
            total_predictions = self.performance_stats['total_predictions']
            correct_predictions = self.performance_stats['correct_predictions']
            
            accuracy = 0
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
            
            # Calcular sesgo de forma segura
            bias_stats = self.performance_stats['bias_tracking']
            total_bias = sum(bias_stats.values())
            
            bias_alza_pct = 0
            bias_baja_pct = 0
            if total_bias > 0:
                bias_alza_pct = (bias_stats['alza_count'] / total_bias) * 100
                bias_baja_pct = (bias_stats['baja_count'] / total_bias) * 100
            
            debug_info = [
                {
                    'level': 'info',
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'message': f"Sesgo: ALZA {bias_alza_pct:.1f}%, BAJA {bias_baja_pct:.1f}%"
                }
            ]
            
            # Retornar estructura completa y consistente
            return {
                "accuracy": round(accuracy, 1),
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "current_streak": self.performance_stats['current_streak'],
                "best_streak": self.performance_stats['best_streak'],
                "today_signals": self.performance_stats['today_signals'],
                "total_signals": self.performance_stats['total_signals'],
                "bias_tracking": bias_stats,
                "debug_info": debug_info + list(self.debug_logs)[-2:]
            }
            
        except Exception as e:
            self._add_debug_log("error", f"Error en get_performance_stats: {e}")
            # Retornar estructura por defecto en caso de error
            return {
                "accuracy": 0,
                "total_predictions": 0,
                "correct_predictions": 0,
                "current_streak": 0,
                "best_streak": 0,
                "today_signals": 0,
                "total_signals": 0,
                "bias_tracking": {'alza_count': 1, 'baja_count': 1, 'lateral_count': 1},
                "debug_info": [{'level': 'error', 'timestamp': datetime.now().strftime("%H:%M:%S"), 'message': f"Error: {e}"}]
            }
    
    def get_enhanced_performance_stats(self):
        """Obtiene estadísticas extendidas incluyendo aprendizaje"""
        base_stats = self.get_performance_stats()
        learning_stats = self.learning_system.get_learning_stats()
        
        return {
            **base_stats,
            'learning_system': learning_stats,
            'auto_learning_active': self.auto_learning_active,
            'market_context': self.market_context
        }
    
    def reset(self):
        """Reinicia el predictor manteniendo el aprendizaje"""
        self.analyzer.reset()
        # No reiniciamos el sistema de aprendizaje

# ------------------ DASHBOARD MEJORADO CON ESTADÍSTICAS DE APRENDIZAJE ------------------
class EnhancedResponsiveDashboard:
    def __init__(self):
        self.dashboard_data = {
            "current_prediction": {
                "direction": "N/A",
                "confidence": 0,
                "arrow": "⏳",
                "color": "gray",
                "signal_strength": "NORMAL",
                "timestamp": "00:00:00",
                "method": "TRADICIONAL",
                "status": "ANALIZANDO",
                "debug_info": []
            },
            "current_candle": {
                "progress": 0,
                "time_remaining": 60,
                "price": 0.0,
                "ticks_processed": 0,
                "is_last_5_seconds": False,
                "current_phase": "INICIAL"
            },
            "metrics": {
                "density": 0,
                "velocity": 0,
                "acceleration": 0,
                "phase": "INICIAL",
                "signal_count": 0
            },
            "performance": {
                "today_accuracy": 0,
                "today_profit": 0,
                "total_signals": 0,
                "win_streak": 0,
                "current_streak": 0,
                "debug_info": []
            },
            "system_status": {
                "iq_connection": "DISCONNECTED",
                "ai_status": "INITIALIZING",
                "metronome_sync": "UNSYNCED",
                "last_update": "N/A",
                "debug_info": []
            },
            "learning_stats": {
                "model_accuracy": 0,
                "training_samples": 0,
                "learning_status": "ACTIVO",
                "last_training": "N/A",
                "top_features": [],
                "debug_info": []
            },
            "visual_effects": {
                "pulse_animation": False,
                "flash_signal": False,
                "countdown_active": False,
                "prediction_change": False
            }
        }
        self.last_prediction = None
        self.prediction_history = []
        
    def update_prediction(self, direction: str, confidence: int, signal_strength: str = "NORMAL", 
                         method: str = "TRADICIONAL", debug_info: list = None, status: str = "ANALIZANDO"):
        arrow, color = self._get_direction_arrow(direction, confidence)
        
        prediction_change = (
            self.last_prediction and 
            self.last_prediction.get('direction') != direction
        )
        
        self.dashboard_data["current_prediction"] = {
            "direction": direction,
            "confidence": confidence,
            "arrow": arrow,
            "color": color,
            "signal_strength": signal_strength,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "method": method,
            "status": status,
            "debug_info": debug_info or []
        }
        
        self.dashboard_data["visual_effects"]["prediction_change"] = prediction_change
        self.dashboard_data["visual_effects"]["flash_signal"] = True
        
        self.prediction_history.append({
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "method": method
        })
        
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            
        self.last_prediction = self.dashboard_data["current_prediction"].copy()
        
        # CORRECCIÓN: Manejo correcto de corrutinas
        try:
            # Obtener el loop de eventos actual
            loop = asyncio.get_event_loop()
            
            # Programar reset de efectos visuales
            if loop.is_running():
                if self.dashboard_data["visual_effects"]["prediction_change"]:
                    asyncio.create_task(self._reset_visual_effect("prediction_change", 2))
                if self.dashboard_data["visual_effects"]["flash_signal"]:
                    asyncio.create_task(self._reset_visual_effect("flash_signal", 1))
        except Exception as e:
            logging.debug(f"🔧 Error programando tareas asyncio: {e}")

    async def _reset_visual_effect(self, effect: str, delay: float):
        try:
            await asyncio.sleep(delay)
            self.dashboard_data["visual_effects"][effect] = False
        except Exception as e:
            logging.debug(f"🔧 Error resetando efecto visual: {e}")

    def update_candle_progress(self, metronome: IQOptionMetronome, current_price: float, ticks_processed: int):
        remaining_time = metronome.get_remaining_time()
        progress = ((60 - remaining_time) / 60) * 100
        is_last_5 = metronome.is_last_5_seconds()
        
        # Determinar fase actual
        if remaining_time > 45:
            current_phase = "FASE 1 (0-15s)"
        elif remaining_time > 25:
            current_phase = "FASE 2 (15-35s)"
        elif remaining_time > 5:
            current_phase = "FASE 3 (35-55s)"
        else:
            current_phase = "PREDICCIÓN (55-60s)"
        
        if is_last_5 and not self.dashboard_data["current_candle"]["is_last_5_seconds"]:
            self.dashboard_data["visual_effects"]["pulse_animation"] = True
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._reset_visual_effect("pulse_animation", 5))
            except Exception as e:
                logging.debug(f"🔧 Error programando tarea pulse: {e}")
        
        self.dashboard_data["current_candle"] = {
            "progress": progress,
            "time_remaining": remaining_time,
            "price": current_price,
            "ticks_processed": ticks_processed,
            "is_last_5_seconds": is_last_5,
            "current_phase": current_phase
        }
        
        self.dashboard_data["visual_effects"]["countdown_active"] = is_last_5

    def update_metrics(self, density: float, velocity: float, acceleration: float, phase: str, signal_count: int = 0):
        self.dashboard_data["metrics"] = {
            "density": density,
            "velocity": velocity,
            "acceleration": acceleration,
            "phase": phase,
            "signal_count": signal_count
        }

    def update_performance(self, accuracy: float, profit: float, signals: int, streak: int, 
                          current_streak: int = 0, debug_info: list = None):
        self.dashboard_data["performance"] = {
            "today_accuracy": accuracy,
            "today_profit": profit,
            "total_signals": signals,
            "win_streak": streak,
            "current_streak": current_streak,
            "debug_info": debug_info or []
        }

    def update_learning_stats(self, accuracy: float, samples: int, status: str, last_training: str, 
                             top_features: list, debug_info: list = None):
        self.dashboard_data["learning_stats"] = {
            "model_accuracy": accuracy,
            "training_samples": samples,
            "learning_status": status,
            "last_training": last_training,
            "top_features": top_features,
            "debug_info": debug_info or []
        }

    def update_system_status(self, iq_status: str, ai_status: str, metronome_status: str = "UNSYNCED", 
                            debug_info: list = None):
        self.dashboard_data["system_status"] = {
            "iq_connection": iq_status,
            "ai_status": ai_status,
            "metronome_sync": metronome_status,
            "last_update": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "debug_info": debug_info or []
        }

    def _get_direction_arrow(self, direction: str, confidence: int):
        if direction == "ALZA":
            if confidence >= 90:
                return "↗️", "green-bright"
            elif confidence >= 80:
                return "↗️", "green"
            else:
                return "↗️", "green-light"
        elif direction == "BAJA":
            if confidence >= 90:
                return "↘️", "red-bright"
            elif confidence >= 80:
                return "↘️", "red"
            else:
                return "↘️", "red-light"
        else:
            return "═", "yellow"

# ------------------ WEBSOCKET MANAGER CORREGIDO ------------------
class AdvancedConnectionManager:
    def __init__(self):
        self.active_connections = set()
        self.dashboard = EnhancedResponsiveDashboard()
        self.metronome = IQOptionMetronome()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logging.info(f"✅ Cliente WebSocket conectado. Total: {len(self.active_connections)}")
        
        await self.send_dashboard_update(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logging.info(f"❌ Cliente WebSocket desconectado. Total: {len(self.active_connections)}")

    async def send_dashboard_update(self, websocket: WebSocket):
        try:
            await websocket.send_json({
                "type": "dashboard_update",
                "data": self.dashboard.dashboard_data
            })
        except Exception as e:
            logging.error(f"Error enviando actualización: {e}")
            self.disconnect(websocket)

    async def broadcast_dashboard_update(self):
        if not self.active_connections:
            return
            
        message = {
            "type": "dashboard_update", 
            "data": self.dashboard.dashboard_data
        }
        
        # CORRECCIÓN: Crear copia del conjunto para evitar modificación durante iteración
        connections = list(self.active_connections)
        disconnected = []
        
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Error broadcast a cliente: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

# ------------------ SISTEMA PRINCIPAL MEJORADO ------------------
# Instancias globales
iq_connector = RealIQOptionConnector(IQ_EMAIL, IQ_PASSWORD, PAR)
predictor = EnhancedNextCandlePredictor()
dashboard_manager = AdvancedConnectionManager()

# Variables globales
_last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
_prediction_made_this_candle = False
_last_price = None

# ------------------ FASTAPI APP ------------------
app = FastAPI(
    title="Delowyss Trading AI V6.3 MEJORADO - Análisis Completo de Vela + Autoaprendizaje MEJORADO",
    description="Sistema de IA con análisis completo de vela actual para predecir siguiente vela con autoaprendizaje MEJORADO - VERSIÓN DEFINITIVA",
    version="6.3.0"
)

# SOLUCIÓN DEFINITIVA: Configuración CORS para Render.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://delowyss-trading.onrender.com",
        "http://delowyss-trading.onrender.com", 
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:10000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configuración específica para WebSockets en Render
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "https://delowyss-trading.onrender.com"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ------------------ CONFIGURACIÓN RUTAS MEJORADAS ------------------
def setup_enhanced_routes(app: FastAPI, manager: AdvancedConnectionManager, iq_connector):
    @app.get("/", response_class=HTMLResponse)
    async def get_enhanced_dashboard():
        return HTML_RESPONSIVE

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.get("/api/prediction")
    async def get_prediction():
        analysis = predictor.analyzer.get_candle_analysis()
        if analysis.get('status') == 'COMPLETE_ANALYSIS':
            prediction = predictor.predict_next_candle()
            return prediction
        return {"status": "ANALYZING", "message": "Analizando vela actual..."}

    @app.get("/api/performance")
    async def get_performance():
        stats = predictor.get_performance_stats()
        return {
            "performance": stats,
            "system_status": "CANDLE_ANALYSIS_ACTIVE",
            "timestamp": now_iso()
        }

    @app.get("/api/learning-stats")
    async def get_learning_stats():
        stats = predictor.get_enhanced_performance_stats()
        return {
            "learning_system": stats.get('learning_system', {}),
            "auto_learning": stats.get('auto_learning_active', False),
            "timestamp": now_iso()
        }

    @app.post("/api/retrain-model")
    async def retrain_model():
        success = predictor.learning_system.train_model()
        return {
            "success": success,
            "message": "Modelo reentrenado" if success else "Error en reentrenamiento",
            "timestamp": now_iso()
        }

    @app.get("/api/analysis")
    async def get_analysis():
        analysis = predictor.analyzer.get_candle_analysis()
        return {
            "analysis": analysis,
            "timestamp": now_iso()
        }

    @app.get("/api/status")
    async def get_status():
        return {
            "status": "operational_mejorado_v6.3",
            "version": "6.3.0",
            "pair": PAR,
            "timeframe": "1min",
            "iq_connected": iq_connector.connected,
            "current_price": iq_connector.current_price,
            "prediction_window": f"{PREDICTION_WINDOW}s",
            "auto_learning": predictor.auto_learning_active,
            "learning_samples": len(predictor.learning_system.training_data),
            "debug_enabled": True,
            "timestamp": now_iso()
        }

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(enhanced_continuous_dashboard_updates(manager, iq_connector))

async def enhanced_continuous_dashboard_updates(manager: AdvancedConnectionManager, iq_connector):
    while True:
        try:
            if time.time() - manager.metronome.last_sync_time > 30:
                try:
                    await manager.metronome.sync_with_iqoption(iq_connector)
                    manager.dashboard.update_system_status(
                        "CONNECTED" if iq_connector.connected else "DISCONNECTED",
                        "OPERATIONAL_MEJORADO_V6.3",
                        "SYNCED" if manager.metronome.last_sync_time > 0 else "UNSYNCED"
                    )
                except Exception as e:
                    logging.warning(f"⚠️ Error sincronizando metrónomo: {e}")
            
            current_price = iq_connector.current_price or 0.0
            ticks_processed = iq_connector.tick_count
            
            manager.dashboard.update_candle_progress(
                manager.metronome, 
                current_price, 
                ticks_processed
            )
            
            # Actualizar estadísticas de aprendizaje más frecuentemente
            current_time = time.time()
            if hasattr(enhanced_continuous_dashboard_updates, 'last_learning_update'):
                if current_time - enhanced_continuous_dashboard_updates.last_learning_update > 8:  # 8 segundos
                    stats = predictor.get_performance_stats()
                    learning_stats = stats.get('learning_system', {})
                    
                    # Actualizar dashboard con estadísticas de aprendizaje
                    top_features = learning_stats.get('top_features', [])
                    
                    manager.dashboard.update_learning_stats(
                        learning_stats.get('model_accuracy', 0),
                        learning_stats.get('training_samples', 0),
                        "ACTIVO",
                        learning_stats.get('last_training', 'N/A'),
                        top_features,
                        learning_stats.get('debug_info', [])
                    )
                    
                    # Actualizar performance con debug info
                    manager.dashboard.update_performance(
                        stats.get('accuracy', 0),
                        0,
                        stats.get('total_signals', 0),
                        stats.get('best_streak', 0),
                        stats.get('current_streak', 0),
                        stats.get('debug_info', [])
                    )
                    
                    enhanced_continuous_dashboard_updates.last_learning_update = current_time
            else:
                enhanced_continuous_dashboard_updates.last_learning_update = current_time
            
            await manager.broadcast_dashboard_update()
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error en actualización mejorada: {e}")
            await asyncio.sleep(1)

# Configurar rutas
setup_enhanced_routes(app, dashboard_manager, iq_connector)

# ------------------ INICIALIZACIÓN MEJORADA - VERSIÓN DEFINITIVA ------------------
def start_enhanced_system():
    try:
        logging.info("🔧 INICIANDO SISTEMA V6.3 MEJORADO DEFINITIVO - DEBUG ACTIVADO")
        logging.info("🎯 SISTEMA DE PREDICCIÓN HÍBRIDO MEJORADO (TRADICIONAL + ML) - VERSIÓN DEFINITIVA")
        logging.info("🐛 CORRECCIONES DEFINITIVAS: ML estable, sesgo controlado, entrenamiento robusto")
        
        # Verificar sistema de aprendizaje MEJORADO
        if hasattr(predictor, 'learning_system'):
            learning_status = "ACTIVO"
            model_status = "CARGADO" if predictor.learning_system.model is not None else "PREPARADO"
            logging.info(f"🧠 SISTEMA DE AUTOAPRENDIZAJE MEJORADO: {learning_status} - MODELO: {model_status}")
            
            if predictor.learning_system.model is not None:
                logging.info(f"📊 Accuracy del modelo: {predictor.learning_system.model_accuracy:.3f}")
            else:
                logging.info("📊 Sistema preparado - ML se activará con datos suficientes")
        
        # ✅ INICIAR CONEXIÓN IQ OPTION
        logging.info("🔄 Iniciando conexión a IQ Option...")
        connection_result = iq_connector.connect()
        logging.info(f"🔧 Resultado conexión IQ Option: {connection_result}")
        
        if connection_result:
            logging.info("✅ Conexión IQ Option exitosa al inicio")
            dashboard_manager.dashboard.update_system_status("CONNECTED", "OPERATIONAL_MEJORADO_V6.3", "SYNCED")
        else:
            logging.error("❌ Conexión IQ Option falló al inicio")
            dashboard_manager.dashboard.update_system_status("DISCONNECTED", "ERROR", "SYNCED")
        
        # ✅ INICIAR THREAD DE ANÁLISIS
        logging.info("🔧 Iniciando thread de análisis de vela...")
        trading_thread = threading.Thread(target=premium_candle_analysis_loop, daemon=True)
        trading_thread.start()
        logging.info("🔧 Thread de análisis de vela iniciado")
        
        logging.info(f"⭐ DELOWYSS AI V6.3 MEJORADA DEFINITIVA INICIADA - DEBUG ACTIVADO")
        logging.info("🎯 PREDICCIÓN A 5s - SISTEMA HÍBRIDO TRADICIONAL + ML MEJORADO")
        logging.info("🐛 SEGUIMIENTO DE SESGO: Controlado con correcciones definitivas")
        logging.info("📊 VALIDACIÓN: Sistema de tracking robusto implementado")
        logging.info("🌐 DASHBOARD DISPONIBLE EN: http://0.0.0.0:10000")
        
        time.sleep(2)
        logging.info(f"🔧 Threads activos: {threading.active_count()}")
        
    except Exception as e:
        logging.error(f"❌ Error iniciando sistema mejorado: {e}")
        import traceback
        logging.error(f"❌ Traceback: {traceback.format_exc()}")

# ------------------ LOOP PRINCIPAL MEJORADO ------------------
def premium_candle_analysis_loop():
    global _last_candle_start, _prediction_made_this_candle, _last_price
    
    logging.info(f"🚀 LOOP DE ANÁLISIS DE VELA COMPLETA MEJORADO INICIADO")
    
    # Variables locales para evitar problemas de scope
    _last_candle_start = int(time.time() // TIMEFRAME * TIMEFRAME)
    _prediction_made_this_candle = False
    _last_price = None
    
    while True:
        try:
            if not iq_connector.connected:
                logging.warning("🔌 IQ Option desconectado, intentando reconectar...")
                if iq_connector.connect():
                    logging.info("✅ Reconexión exitosa a IQ Option")
                    dashboard_manager.dashboard.update_system_status("CONNECTED", "OPERATIONAL_MEJORADO_V6.3", "SYNCED")
                else:
                    logging.error("❌ No se pudo reconectar a IQ Option")
                    dashboard_manager.dashboard.update_system_status("DISCONNECTED", "ERROR", "SYNCED")
                    time.sleep(10)
                continue
            
            current_time = time.time()
            current_candle_start = int(current_time // TIMEFRAME * TIMEFRAME)
            seconds_remaining = iq_connector.get_remaining_time()
            
            # Obtener precio actual
            price = iq_connector.get_realtime_price()
            if price and price > 0:
                _last_price = price
                
                # Procesar tick para análisis
                predictor.process_tick(price, seconds_remaining)
                
                # Hacer predicción en los últimos 5 segundos si no se ha hecho
                if (dashboard_manager.metronome.is_prediction_time() and 
                    not _prediction_made_this_candle and
                    predictor.analyzer.is_ready_for_prediction()):
                    
                    logging.info(f"🎯 HACIENDO PREDICCIÓN MEJORADA A {seconds_remaining:.1f}s")
                    
                    prediction = predictor.predict_next_candle()
                    
                    # Actualizar dashboard
                    dashboard_manager.dashboard.update_prediction(
                        prediction['direction'],
                        prediction['confidence'],
                        method=prediction.get('method', 'TRADICIONAL'),
                        debug_info=prediction.get('debug_info', []),
                        status=prediction.get('status', 'PREDICTION_READY')
                    )
                    
                    # Actualizar métricas de performance
                    stats = predictor.get_performance_stats()
                    dashboard_manager.dashboard.update_performance(
                        stats.get('accuracy', 0),
                        0,
                        stats.get('total_signals', 0),
                        stats.get('best_streak', 0),
                        stats.get('current_streak', 0),
                        stats.get('debug_info', [])
                    )
                    
                    _prediction_made_this_candle = True
                    logging.info(f"✅ PREDICCIÓN MEJORADA: {prediction['direction']} {prediction['confidence']}% - Método: {prediction.get('method', 'TRADICIONAL')}")
            
            # Detectar nueva vela MEJORADO
            if current_candle_start > _last_candle_start:
                if _last_price is not None:
                    # Determinar dirección real de la vela que acaba de cerrar
                    if (predictor.analyzer.current_candle_close is not None and 
                        predictor.analyzer.current_candle_open is not None):
                        
                        price_change = predictor.analyzer.current_candle_close - predictor.analyzer.current_candle_open
                        actual_direction = "ALCISTA" if price_change > 0.00001 else "BAJISTA" if price_change < -0.00001 else "LATERAL"
                        
                        # VALIDAR PREDICCIÓN CON DEBUG MEJORADO
                        validation = predictor.validate_prediction(actual_direction)
                        if validation:
                            result_icon = '✅' if validation['correct'] else '❌'
                            logging.info(f"📊 VALIDACIÓN MEJORADA: Predicho {validation['predicted']} vs Real {validation['actual']} - {result_icon}")
                
                # REINICIO MEJORADO
                _last_candle_start = current_candle_start
                _prediction_made_this_candle = False
                predictor.analyzer.reset()
                logging.info("🕯️ NUEVA VELA - Análisis completo reiniciado")
            
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"💥 Error en loop principal mejorado: {e}")
            time.sleep(1)

# ------------------ EJECUCIÓN PRINCIPAL MEJORADA ------------------
if __name__ == "__main__":
    start_enhanced_system()
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info",
        access_log=True
    )
else:
    start_enhanced_system()
