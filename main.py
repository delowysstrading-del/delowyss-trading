# main.py
import os
import time
import threading
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from iqoptionapi.stable_api import IQ_Option
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import json
import asyncio
import pickle
import warnings
warnings.filterwarnings('ignore')

# ✅ Cargar variables de entorno
load_dotenv()

# -------------------------------------------
# CONFIGURACIÓN AVANZADA
# -------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Delowyss Professional")

app = FastAPI(title="Delowyss Trading AI Professional")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# SISTEMA DE APRENDIZAJE CONTINUO MEJORADO
# -------------------------------------------
class ContinuousLearningModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=1000)
        self.is_trained = False
        self.training_history = []
        self.historical_data = pd.DataFrame()
        self.model_path = "models/"
        os.makedirs(self.model_path, exist_ok=True)
        
    def save_model(self):
        """Guardar modelo para aprendizaje continuo"""
        try:
            with open(f"{self.model_path}ai_model.pkl", 'wb') as f:
                pickle.dump({
                    'rf_model': self.rf_model,
                    'nn_model': self.nn_model,
                    'training_history': self.training_history,
                    'historical_data': self.historical_data
                }, f)
            logger.info("💾 Modelo guardado para aprendizaje continuo")
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")

    def load_model(self):
        """Cargar modelo existente para continuar aprendizaje"""
        try:
            with open(f"{self.model_path}ai_model.pkl", 'rb') as f:
                data = pickle.load(f)
                self.rf_model = data['rf_model']
                self.nn_model = data['nn_model']
                self.training_history = data['training_history']
                self.historical_data = data['historical_data']
                self.is_trained = True
            logger.info("📂 Modelo cargado exitosamente")
        except FileNotFoundError:
            logger.info("🆕 No se encontró modelo previo, creando nuevo")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")

    def incremental_train(self, new_data):
        """Entrenamiento incremental sin borrar datos anteriores"""
        try:
            # Combinar con datos históricos
            if self.historical_data.empty:
                self.historical_data = new_data
            else:
                # Mantener solo los últimos 5000 datos para evitar sobrecarga
                combined_data = pd.concat([self.historical_data, new_data], ignore_index=True)
                self.historical_data = combined_data.tail(5000)
            
            if len(self.historical_data) < 100:
                logger.warning("📊 Datos insuficientes para entrenamiento")
                return

            df = self._create_features(self.historical_data)
            X = df[["rsi", "macd", "ema", "bb_upper", "bb_lower", "atr", "momentum", "volatility"]].fillna(0)
            y = np.where(df["close"].shift(-1) > df["close"], 1, 0)
            
            # Eliminar última fila sin target
            X = X.iloc[:-1]
            y = y[:-1]
            
            if len(X) < 50:
                return

            # Entrenamiento incremental
            if self.is_trained:
                # Fine-tuning del modelo existente
                self.rf_model.fit(X, y)  # RandomForest con warm_start=True
                self.nn_model.partial_fit(X, y, classes=[0, 1])
            else:
                # Entrenamiento inicial
                self.rf_model.fit(X, y)
                self.nn_model.fit(X, y)
                self.is_trained = True

            # Guardar métricas
            accuracy_rf = self.rf_model.score(X, y)
            accuracy_nn = self.nn_model.score(X, y)
            
            training_record = {
                "timestamp": datetime.now(),
                "samples": len(X),
                "accuracy_rf": accuracy_rf,
                "accuracy_nn": accuracy_nn
            }
            self.training_history.append(training_record)
            
            logger.info(f"🧠 Modelo actualizado | Muestras: {len(X)} | RF: {accuracy_rf:.3f} | NN: {accuracy_nn:.3f}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento incremental: {e}")

    def predict(self, df):
        if not self.is_trained:
            return {"status": "error", "message": "Modelo no entrenado"}
            
        try:
            df = self._create_features(df)
            last_row = df[["rsi", "macd", "ema", "bb_upper", "bb_lower", "atr", "momentum", "volatility"]].iloc[-1:].fillna(0)
            
            rf_pred = self.rf_model.predict(last_row)[0]
            nn_pred = self.nn_model.predict(last_row)[0]
            
            # Combinar predicciones con pesos
            rf_confidence = max(self.rf_model.predict_proba(last_row)[0])
            nn_confidence = max(self.nn_model.predict_proba(last_row)[0])
            
            # Predicción final basada en confianza combinada
            final_pred = 1 if (rf_pred * rf_confidence + nn_pred * nn_confidence) >= 0.5 else 0
            
            return {
                "signal": "CALL" if final_pred == 1 else "PUT",
                "confidence": float((rf_confidence + nn_confidence) / 2),
                "confidence_rf": float(rf_confidence),
                "confidence_nn": float(nn_confidence),
                "timestamp": datetime.now().isoformat(),
                "next_candle_prediction": "SUBIRÁ" if final_pred == 1 else "BAJARÁ"
            }
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {"status": "error", "message": str(e)}

    def _create_features(self, df):
        df = compute_advanced_indicators(df)
        return df

# -------------------------------------------
# INTERFAZ WEB FINAL - SIN BARRAS RF/NN
# -------------------------------------------
html_interface = """
<!DOCTYPE html>
<html>
<head>
    <title>Delowyss Trading</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f1b2b 0%, #1a2b3c 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .trading-card {
            background: rgba(30, 43, 60, 0.95);
            border-radius: 20px;
            padding: 30px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        .header {
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 28px;
            font-weight: bold;
            background: linear-gradient(45deg, #2563eb, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .asset-info {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 25px;
        }
        
        .asset-name {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .time-info {
            font-size: 16px;
            opacity: 0.8;
        }
        
        .candle-timer {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        
        .timer-normal {
            color: #3b82f6;
        }
        
        .timer-warning {
            color: #f59e0b;
            animation: pulse 1s infinite;
        }
        
        .timer-critical {
            color: #10b981;
            animation: blink 0.5s infinite;
            font-weight: 900;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.7; }
        }
        
        .percentage {
            font-size: 72px;
            font-weight: bold;
            background: linear-gradient(45deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 20px 0;
            text-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
        }
        
        .trading-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 30px 0;
        }
        
        .action-btn {
            padding: 20px 15px;
            border: none;
            border-radius: 15px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 16px;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .action-analyze {
            background: linear-gradient(45deg, #10b981, #34d399);
            color: white;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
        }
        
        .action-exit {
            background: linear-gradient(45deg, #ef4444, #f87171);
            color: white;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.4);
        }
        
        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
        }
        
        .action-btn:active {
            transform: translateY(0);
        }
        
        .btn-icon {
            font-size: 24px;
            margin-bottom: 8px;
        }
        
        .btn-text {
            font-size: 14px;
            line-height: 1.2;
        }
        
        .warning {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.5);
            padding: 15px;
            border-radius: 10px;
            font-size: 12px;
            margin: 20px 0;
            text-align: left;
            line-height: 1.4;
        }
        
        .ceo-info {
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 14px;
            opacity: 0.8;
        }
        
        .prediction-result {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 15px;
            margin: 15px 0;
            font-size: 18px;
            font-weight: bold;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .signal-call {
            color: #10b981;
            border: 2px solid #10b981;
        }
        
        .signal-put {
            color: #ef4444;
            border: 2px solid #ef4444;
        }
        
        .signal-neutral {
            color: #6b7280;
            border: 2px solid #6b7280;
        }
        
        .loading {
            opacity: 0.7;
            pointer-events: none;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-connected {
            background: #10b981;
        }
        
        .status-disconnected {
            background: #ef4444;
        }
        
        .timer-container {
            margin: 10px 0;
        }
        
        .timer-label {
            font-size: 12px;
            opacity: 0.8;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="trading-card">
        <div class="header">
            <div class="logo">DELOWYSS TRADING</div>
            <div class="status">
                <span class="status-indicator" id="statusIndicator"></span>
                <span id="statusText">Conectando...</span>
            </div>
        </div>
        
        <div class="asset-info">
            <div class="asset-name" id="assetName">EUR/USD OTC</div>
            <div class="time-info">MINUTE</div>
            
            <div class="timer-container">
                <div class="timer-label">TIEMPO RESTANTE VELA</div>
                <div class="candle-timer timer-normal" id="candleTimer">60s</div>
            </div>
        </div>
        
        <div class="percentage" id="confidencePercentage">--%</div>
        
        <div class="prediction-result signal-neutral" id="predictionResult">
            ESPERANDO ANÁLISIS
        </div>
        
        <div class="trading-actions">
            <button class="action-btn action-analyze" onclick="analyzeAndPredict()" id="analyzeBtn">
                <div class="btn-icon">✅</div>
                <div class="btn-text">A. ANALIZA Y PREDICE<br>LA SIGUIENTE VELA</div>
            </button>
            <button class="action-btn action-exit" onclick="exitSystem()" id="exitBtn">
                <div class="btn-icon">❌</div>
                <div class="btn-text">B. SALIR DEL SISTEMA</div>
            </button>
        </div>
        
        <div class="warning">
            ⚠️ Advertencia: El trading de opciones binarias es riesgoso y puede resultar en pérdida de su capital. Opere con precaución.
        </div>
        
        <div class="ceo-info">
            CEO Eduardo Solis<br>
            <small>Sistema IA Avanzada - Análisis en Tiempo Real</small>
        </div>
    </div>

    <script>
        // Temporizador de vela de 1 minuto
        let candleTimer;
        let secondsRemaining = 60;
        
        function startCandleTimer() {
            // Reiniciar el temporizador
            secondsRemaining = 60;
            updateCandleTimer();
            
            // Actualizar cada segundo
            candleTimer = setInterval(() => {
                secondsRemaining--;
                updateCandleTimer();
                
                if (secondsRemaining <= 0) {
                    // Reiniciar cuando llegue a cero
                    secondsRemaining = 60;
                    // Análisis automático cuando termina la vela
                    if (document.getElementById('statusIndicator').classList.contains('status-connected')) {
                        analyzeAndPredict();
                    }
                }
            }, 1000);
        }
        
        function updateCandleTimer() {
            const timerElement = document.getElementById('candleTimer');
            timerElement.textContent = secondsRemaining + 's';
            
            // Cambiar colores según el tiempo restante
            if (secondsRemaining <= 10) {
                // Verde y animación crítica para últimos 10 segundos
                timerElement.className = 'candle-timer timer-critical';
            } else if (secondsRemaining <= 20) {
                // Amarillo/naranja para últimos 20 segundos
                timerElement.className = 'candle-timer timer-warning';
            } else {
                // Azul normal para el resto
                timerElement.className = 'candle-timer timer-normal';
            }
        }
        
        // Estado del sistema
        function updateSystemStatus(connected) {
            const indicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            
            if (connected) {
                indicator.className = 'status-indicator status-connected';
                statusText.textContent = 'Conectado';
                // Iniciar temporizador cuando se conecte
                if (!candleTimer) {
                    startCandleTimer();
                }
            } else {
                indicator.className = 'status-indicator status-disconnected';
                statusText.textContent = 'Desconectado';
                // Pausar temporizador si está desconectado
                if (candleTimer) {
                    clearInterval(candleTimer);
                    candleTimer = null;
                }
            }
        }
        
        // Análisis y predicción
        async function analyzeAndPredict() {
            const analyzeBtn = document.getElementById('analyzeBtn');
            const originalText = analyzeBtn.innerHTML;
            
            // Estado de carga
            analyzeBtn.classList.add('loading');
            analyzeBtn.innerHTML = '<div class="btn-icon">⏳</div><div class="btn-text">ANALIZANDO...</div>';
            
            try {
                const response = await fetch('/api/analyze-and-predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Actualizar porcentaje de confianza COMBINADA (78%)
                    const confidence = Math.round(result.prediction.confidence * 100);
                    document.getElementById('confidencePercentage').textContent = confidence + '%';
                    
                    // Mostrar predicción
                    const predictionElement = document.getElementById('predictionResult');
                    predictionElement.textContent = `PREDICCIÓN: ${result.prediction.next_candle_prediction}`;
                    
                    if (result.prediction.signal === 'CALL') {
                        predictionElement.className = 'prediction-result signal-call';
                    } else {
                        predictionElement.className = 'prediction-result signal-put';
                    }
                    
                    // Mostrar notificación de éxito
                    showNotification('✅ Análisis completado correctamente', 'success');
                    
                } else {
                    showNotification('❌ Error en el análisis: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('❌ Error de conexión: ' + error.message, 'error');
            } finally {
                // Restaurar botón
                analyzeBtn.classList.remove('loading');
                analyzeBtn.innerHTML = originalText;
            }
        }
        
        // Salir del sistema
        async function exitSystem() {
            const exitBtn = document.getElementById('exitBtn');
            const originalText = exitBtn.innerHTML;
            
            if (!confirm('¿Estás seguro de que quieres salir del sistema y cerrar todas las operaciones?')) {
                return;
            }
            
            // Estado de carga
            exitBtn.classList.add('loading');
            exitBtn.innerHTML = '<div class="btn-icon">⏳</div><div class="btn-text">CERRANDO...</div>';
            
            try {
                const response = await fetch('/api/close-all-trades', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    showNotification('✅ Sistema cerrado correctamente. Operaciones finalizadas.', 'success');
                    
                    // Resetear interfaz
                    document.getElementById('confidencePercentage').textContent = '--%';
                    document.getElementById('predictionResult').textContent = 'SISTEMA CERRADO';
                    document.getElementById('predictionResult').className = 'prediction-result signal-neutral';
                    
                } else {
                    showNotification('❌ Error al cerrar el sistema: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('❌ Error de conexión: ' + error.message, 'error');
            } finally {
                // Restaurar botón
                exitBtn.classList.remove('loading');
                exitBtn.innerHTML = originalText;
            }
        }
        
        // Mostrar notificación
        function showNotification(message, type) {
            // Crear elemento de notificación
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                z-index: 1000;
                max-width: 300px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            `;
            
            if (type === 'success') {
                notification.style.background = 'linear-gradient(45deg, #10b981, #34d399)';
            } else {
                notification.style.background = 'linear-gradient(45deg, #ef4444, #f87171)';
            }
            
            notification.textContent = message;
            document.body.appendChild(notification);
            
            // Remover después de 3 segundos
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => {
                    document.body.removeChild(notification);
                }, 300);
            }, 3000);
        }
        
        // WebSocket para estado en tiempo real
        let ws;
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket conectado');
                updateSystemStatus(true);
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status') {
                    updateSystemStatus(data.connected);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket desconectado');
                updateSystemStatus(false);
                // Reconectar después de 5 segundos
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function() {
                updateSystemStatus(false);
            };
        }
        
        // Iniciar temporizador cuando la página cargue
        window.addEventListener('load', function() {
            startCandleTimer();
            connectWebSocket();
        });
    </script>
</body>
</html>
"""

# -------------------------------------------
# CONEXIÓN IQ OPTION (Código original funcional)
# -------------------------------------------
class IQConnector:
    def __init__(self, email: str, password: str, mode: str = "PRACTICE"):
        self.email = email
        self.password = password
        self.mode = mode
        self.api = None
        self.connected = False
        self.available_assets = {}
        self.realtime_data = {}
        self.reconnect_lock = threading.Lock()
        self.active_trades = {}
        self._connect()

    def _connect(self):
        try:
            logger.info("🔗 Conectando a IQ Option...")
            if not self.email or not self.password:
                raise Exception("Credenciales no configuradas")
                
            self.api = IQ_Option(self.email, self.password)
            check = self.api.connect()
            
            if not check:
                raise Exception("No se pudo autenticar")
                
            if self.mode.upper() == "REAL":
                self.api.change_balance("REAL")
                logger.info("💰 Modo REAL activado")
            else:
                self.api.change_balance("PRACTICE")
                logger.info("🎯 Modo DEMO activado")

            self.connected = True
            self.available_assets = self.api.get_all_open_time()
            logger.info("✅ Conectado correctamente a IQ Option")
            
        except Exception as e:
            logger.error(f"❌ Error en conexión: {str(e)}")
            self.connected = False
            raise

    def reconnect(self):
        with self.reconnect_lock:
            if not self.connected:
                logger.warning("🔁 Intentando reconexión...")
                self._connect()

    def start_realtime_stream(self, asset: str = "EURUSD-OTC", timeframe_min: int = 1):
        if not self.connected:
            self.reconnect()
            return

        try:
            period = timeframe_min * 60
            logger.info(f"📡 Iniciando stream en tiempo real para {asset} ({timeframe_min}m)")
            self.api.start_candles_stream(asset, period, 1000)
            
            def stream_loop():
                while self.connected:
                    try:
                        candles = self.api.get_realtime_candles(asset, period)
                        if candles:
                            df = pd.DataFrame.from_dict(candles, orient="index")
                            df["timestamp"] = pd.to_datetime(df["from"], unit="s")
                            df.sort_values("timestamp", inplace=True)
                            df = df[["timestamp", "open", "max", "min", "close", "volume"]]
                            self.realtime_data[asset] = df
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"⚠️ Error en stream: {e}")
                        time.sleep(5)
                        self.reconnect()

            t = threading.Thread(target=stream_loop, daemon=True)
            t.start()
            
        except Exception as e:
            logger.error(f"❌ Error iniciando stream: {e}")
            self.reconnect()

    def get_realtime_candles(self, asset: str):
        return self.realtime_data.get(asset, None)

    def get_candles(self, asset: str, timeframe_min: int = 1, count: int = 1000):
        try:
            if not self.connected:
                self.reconnect()
                
            candles = self.api.get_candles(asset, timeframe_min * 60, count, time.time())
            if not candles:
                return None
                
            df = pd.DataFrame([
                {
                    "timestamp": datetime.fromtimestamp(c["from"]),
                    "open": c["open"],
                    "high": c["max"],
                    "low": c["min"],
                    "close": c["close"],
                    "volume": c["volume"],
                } for c in candles
            ])
            return df
        except Exception as e:
            logger.error(f"❌ Error obteniendo velas: {e}")
            return None

    def place_trade(self, asset: str, action: str, amount: float = 1.0):
        try:
            if action.upper() not in ["CALL", "PUT"]:
                raise ValueError("Acción debe ser CALL o PUT")
                
            check, id = self.api.buy(amount, asset, action.upper(), 1)
            
            if check:
                trade_info = {
                    "id": id,
                    "asset": asset,
                    "action": action,
                    "amount": amount,
                    "timestamp": datetime.now(),
                    "status": "active"
                }
                self.active_trades[id] = trade_info
                logger.info(f"✅ Orden ejecutada: {action} en {asset}")
                return {"status": "success", "trade_id": id}
            else:
                logger.error(f"❌ Error ejecutando orden: {id}")
                return {"status": "error", "message": id}
                
        except Exception as e:
            logger.error(f"❌ Error en place_trade: {e}")
            return {"status": "error", "message": str(e)}

    def close_trade(self, trade_id: int):
        try:
            result = self.api.close_option(trade_id)
            if result:
                if trade_id in self.active_trades:
                    self.active_trades[trade_id]["status"] = "closed"
                logger.info(f"✅ Orden {trade_id} cerrada")
                return {"status": "success"}
            else:
                return {"status": "error", "message": "No se pudo cerrar la orden"}
        except Exception as e:
            logger.error(f"❌ Error cerrando orden: {e}")
            return {"status": "error", "message": str(e)}

# -------------------------------------------
# INDICADORES TÉCNICOS (Código original)
# -------------------------------------------
def compute_advanced_indicators(df):
    try:
        # EMA
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["ema"] = df["close"].ewm(span=14, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, min_periods=14).mean()
        avg_loss = loss.ewm(span=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Momentum y Volatilidad
        df['momentum'] = df['close'].pct_change(periods=5)
        df['volatility'] = df['close'].rolling(20).std()
        
        df.dropna(inplace=True)
        return df
        
    except Exception as e:
        logger.error(f"❌ Error calculando indicadores: {e}")
        return df

# -------------------------------------------
# WEBSOCKET MANAGER (Código original)
# -------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# -------------------------------------------
# INICIALIZACIÓN DEL SISTEMA
# -------------------------------------------
iq_conn = IQConnector(
    email=os.getenv("IQ_EMAIL", ""),
    password=os.getenv("IQ_PASSWORD", "")
)

ai_model = ContinuousLearningModel()
ai_model.load_model()

scheduler = BackgroundScheduler()
scheduler.start()

if iq_conn.connected:
    iq_conn.start_realtime_stream("EURUSD-OTC")
    iq_conn.start_realtime_stream("EURUSD")

# -------------------------------------------
# ENDPOINTS PRINCIPALES (Código original)
# -------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return html_interface

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            status_msg = {
                "type": "status",
                "connected": iq_conn.connected,
                "message": "Sistema activo" if iq_conn.connected else "Sistema desconectado"
            }
            await websocket.send_json(status_msg)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/analyze-and-predict")
async def analyze_and_predict(asset: str = "EURUSD-OTC"):
    """Analiza y predice la siguiente vela"""
    try:
        df = iq_conn.get_realtime_candles(asset)
        if df is None or df.empty:
            raise HTTPException(500, "No hay datos disponibles para análisis")
            
        prediction = ai_model.predict(df)
        
        # Entrenamiento automático con nuevos datos (aprendizaje continuo)
        if len(df) >= 50:
            ai_model.incremental_train(df.tail(100))
            
        return {
            "status": "success",
            "asset": asset,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Error en análisis: {str(e)}")

@app.post("/api/execute-trade")
async def execute_trade_action(asset: str = "EURUSD-OTC", action: str = "CALL", amount: float = 1.0):
    """Ejecuta operación basada en análisis"""
    try:
        # Primero analizar
        df = iq_conn.get_realtime_candles(asset)
        if df is None or df.empty:
            raise HTTPException(500, "No hay datos para análisis")
            
        prediction = ai_model.predict(df)
        
        # Si la acción es automática, usar la predicción del modelo
        if action == "AUTO":
            action = prediction["signal"]
            
        # Ejecutar orden
        trade_result = iq_conn.place_trade(asset, action, amount)
        
        return {
            "status": "success",
            "prediction": prediction,
            "trade": trade_result,
            "executed_action": action,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error ejecutando operación: {str(e)}")

@app.post("/api/close-all-trades")
async def close_all_trades():
    """Cierra todas las operaciones activas"""
    try:
        results = []
        for trade_id in list(iq_conn.active_trades.keys()):
            result = iq_conn.close_trade(trade_id)
            results.append({"trade_id": trade_id, "result": result})
            
        return {
            "status": "success",
            "closed_trades": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Error cerrando operaciones: {str(e)}")

@app.post("/api/train-model")
async def train_model(asset: str = "EURUSD-OTC"):
    """Reentrenamiento manual del modelo"""
    try:
        df = iq_conn.get_candles(asset, count=1000)
        if df is None or df.empty:
            raise HTTPException(500, "No se pudieron obtener datos para entrenamiento")
            
        ai_model.incremental_train(df)
        return {
            "status": "success",
            "message": "Modelo IA reentrenado correctamente",
            "training_samples": len(ai_model.historical_data),
            "accuracy_rf": ai_model.training_history[-1]["accuracy_rf"] if ai_model.training_history else 0,
            "accuracy_nn": ai_model.training_history[-1]["accuracy_nn"] if ai_model.training_history else 0
        }
    except Exception as e:
        raise HTTPException(500, f"Error en entrenamiento: {str(e)}")

@app.get("/api/system-status")
async def system_status():
    return {
        "connected": iq_conn.connected,
        "model_trained": ai_model.is_trained,
        "training_samples": len(ai_model.historical_data),
        "active_trades": len(iq_conn.active_trades),
        "last_training": ai_model.training_history[-1]["timestamp"] if ai_model.training_history else None,
        "timestamp": datetime.now().isoformat()
    }

# -------------------------------------------
# TAREAS PROGRAMADAS (Código original)
# -------------------------------------------
def scheduled_training():
    try:
        if iq_conn.connected:
            logger.info("🔄 Entrenamiento automático programado...")
            df = iq_conn.get_candles("EURUSD-OTC", count=200)
            if df is not None and len(df) > 100:
                ai_model.incremental_train(df)
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento programado: {e}")

def scheduled_analysis():
    try:
        if iq_conn.connected and ai_model.is_trained:
            df = iq_conn.get_realtime_candles("EURUSD-OTC")
            if df is not None and len(df) > 50:
                # Análisis automático cada 5 minutos para aprendizaje continuo
                ai_model.incremental_train(df.tail(100))
    except Exception as e:
        logger.error(f"❌ Error en análisis programado: {e}")

# Programar tareas
scheduler.add_job(scheduled_training, 'interval', minutes=30)
scheduler.add_job(scheduled_analysis, 'interval', minutes=5)

# -------------------------------------------
# INICIALIZACIÓN
# -------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
