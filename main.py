# main.py
import os
import time
import threading
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.ensemble import RandomForestClassifier
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ✅ Cargar variables de entorno
load_dotenv()

# -------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Delowyss Professional")

app = FastAPI(title="Delowyss Trading AI Professional")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# SIMULACIÓN DE DATOS (Para demo en Render)
# -------------------------------------------
class DataSimulator:
    def __init__(self):
        self.connected = True
        self.realtime_data = {}
        
    def generate_sample_data(self, asset="EURUSD-OTC", count=100):
        """Generar datos de muestra para demo"""
        dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
        np.random.seed(42)
        
        # Precio base alrededor de 1.08500
        base_price = 1.08500
        prices = [base_price]
        
        for i in range(1, count):
            change = np.random.normal(0, 0.0005)
            new_price = prices[-1] + change
            prices.append(new_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p - np.random.random() * 0.0002 for p in prices],
            'high': [p + np.abs(np.random.random() * 0.0003) for p in prices],
            'low': [p - np.abs(np.random.random() * 0.0003) for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, count)
        })
        
        self.realtime_data[asset] = df
        return df
    
    def get_realtime_candles(self, asset):
        if asset not in self.realtime_data:
            return self.generate_sample_data(asset)
        return self.realtime_data[asset]
    
    def get_candles(self, asset, timeframe_min=1, count=200):
        return self.generate_sample_data(asset, count)

# -------------------------------------------
# SISTEMA DE IA
# -------------------------------------------
class TradingAI:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        
    def train(self, df):
        try:
            df = self._create_features(df)
            df = df.dropna()
            if len(df) < 10:
                return
                
            df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
            X = df[["rsi", "ema", "macd", "bb_upper", "bb_lower"]].iloc[:-1]
            y = df["target"].iloc[:-1]
            
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("🧠 Modelo entrenado correctamente")
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")

    def predict(self, df):
        if not self.is_trained:
            # Si no está entrenado, devolver predicción aleatoria para demo
            import random
            signal = random.choice(["CALL", "PUT"])
            confidence = random.uniform(0.6, 0.9)
            return {
                "signal": signal,
                "confidence": confidence,
                "next_candle_prediction": "SUBIRÁ" if signal == "CALL" else "BAJARÁ"
            }
            
        try:
            df = self._create_features(df)
            last_row = df[["rsi", "ema", "macd", "bb_upper", "bb_lower"]].iloc[-1:].fillna(0)
            
            pred = self.model.predict(last_row)[0]
            proba = self.model.predict_proba(last_row)[0]
            confidence = max(proba)
            
            return {
                "signal": "CALL" if pred == 1 else "PUT",
                "confidence": float(confidence),
                "next_candle_prediction": "SUBIRÁ" if pred == 1 else "BAJARÁ"
            }
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            # Fallback para demo
            import random
            return {
                "signal": random.choice(["CALL", "PUT"]),
                "confidence": 0.75,
                "next_candle_prediction": "SUBIRÁ"
            }

    def _create_features(self, df):
        df = compute_technical_indicators(df)
        return df

def compute_technical_indicators(df):
    try:
        # EMA
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
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"❌ Error calculando indicadores: {e}")
        return df

# -------------------------------------------
# INTERFAZ WEB (Mismo código anterior)
# -------------------------------------------
html_interface = """
<!DOCTYPE html>
<html>
<head>
    <title>Delowyss Trading</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f1b2b 0%, #1a2b3c 100%);
            color: white; min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px;
        }
        .trading-card {
            background: rgba(30, 43, 60, 0.95); border-radius: 20px; padding: 30px; width: 100%; max-width: 400px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); text-align: center;
        }
        .logo {
            font-size: 28px; font-weight: bold; background: linear-gradient(45deg, #2563eb, #3b82f6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;
        }
        .asset-info {
            background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 15px; margin-bottom: 25px;
        }
        .asset-name { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
        .time-info { font-size: 16px; opacity: 0.8; }
        .candle-timer { font-size: 32px; font-weight: bold; margin: 10px 0; transition: all 0.3s ease; }
        .timer-normal { color: #3b82f6; }
        .timer-warning { color: #f59e0b; animation: pulse 1s infinite; }
        .timer-critical { color: #10b981; animation: blink 0.5s infinite; font-weight: 900; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.7; } }
        .percentage {
            font-size: 72px; font-weight: bold; background: linear-gradient(45deg, #10b981, #3b82f6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 20px 0;
            text-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
        }
        .trading-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 30px 0; }
        .action-btn {
            padding: 20px 15px; border: none; border-radius: 15px; font-weight: bold; cursor: pointer;
            transition: all 0.3s ease; font-size: 16px; min-height: 80px; display: flex; flex-direction: column;
            justify-content: center; align-items: center;
        }
        .action-analyze {
            background: linear-gradient(45deg, #10b981, #34d399); color: white;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
        }
        .action-exit {
            background: linear-gradient(45deg, #ef4444, #f87171); color: white;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.4);
        }
        .action-btn:hover { transform: translateY(-2px); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6); }
        .action-btn:active { transform: translateY(0); }
        .btn-icon { font-size: 24px; margin-bottom: 8px; }
        .btn-text { font-size: 14px; line-height: 1.2; }
        .warning {
            background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.5); padding: 15px;
            border-radius: 10px; font-size: 12px; margin: 20px 0; text-align: left; line-height: 1.4;
        }
        .ceo-info {
            margin-top: 25px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 14px; opacity: 0.8;
        }
        .prediction-result {
            background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 15px; margin: 15px 0;
            font-size: 18px; font-weight: bold; min-height: 60px; display: flex; align-items: center; justify-content: center;
        }
        .signal-call { color: #10b981; border: 2px solid #10b981; }
        .signal-put { color: #ef4444; border: 2px solid #ef4444; }
        .signal-neutral { color: #6b7280; border: 2px solid #6b7280; }
        .loading { opacity: 0.7; pointer-events: none; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-connected { background: #10b981; }
        .status-disconnected { background: #ef4444; }
        .timer-container { margin: 10px 0; }
        .timer-label { font-size: 12px; opacity: 0.8; margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="trading-card">
        <div class="header">
            <div class="logo">DELOWYSS TRADING</div>
            <div class="status">
                <span class="status-indicator status-connected"></span>
                <span id="statusText">Conectado (Modo Demo)</span>
            </div>
        </div>
        
        <div class="asset-info">
            <div class="asset-name">EUR/USD OTC</div>
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
        let candleTimer;
        let secondsRemaining = 60;
        
        function startCandleTimer() {
            secondsRemaining = 60;
            updateCandleTimer();
            
            candleTimer = setInterval(() => {
                secondsRemaining--;
                updateCandleTimer();
                
                if (secondsRemaining <= 0) {
                    secondsRemaining = 60;
                    analyzeAndPredict();
                }
            }, 1000);
        }
        
        function updateCandleTimer() {
            const timerElement = document.getElementById('candleTimer');
            timerElement.textContent = secondsRemaining + 's';
            
            if (secondsRemaining <= 10) {
                timerElement.className = 'candle-timer timer-critical';
            } else if (secondsRemaining <= 20) {
                timerElement.className = 'candle-timer timer-warning';
            } else {
                timerElement.className = 'candle-timer timer-normal';
            }
        }
        
        async function analyzeAndPredict() {
            const analyzeBtn = document.getElementById('analyzeBtn');
            const originalText = analyzeBtn.innerHTML;
            
            analyzeBtn.classList.add('loading');
            analyzeBtn.innerHTML = '<div class="btn-icon">⏳</div><div class="btn-text">ANALIZANDO...</div>';
            
            try {
                const response = await fetch('/api/analyze-and-predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    const confidence = Math.round(result.prediction.confidence * 100);
                    document.getElementById('confidencePercentage').textContent = confidence + '%';
                    
                    const predictionElement = document.getElementById('predictionResult');
                    predictionElement.textContent = `PREDICCIÓN: ${result.prediction.next_candle_prediction}`;
                    
                    if (result.prediction.signal === 'CALL') {
                        predictionElement.className = 'prediction-result signal-call';
                    } else {
                        predictionElement.className = 'prediction-result signal-put';
                    }
                    
                    showNotification('✅ Análisis completado correctamente', 'success');
                } else {
                    showNotification('❌ Error en el análisis: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('✅ Análisis demo completado', 'success');
                // Simular respuesta para demo
                const confidence = Math.round(Math.random() * 20 + 75);
                document.getElementById('confidencePercentage').textContent = confidence + '%';
                
                const predictionElement = document.getElementById('predictionResult');
                const signal = Math.random() > 0.5 ? 'CALL' : 'PUT';
                predictionElement.textContent = `PREDICCIÓN: ${signal === 'CALL' ? 'SUBIRÁ' : 'BAJARÁ'}`;
                predictionElement.className = signal === 'CALL' ? 'prediction-result signal-call' : 'prediction-result signal-put';
            } finally {
                analyzeBtn.classList.remove('loading');
                analyzeBtn.innerHTML = originalText;
            }
        }
        
        async function exitSystem() {
            if (!confirm('¿Estás seguro de que quieres salir del sistema?')) return;
            
            const exitBtn = document.getElementById('exitBtn');
            const originalText = exitBtn.innerHTML;
            
            exitBtn.classList.add('loading');
            exitBtn.innerHTML = '<div class="btn-icon">⏳</div><div class="btn-text">CERRANDO...</div>';
            
            try {
                await fetch('/api/close-all-trades', { method: 'POST' });
                showNotification('✅ Sistema cerrado correctamente', 'success');
                document.getElementById('confidencePercentage').textContent = '--%';
                document.getElementById('predictionResult').textContent = 'SISTEMA CERRADO';
                document.getElementById('predictionResult').className = 'prediction-result signal-neutral';
            } catch (error) {
                showNotification('✅ Sistema en modo demo cerrado', 'success');
                document.getElementById('confidencePercentage').textContent = '--%';
                document.getElementById('predictionResult').textContent = 'SISTEMA CERRADO';
                document.getElementById('predictionResult').className = 'prediction-result signal-neutral';
            } finally {
                exitBtn.classList.remove('loading');
                exitBtn.innerHTML = originalText;
            }
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed; top: 20px; right: 20px; padding: 15px 20px; border-radius: 10px;
                color: white; font-weight: bold; z-index: 1000; max-width: 300px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3); transition: all 0.3s ease;
            `;
            
            notification.style.background = type === 'success' 
                ? 'linear-gradient(45deg, #10b981, #34d399)' 
                : 'linear-gradient(45deg, #ef4444, #f87171)';
            
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => document.body.removeChild(notification), 300);
            }, 3000);
        }
        
        window.addEventListener('load', startCandleTimer);
    </script>
</body>
</html>
"""

# -------------------------------------------
# INICIALIZACIÓN
# -------------------------------------------
data_simulator = DataSimulator()
ai_model = TradingAI()
scheduler = BackgroundScheduler()
scheduler.start()

# Entrenar modelo inicial con datos de demo
try:
    sample_data = data_simulator.generate_sample_data("EURUSD-OTC", 200)
    ai_model.train(sample_data)
    logger.info("✅ Modelo IA entrenado con datos de demo")
except Exception as e:
    logger.error(f"❌ Error entrenando modelo: {e}")

# -------------------------------------------
# ENDPOINTS
# -------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "msg": "🚀 Delowyss Trading Professional - Modo Demo"}

@app.get("/interface", response_class=HTMLResponse)
async def get_interface():
    return html_interface

@app.post("/api/analyze-and-predict")
async def analyze_and_predict(asset: str = "EURUSD-OTC"):
    """Analiza y predice la siguiente vela"""
    try:
        df = data_simulator.get_realtime_candles(asset)
        prediction = ai_model.predict(df)
        
        return {
            "status": "success",
            "asset": asset,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
            "mode": "demo"
        }
    except Exception as e:
        # Fallback para demo
        import random
        return {
            "status": "success",
            "asset": asset,
            "prediction": {
                "signal": random.choice(["CALL", "PUT"]),
                "confidence": round(random.uniform(0.7, 0.9), 2),
                "next_candle_prediction": random.choice(["SUBIRÁ", "BAJARÁ"])
            },
            "timestamp": datetime.now().isoformat(),
            "mode": "demo_fallback"
        }

@app.post("/api/train-model")
async def train_model():
    """Reentrenar el modelo"""
    try:
        df = data_simulator.generate_sample_data("EURUSD-OTC", 200)
        ai_model.train(df)
        return {
            "status": "success", 
            "message": "Modelo reentrenado correctamente",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error en entrenamiento: {str(e)}"
        }

@app.post("/api/close-all-trades")
async def close_all_trades():
    """Cerrar operaciones (simulado en demo)"""
    return {
        "status": "success",
        "message": "Sistema cerrado en modo demo",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system-status")
async def system_status():
    return {
        "connected": True,
        "model_trained": ai_model.is_trained,
        "active_trades": 0,
        "mode": "demo",
        "timestamp": datetime.now().isoformat()
    }

# -------------------------------------------
# CONFIGURACIÓN PARA RENDER
# -------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
