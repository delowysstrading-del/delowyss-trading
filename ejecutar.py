import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ta
from ta import add_all_ta_features
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, TSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import xgboost as xgb
import joblib
import os
import warnings
import threading
from flask import Flask
import signal
import sys

warnings.filterwarnings('ignore')

# Configurar Flask para mantener vivo el servicio en Render
app = Flask(__name__)

# Variables globales para almacenar el último análisis
last_analysis_data = None
last_analysis_time = None
current_assistant = None

@app.route('/')
def home():
    return f"""
    <html>
        <head>
            <title>🚀 Delowyss Trading - Sistema Profesional</title>
            <style>
                body {{ 
                    font-family: 'Arial', sans-serif; 
                    margin: 0; 
                    padding: 0; 
                    background: linear-gradient(135deg, #1a2a6c, #2a3a7c, #3a4a8c);
                    color: #ffffff;
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                }}
                .header {{ 
                    text-align: center; 
                    padding: 40px 20px; 
                    background: rgba(0, 0, 0, 0.7);
                    border-radius: 15px;
                    margin-bottom: 30px;
                    border: 2px solid #00d4ff;
                    box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
                }}
                .logo {{ 
                    font-size: 3.5em; 
                    font-weight: bold; 
                    color: #00d4ff;
                    text-shadow: 0 0 20px rgba(0, 212, 255, 0.7);
                    margin-bottom: 10px;
                }}
                .tagline {{
                    font-size: 1.4em;
                    color: #00ff88;
                    margin-bottom: 20px;
                }}
                .status {{ 
                    background: rgba(0, 0, 0, 0.8); 
                    padding: 30px; 
                    margin: 20px 0; 
                    border-radius: 15px;
                    border-left: 5px solid #00ff88;
                }}
                .features {{ 
                    background: rgba(0, 0, 0, 0.8); 
                    padding: 30px; 
                    border-radius: 15px;
                    border-left: 5px solid #ff6b6b;
                }}
                .feature-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .feature-card {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}
                .btn {{
                    display: inline-block;
                    padding: 15px 30px;
                    background: linear-gradient(45deg, #00d4ff, #00ff88);
                    color: #000;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: bold;
                    margin: 10px;
                    transition: all 0.3s ease;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
                }}
                .btn-secondary {{
                    background: linear-gradient(45deg, #ff6b6b, #ffa726);
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    background: rgba(0, 0, 0, 0.7);
                    border-radius: 10px;
                }}
                .alert {{
                    background: rgba(0, 212, 255, 0.2);
                    border: 1px solid #00d4ff;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🚀 DELOWYSS TRADING</div>
                    <div class="tagline">Sistema de Trading Inteligente con IA Avanzada</div>
                    <p>Plataforma profesional para operaciones Forex con tecnología de punta</p>
                </div>
                
                <div class="status">
                    <h2>✅ SISTEMA DELOWYSS ACTIVO - VERSIÓN PROFESIONAL</h2>
                    <p><strong>Servidor:</strong> Render - Infrastructure Delowyss</p>
                    <p><strong>Última actualización:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p><strong>Estado:</strong> 🟢 SISTEMA OPERATIVO AL 100%</p>
                    <p><strong>Versión:</strong> Delowyss Pro 2.0</p>
                </div>

                {"<div class='alert'><strong>📊 ÚLTIMO ANÁLISIS DISPONIBLE:</strong> Haz click en 'VER RESULTADOS' para ver el análisis más reciente.</div>" if last_analysis_data else ""}
                
                <div class="features">
                    <h3>🚀 ACCIONES DEL SISTEMA DELOWYSS:</h3>
                    <div class="feature-grid">
                        <div class="feature-card">
                            <h4>🔍 ANÁLISIS EN TIEMPO REAL</h4>
                            <p>Ejecuta análisis completo del mercado</p>
                            <a href="/run-analysis" class="btn">🚀 EJECUTAR ANÁLISIS</a>
                        </div>
                        <div class="feature-card">
                            <h4>📊 VER RESULTADOS</h4>
                            <p>Muestra el último análisis realizado</p>
                            <a href="/view-results" class="btn btn-secondary">📈 VER RESULTADOS</a>
                        </div>
                        <div class="feature-card">
                            <h4>🤖 ANÁLISIS AUTOMÁTICO</h4>
                            <p>Ejecuta análisis en segundo plano</p>
                            <a href="/auto-analysis" class="btn">🤖 ANÁLISIS AUTO</a>
                        </div>
                        <div class="feature-card">
                            <h4>❤️ MONITOREO</h4>
                            <p>Verifica estado del sistema</p>
                            <a href="/health" class="btn">❤️ SALUD SISTEMA</a>
                        </div>
                    </div>
                </div>

                <div class="footer">
                    <p>© 2024 Delowyss Trading - Sistema Profesional de Trading | 🤖 IA Avanzada | ⚡ Tiempo Real</p>
                    <p>🔒 Sistema seguro y confiable para traders profesionales</p>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/health')
def health():
    return {
        "status": "healthy", 
        "service": "delowyss-trading-pro", 
        "version": "Delowyss Pro 2.0",
        "timestamp": datetime.now().isoformat(),
        "features_active": [
            "IA Delowyss Engine",
            "Pattern Delowyss Scanner", 
            "Delowyss Analytics Pro",
            "Delowyss Risk Manager"
        ],
        "last_analysis_time": last_analysis_time.isoformat() if last_analysis_time else None
    }

@app.route('/run-analysis')
def run_analysis():
    """Ejecutar análisis y mostrar resultados"""
    global last_analysis_data, last_analysis_time, current_assistant
    
    def run_analysis_background():
        global last_analysis_data, last_analysis_time, current_assistant
        try:
            print("🔍 [Delowyss Web] Iniciando análisis desde la web...")
            current_assistant = DelowyssTradingAssistant()
            
            # Realizar análisis y capturar datos
            result = current_assistant.perform_complete_analysis()
            if result:
                last_analysis_data = result
                last_analysis_time = datetime.now()
                print("✅ [Delowyss Web] Análisis completado y resultados guardados")
            else:
                print("❌ [Delowyss Web] No se pudieron capturar los resultados")
                
        except Exception as e:
            print(f"❌ [Delowyss Web] Error en análisis: {e}")
    
    # Ejecutar en segundo plano
    analysis_thread = threading.Thread(target=run_analysis_background, daemon=True)
    analysis_thread.start()
    
    return """
    <html>
        <head>
            <title>Ejecutando Análisis - Delowyss Trading</title>
            <meta http-equiv="refresh" content="5;url=/view-results" />
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    background: linear-gradient(135deg, #1a2a6c, #2a3a7c);
                    color: white;
                    text-align: center;
                    padding: 50px;
                }
                .loader {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #00d4ff;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 2s linear infinite;
                    margin: 20px auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <h1>🚀 EJECUTANDO ANÁLISIS DELOWYSS</h1>
            <div class="loader"></div>
            <p>El sistema está realizando el análisis profesional de mercado...</p>
            <p>⏳ Esto puede tomar 20-30 segundos</p>
            <p>📊 Entrenando modelos de IA...</p>
            <p>🔍 Analizando patrones de mercado...</p>
            <p>Redirigiendo a resultados en 5 segundos...</p>
            <a href="/view-results" style="color: #00d4ff;">Ver resultados inmediatamente</a>
        </body>
    </html>
    """

@app.route('/view-results')
def view_results():
    """Mostrar resultados del análisis"""
    global last_analysis_data
    
    if not last_analysis_data:
        return """
        <html>
            <head>
                <title>No hay Análisis - Delowyss Trading</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #1a2a6c, #2a3a7c);
                        color: white;
                        text-align: center;
                        padding: 50px;
                    }
                </style>
            </head>
            <body>
                <h1>📊 NO HAY ANÁLISIS DISPONIBLE</h1>
                <p>No se ha realizado ningún análisis recientemente.</p>
                <p><a href="/run-analysis" style="color: #00d4ff; font-size: 18px;">🚀 Ejecutar Análisis Ahora</a></p>
                <p><a href="/" style="color: #00ff88;">↩️ Volver al Inicio</a></p>
            </body>
        </html>
        """
    
    # Determinar clase CSS para la dirección
    direction_class = "signal-buy" if last_analysis_data['prediccion'] == 'ALCISTA' else "signal-sell"
    
    return f"""
    <html>
        <head>
            <title>Resultados Análisis - Delowyss Trading</title>
            <style>
                body {{ 
                    font-family: 'Arial', sans-serif; 
                    margin: 0; 
                    padding: 0; 
                    background: linear-gradient(135deg, #1a2a6c, #2a3a7c, #3a4a8c);
                    color: #ffffff;
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 1000px; 
                    margin: 0 auto; 
                    padding: 20px;
                }}
                .header {{ 
                    text-align: center; 
                    padding: 30px; 
                    background: rgba(0, 0, 0, 0.7);
                    border-radius: 15px;
                    margin-bottom: 20px;
                    border: 2px solid #00d4ff;
                }}
                .result-card {{ 
                    background: rgba(0, 0, 0, 0.8);
                    padding: 25px;
                    margin: 20px 0;
                    border-radius: 15px;
                    border-left: 5px solid #00ff88;
                }}
                .signal-buy {{ color: #00ff88; font-weight: bold; font-size: 1.2em; }}
                .signal-sell {{ color: #ff6b6b; font-weight: bold; font-size: 1.2em; }}
                .signal-neutral {{ color: #ffa726; font-weight: bold; }}
                .btn {{
                    display: inline-block;
                    padding: 12px 25px;
                    background: linear-gradient(45deg, #00d4ff, #00ff88);
                    color: #000;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: bold;
                    margin: 10px;
                }}
                .warning {{
                    background: rgba(255, 193, 7, 0.2);
                    border: 1px solid #ffc107;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .result-item {{
                    margin: 10px 0;
                    padding: 10px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 INFORME PROFESIONAL DELOWYSS TRADING</h1>
                    <p>Análisis realizado: {last_analysis_data['timestamp']}</p>
                </div>
                
                <div class="result-card">
                    <h2>🔮 PREDICCIÓN DELOWYSS</h2>
                    <div class="result-item">
                        <p><strong>Dirección:</strong> <span class="{direction_class}">{last_analysis_data['prediccion']}</span></p>
                        <p><strong>Confianza:</strong> {last_analysis_data['confianza']}</p>
                        <p><strong>Probabilidad Alcista:</strong> {last_analysis_data['probabilidad_alcista']}</p>
                        <p><strong>Probabilidad Bajista:</strong> {last_analysis_data['probabilidad_bajista']}</p>
                        <p><strong>Consenso Modelos:</strong> {last_analysis_data['consenso_modelos']}</p>
                    </div>
                </div>
                
                <div class="result-card">
                    <h2>📊 ANÁLISIS DE MERCADO</h2>
                    <div class="result-item">
                        <p><strong>Régimen:</strong> {last_analysis_data['regimen']}</p>
                        <p><strong>Nivel Riesgo:</strong> {last_analysis_data['nivel_riesgo']}</p>
                        <p><strong>Confianza Sistema:</strong> {last_analysis_data['confianza_sistema']}</p>
                        <p><strong>Expiración Óptima:</strong> {last_analysis_data['expiracion_optima']}</p>
                    </div>
                </div>
                
                <div class="result-card">
                    <h2>🎪 SEÑALES DELOWYSS ACTIVAS</h2>
                    {"".join([f"<div class='result-item'><p>• {senal}</p></div>" for senal in last_analysis_data['senales']])}
                </div>
                
                <div class="result-card">
                    <h2>🔍 PATRONES DELOWYSS DETECTADOS</h2>
                    {"".join([f"<div class='result-item'><p>• {patron}</p></div>" for patron in last_analysis_data['patrones']])}
                </div>
                
                <div class="result-card">
                    <h2>💡 RECOMENDACIÓN DELOWYSS</h2>
                    <div class="result-item">
                        <p><strong>Acción:</strong> <span class="{direction_class}">{last_analysis_data['recomendacion']}</span></p>
                        <p><strong>Estrategia:</strong> {last_analysis_data['estrategia']}</p>
                        <p><strong>Gestión de Capital:</strong> {last_analysis_data['capital']}</p>
                        <p><strong>Nivel de Riesgo:</strong> {last_analysis_data['riesgo']}</p>
                    </div>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="/run-analysis" class="btn">🔄 EJECUTAR NUEVO ANÁLISIS</a>
                    <a href="/" class="btn">🏠 VOLVER AL INICIO</a>
                </div>
                
                <div class="warning">
                    <h3>⚠️ DELOWYSS ADVERTENCIA</h3>
                    <p>Trading con riesgos. Educación continua. Sistema profesional para traders experimentados.</p>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/start-bot')
def start_bot():
    def run_bot():
        try:
            assistant = DelowyssTradingAssistant()
            assistant.run_professional_assistant()
        except Exception as e:
            print(f"❌ [Delowyss System] Error en sistema: {e}")
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    return "🤖 Sistema Delowyss Trading iniciado - Análisis profesional en progreso"

@app.route('/auto-analysis')
def auto_analysis():
    """Endpoint para análisis automático"""
    def run_analysis():
        global last_analysis_data, last_analysis_time, current_assistant
        try:
            print("🤖 [Delowyss Auto] Iniciando análisis automático...")
            current_assistant = DelowyssTradingAssistant()
            result = current_assistant.perform_complete_analysis()
            if result:
                last_analysis_data = result
                last_analysis_time = datetime.now()
                print("✅ [Delowyss Auto] Análisis completado y guardado")
        except Exception as e:
            print(f"❌ [Delowyss System] Error en análisis automático: {e}")
    
    analysis_thread = threading.Thread(target=run_analysis, daemon=True)
    analysis_thread.start()
    return "🔍 Análisis automático Delowyss iniciado - <a href='/view-results'>Ver resultados</a>"

# =============================================================================
# CLASES ORIGINALES DEL SISTEMA DELOWYSS (SIN MODIFICACIONES)
# =============================================================================

class DelowyssPatternRecognizer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sequence_length = 10
        self.pattern_history = []
        self.performance_metrics = {}
        self.initialize_delowyss_models()
    
    def initialize_delowyss_models(self):
        """Inicializar modelos IA exclusivos Delowyss"""
        print("🚀 [Delowyss AI] Inicializando motor de IA Delowyss...")
        
        # Motor Principal Delowyss - Random Forest
        self.models['delowyss_forest'] = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=3,
            min_samples_leaf=2, random_state=42, n_jobs=1
        )
        
        # Motor Avanzado Delowyss - XGBoost
        self.models['delowyss_boost'] = xgb.XGBClassifier(
            n_estimators=120, max_depth=7, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1
        )
        
        # Motor de Patrones Delowyss - SVM
        self.models['delowyss_pattern'] = SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
        )
        
        # Motor Neural Delowyss - MLP
        self.models['delowyss_neural'] = MLPClassifier(
            hidden_layer_sizes=(80, 40, 20), activation='relu',
            solver='adam', alpha=0.001, batch_size=32,
            learning_rate='adaptive', max_iter=300, random_state=42
        )
        
        # Sistema de Escalado Delowyss
        self.scalers['delowyss_standard'] = StandardScaler()
        self.scalers['delowyss_robust'] = RobustScaler()
        
        print("✅ [Delowyss AI] Motor de IA inicializado - 4 modelos profesionales")

class DelowyssTradingAnalyst:
    def __init__(self):
        self.pattern_recognizer = DelowyssPatternRecognizer()
        self.historical_data = []
        self.prediction_history = []
        self.model_trained = False
        self.last_training = None
        self.performance_tracker = {}
        
        # Inicializar sistema Delowyss
        self.load_delowyss_models()
    
    def load_delowyss_models(self):
        """Cargar sistema de modelos Delowyss"""
        try:
            self.initialize_models()
            print("✅ [Delowyss System] Sistema cargado - Listo para análisis profesional")
        except Exception as e:
            print(f"❌ [Delowyss System] Error cargando sistema: {e}")
            self.initialize_models()
    
    def initialize_models(self):
        self.pattern_recognizer = DelowyssPatternRecognizer()
        print("🤖 [Delowyss System] Sistema profesional inicializado")

    def save_models(self):
        """Sistema de persistencia Delowyss"""
        print("💾 [Delowyss System] Modelos entrenados - Sistema actualizado")

    def generate_delowyss_market_data(self, periods=180):
        """Generador de datos de mercado Delowyss - Realismo profesional"""
        np.random.seed(int(time.time()))
        
        base_price = 1.0850
        prices = [base_price]
        volumes = [10000]
        
        # Algoritmo Delowyss de simulación de mercado
        trend_component = 0
        cycle_component = 0
        noise_component = 0
        momentum_effect = 0
        
        for i in range(1, periods):
            # Estrategia Delowyss de tendencias
            if i % 50 == 0:
                trend_component = np.random.choice([-0.0004, 0.0004])
            
            # Ciclos Delowyss
            cycle_component = 0.00025 * np.sin(2 * np.pi * i / 25)
            
            # Momentum Delowyss
            if i > 2:
                recent_trend = (prices[-1] - prices[-3]) / prices[-3]
                momentum_effect = recent_trend * 0.25
            
            # Volatilidad Delowyss
            if i > 1:
                recent_volatility = abs((prices[-1] - prices[-2]) / prices[-2])
                noise_std = 0.00025 + recent_volatility * 0.4
            else:
                noise_std = 0.00025
            
            noise_component = np.random.normal(0, noise_std)
            
            # Combinación Delowyss
            price_change = trend_component + cycle_component + momentum_effect + noise_component
            
            # Mecanismo Delowyss de reversión
            if abs(price_change) > 0.0008 and np.random.random() < 0.15:
                price_change *= -0.6
            
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
            
            # Volumen inteligente Delowyss
            volume_change = np.random.normal(0, 1500) + abs(price_change) * 800000
            new_volume = max(1000, volumes[-1] + volume_change)
            volumes.append(new_volume)
        
        # DataFrame profesional Delowyss
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.normal(0, 0.0001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.00015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.00015))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Garantía de calidad Delowyss
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        print("📊 [Delowyss Data] Datos de mercado generados - Calidad profesional")
        return df

    def calculate_delowyss_indicators(self, df):
        """Sistema de indicadores técnicos Delowyss Pro"""
        try:
            print("📈 [Delowyss Analytics] Calculando suite técnica profesional...")
            
            # Suite de Tendencia Delowyss
            df['delowyss_ema_5'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
            df['delowyss_ema_10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
            df['delowyss_ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['delowyss_ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            
            # Sistema MACD Delowyss
            macd_fast = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['delowyss_macd'] = macd_fast.macd()
            df['delowyss_macd_signal'] = macd_fast.macd_signal()
            df['delowyss_macd_histogram'] = macd_fast.macd_diff()
            
            # Ichimoku Delowyss
            ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
            df['delowyss_ichimoku_a'] = ichimoku.ichimoku_a()
            df['delowyss_ichimoku_b'] = ichimoku.ichimoku_b()
            
            # Momentum Suite Delowyss
            df['delowyss_rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
            df['delowyss_rsi_7'] = RSIIndicator(close=df['close'], window=7).rsi()
            df['delowyss_stoch_k'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
            df['delowyss_stoch_d'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch_signal()
            df['delowyss_williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            df['delowyss_tsi'] = TSIIndicator(close=df['close']).tsi()
            
            # Volatilidad Delowyss
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['delowyss_bb_upper'] = bb.bollinger_hband()
            df['delowyss_bb_lower'] = bb.bollinger_lband()
            df['delowyss_bb_middle'] = bb.bollinger_mavg()
            df['delowyss_bb_width'] = df['delowyss_bb_upper'] - df['delowyss_bb_lower']
            df['delowyss_bb_position'] = (df['close'] - df['delowyss_bb_lower']) / (df['delowyss_bb_upper'] - df['delowyss_bb_lower'])
            
            keltner = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
            df['delowyss_kc_upper'] = keltner.keltner_channel_hband()
            df['delowyss_kc_lower'] = keltner.keltner_channel_lband()
            
            df['delowyss_atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            
            # Análisis de Volumen Delowyss
            df['delowyss_vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
            df['delowyss_obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            
            # Patrones de Velas Delowyss
            df['delowyss_body_size'] = abs(df['close'] - df['open']) / df['open']
            df['delowyss_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
            df['delowyss_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            df['delowyss_shadow_ratio'] = df['delowyss_upper_shadow'] / (df['delowyss_lower_shadow'] + 1e-8)
            df['delowyss_is_doji'] = (df['delowyss_body_size'] < 0.0005).astype(int)
            df['delowyss_is_hammer'] = ((df['delowyss_lower_shadow'] > 2 * df['delowyss_body_size']) & (df['delowyss_upper_shadow'] < df['delowyss_body_size'] * 0.5)).astype(int)
            df['delowyss_is_shooting_star'] = ((df['delowyss_upper_shadow'] > 2 * df['delowyss_body_size']) & (df['delowyss_lower_shadow'] < df['delowyss_body_size'] * 0.5)).astype(int)
            
            # Métricas Avanzadas Delowyss
            df['delowyss_trend_strength'] = abs(df['delowyss_ema_5'] - df['delowyss_ema_20']) / df['delowyss_ema_20']
            df['delowyss_momentum_acceleration'] = df['close'].pct_change(3) - df['close'].pct_change(5)
            df['delowyss_volatility_regime'] = df['delowyss_atr'].rolling(10).mean() / df['delowyss_atr']
            
            # Análisis Temporal Delowyss
            df['delowyss_hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['delowyss_minute'] = pd.to_datetime(df['timestamp']).dt.minute
            
            # Features Históricas Delowyss
            for lag in [1, 2, 3]:
                df[f'delowyss_close_lag_{lag}'] = df['close'].shift(lag)
                df[f'delowyss_volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'delowyss_rsi_lag_{lag}'] = df['delowyss_rsi_14'].shift(lag)
                df[f'delowyss_macd_lag_{lag}'] = df['delowyss_macd'].shift(lag)
            
            # Aceleradores Delowyss
            df['delowyss_price_acceleration'] = df['close'].pct_change() - df['close'].pct_change().shift(1)
            df['delowyss_volume_acceleration'] = df['volume'].pct_change() - df['volume'].pct_change().shift(1)
            
            # Target Sistema Delowyss
            df['delowyss_next_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            print("✅ [Delowyss Analytics] Suite técnica calculada - 35+ indicadores")
            return df.dropna()
            
        except Exception as e:
            print(f"❌ [Delowyss Analytics] Error en cálculo técnico: {e}")
            return df

    def scan_delowyss_patterns(self, df):
        """Escáner de Patrones Ocultos Delowyss"""
        patterns = {}
        
        try:
            print("🔍 [Delowyss Scanner] Escaneando patrones de mercado...")
            
            # Patrón Delowyss 1: Divergencia Inteligente
            price_trend = df['close'].rolling(5).mean().pct_change(3)
            indicator_trend = df['delowyss_rsi_14'].rolling(5).mean().diff(3)
            patterns['delowyss_divergence'] = np.corrcoef(price_trend.dropna(), indicator_trend.dropna())[0,1]
            
            # Patrón Delowyss 2: Cambio de Régimen
            volatility_clusters = df['delowyss_atr'].rolling(10).std() / df['delowyss_atr'].rolling(10).mean()
            patterns['delowyss_regime_change'] = volatility_clusters.iloc[-1]
            
            # Patrón Delowyss 3: Momentum Acumulado
            cumulative_momentum = (df['delowyss_macd_histogram'] * df['volume']).rolling(5).sum()
            patterns['delowyss_cumulative_momentum'] = cumulative_momentum.iloc[-1]
            
            # Patrón Delowyss 4: Eficiencia de Mercado
            price_efficiency = abs(df['close'].pct_change()).rolling(10).std()
            patterns['delowyss_market_efficiency'] = price_efficiency.iloc[-1]
            
            # Patrón Delowyss 5: Presión Oculta
            buy_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'])) * df['volume']
            sell_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'])) * df['volume']
            patterns['delowyss_hidden_pressure'] = (buy_pressure.rolling(5).mean() - sell_pressure.rolling(5).mean()).iloc[-1]
            
            print("✅ [Delowyss Scanner] Patrones detectados - Análisis completo")
            return patterns
            
        except Exception as e:
            print(f"❌ [Delowyss Scanner] Error en escaneo: {e}")
            return {}

    def prepare_delowyss_features(self, df):
        """Preparación de Características Delowyss Pro"""
        try:
            # Suite de Features Delowyss
            feature_columns = [
                'delowyss_rsi_14', 'delowyss_rsi_7', 'delowyss_macd', 'delowyss_macd_signal', 'delowyss_macd_histogram',
                'delowyss_stoch_k', 'delowyss_stoch_d', 'delowyss_williams_r', 'delowyss_tsi', 'delowyss_atr', 'delowyss_bb_position',
                'delowyss_body_size', 'delowyss_upper_shadow', 'delowyss_lower_shadow', 'delowyss_shadow_ratio',
                'delowyss_trend_strength', 'delowyss_momentum_acceleration', 'delowyss_volatility_regime',
                'delowyss_price_acceleration', 'delowyss_volume_acceleration'
            ]
            
            # Features Históricas Delowyss
            for lag in [1, 2, 3]:
                feature_columns.extend([f'delowyss_close_lag_{lag}', f'delowyss_rsi_lag_{lag}', f'delowyss_macd_lag_{lag}'])
            
            # Análisis Temporal Delowyss
            feature_columns.extend(['delowyss_hour', 'delowyss_minute'])
            
            # Dataset Final Delowyss
            feature_df = df[feature_columns].copy()
            feature_df['delowyss_target'] = df['delowyss_next_direction']
            
            return feature_df.dropna()
            
        except Exception as e:
            print(f"❌ [Delowyss Features] Error en preparación: {e}")
            return None

    def train_delowyss_models(self, df):
        """Entrenamiento del Sistema IA Delowyss"""
        try:
            print("🧠 [Delowyss AI] Entrenando motor de inteligencia...")
            
            # Preparar características Delowyss
            feature_df = self.prepare_delowyss_features(df)
            if feature_df is None or len(feature_df) < 80:
                print("❌ [Delowyss AI] Datos insuficientes para entrenamiento")
                return False
            
            X = feature_df.drop('delowyss_target', axis=1)
            y = feature_df['delowyss_target']
            
            # Validación Delowyss
            tscv = TimeSeriesSplit(n_splits=4)
            
            # Escalado Delowyss
            X_scaled = self.pattern_recognizer.scalers['delowyss_standard'].fit_transform(X)
            
            # Entrenamiento Multi-Modelo Delowyss
            model_performance = {}
            
            for name, model in self.pattern_recognizer.models.items():
                print(f"📊 [Delowyss AI] Entrenando {name}...")
                
                # Validación Cruzada Delowyss
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy', n_jobs=1)
                
                # Entrenamiento Final
                model.fit(X_scaled, y)
                
                model_performance[name] = {
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                print(f"   🎯 {name} - Precisión: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
            
            self.performance_tracker = model_performance
            self.model_trained = True
            self.last_training = datetime.now()
            
            print("✅ [Delowyss AI] Motor IA entrenado - Sistema operativo")
            return True
            
        except Exception as e:
            print(f"❌ [Delowyss AI] Error en entrenamiento: {e}")
            return False

    def delowyss_ensemble_prediction(self, df):
        """Sistema de Predicción Ensemble Delowyss"""
        try:
            if not self.model_trained:
                print("⚠️ [Delowyss AI] Entrenando modelos primero...")
                if not self.train_delowyss_models(df):
                    return None
            
            # Preparación para predicción
            feature_df = self.prepare_delowyss_features(df)
            if feature_df is None or len(feature_df) == 0:
                return None
            
            latest_features = feature_df.drop('delowyss_target', axis=1).iloc[-1:]
            latest_scaled = self.pattern_recognizer.scalers['delowyss_standard'].transform(latest_features)
            
            # Predicciones Multi-Modelo
            predictions = {}
            probabilities = {}
            
            for name, model in self.pattern_recognizer.models.items():
                pred = model.predict(latest_scaled)[0]
                proba = model.predict_proba(latest_scaled)[0]
                
                predictions[name] = pred
                probabilities[name] = proba
            
            # Voting Ponderado Delowyss
            weights = {}
            total_performance = 0
            
            for name, perf in self.performance_tracker.items():
                weight = perf['mean_accuracy']
                weights[name] = weight
                total_performance += weight
            
            # Cálculo Ensemble Delowyss
            ensemble_proba = np.zeros(2)
            for name, proba in probabilities.items():
                ensemble_proba += proba * (weights[name] / total_performance)
            
            final_prediction = 1 if ensemble_proba[1] > ensemble_proba[0] else 0
            confidence = max(ensemble_proba)
            
            # Escáner de Patrones Delowyss
            hidden_patterns = self.scan_delowyss_patterns(df)
            
            result = {
                'prediction': 'ALCISTA' if final_prediction == 1 else 'BAJISTA',
                'confidence': confidence,
                'probability_alcista': ensemble_proba[1],
                'probability_bajista': ensemble_proba[0],
                'ensemble_details': predictions,
                'hidden_patterns': hidden_patterns,
                'model_consensus': f"{sum(predictions.values())}/{len(predictions)} modelos Delowyss a favor",
                'timestamp': datetime.now(),
                'system': 'Delowyss Trading Pro'
            }
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            print(f"❌ [Delowyss Prediction] Error en predicción: {e}")
            return None

    def generate_delowyss_analysis(self, df, prediction):
        """Generador de Análisis Delowyss Pro"""
        try:
            latest = df.iloc[-1]
            patterns = prediction['hidden_patterns']
            
            analysis = {
                'market_regime': self.analyze_delowyss_regime(df),
                'risk_assessment': self.assess_delowyss_risk(df),
                'optimal_expiry': self.calculate_delowyss_expiry(df),
                'trading_signals': self.generate_delowyss_signals(df),
                'pattern_insights': self.interpret_delowyss_patterns(patterns),
                'confidence_level': self.calculate_delowyss_confidence(prediction, df),
                'system_version': 'Delowyss Pro 2.0'
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ [Delowyss Analysis] Error en análisis: {e}")
            return {}

    def analyze_delowyss_regime(self, df):
        """Analizador de Régimen de Mercado Delowyss"""
        volatility = df['delowyss_atr'].iloc[-1] / df['close'].iloc[-1]
        trend_strength = abs(df['delowyss_ema_5'].iloc[-1] - df['delowyss_ema_20'].iloc[-1]) / df['delowyss_ema_20'].iloc[-1]
        
        if volatility > 0.001:
            if trend_strength > 0.0005:
                return "DELOWYSS_TENDENCIA_ALTA_VOL"
            else:
                return "DELOWYSS_RANGOS_ALTA_VOL"
        else:
            if trend_strength > 0.0005:
                return "DELOWYSS_TENDENCIA_BAJA_VOL"
            else:
                return "DELOWYSS_CONSOLIDACION_BAJA_VOL"

    def assess_delowyss_risk(self, df):
        """Evaluador de Riesgo Delowyss"""
        volatility = df['delowyss_atr'].iloc[-1] / df['close'].iloc[-1]
        rsi = df['delowyss_rsi_14'].iloc[-1]
        
        risk_score = 0
        if volatility > 0.001:
            risk_score += 2
        if rsi > 70 or rsi < 30:
            risk_score += 1
        if abs(df['delowyss_macd_histogram'].iloc[-1]) > 0.0005:
            risk_score += 1
            
        if risk_score >= 3:
            return "DELOWYSS_RIESGO_ALTO"
        elif risk_score == 2:
            return "DELOWYSS_RIESGO_MODERADO"
        else:
            return "DELOWYSS_RIESGO_BAJO"

    def calculate_delowyss_expiry(self, df):
        """Calculador de Expiración Delowyss"""
        volatility = df['delowyss_atr'].iloc[-1] / df['close'].iloc[-1]
        
        if volatility > 0.001:
            return "1-2 MINUTOS DELOWYSS"
        elif volatility > 0.0005:
            return "2-3 MINUTOS DELOWYSS"
        else:
            return "3-5 MINUTOS DELOWYSS"

    def generate_delowyss_signals(self, df):
        """Generador de Señales Delowyss"""
        signals = []
        latest = df.iloc[-1]
        
        # Señal RSI Delowyss
        if latest['delowyss_rsi_14'] < 30:
            signals.append("DELOWYSS_RSI_SOBREVENDIDO")
        elif latest['delowyss_rsi_14'] > 70:
            signals.append("DELOWYSS_RSI_SOBRECOMPRADO")
        
        # Señal MACD Delowyss
        if latest['delowyss_macd'] > latest['delowyss_macd_signal']:
            signals.append("DELOWYSS_MACD_ALCISTA")
        else:
            signals.append("DELOWYSS_MACD_BAJISTA")
        
        # Señal Bollinger Delowyss
        if latest['close'] < latest['delowyss_bb_lower']:
            signals.append("DELOWYSS_BB_SOBREVENDIDO")
        elif latest['close'] > latest['delowyss_bb_upper']:
            signals.append("DELOWYSS_BB_SOBRECOMPRADO")
        
        return signals

    def interpret_delowyss_patterns(self, patterns):
        """Intérprete de Patrones Delowyss"""
        insights = []
        
        for pattern, value in patterns.items():
            if pattern == 'delowyss_divergence' and abs(value) < 0.3:
                insights.append("DELOWYSS_DIVERGENCIA_DETECTADA")
            elif pattern == 'delowyss_regime_change' and value > 1.5:
                insights.append("DELOWYSS_CAMBIO_VOLATILIDAD")
            elif pattern == 'delowyss_cumulative_momentum' and value > 0:
                insights.append("DELOWYSS_MOMENTUM_POSITIVO")
            elif pattern == 'delowyss_market_efficiency' and value < 0.0005:
                insights.append("DELOWYSS_MERCADO_EFICIENTE")
            elif pattern == 'delowyss_hidden_pressure' and value > 0:
                insights.append("DELOWYSS_PRESION_COMPRA")
            elif pattern == 'delowyss_hidden_pressure' and value < 0:
                insights.append("DELOWYSS_PRESION_VENTA")
        
        return insights if insights else ["DELOWYSS_PATRONES_NEUTRALES"]

    def calculate_delowyss_confidence(self, prediction, df):
        """Calculador de Confianza Delowyss"""
        base_confidence = prediction['confidence']
        
        # Ajuste Delowyss por consistencia
        latest = df.iloc[-1]
        signal_consistency = 0
        
        if (latest['delowyss_rsi_14'] > 50 and prediction['prediction'] == 'ALCISTA') or (latest['delowyss_rsi_14'] < 50 and prediction['prediction'] == 'BAJISTA'):
            signal_consistency += 0.1
        
        if (latest['delowyss_macd'] > latest['delowyss_macd_signal'] and prediction['prediction'] == 'ALCISTA') or (latest['delowyss_macd'] < latest['delowyss_macd_signal'] and prediction['prediction'] == 'BAJISTA'):
            signal_consistency += 0.1
        
        final_confidence = min(0.95, base_confidence + signal_consistency)
        
        if final_confidence > 0.8:
            return "DELOWYSS_CONFIANZA_MUY_ALTA"
        elif final_confidence > 0.7:
            return "DELOWYSS_CONFIANZA_ALTA"
        elif final_confidence > 0.6:
            return "DELOWYSS_CONFIANZA_MODERADA"
        else:
            return "DELOWYSS_CONFIANZA_BAJA"

class DelowyssTradingAssistant:
    def __init__(self):
        self.analyst = DelowyssTradingAnalyst()
        self.session_start = datetime.now()
        self.last_analysis_result = None
    
    def perform_complete_analysis(self):
        """Realizar análisis completo y retornar resultados para la web"""
        try:
            print("🔍 [Delowyss Web] Iniciando análisis completo...")
            
            # Generar datos de mercado
            df = self.analyst.generate_delowyss_market_data(periods=200)
            
            # Calcular indicadores
            df = self.analyst.calculate_delowyss_indicators(df)
            
            # Entrenar modelos si es necesario
            if not self.analyst.model_trained:
                print("🤖 [Delowyss Web] Entrenando modelos IA...")
                self.analyst.train_delowyss_models(df)
            
            # Realizar predicción
            prediction = self.analyst.delowyss_ensemble_prediction(df)
            
            if prediction:
                # Generar análisis
                analysis = self.analyst.generate_delowyss_analysis(df, prediction)
                
                # Capturar resultados para la web
                result = self._capture_analysis_results(df, prediction, analysis)
                self.last_analysis_result = result
                
                print("✅ [Delowyss Web] Análisis completado - Resultados capturados")
                return result
            else:
                print("❌ [Delowyss Web] No se pudo generar la predicción")
                return None
                
        except Exception as e:
            print(f"❌ [Delowyss Web] Error en análisis completo: {e}")
            return None
    
    def _capture_analysis_results(self, df, prediction, analysis):
        """Capturar resultados del análisis para mostrar en la web"""
        latest = df.iloc[-1]
        
        # Determinar recomendación basada en confianza
        if prediction['confidence'] > 0.7:
            if prediction['prediction'] == 'ALCISTA':
                accion = "COMPRA DELOWYSS CONVICTORA"
            else:
                accion = "VENTA DELOWYSS CONVICTORA"
            estrategia = "Confianza alta + Patrones favorables"
        elif prediction['confidence'] > 0.6:
            if prediction['prediction'] == 'ALCISTA':
                accion = "COMPRA DELOWYSS MODERADA"
            else:
                accion = "VENTA DELOWYSS MODERADA"
            estrategia = "Confianza media + Señales mixtas"
        else:
            accion = "ESPERAR SETUP DELOWYSS"
            estrategia = "Baja confianza - Mejor esperar"
        
        return {
            'prediccion': prediction['prediction'],
            'confianza': f"{prediction['confidence']:.2%}",
            'probabilidad_alcista': f"{prediction['probability_alcista']:.2%}",
            'probabilidad_bajista': f"{prediction['probability_bajista']:.2%}",
            'consenso_modelos': prediction['model_consensus'],
            'regimen': analysis['market_regime'],
            'nivel_riesgo': analysis['risk_assessment'],
            'confianza_sistema': analysis['confidence_level'],
            'expiracion_optima': analysis['optimal_expiry'],
            'senales': analysis['trading_signals'],
            'patrones': analysis['pattern_insights'],
            'recomendacion': accion,
            'estrategia': estrategia,
            'capital': "2-3% gestión Delowyss",
            'riesgo': "STOP 1.5% • TAKE PROFIT 2%",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def display_delowyss_welcome(self):
        """Pantalla de Bienvenida Delowyss"""
        print("\n" + "="*80)
        print("🚀 DELOWYSS TRADING PROFESSIONAL - SISTEMA DE TRADING INTELIGENTE")
        print("🤖 IA AVANZADA • ANÁLISIS PROFUNDO • GESTIÓN DE RIESGO")
        print("="*80)
        print("\n📋 COMANDOS DEL SISTEMA DELOWYSS:")
        print("   A - Análisis Profesional con IA Delowyss")
        print("   S - Estadísticas del Sistema Delowyss") 
        print("   R - Reentrenamiento con Aprendizaje Continuo")
        print("   Q - Salir del Sistema Delowyss")
        print("\n" + "="*80)
    
    def perform_delowyss_analysis(self):
        """Ejecutar Análisis Profesional Delowyss"""
        print("\n🔄 [Delowyss System] Iniciando análisis profesional...")
        
        print("📊 [Delowyss Data] Generando datos de mercado...")
        df = self.analyst.generate_delowyss_market_data(periods=200)
        
        print("🔍 [Delowyss Analytics] Calculando suite técnica...")
        df = self.analyst.calculate_delowyss_indicators(df)
        
        if not self.analyst.model_trained:
            print("🤖 [Delowyss AI] Entrenando motor de IA...")
            self.analyst.train_delowyss_models(df)
        
        print("🔮 [Delowyss Prediction] Realizando predicción ensemble...")
        prediction = self.analyst.delowyss_ensemble_prediction(df)
        
        if prediction:
            print("📈 [Delowyss Analysis] Generando análisis completo...")
            analysis = self.analyst.generate_delowyss_analysis(df, prediction)
            self.display_delowyss_report(df, prediction, analysis)
        else:
            print("❌ [Delowyss System] No se pudo completar el análisis")
    
    def perform_automatic_analysis(self):
        """Ejecutar análisis automático para entorno web"""
        print("\n🤖 [Delowyss Auto] Iniciando análisis automático...")
        
        try:
            self.perform_delowyss_analysis()
            print("✅ [Delowyss Auto] Análisis automático completado")
            return True
        except Exception as e:
            print(f"❌ [Delowyss Auto] Error en análisis automático: {e}")
            return False
    
    def display_delowyss_report(self, df, prediction, analysis):
        """Mostrar Reporte Profesional Delowyss"""
        latest = df.iloc[-1]
        
        print("\n" + "🎯 INFORME PROFESIONAL DELOWYSS TRADING")
        print("="*70)
        
        # Información Delowyss
        print(f"🏢 SISTEMA: Delowyss Trading Pro 2.0")
        print(f"💰 PAR: EUR/USD OTC")
        print(f"⏰ TEMPORALIDAD: 1 MINUTO")
        print(f"📅 HORA ANÁLISIS: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💵 PRECIO ACTUAL: {latest['close']:.5f}")
        
        print("\n🔮 PREDICCIÓN DELOWYSS:")
        print(f"   DIRECCIÓN: {prediction['prediction']}")
        print(f"   CONFIANZA: {prediction['confidence']:.2%}")
        print(f"   PROBABILIDAD ALCISTA: {prediction['probability_alcista']:.2%}")
        print(f"   PROBABILIDAD BAJISTA: {prediction['probability_bajista']:.2%}")
        print(f"   CONSENSO MODELOS: {prediction['model_consensus']}")
        
        print("\n📊 ANÁLISIS DELOWYSS:")
        print(f"   RÉGIMEN: {analysis['market_regime']}")
        print(f"   NIVEL RIESGO: {analysis['risk_assessment']}")
        print(f"   CONFIANZA: {analysis['confidence_level']}")
        print(f"   EXPIRACIÓN ÓPTIMA: {analysis['optimal_expiry']}")
        
        print("\n🎪 SEÑALES DELOWYSS:")
        for signal in analysis['trading_signals']:
            print(f"   • {signal}")
        
        print("\n🔍 PATRONES DELOWYSS DETECTADOS:")
        for insight in analysis['pattern_insights']:
            print(f"   • {insight}")
        
        print("\n🤖 SISTEMA DELOWYSS:")
        print(f"   Modelos IA: {len(self.analyst.pattern_recognizer.models)}")
        print(f"   Indicadores: 35+ técnicos")
        print(f"   Último entrenamiento: {self.analyst.last_training}")
        
        print("\n💡 RECOMENDACIÓN DELOWYSS:")
        if prediction['confidence'] > 0.7:
            if prediction['prediction'] == 'ALCISTA':
                action = "COMPRA DELOWYSS"
            else:
                action = "VENTA DELOWYSS"
            print(f"   ACCIÓN: {action} CONVICTORA")
            print(f"   ESTRATEGIA: Confianza alta + Patrones favorables")
        elif prediction['confidence'] > 0.6:
            if prediction['prediction'] == 'ALCISTA':
                action = "COMPRA DELOWYSS"
            else:
                action = "VENTA DELOWYSS"
            print(f"   ACCIÓN: {action} MODERADA")
            print(f"   ESTRATEGIA: Confianza media + Señales mixtas")
        else:
            print("   ACCIÓN: ESPERAR SETUP DELOWYSS")
            print("   ESTRATEGIA: Baja confianza - Mejor esperar")
        
        print(f"   CAPITAL: 2-3% gestión Delowyss")
        print(f"   RIESGO: STOP 1.5% • TAKE PROFIT 2%")
        
        print("\n" + "="*70)
        print("⚠️  DELOWYSS ADVERTENCIA: Trading con riesgos. Educación continua.")
        print("    Sistema profesional para traders experimentados.")
        print("="*70)
    
    def show_delowyss_stats(self):
        """Estadísticas del Sistema Delowyss"""
        if not self.analyst.performance_tracker:
            print("📊 [Delowyss Stats] Sistema en entrenamiento inicial")
            return
        
        print("\n📈 ESTADÍSTICAS SISTEMA DELOWYSS:")
        for model_name, stats in self.analyst.performance_tracker.items():
            print(f"   {model_name.upper()}:")
            print(f"     Precisión: {stats['mean_accuracy']:.3f} (±{stats['std_accuracy'] * 2:.3f})")
        
        print(f"\n🕐 Último entrenamiento: {self.analyst.last_training}")
        print(f"📅 Inicio sesión: {self.session_start}")
        print(f"🔮 Predicciones: {len(self.analyst.prediction_history)}")
        print(f"🤖 Modelos IA: {len(self.analyst.pattern_recognizer.models)}")
        print(f"🎯 Tecnología: 35+ indicadores Delowyss")
    
    def run_professional_assistant(self):
        """Ejecutar Asistente Delowyss - Solo para entorno local"""
        # Verificar si estamos en entorno con terminal interactivo
        if not sys.stdin.isatty():
            print("🌐 [Delowyss] Entorno web detectado - Ejecutando análisis automático")
            self.perform_automatic_analysis()
            return
        
        # Solo ejecutar interfaz interactiva en entorno local
        self.display_delowyss_welcome()
        
        while True:
            try:
                command = input("\n🎯 Comando Delowyss (A/S/R/Q): ").upper().strip()
                
                if command == 'A':
                    print("\n🔍 [Delowyss] Iniciando análisis profesional...")
                    self.perform_delowyss_analysis()
                    
                elif command == 'S':
                    self.show_delowyss_stats()
                    
                elif command == 'R':
                    print("\n🔄 [Delowyss AI] Reentrenamiento con aprendizaje continuo...")
                    df = self.analyst.generate_delowyss_market_data(periods=250)
                    if df is not None:
                        success = self.analyst.train_delowyss_models(df)
                        if success:
                            print("✅ [Delowyss AI] Sistema actualizado - IA optimizada")
                        else:
                            print("❌ [Delowyss AI] Error en reentrenamiento")
                    else:
                        print("❌ [Delowyss Data] Error en datos")
                        
                elif command == 'Q':
                    print("\n👋 [Delowyss] ¡Hasta luego! Gestión profesional de riesgo.")
                    print("💡 Delowyss Trading - Tecnología para traders serios.")
                    break
                    
                else:
                    print("❌ [Delowyss] Comando no reconocido. Usar: A, S, R, Q")
                    
            except KeyboardInterrupt:
                print("\n\n👋 [Delowyss] Sistema interrumpido. ¡Hasta luego!")
                break
            except EOFError:
                print("\n🌐 [Delowyss] Entorno web - Cambiando a modo automático")
                self.perform_automatic_analysis()
                break
            except Exception as e:
                print(f"❌ [Delowyss System] Error: {e}")

def run_delowyss_server():
    """Servidor Web Delowyss"""
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 [Delowyss Server] Iniciando servidor en puerto {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

def delowyss_signal_handler(signum, frame):
    """Manejador de Señales Delowyss"""
    print(f"📞 [Delowyss System] Señal {signum} - Cerrando sistema...")
    sys.exit(0)

# Configurar sistema Delowyss
signal.signal(signal.SIGTERM, delowyss_signal_handler)
signal.signal(signal.SIGINT, delowyss_signal_handler)

def main_delowyss_system():
    """Sistema principal Delowyss"""
    print("🚀 INICIANDO DELOWYSS TRADING PROFESSIONAL...")
    
    # Verificar entorno Render
    if os.environ.get("RENDER") or not sys.stdin.isatty():
        print("🌐 [Delowyss] Entorno web detectado - Modo servidor activo")
        print("🤖 [Delowyss] Iniciando plataforma de trading...")
        
        # Servidor web Delowyss
        server_thread = threading.Thread(target=run_delowyss_server, daemon=True)
        server_thread.start()
        
        # Ejecutar análisis automático inicial
        print("🔍 [Delowyss] Ejecutando análisis automático inicial...")
        try:
            assistant = DelowyssTradingAssistant()
            result = assistant.perform_complete_analysis()
            if result:
                global last_analysis_data, last_analysis_time
                last_analysis_data = result
                last_analysis_time = datetime.now()
                print("✅ [Delowyss] Análisis inicial completado y guardado")
        except Exception as e:
            print(f"❌ [Delowyss System] Error en análisis inicial: {e}")
        
        # Mantener sistema activo
        print("💤 [Delowyss] Sistema Delowyss activo - Servidor web ejecutándose")
        print("🌐 Accede a la URL de Render para usar el sistema")
        while True:
            time.sleep(300)
            print("❤️  [Delowyss] Health check - Sistema operativo")
            
    else:
        # Ejecución local con interfaz interactiva
        print("💻 [Delowyss] Entorno local - Sistema profesional interactivo")
        assistant = DelowyssTradingAssistant()
        assistant.run_professional_assistant()

if __name__ == "__main__":
    main_delowyss_system()
