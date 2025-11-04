import time
import threading
import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np
import socket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from iqoptionapi.stable_api import IQ_Option
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import uvicorn
import joblib

# --- INTENTO DE IMPORTAR TENSORFLOW (si falla, usa solo MLP) ---
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    USE_TF = False

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- CREDENCIALES IQ OPTION ---
EMAIL = "vozhechacancion1@gmail.com"
PASSWORD = "Eduyesy1986/"

# --- FASTAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IA MODELOS ---
mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1500, activation='relu')
scaler = StandardScaler()
model_trained = False
latest_candles = []
IQ = None

# --- RUTAS Y CONFIGURACIONES ---
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.gz")
MLP_PATH = os.path.join(MODELS_DIR, "mlp_joblib.gz")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_model")
SEQ_LEN = 30
MIN_SAMPLES_TO_TRAIN = 60
lstm_model = None
lstm_ready = False

# ---------------------------------------------------------------
# ✅ CONEXIÓN A IQ OPTION
# ---------------------------------------------------------------
def conectar_iq():
    global IQ
    IQ = IQ_Option(EMAIL, PASSWORD)
    for intento in range(5):
        try:
            logging.info("🔗 Conectando a IQ Option...")
            IQ.connect()
            IQ.change_balance("PRACTICE")
            logging.info("✅ Conectado a IQ Option (modo DEMO).")
            return True
        except Exception as e:
            logging.error(f"❌ Error de conexión: {e}")
            time.sleep(2)
    return False

# ---------------------------------------------------------------
# ✅ OBTENER VELAS
# ---------------------------------------------------------------
def obtener_velas(par="EURUSD-OTC", duracion=60, velas=200):
    global latest_candles
    try:
        IQ.start_candles_stream(par, duracion, velas)
        candles = IQ.get_candles(par, duracion, velas, time.time())
        latest_candles = candles
        return candles
    except Exception as e:
        logging.error(f"❌ Error obteniendo velas: {e}")
        return []

# ---------------------------------------------------------------
# ✅ INDICADORES TÉCNICOS
# ---------------------------------------------------------------
def calcular_indicadores(df):
    df = df.copy()
    if 'high' in df.columns and 'max' not in df.columns:
        df.rename(columns={'high': 'max', 'low': 'min'}, inplace=True)

    df['SMA_3'] = df['close'].rolling(3).mean()
    df['SMA_5'] = df['close'].rolling(5).mean()
    df['EMA_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['TR'] = pd.concat([
        df['max'] - df['min'],
        (df['max'] - df['close'].shift()).abs(),
        (df['min'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['+DM'] = df['max'].diff().where(lambda x: x > 0, 0.0)
    df['-DM'] = -df['min'].diff().where(lambda x: x < 0, 0.0)
    tr14 = df['TR'].rolling(14).sum()
    df['+DI'] = 100 * (df['+DM'].rolling(14).sum() / tr14.replace(0, np.nan))
    df['-DI'] = 100 * (df['-DM'].rolling(14).sum() / tr14.replace(0, np.nan))
    df['ADX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan)) * 100
    df['ADX'] = df['ADX'].ewm(span=14).mean().fillna(0)

    df['BB_middle'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']).fillna(0)

    df['TP'] = (df['close'] + df['max'] + df['min']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(20).mean()) / (0.015 * df['TP'].rolling(20).std()).replace(0, np.nan)

    df['momentum'] = df['close'].diff(3).fillna(0)
    df['ATR'] = df['TR'].rolling(14).mean().fillna(0)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ---------------------------------------------------------------
# ✅ PREPARAR FEATURES Y LABEL (CORREGIDO para predecir la siguiente vela)
# ---------------------------------------------------------------
def prepare_features_targets(df):
    df = calcular_indicadores(df)
    features = ['open','close','min','max','volume','SMA_3','SMA_5','EMA_3','EMA_5',
                'RSI','MACD','MACD_signal','ADX','+DI','-DI','BB_upper','BB_lower','BB_width',
                'CCI','momentum','ATR']
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    X = df[features]
    # ✅ target = close(next) > open(next)
    y = (df['close'].shift(-1) > df['open'].shift(-1)).astype(int).fillna(0)
    return X, y, features

# ---------------------------------------------------------------
# ✅ CONSTRUIR MODELOS Y ENTRENAR
# ---------------------------------------------------------------
def build_lstm_model(n_features):
    model = Sequential([
        LSTM(128, input_shape=(SEQ_LEN, n_features), return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def entrenar_modelo():
    global model_trained, latest_candles, scaler, mlp_model, lstm_model, lstm_ready
    try:
        if len(latest_candles) < MIN_SAMPLES_TO_TRAIN:
            return
        df = pd.DataFrame(latest_candles)
        X_df, y_series, _ = prepare_features_targets(df)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            scaler.fit(X_df.values)
            joblib.dump(scaler, SCALER_PATH)
        X_scaled = scaler.transform(X_df.values)
        mlp_model.fit(X_scaled, y_series.values)
        joblib.dump(mlp_model, MLP_PATH)
        model_trained = True
        logging.info("✅ MLP entrenado.")

        # Fine-tuning del LSTM
        if USE_TF:
            sequences = []
            labels = []
            for i in range(len(X_scaled) - SEQ_LEN - 1):
                sequences.append(X_scaled[i:i+SEQ_LEN])
                labels.append(y_series.values[i+SEQ_LEN])
            X_seq = np.array(sequences)
            y_seq = np.array(labels)
            if len(X_seq) >= 20:
                if lstm_model is None:
                    lstm_model = build_lstm_model(X_seq.shape[2])
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                lstm_model.fit(X_seq, y_seq, epochs=5, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
                lstm_model.save(LSTM_PATH)
                lstm_ready = True
                logging.info("✅ LSTM entrenado o actualizado.")
    except Exception as e:
        logging.error(f"Error entrenando modelo: {e}")

# ---------------------------------------------------------------
# ✅ PREDICCIÓN SIGUIENTE VELA
# ---------------------------------------------------------------
def predecir_siguiente_vela():
    if not model_trained or len(latest_candles) == 0:
        return {"prediction": "N/A", "confidence": 0}
    try:
        df = pd.DataFrame(latest_candles)
        X_df, _, _ = prepare_features_targets(df)
        X_scaled = scaler.transform(X_df.values)
        if USE_TF and lstm_ready and lstm_model is not None and len(X_scaled) >= SEQ_LEN:
            seq = X_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X_scaled.shape[1])
            prob = float(lstm_model.predict(seq, verbose=0)[0][0])
            pred = 1 if prob >= 0.5 else 0
            conf = round(max(prob, 1 - prob) * 100, 2)
            return {"prediction": pred, "confidence": conf}
        else:
            X_new = X_scaled[-1].reshape(1, -1)
            prob_arr = mlp_model.predict_proba(X_new)[0]
            pred = int(np.argmax(prob_arr))
            conf = round(float(max(prob_arr) * 100), 2)
            return {"prediction": pred, "confidence": conf}
    except Exception as e:
        logging.error(f"❌ Error prediciendo: {e}")
        return {"prediction": "N/A", "confidence": 0}

# ---------------------------------------------------------------
# ✅ SERVIDOR Y FRONTEND (HTML CORREGIDO)
# ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <html>
    <head>
        <title>Delowyss Trading - CEO Eduardo Solis</title>
        <style>
            body { font-family: Arial; text-align: center; background: #111; color: white; }
            button { font-size: 20px; padding: 15px 30px; border-radius: 12px; border: none; cursor: pointer; }
            #timer { font-size: 18px; margin-top: 10px; }
            #result { font-size: 20px; margin-top: 15px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🚀 Delowyss Trading Bot - EUR/USD OTC</h1>
        <button id="btn" onclick="analyze()">ANALIZAR Y PREDECIR</button>
        <p id="timer">Tiempo restante: --s</p>
        <p id="result"></p>

        <script>
        async function getServerSeconds() {
            try {
                const r = await fetch('/time');
                const j = await r.json();
                return j.seconds_remaining;
            } catch { return 60; }
        }

        let timeLeft = 60;
        const timerEl = document.getElementById('timer');
        const btn = document.getElementById('btn');
        async function syncAndStart() {
            timeLeft = await getServerSeconds();
            setInterval(() => {
                timeLeft--;
                if (timeLeft <= 0) syncAndStart();
                timerEl.innerHTML = "Tiempo restante: " + Math.max(0, timeLeft) + "s";
                btn.style.backgroundColor = timeLeft <= 10 ? "red" : "#333";
            }, 1000);
        }
        syncAndStart();

        async function analyze() {
            const res = await fetch('/predict');
            const data = await res.json();
            const div = document.getElementById('result');
            if (data.prediction === "N/A") {
                div.innerHTML = "Predicción: N/D | Confianza: 0%";
            } else {
                const txt = data.prediction === 1 ? "📈 SUBIRÁ ↑" : "📉 BAJARÁ ↓";
                div.innerHTML = "Predicción: " + txt + " | Confianza: " + data.confidence + "%";
                btn.style.backgroundColor = data.prediction === 1 ? "green" : "red";
            }
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/predict")
async def predict():
    return JSONResponse(predecir_siguiente_vela())

@app.get("/time")
async def time_endpoint():
    now = time.time()
    sec_remaining = int(60 - (now % 60))
    return JSONResponse({"seconds_remaining": sec_remaining})

# ---------------------------------------------------------------
# ✅ INICIO
# ---------------------------------------------------------------
def iniciar_bot():
    conectar_iq()
    obtener_velas()
    threading.Thread(target=lambda: (obtener_velas(), entrenar_modelo()), daemon=True).start()

if __name__ == "__main__":
    iniciar_bot()
    uvicorn.run(app, host="0.0.0.0", port=10000)
