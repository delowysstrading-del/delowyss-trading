# 🚀 Delowyss Trading Professional

Sistema IA + FastAPI + IQ Option con análisis en tiempo real.

## 🧠 Funciones
- Conexión directa IQ Option
- Stream de velas en vivo (EUR/USD OTC)
- Análisis técnico con RSI, EMA, MACD, Bollinger Bands
- Predicción CALL/PUT con IA (RandomForest)
- API lista para conectar a una extensión Chrome o front-end

## ⚙️ Despliegue en Render
1. Crea un nuevo proyecto en [Render](https://render.com/)
2. Conecta tu repositorio de GitHub
3. Render detectará `requirements.txt` y `Procfile`
4. Configura las variables de entorno:
   - `IQ_EMAIL`
   - `IQ_PASSWORD`

## 📡 Endpoints principales
- `/` → Estado del sistema  
- `/api/train` → Entrena el modelo IA  
- `/api/analyze` → Devuelve la señal CALL/PUT en tiempo real  

---

Hecho con ❤️ por Delowyss AI Trading System (Eduardo Solis)
