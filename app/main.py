# app/main.py
"""
Delowyss Trading AI - Render Ready
CEO: Eduardo Solis — © 2025
Sistema de trading con IA avanzada
"""

import os
import asyncio
import threading
import logging
import time
from contextvars import ContextVar

from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse

from sqlalchemy import func
from sqlalchemy.sql import text

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas import InferPayload, PredictionOut
from app.auth import require_api_key, verify_master, gen_api_key, hash_password
from app.db import init_db, get_session, Tick, Prediction, User
from app.ml_pipeline import (
    load_model,
    extract_features_from_ticks,
    predict_from_features,
    train_from_dataset
)
from app.fxcm_streamer import start_streaming

# ================================
# LOGGING
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("DelowyssAI")

# ================================
# RATE LIMITER
# ================================
def _key_func(req: Request):
    xff = req.headers.get("x-forwarded-for")
    return xff.split(",")[0].strip() if xff else get_remote_address(req)

limiter = Limiter(key_func=_key_func)
app = FastAPI(title="Delowyss Trading API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ================================
# DATABASE INIT
# ================================
init_db()

# ================================
# MODEL CONTEXT
# ================================
MODEL: ContextVar = ContextVar("model")
model_lock = threading.Lock()
MIN_TRAINING_SAMPLES = 1000

def update_model(new_model):
    with model_lock:
        MODEL.set(new_model)
        logger.info("Modelo actualizado exitosamente")

def get_current_model():
    try:
        return MODEL.get()
    except LookupError:
        return None

# Load initial model
try:
    initial_model = load_model()
    update_model(initial_model)
    logger.info("Modelo inicial cargado")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    update_model(None)

# ================================
# FXCM STREAMING
# ================================
async def start_streaming_async():
    try:
        await asyncio.to_thread(start_streaming)
        logger.info("Streamer FXCM finalizó")
    except Exception as e:
        logger.error(f"Error en streamer FXCM: {e}")

# ================================
# RETRAIN LOOP
# ================================
async def retrain_loop():
    interval = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 86400))
    while True:
        try:
            logger.info("Iniciando retrain automático...")
            one_day_ago = int(time.time() * 1000) - 86_400_000

            with get_session() as session:
                ticks = session.query(Tick).filter(Tick.t_ms >= one_day_ago).order_by(Tick.t_ms).all()

            logger.info(f"Ticks para entrenamiento: {len(ticks)}")
            if len(ticks) >= MIN_TRAINING_SAMPLES:
                dataset, labels = [], []
                window_size_ms = 60_000
                start_time = ticks[0].t_ms if ticks else 0
                window_ticks = []

                for t in ticks:
                    if t.t_ms < start_time + window_size_ms:
                        window_ticks.append({"mid": t.mid})
                    else:
                        if len(window_ticks) >= 5:
                            features = extract_features_from_ticks(window_ticks)
                            if features:
                                dataset.append(features)
                                next_tick = t
                                delta = next_tick.mid - window_ticks[-1]["mid"]
                                labels.append(1 if delta > 0 else 0)
                        start_time = t.t_ms
                        window_ticks = [{"mid": t.mid}]

                if dataset and labels:
                    new_model = train_from_dataset(dataset, labels)
                    update_model(new_model)
                    try:
                        from app.ml_pipeline import save_model as _save
                        _save(new_model)
                        logger.info("Modelo guardado en disco")
                    except:
                        logger.info("save_model no disponible")
            else:
                logger.warning("No hay datos suficientes para retrain")

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Retrain loop cancelado")
            break
        except Exception as e:
            logger.error(f"Error en retrain: {e}")
            await asyncio.sleep(300)

# ================================
# STARTUP
# ================================
@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando Delowyss Trading API Render-ready...")
    if os.environ.get("RUN_WORKERS", "1") == "1":
        asyncio.create_task(start_streaming_async())
        asyncio.create_task(retrain_loop())
    else:
        logger.warning("RUN_WORKERS=0 → Workers deshabilitados")
    logger.info("API iniciada correctamente")

# ================================
# HEALTH CHECK
# ================================
@app.get("/health")
async def health_check():
    model_status = "loaded" if get_current_model() else "not_loaded"
    db_status = "connected"
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"
    return {"status": "healthy", "model": model_status, "database": db_status}

# ================================
# INFERENCIA
# ================================
@app.post("/api/infer", response_model=PredictionOut)
@limiter.limit("10/minute")
async def infer(request: Request, payload: InferPayload, user=Depends(require_api_key)):
    valid_symbols = ["EUR/USD"]
    if payload.symbol not in valid_symbols:
        raise HTTPException(400, f"Símbolo inválido. Válidos: {valid_symbols}")
    if not payload.ticks or len(payload.ticks) < 10:
        raise HTTPException(400, "Se necesitan al menos 10 ticks")

    model = get_current_model()
    if not model:
        raise HTTPException(503, "Modelo no disponible")

    try:
        ticks = [t.dict() for t in payload.ticks]
        features = extract_features_from_ticks(ticks)
        if not features:
            raise HTTPException(400, "No se pudieron extraer features")

        p_up, p_down = predict_from_features(model, features)
        conf = max(p_up, p_down)
        signal = "CALL" if p_up > p_down else "PUT"
        if conf < 0.7:
            signal = "NONE"

        with get_session() as session:
            pred = Prediction(
                symbol=payload.symbol,
                time_now_ms=payload.time_now,
                p_up=p_up,
                p_down=p_down,
                confidence=conf,
                signal=signal,
                user_id=user.id
            )
            session.add(pred)
            session.commit()

        logger.info(f"Predicción {signal} ({conf:.3f}) — {payload.symbol}")
        return {"signal": signal, "p_up": p_up, "p_down": p_down, "confidence": conf, "time": payload.time_now}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inferencia ({user.username}): {e}")
        raise HTTPException(500, "Error interno")

# ================================
# ADMIN — CREAR USUARIO
# ================================
@app.post("/admin/create_user")
@limiter.limit("5/minute")
async def create_user(request: Request, username: str, password: str, plan: str = "weekly", x_master: str = Header(None)):
    verify_master(x_master)
    if plan not in ["weekly", "monthly", "premium"]:
        raise HTTPException(400, "Plan inválido")
    if len(password) < 8:
        raise HTTPException(400, "Contraseña muy corta")

    try:
        api_key = gen_api_key()
        with get_session() as session:
            if session.query(User).filter(User.username == username).first():
                raise HTTPException(400, "Usuario ya existe")
            user = User(username=username, password_hash=hash_password(password), api_key=api_key, plan=plan)
            session.add(user)
            session.commit()

        logger.info(f"Usuario creado: {username}")
        return {"username": username, "api_key": api_key, "plan": plan}
    except Exception as e:
        logger.error(f"Error creando usuario {username}: {e}")
        raise HTTPException(500, "Error interno")

# ================================
# GLOBAL EXCEPTION
# ================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Excepción no manejada en {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})
