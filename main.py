# app/main.py
"""
Delowyss Trading AI
CEO: Eduardo Solis — © 2025
Sistema de trading con IA avanzada
"""

import os
import asyncio
import threading
import logging
import time
from contextvars import ContextVar
from typing import Optional
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
from app.ml_pipeline import load_model, extract_features_from_ticks, predict_from_features, train_from_dataset
from app.fxcm_streamer import start_streaming

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate Limiter — intenta usar X-Forwarded-For si existe (útil detrás de proxies)
def _key_func(req: Request):
    # slowapi's get_remote_address expects a request-like object; prefer X-Forwarded-For
    xff = req.headers.get("x-forwarded-for")
    if xff:
        # X-Forwarded-For may contain multiple IPs — tomar la primera
        return xff.split(",")[0].strip()
    return get_remote_address(req)

limiter = Limiter(key_func=_key_func)
app = FastAPI(title="Delowyss Trading API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# DB
init_db()

# Modelo seguro
MODEL: ContextVar = ContextVar('model')
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

# Cargar modelo inicial
try:
    initial_model = load_model()
    update_model(initial_model)
    logger.info("Modelo inicial cargado exitosamente")
except Exception as e:
    logger.error(f"Error cargando modelo inicial: {e}")
    update_model(None)

# FXCM streamer
async def start_streaming_async():
    try:
        # start_streaming puede ser bloqueante/infinito — ejecutarlo en hilo separado
        await asyncio.to_thread(start_streaming)
        logger.info("Streamer FXCM finalizó correctamente")
    except Exception as e:
        logger.error(f"Error iniciando streamer FXCM: {e}")

# Loop de retreinado por ventanas 1 min
async def retrain_loop():
    while True:
        try:
            logger.info("Iniciando proceso de retreinado...")
            # consultar solo últimos 24 horas para evitar cargar toda la tabla
            one_day_ago = int(time.time() * 1000) - 86_400_000
            with get_session() as session:
                ticks = session.query(Tick).filter(Tick.t_ms >= one_day_ago).order_by(Tick.t_ms).all()
            logger.info(f"Recolectados {len(ticks)} ticks para entrenamiento")

            if len(ticks) >= MIN_TRAINING_SAMPLES:
                # Construir dataset por ventana 1 min
                dataset = []
                labels = []
                window_size_ms = 60_000
                if not ticks:
                    logger.warning("No hay ticks después del filtro de tiempo")
                else:
                    start_time = ticks[0].t_ms
                    window_ticks = []
                    for t in ticks:
                        # si el tick cae dentro de la ventana actual
                        if t.t_ms < start_time + window_size_ms:
                            window_ticks.append({"mid": t.mid})
                        else:
                            # procesar ventana anterior
                            if len(window_ticks) >= 5:
                                features = extract_features_from_ticks(window_ticks)
                                if features:
                                    dataset.append(features)
                                    # etiqueta: comparar siguiente tick con último de la ventana
                                    next_tick = t
                                    delta = next_tick.mid - window_ticks[-1]["mid"]
                                    label = 1 if delta > 0 else 0
                                    labels.append(label)
                            # empezar nueva ventana desde el tick actual
                            start_time = t.t_ms
                            window_ticks = [{"mid": t.mid}]
                    # nota: no olvidamos la última ventana (no tiene "next_tick")
                if dataset and labels:
                    new_model = train_from_dataset(dataset, labels)
                    update_model(new_model)
                    # intentar persistir modelo si ml_pipeline ofrece save_model
                    try:
                        from app.ml_pipeline import save_model as _save_model
                        _save_model(new_model)
                        logger.info("Modelo guardado en disco")
                    except Exception:
                        # no fallamos si no existe save_model; ya está actualizado en memoria
                        logger.info("save_model no disponible o falló; modelo en memoria solamente")
                    logger.info("Modelo retreinado y actualizado exitosamente")
                else:
                    logger.warning("No se generó dataset válido para entrenamiento")
            else:
                logger.warning(f"Insuficientes datos: {len(ticks)} < {MIN_TRAINING_SAMPLES}")
            # dormir 24 horas (se puede ajustar con variable de entorno)
            sleep_seconds = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 3600 * 24))
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            logger.info("Loop de retreinado cancelado")
            break
        except Exception as e:
            logger.error(f"Error en loop de retreinado: {e}")
            # esperar un poco antes de reintentar
            await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando Delowyss Trading API")

    # Evitar loops duplicados si Render/otro PaaS inicia múltiples procesos
    if os.environ.get("RUN_WORKERS", "1") == "1":
        asyncio.create_task(start_streaming_async())
        asyncio.create_task(retrain_loop())
    else:
        logger.warning("RUN_WORKERS=0 → workers desactivados; solo servidor web iniciado")

    logger.info("Aplicación iniciada exitosamente")

# Health check
@app.get("/health")
async def health_check():
    model_status = "loaded" if get_current_model() else "not_loaded"
    db_status = "connected"
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"
        logger.error(f"Health check DB error: {e}")
    return {
        "status": "healthy",
        "model": model_status,
        "database": db_status
    }

# Infer endpoint
@app.post("/api/infer", response_model=PredictionOut)
@limiter.limit("10/minute")
async def infer(request: Request, payload: InferPayload, user=Depends(require_api_key)):
    valid_symbols = ["EUR/USD"]
    if payload.symbol not in valid_symbols:
        raise HTTPException(status_code=400, detail=f"Símbolo inválido. Válidos: {valid_symbols}")
    if not payload.ticks or len(payload.ticks) < 10:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 10 ticks")
    current_model = get_current_model()
    if not current_model:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        ticks = [t.dict() for t in payload.ticks]
        features = extract_features_from_ticks(ticks)
        if not features:
            raise HTTPException(status_code=400, detail="No se pudieron extraer features")
        p_up, p_down = predict_from_features(current_model, features)
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
        logger.info(f"Predicción {signal} (conf: {conf:.3f}) para {payload.symbol}")
        return {"signal": signal, "p_up": p_up, "p_down": p_down, "confidence": conf, "time": payload.time_now}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inferencia usuario {getattr(user, 'username', 'unknown')}: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

# Admin endpoints
@app.post("/admin/create_user")
@limiter.limit("5/minute")
async def create_user(request: Request, username: str, password: str, plan: str = "weekly", x_master: str = Header(None)):
    verify_master(x_master)
    if plan not in ["weekly", "monthly", "premium"]:
        raise HTTPException(status_code=400, detail="Plan inválido")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Contraseña muy corta")
    try:
        api_key = gen_api_key()
        with get_session() as session:
            if session.query(User).filter(User.username == username).first():
                raise HTTPException(status_code=400, detail="Usuario ya existe")
            user = User(username=username, password_hash=hash_password(password), api_key=api_key, plan=plan)
            session.add(user)
            session.commit()
        logger.info(f"Usuario creado: {username} (plan: {plan})")
        return {"username": username, "api_key": api_key, "plan": plan}
    except Exception as e:
        logger.error(f"Error creando usuario {username}: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.get("/admin/stats")
@limiter.limit("30/minute")
async def admin_stats(request: Request, x_master: str = Header(None)):
    verify_master(x_master)
    try:
        with get_session() as session:
            tot_ticks = session.query(Tick).count()
            tot_preds = session.query(Prediction).count()
            tot_users = session.query(User).count()
            user_stats = session.query(
                User.username,
                User.plan,
                User.active,
                Prediction.signal,
                func.count(Prediction.id).label('pred_count')
            ).join(Prediction, User.id==Prediction.user_id, isouter=True).group_by(
                User.username, User.plan, User.active, Prediction.signal
            ).all()
        model_status = "loaded" if get_current_model() else "not_loaded"
        return {
            "ticks": tot_ticks,
            "predictions": tot_preds,
            "users": tot_users,
            "model_status": model_status,
            "user_stats": [{"username": s.username,"plan":s.plan,"active":s.active,"signal":s.signal,"prediction_count":s.pred_count} for s in user_stats]
        }
    except Exception as e:
        logger.error(f"Error stats admin: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@app.get("/me")
@limiter.limit("30/minute")
async def me(request: Request, user=Depends(require_api_key)):
    return {"username": user.username,"plan": user.plan,"active": user.active,"joined_date": getattr(user,'created_at',None)}

@app.get("/admin/model_status")
async def model_status(x_master: str = Header(None)):
    verify_master(x_master)
    model = get_current_model()
    return {"model_loaded": model is not None, "model_type": type(model).__name__ if model else None, "last_retrain": "TODO"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Excepción no manejada en {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})
