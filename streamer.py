# app/streamer.py
"""
Streamer genérico para Delowyss Trading AI.

Exposición:
- start_streaming(): loop infinito (bloqueante) pensado para correr con asyncio.to_thread()

Características:
- Si configuras FXCM_TOKEN y la librería fxcmpy está instalada intentará conectar.
- Si no, usa un simulador (random-walk) seguro para pruebas en Render.
- Intenta almacenar ticks en DB si detecta modelos/session helpers.
"""

import os
import time
import random
import logging

logger = logging.getLogger(__name__)

# Intento de import DB (soporta get_session o get_db)
_DB_AVAILABLE = False
_get_session_fn = None
_TICK_MODEL = None

try:
    # preferencia: get_session (nombre usado en main.py anterior)
    from app.db import get_session, Tick  # type: ignore
    _DB_AVAILABLE = True
    _get_session_fn = get_session
    _TICK_MODEL = Tick
except Exception:
    try:
        # alternativa: get_db (nombre usado en otra versión)
        from app.db import get_db, Tick  # type: ignore
        _DB_AVAILABLE = True
        _get_session_fn = get_db
        _TICK_MODEL = Tick
    except Exception:
        _DB_AVAILABLE = False
        _get_session_fn = None
        _TICK_MODEL = None

# Intento FXCM (opcional)
try:
    import fxcmpy  # pragma: no cover
    _FXCMPY_AVAILABLE = True
except Exception:
    _FXCMPY_AVAILABLE = False


def _now_ms():
    return int(time.time() * 1000)


def _simulate_tick(prev_mid):
    change = random.uniform(-0.00007, 0.00007)
    mid = round(prev_mid + change, 5)
    return {"mid": float(mid), "t_ms": _now_ms()}


def _store_tick_db(tick_dict):
    if not _DB_AVAILABLE or _get_session_fn is None or _TICK_MODEL is None:
        return False
    try:
        # _get_session_fn may be a generator (yield) or function returning session
        sess_gen = _get_session_fn()
        # if generator, use as context manager-like
        if hasattr(sess_gen, "__iter__"):
            session = next(sess_gen)
            try:
                t = _TICK_MODEL(mid=float(tick_dict["mid"]), t_ms=int(tick_dict["t_ms"]))
                session.add(t)
                session.commit()
            finally:
                try:
                    next(sess_gen)
                except StopIteration:
                    pass
        else:
            # callable returns session directly
            session = sess_gen
            t = _TICK_MODEL(mid=float(tick_dict["mid"]), t_ms=int(tick_dict["t_ms"]))
            session.add(t)
            session.commit()
        return True
    except Exception as e:
        logger.error(f"[streamer] Error guardando tick en DB: {e}")
        return False


def start_streaming():
    """
    Función principal para ejecutar en background:
        await asyncio.to_thread(start_streaming)
    """

    logger.info("[streamer] start_streaming iniciado")

    fxcm_token = os.environ.get("FXCM_TOKEN")
    symbol = os.environ.get("STREAM_SYMBOL", "EUR/USD")
    sim_start_mid = float(os.environ.get("SIM_START_MID", "1.10000"))
    sim_interval = float(os.environ.get("SIM_TICK_INTERVAL", "0.5"))

    # Intentar FXCM si está disponible y token presente
    if _FXCMPY_AVAILABLE and fxcm_token:
        try:
            logger.info("[streamer] Intentando conectar a FXCM...")
            con = fxcmpy.fxcmpy(access_token=fxcm_token, log_level="error")
            logger.info("[streamer] Conectado a FXCM")
            # Notar: API real puede diferir; este bloque es orientativo.
            while True:
                try:
                    price = con.get_last_price(symbol)
                    if price is not None:
                        mid = float((price["Bid"] + price["Ask"]) / 2)
                        tick = {"mid": mid, "t_ms": _now_ms()}
                        _store_tick_db(tick)
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"[streamer] Error loop FXCM: {e}")
                    time.sleep(5)
        except Exception as e:
            logger.error(f"[streamer] Error conectando FXCM: {e}")
            logger.info("[streamer] Caída a modo simulador")

    # Simulador (modo seguro)
    prev_mid = sim_start_mid
    logger.info("[streamer] Modo simulador activo")
    try:
        while True:
            tick = _simulate_tick(prev_mid)
            prev_mid = tick["mid"]
            logger.debug(f"[streamer] Tick sim: {tick}")
            _store_tick_db(tick)
            time.sleep(sim_interval)
    except KeyboardInterrupt:
        logger.info("[streamer] Simulador detenido por KeyboardInterrupt")
    except Exception as e:
        logger.error(f"[streamer] Error en simulador: {e}")
