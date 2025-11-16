# app/fxcm_streamer.py
"""
Streamer FXCM mínimo para integración con main.py.

Función principal:
- start_streaming(): loop infinito que intenta obtener ticks reales de FXCM si configuras credenciales.
  Si no hay credenciales o fxcmpy no está disponible, usa un simulador de ticks (random walk).
  Inserta ticks en la base de datos usando app.db.get_session() si el modelo Tick existe.
"""

import os
import time
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Intentamos soporte opcional para fxcmpy (solo si lo instalas y configuras)
try:
    import fxcmpy  # pragma: no cover
    _FXCMPY_AVAILABLE = True
except Exception:
    _FXCMPY_AVAILABLE = False

# Importar DB si existe para insertar ticks (main.py ya tiene app.db)
try:
    from app.db import get_session, Tick
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

def _now_ms():
    return int(time.time() * 1000)

def _simulate_tick(prev_mid):
    # random walk pequeño, simula spread y mid
    change = random.uniform(-0.00005, 0.00005)
    mid = float(prev_mid + change)
    # normalizar a 5 decimales típico EUR/USD
    mid = round(mid, 5)
    return {"mid": mid, "t_ms": _now_ms()}

def _store_tick_db(tick_dict):
    if not _DB_AVAILABLE:
        return
    try:
        with get_session() as session:
            t = Tick(mid=float(tick_dict["mid"]), t_ms=int(tick_dict["t_ms"]))
            session.add(t)
            session.commit()
    except Exception as e:
        logger.error(f"[fxcm_streamer] Error guardando tick en DB: {e}")

def start_streaming():
    """
    Funcion principal. Se ejecuta en un hilo (asyncio.to_thread) desde main.py.
    - Si FXCMPY está configurado y hay token, intentará conectarse.
    - En caso contrario usará un simulador local.
    """
    logger.info("[fxcm_streamer] start_streaming iniciado")

    # Si el usuario configuró FXCM_TOKEN y instaló fxcmpy, intentar conectar (opcional)
    fxcm_token = os.environ.get("FXCM_TOKEN")
    symbol = os.environ.get("STREAM_SYMBOL", "EUR/USD")

    if _FXCMPY_AVAILABLE and fxcm_token:
        try:
            logger.info("[fxcm_streamer] Intentando conectar a FXCM...")
            con = fxcmpy.fxcmpy(access_token=fxcm_token, log_level='error')
            logger.info("[fxcm_streamer] Conectado a FXCM")

            # subscribe a ticks
            # Nota: la API real de fxcmpy puede variar; este bloque es orientativo.
            while True:
                try:
                    # obtener último precio cada segundo
                    price = con.get_last_price(symbol)
                    if price is not None:
                        mid = float((price['Bid'] + price['Ask']) / 2)
                        tick = {"mid": mid, "t_ms": _now_ms()}
                        _store_tick_db(tick)
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"[fxcm_streamer] Error en loop FXCM: {e}")
                    time.sleep(5)

        except Exception as e:
            logger.error(f"[fxcm_streamer] Error conectando a FXCM: {e}")
            logger.info("[fxcm_streamer] Cayendo a modo simulador")

    # Modo simulador (seguro, sin dependencias)
    prev_mid = float(os.environ.get("SIM_START_MID", "1.10000"))
    interval = float(os.environ.get("SIM_TICK_INTERVAL", "0.5"))  # segundos

    logger.info("[fxcm_streamer] Modo simulador de ticks activo")
    try:
        while True:
            tick = _simulate_tick(prev_mid)
            prev_mid = tick["mid"]
            # opcional: imprimir o loguear cada N ticks
            logger.debug(f"[fxcm_streamer] Tick sim: {tick}")
            # intentar guardar en DB si está disponible
            _store_tick_db(tick)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("[fxcm_streamer] Simulador detenido por KeyboardInterrupt")
    except Exception as e:
        logger.error(f"[fxcm_streamer] Error en simulador: {e}")
        # el hilo terminará y main.py puede manejar reconexión si lo desea
