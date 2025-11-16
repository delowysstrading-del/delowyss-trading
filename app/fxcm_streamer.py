# app/fxcm_streamer.py
"""
FXCM Streamer
Conecta con FXCM real, guarda ticks y dispara inferencia anticipada
"""

import logging
from fxcmpy import fxcmpy, FxcmpyError
import time
from datetime import datetime
from .db import get_session, Tick
from threading import Thread

logger = logging.getLogger("DelowyssFXCM")

FXCM_API_TOKEN = "TU_TOKEN_FXCM_REAL"  # Token FXCM real, poner en variable de entorno
SYMBOLS = ["EUR/USD"]  # Ajustar según necesidad
STREAM_INTERVAL = 1  # segundos entre ticks
PREDICTION_LEAD_TIME = 3  # segundos antes de cerrar la vela para predecir

def start_streaming(current_window_ticks):
    """
    Conecta a FXCM real y guarda ticks en DB.
    current_window_ticks: lista compartida para enviar ticks al main.py
    """
    try:
        con = fxcmpy(access_token=FXCM_API_TOKEN, log_level='error', server='real')  # cambio a real
        logger.info("Conectado a FXCM REAL exitosamente")

        last_candle_time = None

        while True:
            for symbol in SYMBOLS:
                try:
                    df = con.get_candles(symbol, period='m1', number=1)  # última vela 1 min
                    if df.empty:
                        continue

                    last = df.iloc[-1]
                    candle_time = last.name.timestamp()
                    mid = (last['bidclose'] + last['askclose']) / 2
                    t_ms = int(candle_time * 1000)

                    # Guardar tick en DB
                    with get_session() as session:
                        tick = Tick(t_ms=t_ms, mid=mid)
                        session.add(tick)
                        session.commit()
                    logger.info(f"Tick guardado {symbol}: {mid}")

                    # Guardar en ventana actual para inferencia anticipada
                    current_window_ticks.append({"mid": mid, "t_ms": t_ms})

                    # Disparar predicción 3-5 segundos antes de cerrar vela
                    if last_candle_time and candle_time - last_candle_time >= 57:  # 60-3 segundos
                        try:
                            from app.main import infer_tick_loop
                            Thread(target=infer_tick_loop, args=(symbol,)).start()
                        except Exception as e:
                            logger.error(f"No se pudo ejecutar infer_tick_loop: {e}")

                    last_candle_time = candle_time

                except FxcmpyError as e:
                    logger.error(f"Error FXCM {symbol}: {e}")
            time.sleep(STREAM_INTERVAL)

    except FxcmpyError as e:
        logger.error(f"No se pudo conectar a FXCM REAL: {e}")
        raise
