# app/fxcm_streamer.py
import logging
import time
from fxcmpy import fxcmpy, FxcmpyError
from .db import get_session, Tick

logger = logging.getLogger("DelowyssFXCM")

FXCM_API_TOKEN = "TU_TOKEN_FXCM"  # Mejor poner en variables de entorno
SYMBOLS = ["EUR/USD"]
SLEEP_SECONDS = 1

def start_streaming():
    """
    Conecta a FXCM y guarda ticks en la DB.
    Método bloqueante, ejecutar en hilo separado o asyncio.to_thread
    """
    con = None
    while True:
        try:
            if con is None or not con.is_connected():
                con = fxcmpy(access_token=FXCM_API_TOKEN, log_level='error', server='demo')
                logger.info("Conectado a FXCM exitosamente")

            for symbol in SYMBOLS:
                try:
                    df = con.get_candles(symbol, period='m1', number=1)
                    if not df.empty:
                        last = df.iloc[-1]
                        mid = (last['bidclose'] + last['askclose']) / 2
                        t_ms = int(last.name.timestamp() * 1000)
                        tick = Tick(t_ms=t_ms, mid=mid)

                        with get_session() as session:
                            session.add(tick)
                            session.commit()

                        logger.info(f"Tick guardado {symbol}: {mid}")
                except FxcmpyError as e:
                    logger.error(f"Error FXCM {symbol}: {e}")

            time.sleep(SLEEP_SECONDS)

        except FxcmpyError as e:
            logger.error(f"Error conexión FXCM: {e}, reintentando en 5s...")
            time.sleep(5)
            con = None
        except Exception as e:
            logger.error(f"Error inesperado en streamer FXCM: {e}")
            time.sleep(5)
