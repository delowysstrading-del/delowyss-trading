import logging
from fxcmpy import fxcmpy, FxcmpyError  # Asegúrate de instalar fxcmpy
import time
from .db import get_session, Tick

logger = logging.getLogger("DelowyssFXCM")

FXCM_API_TOKEN = "TU_TOKEN_FXCM"  # Poner en variables de entorno en producción
SYMBOLS = ["EUR/USD"]  # Ajusta según tu necesidad

def start_streaming():
    """
    Conecta a FXCM y guarda ticks en la DB.
    Este método es bloqueante, se ejecuta en hilo separado.
    """
    try:
        con = fxcmpy(access_token=FXCM_API_TOKEN, log_level='error', server='demo')  # demo o real
        logger.info("Conectado a FXCM exitosamente")

        while True:
            for symbol in SYMBOLS:
                try:
                    # Obtener el precio actual (bid+ask)/2 = mid
                    df = con.get_candles(symbol, period='m1', number=1)  # última vela 1 minuto
                    if not df.empty:
                        last = df.iloc[-1]
                        mid = (last['bidclose'] + last['askclose']) / 2
                        t_ms = int(last.name.timestamp() * 1000)
                        tick = Tick(t_ms=t_ms, mid=mid)
                        # Guardar en DB
                        for session in get_session():
                            session.add(tick)
                            session.commit()
                        logger.info(f"Tick guardado {symbol}: {mid}")
                except FxcmpyError as e:
                    logger.error(f"Error FXCM {symbol}: {e}")
            time.sleep(1)  # esperar 1 segundo antes del siguiente tick
    except FxcmpyError as e:
        logger.error(f"No se pudo conectar a FXCM: {e}")
        raise
