import websocket
import json
import time
import threading
from datetime import datetime
import requests

class IQOptionWebSocket:
    def __init__(self, email, password):
        self.email = email
        self.password = password
        self.ws = None
        self.connected = False
        self.candles_data = []
        self.ssid = None
        
    def connect(self):
        """Conectar a IQ Option via WebSocket"""
        try:
            # Primero obtener SSID via login HTTP
            login_success = self._http_login()
            if not login_success:
                return False, "Error en login HTTP"
            
            # Conectar WebSocket
            self.ws = websocket.WebSocketApp(
                "wss://iqoption.com/echo/websocket",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Ejecutar en hilo separado
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Esperar conexión
            timeout = 10
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            return self.connected, "Conectado" if self.connected else "Timeout"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _http_login(self):
        """Login via HTTP para obtener SSID"""
        try:
            session = requests.Session()
            
            # Headers para simular navegador
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Content-Type': 'application/json'
            }
            
            # Datos de login
            login_data = {
                'identifier': self.email,
                'password': self.password
            }
            
            # Hacer login
            response = session.post(
                'https://auth.iqoption.com/api/v2/login',
                json=login_data,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ssid'):
                    self.ssid = data['ssid']
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error en login HTTP: {e}")
            return False
    
    def _on_message(self, ws, message):
        """Manejar mensajes WebSocket"""
        try:
            data = json.loads(message)
            print(f"📨 Mensaje WebSocket: {data}")
            
            # Aquí procesarías los datos de velas reales
            if 'candles' in str(data):
                self.candles_data.append(data)
                
        except Exception as e:
            print(f"Error procesando mensaje: {e}")
    
    def _on_error(self, ws, error):
        print(f"❌ Error WebSocket: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        print("🔌 WebSocket cerrado")
        self.connected = False
    
    def _on_open(self, ws):
        print("✅ WebSocket conectado")
        self.connected = True
        
        # Enviar autenticación
        if self.ssid:
            auth_msg = {
                'name': 'ssid',
                'msg': self.ssid
            }
            ws.send(json.dumps(auth_msg))
    
    def get_candles(self, asset, timeframe, count):
        """Obtener velas históricas (simulado por ahora)"""
        # Por simplicidad, devolver datos simulados
        # En producción, aquí pedirías velas reales via API
        import random
        candles = []
        base_price = 1.08  # EURUSD base
        
        for i in range(count):
            candles.append({
                'from': int(time.time()) - (i * timeframe),
                'open': base_price + random.uniform(-0.01, 0.01),
                'max': base_price + random.uniform(0, 0.02),
                'min': base_price + random.uniform(-0.02, 0),
                'close': base_price + random.uniform(-0.01, 0.01),
                'volume': random.uniform(1000, 5000)
            })
        
        return candles
    
    def get_realtime_candles(self, asset, timeframe, count):
        """Obtener velas en tiempo real"""
        return self.get_candles(asset, timeframe, count)

# Para usar en main.py
class IQ_Option:
    def __init__(self, email, password):
        self.ws_client = IQOptionWebSocket(email, password)
        self.connected = False
        
    def connect(self):
        self.connected, message = self.ws_client.connect()
        return self.connected, message
        
    def get_candles(self, asset, timeframe, count, timestamp):
        return self.ws_client.get_candles(asset, timeframe, count)
        
    def change_balance(self, mode):
        print(f"💰 Modo cuenta: {mode}")
        
    def get_balance(self):
        return 10000.0
        
    def get_all_open_time(self):
        return {"binary": {"EURUSD-OTC": {"open": True}}}
        
    def get_realtime_candles(self, asset, timeframe, count):
        return self.ws_client.get_realtime_candles(asset, timeframe, count)
