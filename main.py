# En lugar de:
# from iqoptionapi.stable_api import IQ_Option

# Usa esta simulación:
class IQ_Option:
    def __init__(self, email, password):
        self.email = email
        self.password = password
        
    def connect(self):
        print("🔗 Simulando conexión a IQ Option")
        return True, "Simulated connection"
        
    def get_candles(self, asset, timeframe, count, timestamp):
        # Simular datos de velas
        import random
        candles = []
        for i in range(count):
            candles.append({
                'from': timestamp - (i * timeframe),
                'open': random.uniform(1.0, 1.2),
                'max': random.uniform(1.1, 1.3),
                'min': random.uniform(0.9, 1.1),
                'close': random.uniform(1.0, 1.2),
                'volume': random.uniform(1000, 5000)
            })
        return candles
        
    def change_balance(self, mode):
        print(f"💰 Modo de cuenta: {mode}")
        
    def get_balance(self):
        return 10000.0  # Balance simulado
        
    def get_all_open_time(self):
        return {"binary": {"EURUSD-OTC": {"open": True}}}
