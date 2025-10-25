import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time
import os
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, TSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import logging
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DelowyssTrading')

class DelowyssIQOptionAPI:
    """Conexión premium Delowyss con IQ Option API"""
    
    def __init__(self, email: str = None, password: str = None):
        self.email = email or os.getenv('IQ_OPTION_EMAIL', 'delowyss@trading.com')
        self.password = password or os.getenv('IQ_OPTION_PASSWORD', 'delowyss_pro_2024')
        self.api = None
        self.connected = False
        self.balance = 0.0
        self.account_type = "REAL"
        
    def connect(self) -> bool:
        """Establecer conexión con IQ Option"""
        try:
            logger.info("🚀 DELOWYSS: Iniciando conexión premium con IQ Option...")
            
            # Para conexión real, instalar: pip install iqoptionapi
            try:
                from iqoptionapi import IQOptionAPI
                self.api = IQOptionAPI(self.email, self.password)
                
                if self.api.connect():
                    self.connected = True
                    self.balance = self.api.get_balance()
                    logger.info(f"✅ DELOWYSS: Conectado exitosamente - Balance: ${self.balance:.2f}")
                    return True
                else:
                    logger.warning("❌ DELOWYSS: Falló conexión real, usando modo simulación avanzada")
                    self.connected = False
                    return False
                    
            except ImportError:
                logger.warning("📦 DELOWYSS: iqoptionapi no instalado. Usando simulación avanzada.")
                logger.info("💡 Para conexión real: pip install iqoptionapi")
                self.connected = False
                return True  # Permitir continuar en modo simulación
                
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error conexión: {e}")
            self.connected = False
            return False

    def get_real_time_data(self, asset: str = "EURUSD-OTC", timeframe: int = 60, count: int = 200) -> Optional[pd.DataFrame]:
        """Obtener datos en tiempo real de IQ Option"""
        try:
            if self.connected and self.api:
                # Datos reales de IQ Option
                candles = self.api.get_candles(asset, timeframe, count)
                if candles:
                    df = self._process_real_candles(candles)
                    logger.info(f"📊 DELOWYSS: {len(df)} velas reales obtenidas")
                    return df
            
            # Datos simulados de alta fidelidad
            return self._generate_high_fidelity_data(count)
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error obteniendo datos: {e}")
            return self._generate_high_fidelity_data(count)

    def _process_real_candles(self, candles: List) -> pd.DataFrame:
        """Procesar velas reales de IQ Option"""
        df = pd.DataFrame(candles)
        df.rename(columns={
            'from': 'timestamp',
            'open': 'open',
            'max': 'high',
            'min': 'low', 
            'close': 'close',
            'volume': 'volume'
        }, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    def _generate_high_fidelity_data(self, count: int) -> pd.DataFrame:
        """Generar datos simulados de alta fidelidad con patrones realistas"""
        np.random.seed(int(time.time()))
        
        base_price = 1.08500
        prices = [base_price]
        volumes = [10000]
        
        # Parámetros de mercado realistas
        volatility_regimes = [0.0002, 0.0004, 0.0006]  # Baja, Media, Alta volatilidad
        current_volatility = 0.0004
        trend_direction = 0
        trend_strength = 0
        
        for i in range(1, count):
            # Cambios de régimen de volatilidad
            if i % 50 == 0:
                current_volatility = np.random.choice(volatility_regimes)
            
            # Cambios de tendencia
            if i % 75 == 0:
                trend_direction = np.random.choice([-1, 1])
                trend_strength = np.random.uniform(0.0001, 0.0003)
            
            # Componentes de precio
            trend = trend_direction * trend_strength
            noise = np.random.normal(0, current_volatility)
            cycle = 0.00015 * np.sin(2 * np.pi * i / 25)
            
            price_change = trend + noise + cycle
            
            # Patrones de reversión
            if abs(price_change) > current_volatility * 3:
                price_change *= -0.5
            
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
            
            # Volumen correlacionado con volatilidad
            volume_factor = 1 + (current_volatility / 0.0002)
            base_volume = 8000
            volume_noise = np.random.normal(0, 2000)
            new_volume = max(5000, base_volume * volume_factor + volume_noise)
            volumes.append(new_volume)
        
        dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.normal(0, 0.0001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.00015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.00015))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Asegurar consistencia OHLC
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        logger.info(f"📊 DELOWYSS: {len(df)} velas simuladas (alta fidelidad)")
        return df

    def place_trade(self, direction: str, amount: float, expiry: int = 2) -> Dict:
        """Colocar operación en IQ Option"""
        try:
            if self.connected and self.api:
                # Operación real
                result = self.api.buy(amount, "EURUSD-OTC", direction.lower(), expiry)
                return {
                    'success': result,
                    'type': 'REAL_TRADE',
                    'direction': direction,
                    'amount': amount,
                    'expiry': expiry
                }
            else:
                # Simulación de operación
                return {
                    'success': True,
                    'type': 'SIMULATION_TRADE',
                    'direction': direction,
                    'amount': amount,
                    'expiry': expiry,
                    'message': 'Modo simulación - Operación virtual'
                }
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error en operación: {e}")
            return {'success': False, 'error': str(e)}

class DelowyssPatternDetector:
    """Motor avanzado de detección de patrones Delowyss"""
    
    def __init__(self):
        self.pattern_history = []
        self.pattern_thresholds = {
            'divergence': 0.3,
            'momentum': 0.1,
            'volatility': 1.2,
            'accumulation': 1000
        }
    
    def detect_comprehensive_patterns(self, df: pd.DataFrame) -> Dict:
        """Detección exhaustiva de patrones de mercado"""
        patterns = {}
        
        try:
            # 1. ANÁLISIS DE DIVERGENCIA AVANZADA
            patterns.update(self._detect_divergence_patterns(df))
            
            # 2. ANÁLISIS DE MOMENTUM OCULTO
            patterns.update(self._detect_hidden_momentum(df))
            
            # 3. ANÁLISIS DE ACUMULACIÓN/DISTRIBUCIÓN
            patterns.update(self._detect_accumulation_patterns(df))
            
            # 4. ANÁLISIS DE VOLATILIDAD
            patterns.update(self._detect_volatility_patterns(df))
            
            # 5. ANÁLISIS DE TENDENCIA
            patterns.update(self._detect_trend_patterns(df))
            
            # 6. ANÁLISIS DE CICLOS TEMPORALES
            patterns.update(self._detect_time_cycles(df))
            
            # 7. ANÁLISIS DE PERFIL DE VOLUMEN
            patterns.update(self._detect_volume_profile(df))
            
            self.pattern_history.append({
                'timestamp': datetime.now(),
                'patterns': patterns,
                'market_regime': self._determine_market_regime(patterns)
            })
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error en detección de patrones: {e}")
            return {}

    def _detect_divergence_patterns(self, df: pd.DataFrame) -> Dict:
        """Detectar divergencias regulares y ocultas"""
        patterns = {}
        
        # Divergencia RSI
        price_trend = df['close'].rolling(8).mean().pct_change(5)
        rsi_trend = df['rsi_14'].rolling(8).mean().diff(5)
        patterns['divergence_rsi'] = np.corrcoef(
            price_trend.dropna(), rsi_trend.dropna()
        )[0,1]
        
        # Divergencia MACD
        macd_trend = df['macd'].rolling(5).mean().diff(3)
        patterns['divergence_macd'] = np.corrcoef(
            price_trend.dropna(), macd_trend.dropna()
        )[0,1]
        
        # Divergencia Estocástico
        stoch_trend = df['stoch_k'].rolling(5).mean().diff(3)
        patterns['divergence_stoch'] = np.corrcoef(
            price_trend.dropna(), stoch_trend.dropna()
        )[0,1]
        
        return patterns

    def _detect_hidden_momentum(self, df: pd.DataFrame) -> Dict:
        """Detectar momentum oculto y fuerzas de mercado no visibles"""
        patterns = {}
        
        # Momentum de precio vs volumen
        price_momentum = df['close'].pct_change(3)
        volume_momentum = df['volume'].pct_change(3)
        patterns['hidden_momentum'] = (price_momentum * volume_momentum).rolling(5).mean().iloc[-1]
        
        # Aceleración de momentum
        patterns['momentum_acceleration'] = (
            df['close'].pct_change(2) - df['close'].pct_change(5)
        ).iloc[-1]
        
        # Momento acumulativo ponderado por volumen
        patterns['volume_weighted_momentum'] = (
            df['close'].pct_change() * df['volume']
        ).rolling(8).sum().iloc[-1]
        
        return patterns

    def _detect_accumulation_patterns(self, df: pd.DataFrame) -> Dict:
        """Detectar patrones de acumulación y distribución"""
        patterns = {}
        
        # Línea de Acumulación/Distribución
        ad_line = ((2*df['close'] - df['high'] - df['low']) / 
                  (df['high'] - df['low'] + 1e-8)) * df['volume']
        patterns['accumulation_line'] = ad_line.rolling(10).mean().iloc[-1]
        
        # Ratio de Fuerza Compradora/Vendedora
        buy_pressure = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
        sell_pressure = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
        patterns['buy_sell_ratio'] = (
            buy_pressure.rolling(5).mean() / sell_pressure.rolling(5).mean()
        ).iloc[-1]
        
        # Flujo de dinero
        mf_volume = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
        patterns['money_flow'] = (mf_volume * df['volume']).rolling(8).sum().iloc[-1]
        
        return patterns

    def _detect_volatility_patterns(self, df: pd.DataFrame) -> Dict:
        """Detectar patrones de volatilidad y compresión"""
        patterns = {}
        
        # Ratio de volatilidad
        patterns['volatility_ratio'] = df['atr'].iloc[-1] / df['atr'].rolling(20).mean().iloc[-1]
        
        # Compresión de Bollinger Bands
        bb_width = df['bb_upper'] - df['bb_lower']
        patterns['bb_squeeze'] = bb_width.iloc[-1] / bb_width.rolling(20).mean().iloc[-1]
        
        # Regímenes de volatilidad
        volatility_regime = df['atr'].rolling(10).std() / df['atr'].rolling(10).mean()
        patterns['volatility_regime_change'] = volatility_regime.iloc[-1]
        
        return patterns

    def _detect_trend_patterns(self, df: pd.DataFrame) -> Dict:
        """Detectar fuerza y calidad de la tendencia"""
        patterns = {}
        
        # Fuerza de tendencia ADX
        patterns['trend_strength'] = df['adx'].iloc[-1] if 'adx' in df.columns else 25
        
        # Calidad de tendencia (suavidad)
        price_changes = df['close'].pct_change().abs()
        patterns['trend_quality'] = price_changes.rolling(10).std().iloc[-1]
        
        # Convergencia/Divergencia de medias
        patterns['ema_convergence'] = (
            (df['ema_9'] - df['ema_21']).abs() / df['ema_21']
        ).iloc[-1]
        
        return patterns

    def _detect_time_cycles(self, df: pd.DataFrame) -> Dict:
        """Detectar patrones temporales y ciclos"""
        patterns = {}
        
        # Análisis de hora del día
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hour_volatility = df.groupby('hour')['atr'].mean()
        patterns['optimal_trading_hour'] = hour_volatility.idxmax()
        
        # Ciclos intraday
        patterns['intraday_cycle_strength'] = np.sin(2 * np.pi * datetime.now().hour / 24)
        
        return patterns

    def _detect_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analizar perfil de volumen y liquidez"""
        patterns = {}
        
        # Relación volumen/precio
        patterns['volume_price_correlation'] = df['volume'].corr(df['close'])
        
        # Clusters de volumen
        volume_zscore = (df['volume'] - df['volume'].mean()) / df['volume'].std()
        patterns['volume_anomaly'] = volume_zscore.iloc[-1]
        
        return patterns

    def _determine_market_regime(self, patterns: Dict) -> str:
        """Determinar el régimen actual del mercado"""
        if patterns.get('trend_strength', 0) > 40:
            return "FUERTE_TENDENCIA"
        elif patterns.get('volatility_ratio', 1) > 1.5:
            return "ALTA_VOLATILIDAD"
        elif patterns.get('bb_squeeze', 1) < 0.7:
            return "COMPRESION_VOLATILIDAD"
        elif abs(patterns.get('divergence_rsi', 0)) > 0.4:
            return "DIVERGENCIA_ACTIVA"
        else:
            return "MERCADO_NEUTRO"

class DelowyssAIAnalyst:
    """Motor de IA avanzado Delowyss con ensemble profundo"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.model_trained = False
        self.last_training = None
        
        self._initialize_advanced_models()
    
    def _initialize_advanced_models(self):
        """Inicializar ensemble avanzado de modelos"""
        logger.info("🧠 DELOWYSS AI: Inicializando ensemble avanzado...")
        
        # 1. Random Forest Optimizado
        self.models['random_forest_pro'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 2. XGBoost Avanzado
        self.models['xgboost_advanced'] = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=7,
            learning_rate=0.12,
            subsample=0.85,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # 3. Gradient Boosting Mejorado
        self.models['gradient_boosting_pro'] = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=6,
            learning_rate=0.15,
            subsample=0.8,
            random_state=42
        )
        
        # 4. SVM Optimizado
        self.models['svm_optimized'] = SVC(
            kernel='rbf',
            C=1.2,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # 5. Red Neuronal Profunda
        self.models['neural_network_deep'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=400,
            random_state=42
        )
        
        # Scalers avanzados
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        logger.info("✅ DELOWYSS AI: Ensemble avanzado inicializado")

    def calculate_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular suite completa de indicadores técnicos"""
        try:
            logger.info("📊 DELOWYSS: Calculando indicadores avanzados...")
            
            # ===== TENDENCIA AVANZADA =====
            # EMA Múltiples
            for window in [3, 5, 9, 12, 21, 26, 50]:
                df[f'ema_{window}'] = EMAIndicator(close=df['close'], window=window).ema_indicator()
            
            # MACD Completo
            macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            df['macd_histogram_change'] = df['macd_histogram'].diff()
            
            # ADX y Direccionales
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx.adx()
            df['di_plus'] = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            
            # Ichimoku Cloud
            ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            
            # ===== MOMENTUM COMPLETO =====
            # RSI Múltiple
            for window in [7, 14, 21]:
                df[f'rsi_{window}'] = RSIIndicator(close=df['close'], window=window).rsi()
            
            # Estocástico Completo
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            
            # TSI (True Strength Index)
            df['tsi'] = TSIIndicator(close=df['close']).tsi()
            
            # ===== VOLATILIDAD AVANZADA =====
            # Bollinger Bands Mejoradas
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(20).mean()
            
            # Keltner Channel
            keltner = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
            df['kc_upper'] = keltner.keltner_channel_hband()
            df['kc_lower'] = keltner.keltner_channel_lband()
            df['kc_middle'] = keltner.keltner_channel_mband()
            
            # ATR y Volatilidad
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            df['atr_ratio'] = df['atr'] / df['close']
            df['volatility_regime'] = df['atr'].rolling(10).std() / df['atr'].rolling(10).mean()
            
            # ===== ANÁLISIS DE VOLUMEN =====
            # VWAP
            df['vwap'] = VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            ).volume_weighted_average_price()
            
            # OBV
            df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            df['obv_trend'] = df['obv'].diff().rolling(5).mean()
            
            # ===== PATRONES DE VELAS AVANZADOS =====
            df['body_size'] = abs(df['close'] - df['open']) / df['open']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-8)
            df['total_range'] = (df['high'] - df['low']) / df['open']
            
            # Patrones específicos
            df['is_doji'] = (df['body_size'] < 0.0005).astype(int)
            df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                             (df['upper_shadow'] < df['body_size'] * 0.3)).astype(int)
            df['is_shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                    (df['lower_shadow'] < df['body_size'] * 0.3)).astype(int)
            df['is_marubozu'] = ((df['body_size'] > 0.001) & 
                               (df['upper_shadow'] < 0.0001) & 
                               (df['lower_shadow'] < 0.0001)).astype(int)
            
            # ===== CARACTERÍSTICAS PERSONALIZADAS =====
            # Fuerza de tendencia
            df['trend_strength_custom'] = abs(df['ema_9'] - df['ema_21']) / df['ema_21']
            
            # Momento aceleración
            df['momentum_acceleration'] = df['close'].pct_change(3) - df['close'].pct_change(5)
            
            # Eficiencia del mercado
            df['market_efficiency'] = df['close'].pct_change().abs().rolling(10).std()
            
            # Relación riesgo/recompensa
            df['risk_reward_ratio'] = df['atr'] / (df['close'].pct_change().abs().rolling(10).mean() * df['close'])
            
            # ===== FEATURES TEMPORALES =====
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            
            # ===== LAG FEATURES AVANZADAS =====
            for lag in [1, 2, 3, 5, 8, 13]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
                df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
                df[f'atr_lag_{lag}'] = df['atr'].shift(lag)
            
            # ===== FEATURES DE DIFERENCIA =====
            df['price_acceleration'] = df['close'].pct_change() - df['close'].pct_change().shift(1)
            df['volume_acceleration'] = df['volume'].pct_change() - df['volume'].pct_change().shift(1)
            df['volatility_acceleration'] = df['atr'].pct_change() - df['atr'].pct_change().shift(1)
            
            # ===== TARGET VARIABLE =====
            df['next_candle_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
            df['next_candle_magnitude'] = df['close'].shift(-1).pct_change()
            
            logger.info("✅ DELOWYSS: Indicadores avanzados calculados")
            return df.dropna()
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error calculando indicadores: {e}")
            return df

    def prepare_advanced_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preparar características avanzadas para el modelo"""
        try:
            # Selección de características más importantes
            feature_columns = [
                # Tendencia
                'ema_3', 'ema_5', 'ema_9', 'ema_12', 'ema_21', 'ema_26',
                'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_change',
                'adx', 'di_plus', 'di_minus',
                'ichimoku_a', 'ichimoku_b', 'ichimoku_base', 'ichimoku_conversion',
                
                # Momentum
                'rsi_7', 'rsi_14', 'rsi_21',
                'stoch_k', 'stoch_d', 'williams_r', 'tsi',
                
                # Volatilidad
                'bb_position', 'bb_width', 'bb_squeeze',
                'atr', 'atr_ratio', 'volatility_regime',
                
                # Volumen
                'vwap', 'obv', 'obv_trend',
                
                # Velas
                'body_size', 'upper_shadow', 'lower_shadow', 'shadow_ratio', 'total_range',
                'is_doji', 'is_hammer', 'is_shooting_star', 'is_marubozu',
                
                # Personalizadas
                'trend_strength_custom', 'momentum_acceleration', 'market_efficiency', 'risk_reward_ratio',
                
                # Temporales
                'hour', 'minute', 'day_of_week', 'is_london_session', 'is_ny_session'
            ]
            
            # Agregar lag features seleccionadas
            for lag in [1, 2, 3]:
                feature_columns.extend([
                    f'close_lag_{lag}', f'volume_lag_{lag}', 
                    f'rsi_lag_{lag}', f'macd_lag_{lag}'
                ])
            
            # Features de diferencia
            feature_columns.extend(['price_acceleration', 'volume_acceleration', 'volatility_acceleration'])
            
            # Crear dataset final
            available_features = [col for col in feature_columns if col in df.columns]
            feature_df = df[available_features].copy()
            feature_df['target'] = df['next_candle_direction']
            
            logger.info(f"📊 DELOWYSS: {len(available_features)} características preparadas")
            return feature_df.dropna()
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error preparando características: {e}")
            return None

    def train_advanced_ensemble(self, df: pd.DataFrame) -> bool:
        """Entrenar ensemble avanzado con validación robusta"""
        try:
            logger.info("🧠 DELOWYSS AI: Iniciando entrenamiento avanzado...")
            
            feature_df = self.prepare_advanced_features(df)
            if feature_df is None or len(feature_df) < 100:
                logger.warning("❌ DELOWYSS: Datos insuficientes para entrenamiento avanzado")
                return False
            
            X = feature_df.drop('target', axis=1)
            y = feature_df['target']
            
            # Validación temporal robusta
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Escalado avanzado
            X_scaled = self.scalers['standard'].fit_transform(X)
            
            # Entrenamiento y evaluación de modelos
            model_performance = {}
            
            for name, model in self.models.items():
                try:
                    logger.info(f"📊 DELOWYSS: Entrenando {name}...")
                    
                    # Cross-validation temporal
                    cv_scores = cross_val_score(
                        model, X_scaled, y, cv=tscv, 
                        scoring='accuracy', n_jobs=-1
                    )
                    
                    # Entrenar modelo final
                    model.fit(X_scaled, y)
                    
                    # Predicciones de validación
                    y_pred = model.predict(X_scaled)
                    accuracy = accuracy_score(y, y_pred)
                    
                    model_performance[name] = {
                        'mean_accuracy': cv_scores.mean(),
                        'std_accuracy': cv_scores.std(),
                        'final_accuracy': accuracy,
                        'cv_scores': cv_scores.tolist()
                    }
                    
                    logger.info(f"   ✅ {name}: CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    logger.error(f"   ❌ DELOWYSS: Error entrenando {name}: {e}")
                    continue
            
            if not model_performance:
                logger.error("❌ DELOWYSS: Todos los modelos fallaron en el entrenamiento")
                return False
            
            self.performance_metrics = model_performance
            self.model_trained = True
            self.last_training = datetime.now()
            
            # Guardar modelos
            self._save_models()
            
            logger.info(f"🎯 DELOWYSS AI: Ensemble entrenado - {len(model_performance)} modelos activos")
            return True
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error en entrenamiento: {e}")
            return False

    def ensemble_prediction(self, df: pd.DataFrame) -> Optional[Dict]:
        """Predicción ensemble avanzada con ponderación inteligente"""
        try:
            if not self.model_trained:
                logger.info("🔄 DELOWYSS: Entrenando modelos primero...")
                if not self.train_advanced_ensemble(df):
                    return None
            
            feature_df = self.prepare_advanced_features(df)
            if feature_df is None or len(feature_df) == 0:
                return None
            
            latest_features = feature_df.drop('target', axis=1).iloc[-1:]
            latest_scaled = self.scalers['standard'].transform(latest_features)
            
            # Recolección de predicciones
            predictions = {}
            probabilities = {}
            successful_models = 0
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(latest_scaled)[0]
                    proba = model.predict_proba(latest_scaled)[0]
                    
                    predictions[name] = pred
                    probabilities[name] = proba
                    successful_models += 1
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ DELOWYSS: Modelo {name} no disponible: {e}")
                    continue
            
            if successful_models == 0:
                logger.error("❌ DELOWYSS: Ningún modelo disponible para predicción")
                return None
            
            # Ensemble voting ponderado por performance
            weights = {}
            total_weight = 0
            
            for name, perf in self.performance_metrics.items():
                if name in predictions:
                    # Ponderación basada en accuracy y estabilidad
                    weight = perf['mean_accuracy'] * (1 - perf['std_accuracy'])
                    weights[name] = weight
                    total_weight += weight
            
            # Predicción ensemble ponderada
            ensemble_proba = np.zeros(2)
            for name, proba in probabilities.items():
                weight = weights.get(name, 0) / total_weight
                ensemble_proba += proba * weight
            
            final_prediction = 1 if ensemble_proba[1] > ensemble_proba[0] else 0
            confidence = max(ensemble_proba)
            
            # Métricas de consenso
            bullish_votes = sum(predictions.values())
            bearish_votes = len(predictions) - bullish_votes
            consensus_ratio = bullish_votes / len(predictions)
            
            result = {
                'prediction': 'ALCISTA' if final_prediction == 1 else 'BAJISTA',
                'confidence': confidence,
                'probability_alcista': ensemble_proba[1],
                'probability_bajista': ensemble_proba[0],
                'ensemble_details': predictions,
                'models_used': f"{successful_models}/{len(self.models)}",
                'consensus_ratio': consensus_ratio,
                'bullish_votes': bullish_votes,
                'bearish_votes': bearish_votes,
                'timestamp': datetime.now()
            }
            
            logger.info(f"🎯 DELOWYSS: Predicción completada - {result['prediction']} ({confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error en predicción ensemble: {e}")
            return None

    def _save_models(self):
        """Guardar modelos entrenados"""
        try:
            for name, model in self.models.items():
                filename = f'delowyss_{name}_model.pkl'
                joblib.dump(model, filename)
            
            joblib.dump(self.scalers['standard'], 'delowyss_scaler.pkl')
            logger.info("💾 DELOWYSS: Modelos guardados exitosamente")
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error guardando modelos: {e}")

    def load_models(self):
        """Cargar modelos previamente entrenados"""
        try:
            loaded_models = 0
            for name in self.models.keys():
                filename = f'delowyss_{name}_model.pkl'
                if os.path.exists(filename):
                    self.models[name] = joblib.load(filename)
                    loaded_models += 1
            
            if os.path.exists('delowyss_scaler.pkl'):
                self.scalers['standard'] = joblib.load('delowyss_scaler.pkl')
                loaded_models += 1
            
            if loaded_models > 0:
                self.model_trained = True
                logger.info(f"🔄 DELOWYSS: {loaded_models} modelos cargados")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ DELOWYSS: Error cargando modelos: {e}")
            return False

class DelowyssTradingAssistant:
    """ASISTENTE PREDILECTIVO DELOWYSS TRADING - VERSIÓN COMPLETA"""
    
    def __init__(self):
        self.api = DelowyssIQOptionAPI()
        self.ai_analyst = DelowyssAIAnalyst()
        self.pattern_detector = DelowyssPatternDetector()
        self.analysis_count = 0
        self.performance_history = []
        
        # Inicializar sistema
        self._initialize_system()
    
    def _initialize_system(self):
        """Inicializar sistema Delowyss completo"""
        print(f"\n{'🚀 DELOWYSS TRADING PROFESSIONAL ':^70}")
        print(f"{'='*70}")
        print("🤖 Asistente Predilectivo v4.0 - Sistema Completo")
        print("⚡ IA Avanzada + Patrones Ocultos + Conexión Real")
        print("🔗 IQ Option EUR/USD OTC - Temporalidad 1 Minuto")
        print(f"{'='*70}")
        
        # Conectar a IQ Option
        connection_success = self.api.connect()
        
        # Cargar modelos existentes
        models_loaded = self.ai_analyst.load_models()
        
        if connection_success:
            if self.api.connected:
                print("✅ CONEXIÓN: IQ Option REAL establecida")
            else:
                print("🔶 CONEXIÓN: Modo simulación avanzada activado")
            
            if models_loaded:
                print("✅ MODELOS: IA precargada lista")
            else:
                print("🔄 MODELOS: Entrenamiento requerido en primer análisis")
        
        print(f"{'='*70}")

    def perform_comprehensive_analysis(self):
        """Ejecutar análisis completo Delowyss"""
        self.analysis_count += 1
        start_time = time.time()
        
        print(f"\n{'🎯 DELOWYSS - ANÁLISIS PREDILECTIVO COMPLETO ':^70}")
        print(f"{'='*70}")
        
        # FASE 1: OBTENCIÓN DE DATOS
        print("\n📊 FASE 1: CONEXIÓN IQ OPTION Y DATOS...")
        df = self.api.get_real_time_data(count=200)
        if df is None:
            print("❌ ERROR CRÍTICO: No se pudieron obtener datos del mercado")
            return
        
        # FASE 2: ANÁLISIS TÉCNICO AVANZADO
        print("📈 FASE 2: ANÁLISIS TÉCNICO COMPLETO...")
        df = self.ai_analyst.calculate_comprehensive_indicators(df)
        
        # FASE 3: DETECCIÓN DE PATRONES OCULTOS
        print("🔍 FASE 3: DETECCIÓN PATRONES OCULTOS...")
        hidden_patterns = self.pattern_detector.detect_comprehensive_patterns(df)
        
        # FASE 4: PREDICCIÓN IA AVANZADA
        print("🧠 FASE 4: PREDICCIÓN IA DELOWYSS...")
        prediction = self.ai_analyst.ensemble_prediction(df)
        
        if prediction is None:
            print("❌ ERROR: No se pudo generar predicción")
            return
        
        # FASE 5: GENERACIÓN DE REPORTE
        analysis_duration = time.time() - start_time
        self._display_comprehensive_report(df, prediction, hidden_patterns, analysis_duration)
        
        # Guardar en historial
        self.performance_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'patterns': hidden_patterns,
            'duration': analysis_duration
        })

    def _display_comprehensive_report(self, df: pd.DataFrame, prediction: Dict, 
                                    patterns: Dict, duration: float):
        """Mostrar reporte completo Delowyss"""
        latest = df.iloc[-1]
        
        print(f"\n{'🎯 INFORME DELOWYSS TRADING - ANÁLISIS COMPLETO ':^70}")
        print(f"{'='*70}")
        
        # HEADER INFORMATIVO
        print(f"🏢 SISTEMA: Delowyss Trading Professional v4.0")
        print(f"💰 ACTIVO: EUR/USD OTC | ⏰ TEMPORALIDAD: 1 MINUTO")
        print(f"📅 HORA ANÁLISIS: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💵 PRECIO ACTUAL: {latest['close']:.5f}")
        print(f"⚡ VELOCIDAD: {duration:.2f}s | 🔢 ANÁLISIS: #{self.analysis_count}")
        
        # SECCIÓN PREDICCIÓN
        print(f"\n{'🔮 PREDICCIÓN IA DELOWYSS ':^70}")
        print(f"┌{'─'*68}┐")
        print(f"│ {'DIRECCIÓN:':<12} {prediction['prediction']:<18} {'CONFIANZA:':<12} {prediction['confidence']:.2%} │")
        print(f"│ {'PROB. ALCISTA:':<12} {prediction['probability_alcista']:.2%:<18} {'PROB. BAJISTA:':<12} {prediction['probability_bajista']:.2%} │")
        print(f"│ {'MODELOS:':<12} {prediction['models_used']:<18} {'CONSENSO:':<12} {prediction['consensus_ratio']:.1%} │")
        print(f"│ {'VOTOS ALCISTAS:':<12} {prediction['bullish_votes']:<18} {'VOTOS BAJISTAS:':<12} {prediction['bearish_votes']} │")
        print(f"└{'─'*68}┘")
        
        # SECCIÓN INDICADORES CLAVE
        print(f"\n{'📊 INDICADORES TÉCNICOS CLAVE ':^70}")
        key_indicators = [
            ("RSI 14", f"{latest['rsi_14']:.1f}", 
             "SOBREC" if latest['rsi_14'] > 70 else "SOBREV" if latest['rsi_14'] < 30 else "NEUTRO"),
            ("MACD", f"{latest['macd']:.6f}", 
             "ALCISTA" if latest['macd'] > latest['macd_signal'] else "BAJISTA"),
            ("ADX", f"{latest.get('adx', 25):.1f}", 
             "FUERTE" if latest.get('adx', 0) > 25 else "DEBIL"),
            ("BB POS", f"{latest['bb_position']:.2f}", 
             "SOBREC" if latest['bb_position'] > 0.8 else "SOBREV" if latest['bb_position'] < 0.2 else "NEUTRO"),
            ("ATR", f"{latest['atr']:.5f}", 
             "ALTA" if latest['atr'] > 0.0005 else "BAJA"),
            ("VOLATILIDAD", f"{latest['volatility_regime']:.2f}", 
             "ALTA" if latest['volatility_regime'] > 1.2 else "BAJA")
        ]
        
        for name, value, status in key_indicators:
            print(f"   {name:<15} {value:<12} [{status}]")
        
        # SECCIÓN PATRONES OCULTOS
        print(f"\n{'🔍 PATRONES OCULTOS DETECTADOS ':^70}")
        if patterns:
            significant_patterns = []
            for pattern, value in patterns.items():
                pattern_name = pattern.replace('_', ' ').upper()
                significance = "ALTA" if abs(value) > 0.2 else "MEDIA" if abs(value) > 0.1 else "BAJA"
                if significance != "BAJA":
                    significant_patterns.append((pattern_name, value, significance))
            
            if significant_patterns:
                for name, value, sig in significant_patterns[:6]:  # Mostrar máximo 6
                    print(f"   {name:<25} {value:>8.3f} [{sig}]")
            else:
                print("   ⚠️  No se detectaron patrones significativos")
        else:
            print("   ❌ No se pudieron analizar patrones")
        
        # RÉGIMEN DE MERCADO
        market_regime = patterns.get('market_regime', 'DESCONOCIDO') if patterns else 'DESCONOCIDO'
        print(f"\n{'🌡️  RÉGIMEN DE MERCADO ':^70}")
        print(f"   ESTADO ACTUAL: {market_regime}")
        
        # SECCIÓN RECOMENDACIÓN
        print(f"\n{'💡 RECOMENDACIÓN DELOWYSS ':^70}")
        confidence = prediction['confidence']
        market_condition = self._assess_market_condition(latest, patterns)
        
        recommendation = self._generate_trading_recommendation(
            prediction, market_condition, confidence
        )
        
        print(f"┌{'─'*68}┐")
        for line in recommendation:
            print(f"│ {line} │")
        print(f"└{'─'*68}┘")
        
        # FOOTER
        print(f"\n{'⚠️  DELOWYSS TRADING - GESTIÓN DE RIESGO ':^70}")
        print("    • Nunca invierta más del 5% de su capital en una operación")
        print("    • Use stop-loss y take-profit en todas sus operaciones")  
        print("    • Los análisis son educativos - Verifique siempre las señales")
        print("    • Delowyss Trading no se responsabiliza por pérdidas")
        print(f"{'='*70}")

    def _assess_market_condition(self, latest: pd.Series, patterns: Dict) -> Dict:
        """Evaluar condición completa del mercado"""
        condition = {
            'trend': 'NEUTRAL',
            'volatility': 'NORMAL', 
            'momentum': 'NEUTRAL',
            'risk': 'MODERADO'
        }
        
        # Evaluar tendencia
        if latest.get('adx', 0) > 25:
            if latest['ema_9'] > latest['ema_21']:
                condition['trend'] = 'FUERTE_ALCISTA'
            else:
                condition['trend'] = 'FUERTE_BAJISTA'
        elif latest['ema_9'] > latest['ema_21']:
            condition['trend'] = 'ALCISTA_SUAVE'
        elif latest['ema_9'] < latest['ema_21']:
            condition['trend'] = 'BAJISTA_SUAVE'
        
        # Evaluar volatilidad
        if latest['atr'] > 0.0006:
            condition['volatility'] = 'ALTA'
        elif latest['atr'] < 0.0002:
            condition['volatility'] = 'BAJA'
        
        # Evaluar momentum
        if latest['rsi_14'] > 70:
            condition['momentum'] = 'SOBRECOMPRADO'
        elif latest['rsi_14'] < 30:
            condition['momentum'] = 'SOBREVENDIDO'
        elif latest['macd'] > latest['macd_signal']:
            condition['momentum'] = 'ALCISTA'
        else:
            condition['momentum'] = 'BAJISTA'
        
        # Evaluar riesgo
        risk_factors = 0
        if condition['volatility'] == 'ALTA': risk_factors += 2
        if condition['momentum'] in ['SOBRECOMPRADO', 'SOBREVENDIDO']: risk_factors += 1
        if patterns.get('volatility_regime_change', 1) > 1.5: risk_factors += 1
        
        if risk_factors >= 3:
            condition['risk'] = 'ALTO'
        elif risk_factors >= 2:
            condition['risk'] = 'MODERADO_ALTO'
        elif risk_factors >= 1:
            condition['risk'] = 'MODERADO'
        else:
            condition['risk'] = 'BAJO'
        
        return condition

    def _generate_trading_recommendation(self, prediction: Dict, 
                                       market_condition: Dict, confidence: float) -> List[str]:
        """Generar recomendación de trading personalizada"""
        recommendation = []
        
        # Línea 1: Acción principal
        if confidence > 0.75:
            action = f"🎯 ACCIÓN: {prediction['prediction']} CONVICTORA"
            confidence_stars = "⭐⭐⭐⭐⭐"
        elif confidence > 0.65:
            action = f"🎯 ACCIÓN: {prediction['prediction']} MODERADA" 
            confidence_stars = "⭐⭐⭐⭐"
        else:
            action = "🎯 ACCIÓN: ESPERAR MEJOR SETUP"
            confidence_stars = "⭐⭐⭐"
        
        recommendation.append(f"{action:<45} {confidence_stars:>20}")
        
        # Línea 2: Tiempo y riesgo
        if market_condition['volatility'] == 'ALTA':
            expiry = "1-2 MINUTOS"
        elif market_condition['volatility'] == 'BAJA':
            expiry = "3-5 MINUTOS" 
        else:
            expiry = "2-3 MINUTOS"
        
        risk_emoji = "🔴" if market_condition['risk'] == 'ALTO' else "🟡" if 'MODERADO' in market_condition['risk'] else "🟢"
        recommendation.append(f"⏰ EXPIRACIÓN: {expiry:<38} RIESGO: {risk_emoji} {market_condition['risk']:>10}")
        
        # Línea 3: Inversión y potencial
        if confidence > 0.75 and market_condition['risk'] in ['BAJO', 'MODERADO']:
            investment = "3-5% CAPITAL"
            potential = "🚀 ALTO"
        elif confidence > 0.65:
            investment = "2-3% CAPITAL" 
            potential = "📈 MEDIO"
        else:
            investment = "0-1% CAPITAL"
            potential = "📊 BAJO"
        
        recommendation.append(f"💰 INVERSIÓN: {investment:<40} POTENCIAL: {potential:>12}")
        
        return recommendation

    def run_assistant(self):
        """Ejecutar el asistente Delowyss"""
        try:
            while True:
                print(f"\n{'🎯 COMANDO DELOWYSS ':^70}")
                print("    Comando disponible: A - Análisis completo del mercado")
                print("    Comando especial: Q - Salir del sistema Delowyss")
                print(f"{'─'*70}")
                
                command = input("\n🎯 Ingrese comando Delowyss (A/Q): ").upper().strip()
                
                if command == 'A':
                    self.perform_comprehensive_analysis()
                elif command == 'Q':
                    print(f"\n{'👋 DESCONECTANDO DELOWYSS TRADING ':^70}")
                    print("    ¡Gracias por usar Delowyss Trading Professional!")
                    print("    Recuerde: La disciplina es la clave del éxito.")
                    print(f"{'='*70}")
                    break
                else:
                    print("❌ Comando no reconocido. Use 'A' para analizar o 'Q' para salir.")
                    
        except KeyboardInterrupt:
            print(f"\n\n{'🛑 SISTEMA INTERRUMPIDO ':^70}")
            print("    Delowyss Trading se ha detenido de forma segura.")
        except Exception as e:
            print(f"\n{'❌ ERROR CRÍTICO DELOWYSS ':^70}")
            print(f"    Error: {e}")
            print("    Contacte al soporte técnico de Delowyss Trading.")

def main():
    """Función principal"""
    print(f"\n{'🚀 INICIANDO DELOWYSS TRADING PROFESSIONAL ':^70}")
    print(f"{'='*70}")
    
    try:
        assistant = DelowyssTradingAssistant()
        assistant.run_assistant()
    except Exception as e:
        print(f"\n{'💥 ERROR INICIAL DELOWYSS ':^70}")
        print(f"    Error: {e}")
        print("    Verifique la instalación y configuración.")

if __name__ == "__main__":
    main()
