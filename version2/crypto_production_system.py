import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")



# CONFIGURATION
@dataclass
class TradingConfig:
    """System configuration with sensible defaults"""
    
    # API Keys
    BINANCE_API_KEY: str = ""  # Optional for public endpoints
    BINANCE_SECRET: str = ""
    
    # Data Parameters
    TIMEFRAME: str = "1h"  # Candlestick interval: 1m, 5m, 15m, 1h, 4h, 1d
    LOOKBACK_PERIODS: int = 1000  # Historical candles to fetch (max allowed)
    SEQUENCE_LENGTH: int = 60  # Input sequence length (60 hours = 2.5 days)
    PREDICTION_HORIZON: int = 12  # Predict 12 hours ahead
    
    # Trading Pairs
    TRADING_PAIRS: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        'XRPUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'AVAXUSDT'
    ])
    
    # Model Parameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    
    # Risk Management
    INITIAL_CAPITAL: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    MAX_POSITION_SIZE: float = 0.25
    TRANSACTION_FEE: float = 0.001  # 0.1% per trade
    SLIPPAGE: float = 0.0005  # 0.05% slippage
    MIN_CONFIDENCE: float = 0.65
    
    # System
    UPDATE_INTERVAL: int = 3600  # Update every hour
    RETRAIN_INTERVAL: int = 7  # Retrain every 7 days
    LOG_LEVEL: str = "DEBUG"  # Set to DEBUG to see data processing details
    
    # Paths
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"
    BACKTEST_DIR: str = "backtests"
    
    def __post_init__(self):
        """Create directories"""
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR, self.BACKTEST_DIR]:
            os.makedirs(directory, exist_ok=True)



# LOGGING SETUP
def setup_logging(config: TradingConfig) -> logging.Logger:
    """Configure logging with proper handlers"""
    logger = logging.getLogger('CryptoTrader')
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_fmt)
    
    # File handler
    log_file = os.path.join(
        config.LOG_DIR,
        f'trading_{datetime.now():%Y%m%d_%H%M%S}.log'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_fmt)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger



# RATE LIMITER
class RateLimiter:
    """Thread-safe API rate limiter"""
    
    def __init__(self, calls_per_minute: int, logger: logging.Logger):
        self.calls_per_minute = calls_per_minute
        self.call_times = deque(maxlen=calls_per_minute)
        self.lock = threading.Lock()
        self.logger = logger
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        with self.lock:
            now = datetime.now()
            
            # Remove old timestamps
            while self.call_times and (now - self.call_times[0]) > timedelta(minutes=1):
                self.call_times.popleft()
            
            # Wait if at limit
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0]).total_seconds()
                if sleep_time > 0:
                    self.logger.warning(f"Rate limit reached. Waiting {sleep_time:.1f}s")
                    time.sleep(sleep_time)
            
            self.call_times.append(now)


# MARKET DATA FETCHER (BINANCE)
class BinanceDataFetcher:
    """Fetch real OHLCV candlestick data from Binance"""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.rate_limiter = RateLimiter(1200, logger)  # 1200 requests/min
        self.session = requests.Session()
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (max 1000)
        
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/klines",
                params={
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(limit, 1000)
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse klines
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Keep only OHLCV
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.set_index('timestamp')
            
            self.logger.debug(f"Fetched {len(df)} candles for {symbol}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple(self, symbols: List[str], interval: str, limit: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        data = {}
        
        for symbol in symbols:
            df = self.fetch_klines(symbol, interval, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.1)  # Be nice to API
        
        return data


# TECHNICAL INDICATORS (CORRECT IMPLEMENTATIONS)
class TechnicalIndicators:
    """Properly implemented technical indicators"""
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = [20, 50]) -> pd.DataFrame:
        """Simple Moving Average"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
        """Exponential Moving Average"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['atr'] = true_range.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index (PROPER implementation)"""
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = pd.Series(true_range).rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (smoothed DX)
        df['adx'] = dx.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator (PROPER implementation)"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators"""
        df = TechnicalIndicators.add_sma(df)
        df = TechnicalIndicators.add_ema(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        df = TechnicalIndicators.add_adx(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_obv(df)
        
        # Additional features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df



# DATA PROCESSOR (NO LEAKAGE)
class TimeSeriesDataProcessor:
    """Process time series data without leakage"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scalers = {}
    
    def create_labels(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Create labels for prediction
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Number of periods ahead to predict
        
        Returns:
            Series with labels: 0=SELL, 1=HOLD, 2=BUY
        """
        # Calculate future returns
        future_close = df['close'].shift(-horizon)
        future_return = (future_close - df['close']) / df['close']
        
        # Classify returns
        labels = pd.Series(1, index=df.index)  # Default HOLD
        labels[future_return > 0.02] = 2  # BUY (>2% gain)
        labels[future_return < -0.02] = 0  # SELL (<-2% loss)
        
        return labels
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int, 
                         horizon: int, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            df: DataFrame with features and OHLCV
            sequence_length: Length of input sequences
            horizon: Prediction horizon
            fit_scaler: Whether to fit scaler (True for train, False for test)
        
        Returns:
            X: (samples, sequence_length, features)
            y: (samples, 3) one-hot encoded labels
        """
        # Add technical indicators
        df_with_indicators = TechnicalIndicators.add_all_indicators(df.copy())
        
        # Drop NaN rows
        df_clean = df_with_indicators.dropna()
        
        # Debug logging
        self.logger.debug(f"Original data: {len(df)} rows")
        self.logger.debug(f"After indicators: {len(df_with_indicators)} rows")
        self.logger.debug(f"After dropna: {len(df_clean)} rows")
        self.logger.debug(f"Need minimum: {sequence_length + horizon} rows")
        
        if len(df_clean) < sequence_length + horizon:
            self.logger.warning(
                f"Insufficient data: {len(df_clean)} rows available, "
                f"need {sequence_length + horizon} (seq={sequence_length} + horizon={horizon})"
            )
            return np.array([]), np.array([])
        
        # Create labels
        labels = self.create_labels(df_clean, horizon)
        
        # Select features (exclude original OHLC to avoid redundancy)
        feature_cols = [col for col in df_clean.columns if col not in ['open', 'high', 'low']]
        features = df_clean[feature_cols].values
        
        self.logger.debug(f"Number of features: {len(feature_cols)}")
        
        # Fit or use existing scaler
        symbol = df.name if hasattr(df, 'name') else 'default'
        
        if fit_scaler:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers[symbol] = scaler
        else:
            scaler = self.scalers.get(symbol)
            if scaler is None:
                self.logger.error(f"No scaler found for {symbol}")
                return np.array([]), np.array([])
            features_scaled = scaler.transform(features)
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(features_scaled) - sequence_length - horizon):
            X.append(features_scaled[i:i + sequence_length])
            y.append(labels.iloc[i + sequence_length])
        
        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=3)
        
        self.logger.debug(f"Created {len(X)} sequences")
        
        return X, y
    
    def walk_forward_split(self, df: pd.DataFrame, n_splits: int = 5):
        """
        Create walk-forward validation splits
        
        Yields:
            (train_df, val_df) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for train_idx, val_idx in tscv.split(df):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            yield train_df, val_df


# MODEL ARCHITECTURES
def build_lstm_model(input_shape: Tuple[int, int], num_classes: int = 3) -> Model:
    """
    Build LSTM model with proper dimensions
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled model
    """
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def build_transformer_model(input_shape: Tuple[int, int], num_classes: int = 3) -> Model:
    """
    Build Transformer model with CORRECT dimensions
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled model
    """
    sequence_length, num_features = input_shape
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    
    inputs = layers.Input(shape=input_shape)
    
    # Project to embed_dim
    x = layers.Dense(embed_dim)(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embedding_layer = layers.Embedding(
        input_dim=sequence_length,
        output_dim=embed_dim
    )
    position_embeddings = position_embedding_layer(positions)
    position_embeddings = tf.expand_dims(position_embeddings, 0)
    
    x = x + position_embeddings
    
    # Transformer blocks
    for _ in range(2):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        ffn_output = ffn(x)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model



# ENSEMBLE MODEL
class EnsembleModel:
    """Ensemble of models with proper calibration"""
    
    def __init__(self, input_shape: Tuple[int, int], config: TradingConfig, logger: logging.Logger):
        self.input_shape = input_shape
        self.config = config
        self.logger = logger
        self.models = []
        self.weights = []
        self.is_trained = False
    
    def build(self):
        """Build ensemble models"""
        self.logger.info("Building ensemble models...")
        
        # LSTM model
        lstm_model = build_lstm_model(self.input_shape)
        self.models.append(('LSTM', lstm_model))
        
        # Transformer model
        transformer_model = build_transformer_model(self.input_shape)
        self.models.append(('Transformer', transformer_model))
        
        # Equal weights initially
        self.weights = [1.0 / len(self.models)] * len(self.models)
        
        self.logger.info(f"Built {len(self.models)} models")
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train all models"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        for name, model in self.models:
            self.logger.info(f"Training {name}...")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )
            
            best_val_acc = max(history.history['val_accuracy'])
            self.logger.info(f"{name} - Best Val Accuracy: {best_val_acc:.4f}")
        
        # Update weights based on validation performance
        self.update_weights(X_val, y_val)
        self.is_trained = True
    
    def update_weights(self, X_val, y_val):
        """Update ensemble weights based on validation accuracy"""
        accuracies = []
        
        for name, model in self.models:
            pred = model.predict(X_val, verbose=0)
            acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_val, axis=1))
            accuracies.append(acc)
            self.logger.debug(f"{name} validation accuracy: {acc:.4f}")
        
        # Normalize to weights
        total = sum(accuracies)
        if total > 0:
            self.weights = [acc / total for acc in accuracies]
        
        self.logger.info(f"Updated weights: {dict(zip([n for n, _ in self.models], self.weights))}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        predictions = []
        for name, model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def save(self, path: str):
        """Save all models"""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models:
            model_path = os.path.join(path, f"{name.lower()}.h5")
            model.save(model_path)
        
        # Save weights
        weights_path = os.path.join(path, "ensemble_weights.json")
        with open(weights_path, 'w') as f:
            json.dump({
                'weights': self.weights,
                'model_names': [name for name, _ in self.models]
            }, f)
        
        self.logger.info(f"Saved ensemble to {path}")
    
    def load(self, path: str):
        """Load all models"""
        self.models = []
        
        # Load weights
        weights_path = os.path.join(path, "ensemble_weights.json")
        with open(weights_path, 'r') as f:
            data = json.load(f)
            self.weights = data['weights']
            model_names = data['model_names']
        
        # Load models
        for name in model_names:
            model_path = os.path.join(path, f"{name.lower()}.h5")
            model = keras.models.load_model(model_path)
            self.models.append((name, model))
        
        self.is_trained = True
        self.logger.info(f"Loaded ensemble from {path}")



# BACKTESTER
@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    
    def close(self, exit_price: float, exit_time: datetime, fee_rate: float):
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # Calculate P&L
        if self.direction == 'LONG':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - exit_price) * self.size
        
        # Subtract fees
        self.fees = (self.entry_price * self.size * fee_rate) + (exit_price * self.size * fee_rate)
        self.pnl -= self.fees
        
        # Calculate percentage
        self.pnl_pct = (self.pnl / (self.entry_price * self.size)) * 100


class Backtester:
    """Backtest trading strategy with realistic costs"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.trades = []
        self.equity_curve = []
    
    def run(self, df: pd.DataFrame, predictions: np.ndarray, symbol: str) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            df: DataFrame with OHLCV data (must be aligned with predictions)
            predictions: Model predictions (samples, 3)
            symbol: Trading pair symbol
        
        Returns:
            Dictionary with performance metrics
        """
        capital = self.config.INITIAL_CAPITAL
        position = None
        
        equity = [capital]
        
        for i in range(len(predictions)):
            if i >= len(df):
                break
            
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            
            # Get prediction
            pred_class = np.argmax(predictions[i])
            confidence = predictions[i][pred_class]
            
            # Skip low confidence
            if confidence < self.config.MIN_CONFIDENCE:
                equity.append(capital)
                continue
            
            # Trading logic
            if pred_class == 2 and position is None:  # BUY signal, no position
                # Open long position
                position_size = min(
                    capital * self.config.RISK_PER_TRADE * confidence,
                    capital * self.config.MAX_POSITION_SIZE
                )
                
                # Account for slippage
                entry_price = price * (1 + self.config.SLIPPAGE)
                
                shares = position_size / entry_price
                
                position = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    symbol=symbol,
                    direction='LONG',
                    entry_price=entry_price,
                    exit_price=None,
                    size=shares
                )
                
                capital -= position_size
            
            elif pred_class == 0 and position is not None:  # SELL signal, have position
                # Close position
                exit_price = price * (1 - self.config.SLIPPAGE)
                
                position.close(
                    exit_price=exit_price,
                    exit_time=timestamp,
                    fee_rate=self.config.TRANSACTION_FEE
                )
                
                capital += (position.exit_price * position.size)
                capital += position.pnl
                
                self.trades.append(position)
                position = None
            
            # Update equity
            current_equity = capital
            if position:
                current_equity += position.size * price
            
            equity.append(current_equity)
        
        # Close any open position at end
        if position:
            exit_price = df['close'].iloc[-1] * (1 - self.config.SLIPPAGE)
            position.close(
                exit_price=exit_price,
                exit_time=df.index[-1],
                fee_rate=self.config.TRANSACTION_FEE
            )
            capital += (position.exit_price * position.size)
            capital += position.pnl
            self.trades.append(position)
        
        self.equity_curve = equity
        
        # Calculate metrics
        metrics = self.calculate_metrics(symbol)
        
        return metrics
    
    def calculate_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Trade statistics
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Returns
        total_return = (self.equity_curve[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL
        
        # Sharpe ratio
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        metrics = {
            'symbol': symbol,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'final_equity': self.equity_curve[-1]
        }
        
        return metrics
    
    def save_results(self, metrics: Dict[str, Any], filename: str):
        """Save backtest results"""
        filepath = os.path.join(self.config.BACKTEST_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Saved backtest results to {filepath}")

# MAIN TRADING SYSTEM
class CryptoTradingSystem:
    """Production-grade cryptocurrency trading system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logging(config)
        
        self.data_fetcher = BinanceDataFetcher(config, self.logger)
        self.data_processor = TimeSeriesDataProcessor(config, self.logger)
        
        self.ensemble = None
        self.last_train_time = None
        
        self.logger.info("=" * 60)
        self.logger.info("Crypto Trading System v6.0 - Production")
        self.logger.info("=" * 60)
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch current market data"""
        self.logger.info(f"Fetching data for {len(self.config.TRADING_PAIRS)} pairs...")
        
        data = self.data_fetcher.fetch_multiple(
            self.config.TRADING_PAIRS,
            self.config.TIMEFRAME,
            self.config.LOOKBACK_PERIODS
        )
        
        self.logger.info(f"Fetched data for {len(data)} pairs")
        
        return data
    
    def train_models(self, market_data: Dict[str, pd.DataFrame]):
        """Train models with walk-forward validation"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING MODELS")
        self.logger.info("=" * 60)
        
        # Combine data from all symbols for training
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        
        for symbol, df in market_data.items():
            self.logger.info(f"Processing {symbol}...")
            
            # Use walk-forward split
            for train_df, val_df in self.data_processor.walk_forward_split(df, n_splits=3):
                # Prepare train data (fit scaler)
                X_train, y_train = self.data_processor.prepare_sequences(
                    train_df,
                    self.config.SEQUENCE_LENGTH,
                    self.config.PREDICTION_HORIZON,
                    fit_scaler=True
                )
                
                if len(X_train) == 0:
                    continue
                
                # Prepare validation data (use fitted scaler)
                X_val, y_val = self.data_processor.prepare_sequences(
                    val_df,
                    self.config.SEQUENCE_LENGTH,
                    self.config.PREDICTION_HORIZON,
                    fit_scaler=False
                )
                
                if len(X_val) == 0:
                    continue
                
                all_X_train.append(X_train)
                all_y_train.append(y_train)
                all_X_val.append(X_val)
                all_y_val.append(y_val)
                
                # Use only first split for faster training
                break
        
        if not all_X_train:
            self.logger.error("No training data available")
            return
        
        # Combine all data
        X_train = np.vstack(all_X_train)
        y_train = np.vstack(all_y_train)
        X_val = np.vstack(all_X_val)
        y_val = np.vstack(all_y_val)
        
        self.logger.info(f"Training data: {X_train.shape}")
        self.logger.info(f"Validation data: {X_val.shape}")
        
        # Build and train ensemble
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.ensemble = EnsembleModel(input_shape, self.config, self.logger)
        self.ensemble.build()
        self.ensemble.train(X_train, y_train, X_val, y_val)
        
        # Save models
        model_path = os.path.join(self.config.MODEL_DIR, f"ensemble_{datetime.now():%Y%m%d_%H%M%S}")
        self.ensemble.save(model_path)
        
        self.last_train_time = datetime.now()
        
        self.logger.info("Model training complete")
    
    def backtest(self, market_data: Dict[str, pd.DataFrame]):
        """Run backtest on all symbols"""
        self.logger.info("=" * 60)
        self.logger.info("BACKTESTING")
        self.logger.info("=" * 60)
        
        if not self.ensemble or not self.ensemble.is_trained:
            self.logger.error("Models not trained")
            return
        
        all_metrics = []
        
        for symbol, df in market_data.items():
            self.logger.info(f"Backtesting {symbol}...")
            
            # Prepare data
            X, y = self.data_processor.prepare_sequences(
                df,
                self.config.SEQUENCE_LENGTH,
                self.config.PREDICTION_HORIZON,
                fit_scaler=False
            )
            
            if len(X) == 0:
                continue
            
            # Get predictions
            predictions = self.ensemble.predict(X)
            
            # Align dataframe with predictions
            df_aligned = df.iloc[self.config.SEQUENCE_LENGTH:self.config.SEQUENCE_LENGTH + len(predictions)]
            
            # Run backtest
            backtester = Backtester(self.config, self.logger)
            metrics = backtester.run(df_aligned, predictions, symbol)
            
            all_metrics.append(metrics)
            
            # Log results
            self.logger.info(f"{symbol} Results:")
            self.logger.info(f"  Trades: {metrics['total_trades']}")
            self.logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
            self.logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Save aggregate results
        if all_metrics:
            summary_file = f"backtest_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
            backtester.save_results({
                'timestamp': datetime.now().isoformat(),
                'config': asdict(self.config),
                'results': all_metrics
            }, summary_file)
        
        self.logger.info("Backtesting complete")
    
    def run(self):
        """Main execution loop"""
        try:
            # Fetch initial data
            market_data = self.fetch_data()
            
            if not market_data:
                self.logger.error("No market data available")
                return
            
            # Train models
            self.train_models(market_data)
            
            # Run backtest
            self.backtest(market_data)
            
            # Main loop
            while True:
                self.logger.info(f"\nWaiting {self.config.UPDATE_INTERVAL}s for next update...")
                time.sleep(self.config.UPDATE_INTERVAL)
                
                # Fetch new data
                market_data = self.fetch_data()
                
                # Check if retrain needed
                if self.last_train_time:
                    days_since_train = (datetime.now() - self.last_train_time).days
                    if days_since_train >= self.config.RETRAIN_INTERVAL:
                        self.logger.info("Retraining models...")
                        self.train_models(market_data)
                
                # Generate current signals
                self.generate_signals(market_data)
        
        except KeyboardInterrupt:
            self.logger.info("\nShutdown requested by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.logger.info("System shutdown complete")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]):
        """Generate trading signals for current market"""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING SIGNALS")
        self.logger.info("=" * 60)
        
        if not self.ensemble or not self.ensemble.is_trained:
            self.logger.warning("Models not trained, skipping signals")
            return
        
        for symbol, df in market_data.items():
            # Get latest sequence
            X, _ = self.data_processor.prepare_sequences(
                df.tail(self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON),
                self.config.SEQUENCE_LENGTH,
                self.config.PREDICTION_HORIZON,
                fit_scaler=False
            )
            
            if len(X) == 0:
                continue
            
            # Predict
            prediction = self.ensemble.predict(X[-1:])
            pred_class = np.argmax(prediction[0])
            confidence = prediction[0][pred_class]
            
            actions = ['SELL', 'HOLD', 'BUY']
            action = actions[pred_class]
            
            current_price = df['close'].iloc[-1]
            
            if confidence >= self.config.MIN_CONFIDENCE:
                self.logger.info(f"{symbol}: {action} (Confidence: {confidence:.2%}, Price: ${current_price:,.2f})")
            else:
                self.logger.debug(f"{symbol}: Low confidence ({confidence:.2%}), skipping")


# PERFORMANCE VISUALIZATION
class PerformanceVisualizer:
    """Visualize trading performance and model metrics"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def plot_equity_curve(self, equity_curve: List[float], symbol: str, save: bool = True):
        """Plot equity curve over time"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve, linewidth=2)
            plt.title(f'Equity Curve - {symbol}', fontsize=14, fontweight='bold')
            plt.xlabel('Trade Number')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
            max_equity = max(equity_curve)
            min_equity = min(equity_curve)
            
            stats_text = f'Total Return: {total_return:.2f}%\n'
            stats_text += f'Max: ${max_equity:,.2f}\n'
            stats_text += f'Min: ${min_equity:,.2f}'
            
            plt.text(0.02, 0.98, stats_text, 
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save:
                filename = os.path.join(self.config.BACKTEST_DIR, f'equity_curve_{symbol}.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                return filename
            else:
                plt.show()
        
        except ImportError:
            pass  # matplotlib not available
    
    def plot_trade_analysis(self, trades: List[Trade], symbol: str):
        """Plot trade analysis charts"""
        try:
            import matplotlib.pyplot as plt
            
            if not trades:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # P&L Distribution
            pnls = [t.pnl_pct for t in trades]
            axes[0, 0].hist(pnls, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('P&L Distribution (%)')
            axes[0, 0].set_xlabel('P&L %')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            # Cumulative P&L
            cumulative_pnl = np.cumsum([t.pnl for t in trades])
            axes[0, 1].plot(cumulative_pnl, linewidth=2)
            axes[0, 1].set_title('Cumulative P&L ($)')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Cumulative P&L ($)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Win/Loss Streaks
            wins = [1 if t.pnl > 0 else 0 for t in trades]
            axes[1, 0].bar(range(len(wins)), wins, color=['green' if w else 'red' for w in wins])
            axes[1, 0].set_title('Win/Loss Pattern')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('Win (1) / Loss (0)')
            
            # Trade Duration
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 
                        for t in trades if t.exit_time]
            axes[1, 1].hist(durations, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Trade Duration (hours)')
            axes[1, 1].set_xlabel('Duration (hours)')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            filename = os.path.join(self.config.BACKTEST_DIR, f'trade_analysis_{symbol}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
        
        except ImportError:
            pass



# MODEL EVALUATION & METRICS
class ModelEvaluator:
    """Evaluate model performance with detailed metrics"""
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        class_names: List[str] = ['SELL', 'HOLD', 'BUY']) -> Dict:
        """Calculate confusion matrix and classification metrics"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Convert one-hot to class indices
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Classification report
        report = classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'overall_accuracy': report['accuracy']
        }
    
    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        y_classes = np.argmax(y, axis=1)
        class_counts = np.bincount(y_classes)
        total = len(y_classes)
        
        weights = {i: total / (len(class_counts) * count) 
                  for i, count in enumerate(class_counts)}
        
        return weights


# ADVANCED BACKTESTING WITH DETAILED REPORTING
class AdvancedBacktester(Backtester):
    """Extended backtester with additional analytics"""
    
    def calculate_advanced_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        base_metrics = self.calculate_metrics(symbol)
        
        if not self.trades:
            return base_metrics
        
        # Win/Loss streaks
        streaks = self._calculate_streaks()
        
        # Recovery time
        recovery_time = self._calculate_recovery_time()
        
        # Risk-adjusted returns
        sortino_ratio = self._calculate_sortino_ratio()
        calmar_ratio = self._calculate_calmar_ratio()
        
        # Trade quality
        expectancy = self._calculate_expectancy()
        
        advanced_metrics = {
            **base_metrics,
            'max_consecutive_wins': streaks['max_wins'],
            'max_consecutive_losses': streaks['max_losses'],
            'avg_recovery_time_hours': recovery_time,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'expectancy': expectancy,
            'total_fees_paid': sum(t.fees for t in self.trades),
            'avg_trade_duration_hours': np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 3600 
                for t in self.trades if t.exit_time
            ])
        }
        
        return advanced_metrics
    
    def _calculate_streaks(self) -> Dict[str, int]:
        """Calculate win/loss streaks"""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {'max_wins': max_wins, 'max_losses': max_losses}
    
    def _calculate_recovery_time(self) -> float:
        """Calculate average time to recover from drawdown"""
        peak = self.config.INITIAL_CAPITAL
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, equity in enumerate(self.equity_curve):
            if equity > peak:
                if in_drawdown and drawdown_start is not None:
                    recovery_times.append(i - drawdown_start)
                    in_drawdown = False
                peak = equity
            elif equity < peak and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)"""
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        if downside_std == 0:
            return 0
        
        return mean_return / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        total_return = (self.equity_curve[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL
        
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        if max_drawdown == 0:
            return 0
        
        return total_return / max_drawdown
    
    def _calculate_expectancy(self) -> float:
        """Calculate trade expectancy"""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        
        if not self.trades:
            return 0
        
        win_rate = len(wins) / len(self.trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return expectancy



# LIVE TRADING SIMULATOR (PAPER TRADING)
class LiveTradingSimulator:
    """Simulate live trading with paper money"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.capital = config.INITIAL_CAPITAL
        self.positions = {}
        self.trade_history = []
        self.equity_history = [config.INITIAL_CAPITAL]
        
    def process_signal(self, symbol: str, action: str, confidence: float, 
                      current_price: float, timestamp: datetime) -> Optional[Trade]:
        """Process a trading signal"""
        
        if confidence < self.config.MIN_CONFIDENCE:
            self.logger.debug(f"{symbol}: Signal filtered (low confidence: {confidence:.2%})")
            return None
        
        # Close existing position on opposite signal
        if symbol in self.positions:
            if (action == 'SELL' and self.positions[symbol].direction == 'LONG'):
                return self._close_position(symbol, current_price, timestamp)
            elif (action == 'BUY' and self.positions[symbol].direction == 'SHORT'):
                return self._close_position(symbol, current_price, timestamp)
        
        # Open new position
        if action == 'BUY' and symbol not in self.positions:
            return self._open_position(symbol, 'LONG', confidence, current_price, timestamp)
        elif action == 'SELL' and symbol not in self.positions:
            # For crypto spot trading, we typically don't short, so skip
            self.logger.debug(f"{symbol}: SELL signal but can't short in spot market")
            return None
        
        return None
    
    def _open_position(self, symbol: str, direction: str, confidence: float,
                      price: float, timestamp: datetime) -> Trade:
        """Open a new position"""
        position_size = min(
            self.capital * self.config.RISK_PER_TRADE * confidence,
            self.capital * self.config.MAX_POSITION_SIZE
        )
        
        # Apply slippage
        entry_price = price * (1 + self.config.SLIPPAGE if direction == 'LONG' else 1 - self.config.SLIPPAGE)
        
        shares = position_size / entry_price
        
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            size=shares
        )
        
        self.positions[symbol] = trade
        self.capital -= position_size
        
        self.logger.info(f"OPENED {direction} {symbol} @ ${entry_price:,.2f} | Size: {shares:.6f} | Confidence: {confidence:.2%}")
        
        return trade
    
    def _close_position(self, symbol: str, price: float, timestamp: datetime) -> Trade:
        """Close an existing position"""
        position = self.positions.pop(symbol)
        
        # Apply slippage
        exit_price = price * (1 - self.config.SLIPPAGE if position.direction == 'LONG' else 1 + self.config.SLIPPAGE)
        
        position.close(exit_price, timestamp, self.config.TRANSACTION_FEE)
        
        self.capital += (position.exit_price * position.size) + position.pnl
        self.trade_history.append(position)
        
        self.logger.info(
            f"CLOSED {position.direction} {symbol} @ ${exit_price:,.2f} | "
            f"P&L: ${position.pnl:,.2f} ({position.pnl_pct:+.2f}%) | "
            f"Duration: {(position.exit_time - position.entry_time).total_seconds() / 3600:.1f}h"
        )
        
        return position
    
    def update_equity(self, current_prices: Dict[str, float]):
        """Update equity based on current market prices"""
        equity = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                equity += position.size * current_prices[symbol]
        
        self.equity_history.append(equity)
        
        return equity
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'current_capital': self.capital,
                'total_return': 0.0
            }
        
        wins = [t for t in self.trade_history if t.pnl > 0]
        
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(wins),
            'win_rate': len(wins) / len(self.trade_history) * 100,
            'total_pnl': sum(t.pnl for t in self.trade_history),
            'total_return': (self.equity_history[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL * 100,
            'current_capital': self.capital,
            'current_equity': self.equity_history[-1],
            'open_positions': len(self.positions),
            'max_equity': max(self.equity_history),
            'min_equity': min(self.equity_history)
        }



# ENHANCED TRADING SYSTEM
class EnhancedCryptoTradingSystem(CryptoTradingSystem):
    """Enhanced system with visualization and live simulation"""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.visualizer = PerformanceVisualizer(config)
        self.live_simulator = None
    
    def backtest(self, market_data: Dict[str, pd.DataFrame]):
        """Enhanced backtest with visualization"""
        self.logger.info("=" * 60)
        self.logger.info("BACKTESTING")
        self.logger.info("=" * 60)
        
        if not self.ensemble or not self.ensemble.is_trained:
            self.logger.error("Models not trained")
            return
        
        all_metrics = []
        
        for symbol, df in market_data.items():
            self.logger.info(f"Backtesting {symbol}...")
            
            # Prepare data
            X, y = self.data_processor.prepare_sequences(
                df,
                self.config.SEQUENCE_LENGTH,
                self.config.PREDICTION_HORIZON,
                fit_scaler=False
            )
            
            if len(X) == 0:
                continue
            
            # Get predictions
            predictions = self.ensemble.predict(X)
            
            # Calculate confusion matrix
            eval_metrics = ModelEvaluator.confusion_matrix(y, predictions)
            self.logger.info(f"{symbol} Model Accuracy: {eval_metrics['overall_accuracy']:.4f}")
            
            # Align dataframe with predictions
            df_aligned = df.iloc[self.config.SEQUENCE_LENGTH:self.config.SEQUENCE_LENGTH + len(predictions)]
            
            # Run advanced backtest
            backtester = AdvancedBacktester(self.config, self.logger)
            metrics = backtester.calculate_advanced_metrics(symbol)
            metrics['model_metrics'] = eval_metrics
            
            all_metrics.append(metrics)
            
            # Log results
            self.logger.info(f"{symbol} Results:")
            self.logger.info(f"  Trades: {metrics['total_trades']}")
            self.logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
            self.logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
            self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            self.logger.info(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            self.logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
            self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
            self.logger.info(f"  Expectancy: ${metrics['expectancy']:.2f}")
            
            # Visualize
            self.visualizer.plot_equity_curve(backtester.equity_curve, symbol)
            self.visualizer.plot_trade_analysis(backtester.trades, symbol)
        
        # Save aggregate results
        if all_metrics:
            summary_file = f"backtest_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(all_metrics)
            
            backtester.save_results({
                'timestamp': datetime.now().isoformat(),
                'config': asdict(self.config),
                'individual_results': all_metrics,
                'portfolio_metrics': portfolio_metrics
            }, summary_file)
        
        self.logger.info("Backtesting complete")
    
    def _calculate_portfolio_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Calculate portfolio-level performance metrics"""
        total_trades = sum(m['total_trades'] for m in all_metrics)
        total_winning = sum(m['winning_trades'] for m in all_metrics)
        
        avg_return = np.mean([m['total_return'] for m in all_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])
        worst_drawdown = min([m['max_drawdown'] for m in all_metrics])
        
        return {
            'total_symbols': len(all_metrics),
            'total_trades': total_trades,
            'portfolio_win_rate': (total_winning / total_trades * 100) if total_trades > 0 else 0,
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'worst_drawdown': worst_drawdown,
            'best_performer': max(all_metrics, key=lambda x: x['total_return'])['symbol'],
            'worst_performer': min(all_metrics, key=lambda x: x['total_return'])['symbol']
        }
    
    def run_live_simulation(self):
        """Run live paper trading simulation"""
        self.logger.info("=" * 60)
        self.logger.info("LIVE PAPER TRADING SIMULATION")
        self.logger.info("=" * 60)
        
        if not self.ensemble or not self.ensemble.is_trained:
            self.logger.error("Models not trained")
            return
        
        self.live_simulator = LiveTradingSimulator(self.config, self.logger)
        
        try:
            cycle = 0
            while True:
                cycle += 1
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"LIVE CYCLE #{cycle}")
                self.logger.info(f"{'='*60}")
                
                # Fetch current data
                market_data = self.fetch_data()
                
                current_prices = {}
                
                # Generate signals for each symbol
                for symbol, df in market_data.items():
                    current_price = df['close'].iloc[-1]
                    current_prices[symbol] = current_price
                    
                    # Get latest sequence
                    X, _ = self.data_processor.prepare_sequences(
                        df.tail(self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON),
                        self.config.SEQUENCE_LENGTH,
                        self.config.PREDICTION_HORIZON,
                        fit_scaler=False
                    )
                    
                    if len(X) == 0:
                        continue
                    
                    # Predict
                    prediction = self.ensemble.predict(X[-1:])
                    pred_class = np.argmax(prediction[0])
                    confidence = prediction[0][pred_class]
                    
                    actions = ['SELL', 'HOLD', 'BUY']
                    action = actions[pred_class]
                    
                    # Process signal
                    self.live_simulator.process_signal(
                        symbol, action, confidence, current_price, datetime.now()
                    )
                
                # Update equity
                current_equity = self.live_simulator.update_equity(current_prices)
                
                # Display performance
                perf = self.live_simulator.get_performance_summary()
                self.logger.info(f"\n📊 PERFORMANCE SUMMARY:")
                self.logger.info(f"  Current Equity: ${current_equity:,.2f}")
                self.logger.info(f"  Total Return: {perf['total_return']:+.2f}%")
                self.logger.info(f"  Total Trades: {perf['total_trades']}")
                self.logger.info(f"  Win Rate: {perf['win_rate']:.2f}%")
                self.logger.info(f"  Open Positions: {perf['open_positions']}")
                
                # Check if retrain needed
                if self.last_train_time:
                    days_since_train = (datetime.now() - self.last_train_time).days
                    if days_since_train >= self.config.RETRAIN_INTERVAL:
                        self.logger.info("Retraining models...")
                        self.train_models(market_data)
                
                # Wait for next update
                self.logger.info(f"\n⏳ Waiting {self.config.UPDATE_INTERVAL}s for next update...")
                time.sleep(self.config.UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Stopping live simulation...")
            
            # Final report
            final_perf = self.live_simulator.get_performance_summary()
            self.logger.info("\n" + "=" * 60)
            self.logger.info("FINAL PERFORMANCE REPORT")
            self.logger.info("=" * 60)
            for key, value in final_perf.items():
                self.logger.info(f"  {key}: {value}")



# COMMAND LINE INTERFACE
def main():
    """Main entry point with CLI options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production-Grade Cryptocurrency Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crypto_production_system.py --mode backtest
  python crypto_production_system.py --mode live --pairs BTCUSDT ETHUSDT
  python crypto_production_system.py --mode backtest --timeframe 4h --epochs 50
        """
    )
    
    parser.add_argument('--mode', choices=['backtest', 'live', 'train'], 
                       default='backtest',
                       help='Operating mode (default: backtest)')
    
    parser.add_argument('--pairs', nargs='+', 
                       help='Trading pairs to analyze (e.g., BTCUSDT ETHUSDT)')
    
    parser.add_argument('--timeframe', default='1h',
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Candlestick timeframe (default: 1h)')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital (default: 10000)')
    
    parser.add_argument('--min-confidence', type=float, default=0.65,
                       help='Minimum confidence threshold (default: 0.65)')
    
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                       help='Risk per trade as decimal (default: 0.02)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TradingConfig()
    
    # Apply CLI arguments
    if args.pairs:
        config.TRADING_PAIRS = args.pairs
    
    config.TIMEFRAME = args.timeframe
    config.EPOCHS = args.epochs
    config.INITIAL_CAPITAL = args.capital
    config.MIN_CONFIDENCE = args.min_confidence
    config.RISK_PER_TRADE = args.risk_per_trade
    
    # Print banner
    print("\n" + "=" * 70)
    print("PRODUCTION CRYPTOCURRENCY TRADING SYSTEM v6.0")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Pairs: {', '.join(config.TRADING_PAIRS)}")
    print(f"  Timeframe: {config.TIMEFRAME}")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Min Confidence: {config.MIN_CONFIDENCE:.0%}")
    print(f"\n⚠️  IMPORTANT NOTES:")
    print("  1. Uses REAL OHLCV candlestick data from Binance")
    print("  2. Walk-forward validation prevents data leakage")
    print("  3. Includes realistic transaction costs and slippage")
    print("  4. Proper time series handling and backtesting")
    print("  5. No API keys needed for public market data")
    print(f"\n⚠️  DISCLAIMER:")
    print("  This is for educational purposes only.")
    print("  Cryptocurrency trading involves substantial risk.")
    print("  Past performance does not guarantee future results.")
    print("=" * 70 + "\n")
    
    # Create and run system
    try:
        system = EnhancedCryptoTradingSystem(config)
        
        if args.mode == 'backtest':
            # Fetch data and run backtest
            market_data = system.fetch_data()
            if market_data:
                system.train_models(market_data)
                system.backtest(market_data)
        
        elif args.mode == 'live':
            # Run live paper trading simulation
            market_data = system.fetch_data()
            if market_data:
                system.train_models(market_data)
                system.run_live_simulation()
        
        elif args.mode == 'train':
            # Only train models
            market_data = system.fetch_data()
            if market_data:
                system.train_models(market_data)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        print("💾 All state has been saved")
    
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0



# ENTRY POINT
if __name__ == "__main__":
    sys.exit(main())
