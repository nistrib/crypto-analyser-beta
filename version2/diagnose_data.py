"""
Data Diagnostics Script

This script helps diagnose data fetching and processing issues.
"""

from crypto_production_system import (
    TradingConfig, 
    BinanceDataFetcher, 
    TechnicalIndicators,
    setup_logging
)
import pandas as pd

def diagnose_data():
    """Run diagnostics on data fetching and processing"""
    
    config = TradingConfig()
    logger = setup_logging(config)
    
    print("\n" + "=" * 70)
    print("CRYPTOCURRENCY DATA DIAGNOSTICS")
    print("=" * 70)
    
    # Test 1: Fetch data from Binance
    print("\n📡 TEST 1: Fetching data from Binance...")
    print("-" * 70)
    
    fetcher = BinanceDataFetcher(config, logger)
    
    test_symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        
        # Try different timeframes
        for timeframe in ['1h', '4h', '1d']:
            df = fetcher.fetch_klines(symbol, timeframe, limit=100)
            
            if df.empty:
                print(f"  ❌ {timeframe}: Failed to fetch data")
            else:
                print(f"  ✅ {timeframe}: {len(df)} candles")
                print(f"     Date range: {df.index[0]} to {df.index[-1]}")
                print(f"     Columns: {', '.join(df.columns)}")
    
    # Test 2: Check OHLCV validity
    print("\n\n📊 TEST 2: Checking OHLCV data validity...")
    print("-" * 70)
    
    df = fetcher.fetch_klines('BTCUSDT', '1h', 100)
    
    if not df.empty:
        print(f"\nData shape: {df.shape}")
        print(f"No NaN values: {not df.isnull().any().any()}")
        print(f"High >= Close: {(df['high'] >= df['close']).all()}")
        print(f"Low <= Close: {(df['low'] <= df['close']).all()}")
        print(f"High >= Low: {(df['high'] >= df['low']).all()}")
        
        print(f"\nPrice statistics:")
        print(f"  Current price: ${df['close'].iloc[-1]:,.2f}")
        print(f"  Max price: ${df['high'].max():,.2f}")
        print(f"  Min price: ${df['low'].min():,.2f}")
        print(f"  Avg volume: {df['volume'].mean():,.0f}")
    
    # Test 3: Technical indicators
    print("\n\n📈 TEST 3: Testing technical indicators...")
    print("-" * 70)
    
    if not df.empty:
        df_with_indicators = TechnicalIndicators.add_all_indicators(df.copy())
        
        print(f"\nOriginal columns: {len(df.columns)}")
        print(f"After indicators: {len(df_with_indicators.columns)}")
        print(f"Added indicators: {len(df_with_indicators.columns) - len(df.columns)}")
        
        # Check for NaN values
        print(f"\nNaN values per column:")
        nan_counts = df_with_indicators.isnull().sum()
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count} NaNs ({count/len(df)*100:.1f}%)")
        
        # After dropping NaNs
        df_clean = df_with_indicators.dropna()
        print(f"\nRows after dropna: {len(df_clean)} (lost {len(df) - len(df_clean)} rows)")
        
        if len(df_clean) > 0:
            print(f"\nSample indicators (latest values):")
            print(f"  RSI: {df_clean['rsi'].iloc[-1]:.2f}")
            print(f"  MACD: {df_clean['macd'].iloc[-1]:.4f}")
            print(f"  ATR: {df_clean['atr'].iloc[-1]:.4f}")
            print(f"  Bollinger Width: {df_clean['bb_width'].iloc[-1]:.4f}")
    
    # Test 4: Sequence creation
    print("\n\n🔢 TEST 4: Testing sequence creation...")
    print("-" * 70)
    
    test_configs = [
        {'seq': 30, 'horizon': 6, 'name': 'Conservative'},
        {'seq': 60, 'horizon': 12, 'name': 'Standard'},
        {'seq': 168, 'horizon': 24, 'name': 'Aggressive'}
    ]
    
    for test in test_configs:
        seq_len = test['seq']
        horizon = test['horizon']
        
        required_rows = seq_len + horizon
        available_rows = len(df_clean) if not df.empty else 0
        
        status = "✅" if available_rows >= required_rows else "❌"
        
        print(f"\n{status} {test['name']} (seq={seq_len}, horizon={horizon}):")
        print(f"   Requires: {required_rows} rows")
        print(f"   Available: {available_rows} rows")
        
        if available_rows >= required_rows:
            possible_sequences = available_rows - required_rows
            print(f"   Can create: ~{possible_sequences} sequences")
        else:
            shortage = required_rows - available_rows
            print(f"   Shortage: {shortage} rows")
    
    # Recommendations
    print("\n\n💡 RECOMMENDATIONS:")
    print("=" * 70)
    
    if df.empty:
        print("❌ Cannot fetch data from Binance")
        print("   - Check internet connection")
        print("   - Verify Binance API is accessible")
        print("   - Try: curl https://api.binance.com/api/v3/ping")
    
    elif len(df_clean) < 100:
        print("⚠️  Insufficient data after adding indicators")
        print("   - Use 4h or 1d timeframe instead of 1h")
        print("   - Increase LOOKBACK_PERIODS to 1000 (max)")
        print("   - Reduce number of indicators")
    
    elif len(df_clean) < 200:
        print("⚠️  Limited data for training")
        print("   - Use SEQUENCE_LENGTH=30 or less")
        print("   - Use PREDICTION_HORIZON=6 or less")
        print("   - Consider 4h timeframe for more data")
    
    else:
        print("✅ Data looks good!")
        print("   - You have sufficient data for training")
        print("   - Can use standard configuration")
        print("   - Ready to run backtest")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    diagnose_data()
