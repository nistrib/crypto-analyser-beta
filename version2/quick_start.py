"""
Quick Start Script - Minimal Configuration

This script uses conservative settings to ensure the system runs successfully
with limited data.
"""

from crypto_production_system import TradingConfig, EnhancedCryptoTradingSystem
import sys

def main():
    # Create configuration with MINIMAL requirements
    config = TradingConfig()
    
    # Fewer pairs to start
    config.TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Larger timeframe for more stable data
    config.TIMEFRAME = '4h'  # 4-hour candles instead of 1-hour
    
    # Fetch maximum data
    config.LOOKBACK_PERIODS = 1000  # Maximum from Binance
    
    # Reduce sequence requirements
    config.SEQUENCE_LENGTH = 30  # 30 candles = 5 days of 4h data
    config.PREDICTION_HORIZON = 6  # Predict 6 candles ahead (24 hours for 4h)
    
    # Faster training for testing
    config.EPOCHS = 20  # Just 20 epochs for quick test
    config.BATCH_SIZE = 16
    
    # Less strict confidence
    config.MIN_CONFIDENCE = 0.60
    
    print("\n" + "=" * 70)
    print("QUICK START - MINIMAL CONFIGURATION")
    print("=" * 70)
    print(f"\n📊 Settings:")
    print(f"  Pairs: {', '.join(config.TRADING_PAIRS)}")
    print(f"  Timeframe: {config.TIMEFRAME}")
    print(f"  Sequence Length: {config.SEQUENCE_LENGTH}")
    print(f"  Prediction Horizon: {config.PREDICTION_HORIZON}")
    print(f"  Training Epochs: {config.EPOCHS}")
    print(f"\n🎯 This configuration is optimized to work with limited data.")
    print("=" * 70 + "\n")
    
    try:
        # Create and run system
        system = EnhancedCryptoTradingSystem(config)
        
        # Fetch data
        print("Fetching market data...")
        market_data = system.fetch_data()
        
        if not market_data:
            print("❌ Failed to fetch data from Binance")
            print("   Check your internet connection")
            return 1
        
        print(f"✅ Fetched data for {len(market_data)} pairs\n")
        
        # Show data info
        for symbol, df in market_data.items():
            print(f"  {symbol}: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Train models
        print("\n🤖 Training models (this may take a few minutes)...")
        system.train_models(market_data)
        
        # Run backtest
        print("\n📊 Running backtest...")
        system.backtest(market_data)
        
        print("\n✅ Quick start complete!")
        print("📁 Check the following directories:")
        print(f"   - {config.LOG_DIR}/ for logs")
        print(f"   - {config.MODEL_DIR}/ for saved models")
        print(f"   - {config.BACKTEST_DIR}/ for backtest results")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        return 0
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
