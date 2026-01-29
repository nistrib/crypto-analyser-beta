# Cryptocurrency Trading Analysis Systems 🚀

**Advanced AI-Powered Cryptocurrency Analysis & Trading Bots**  
*⚠️ BETA VERSION - For Educational and Research Purposes Only*

---

## 📋 Overview

This repository contains **two cryptocurrency trading analysis systems** with different approaches and capabilities:

1. **Version 2.0 (Production System)** - Advanced ML-based system with proper validation and backtesting
2. **Version 1.0 (Original Bot)** - AI-powered analysis bot with DeepSeek integration

Both systems are currently in **BETA** and are designed for educational purposes, backtesting, and paper trading only.

---

# 🆕 Version 2.0 - Production-Grade ML System

**Location:** `version2/`

A professionally implemented cryptocurrency trading system featuring state-of-the-art machine learning models, proper time series validation, and realistic backtesting.

## ✨ Key Features (v2.0)

### 🤖 Advanced Machine Learning
- **Transformer Models**: Attention-based architecture for time series prediction
- **LSTM Networks**: Deep learning with attention mechanisms
- **CNN-LSTM Hybrid**: Pattern recognition + temporal analysis
- **Ensemble Learning**: Combines multiple models weighted by validation accuracy
- **Walk-Forward Validation**: Prevents data leakage with proper time series splits

### 📊 Professional Technical Analysis
- **50+ Technical Indicators**:
  - Moving Averages (SMA, EMA)
  - Momentum (RSI, MACD, Stochastic)
  - Volatility (Bollinger Bands, ATR, ADX)
  - Volume (OBV, Volume Ratio)
  - All indicators correctly implemented and tested

### 💰 Realistic Backtesting
- Transaction fees (0.1% per trade)
- Market slippage (0.05%)
- Position sizing with risk limits
- Maximum drawdown monitoring
- Complete trade lifecycle tracking
- Advanced metrics (Sharpe, Sortino, Calmar ratios)

### 📈 Real Market Data
- **Binance API Integration**: Real OHLCV candlestick data
- **No API Keys Required**: Uses public endpoints
- **Multiple Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **Proper OHLC**: Actual high/low ranges (not synthetic)

### 🧪 Quality Assurance
- **24 Comprehensive Tests**: Data quality, indicator validation, leakage detection
- **Performance Visualization**: Auto-generated equity curves and trade analysis
- **Live Paper Trading**: Real-time simulation with paper money
- **Model Persistence**: Save/load trained models

## 🚀 Quick Start (v2.0)

### Installation

```bash
cd version2/
pip install -r requirements_production.txt
```

### Three Ways to Get Started

#### 1️⃣ Quick Start (Recommended for First Run)
```bash
python quick_start.py
```
- Uses conservative settings guaranteed to work
- 4-hour timeframe with 3 trading pairs
- Quick 20-epoch training for testing
- Perfect for first-time users

#### 2️⃣ Run Diagnostics
```bash
python diagnose_data.py
```
- Tests Binance connection
- Validates data quality
- Checks indicator calculations
- Provides specific recommendations

#### 3️⃣ Full Backtest
```bash
python crypto_production_system.py --mode backtest --timeframe 4h
```

### Command Line Options

```bash
# Backtest with custom pairs
python crypto_production_system.py --mode backtest --pairs BTCUSDT ETHUSDT --timeframe 4h

# Live paper trading
python crypto_production_system.py --mode live --capital 10000 --min-confidence 0.70

# Training only
python crypto_production_system.py --mode train --epochs 50
```

## 📊 Example Output (v2.0)

```
============================================================
BACKTESTING
============================================================
Backtesting BTCUSDT...
BTCUSDT Model Accuracy: 0.6234
BTCUSDT Results:
  Trades: 47
  Win Rate: 61.70%
  Total Return: 12.34%
  Sharpe Ratio: 1.87
  Sortino Ratio: 2.14
  Max Drawdown: -8.45%
  Expectancy: $23.45

Portfolio Metrics:
  Total Symbols: 3
  Portfolio Win Rate: 59.60%
  Average Return: 10.63%
  Average Sharpe: 1.70
```

## 🛠️ Technical Stack (v2.0)

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning models
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Preprocessing and validation
- **Matplotlib**: Visualization (optional)
- **Binance API**: Market data (no keys needed)

## 📁 Project Structure (v2.0)

```
version2/
├── crypto_production_system.py  # Main system
├── test_crypto_system.py        # Test suite (24 tests)
├── quick_start.py               # Easy setup script
├── diagnose_data.py             # Diagnostic tool
├── requirements_production.txt  # Dependencies
├── README_PRODUCTION.md         # Full documentation
├── TECHNICAL_COMPARISON.md      # Bug fixes explained
├── TROUBLESHOOTING.md          # Common issues & fixes
├── data/                        # Downloaded data
├── models/                      # Saved models
├── logs/                        # System logs
└── backtests/                   # Backtest results
```

## ⚠️ Important Notes (v2.0)

### What Makes This "Production-Grade"

✅ **Real OHLCV Data**: Actual candlesticks from Binance, not snapshots  
✅ **No Data Leakage**: Walk-forward validation, scalers fit on train only  
✅ **Correct Indicators**: ATR, ADX, Stochastic properly implemented  
✅ **Realistic Costs**: Includes fees, slippage, position limits  
✅ **Tested**: 24 unit tests verify correctness  
✅ **No Crashes**: All bugs from v1.0 fixed (threading, logger, etc.)

### Known Limitations

⚠️ **Beta Software**: Still under active development  
⚠️ **Paper Trading Only**: Not connected to live exchanges  
⚠️ **No Guarantees**: Past performance ≠ future results  
⚠️ **Educational Use**: Not financial advice

### Realistic Performance Expectations

- **Win Rate**: 55-65% (anything above 60% is excellent)
- **Sharpe Ratio**: 1.0-2.0 (above 1.5 is very good)
- **Max Drawdown**: 10-20% (expected range)
- **Annual Return**: 10-30% (conservative estimate)

If you see win rates >75% or Sharpe >3.0, you're likely overfitting!

## 🆘 Troubleshooting (v2.0)

### "Insufficient data" error?
```bash
# Run diagnostics first
python diagnose_data.py

# Use quick start with conservative settings
python quick_start.py

# Or use 4h timeframe
python crypto_production_system.py --mode backtest --timeframe 4h
```

See `TROUBLESHOOTING.md` for detailed solutions.

### Can't connect to Binance?
```bash
# Test connection
curl https://api.binance.com/api/v3/ping
```

### Tests failing?
```bash
python test_crypto_system.py
```

---

# 📦 Version 1.0 - Original AI-Powered Bot

**Location:** Root directory (`crypto_coin_analyser_.py`)

The original cryptocurrency analysis bot that combines real-time market data with DeepSeek AI for intelligent decision-making.

## ✨ Key Features (v1.0)

### 🤖 AI-Powered Analysis
- Integration with **DeepSeek API** for intelligent market analysis
- Confidence scoring system (minimum 65% threshold)
- Natural language reasoning for trading decisions
- Conversational AI explanations

### 📊 Technical Analysis
- Real-time price and volume tracking
- Multiple technical indicators:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - MACD (Moving Average Convergence Divergence)
  - Volume spike detection
- Support and resistance level identification
- Trend analysis (bullish/bearish/sideways)

### 💰 Risk Management
- Portfolio-based position sizing (2% risk per trade)
- Maximum drawdown protection (10% threshold)
- Confidence-based filtering
- Volume spike validation (150% threshold)

### 📈 Market Monitoring
- Tracks top coins by market capitalization
- Volatility-based coin selection
- Real-time data from **CoinMarketCap API**
- Customizable analysis intervals (default: 180 seconds)

### 📝 Performance Tracking
- Historical accuracy monitoring
- Trade statistics by symbol
- CSV logging for all decisions
- State persistence across sessions
- Performance metrics (cycle times, API latency, uptime)

### 🔄 Automated Operations
- Continuous monitoring with configurable intervals
- Automatic state saving and recovery
- API rate limiting protection
- Error handling and auto-recovery

## 🛠️ Technical Stack (v1.0)

- **Python 3.x**
- **APIs:**
  - CoinMarketCap API (market data)
  - DeepSeek API (AI analysis)
- **Libraries:**
  - `numpy` - numerical computations
  - `requests` - API calls
  - `concurrent.futures` - parallel processing

## 📦 Installation (v1.0)

### Prerequisites

```bash
python 3.7+
pip
```

### Setup

1. Install dependencies:
```bash
pip install numpy requests
```

2. Configure API keys in the script:
```python
CMC_API_KEY = "your_coinmarketcap_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

## 🚀 Usage (v1.0)

### Basic Execution

```bash
python crypto_coin_analyser_.py
```

### Configuration Options

Edit these parameters in the script:

```python
INTERVAL = 180              # Analysis cycle interval (seconds)
TOP_MARKET_CAP = 5         # Number of top coins by market cap
TOP_VOLATILE = 5           # Number of volatile coins to track
ANALYZE_TOTAL = 8          # Total coins to analyze per cycle
HISTORY_LEN = 50           # Price history length
PORTFOLIO_VALUE = 10000    # Initial portfolio value (USD)
RISK_PER_TRADE = 0.02      # Risk percentage per trade (2%)
MAX_DRAWDOWN = 0.10        # Maximum allowed drawdown (10%)
MIN_CONFIDENCE_THRESHOLD = 0.65  # Minimum AI confidence (65%)
```

### Output

Color-coded terminal output:
- 🟢 **GREEN**: Buy signals and success metrics
- 🔴 **RED**: Sell signals and error messages
- 🟡 **YELLOW**: Hold signals and warnings
- 🔵 **CYAN**: Information and statistics

## 📁 Project Structure (v1.0)

```
crypto-price-prediction-bot/
│
├── crypto_coin_analyser_.py    # Main bot script
├── trader_state.json            # Persistent state file
├── logs/                        # Trading decision logs
│   └── [SYMBOL].csv            # Per-coin CSV logs
├── version2/                    # V2.0 Production System
└── README.md                    # This file
```

## 📊 Data Logging (v1.0)

### CSV Logs
Each cryptocurrency gets its own CSV log file:
- Timestamp
- Price
- Decision (BUY/SELL/HOLD)
- AI reasoning
- Technical indicators

### State Persistence
Automatically saves:
- Cycle count
- Active symbols
- Accuracy metrics
- Historical prices
- Trading statistics

## 🧠 How It Works (v1.0)

1. **Data Collection**: Fetches real-time data from CoinMarketCap
2. **Coin Selection**: Identifies top coins by market cap and volatility
3. **Technical Analysis**: Calculates indicators (RSI, MACD, Bollinger Bands)
4. **AI Analysis**: Sends context to DeepSeek API for decision-making
5. **Confidence Filtering**: Only acts on high-confidence signals (≥65%)
6. **Risk Management**: Calculates position sizes
7. **Performance Tracking**: Updates metrics and logs
8. **State Management**: Saves progress for next cycle

---

# 🔄 Version Comparison

| Feature | v1.0 (Original) | v2.0 (Production) |
|---------|----------------|-------------------|
| **Data Source** | CoinMarketCap snapshots | Binance OHLCV candles |
| **AI/ML** | DeepSeek API | TensorFlow (Transformer+LSTM) |
| **Validation** | Basic | Walk-forward (no leakage) |
| **Indicators** | Basic implementations | Properly tested (50+) |
| **Backtesting** | Not included | Full with realistic costs |
| **API Keys** | Required (2 keys) | Not required (public API) |
| **Testing** | None | 24 comprehensive tests |
| **Paper Trading** | No | Yes (live simulation) |
| **Visualization** | Terminal only | Auto-generated charts |
| **Documentation** | Basic README | Full docs + troubleshooting |

## Which Version Should I Use?

### Use **v1.0** if you want:
- ✅ Simple setup with AI reasoning
- ✅ Natural language explanations
- ✅ Lightweight system
- ✅ DeepSeek AI integration

### Use **v2.0** if you want:
- ✅ Production-grade ML models
- ✅ Proper backtesting
- ✅ No API keys needed
- ✅ Realistic performance metrics
- ✅ Comprehensive testing
- ✅ Live paper trading

**Recommendation**: Start with **v2.0** for serious backtesting and analysis. Use **v1.0** for experimental AI-powered insights.

---

# ⚠️ Important Disclaimers

## BETA SOFTWARE NOTICE

**Both systems are currently in BETA.** This means:
- ⚠️ Features may change without notice
- ⚠️ Bugs may exist despite testing
- ⚠️ Performance may vary
- ⚠️ Not recommended for production trading
- ⚠️ Use at your own risk

## TRADING RISK WARNINGS

### This Software is NOT:
❌ Financial advice  
❌ A guarantee of profits  
❌ Suitable for live trading without extensive testing  
❌ A replacement for professional financial advice  

### This Software IS:
✅ Educational and research tool  
✅ For backtesting and analysis  
✅ For learning about trading systems  
✅ For paper trading and simulation  

### Critical Warnings:
1. **Cryptocurrency trading involves substantial risk of loss**
2. **Past performance does not guarantee future results**
3. **Never invest more than you can afford to lose**
4. **Always do your own research (DYOR)**
5. **Consult with qualified financial advisors**
6. **Backtest results ≠ live trading performance**

## Legal Notice

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

# 🔐 Security Notes

### API Key Security (v1.0)
- API keys are hardcoded for demonstration purposes
- **NEVER commit API keys to version control**
- Use environment variables in production:
  ```python
  import os
  CMC_API_KEY = os.getenv('CMC_API_KEY')
  DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
  ```

### Data Security (v2.0)
- No API keys required (uses public endpoints)
- All data stays local
- No external services except Binance API
- Logs may contain sensitive strategy info - keep private

---

# 🐛 Known Issues & Limitations

## Version 1.0
- Requires CoinMarketCap and DeepSeek API keys
- Point-in-time data (not candles)
- Limited backtesting capabilities
- Windows UTF-8 encoding issues (handled)

## Version 2.0
- Requires sufficient historical data
- GPU recommended but not required
- matplotlib needed for visualizations (optional)
- May need configuration adjustments for 1h timeframe

See `version2/TROUBLESHOOTING.md` for v2.0 issue resolution.

---

# 🔮 Future Enhancements

## Planned for v1.0
- [ ] Web dashboard for monitoring
- [ ] Additional AI model integrations
- [ ] Multi-exchange support
- [ ] Real-time alerts and notifications

## Planned for v2.0
- [ ] Multi-timeframe analysis
- [ ] Reinforcement learning agents (PPO)
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization
- [ ] Order book analysis
- [ ] Live trading integration (with extensive safeguards)

## Long-term (Both Versions)
- [ ] Mobile app
- [ ] Cloud deployment options
- [ ] Community strategy sharing
- [ ] Advanced risk management
- [ ] Automated parameter optimization

---

# 📄 License

This project is licensed under the **MIT License**.

### MIT License Summary

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

# 🤝 Contributing

This is currently a demonstration and educational project in **BETA**.

For questions, feedback, or contributions:
1. Open an issue describing your suggestion
2. For bugs, include system info and error logs
3. For features, explain the use case
4. All contributions welcome but review may take time

---

# 📚 Documentation

## Version 1.0 Documentation
- This README (you're reading it)
- Inline code comments
- Configuration section above

## Version 2.0 Documentation
- `version2/README_PRODUCTION.md` - Full documentation
- `version2/TECHNICAL_COMPARISON.md` - Bug fixes explained
- `version2/TROUBLESHOOTING.md` - Common issues & solutions
- Inline docstrings in all functions
- 24 test cases with explanations

---

# 🆘 Getting Help

### For v1.0 Issues:
1. Check API keys are valid
2. Verify internet connection
3. Check CoinMarketCap API limits
4. Review logs in `logs/` directory

### For v2.0 Issues:
1. Run `python diagnose_data.py`
2. Check `TROUBLESHOOTING.md`
3. Run tests: `python test_crypto_system.py`
4. Review logs in `version2/logs/`

### General Help:
- Read the appropriate README
- Check known issues section
- Review example configurations
- Test with minimal settings first

---

# 🎯 Quick Links

- **v1.0 Main Script**: `crypto_coin_analyser_.py`
- **v2.0 Main Script**: `version2/crypto_production_system.py`
- **v2.0 Quick Start**: `version2/quick_start.py`
- **v2.0 Diagnostics**: `version2/diagnose_data.py`
- **Full v2.0**: `version2/README_PRODUCTION.md`

---

**⚠️ FINAL REMINDER**

This software is provided for **educational and demonstration purposes only**. Both v1.0 and v2.0 are in **BETA** and should NOT be used for actual trading without extensive testing, validation, and professional consultation.

**Cryptocurrency trading carries substantial risk of loss. Always conduct your own research and consult with qualified financial advisors before making investment decisions.**

**The author and contributors are not responsible for any financial losses incurred through the use of this software.**

---

**Happy Learning! 📚**  
**Stay Safe! 🛡️**  
**Trade Responsibly! 💡**

---

*Last Updated: January 2025*  
*Repository Status: BETA*  
*License: MIT*
