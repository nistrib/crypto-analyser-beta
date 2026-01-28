# Crypto Price Prediction Bot

**Price Prediction Bot for Crypto Currencies**  
*Distributed for demonstration purposes. Not a final product.*

---

## 📋 Overview

An automated cryptocurrency trading analysis bot that combines real-time market data with AI-powered decision making. The bot monitors top cryptocurrencies, analyzes market conditions, and provides BUY/SELL/HOLD recommendations using technical indicators and machine learning.

## ✨ Key Features

### 🤖 AI-Powered Analysis
- Integration with DeepSeek API for intelligent market analysis
- Confidence scoring system (minimum 65% threshold)
- Natural language reasoning for trading decisions

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
- Real-time data from CoinMarketCap API
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

## 🛠️ Technical Stack

- **Python 3.x**
- **APIs:**
  - CoinMarketCap API (market data)
  - DeepSeek API (AI analysis)
- **Libraries:**
  - `numpy` - numerical computations
  - `requests` - API calls
  - `concurrent.futures` - parallel processing

## 📦 Installation

### Prerequisites

```bash
python 3.7+
pip
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd crypto-price-prediction-bot
```

2. Install required dependencies:
```bash
pip install numpy requests
```

3. Configure API keys in the script:
```python
CMC_API_KEY = "your_coinmarketcap_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

## 🚀 Usage

### Basic Execution

```bash
python crypto_coin_analyser_.py
```

### Configuration Options

Edit the following parameters in the script to customize behavior:

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

The bot provides color-coded terminal output:
- 🟢 **GREEN**: Buy signals and success metrics
- 🔴 **RED**: Sell signals and error messages
- 🟡 **YELLOW**: Hold signals and warnings
- 🔵 **CYAN**: Information and statistics

## 📁 Project Structure

```
crypto-price-prediction-bot/
│
├── crypto_coin_analyser_.py    # Main bot script
├── trader_state.json            # Persistent state file
├── logs/                        # Trading decision logs
│   └── [SYMBOL].csv            # Per-coin CSV logs
└── README.md                    # This file
```

## 📊 Data Logging

### CSV Logs
Each cryptocurrency gets its own CSV log file in the `logs/` directory:
- Timestamp
- Price
- Decision (BUY/SELL/HOLD)
- AI reasoning
- Technical indicators

### State Persistence
The bot automatically saves its state including:
- Cycle count
- Active symbols
- Accuracy metrics
- Historical prices
- Trading statistics

## 🧠 How It Works

1. **Data Collection**: Fetches real-time market data from CoinMarketCap
2. **Coin Selection**: Identifies top coins by market cap and volatility
3. **Technical Analysis**: Calculates indicators (RSI, MACD, Bollinger Bands, etc.)
4. **AI Analysis**: Sends market context to DeepSeek API for decision-making
5. **Confidence Filtering**: Only executes high-confidence signals (≥65%)
6. **Risk Management**: Calculates position sizes based on portfolio and confidence
7. **Performance Tracking**: Updates accuracy metrics and logs decisions
8. **State Management**: Saves progress and prepares for next cycle

## ⚠️ Limitations & Disclaimers

- **DEMONSTRATION ONLY**: This is not financial advice or a production-ready trading system
- **API Keys**: Requires valid CoinMarketCap and DeepSeek API keys
- **Rate Limits**: Respects API rate limits (configurable)
- **No Automated Trading**: Bot provides analysis only; does not execute actual trades
- **Historical Performance**: Past predictions do not guarantee future results
- **Risk**: Cryptocurrency trading involves substantial risk of loss

## 🔐 Security Notes

- API keys are hardcoded for demonstration purposes
- In production, use environment variables or secure key management
- Never commit API keys to version control
- Consider implementing additional security measures for real trading

## 🐛 Known Issues

- Console output formatting may vary on different operating systems
- Requires stable internet connection for API calls
- Windows-specific UTF-8 encoding handling included

## 🔮 Future Enhancements

- [ ] Web dashboard for monitoring
- [ ] Additional AI model integrations
- [ ] More sophisticated risk management strategies
- [ ] Backtesting capabilities
- [ ] Multi-exchange support
- [ ] Real-time alerts and notifications
- [ ] Portfolio optimization algorithms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

## 🤝 Contributing

This is a demonstration project. For questions or feedback, please refer to the project maintainer.

---

**⚠️ IMPORTANT DISCLAIMER**

This software is provided for educational and demonstration purposes only. It is NOT intended for actual trading or investment decisions. Cryptocurrency trading carries substantial risk of loss. Always conduct your own research and consult with qualified financial advisors before making investment decisions. The authors and contributors are not responsible for any financial losses incurred through the use of this software.
