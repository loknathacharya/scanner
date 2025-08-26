# 📊 Stock Scanner & Filter App (Chartink-like)

A powerful Streamlit application for scanning and filtering stock data with technical indicators, similar to Chartink. Built with performance optimization for large datasets.

## 🚀 Features

### Core Features
- **File Upload Support**: CSV, Excel, and Parquet files
- **Automatic Column Detection**: Smart detection of Date, Symbol, OHLCV columns  
- **Technical Indicators**: 20+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **Filter Builder**: Pre-built templates and custom filter creation
- **Interactive Results**: Column selection, sorting, pagination
- **Export Options**: CSV, Excel, JSON downloads
- **Performance Charts**: Candlestick charts with indicator overlays

### Filter Capabilities
- **Pre-built Templates**: Momentum, oversold, breakout, and technical signal filters
- **Custom Filters**: Build complex conditions with AND/OR logic
- **Special Operators**: Crossover detection (crosses_above, crosses_below)
- **Save/Load Filters**: Persist custom filters as JSON

### Performance Optimizations
- **Caching**: Streamlit caching for repeated operations
- **Vectorized Operations**: Pandas/NumPy optimized calculations
- **Memory Efficient**: Optimized data types and chunked processing
- **Fast Indicators**: NumPy-based technical indicator calculations

## 📁 Project Structure

```
stock_scanner/
├── stock_scanner_main.py              # Main Streamlit application
├── filters_module.py          # Filter engine and templates
├── indicators_module.py       # Technical indicators calculation
├── utils_module.py           # Data processing utilities  
├── ui_components_module.py   # Reusable UI components
├── requirements_file.txt   # Python dependencies
└── readme_file.md         # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download Files
Save all the Python files (`app.py`, `filters.py`, `indicators.py`, `utils.py`, `ui_components.py`) in a single directory.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### 1. Upload Data
- Go to the **Upload Data** tab
- Upload CSV, Excel, or Parquet files with OHLCV data
- The app will auto-detect columns (Date, Symbol, Open, High, Low, Close, Volume)
- Verify column mappings and click **Process Data**

### 2. Build Filters

#### Pre-built Templates
- Select from ready-made filters like "Price Above SMA(20)", "High Volume", "RSI Overbought"
- Click **Apply Template Filter** to run

#### Custom Filters  
- Add conditions using the filter builder
- Choose Column, Operator, and Value for each condition
- Combine conditions with AND/OR logic
- Save frequently used filters for later

### 3. View Results
- Results show in an interactive table
- Select which columns to display
- Sort and paginate through results
- View individual stock charts
- Export results as CSV, Excel, or JSON

## 🎯 Sample Filters

### Momentum Filters
```
close > sma_20 AND volume > volume_sma_20 * 1.5
rsi > 60 AND macd > macd_signal
```

### Oversold Opportunities
```
rsi < 30 AND close > sma_5
bb_lower > close AND volume > volume_sma_20
```

### Breakout Patterns
```
close > high_20 AND volume > volume_sma_50 * 2
close crosses_above sma_50 AND rsi > 50
```

## 📈 Technical Indicators

### Moving Averages
- **Simple MA**: 5, 10, 20, 50, 200 periods
- **Exponential MA**: 12, 20, 26, 50 periods

### Oscillators  
- **RSI**: Relative Strength Index (14 periods)
- **Stochastic**: %K and %D
- **Williams %R**: Williams Percent Range
- **CCI**: Commodity Channel Index

### Trend Indicators
- **MACD**: Line, Signal, and Histogram
- **ADX**: Average Directional Index (future)

### Volatility
- **Bollinger Bands**: Upper, Middle, Lower, Width
- **ATR**: Average True Range

### Volume
- **Volume SMA**: 20 and 50-period averages
- **Volume Ratio**: Current vs average volume

## 💾 Data Format Requirements

Your data should contain these columns (exact names may vary):
- **Date/DateTime**: Date column in various formats
- **Symbol/Ticker**: Stock identifier  
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Trading volume

### Sample CSV Format
```csv
Date,Symbol,Open,High,Low,Close,Volume
2024-01-01,AAPL,180.50,182.30,179.20,181.75,45000000
2024-01-02,AAPL,181.80,183.50,181.00,182.90,38000000
2024-01-01,MSFT,375.20,378.50,374.10,377.80,25000000
```

## 🚀 Performance Tips

### For Large Datasets (>100MB)
- Use **Parquet** format for fastest loading
- Filter data by date range before processing
- Use pre-built templates for common scans
- Enable chunked processing in code if needed

### Memory Optimization
- The app automatically optimizes data types
- Remove unused columns before upload
- Use date filters to limit data range

## 🔧 Advanced Configuration

### Adding Custom Indicators
Edit `indicators.py` to add new technical indicators:

```python
@staticmethod
def my_custom_indicator(data: pd.Series, window: int = 14) -> pd.Series:
    """Your custom indicator logic"""
    return data.rolling(window).apply(your_function)
```

### Custom Filter Templates
Add templates in `filters.py`:

```python
"My Custom Filter": "close > sma_20 AND your_indicator > threshold"
```

## 🐛 Troubleshooting

### Common Issues

**"Column not found" error**
- Check if your data has all required OHLCV columns
- Verify column names match detected columns

**"Memory error" with large files**
- Try smaller file size or date range
- Increase system memory allocation
- Use Parquet format instead of CSV

**Slow performance**
- Clear browser cache and restart Streamlit
- Check data quality (remove duplicates, invalid dates)
- Use pre-built filters instead of complex custom ones

### Getting Help
1. Check data format requirements
2. Verify all dependencies are installed
3. Ensure Python version 3.8+
4. Try with smaller sample data first

## 📝 Future Enhancements

- **Multi-timeframe Analysis**: Daily + Intraday scans
- **Backtesting Engine**: Test filter performance historically  
- **Alert System**: Email/webhook notifications
- **Pattern Recognition**: Automated chart pattern detection
- **Portfolio Management**: Track multiple watchlists
- **Real-time Data**: Live market data integration

## 🤝 Contributing

Feel free to contribute by:
- Adding new technical indicators
- Creating more filter templates  
- Improving performance optimizations
- Adding new chart types
- Enhancing UI/UX components

## 📄 License

This project is open source. Feel free to use, modify, and distribute as needed.

---

**Happy Scanning! 📊🚀**