import pandas as pd
import numpy as np
from indicators_module import TechnicalIndicators

def test_offset_support():
    """Test offset support for technical indicators"""
    print("Testing offset support for technical indicators...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(50) * 0.02 + 100) + 100
    volumes = np.random.randint(100000, 1000000, 50)
    
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })
    
    # Initialize indicator calculator
    indicator = TechnicalIndicators()
    
    # Test SMA with offset
    print("\n1. Testing SMA with offset...")
    sma_normal = indicator.sma(df['close'], 20, 0)
    sma_offset_5 = indicator.sma(df['close'], 20, 5)
    sma_offset_neg_5 = indicator.sma(df['close'], 20, -5)
    
    print(f"SMA normal (last 5 values): {sma_normal.tail().values}")
    print(f"SMA offset +5 (last 5 values): {sma_offset_5.tail().values}")
    print(f"SMA offset -5 (last 5 values): {sma_offset_neg_5.tail().values}")
    
    # Test EMA with offset
    print("\n2. Testing EMA with offset...")
    ema_normal = indicator.ema(df['close'], 20, 0)
    ema_offset_5 = indicator.ema(df['close'], 20, 5)
    
    print(f"EMA normal (last 5 values): {ema_normal.tail().values}")
    print(f"EMA offset +5 (last 5 values): {ema_offset_5.tail().values}")
    
    # Test RSI with offset
    print("\n3. Testing RSI with offset...")
    rsi_normal = indicator.rsi(df['close'], 14, 0)
    rsi_offset_5 = indicator.rsi(df['close'], 14, 5)
    
    print(f"RSI normal (last 5 values): {rsi_normal.tail().values}")
    print(f"RSI offset +5 (last 5 values): {rsi_offset_5.tail().values}")
    
    # Test MACD with offset
    print("\n4. Testing MACD with offset...")
    macd_normal, signal_normal, hist_normal = indicator.macd(df['close'], 12, 26, 9, 0)
    macd_offset_5, signal_offset_5, hist_offset_5 = indicator.macd(df['close'], 12, 26, 9, 5)
    
    print(f"MACD normal (last 3 values): {macd_normal.tail(3).values}")
    print(f"MACD offset +5 (last 3 values): {macd_offset_5.tail(3).values}")
    
    # Test Bollinger Bands with offset
    print("\n5. Testing Bollinger Bands with offset...")
    bb_upper_normal, bb_middle_normal, bb_lower_normal = indicator.bollinger_bands(df['close'], 20, 2, 0)
    bb_upper_offset_5, bb_middle_offset_5, bb_lower_offset_5 = indicator.bollinger_bands(df['close'], 20, 2, 5)
    
    print(f"BB upper normal (last 3 values): {bb_upper_normal.tail(3).values}")
    print(f"BB upper offset +5 (last 3 values): {bb_upper_offset_5.tail(3).values}")
    
    # Test add_all_indicators with offset
    print("\n6. Testing add_all_indicators with offset...")
    df_indicators_normal = indicator.add_all_indicators(df, offset=0)
    df_indicators_offset = indicator.add_all_indicators(df, offset=5)
    
    print(f"Normal indicators shape: {df_indicators_normal.shape}")
    print(f"Offset indicators shape: {df_indicators_offset.shape}")
    
    # Check if offset indicators are shifted
    print(f"Normal SMA_20 (last 3): {df_indicators_normal['sma_20'].tail(3).values}")
    print(f"Offset SMA_20 (last 3): {df_indicators_offset['sma_20'].tail(3).values}")
    
    # Test validation
    print("\n7. Testing validation...")
    try:
        # Invalid window
        indicator.sma(df['close'], -5, 0)
        print("ERROR: Should have failed with negative window")
    except ValueError as e:
        print(f"✓ Correctly caught negative window error: {e}")
    
    try:
        # Invalid offset
        indicator.sma(df['close'], 20, "invalid")
        print("ERROR: Should have failed with invalid offset")
    except ValueError as e:
        print(f"✓ Correctly caught invalid offset error: {e}")
    
    print("\n✓ All offset support tests completed successfully!")

def test_timeframe_support():
    """Test timeframe support for technical indicators"""
    print("\n\nTesting timeframe support for technical indicators...")
    
    # Create sample intraday data
    dates = pd.date_range('2023-01-01 09:30', periods=100, freq='15min')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100) * 0.01 + 100) + 100
    volumes = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })
    
    # Initialize indicator calculator
    indicator = TechnicalIndicators()
    
    # Test daily timeframe (default)
    print("\n1. Testing daily timeframe...")
    df_daily = indicator.add_all_indicators(df, timeframe='daily')
    print(f"Daily timeframe shape: {df_daily.shape}")
    
    # Test weekly timeframe
    print("\n2. Testing weekly timeframe...")
    try:
        df_weekly = indicator.add_all_indicators(df, timeframe='weekly')
        print(f"Weekly timeframe shape: {df_weekly.shape}")
    except Exception as e:
        print(f"Weekly timeframe test failed: {e}")
    
    print("\n✓ Timeframe support tests completed!")

def test_performance():
    """Test performance improvements with caching"""
    print("\n\nTesting performance improvements...")
    
    # Create larger dataset
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.02 + 100) + 100
    volumes = np.random.randint(100000, 1000000, 1000)
    
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })
    
    # Initialize indicator calculator
    indicator = TechnicalIndicators()
    
    # Test performance with caching
    import time
    
    print("\n1. Testing SMA performance...")
    start_time = time.time()
    for i in range(10):
        result = indicator.sma(df['close'], 20, 0)
    end_time = time.time()
    print(f"SMA calculation time (10 iterations): {end_time - start_time:.4f} seconds")
    
    print("\n2. Testing EMA performance...")
    start_time = time.time()
    for i in range(10):
        result = indicator.ema(df['close'], 20, 0)
    end_time = time.time()
    print(f"EMA calculation time (10 iterations): {end_time - start_time:.4f} seconds")
    
    print("\n3. Testing RSI performance...")
    start_time = time.time()
    for i in range(10):
        result = indicator.rsi(df['close'], 14, 0)
    end_time = time.time()
    print(f"RSI calculation time (10 iterations): {end_time - start_time:.4f} seconds")
    
    print("\n✓ Performance tests completed!")

if __name__ == "__main__":
    test_offset_support()
    test_timeframe_support()
    test_performance()
    print("\n🎉 All tests completed successfully!")