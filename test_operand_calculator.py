import pandas as pd
import numpy as np
import pytest
from operand_calculator import OperandCalculator


class TestOperandCalculator:
    """Test suite for OperandCalculator class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)  # For reproducible results
        
        data = pd.DataFrame({
            'date': dates,
            'symbol': 'AAPL',
            'open': np.random.uniform(150, 160, 30),
            'high': np.random.uniform(160, 170, 30),
            'low': np.random.uniform(140, 150, 30),
            'close': np.random.uniform(145, 165, 30),
            'volume': np.random.randint(1000000, 5000000, 30)
        })
        
        # Ensure OHLC relationships are valid
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def calculator(self, sample_data):
        """Create OperandCalculator instance with sample data"""
        return OperandCalculator(sample_data)
    
    def test_initialization(self, sample_data):
        """Test OperandCalculator initialization"""
        # Test successful initialization
        calc = OperandCalculator(sample_data)
        assert calc.data is not None
        assert len(calc.data) == 30
        
        # Test missing required columns
        invalid_data = sample_data.drop(columns=['close'])
        with pytest.raises(ValueError, match="Missing required columns"):
            OperandCalculator(invalid_data)
    
    def test_calculate_column(self, calculator):
        """Test column calculation with offset support"""
        # Test basic column calculation
        operand = {'type': 'column', 'name': 'close'}
        result = calculator.calculate_column(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        assert result.name == 'close'
        
        # Test column with positive offset
        operand = {'type': 'column', 'name': 'close', 'offset': 3}
        result = calculator.calculate_column(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        assert result.iloc[-3:].isna().all()  # Last 3 values should be NaN
        
        # Test column with negative offset
        operand = {'type': 'column', 'name': 'close', 'offset': -3}
        result = calculator.calculate_column(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        assert result.iloc[:3].isna().all()  # First 3 values should be NaN
        
        # Test missing column name
        with pytest.raises(ValueError, match="Column operand requires 'name' field"):
            calculator.calculate_column({'type': 'column'})
        
        # Test invalid column name
        with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
            calculator.calculate_column({'type': 'column', 'name': 'invalid_column'})
    
    def test_calculate_indicator(self, calculator):
        """Test indicator calculation with offset support"""
        # Test SMA calculation
        operand = {
            'type': 'indicator',
            'name': 'sma',
            'column': 'close',
            'params': [5]
        }
        result = calculator.calculate_indicator(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        assert not result.isna().all()  # Should have some non-NaN values
        
        # Test EMA calculation
        operand = {
            'type': 'indicator',
            'name': 'ema',
            'column': 'close',
            'params': [10]
        }
        result = calculator.calculate_indicator(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        
        # Test RSI calculation
        operand = {
            'type': 'indicator',
            'name': 'rsi',
            'column': 'close',
            'params': [14]
        }
        result = calculator.calculate_indicator(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        
        # Test MACD calculation
        operand = {
            'type': 'indicator',
            'name': 'macd',
            'column': 'close',
            'params': [12, 26, 9]
        }
        result = calculator.calculate_indicator(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        
        # Test Bollinger Bands calculation
        operand = {
            'type': 'indicator',
            'name': 'bollinger_bands',
            'column': 'close',
            'params': [20, 2]
        }
        result = calculator.calculate_indicator(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        
        # Test indicator with offset
        operand = {
            'type': 'indicator',
            'name': 'sma',
            'column': 'close',
            'params': [5],
            'offset': 2
        }
        result = calculator.calculate_indicator(operand)
        assert isinstance(result, pd.Series)
        assert len(result) == 30
        assert result.iloc[-2:].isna().all()  # Last 2 values should be NaN
        
        # Test missing indicator name
        with pytest.raises(ValueError, match="Indicator operand requires 'name' field"):
            calculator.calculate_indicator({'type': 'indicator', 'column': 'close'})
        
        # Test missing column name
        with pytest.raises(ValueError, match="Indicator operand requires 'column' field"):
            calculator.calculate_indicator({'type': 'indicator', 'name': 'sma'})
        
        # Test invalid column name
        with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
            calculator.calculate_indicator({
                'type': 'indicator',
                'name': 'sma',
                'column': 'invalid_column'
            })
        
        # Test unsupported indicator
        with pytest.raises(ValueError, match="Unsupported indicator"):
            calculator.calculate_indicator({
                'type': 'indicator',
                'name': 'unsupported_indicator',
                'column': 'close'
            })
    
    def test_apply_offset(self, calculator):
        """Test offset application functionality"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test zero offset
        result = calculator.apply_offset(series, 0)
        assert result.equals(series)
        
        # Test positive offset (future)
        result = calculator.apply_offset(series, 3)
        expected = pd.Series([1, 2, 3, 4, 5, 6, 7, np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
        
        # Test negative offset (past)
        result = calculator.apply_offset(series, -3)
        expected = pd.Series([np.nan, np.nan, np.nan, 4, 5, 6, 7, 8, 9, 10])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
        
        # Test invalid offset type
        with pytest.raises(ValueError, match="Offset must be an integer"):
            calculator.apply_offset(series, "invalid_offset")
    
    def test_calculate_constant(self, calculator):
        """Test constant value calculation"""
        # Test integer constant
        operand = {'type': 'constant', 'value': 42}
        result = calculator.calculate_constant(operand)
        assert isinstance(result, float)
        assert result == 42.0
        
        # Test float constant
        operand = {'type': 'constant', 'value': 3.14}
        result = calculator.calculate_constant(operand)
        assert isinstance(result, float)
        assert result == 3.14
        
        # Test missing value
        with pytest.raises(ValueError, match="Constant operand requires 'value' field"):
            calculator.calculate_constant({'type': 'constant'})
        
        # Test invalid value type
        with pytest.raises(ValueError, match="Constant 'value' must be a number"):
            calculator.calculate_constant({'type': 'constant', 'value': 'invalid'})
    
    def test_calculate_operand(self, calculator):
        """Test generic operand calculation"""
        # Test column operand
        operand = {'type': 'column', 'name': 'close'}
        result = calculator.calculate_operand(operand)
        assert isinstance(result, pd.Series)
        
        # Test indicator operand
        operand = {
            'type': 'indicator',
            'name': 'sma',
            'column': 'close',
            'params': [5]
        }
        result = calculator.calculate_operand(operand)
        assert isinstance(result, pd.Series)
        
        # Test constant operand
        operand = {'type': 'constant', 'value': 42}
        result = calculator.calculate_operand(operand)
        assert isinstance(result, float)
        assert result == 42.0
        
        # Test invalid operand type
        with pytest.raises(ValueError, match="Invalid operand type"):
            calculator.calculate_operand({'type': 'invalid_type'})
    
    def test_get_supported_indicators(self, calculator):
        """Test getting supported indicators"""
        indicators = calculator.get_supported_indicators()
        assert isinstance(indicators, list)
        assert 'sma' in indicators
        assert 'ema' in indicators
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'bollinger_bands' in indicators
    
    def test_get_supported_columns(self, calculator):
        """Test getting supported columns"""
        columns = calculator.get_supported_columns()
        assert isinstance(columns, list)
        assert 'open' in columns
        assert 'high' in columns
        assert 'low' in columns
        assert 'close' in columns
        assert 'volume' in columns
    
    def test_validate_operand(self, calculator):
        """Test operand validation"""
        # Test valid column operand
        operand = {'type': 'column', 'name': 'close'}
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is True
        assert "validation successful" in message
        
        # Test valid indicator operand
        operand = {
            'type': 'indicator',
            'name': 'sma',
            'column': 'close',
            'params': [5]
        }
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is True
        
        # Test valid constant operand
        operand = {'type': 'constant', 'value': 42}
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is True
        
        # Test invalid operand type
        operand = {'type': 'invalid_type'}
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is False
        assert "Invalid operand type" in message
        
        # Test missing column name
        operand = {'type': 'column'}
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is False
        assert "requires 'name' field" in message
        
        # Test invalid column name
        operand = {'type': 'column', 'name': 'invalid_column'}
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is False
        assert "not found in data" in message
        
        # Test invalid constant value type
        operand = {'type': 'constant', 'value': 'invalid'}
        is_valid, message = calculator.validate_operand(operand)
        assert is_valid is False
        assert "must be a number" in message


if __name__ == "__main__":
    pytest.main([__file__])