import pandas as pd
import numpy as np
import pytest
from advanced_filter_engine import AdvancedFilterEngine


class TestAdvancedFilterEngine:
    """Test cases for AdvancedFilterEngine class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'date': dates,
            'symbol': ['AAPL'] * 10,
            'open': np.random.uniform(150, 160, 10),
            'high': np.random.uniform(155, 165, 10),
            'low': np.random.uniform(145, 155, 10),
            'close': np.random.uniform(148, 162, 10),
            'volume': np.random.randint(1000000, 5000000, 10)
        })
        
        # Add some indicators for testing
        data['sma_20'] = data['close'].rolling(window=5).mean()  # Using smaller window for test
        data['rsi'] = np.random.uniform(30, 70, 10)
        
        return data
    
    @pytest.fixture
    def filter_engine(self):
        """Create AdvancedFilterEngine instance"""
        return AdvancedFilterEngine()
    
    def test_init(self):
        """Test AdvancedFilterEngine initialization"""
        engine = AdvancedFilterEngine()
        assert engine.parser is not None
        assert engine.calculator is None
    
    def test_apply_filter_empty_data(self, filter_engine):
        """Test applying filter to empty DataFrame"""
        empty_df = pd.DataFrame()
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 100}
                }
            ]
        }
        
        result = filter_engine.apply_filter(empty_df, json_filter)
        assert result.empty
    
    def test_apply_filter_invalid_data_type(self, filter_engine):
        """Test applying filter with invalid data type"""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            filter_engine.apply_filter("not_a_dataframe", {})
    
    def test_apply_filter_invalid_filter_type(self, filter_engine, sample_data):
        """Test applying filter with invalid filter type"""
        with pytest.raises(TypeError, match="json_filter must be a dictionary"):
            filter_engine.apply_filter(sample_data, "not_a_dict")
    
    def test_apply_filter_simple_condition(self, filter_engine, sample_data):
        """Test applying filter with simple condition"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 150}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        assert not result.empty
        assert all(result['close'] > 150)
    
    def test_apply_filter_column_comparison(self, filter_engine, sample_data):
        """Test applying filter with column comparison"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "column", "name": "open"}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        assert not result.empty
        assert all(result['close'] > result['open'])
    
    def test_apply_filter_with_indicator(self, filter_engine, sample_data):
        """Test applying filter with indicator"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "indicator", "name": "sma", "column": "close", "params": [5]},
                    "operator": ">",
                    "right": {"type": "constant", "value": 150}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        # Result should not be empty, but we can't easily test the exact values
        # without knowing the exact SMA calculations
        assert isinstance(result, pd.DataFrame)
    
    def test_apply_filter_multiple_conditions_and(self, filter_engine, sample_data):
        """Test applying filter with multiple conditions using AND"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 150}
                },
                {
                    "left": {"type": "column", "name": "volume"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 2000000}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        assert not result.empty
        assert all(result['close'] > 150)
        assert all(result['volume'] > 2000000)
    
    def test_apply_filter_multiple_conditions_or(self, filter_engine, sample_data):
        """Test applying filter with multiple conditions using OR"""
        json_filter = {
            "logic": "OR",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 160}
                },
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": "<",
                    "right": {"type": "constant", "value": 140}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        assert not result.empty
        # Check that all rows satisfy either condition
        close_values = result['close']
        assert all((close_values > 160) | (close_values < 140))
    
    def test_evaluate_condition_invalid_structure(self, filter_engine, sample_data):
        """Test evaluating condition with invalid structure"""
        with pytest.raises(KeyError, match="Missing required field"):
            filter_engine.evaluate_condition(sample_data, {})
    
    def test_evaluate_condition_invalid_operator(self, filter_engine, sample_data):
        """Test evaluating condition with invalid operator"""
        condition = {
            "left": {"type": "column", "name": "close"},
            "operator": "INVALID",
            "right": {"type": "constant", "value": 150}
        }
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            filter_engine.evaluate_condition(sample_data, condition)
    
    def test_combine_results_empty_list(self, filter_engine):
        """Test combining results with empty list"""
        with pytest.raises(ValueError, match="Cannot combine empty results list"):
            filter_engine.combine_results([], "AND")
    
    def test_combine_results_invalid_logic(self, filter_engine):
        """Test combining results with invalid logic operator"""
        results = [pd.Series([True, False, True])]
        
        with pytest.raises(ValueError, match="Invalid logic operator"):
            filter_engine.combine_results(results, "INVALID")
    
    def test_combine_results_single_result(self, filter_engine):
        """Test combining results with single result"""
        results = [pd.Series([True, False, True])]
        result = filter_engine.combine_results(results, "AND")
        assert result.equals(pd.Series([True, False, True]))
    
    def test_combine_results_and_logic(self, filter_engine):
        """Test combining results with AND logic"""
        results = [
            pd.Series([True, False, True]),
            pd.Series([True, True, False])
        ]
        result = filter_engine.combine_results(results, "AND")
        expected = pd.Series([True, False, False])
        assert result.equals(expected)
    
    def test_combine_results_or_logic(self, filter_engine):
        """Test combining results with OR logic"""
        results = [
            pd.Series([True, False, True]),
            pd.Series([False, True, False])
        ]
        result = filter_engine.combine_results(results, "OR")
        expected = pd.Series([True, True, True])
        assert result.equals(expected)
    
    def test_validate_filter_valid(self, filter_engine):
        """Test validating valid filter"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 150}
                }
            ]
        }
        
        is_valid, error_msg = filter_engine.validate_filter(json_filter)
        assert is_valid
        assert "successful" in error_msg
    
    def test_validate_filter_invalid(self, filter_engine):
        """Test validating invalid filter"""
        json_filter = {
            "logic": "INVALID",
            "conditions": []
        }
        
        is_valid, error_msg = filter_engine.validate_filter(json_filter)
        assert not is_valid
        assert "error" in error_msg.lower()
    
    def test_get_supported_operators(self, filter_engine):
        """Test getting supported operators"""
        operators = filter_engine.get_supported_operators()
        assert isinstance(operators, list)
        assert ">" in operators
        assert "<" in operators
        assert ">=" in operators
        assert "<=" in operators
        assert "==" in operators
        assert "!=" in operators
    
    def test_get_supported_indicators(self, filter_engine, sample_data):
        """Test getting supported indicators"""
        # Initialize calculator first by applying a simple filter
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 0}
                }
            ]
        }
        filter_engine.apply_filter(sample_data, json_filter)
        
        indicators = filter_engine.get_supported_indicators()
        assert isinstance(indicators, list)
        assert 'sma' in indicators
    
    def test_get_supported_columns(self, filter_engine, sample_data):
        """Test getting supported columns"""
        # Initialize calculator first by applying a simple filter
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 0}
                }
            ]
        }
        filter_engine.apply_filter(sample_data, json_filter)
        
        columns = filter_engine.get_supported_columns()
        assert isinstance(columns, list)
        assert 'close' in columns
        assert 'volume' in columns
    
    def test_apply_filter_with_offset(self, filter_engine, sample_data):
        """Test applying filter with offset"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close", "offset": 1},
                    "operator": ">",
                    "right": {"type": "constant", "value": 150}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        # Should handle offset correctly
        assert isinstance(result, pd.DataFrame)
    
    def test_apply_filter_complex_expression(self, filter_engine, sample_data):
        """Test applying filter with complex expression"""
        json_filter = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {"type": "column", "name": "close"},
                    "operator": ">",
                    "right": {"type": "constant", "value": 150}
                },
                {
                    "left": {"type": "indicator", "name": "sma", "column": "close", "params": [5]},
                    "operator": "<",
                    "right": {"type": "column", "name": "close"}
                }
            ]
        }
        
        result = filter_engine.apply_filter(sample_data, json_filter)
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])