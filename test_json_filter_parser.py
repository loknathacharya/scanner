import unittest
import json
import pandas as pd
import sys
import os

# Add the current directory to the path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from json_filter_parser import JSONFilterParser


class TestJSONFilterParser(unittest.TestCase):
    """Test cases for JSONFilterParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = JSONFilterParser()
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'close': [100, 102, 105, 103, 108, 110, 107, 112, 115, 118],
            'volume': [1000000, 1200000, 950000, 1100000, 1300000, 1250000, 1150000, 1400000, 1350000, 1500000],
            'open': [99, 101, 103, 104, 106, 109, 108, 111, 113, 116]
        })
    
    def test_init(self):
        """Test initialization of JSONFilterParser."""
        self.assertIsNotNone(self.parser.schema)
        self.assertIsInstance(self.parser.schema, dict)
    
    def test_valid_json_schema(self):
        """Test valid JSON schema validation."""
        valid_json = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {
                        "type": "column",
                        "name": "close",
                        "timeframe": "daily",
                        "offset": 0
                    },
                    "operator": ">",
                    "right": {
                        "type": "constant",
                        "value": 100
                    }
                }
            ]
        }
        
        is_valid, message = self.parser.validate_json(valid_json)
        self.assertTrue(is_valid)
        self.assertEqual(message, "JSON validation successful")
    
    def test_invalid_json_missing_logic(self):
        """Test JSON validation with missing logic field."""
        invalid_json = {
            "conditions": [
                {
                    "left": {
                        "type": "column",
                        "name": "close"
                    },
                    "operator": ">",
                    "right": {
                        "type": "constant",
                        "value": 100
                    }
                }
            ]
        }
        
        is_valid, message = self.parser.validate_json(invalid_json)
        self.assertFalse(is_valid)
        self.assertIn("JSON validation error", message)
    
    def test_invalid_json_invalid_operator(self):
        """Test JSON validation with invalid operator."""
        invalid_json = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {
                        "type": "column",
                        "name": "close"
                    },
                    "operator": "INVALID",
                    "right": {
                        "type": "constant",
                        "value": 100
                    }
                }
            ]
        }
        
        is_valid, message = self.parser.validate_json(invalid_json)
        self.assertFalse(is_valid)
        self.assertIn("JSON validation error", message)
    
    def test_parse_column_operand(self):
        """Test parsing column operand."""
        operand_data = {
            "type": "column",
            "name": "close",
            "timeframe": "daily",
            "offset": 1
        }
        
        parsed = self.parser.parse_operands(operand_data)
        
        self.assertEqual(parsed["type"], "column")
        self.assertEqual(parsed["name"], "close")
        self.assertEqual(parsed["timeframe"], "daily")
        self.assertEqual(parsed["offset"], 1)
    
    def test_parse_indicator_operand(self):
        """Test parsing indicator operand."""
        operand_data = {
            "type": "indicator",
            "name": "sma",
            "params": [20],
            "column": "close",
            "timeframe": "daily",
            "offset": 0
        }
        
        parsed = self.parser.parse_operands(operand_data)
        
        self.assertEqual(parsed["type"], "indicator")
        self.assertEqual(parsed["name"], "sma")
        self.assertEqual(parsed["params"], [20])
        self.assertEqual(parsed["column"], "close")
    
    def test_parse_constant_operand(self):
        """Test parsing constant operand."""
        operand_data = {
            "type": "constant",
            "value": 100.5
        }
        
        parsed = self.parser.parse_operands(operand_data)
        
        self.assertEqual(parsed["type"], "constant")
        self.assertEqual(parsed["value"], 100.5)
    
    def test_parse_invalid_operand_type(self):
        """Test parsing invalid operand type."""
        operand_data = {
            "type": "invalid_type",
            "name": "close"
        }
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse_operands(operand_data)
        
        self.assertIn("Invalid operand type", str(context.exception))
    
    def test_parse_column_missing_name(self):
        """Test parsing column operand missing name field."""
        operand_data = {
            "type": "column",
            "timeframe": "daily"
        }
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse_operands(operand_data)
        
        self.assertIn("Column operand requires 'name' field", str(context.exception))
    
    def test_parse_indicator_missing_column(self):
        """Test parsing indicator operand missing column field."""
        operand_data = {
            "type": "indicator",
            "name": "sma",
            "params": [20]
        }
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse_operands(operand_data)
        
        self.assertIn("Indicator operand requires 'column' field", str(context.exception))
    
    def test_parse_constant_missing_value(self):
        """Test parsing constant operand missing value field."""
        operand_data = {
            "type": "constant"
        }
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse_operands(operand_data)
        
        self.assertIn("Constant operand requires 'value' field", str(context.exception))
    
    def test_build_filter_expression_single_condition(self):
        """Test building filter expression with single condition."""
        conditions = [
            {
                "left": {
                    "type": "column",
                    "name": "close",
                    "offset": 0
                },
                "operator": ">",
                "right": {
                    "type": "constant",
                    "value": 100
                }
            }
        ]
        
        expression = self.parser.build_filter_expression(conditions, "AND")
        
        self.assertEqual(expression, "(data['close'].iloc[0] > 100)")
    
    def test_build_filter_expression_multiple_conditions_and(self):
        """Test building filter expression with multiple conditions using AND."""
        conditions = [
            {
                "left": {
                    "type": "column",
                    "name": "close",
                    "offset": 0
                },
                "operator": ">",
                "right": {
                    "type": "constant",
                    "value": 100
                }
            },
            {
                "left": {
                    "type": "column",
                    "name": "volume",
                    "offset": 0
                },
                "operator": ">",
                "right": {
                    "type": "constant",
                    "value": 1000000
                }
            }
        ]
        
        expression = self.parser.build_filter_expression(conditions, "AND")
        
        expected = "(data['close'].iloc[0] > 100) AND (data['volume'].iloc[0] > 1000000)"
        self.assertEqual(expression, expected)
    
    def test_build_filter_expression_multiple_conditions_or(self):
        """Test building filter expression with multiple conditions using OR."""
        conditions = [
            {
                "left": {
                    "type": "column",
                    "name": "close",
                    "offset": 0
                },
                "operator": ">",
                "right": {
                    "type": "constant",
                    "value": 100
                }
            },
            {
                "left": {
                    "type": "column",
                    "name": "volume",
                    "offset": 0
                },
                "operator": ">",
                "right": {
                    "type": "constant",
                    "value": 1000000
                }
            }
        ]
        
        expression = self.parser.build_filter_expression(conditions, "OR")
        
        expected = "(data['close'].iloc[0] > 100) OR (data['volume'].iloc[0] > 1000000)"
        self.assertEqual(expression, expected)
    
    def test_build_filter_expression_with_indicators(self):
        """Test building filter expression with indicator operands."""
        conditions = [
            {
                "left": {
                    "type": "indicator",
                    "name": "sma",
                    "params": [20],
                    "column": "close",
                    "offset": 0
                },
                "operator": ">",
                "right": {
                    "type": "column",
                    "name": "close",
                    "offset": 0
                }
            }
        ]
        
        expression = self.parser.build_filter_expression(conditions, "AND")
        
        expected = "(self._calculate_indicator('sma', data['close'], [20]) > data['close'].iloc[0])"
        self.assertEqual(expression, expected)
    
    def test_build_filter_expression_invalid_logic(self):
        """Test building filter expression with invalid logic operator."""
        conditions = [
            {
                "left": {
                    "type": "column",
                    "name": "close"
                },
                "operator": ">",
                "right": {
                    "type": "constant",
                    "value": 100
                }
            }
        ]
        
        with self.assertRaises(ValueError) as context:
            self.parser.build_filter_expression(conditions, "INVALID")
        
        self.assertIn("Invalid logic operator", str(context.exception))
    
    def test_build_filter_expression_empty_conditions(self):
        """Test building filter expression with empty conditions."""
        conditions = []
        
        with self.assertRaises(ValueError) as context:
            self.parser.build_filter_expression(conditions, "AND")
        
        self.assertIn("Cannot build expression with empty conditions", str(context.exception))
    
    def test_get_supported_operators(self):
        """Test getting supported operators."""
        operators = self.parser.get_supported_operators()
        
        expected_operators = [">", "<", ">=", "<=", "==", "!="]
        self.assertEqual(operators, expected_operators)
    
    def test_get_supported_indicators(self):
        """Test getting supported indicators."""
        indicators = self.parser.get_supported_indicators()
        
        expected_indicators = ["sma", "ema"]
        self.assertEqual(indicators, expected_indicators)
    
    def test_get_supported_timeframes(self):
        """Test getting supported timeframes."""
        timeframes = self.parser.get_supported_timeframes()
        
        expected_timeframes = ["daily", "weekly", "intraday"]
        self.assertEqual(timeframes, expected_timeframes)
    
    def test_complex_json_validation(self):
        """Test validation of complex JSON with multiple conditions and indicators."""
        complex_json = {
            "logic": "OR",
            "conditions": [
                {
                    "left": {
                        "type": "column",
                        "name": "close",
                        "timeframe": "daily",
                        "offset": 0
                    },
                    "operator": ">",
                    "right": {
                        "type": "indicator",
                        "name": "sma",
                        "params": [50],
                        "column": "close",
                        "timeframe": "daily",
                        "offset": 0
                    }
                },
                {
                    "left": {
                        "type": "indicator",
                        "name": "ema",
                        "params": [20],
                        "column": "close",
                        "timeframe": "daily",
                        "offset": 0
                    },
                    "operator": "<",
                    "right": {
                        "type": "constant",
                        "value": 105.0
                    }
                }
            ]
        }
        
        is_valid, message = self.parser.validate_json(complex_json)
        self.assertTrue(is_valid)
        self.assertEqual(message, "JSON validation successful")


if __name__ == '__main__':
    unittest.main()