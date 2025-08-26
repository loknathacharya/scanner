import unittest
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from json_filter_ui import JSONFilterUI
from json_filter_parser import JSONFilterParser

# Mock Streamlit before importing
mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
mock_st.subheader = MagicMock()
mock_st.write = MagicMock()
mock_st.success = MagicMock()
mock_st.error = MagicMock()
mock_st.warning = MagicMock()
mock_st.info = MagicMock()
mock_st.metric = MagicMock()
mock_st.spinner = MagicMock()
mock_st.dataframe = MagicMock()
mock_st.download_button = MagicMock()
mock_st.expander = MagicMock(return_value=MagicMock())
mock_st.code = MagicMock()
mock_st.selectbox = MagicMock()
mock_st.button = MagicMock()
mock_st.text_area = MagicMock()

# Patch Streamlit in the json_filter_ui module
sys.modules['streamlit'] = mock_st

class TestJSONFilterUI(unittest.TestCase):
    """Test cases for JSONFilterUI class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ui = JSONFilterUI()
        self.parser = JSONFilterParser()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'symbol': ['AAPL'] * 100,
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(155, 205, 100),
            'low': np.random.uniform(145, 195, 100),
            'close': np.random.uniform(152, 202, 100),
            'volume': np.random.uniform(1000000, 5000000, 100),
            'sma_20': np.random.uniform(160, 190, 100),
            'sma_50': np.random.uniform(155, 195, 100),
            'rsi': np.random.uniform(20, 80, 100)
        })
    
    def test_init(self):
        """Test JSONFilterUI initialization"""
        self.assertIsInstance(self.ui.parser, JSONFilterParser)
        self.assertIsInstance(self.ui.example_filters, dict)
        self.assertIn('basic_filters', self.ui.example_filters)
        self.assertIn('technical_indicators', self.ui.example_filters)
        self.assertIn('complex_patterns', self.ui.example_filters)
        self.assertIn('volume_analysis', self.ui.example_filters)
    
    def test_render_json_editor_valid_json(self):
        """Test JSON editor with valid JSON input"""
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
                        "value": 100.0
                    }
                }
            ]
        }
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            mock_st.text_area.return_value = json.dumps(valid_json, indent=2)
            mock_st.button.return_value = True
            
            result = self.ui.render_json_editor()
            
            if result is not None:
                self.assertEqual(result['logic'], "AND")
                self.assertEqual(len(result['conditions']), 1)
                self.assertEqual(result['conditions'][0]['operator'], ">")
    
    def test_render_json_editor_invalid_json(self):
        """Test JSON editor with invalid JSON input"""
        invalid_json = '{"logic": "AND", "conditions": [{"left": {"type": "column", "name": "close"}, "operator": ">", "right": {"type": "constant", "value": 100.0}'
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            mock_st.text_area.return_value = invalid_json
            mock_st.button.return_value = True
            
            result = self.ui.render_json_editor()
            
            self.assertIsNone(result)
    
    def test_render_validation_feedback_valid(self):
        """Test validation feedback with valid JSON"""
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
                        "value": 100.0
                    }
                }
            ]
        }
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            self.ui.render_validation_feedback(valid_json)
            
            # Check that success message was called
            mock_st.success.assert_called()
    
    def test_render_validation_feedback_invalid(self):
        """Test validation feedback with invalid JSON"""
        invalid_json = {
            "logic": "INVALID_LOGIC",
            "conditions": []
        }
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            self.ui.render_validation_feedback(invalid_json)
            
            # Check that error message was called
            mock_st.error.assert_called()
    
    def test_render_filter_preview(self):
        """Test filter preview functionality"""
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
                        "value": 100.0
                    }
                }
            ]
        }
        
        # Mock Streamlit components and AdvancedFilterEngine
        with patch('json_filter_ui.st') as mock_st, \
             patch('advanced_filter_engine.AdvancedFilterEngine') as mock_engine:
            
            # Mock the filter engine
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance
            mock_engine_instance.apply_filter.return_value = self.sample_data.head(10)
            
            self.ui.render_filter_preview(valid_json, self.sample_data)
            
            # Check that spinner and metrics were used
            mock_st.spinner.assert_called()
            mock_st.metric.assert_called()
    
    def test_render_filter_preview_no_data(self):
        """Test filter preview with no data"""
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
                        "value": 100.0
                    }
                }
            ]
        }
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            self.ui.render_filter_preview(valid_json, pd.DataFrame())
            
            # Check that info message was shown
            mock_st.info.assert_called()
    
    def test_get_example_filters(self):
        """Test getting example filters"""
        examples = self.ui.get_example_filters()
        
        self.assertIsInstance(examples, dict)
        self.assertIn('basic_filters', examples)
        self.assertIn('technical_indicators', examples)
        self.assertIn('complex_patterns', examples)
        self.assertIn('volume_analysis', examples)
        
        # Check that each category has examples
        for category in examples.values():
            self.assertIsInstance(category, dict)
            self.assertGreater(len(category), 0)
    
    def test_render_example_selector(self):
        """Test example selector rendering"""
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            mock_st.selectbox.side_effect = ["Basic Filters", "Price Above $100"]
            
            self.ui.render_example_selector()
            
            # Check that selectbox was called
            self.assertEqual(mock_st.selectbox.call_count, 2)
            mock_st.code.assert_called()
    
    def test_render_json_structure_help(self):
        """Test JSON structure help rendering"""
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            self.ui.render_json_structure_help()
            
            # Check that expander was called
            mock_st.expander.assert_called()
    
    def test_load_example_filters(self):
        """Test loading example filters"""
        examples = self.ui._load_example_filters()
        
        # Check structure
        self.assertIsInstance(examples, dict)
        self.assertIn('basic_filters', examples)
        self.assertIn('technical_indicators', examples)
        self.assertIn('complex_patterns', examples)
        self.assertIn('volume_analysis', examples)
        
        # Check basic filters
        basic_filters = examples['basic_filters']
        self.assertIn('Price Above $100', basic_filters)
        self.assertIn('Price Increase', basic_filters)
        self.assertIn('High Volume', basic_filters)
        
        # Check technical indicators
        tech_indicators = examples['technical_indicators']
        self.assertIn('Golden Cross', tech_indicators)
        self.assertIn('RSI Overbought', tech_indicators)
        self.assertIn('RSI Oversold', tech_indicators)
        
        # Check complex patterns
        complex_patterns = examples['complex_patterns']
        self.assertIn('Bullish Confirmation', complex_patterns)
        self.assertIn('OR Logic Example', complex_patterns)
        
        # Check volume analysis
        volume_analysis = examples['volume_analysis']
        self.assertIn('Volume Breakout', volume_analysis)
        self.assertIn('High Volume Spike', volume_analysis)
    
    def test_perform_additional_validations(self):
        """Test additional validation logic"""
        # Test with valid conditions
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
                        "value": 100.0
                    }
                }
            ]
        }
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            self.ui._perform_additional_validations(valid_json)
            
            # Should not show warnings for valid conditions
            mock_st.warning.assert_not_called()
        
        # Test with empty conditions
        empty_json = {
            "logic": "AND",
            "conditions": []
        }
        
        with patch('json_filter_ui.st') as mock_st:
            self.ui._perform_additional_validations(empty_json)
            
            # Should show warning for empty conditions
            mock_st.warning.assert_called()
        
        # Test with constant comparison
        constant_json = {
            "logic": "AND",
            "conditions": [
                {
                    "left": {
                        "type": "constant",
                        "value": 100.0
                    },
                    "operator": ">",
                    "right": {
                        "type": "constant",
                        "value": 50.0
                    }
                }
            ]
        }
        
        with patch('json_filter_ui.st') as mock_st:
            self.ui._perform_additional_validations(constant_json)
            
            # Should show info for constant comparison
            mock_st.info.assert_called()
    
    def test_display_condition_details(self):
        """Test condition details display"""
        condition = {
            "left": {
                "type": "column",
                "name": "close",
                "timeframe": "daily",
                "offset": 0
            },
            "operator": ">",
            "right": {
                "type": "constant",
                "value": 100.0
            }
        }
        
        # Mock Streamlit components
        with patch('json_filter_ui.st') as mock_st:
            self.ui._display_condition_details(condition)
            
            # Check that write was called with expected content
            mock_st.write.assert_called()
    
    def test_integration_with_parser(self):
        """Test integration with JSONFilterParser"""
        # Test that the UI uses the parser correctly
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
                        "value": 100.0
                    }
                }
            ]
        }
        
        # Test validation
        is_valid, error_msg = self.parser.validate_json(valid_json)
        self.assertTrue(is_valid)
        
        # Test that UI uses the same validation
        with patch('json_filter_ui.st') as mock_st:
            self.ui.render_validation_feedback(valid_json)
            mock_st.success.assert_called()

if __name__ == '__main__':
    unittest.main()