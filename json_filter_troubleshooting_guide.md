# JSON-Based Filtering System Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide provides specific scenarios and solutions for common issues encountered when using the JSON-Based Filtering System. Each scenario includes detailed problem descriptions, root cause analysis, step-by-step solutions, and preventive measures.

## Troubleshooting Framework

### Issue Classification

1. **JSON Syntax Errors**: Malformed JSON structure
2. **Schema Validation Errors**: Invalid JSON format or content
3. **Data Processing Errors**: Issues with data handling and transformation
4. **Performance Issues**: Slow execution or resource problems
5. **Indicator Calculation Errors**: Problems with technical indicators
6. **UI Integration Issues**: User interface and interaction problems
7. **Memory and Resource Issues**: System resource limitations
8. **Integration Problems**: Issues with external systems and APIs

### Troubleshooting Methodology

1. **Identify the Problem**: Recognize symptoms and categorize the issue
2. **Gather Information**: Collect relevant logs, error messages, and system information
3. **Analyze Root Cause**: Determine the underlying cause of the issue
4. **Implement Solution**: Apply the appropriate fix or workaround
5. **Verify Resolution**: Confirm the issue is resolved and no new problems are introduced
6. **Document Prevention**: Record preventive measures for future reference

## Scenario 1: JSON Syntax Errors

### Problem Description
Users encounter JSON syntax errors when trying to create or edit filters. The error messages are often cryptic and don't provide clear guidance on how to fix the issue.

**Symptoms:**
- "JSONDecodeError: Expecting value: line X column Y"
- "Invalid JSON format" error messages
- JSON editor shows red highlighting but unclear what's wrong
- Filter application fails with syntax-related errors

**Example Error:**
```
JSONDecodeError: Expecting value: line 3 column 5 (char 42)
```

### Root Cause Analysis
1. **Missing Commas**: Forgetting to add commas between JSON elements
2. **Extra Commas**: Adding trailing commas in invalid positions
3. **Quotation Issues**: Using single quotes instead of double quotes
4. **Bracket Mismatches**: Unclosed or mismatched brackets/braces
5. **Invalid Escape Sequences**: Improper handling of special characters

### Step-by-Step Solutions

#### Solution 1: JSON Syntax Validation

```python
import json
from json.decoder import JSONDecodeError

def validate_json_syntax(json_string: str) -> dict:
    """
    Validate JSON syntax with detailed error reporting.
    
    Args:
        json_string (str): JSON string to validate
        
    Returns:
        dict: Validation result with detailed error information
    """
    try:
        # Basic syntax validation
        parsed_json = json.loads(json_string)
        
        # Additional structural validation
        if not isinstance(parsed_json, dict):
            return {
                "valid": False,
                "error": "JSON must be an object (dictionary)",
                "error_type": "invalid_structure",
                "suggestion": "Wrap your JSON in curly braces {}"
            }
        
        # Check required fields
        required_fields = ["logic", "conditions"]
        missing_fields = [field for field in required_fields if field not in parsed_json]
        
        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "error_type": "missing_fields",
                "suggestion": f"Add the missing field(s): {', '.join(missing_fields)}"
            }
        
        return {
            "valid": True,
            "message": "JSON syntax is valid",
            "parsed_json": parsed_json
        }
        
    except JSONDecodeError as e:
        return {
            "valid": False,
            "error": f"JSON syntax error: {str(e)}",
            "error_type": "syntax_error",
            "line": e.lineno,
            "column": e.colno,
            "position": e.pos,
            "suggestion": get_syntax_suggestion(str(e))
        }

def get_syntax_suggestion(error_msg: str) -> str:
    """Get specific suggestion based on error message."""
    error_lower = error_msg.lower()
    
    if "expecting value" in error_lower:
        return "Check for missing values or incomplete expressions"
    elif "expecting property name" in error_lower:
        return "Check for missing property names or quotes"
    elif "expecting ':' delimiter" in error_lower:
        return "Check for missing colons between keys and values"
    elif "expecting ',' delimiter" in error_lower:
        return "Check for missing commas between elements"
    elif "invalid escape sequence" in error_lower:
        return "Check for invalid escape sequences in strings"
    else:
        return "Review your JSON syntax carefully"

# Usage example
def json_syntax_validation_example():
    """Example of JSON syntax validation."""
    # Test cases
    test_cases = [
        '{"logic": "AND", "conditions": [{"left": {"type": "column", "name": "close"}, "operator": ">", "right": {"type": "constant", "value": 100}}]}',  # Valid
       
