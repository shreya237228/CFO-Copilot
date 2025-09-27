"""
Tests for CFO Copilot Agent
"""

import pytest
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.cfo_agent import CFOAgent
from agent.planner import CFOAgentPlanner, IntentType

def test_planner_intent_classification():
    """Test intent classification"""
    planner = CFOAgentPlanner()
    
    # Test revenue vs budget intent
    intent = planner.classify_intent("What was June 2025 revenue vs budget?")
    assert intent.type == IntentType.REVENUE_VS_BUDGET
    assert intent.confidence >= 0.5
    
    # Test gross margin intent
    intent = planner.classify_intent("Show gross margin trend")
    assert intent.type == IntentType.GROSS_MARGIN_TREND
    assert intent.confidence >= 0.5
    
    # Test opex breakdown intent
    intent = planner.classify_intent("Break down Opex by category")
    assert intent.type == IntentType.OPEX_BREAKDOWN
    assert intent.confidence >= 0.5
    
    # Test cash runway intent
    intent = planner.classify_intent("What is our cash runway?")
    assert intent.type == IntentType.CASH_RUNWAY
    assert intent.confidence >= 0.5

def test_agent_initialization():
    """Test agent initialization"""
    agent = CFOAgent()
    assert agent is not None
    assert agent.planner is not None
    assert agent.calculator is not None

def test_agent_query_processing():
    """Test basic query processing"""
    agent = CFOAgent()
    
    # Test a simple revenue question
    response = agent.process_query("What was revenue vs budget for June 2025?")
    
    assert response is not None
    assert 'success' in response
    assert 'intent' in response
    assert 'result' in response
    
    # The result might have an error if data is missing, but the structure should be correct
    if response['success']:
        assert response['intent'] in [intent.value for intent in IntentType]

def test_data_loading():
    """Test data loading functionality"""
    agent = CFOAgent()
    
    # Check if data was loaded
    data_info = agent.get_available_metrics()
    assert data_info is not None
    assert 'success' in data_info
    
    # If data loading was successful, check structure
    if data_info['success']:
        assert 'data_info' in data_info
        assert 'available_metrics' in data_info

if __name__ == "__main__":
    pytest.main([__file__])
