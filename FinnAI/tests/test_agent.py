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

def test_revenue_vs_budget_calculation():
    """Test revenue vs budget calculation with real data"""
    agent = CFOAgent()
    
    # Test with specific month and year
    response = agent.process_query("What was June 2025 revenue vs budget in USD?")
    
    assert response is not None
    assert response['success'] is True
    assert response['intent'] == 'revenue_vs_budget'
    
    result = response['result']
    
    # Check if we have valid data (not an error)
    if 'error' not in result:
        # Verify the structure of the result
        assert 'actual_revenue' in result
        assert 'budget_revenue' in result
        assert 'variance' in result
        assert 'variance_pct' in result
        
        # Verify data types
        assert isinstance(result['actual_revenue'], (int, float))
        assert isinstance(result['budget_revenue'], (int, float))
        assert isinstance(result['variance'], (int, float))
        assert isinstance(result['variance_pct'], (int, float))
        
        # Verify variance calculation
        expected_variance = result['actual_revenue'] - result['budget_revenue']
        assert abs(result['variance'] - expected_variance) < 0.01
        
        # Verify variance percentage calculation
        if result['budget_revenue'] != 0:
            expected_pct = (result['variance'] / result['budget_revenue']) * 100
            assert abs(result['variance_pct'] - expected_pct) < 0.01

def test_opex_breakdown_calculation():
    """Test OPEX breakdown calculation with real data"""
    agent = CFOAgent()
    
    # Test OPEX breakdown
    response = agent.process_query("Break down Opex by category for June 2025")
    
    assert response is not None
    assert response['success'] is True
    assert response['intent'] == 'opex_breakdown'
    
    result = response['result']
    
    # Check if we have valid data (not an error)
    if 'error' not in result:
        # Verify the structure of the result
        assert 'breakdown' in result
        assert 'breakdown_pct' in result
        assert 'total_opex' in result
        
        # Verify data types
        assert isinstance(result['breakdown'], dict)
        assert isinstance(result['breakdown_pct'], dict)
        assert isinstance(result['total_opex'], (int, float))
        
        # Verify breakdown has categories
        assert len(result['breakdown']) > 0
        
        # Verify percentages sum to 100 (approximately)
        total_percentage = sum(result['breakdown_pct'].values())
        assert abs(total_percentage - 100.0) < 0.1
        
        # Verify breakdown values sum to total
        total_breakdown = sum(result['breakdown'].values())
        assert abs(total_breakdown - result['total_opex']) < 0.01
        
        # Verify all values are positive
        for category, amount in result['breakdown'].items():
            assert amount >= 0
            assert result['breakdown_pct'][category] >= 0

if __name__ == "__main__":
    pytest.main([__file__])
