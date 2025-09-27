"""
CFO Copilot Agent
Main agent that orchestrates intent classification, data operations, and response generation
"""

import os
from typing import Dict, Any, Optional
from .planner import CFOAgentPlanner, IntentType
from .data_tools import DataLoader, FinancialCalculator

class CFOAgent:
    """Main CFO Copilot Agent"""
    
    def __init__(self, fixtures_dir: str = "fixtures"):
        self.fixtures_dir = fixtures_dir
        self.planner = CFOAgentPlanner()
        self.data_loader = DataLoader(fixtures_dir)
        self.calculator = FinancialCalculator(self.data_loader)
        
        # Load data on initialization
        self.data_loader.load_all_data()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a CFO query and return structured response"""
        try:
            # Classify intent
            intent = self.planner.classify_intent(query)
            
            # Execute based on intent
            result = self._execute_intent(intent, query)
            
            return {
                "success": True,
                "intent": intent.type.value,
                "confidence": intent.confidence,
                "query": query,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}",
                "query": query
            }
    
    def _execute_intent(self, intent, query: str) -> Dict[str, Any]:
        """Execute the appropriate action based on classified intent"""
        
        if intent.type == IntentType.REVENUE_VS_BUDGET:
            month = intent.parameters.get('month')
            year = intent.parameters.get('year')
            return self.calculator.get_revenue_vs_budget(month, year)
        
        elif intent.type == IntentType.GROSS_MARGIN_TREND:
            period_months = intent.parameters.get('period_months', 3)
            return self.calculator.get_gross_margin_trend(period_months)
        
        elif intent.type == IntentType.OPEX_BREAKDOWN:
            month = intent.parameters.get('month')
            year = intent.parameters.get('year')
            return self.calculator.get_opex_breakdown(month, year)
        
        elif intent.type == IntentType.CASH_RUNWAY:
            return self.calculator.get_cash_runway()
        
        elif intent.type == IntentType.EBITDA:
            month = intent.parameters.get('month')
            year = intent.parameters.get('year')
            return self.calculator.get_ebitda(month, year)
        
        else:
            return self._handle_general_query(query)
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries that don't match specific intents"""
        return {
            "message": "I can help you with financial analysis. Try asking about:",
            "suggestions": [
                "Revenue vs budget for a specific month",
                "Gross margin trends",
                "Opex breakdown by category",
                "Cash runway calculation",
                "EBITDA analysis"
            ],
            "example": "What was June 2025 revenue vs budget in USD?"
        }
    
    def get_available_metrics(self) -> Dict[str, Any]:
        """Return information about available data and metrics"""
        try:
            data_info = {}
            
            for data_type, df in self.data_loader.data.items():
                if not df.empty:
                    data_info[data_type] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "date_range": self._get_date_range(df)
                    }
            
            return {
                "success": True,
                "data_info": data_info,
                "available_metrics": [
                    "Revenue vs Budget",
                    "Gross Margin Trends", 
                    "Opex Breakdown",
                    "Cash Runway",
                    "EBITDA Analysis"
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting data info: {str(e)}"
            }
    
    def _get_date_range(self, df) -> Dict[str, Any]:
        """Extract date range information from dataframe"""
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return {
                    "first_period": numeric_cols[0],
                    "last_period": numeric_cols[-1],
                    "total_periods": len(numeric_cols)
                }
            return {"message": "No numeric columns found"}
        except:
            return {"message": "Unable to determine date range"}
