"""
CFO Copilot Agent Planner
Handles intent classification and orchestrates data operations
"""

import re
import openai
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IntentType(Enum):
    REVENUE_VS_BUDGET = "revenue_vs_budget"
    GROSS_MARGIN_TREND = "gross_margin_trend"
    OPEX_BREAKDOWN = "opex_breakdown"
    CASH_RUNWAY = "cash_runway"
    EBITDA = "ebitda"
    GENERAL_QUERY = "general_query"

@dataclass
class Intent:
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    parameters: Dict[str, Any]

class CFOAgentPlanner:
    """Main agent planner that classifies intents and extracts parameters"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.openai_client = None
        
        if use_llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
            else:
                print("⚠️ OpenAI API key not found. Falling back to rule-based classification.")
                self.use_llm = False
        
        # Fallback rule-based patterns
        self.intent_patterns = {
            IntentType.REVENUE_VS_BUDGET: [
                r"revenue.*budget",
                r"revenue.*vs.*budget",
                r"actual.*revenue",
                r"budget.*revenue"
            ],
            IntentType.GROSS_MARGIN_TREND: [
                r"gross margin",
                r"margin.*trend",
                r"gross.*margin.*%"
            ],
            IntentType.OPEX_BREAKDOWN: [
                r"opex",
                r"operating expense",
                r"expense.*breakdown",
                r"opex.*category"
            ],
            IntentType.CASH_RUNWAY: [
                r"cash runway",
                r"runway",
                r"cash.*burn",
                r"months.*cash"
            ],
            IntentType.EBITDA: [
                r"ebitda",
                r"earnings.*before.*interest"
            ]
        }
    
    def classify_intent(self, query: str) -> Intent:
        """Classify the user's intent from their query using LLM or fallback to rules"""
        
        if self.use_llm and self.openai_client:
            return self._classify_with_llm(query)
        else:
            return self._classify_with_rules(query)
    
    def _classify_with_llm(self, query: str) -> Intent:
        """Use LLM for intent classification"""
        try:
            prompt = f"""
            You are a CFO assistant. Classify the following financial query into one of these intents:
            
            1. revenue_vs_budget - Questions comparing actual revenue to budget
            2. gross_margin_trend - Questions about gross margin trends over time
            3. opex_breakdown - Questions about operating expense breakdowns
            4. cash_runway - Questions about cash runway/burn rate
            5. ebitda - Questions about EBITDA analysis
            6. general_query - Other general questions
            
            Also extract any entities like months, years, and currencies.
            
            Query: "{query}"
            
            Respond in JSON format:
            {{
                "intent": "intent_name",
                "confidence": 0.95,
                "entities": {{"month": "june", "year": 2025, "currency": "USD"}},
                "parameters": {{"month": "june", "year": 2025}}
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return Intent(
                type=IntentType(result['intent']),
                confidence=result['confidence'],
                entities=result.get('entities', {}),
                parameters=result.get('parameters', {})
            )
            
        except Exception as e:
            print(f"⚠️ LLM classification failed: {e}. Falling back to rules.")
            return self._classify_with_rules(query)
    
    def _classify_with_rules(self, query: str) -> Intent:
        """Fallback rule-based classification"""
        query_lower = query.lower()
        
        # Calculate confidence scores for each intent
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent_type] = score / len(patterns)
        
        # Find the best intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if best_intent[1] > 0:
            intent_type = best_intent[0]
            confidence = best_intent[1]
        else:
            intent_type = IntentType.GENERAL_QUERY
            confidence = 0.1
        
        # Extract entities and parameters
        entities = self._extract_entities(query)
        parameters = self._extract_parameters(query, intent_type)
        
        return Intent(
            type=intent_type,
            confidence=confidence,
            entities=entities,
            parameters=parameters
        )
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities like dates, currencies, etc."""
        entities = {}
        
        # Extract month/year patterns
        month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
        year_pattern = r'\b(20\d{2})\b'
        
        months = re.findall(month_pattern, query.lower())
        years = re.findall(year_pattern, query.lower())
        
        if months:
            entities['month'] = months[0]
        if years:
            entities['year'] = int(years[0])
        
        # Extract currency
        if 'usd' in query.lower() or 'dollar' in query.lower():
            entities['currency'] = 'USD'
        
        return entities
    
    def _extract_parameters(self, query: str, intent_type: IntentType) -> Dict[str, Any]:
        """Extract specific parameters based on intent type"""
        parameters = {}
        
        if intent_type == IntentType.GROSS_MARGIN_TREND:
            # Look for time period specifications
            if 'last' in query.lower():
                numbers = re.findall(r'\b(\d+)\b', query.lower())
                if numbers:
                    parameters['period_months'] = int(numbers[0])
            elif 'months' in query.lower():
                numbers = re.findall(r'\b(\d+)\b', query.lower())
                if numbers:
                    parameters['period_months'] = int(numbers[0])
        
        elif intent_type == IntentType.REVENUE_VS_BUDGET:
            # Extract specific month/year if mentioned
            month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
            year_pattern = r'\b(20\d{2})\b'
            
            months = re.findall(month_pattern, query.lower())
            years = re.findall(year_pattern, query.lower())
            
            if months:
                parameters['month'] = months[0]
            if years:
                parameters['year'] = int(years[0])
        
        elif intent_type == IntentType.OPEX_BREAKDOWN:
            # Extract specific month/year if mentioned
            month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
            year_pattern = r'\b(20\d{2})\b'
            
            months = re.findall(month_pattern, query.lower())
            years = re.findall(year_pattern, query.lower())
            
            if months:
                parameters['month'] = months[0]
            if years:
                parameters['year'] = int(years[0])
        
        elif intent_type == IntentType.EBITDA:
            # Extract specific month/year if mentioned
            month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
            year_pattern = r'\b(20\d{2})\b'
            
            months = re.findall(month_pattern, query.lower())
            years = re.findall(year_pattern, query.lower())
            
            if months:
                parameters['month'] = months[0]
            if years:
                parameters['year'] = int(years[0])
        
        return parameters
