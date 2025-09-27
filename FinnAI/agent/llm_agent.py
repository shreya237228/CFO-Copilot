"""
LLM-Enhanced CFO Agent
Uses Google Gemini API for more intelligent responses and reasoning
"""

import google.generativeai as genai
import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from .cfo_agent import CFOAgent
from .planner import IntentType
from dotenv import load_dotenv

load_dotenv()

class LLMEnhancedCFOAgent(CFOAgent):
    """Enhanced CFO Agent with LLM-powered reasoning and responses"""
    
    def __init__(self, fixtures_dir: str = "fixtures", use_llm: bool = True):
        super().__init__(fixtures_dir)
        
        self.use_llm = use_llm
        self.gemini_model = None
        self.conversation_history = []
        
        if use_llm:
            # Force reload environment variables
            load_dotenv(override=True)
            api_key = os.getenv('GEMINI_API_KEY')
            print(f"🔑 API key loaded: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")
            
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                    print("🤖 LLM-enhanced CFO Agent initialized with Gemini")
                except Exception as e:
                    print(f"⚠️ Gemini initialization failed: {e}")
                    self.use_llm = False
            else:
                print("⚠️ Gemini API key not found. Using rule-based agent.")
                self.use_llm = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a CFO query with LLM-enhanced reasoning"""
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        if self.use_llm and self.gemini_model:
            return self._process_with_llm(query)
        else:
            # Fallback to base agent
            return super().process_query(query)
    
    def _process_with_llm(self, query: str) -> Dict[str, Any]:
        """Process query using LLM with full dataset access"""
        
        try:
            # Get comprehensive data summary for LLM
            data_summary = self._build_comprehensive_data_summary()
            
            # Use LLM to answer the question directly
            llm_response = self._generate_llm_response_with_data(query, data_summary)
            
            return llm_response
            
        except Exception as e:
            print(f"⚠️ LLM processing failed: {e}. Falling back to base agent.")
            return super().process_query(query)
    
    def _build_comprehensive_data_summary(self) -> str:
        """Build a comprehensive summary of all available financial data"""
        
        try:
            data_summary = "=== COMPLETE FINANCIAL DATASET ===\n\n"
            
            # Actuals data - include ALL data
            actuals = self.data_loader.data.get('actuals', pd.DataFrame())
            if not actuals.empty:
                data_summary += "ACTUALS DATA (ALL RECORDS):\n"
                for _, row in actuals.iterrows():
                    data_summary += f"  {row['month']} | {row['entity']} | {row['account_category']} | {row['amount']} {row['currency']}\n"
                data_summary += "\n"
            
            # Budget data - include ALL data
            budget = self.data_loader.data.get('budget', pd.DataFrame())
            if not budget.empty:
                data_summary += "BUDGET DATA (ALL RECORDS):\n"
                for _, row in budget.iterrows():
                    data_summary += f"  {row['month']} | {row['entity']} | {row['account_category']} | {row['amount']} {row['currency']}\n"
                data_summary += "\n"
            
            # FX rates - include ALL data
            fx_data = self.data_loader.data.get('fx', pd.DataFrame())
            if not fx_data.empty:
                data_summary += "FX RATES (ALL RECORDS):\n"
                for _, row in fx_data.iterrows():
                    data_summary += f"  {row['month']} | {row['currency']} | {row['rate_to_usd']} USD\n"
                data_summary += "\n"
            
            # Cash data - include ALL data
            cash_data = self.data_loader.data.get('cash', pd.DataFrame())
            if not cash_data.empty:
                data_summary += "CASH DATA (ALL RECORDS):\n"
                for _, row in cash_data.iterrows():
                    data_summary += f"  {row['month']} | {row['entity']} | {row['cash_usd']} USD\n"
                data_summary += "\n"
            
            return data_summary
            
        except Exception as e:
            return f"Error building data summary: {e}"
    
    def _generate_llm_response_with_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Generate LLM response with full dataset access"""
        
        try:
            system_prompt = """You are an expert CFO assistant with access to the complete financial dataset. Your role is to:

1. Answer ANY financial question using the EXACT data provided below
2. Use the ACTUAL numbers from the dataset - never assume or estimate
3. Perform calculations using the real data values
4. Convert currencies using the provided FX rates
5. Compare actuals vs budget using the real numbers
6. Calculate metrics like margins, growth rates, etc. using actual data
7. Be precise with numbers and professional in tone

CRITICAL INSTRUCTIONS:
- Use ONLY the data provided in the dataset below
- Do NOT assume any values - if data is missing, say so
- Show your calculations using the actual numbers
- Convert currencies using the exact FX rates provided
- Be specific with month, entity, and account details
- NEVER make up numbers or use "assume" or "let's say"

DATE FORMAT UNDERSTANDING:
- Data uses format "YYYY-MM" (e.g., "2023-06" for June 2023)
- When user asks for "June 2023", look for "2023-06" in the data
- When user asks for "2025-12", that means December 2025
- Match the exact date format in the dataset

You have access to the complete dataset including:
- All actuals records with exact amounts and currencies
- All budget records with exact amounts and currencies  
- All FX rates for currency conversion
- All cash balances by entity and month
- Complete historical data for trend analysis"""
            
            user_prompt = f"""
            COMPLETE FINANCIAL DATASET:
            {data_summary}
            
            User Question: "{query}"
            
            Instructions:
            1. Find the EXACT data values in the dataset above
            2. Use the real numbers - do not assume anything
            3. Show your calculations step by step
            4. Convert currencies using the exact FX rates provided
            5. Provide specific numbers, not estimates
            6. If data is missing, clearly state what's not available
            7. Match date formats exactly (e.g., "June 2023" = "2023-06")
            
            Answer the question using only the data provided above.
            """
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more precise financial analysis
                    max_output_tokens=2000
                ),
                request_options={"timeout": 30}  # 30 second timeout
            )
            
            llm_analysis = response.text
            
            # Add LLM response to conversation history
            self.conversation_history.append({"role": "assistant", "content": llm_analysis})
            
            # Determine if charts would be helpful
            chart_data = self._extract_chart_data(query, data_summary)
            
            return {
                "success": True,
                "intent": "llm_analysis",
                "confidence": 0.95,
                "query": query,
                "result": {
                    "llm_analysis": llm_analysis,
                    "data_used": "comprehensive_dataset",
                    "analysis_type": "direct_llm_analysis",
                    "chart_data": chart_data
                },
                "llm_enhanced": True
            }
            
        except Exception as e:
            print(f"⚠️ LLM response generation failed: {e}")
            return super().process_query(query)
    
    def _extract_chart_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract data for chart generation based on query type"""
        
        try:
            query_lower = query.lower()
            chart_data = {"chart_type": None, "data": {}}
            
            # Revenue vs Budget comparison
            if any(word in query_lower for word in ['revenue', 'budget', 'vs', 'comparison', 'actual']):
                chart_data["chart_type"] = "revenue_vs_budget"
                chart_data["data"] = self._extract_revenue_budget_data(query, data_summary)
            
            # OPEX breakdown
            elif any(word in query_lower for word in ['opex', 'operating expense', 'expense', 'breakdown', 'category']):
                chart_data["chart_type"] = "opex_breakdown"
                chart_data["data"] = self._extract_opex_data(query, data_summary)
            
            # Trend analysis
            elif any(word in query_lower for word in ['trend', 'over time', 'monthly', 'quarterly', 'growth']):
                chart_data["chart_type"] = "trend"
                chart_data["data"] = self._extract_trend_data(query, data_summary)
            
            return chart_data
            
        except Exception as e:
            print(f"⚠️ Chart data extraction failed: {e}")
            return {"chart_type": None, "data": {}}
    
    def _extract_revenue_budget_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract revenue vs budget data for charting"""
        
        try:
            # Parse the data summary to extract actual numbers
            actuals_data = []
            budget_data = []
            
            lines = data_summary.split('\n')
            in_actuals = False
            in_budget = False
            
            for line in lines:
                if 'ACTUALS DATA' in line:
                    in_actuals = True
                    in_budget = False
                    continue
                elif 'BUDGET DATA' in line:
                    in_actuals = False
                    in_budget = True
                    continue
                elif 'FX RATES' in line or 'CASH DATA' in line:
                    in_actuals = False
                    in_budget = False
                    continue
                
                if in_actuals and 'Revenue' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        month = parts[0].strip()
                        entity = parts[1].strip()
                        amount = float(parts[3].strip().split()[0])
                        currency = parts[3].strip().split()[1]
                        actuals_data.append({'month': month, 'entity': entity, 'amount': amount, 'currency': currency})
                
                elif in_budget and 'Revenue' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        month = parts[0].strip()
                        entity = parts[1].strip()
                        amount = float(parts[3].strip().split()[0])
                        currency = parts[3].strip().split()[1]
                        budget_data.append({'month': month, 'entity': entity, 'amount': amount, 'currency': currency})
            
            # Convert to USD and sum
            actual_revenue = sum(self._convert_to_usd(item['amount'], item['currency'], item['month']) for item in actuals_data)
            budget_revenue = sum(self._convert_to_usd(item['amount'], item['currency'], item['month']) for item in budget_data)
            
            return {
                'actual_revenue': actual_revenue,
                'budget_revenue': budget_revenue,
                'variance': actual_revenue - budget_revenue,
                'variance_pct': ((actual_revenue - budget_revenue) / budget_revenue * 100) if budget_revenue != 0 else 0
            }
            
        except Exception as e:
            print(f"⚠️ Revenue budget data extraction failed: {e}")
            return {}
    
    def _extract_opex_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract OPEX breakdown data for charting"""
        
        try:
            opex_data = {}
            
            lines = data_summary.split('\n')
            in_actuals = False
            
            for line in lines:
                if 'ACTUALS DATA' in line:
                    in_actuals = True
                    continue
                elif 'BUDGET DATA' in line or 'FX RATES' in line or 'CASH DATA' in line:
                    in_actuals = False
                    continue
                
                if in_actuals and 'Opex' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        category = parts[2].strip()
                        amount = float(parts[3].strip().split()[0])
                        currency = parts[3].strip().split()[1]
                        month = parts[0].strip()
                        
                        if category not in opex_data:
                            opex_data[category] = 0
                        opex_data[category] += self._convert_to_usd(amount, currency, month)
            
            total_opex = sum(opex_data.values())
            breakdown_pct = {k: (v / total_opex * 100) if total_opex > 0 else 0 for k, v in opex_data.items()}
            
            return {
                'breakdown': opex_data,
                'breakdown_pct': breakdown_pct,
                'total_opex': total_opex
            }
            
        except Exception as e:
            print(f"⚠️ OPEX data extraction failed: {e}")
            return {}
    
    def _extract_trend_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract trend data for charting"""
        
        try:
            # This would extract time series data for trend charts
            # Implementation depends on specific trend requirements
            return {}
            
        except Exception as e:
            print(f"⚠️ Trend data extraction failed: {e}")
            return {}
    
    def _convert_to_usd(self, amount: float, currency: str, month: str) -> float:
        """Convert amount to USD using FX rates"""
        
        try:
            if currency == 'USD':
                return amount
            
            # Get FX rate for the month
            fx_data = self.data_loader.data.get('fx', pd.DataFrame())
            if fx_data.empty:
                return amount
            
            month_fx = fx_data[(fx_data['month'] == month) & (fx_data['currency'] == currency)]
            if not month_fx.empty:
                rate = month_fx['rate_to_usd'].iloc[0]
                return amount * rate
            
            # Fallback to latest rate
            currency_fx = fx_data[fx_data['currency'] == currency]
            if not currency_fx.empty:
                rate = currency_fx['rate_to_usd'].iloc[-1]
                return amount * rate
            
            return amount
            
        except Exception as e:
            print(f"⚠️ Currency conversion failed: {e}")
            return amount

    def _generate_llm_response(self, query: str, base_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced response using LLM"""
        
        try:
            # Prepare context for LLM
            context = self._build_context(base_response)
            
            system_prompt = """You are an expert CFO assistant. Your role is to:
1. Analyze financial data and provide clear insights
2. Explain complex financial concepts in simple terms
3. Highlight key trends and anomalies
4. Provide actionable recommendations when appropriate
5. Be concise but comprehensive in your analysis

Always be professional, accurate, and focus on what matters most to executives."""
            
            user_prompt = f"""
            Original Query: "{query}"
            
            Financial Analysis Results:
            {json.dumps(base_response['result'], indent=2)}
            
            Please provide a comprehensive CFO-level analysis that includes:
            1. Key findings and insights
            2. Trend analysis (if applicable)
            3. Potential concerns or opportunities
            4. Executive summary
            
            Keep it professional and actionable for board-level discussions.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            llm_analysis = response.choices[0].message.content
            
            # Add LLM response to conversation history
            self.conversation_history.append({"role": "assistant", "content": llm_analysis})
            
            # Enhance the base response with LLM insights
            enhanced_result = base_response['result'].copy()
            enhanced_result['llm_analysis'] = llm_analysis
            enhanced_result['key_insights'] = self._extract_key_insights(llm_analysis)
            
            return {
                "success": True,
                "intent": base_response['intent'],
                "confidence": base_response['confidence'],
                "query": query,
                "result": enhanced_result,
                "llm_enhanced": True
            }
            
        except Exception as e:
            print(f"⚠️ LLM response generation failed: {e}")
            return base_response
    
    def _build_context(self, base_response: Dict[str, Any]) -> str:
        """Build context from conversation history and data"""
        
        context = f"Intent: {base_response['intent']}\n"
        context += f"Confidence: {base_response['confidence']}\n"
        
        if len(self.conversation_history) > 1:
            context += "\nRecent conversation:\n"
            for msg in self.conversation_history[-3:]:  # Last 3 messages
                role = "User" if msg['role'] == 'user' else "Assistant"
                context += f"{role}: {msg['content'][:100]}...\n"
        
        return context
    
    def _extract_key_insights(self, llm_analysis: str) -> list:
        """Extract key insights from LLM analysis"""
        
        insights = []
        lines = llm_analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*')) or 'key' in line.lower() or 'important' in line.lower():
                insights.append(line)
        
        return insights[:5]  # Limit to top 5 insights
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        
        if not self.use_llm or not self.openai_client:
            return "LLM not available for conversation summary."
        
        try:
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-10:]  # Last 10 messages
            ])
            
            prompt = f"""
            Summarize this CFO conversation in 2-3 sentences, highlighting:
            1. Main topics discussed
            2. Key financial insights discovered
            3. Any action items or concerns raised
            
            Conversation:
            {conversation_text}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def _extract_chart_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract data for chart generation based on query type"""
        
        try:
            query_lower = query.lower()
            chart_data = {"chart_type": None, "data": {}}
            
            # Revenue vs Budget comparison
            if any(word in query_lower for word in ['revenue', 'budget', 'vs', 'comparison', 'actual']):
                chart_data["chart_type"] = "revenue_vs_budget"
                chart_data["data"] = self._extract_revenue_budget_data(query, data_summary)
            
            # OPEX breakdown
            elif any(word in query_lower for word in ['opex', 'operating expense', 'expense', 'breakdown', 'category']):
                chart_data["chart_type"] = "opex_breakdown"
                chart_data["data"] = self._extract_opex_data(query, data_summary)
            
            # Trend analysis
            elif any(word in query_lower for word in ['trend', 'over time', 'monthly', 'quarterly', 'growth']):
                chart_data["chart_type"] = "trend"
                chart_data["data"] = self._extract_trend_data(query, data_summary)
            
            return chart_data
            
        except Exception as e:
            print(f"⚠️ Chart data extraction failed: {e}")
            return {"chart_type": None, "data": {}}
    
    def _extract_revenue_budget_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract revenue vs budget data for charting"""
        
        try:
            # Parse the data summary to extract actual numbers
            actuals_data = []
            budget_data = []
            
            lines = data_summary.split('\n')
            in_actuals = False
            in_budget = False
            
            for line in lines:
                if 'ACTUALS DATA' in line:
                    in_actuals = True
                    in_budget = False
                    continue
                elif 'BUDGET DATA' in line:
                    in_actuals = False
                    in_budget = True
                    continue
                elif 'FX RATES' in line or 'CASH DATA' in line:
                    in_actuals = False
                    in_budget = False
                    continue
                
                if in_actuals and 'Revenue' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        month = parts[0].strip()
                        entity = parts[1].strip()
                        amount = float(parts[3].strip().split()[0])
                        currency = parts[3].strip().split()[1]
                        actuals_data.append({'month': month, 'entity': entity, 'amount': amount, 'currency': currency})
                
                elif in_budget and 'Revenue' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        month = parts[0].strip()
                        entity = parts[1].strip()
                        amount = float(parts[3].strip().split()[0])
                        currency = parts[3].strip().split()[1]
                        budget_data.append({'month': month, 'entity': entity, 'amount': amount, 'currency': currency})
            
            # Convert to USD and sum
            actual_revenue = sum(self._convert_to_usd(item['amount'], item['currency'], item['month']) for item in actuals_data)
            budget_revenue = sum(self._convert_to_usd(item['amount'], item['currency'], item['month']) for item in budget_data)
            
            return {
                'actual_revenue': actual_revenue,
                'budget_revenue': budget_revenue,
                'variance': actual_revenue - budget_revenue,
                'variance_pct': ((actual_revenue - budget_revenue) / budget_revenue * 100) if budget_revenue != 0 else 0
            }
            
        except Exception as e:
            print(f"⚠️ Revenue budget data extraction failed: {e}")
            return {}
    
    def _extract_opex_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract OPEX breakdown data for charting"""
        
        try:
            opex_data = {}
            
            lines = data_summary.split('\n')
            in_actuals = False
            
            for line in lines:
                if 'ACTUALS DATA' in line:
                    in_actuals = True
                    continue
                elif 'BUDGET DATA' in line or 'FX RATES' in line or 'CASH DATA' in line:
                    in_actuals = False
                    continue
                
                if in_actuals and 'Opex' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        category = parts[2].strip()
                        amount = float(parts[3].strip().split()[0])
                        currency = parts[3].strip().split()[1]
                        month = parts[0].strip()
                        
                        if category not in opex_data:
                            opex_data[category] = 0
                        opex_data[category] += self._convert_to_usd(amount, currency, month)
            
            total_opex = sum(opex_data.values())
            breakdown_pct = {k: (v / total_opex * 100) if total_opex > 0 else 0 for k, v in opex_data.items()}
            
            return {
                'breakdown': opex_data,
                'breakdown_pct': breakdown_pct,
                'total_opex': total_opex
            }
            
        except Exception as e:
            print(f"⚠️ OPEX data extraction failed: {e}")
            return {}
    
    def _extract_trend_data(self, query: str, data_summary: str) -> Dict[str, Any]:
        """Extract trend data for charting"""
        
        try:
            # This would extract time series data for trend charts
            # Implementation depends on specific trend requirements
            return {}
            
        except Exception as e:
            print(f"⚠️ Trend data extraction failed: {e}")
            return {}
    
    def _convert_to_usd(self, amount: float, currency: str, month: str) -> float:
        """Convert amount to USD using FX rates"""
        
        try:
            if currency == 'USD':
                return amount
            
            # Get FX rate for the month
            fx_data = self.data_loader.data.get('fx', pd.DataFrame())
            if fx_data.empty:
                return amount
            
            month_fx = fx_data[(fx_data['month'] == month) & (fx_data['currency'] == currency)]
            if not month_fx.empty:
                rate = month_fx['rate_to_usd'].iloc[0]
                return amount * rate
            
            # Fallback to latest rate
            currency_fx = fx_data[fx_data['currency'] == currency]
            if not currency_fx.empty:
                rate = currency_fx['rate_to_usd'].iloc[-1]
                return amount * rate
            
            return amount
            
        except Exception as e:
            print(f"⚠️ Currency conversion failed: {e}")
            return amount

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history reset")
