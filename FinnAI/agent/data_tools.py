"""
CFO Copilot Data Tools
Handles data loading, processing, and calculations for financial metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

class DataLoader:
    """Handles loading and preprocessing of CSV data"""
    
    def __init__(self, fixtures_dir: str = "fixtures"):
        self.fixtures_dir = fixtures_dir
        self.data = {}
        self.fx_rates = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files and return as dictionary"""
        try:
            # Load actuals
            actuals_path = os.path.join(self.fixtures_dir, "actuals.csv")
            if os.path.exists(actuals_path):
                self.data['actuals'] = pd.read_csv(actuals_path)
            
            # Load budget
            budget_path = os.path.join(self.fixtures_dir, "budget.csv")
            if os.path.exists(budget_path):
                self.data['budget'] = pd.read_csv(budget_path)
            
            # Load FX rates
            fx_path = os.path.join(self.fixtures_dir, "fx.csv")
            if os.path.exists(fx_path):
                fx_df = pd.read_csv(fx_path)
                self.data['fx'] = fx_df
                # Create lookup dictionary for FX rates
                self.fx_rates = dict(zip(fx_df['currency'], fx_df['rate_to_usd']))
            
            # Load cash data
            cash_path = os.path.join(self.fixtures_dir, "cash.csv")
            if os.path.exists(cash_path):
                self.data['cash'] = pd.read_csv(cash_path)
            
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}
    
    def get_fx_rate(self, currency: str) -> float:
        """Get FX rate for a given currency to USD"""
        return self.fx_rates.get(currency, 1.0)  # Default to 1.0 if not found

class FinancialCalculator:
    """Handles financial calculations and metrics"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.data = data_loader.data
        
        # Month name to number mapping
        self.month_mapping = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02',
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
    
    def _convert_to_usd_and_sum(self, df: pd.DataFrame) -> float:
        """Convert amounts to USD and sum them"""
        total_usd = 0
        for _, row in df.iterrows():
            amount = row['amount']
            currency = row['currency']
            month = row['month']
            
            # Get FX rate for this month and currency
            fx_rate = self._get_fx_rate_for_month(month, currency)
            total_usd += amount * fx_rate
        
        return total_usd
    
    def _get_fx_rate_for_month(self, month: str, currency: str) -> float:
        """Get FX rate for a specific month and currency"""
        fx_data = self.data.get('fx', pd.DataFrame())
        if fx_data.empty:
            return 1.0
        
        # Filter for the specific month and currency
        month_fx = fx_data[(fx_data['month'] == month) & (fx_data['currency'] == currency)]
        if not month_fx.empty:
            return month_fx['rate_to_usd'].iloc[0]
        
        # If not found, try to get the latest rate for this currency
        currency_fx = fx_data[fx_data['currency'] == currency]
        if not currency_fx.empty:
            return currency_fx['rate_to_usd'].iloc[-1]
        
        return 1.0  # Default to 1.0 if not found
    
    def get_revenue_vs_budget(self, month: Optional[str] = None, year: Optional[int] = None) -> Dict[str, Any]:
        """Calculate revenue vs budget for a specific month/year"""
        try:
            actuals = self.data.get('actuals', pd.DataFrame())
            budget = self.data.get('budget', pd.DataFrame())
            
            if actuals.empty or budget.empty:
                return {"error": "Missing actuals or budget data"}
            
            # Filter for revenue accounts
            revenue_actuals = actuals[actuals['account_category'] == 'Revenue']
            revenue_budget = budget[budget['account_category'] == 'Revenue']
            
            if month and year:
                # Convert month name to number if needed
                month_num = self.month_mapping.get(month.lower(), month)
                # Filter by specific month/year (format: YYYY-MM)
                month_filter = f"{year}-{month_num.zfill(2)}"
                
                # Check if data exists for this month
                available_months = set(actuals['month'].unique()) | set(budget['month'].unique())
                if month_filter not in available_months:
                    return {"error": f"Data not available for {month} {year}. Available months: {sorted(available_months)}"}
                
                revenue_actuals = revenue_actuals[revenue_actuals['month'] == month_filter]
                revenue_budget = revenue_budget[revenue_budget['month'] == month_filter]
                
                if revenue_actuals.empty or revenue_budget.empty:
                    return {"error": f"Revenue data not available for {month} {year}"}
            else:
                # Get latest available data only if no specific month/year requested
                latest_month = actuals['month'].max()
                revenue_actuals = revenue_actuals[revenue_actuals['month'] == latest_month]
                revenue_budget = revenue_budget[revenue_budget['month'] == latest_month]
            
            # Convert to USD if needed and sum
            actual_revenue = self._convert_to_usd_and_sum(revenue_actuals)
            budget_revenue = self._convert_to_usd_and_sum(revenue_budget)
            
            variance = actual_revenue - budget_revenue
            variance_pct = (variance / budget_revenue * 100) if budget_revenue != 0 else 0
            
            return {
                "actual_revenue": round(actual_revenue, 2),
                "budget_revenue": round(budget_revenue, 2),
                "variance": round(variance, 2),
                "variance_pct": round(variance_pct, 2),
                "month": month,
                "year": year
            }
        except Exception as e:
            return {"error": f"Error calculating revenue vs budget: {str(e)}"}
    
    def get_gross_margin_trend(self, period_months: int = 3) -> Dict[str, Any]:
        """Calculate gross margin trend for the last N months"""
        try:
            actuals = self.data.get('actuals', pd.DataFrame())
            
            if actuals.empty:
                return {"error": "Missing actuals data"}
            
            # Get unique months and sort them
            unique_months = sorted(actuals['month'].unique())
            
            # Use user-specified period or default to 3 months
            if period_months is None:
                period_months = 3
            
            if len(unique_months) < period_months:
                period_months = len(unique_months)
            
            # Get last N months of data
            recent_months = unique_months[-period_months:]
            
            margins = []
            months = []
            
            for month in recent_months:
                month_data = actuals[actuals['month'] == month]
                
                # Get revenue and COGS for this month
                revenue_data = month_data[month_data['account_category'] == 'Revenue']
                cogs_data = month_data[month_data['account_category'] == 'COGS']
                
                revenue = self._convert_to_usd_and_sum(revenue_data)
                cogs = self._convert_to_usd_and_sum(cogs_data)
                
                gross_margin = ((revenue - cogs) / revenue * 100) if revenue != 0 else 0
                margins.append(round(gross_margin, 2))
                months.append(month)
            
            return {
                "margins": margins,
                "months": months,
                "period_months": period_months,
                "average_margin": round(sum(margins) / len(margins), 2) if margins else 0
            }
        except Exception as e:
            return {"error": f"Error calculating gross margin trend: {str(e)}"}
    
    def get_opex_breakdown(self, month: Optional[str] = None, year: Optional[int] = None) -> Dict[str, Any]:
        """Get operating expenses breakdown by category"""
        try:
            actuals = self.data.get('actuals', pd.DataFrame())
            
            if actuals.empty:
                return {"error": "Missing actuals data"}
            
            # Filter for Opex accounts
            opex_data = actuals[actuals['account_category'].str.contains('Opex', case=False, na=False)]
            
            if opex_data.empty:
                return {"error": "No Opex data found"}
            
            # Filter by month if specified
            if month and year:
                # Convert month name to number if needed
                month_num = self.month_mapping.get(month.lower(), month)
                month_filter = f"{year}-{month_num.zfill(2)}"
                opex_data = opex_data[opex_data['month'] == month_filter]
                if opex_data.empty:
                    return {"error": f"Data not available for {month} {year}"}
            else:
                # Get latest available data only if no specific month/year requested
                latest_month = opex_data['month'].max()
                opex_data = opex_data[opex_data['month'] == latest_month]
            
            # Group by account category and convert to USD
            breakdown = {}
            for category in opex_data['account_category'].unique():
                category_data = opex_data[opex_data['account_category'] == category]
                total_usd = self._convert_to_usd_and_sum(category_data)
                breakdown[category] = total_usd
            
            total_opex = sum(breakdown.values())
            
            # Calculate percentages
            breakdown_pct = {k: round((v / total_opex * 100), 2) for k, v in breakdown.items()}
            
            return {
                "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
                "breakdown_pct": breakdown_pct,
                "total_opex": round(total_opex, 2),
                "month": month,
                "year": year
            }
        except Exception as e:
            return {"error": f"Error calculating Opex breakdown: {str(e)}"}
    
    def get_cash_runway(self) -> Dict[str, Any]:
        """Calculate cash runway based on current cash and burn rate"""
        try:
            cash_data = self.data.get('cash', pd.DataFrame())
            actuals = self.data.get('actuals', pd.DataFrame())
            
            if cash_data.empty:
                return {"error": "Missing cash data"}
            
            # Get current cash balance (last available data)
            latest_month = cash_data['month'].max()
            current_cash = cash_data[cash_data['month'] == latest_month]['cash_usd'].sum()
            
            if current_cash == 0:
                return {"error": "No cash data found"}
            
            # Calculate burn rate from last 3 months of actuals
            if actuals.empty:
                return {"error": "Missing actuals data for burn rate calculation"}
            
            # Get last 3 months of actuals
            unique_months = sorted(actuals['month'].unique())
            if len(unique_months) < 3:
                months_to_use = len(unique_months)
            else:
                months_to_use = 3
            
            recent_months = unique_months[-months_to_use:]
            
            # Calculate net burn for each month (Revenue - COGS - Opex)
            monthly_burns = []
            for month in recent_months:
                month_data = actuals[actuals['month'] == month]
                
                revenue = self._convert_to_usd_and_sum(month_data[month_data['account_category'] == 'Revenue'])
                cogs = self._convert_to_usd_and_sum(month_data[month_data['account_category'] == 'COGS'])
                opex = self._convert_to_usd_and_sum(month_data[month_data['account_category'].str.contains('Opex', case=False, na=False)])
                
                net_burn = revenue - cogs - opex
                monthly_burns.append(net_burn)
            
            avg_monthly_burn = sum(monthly_burns) / len(monthly_burns)
            
            # Calculate runway
            if avg_monthly_burn < 0:  # If we're burning cash
                runway_months = current_cash / abs(avg_monthly_burn)
            else:
                runway_months = float('inf')  # We're generating cash
            
            return {
                "current_cash": round(current_cash, 2),
                "avg_monthly_burn": round(avg_monthly_burn, 2),
                "runway_months": round(runway_months, 1) if runway_months != float('inf') else "∞",
                "months_analyzed": months_to_use
            }
        except Exception as e:
            return {"error": f"Error calculating cash runway: {str(e)}"}
    
    def get_ebitda(self, month: Optional[str] = None, year: Optional[int] = None) -> Dict[str, Any]:
        """Calculate EBITDA (Revenue - COGS - Opex)"""
        try:
            actuals = self.data.get('actuals', pd.DataFrame())
            
            if actuals.empty:
                return {"error": "Missing actuals data"}
            
            # Filter by month if specified
            if month and year:
                # Convert month name to number if needed
                month_num = self.month_mapping.get(month.lower(), month)
                month_filter = f"{year}-{month_num.zfill(2)}"
                month_data = actuals[actuals['month'] == month_filter]
                if month_data.empty:
                    return {"error": f"Data not available for {month} {year}"}
            else:
                # Get latest available data only if no specific month/year requested
                latest_month = actuals['month'].max()
                month_data = actuals[actuals['month'] == latest_month]
            
            # Get revenue, COGS, and Opex
            revenue_data = month_data[month_data['account_category'] == 'Revenue']
            cogs_data = month_data[month_data['account_category'] == 'COGS']
            opex_data = month_data[month_data['account_category'].str.contains('Opex', case=False, na=False)]
            
            revenue = self._convert_to_usd_and_sum(revenue_data)
            cogs = self._convert_to_usd_and_sum(cogs_data)
            opex = self._convert_to_usd_and_sum(opex_data)
            
            ebitda = revenue - cogs - opex
            
            return {
                "revenue": round(revenue, 2),
                "cogs": round(cogs, 2),
                "opex": round(opex, 2),
                "ebitda": round(ebitda, 2),
                "month": month,
                "year": year
            }
        except Exception as e:
            return {"error": f"Error calculating EBITDA: {str(e)}"}
