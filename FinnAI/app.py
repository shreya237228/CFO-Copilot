"""
CFO Copilot - Streamlit Web App
AI-powered financial assistant for CFOs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from agent.cfo_agent import CFOAgent
from agent.llm_agent import LLMEnhancedCFOAgent

# Page configuration
st.set_page_config(
    page_title="CFO Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card strong {
        color: #1f2937;
        font-weight: 600;
    }
    
    /* Message Styles */
    .success-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.1);
    }
    
    .error-message {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #ef4444;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.1);
    }
    
    .info-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-section h3 {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric Display */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="metric-container"] > div {
        color: #1f2937;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Chart Container */
    .stPlotlyChart {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    /* Dataframe Styles */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Loading Spinner */
    .stSpinner {
        color: #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agent():
    """Initialize the LLM-Enhanced CFO Agent"""
    if 'agent' not in st.session_state:
        st.session_state.agent = LLMEnhancedCFOAgent()
    return st.session_state.agent

def format_currency(amount):
    """Format currency for display"""
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:.2f}"

def _validate_revenue_budget_data(result):
    """Validate revenue vs budget data for chart creation"""
    try:
        # Check if required fields exist
        required_fields = ['actual_revenue', 'budget_revenue']
        for field in required_fields:
            if field not in result:
                return False
        
        # Check if values are numeric and not None
        actual = result['actual_revenue']
        budget = result['budget_revenue']
        
        if actual is None or budget is None:
            return False
        
        # Check if values are numbers
        try:
            actual = float(actual)
            budget = float(budget)
        except (ValueError, TypeError):
            return False
        
        # Check if at least one value is non-zero (meaningful data)
        if actual == 0 and budget == 0:
            return False
        
        # Check if values are reasonable (not extremely large or negative)
        if abs(actual) > 1e12 or abs(budget) > 1e12:  # 1 trillion limit
            return False
        
        return True
        
    except Exception:
        return False

def _validate_trend_data(result):
    """Validate trend data for chart creation"""
    try:
        # Check if required fields exist
        required_fields = ['months', 'margins']
        for field in required_fields:
            if field not in result:
                return False
        
        months = result['months']
        margins = result['margins']
        
        # Check if data is not empty
        if not months or not margins:
            return False
        
        # Check if both lists have the same length
        if len(months) != len(margins):
            return False
        
        # Check if we have at least 2 data points for a meaningful trend
        if len(months) < 2:
            return False
        
        # Check if margins contain valid numeric data
        try:
            margins_float = [float(m) for m in margins]
            # Check if all values are reasonable percentages
            if any(m < -1000 or m > 1000 for m in margins_float):
                return False
        except (ValueError, TypeError):
            return False
        
        return True
        
    except Exception:
        return False

def _validate_opex_data(result):
    """Validate OPEX breakdown data for chart creation"""
    try:
        # Check if required fields exist
        if 'breakdown' not in result:
            return False
        
        breakdown = result['breakdown']
        
        # Check if breakdown is not empty
        if not breakdown:
            return False
        
        # Check if we have at least 2 categories for a meaningful breakdown
        if len(breakdown) < 2:
            return False
        
        # Check if all values are numeric and positive
        total_value = 0
        for category, value in breakdown.items():
            try:
                value_float = float(value)
                if value_float < 0:  # Negative expenses don't make sense for breakdown
                    return False
                total_value += value_float
            except (ValueError, TypeError):
                return False
        
        # Check if total is meaningful (not zero)
        if total_value == 0:
            return False
        
        # Check if total is reasonable
        if total_value > 1e12:  # 1 trillion limit
            return False
        
        return True
        
    except Exception:
        return False

def create_revenue_vs_budget_chart(result):
    """Create a revenue vs budget comparison chart"""
    if "error" in result:
        return None
    
    # Validate data before creating chart
    if not _validate_revenue_budget_data(result):
        return None
    
    fig = go.Figure(data=[
        go.Bar(name='Actual Revenue', x=['Revenue'], y=[result['actual_revenue']], 
               marker_color='#2E8B57'),
        go.Bar(name='Budget Revenue', x=['Revenue'], y=[result['budget_revenue']], 
               marker_color='#4169E1')
    ])
    
    fig.update_layout(
        title='Revenue vs Budget Comparison',
        xaxis_title='',
        yaxis_title='Amount (USD)',
        barmode='group',
        height=400
    )
    
    return fig

def create_gross_margin_trend_chart(result):
    """Create a gross margin trend chart"""
    if "error" in result:
        return None
    
    # Validate data before creating chart
    if not _validate_trend_data(result):
        return None
    
    fig = go.Figure(data=go.Scatter(
        x=result['months'],
        y=result['margins'],
        mode='lines+markers',
        name='Gross Margin %',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Gross Margin Trend',
        xaxis_title='Month',
        yaxis_title='Gross Margin (%)',
        height=400
    )
    
    return fig

def create_opex_breakdown_chart(result):
    """Create an Opex breakdown pie chart"""
    if "error" in result:
        return None
    
    # Validate data before creating chart
    if not _validate_opex_data(result):
        return None
    
    labels = list(result['breakdown'].keys())
    values = list(result['breakdown'].values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title='Operating Expenses Breakdown',
        height=500
    )
    
    return fig

def display_response(response):
    """Display the agent's response"""
    if not response['success']:
        st.markdown(f'<div class="error-message">{response["error"]}</div>', 
                   unsafe_allow_html=True)
        return
    
    intent = response['intent']
    result = response['result']
    confidence = response['confidence']
    
    # Display confidence score with better styling
    confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
    st.markdown(f"""
    <div class="info-message">
        <strong>Intent:</strong> {intent.replace('_', ' ').title()} {confidence_color} <strong>Confidence:</strong> {confidence:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    # Handle different intent types
    if intent == "revenue_vs_budget":
        if "error" not in result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Actual Revenue", format_currency(result['actual_revenue']))
            with col2:
                st.metric("Budget Revenue", format_currency(result['budget_revenue']))
            with col3:
                st.metric("Variance", format_currency(result['variance']))
            with col4:
                st.metric("Variance %", f"{result['variance_pct']:.1f}%")
            
            # Create chart
            chart = create_revenue_vs_budget_chart(result)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.markdown("""
                <div class="info-message">
                    📊 Chart not displayed: Insufficient or invalid data for revenue vs budget comparison.
                </div>
                """, unsafe_allow_html=True)
            
            # Summary text
            variance_text = "above" if result['variance'] > 0 else "below"
            st.markdown(f"""
            <div class="metric-card">
                <strong>Summary:</strong> Actual revenue for {result.get('month', 'latest period')} {result.get('year', '')} 
                was <strong>{format_currency(result['actual_revenue'])}</strong>, 
                {format_currency(abs(result['variance']))} {variance_text} budget of 
                <strong>{format_currency(result['budget_revenue'])}</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(result['error'])
    
    elif intent == "gross_margin_trend":
        if "error" not in result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Margin", f"{result['average_margin']:.1f}%")
            with col2:
                st.metric("Period Analyzed", f"{result['period_months']} months")
            
            # Create chart
            chart = create_gross_margin_trend_chart(result)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.markdown("""
                <div class="info-message">
                    📊 Chart not displayed: Insufficient data points for meaningful trend analysis.
                </div>
                """, unsafe_allow_html=True)
            
            # Summary text
            trend_direction = "improving" if result['margins'][-1] > result['margins'][0] else "declining"
            st.markdown(f"""
            <div class="metric-card">
                <strong>Summary:</strong> Gross margin trend shows {trend_direction} performance over the last 
                {result['period_months']} months, with an average margin of {result['average_margin']:.1f}%.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(result['error'])
    
    elif intent == "opex_breakdown":
        if "error" not in result:
            st.metric("Total Opex", format_currency(result['total_opex']))
            
            # Create chart
            chart = create_opex_breakdown_chart(result)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.markdown("""
                <div class="info-message">
                    📊 Chart not displayed: Insufficient expense categories for meaningful breakdown.
                </div>
                """, unsafe_allow_html=True)
            
            # Breakdown table
            breakdown_df = pd.DataFrame([
                {"Category": k, "Amount (USD)": format_currency(v), "Percentage": f"{result['breakdown_pct'][k]:.1f}%"}
                for k, v in result['breakdown'].items()
            ])
            st.dataframe(breakdown_df, use_container_width=True)
            
            # Summary text
            st.markdown(f"""
            <div class="metric-card">
                <strong>Summary:</strong> Total operating expenses for {result.get('month', 'latest period')} {result.get('year', '')} 
                were <strong>{format_currency(result['total_opex'])}</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(result['error'])
    
    elif intent == "cash_runway":
        if "error" not in result:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Cash", format_currency(result['current_cash']))
            with col2:
                st.metric("Monthly Burn", format_currency(result['avg_monthly_burn']))
            with col3:
                runway_display = f"{result['runway_months']} months" if result['runway_months'] != "∞" else "∞"
                st.metric("Cash Runway", runway_display)
            
            # Summary text
            if result['runway_months'] == "∞":
                st.markdown(f"""
                <div class="success-message">
                    <strong>Excellent!</strong> The company is generating positive cash flow with 
                    <strong>{format_currency(result['current_cash'])}</strong> in cash reserves.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Summary:</strong> Current cash runway is approximately 
                    <strong>{result['runway_months']} months</strong> based on the average monthly burn of 
                    <strong>{format_currency(result['avg_monthly_burn'])}</strong> over the last 
                    {result['months_analyzed']} months.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(result['error'])
    
    elif intent == "ebitda":
        if "error" not in result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Revenue", format_currency(result['revenue']))
            with col2:
                st.metric("COGS", format_currency(result['cogs']))
            with col3:
                st.metric("Opex", format_currency(result['opex']))
            with col4:
                st.metric("EBITDA", format_currency(result['ebitda']))
            
            # Summary text
            profitability = "profitable" if result['ebitda'] > 0 else "unprofitable"
            st.markdown(f"""
            <div class="metric-card">
                <strong>Summary:</strong> EBITDA analysis for {result.get('month', 'latest period')} {result.get('year', '')} 
                shows the company was <strong>{profitability}</strong> with EBITDA of 
                <strong>{format_currency(result['ebitda'])}</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(result['error'])
    
    else:
        # General query response - check if it's a new LLM response format
        if 'llm_analysis' in result:
            st.markdown("### 📊 Financial Analysis")
            st.markdown(result['llm_analysis'])
            
            # Display charts if available
            if 'chart_data' in result and result['chart_data']['chart_type']:
                chart_type = result['chart_data']['chart_type']
                chart_data = result['chart_data']['data']
                
                if chart_type == "revenue_vs_budget" and chart_data:
                    st.markdown("### 📈 Revenue vs Budget Chart")
                    chart = create_revenue_vs_budget_chart(chart_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.markdown("""
                        <div class="info-message">
                            📊 Chart not displayed: Insufficient or invalid data for revenue vs budget comparison.
                        </div>
                        """, unsafe_allow_html=True)
                
                elif chart_type == "opex_breakdown" and chart_data:
                    st.markdown("### 📊 OPEX Breakdown Chart")
                    chart = create_opex_breakdown_chart(chart_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.markdown("""
                        <div class="info-message">
                            📊 Chart not displayed: Insufficient expense categories for meaningful breakdown.
                        </div>
                        """, unsafe_allow_html=True)
                
                elif chart_type == "trend" and chart_data:
                    st.markdown("### 📈 Trend Analysis Chart")
                    chart = create_gross_margin_trend_chart(chart_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.markdown("""
                        <div class="info-message">
                            📊 Chart not displayed: Insufficient data points for meaningful trend analysis.
                        </div>
                        """, unsafe_allow_html=True)
        elif 'message' in result:
            st.markdown(f"""
            <div class="info-message">
                {result['message']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Suggested questions:**")
            for suggestion in result['suggestions']:
                st.markdown(f"• {suggestion}")
            st.markdown(f"**Example:** {result['example']}")
        else:
            st.markdown("""
            <div class="info-message">
                ✅ Analysis completed. Check the AI insights above for detailed results.
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 CFO Copilot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered financial assistant for data-driven insights</p>', unsafe_allow_html=True)
    
    # Initialize agent
    agent = initialize_agent()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Available Metrics")
        
        # Get available data info
        data_info = agent.get_available_metrics()
        if data_info['success']:
            st.markdown("**Data Sources:**")
            for data_type, info in data_info['data_info'].items():
                st.markdown(f"• **{data_type.title()}**: {info['rows']} records")
            
            st.markdown("**Supported Metrics:**")
            for metric in data_info['available_metrics']:
                st.markdown(f"• {metric}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What was June 2025 revenue vs budget in USD?",
            "Show Gross Margin % trend for the last 3 months",
            "Break down Opex by category for June",
            "What is our cash runway right now?",
            "Show me EBITDA for March 2025"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}", use_container_width=True):
                st.session_state.user_query = question
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🔧 Settings")
        if st.button("🔄 Reload Data", use_container_width=True):
            st.session_state.agent = CFOAgent()
            st.success("✅ Data reloaded successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    st.markdown("### 💬 Ask Your CFO Questions")
    
    # Chat input with better styling
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question about your financial data:",
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., What was June 2025 revenue vs budget in USD?",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Process query
    if analyze_clicked or user_query:
        if user_query:
            with st.spinner("🔍 Analyzing your question..."):
                response = agent.process_query(user_query)
                display_response(response)
        else:
            st.warning("⚠️ Please enter a question to analyze.")

if __name__ == "__main__":
    main()
