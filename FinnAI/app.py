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
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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

def create_revenue_vs_budget_chart(result):
    """Create a revenue vs budget comparison chart"""
    if "error" in result:
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
    
    # Display confidence score
    st.info(f"Intent: {intent.replace('_', ' ').title()} (Confidence: {confidence:.2f})")
    
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
                
                elif chart_type == "opex_breakdown" and chart_data:
                    st.markdown("### 📊 OPEX Breakdown Chart")
                    chart = create_opex_breakdown_chart(chart_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                
                elif chart_type == "trend" and chart_data:
                    st.markdown("### 📈 Trend Analysis Chart")
                    # Add trend chart creation here if needed
        elif 'message' in result:
            st.info(result['message'])
            st.write("**Suggested questions:**")
            for suggestion in result['suggestions']:
                st.write(f"• {suggestion}")
            st.write(f"**Example:** {result['example']}")
        else:
            st.info("Analysis completed. Check the AI insights above for detailed results.")

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 CFO Copilot</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered financial assistant for data-driven insights**")
    
    # Initialize agent
    agent = initialize_agent()
    
    # Sidebar
    with st.sidebar:
        st.header("📈 Available Metrics")
        
        # Get available data info
        data_info = agent.get_available_metrics()
        if data_info['success']:
            st.write("**Data Sources:**")
            for data_type, info in data_info['data_info'].items():
                st.write(f"• {data_type.title()}: {info['rows']} records")
            
            st.write("**Supported Metrics:**")
            for metric in data_info['available_metrics']:
                st.write(f"• {metric}")
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "What was June 2025 revenue vs budget in USD?",
            "Show Gross Margin % trend for the last 3 months",
            "Break down Opex by category for June",
            "What is our cash runway right now?",
            "Show me EBITDA for March 2025"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.user_query = question
        
        st.header("🔧 Settings")
        if st.button("Reload Data"):
            st.session_state.agent = CFOAgent()
            st.success("Data reloaded successfully!")
    
    # Main chat interface
    st.header("💬 Ask Your CFO Questions")
    
    # Chat input
    user_query = st.text_input(
        "Ask a question about your financial data:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., What was June 2025 revenue vs budget in USD?"
    )
    
    # Process query button
    if st.button("Analyze", type="primary") or user_query:
        if user_query:
            with st.spinner("Analyzing your question..."):
                response = agent.process_query(user_query)
                display_response(response)
        else:
            st.warning("Please enter a question to analyze.")

if __name__ == "__main__":
    main()
