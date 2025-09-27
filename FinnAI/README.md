# CFO Copilot 📊

Your AI-powered financial assistant that turns messy spreadsheets into clear insights. Just ask questions in plain English and get instant answers with beautiful charts.

## What This Does

Think of it as having a smart CFO assistant who can:
- Answer questions like "How did we do vs budget last month?"
- Show you trends and breakdowns with pretty charts
- Handle multiple currencies automatically
- Give you the insights you need without digging through data

## Quick Start

**1. Install the stuff you need:**
```bash
pip install -r requirements.txt
```

**2. Get a Gemini API key (for the smart analysis):**
- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Create a `.env` file in the project folder:
```
GEMINI_API_KEY=your_api_key_here
```

**3. Run it:**
```bash
streamlit run app.py
```

**4. Open your browser to `http://localhost:8501` and start asking questions!**

## Sample Questions You Can Ask

- "What was June 2025 revenue vs budget in USD?"
- "Show me the gross margin trend for the last 6 months"
- "Break down our operating expenses by category"
- "What's our cash runway looking like?"
- "How profitable were we in March 2025?"

## Your Data

The app looks for CSV files in the `fixtures/` folder:
- `actuals.csv` - Your real financial data
- `budget.csv` - Your budgeted amounts  
- `fx.csv` - Currency exchange rates
- `cash.csv` - Cash balances

Make sure your CSVs have the right columns (check the examples in the fixtures folder).

## What You'll Get

- **Smart Analysis**: The AI explains what the numbers mean
- **Beautiful Charts**: Interactive graphs that actually make sense
- **Real Insights**: Not just data dumps, but actionable insights
- **Multi-Currency**: Handles different currencies automatically


## What's Under the Hood

This is built with:
- **Streamlit** for the web interface
- **Google Gemini** for the smart analysis
- **Plotly** for the charts
- **Pandas** for data processing

The app automatically:
- Understands what you're asking for
- Finds the right data
- Does the calculations
- Creates the charts
- Explains what it all means

