"""
CFO Copilot Demo Script
Demonstrates the agent's capabilities with sample queries
"""

from agent.cfo_agent import CFOAgent
import json

def demo_agent():
    """Run a demo of the CFO Agent with sample queries"""
    
    print("🤖 CFO Copilot Demo")
    print("=" * 50)
    
    # Initialize the agent
    print("Initializing CFO Agent...")
    agent = CFOAgent()
    print("✅ Agent initialized successfully!\n")
    
    # Sample queries to demonstrate different capabilities
    demo_queries = [
        "What was June 2025 revenue vs budget in USD?",
        "Show Gross Margin % trend for the last 3 months",
        "Break down Opex by category for June",
        "What is our cash runway right now?",
        "Show me EBITDA for March 2025"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"📝 Demo Query {i}: {query}")
        print("-" * 50)
        
        try:
            response = agent.process_query(query)
            
            if response['success']:
                print(f"✅ Intent: {response['intent']}")
                print(f"🎯 Confidence: {response['confidence']:.2f}")
                
                result = response['result']
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("📊 Result:")
                    # Pretty print the result
                    print(json.dumps(result, indent=2, default=str))
            else:
                print(f"❌ Failed: {response['error']}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        print("\n" + "="*50 + "\n")
    
    print("🎉 Demo completed! Check the Streamlit app for interactive usage.")

if __name__ == "__main__":
    demo_agent()
