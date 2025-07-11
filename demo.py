"""
Demo script for Dell Organizational Operations Chatbot
Test the system without the Streamlit UI
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chatbot_engine import DellOperationsChatbot
    from data_manager import DataManager, DataTemplates
    from config import DellConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Run demo of the Dell chatbot system"""
    print("🏢 Dell Organizational Operations Chatbot - Demo")
    print("=" * 50)
    
    # Initialize system
    print("Initializing system...")
    try:
        chatbot = DellOperationsChatbot()
        data_manager = DataManager()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Display fiscal year info
    current_fy = DellConfig.get_current_fiscal_year()
    fy_range = DellConfig.get_fiscal_year_range()
    print(f"\n📅 Dell Fiscal Year: FY{current_fy}")
    print(f"📊 Analysis Period: FY{fy_range[0]} - FY{fy_range[1]}")
    
    # Load sample data
    print("\n📦 Loading sample data...")
    sample_data = DataTemplates.get_manual_data_example()
    
    # Add more comprehensive sample data
    additional_samples = [
        {
            "content": "Dell FY2024 Q1 manufacturing automation initiative increased production efficiency by 18% while reducing human errors by 30%, resulting in $4.5M cost savings",
            "department": "Manufacturing",
            "fiscal_year": "FY2024",
            "quarter": "Q1",
            "impact_level": "High",
            "financial_impact": "4500000",
            "error_reduction": "30%",
            "efficiency_gain": "18%"
        },
        {
            "content": "Sales team restructuring in FY2023 Q3 improved customer response time by 40% and increased closure rate by 15%, generating $2.2M additional revenue",
            "department": "Sales", 
            "fiscal_year": "FY2023",
            "quarter": "Q3",
            "impact_level": "High",
            "financial_impact": "2200000",
            "efficiency_gain": "40% response time improvement"
        },
        {
            "content": "IT infrastructure modernization in FY2022 Q4 reduced system downtime from 2% to 0.1%, preventing $1.8M in potential losses and improving operational reliability",
            "department": "IT",
            "fiscal_year": "FY2022",
            "quarter": "Q4",
            "impact_level": "Critical",
            "financial_impact": "1800000",
            "error_prevention": "System downtime reduction",
            "reliability_improvement": "99.9% uptime"
        },
        {
            "content": "Supply chain optimization in FY2024 Q2 reduced delivery times by 25% and logistics costs by $3M through AI-powered route optimization",
            "department": "Supply Chain",
            "fiscal_year": "FY2024", 
            "quarter": "Q2",
            "impact_level": "High",
            "financial_impact": "3000000",
            "efficiency_gain": "25% delivery time reduction"
        },
        {
            "content": "HR training program implementation in FY2023 reduced onboarding errors by 45% and improved employee satisfaction scores by 20%",
            "department": "HR",
            "fiscal_year": "FY2023",
            "impact_level": "Medium",
            "error_reduction": "45%",
            "satisfaction_improvement": "20%"
        }
    ]
    
    all_sample_data = sample_data + additional_samples
    
    if data_manager.add_data_manually(all_sample_data):
        print(f"✅ Loaded {len(all_sample_data)} sample records")
    else:
        print("❌ Failed to load sample data")
        return
    
    # Display data statistics
    stats = data_manager.get_collection_stats()
    print(f"\n📈 Data Statistics:")
    print(f"   Total Records: {stats.get('total_documents', 0)}")
    print(f"   Collection: {stats.get('collection_name', 'N/A')}")
    print(f"   Last Updated: {stats.get('last_updated', 'N/A')}")
    
    # Demo queries
    demo_queries = [
        "What were the major operational changes in the last 3 fiscal years?",
        "Which departments had the highest financial impact?", 
        "Show me error reduction initiatives and their effectiveness",
        "What are the trends in operational efficiency across departments?",
        "Which areas show the most improvement in FY2024?"
    ]
    
    print(f"\n🤖 Running Demo Queries:")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            result = chatbot.process_query(query)
            
            print(f"📊 Confidence Score: {result['confidence_score']:.0%}")
            print(f"📁 Data Sources: {result['data_sources']}")
            print(f"📅 Fiscal Year Context: {result['fiscal_year_context']}")
            
            print(f"\n💬 Response:")
            print(result['response'])
            
            if result['key_insights']:
                print(f"\n💡 Key Insights:")
                for insight in result['key_insights']:
                    print(f"   • {insight}")
            
            if result['recommendations']:
                print(f"\n📋 Recommendations:")
                for rec in result['recommendations'][:3]:  # Show top 3
                    print(f"   • {rec}")
            
            print(f"\n✅ Validation Score: {result['validation'].get('overall_score', 0):.0%}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "="*50)
    
    # Interactive mode
    print(f"\n🎯 Interactive Mode - Ask your own questions!")
    print("(Type 'quit' to exit)")
    
    while True:
        try:
            user_query = input("\n🤔 Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            print("\n🤖 Analyzing...")
            result = chatbot.process_query(user_query)
            
            print(f"\n📊 Analysis Results:")
            print(f"   Confidence: {result['confidence_score']:.0%}")
            print(f"   Data Sources: {result['data_sources']}")
            
            print(f"\n💬 Response:")
            print(result['response'])
            
            if result['key_insights']:
                print(f"\n💡 Key Insights:")
                for insight in result['key_insights']:
                    print(f"   • {insight}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n👋 Demo completed. Thank you for testing the Dell Operations Chatbot!")
    
    # Cleanup note
    print(f"\nℹ️  To clear data: python -c \"from data_manager import DataManager; DataManager().clear_all_data()\"")

if __name__ == "__main__":
    main()