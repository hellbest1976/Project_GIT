"""
Enhanced Dell Organizational Operations Chatbot - CLI Interface
Integrates enhanced executive analysis with existing CLI functionality
"""

import os
import sys
import json
import pprint
from datetime import datetime
from typing import Dict, Any

# Import enhanced chatbot and existing modules
try:
    from enhanced_chatbot_engine import EnhancedDellChatbot
    import he_query_executor
    from he_question_parser import extract_he_filters
    from he_query_executor import load_he_chroma_as_dataframe, apply_he_filters
    from question_parser import extract_filters
    from query_executor import load_chroma_as_dataframe, apply_filters
    
    # Try to import FAQ if available
    try:
        from faq_knowledge import load_faq_knowledge, search_faq, search_faq_keywords
        FAQ_AVAILABLE = True
    except ImportError:
        FAQ_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required files are present and dependencies installed.")
    sys.exit(1)

class EnhancedDellCLI:
    """Enhanced CLI for Dell Operations Chatbot with executive-level analysis"""
    
    def __init__(self):
        self.setup_banner()
        self.enhanced_chatbot = None
        self.faqs = []
        self.initialize_system()
    
    def setup_banner(self):
        """Display startup banner"""
        print("=" * 70)
        print("🏢 DELL ORGANIZATIONAL OPERATIONS CHATBOT - ENHANCED CLI")
        print("=" * 70)
        print("Executive Analysis • Error Prevention • Strategic Insights")
        print("Built for CEO and Board of Directors")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def initialize_system(self):
        """Initialize all system components"""
        print("\n🔧 Initializing Enhanced Dell Operations Chatbot...")
        
        # Initialize enhanced chatbot
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            self.enhanced_chatbot = EnhancedDellChatbot(openai_key)
            print("✅ Enhanced chatbot engine initialized")
        except Exception as e:
            print(f"❌ Failed to initialize enhanced chatbot: {e}")
            sys.exit(1)
        
        # Load FAQ knowledge if available
        if FAQ_AVAILABLE:
            try:
                self.faqs = load_faq_knowledge("faq_knowledge.json")
                print(f"✅ Loaded {len(self.faqs)} FAQ entries")
            except Exception as e:
                print(f"⚠️ Could not load FAQ knowledge: {e}")
        
        # Get system status
        status = self.enhanced_chatbot.get_system_status()
        self.display_system_status(status)
        
        if not status["system_ready"]:
            print("\n❗ System not ready. Please check your ChromaDB configuration.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(1)
    
    def display_system_status(self, status: Dict[str, Any]):
        """Display current system status"""
        print(f"\n📊 System Status:")
        print(f"   Main Operations DB: {'✅ Online' if status['main_db_loaded'] else '❌ Offline'} ({status['main_db_records']:,} records)")
        print(f"   Human Error DB: {'✅ Online' if status['he_db_loaded'] else '❌ Offline'} ({status['he_db_records']:,} records)")
        print(f"   AI Enhancement: {'✅ Enabled' if status['openai_enabled'] else '🟡 Fallback Mode'}")
        print(f"   Current Dell FY: FY{status['current_dell_fy']}")
        print(f"   System Ready: {'✅ Yes' if status['system_ready'] else '❌ No'}")
    
    def run(self):
        """Main CLI loop"""
        print(f"\n💬 Enhanced Executive Analysis Chat")
        print("Type your question or use one of these commands:")
        print("   'samples' - Show sample executive questions")
        print("   'status'  - Show system status")
        print("   'help'    - Show detailed help")
        print("   'exit'    - Quit application")
        print()
        
        while True:
            try:
                question = input("🤔 Your executive question: ").strip()
                
                if not question:
                    continue
                    
                # Handle commands
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Thank you for using Dell Executive Operations Chatbot!")
                    break
                    
                elif question.lower() == 'samples':
                    self.show_sample_questions()
                    continue
                    
                elif question.lower() == 'status':
                    status = self.enhanced_chatbot.get_system_status()
                    self.display_system_status(status)
                    continue
                    
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                
                # Process the question
                self.process_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                import traceback
                print(traceback.format_exc())
    
    def show_sample_questions(self):
        """Display sample executive questions"""
        samples = [
            "What were the major human error incidents in the last 3 fiscal years?",
            "Show me error trends for Dell Technologies in FY2024",
            "Which companies had the highest error rates in Q1 2024?",
            "Give me a summary of P1 incidents for the last fiscal year",
            "What are the most common error triggers across all companies?",
            "Show failed changes due to human error for specific company",
            "Analyze operational efficiency trends by quarter",
            "What preventive measures should we implement based on error patterns?",
            "Show me non-Dell/DTMS problems in the last year",
            "What's the COE India error rate compared to other regions?"
        ]
        
        print("\n💡 Sample Executive Questions:")
        for i, sample in enumerate(samples, 1):
            print(f"   {i:2d}. {sample}")
        print()
    
    def show_help(self):
        """Display detailed help information"""
        print("\n📚 Dell Executive Operations Chatbot - Help")
        print("=" * 50)
        print("\n🎯 Purpose:")
        print("   Analyze organizational operations and prevent human errors")
        print("   Designed for Dell's CEO and Board of Directors")
        print("   Follows Dell's Feb-Jan fiscal year cycle")
        print("\n🗣️ Question Types:")
        print("   • Human Error Analysis: 'Show human errors for [company] in FY2024'")
        print("   • Operational Trends: 'What are the trends in Q1 2024?'")
        print("   • Company Comparisons: 'Which companies had highest error rates?'")
        print("   • Strategic Insights: 'What preventive measures should we implement?'")
        print("\n📅 Dell Fiscal Year:")
        print("   • FY2024 = Feb 2023 - Jan 2024")
        print("   • Default analysis covers last 3 fiscal years")
        print("   • Use 'Q1', 'Q2', 'Q3', 'Q4' for quarterly analysis")
        print("\n🎛️ Commands:")
        print("   • 'samples' - Show sample questions")
        print("   • 'status'  - Show system status")
        print("   • 'help'    - Show this help")
        print("   • 'exit'    - Quit application")
        print()
    
    def process_question(self, question: str):
        """Process user question with enhanced analysis"""
        print(f"\n🔍 Processing: {question}")
        
        # Check FAQ first if available
        if FAQ_AVAILABLE and self.faqs:
            faq_answer = self.check_faq(question)
            if faq_answer:
                print("\n✅ [FAQ MATCHED]:")
                print(faq_answer)
                return
        
        # Use enhanced chatbot for analysis
        try:
            result = self.enhanced_chatbot.process_executive_query(question)
            self.display_enhanced_result(result)
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            
            # Fallback to traditional analysis
            print("🔄 Falling back to traditional analysis...")
            self.fallback_analysis(question)
    
    def check_faq(self, question: str) -> str:
        """Check FAQ for direct answers"""
        try:
            # Try exact match first
            faq_answer = search_faq(question, self.faqs, threshold=0.98)
            if faq_answer:
                return faq_answer
            
            # Try keyword search
            return search_faq_keywords(question, self.faqs)
            
        except Exception as e:
            print(f"⚠️ FAQ search error: {e}")
            return None
    
    def display_enhanced_result(self, result: Dict[str, Any]):
        """Display enhanced analysis result"""
        print("\n" + "="*60)
        print("🤖 ENHANCED EXECUTIVE ANALYSIS")
        print("="*60)
        
        # Main response
        print(f"\n📝 Executive Summary:")
        print(result["response"])
        
        # Metrics dashboard
        print(f"\n📊 Analysis Metrics:")
        print(f"   Confidence Score: {result['confidence_score']:.0%}")
        print(f"   Data Sources: {result['data_sources']} records")
        print(f"   Database: {result['database_type']}")
        print(f"   Fiscal Year Context: {result['fiscal_year_context']}")
        
        # Key insights
        if result["key_insights"]:
            print(f"\n💡 Key Executive Insights:")
            for insight in result["key_insights"]:
                print(f"   • {insight}")
        
        # Strategic recommendations
        if result["recommendations"]:
            print(f"\n📋 Strategic Recommendations:")
            for rec in result["recommendations"]:
                print(f"   • {rec}")
        
        # Validation metrics
        validation = result.get("validation", {})
        if validation:
            print(f"\n✅ Validation Score: {validation.get('overall_score', 0):.0%}")
            print(f"   Data Referenced: {'✅' if validation.get('data_referenced') else '❌'}")
            print(f"   Metrics Included: {'✅' if validation.get('metrics_included') else '❌'}")
            print(f"   Executive Format: {'✅' if validation.get('executive_format') else '❌'}")
        
        # Technical details
        print(f"\n🔍 Technical Details:")
        filters = result.get("filters_applied", {})
        if filters:
            print(f"   Filters Applied:")
            for key, value in filters.items():
                if value and key != "intent":
                    print(f"     - {key}: {value}")
        
        print(f"   Analysis Timestamp: {result['timestamp']}")
        print("="*60)
    
    def fallback_analysis(self, question: str):
        """Fallback to traditional analysis methods"""
        print("\n🔄 Traditional Analysis Mode:")
        
        # Determine question type
        is_he_question = self.is_human_error_question(question)
        
        if is_he_question:
            print("📂 Using Human Error database...")
            filters = extract_he_filters(question)
            print(f"🔍 Extracted filters: {filters}")
            
            try:
                he_df = load_he_chroma_as_dataframe()
                result, filtered_df = apply_he_filters(he_df, filters, return_df=True)
                self.display_traditional_result(result, filtered_df, "Human Error")
            except Exception as e:
                print(f"❌ Human Error analysis failed: {e}")
        
        else:
            print("📂 Using Main Operations database...")
            filters = extract_filters(question)
            print(f"🔍 Extracted filters: {filters}")
            
            try:
                main_df = load_chroma_as_dataframe()
                result = apply_filters(main_df, filters)
                self.display_traditional_result(result, None, "Main Operations")
            except Exception as e:
                print(f"❌ Main operations analysis failed: {e}")
    
    def display_traditional_result(self, result: Dict, filtered_df, db_type: str):
        """Display traditional analysis result"""
        print(f"\n📊 Traditional {db_type} Analysis:")
        
        if not result:
            print("❌ No results found")
            return
        
        result_type = result.get("result_type", "unknown")
        
        if result_type == "count":
            print(f"Total: {result['value']}")
            
        elif result_type == "summary":
            summary = result.get("value", {})
            print(f"Total Records: {summary.get('total', 0)}")
            
            if summary.get("problem_trigger_breakdown"):
                print("\nProblem Trigger Breakdown:")
                for trigger, info in summary["problem_trigger_breakdown"].items():
                    print(f"  • {trigger}: {info.get('total', 0)}")
            
        elif result_type == "company_counts":
            print(f"Total: {result.get('total', 0)}")
            companies = result.get("value", {})
            print("\nCompany Breakdown:")
            for company, count in companies.items():
                print(f"  • {company}: {count}")
        
        else:
            print("Result:")
            pprint.pprint(result)
    
    def is_human_error_question(self, question: str) -> bool:
        """Check if question is about human errors"""
        q = question.lower()
        return any(kw in q for kw in [
            "human error", "human errors", "hea", "error avoidance", 
            "root cause type", "problem tkt trigger", "problem trigger"
        ])

def main():
    """Main entry point"""
    try:
        cli = EnhancedDellCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()