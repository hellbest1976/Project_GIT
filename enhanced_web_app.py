"""
Enhanced Dell Organizational Operations Chatbot - Web Interface
Executive-level analysis integrating with existing ChromaDB system
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import traceback
from typing import Dict, Any, List

# Import enhanced chatbot and existing modules
try:
    from enhanced_chatbot_engine import EnhancedDellChatbot
    import he_query_executor
    from he_question_parser import extract_he_filters
    from question_parser import extract_filters
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all required files are present.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Dell Executive Operations Chatbot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #007DBA;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
}
.dell-blue { color: #007DBA; }
.executive-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #007DBA;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #007DBA 0%, #0056b3 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin: 0.5rem 0;
}
.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #fd7e14; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }
.status-online { color: #28a745; }
.status-warning { color: #fd7e14; }
.status-error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class EnhancedDellChatbotApp:
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'enhanced_chatbot' not in st.session_state:
            try:
                # Get OpenAI API key if available
                openai_key = os.getenv("OPENAI_API_KEY")
                st.session_state.enhanced_chatbot = EnhancedDellChatbot(openai_key)
                st.session_state.system_status = st.session_state.enhanced_chatbot.get_system_status()
            except Exception as e:
                st.error(f"Failed to initialize enhanced chatbot: {e}")
                st.stop()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'selected_sample' not in st.session_state:
            st.session_state.selected_sample = ""
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🏢 Dell Executive Operations Chatbot</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Analysis • Error Prevention • Strategic Insights</p>', 
                   unsafe_allow_html=True)
        
        # Sidebar for system status and controls
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2.5, 1])
        
        with col1:
            self.render_chat_interface()
        
        with col2:
            self.render_executive_dashboard()
    
    def render_sidebar(self):
        """Render sidebar with system status and controls"""
        st.sidebar.title("🎛️ System Control")
        
        # System status
        status = st.session_state.system_status
        st.sidebar.subheader("📊 System Status")
        
        # Database status
        main_status = "🟢 Online" if status["main_db_loaded"] else "🔴 Offline"
        he_status = "🟢 Online" if status["he_db_loaded"] else "🔴 Offline"
        openai_status = "🟢 Enabled" if status["openai_enabled"] else "🟡 Fallback Mode"
        
        st.sidebar.markdown(f"""
        **Main Operations DB:** {main_status}  
        **Human Error DB:** {he_status}  
        **AI Enhancement:** {openai_status}  
        **System Ready:** {'✅ Yes' if status['system_ready'] else '❌ No'}
        """)
        
        # Current Dell FY info
        current_fy = status["current_dell_fy"]
        st.sidebar.info(f"""
        **Current Dell FY:** FY{current_fy}  
        **Cycle:** Feb {current_fy-1} - Jan {current_fy}  
        **Default Analysis:** Last 3 FYs
        """)
        
        # Data statistics
        st.sidebar.subheader("📈 Data Overview")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Main DB", f"{status['main_db_records']:,}" if status['main_db_records'] > 0 else "No Data")
        with col2:
            st.metric("HE DB", f"{status['he_db_records']:,}" if status['he_db_records'] > 0 else "No Data")
        
        # Quick actions
        st.sidebar.subheader("🚀 Quick Actions")
        
        if st.sidebar.button("🔄 Refresh System", use_container_width=True):
            st.session_state.enhanced_chatbot = EnhancedDellChatbot(os.getenv("OPENAI_API_KEY"))
            st.session_state.system_status = st.session_state.enhanced_chatbot.get_system_status()
            st.rerun()
        
        if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.sidebar.success("Chat history cleared!")
            st.rerun()
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        
        # API Key management
        with st.sidebar.expander("🔑 OpenAI Configuration"):
            current_key = "Set" if status["openai_enabled"] else "Not Set"
            st.write(f"Current Status: {current_key}")
            new_key = st.text_input("API Key", type="password", placeholder="sk-...")
            if st.button("Update API Key") and new_key:
                os.environ["OPENAI_API_KEY"] = new_key
                st.session_state.enhanced_chatbot = EnhancedDellChatbot(new_key)
                st.session_state.system_status = st.session_state.enhanced_chatbot.get_system_status()
                st.success("API Key updated!")
                st.rerun()
        
        # Database paths (read-only info)
        with st.sidebar.expander("🗄️ Database Info"):
            if hasattr(he_query_executor, 'CHROMA_PATH'):
                st.code(f"HE DB: {he_query_executor.CHROMA_PATH}")
            st.write("**Collections:**")
            if hasattr(he_query_executor, 'COLLECTION_NAME'):
                st.write(f"• {he_query_executor.COLLECTION_NAME}")
    
    def render_chat_interface(self):
        """Render main chat interface"""
        st.subheader("💬 Executive Analysis Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            self.display_message(message)
        
        # Query input section
        st.subheader("❓ Ask Your Executive Question")
        
        # Sample questions for executives
        executive_samples = [
            "What were the major human error incidents in the last 3 fiscal years?",
            "Show me error trends for Dell Technologies in FY2024",
            "Which companies had the highest error rates in Q1 2024?",
            "Give me a summary of P1 incidents for the last fiscal year",
            "What are the most common error triggers across all companies?",
            "Show failed changes due to human error for specific company",
            "Analyze operational efficiency trends by quarter",
            "What preventive measures should we implement based on error patterns?"
        ]
        
        # Sample question selector
        sample_question = st.selectbox(
            "💡 Sample Executive Questions:",
            [""] + executive_samples,
            key="sample_selector"
        )
        
        # Manual query input
        user_query = st.text_area(
            "Your Executive Question:",
            value=sample_question if sample_question else st.session_state.selected_sample,
            height=120,
            placeholder="Ask about operational changes, error patterns, financial impacts, trends, or strategic insights...",
            help="Examples: 'What were the major operational changes in Q1 FY2024?' or 'Show me human error trends for Dell Technologies'"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            analyze_button = st.button(
                "🔍 **Analyze Executive Query**", 
                type="primary", 
                use_container_width=True,
                disabled=not st.session_state.system_status["system_ready"]
            )
        
        with col2:
            if st.button("📋 Copy Sample", use_container_width=True):
                if sample_question:
                    st.session_state.selected_sample = sample_question
                    st.rerun()
        
        with col3:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.selected_sample = ""
                st.rerun()
        
        # Process query
        if analyze_button:
            if user_query.strip():
                self.process_executive_query(user_query.strip())
            else:
                st.warning("Please enter an executive question to analyze.")
        
        # System readiness check
        if not st.session_state.system_status["system_ready"]:
            st.error("⚠️ System not ready. No databases are loaded. Please check your ChromaDB configuration.")
    
    def process_executive_query(self, query: str):
        """Process executive query and display results"""
        with st.spinner("🤖 Analyzing your executive query..."):
            try:
                result = st.session_state.enhanced_chatbot.process_executive_query(query)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.error(traceback.format_exc())
    
    def display_message(self, message: Dict[str, Any]):
        """Display a chat message with executive formatting"""
        result = message["result"]
        
        # User question header
        st.markdown(f"**🤔 Executive Question:** {message['query']}")
        
        # Response card
        st.markdown('<div class="executive-card">', unsafe_allow_html=True)
        
        # Main response
        st.markdown("**🤖 Executive Analysis:**")
        st.markdown(result["response"])
        
        # Metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = result["confidence_score"]
            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            st.markdown(f'''
            <div class="metric-card">
                <div style="font-size: 0.9rem;">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{confidence:.0%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div style="font-size: 0.9rem;">Data Sources</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{result["data_sources"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            db_type = result["database_type"]
            st.markdown(f'''
            <div class="metric-card">
                <div style="font-size: 0.9rem;">Database</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{db_type}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            validation_score = result.get("validation", {}).get("overall_score", 0)
            st.markdown(f'''
            <div class="metric-card">
                <div style="font-size: 0.9rem;">Validation</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{validation_score:.0%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Key insights
        if result["key_insights"]:
            st.markdown("**💡 Key Executive Insights:**")
            for insight in result["key_insights"]:
                st.markdown(f"• {insight}")
        
        # Strategic recommendations
        if result["recommendations"]:
            st.markdown("**📋 Strategic Recommendations:**")
            for rec in result["recommendations"]:
                st.markdown(f"• {rec}")
        
        # Technical details (expandable)
        with st.expander("🔍 Technical Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Fiscal Year Context:**")
                st.write(result["fiscal_year_context"])
                
                st.write("**Filters Applied:**")
                filters = result.get("filters_applied", {})
                if filters:
                    for key, value in filters.items():
                        if value:
                            st.write(f"• {key}: {value}")
                else:
                    st.write("No specific filters applied")
            
            with col2:
                st.write("**Validation Metrics:**")
                validation = result.get("validation", {})
                for key, value in validation.items():
                    status = "✅" if value else "❌"
                    st.write(f"{status} {key.replace('_', ' ').title()}")
                
                st.write("**Analysis Timestamp:**")
                st.write(result["timestamp"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_executive_dashboard(self):
        """Render executive dashboard"""
        st.subheader("📊 Executive Dashboard")
        
        # Current system overview
        status = st.session_state.system_status
        
        # Dell FY information
        current_fy = status["current_dell_fy"]
        
        st.markdown(f'''
        <div class="executive-card">
            <h4 class="dell-blue">📅 Dell Fiscal Year Context</h4>
            <p><strong>Current FY:</strong> FY{current_fy}</p>
            <p><strong>Period:</strong> Feb {current_fy-1} - Jan {current_fy}</p>
            <p><strong>Analysis Default:</strong> Last 3 Fiscal Years</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # System health indicators
        st.markdown("**🏥 System Health:**")
        
        health_items = [
            ("Main Operations DB", status["main_db_loaded"]),
            ("Human Error DB", status["he_db_loaded"]),
            ("AI Enhancement", status["openai_enabled"]),
            ("System Ready", status["system_ready"])
        ]
        
        for item, status_val in health_items:
            icon = "🟢" if status_val else "🔴"
            st.markdown(f"{icon} {item}")
        
        # Recent activity
        if st.session_state.chat_history:
            st.markdown("**📈 Recent Executive Analysis:**")
            recent = st.session_state.chat_history[-1]
            
            st.markdown(f'''
            <div style="background-color: #f1f3f4; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <strong>Last Query:</strong> {recent['query'][:60]}...
                <br><strong>Confidence:</strong> {recent['result']['confidence_score']:.0%}
                <br><strong>Database:</strong> {recent['result']['database_type']}
            </div>
            ''', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("**📊 Database Statistics:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Main Records", 
                f"{status['main_db_records']:,}" if status['main_db_records'] > 0 else "No Data",
                help="Total records in main operations database"
            )
        
        with col2:
            st.metric(
                "HE Records", 
                f"{status['he_db_records']:,}" if status['he_db_records'] > 0 else "No Data",
                help="Total records in human error database"
            )
        
        # Analysis capabilities
        st.markdown("**🧠 Analysis Capabilities:**")
        
        capabilities = [
            "✅ Multi-database analysis",
            "✅ Dell fiscal year alignment",
            "✅ Executive-level insights",
            "✅ Confidence scoring",
            "✅ Strategic recommendations",
            "✅ Error pattern detection",
            "✅ Trend analysis",
            "✅ Financial impact assessment"
        ]
        
        for cap in capabilities:
            st.markdown(cap)
        
        # Quick actions
        st.markdown("**⚡ Quick Actions:**")
        
        if st.button("📊 System Overview", use_container_width=True):
            st.json(status)
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.session_state.system_status = st.session_state.enhanced_chatbot.get_system_status()
            st.success("Status refreshed!")
            st.rerun()
        
        # Export functionality
        if st.session_state.chat_history:
            if st.button("📤 Export Chat History", use_container_width=True):
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "system_status": status,
                    "chat_history": st.session_state.chat_history
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"dell_executive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

# Run the application
if __name__ == "__main__":
    app = EnhancedDellChatbotApp()
    app.run()