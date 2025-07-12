"""
Dell Organizational Operations Chatbot - Main Application
Executive-level analysis for CEO and Board of Directors
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import traceback
from typing import Dict, Any, List

# Import our custom modules
try:
    from chatbot_engine import DellOperationsChatbot
    from data_manager import DataManager, DataTemplates
    from config import DellConfig, ChatbotConfig
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all required files are present.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Dell Operations Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.executive-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.metric-card {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.confidence-high { color: #4caf50; font-weight: bold; }
.confidence-medium { color: #ff9800; font-weight: bold; }
.confidence-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class DellChatbotApp:
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chatbot' not in st.session_state:
            try:
                st.session_state.chatbot = DellOperationsChatbot()
                st.session_state.data_manager = DataManager()
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {e}")
                st.stop()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🏢 Dell Organizational Operations Chatbot</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Executive Analysis • Error Prevention • Strategic Insights</p>', 
                   unsafe_allow_html=True)
        
        # Sidebar for data management and settings
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_chat_interface()
        
        with col2:
            self.render_dashboard()
    
    def render_sidebar(self):
        """Render sidebar with data management options"""
        st.sidebar.title("📊 Data Management")
        
        # Display current fiscal year info
        current_fy = DellConfig.get_current_fiscal_year()
        fy_range = DellConfig.get_fiscal_year_range()
        
        st.sidebar.info(f"""
        **Dell Fiscal Year:** FY{current_fy}  
        **Analysis Period:** FY{fy_range[0]} - FY{fy_range[1]}  
        **Cycle:** Feb-Jan
        """)
        
        # Data input options
        st.sidebar.subheader("💾 Add Data")
        
        data_input_method = st.sidebar.selectbox(
            "Choose input method:",
            ["Manual Entry", "CSV Upload", "Database Query", "Sample Data"]
        )
        
        if data_input_method == "Manual Entry":
            self.render_manual_data_input()
        elif data_input_method == "CSV Upload":
            self.render_csv_upload()
        elif data_input_method == "Database Query":
            self.render_database_input()
        elif data_input_method == "Sample Data":
            self.render_sample_data()
        
        # Data statistics
        self.render_data_stats()
        
        # Clear data option
        if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
            if st.session_state.data_manager.clear_all_data():
                st.sidebar.success("All data cleared!")
                st.session_state.data_loaded = False
                st.rerun()
    
    def render_manual_data_input(self):
        """Render manual data entry form"""
        st.sidebar.subheader("✍️ Manual Data Entry")
        
        with st.sidebar.form("manual_data_form"):
            content = st.text_area("Content", height=100, 
                                 placeholder="Describe the operational change, impact, and outcomes...")
            
            col1, col2 = st.columns(2)
            with col1:
                department = st.selectbox("Department", 
                    ["Manufacturing", "IT", "Sales", "Supply Chain", "Finance", "HR", "Operations", "Other"])
                fiscal_year = st.selectbox("Fiscal Year", 
                    [f"FY{year}" for year in range(current_fy-5, current_fy+1)])
            
            with col2:
                impact_level = st.selectbox("Impact Level", 
                    ["Low", "Medium", "High", "Critical"])
                error_type = st.selectbox("Error Type", 
                    ["None", "Process Error", "Communication Gap", "Configuration Error", "Training Gap", "Other"])
            
            financial_impact = st.number_input("Financial Impact ($)", min_value=0, step=1000)
            
            submitted = st.form_submit_button("Add Data")
            
            if submitted and content:
                data_entry = {
                    "content": content,
                    "department": department,
                    "fiscal_year": fiscal_year,
                    "impact_level": impact_level,
                    "error_type": error_type,
                    "financial_impact": str(financial_impact),
                    "entry_date": datetime.now().isoformat()
                }
                
                if st.session_state.data_manager.add_data_manually([data_entry]):
                    st.sidebar.success("Data added successfully!")
                    st.session_state.data_loaded = True
                    st.rerun()
                else:
                    st.sidebar.error("Failed to add data")
    
    def render_csv_upload(self):
        """Render CSV upload interface"""
        st.sidebar.subheader("📁 CSV Upload")
        
        # Show template
        if st.sidebar.button("📋 Download CSV Template"):
            template = DataTemplates.get_csv_template()
            st.sidebar.download_button(
                label="Download Template",
                data=template,
                file_name="dell_operations_template.csv",
                mime="text/csv"
            )
        
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Preview data
                df = pd.read_csv(temp_path)
                st.sidebar.write("Preview:")
                st.sidebar.dataframe(df.head(3))
                
                if st.sidebar.button("Upload Data"):
                    if st.session_state.data_manager.add_data_from_csv(temp_path):
                        st.sidebar.success(f"Uploaded {len(df)} records!")
                        st.session_state.data_loaded = True
                        os.remove(temp_path)  # Clean up
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to upload data")
                
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
    
    def render_database_input(self):
        """Render database query interface"""
        st.sidebar.subheader("🗄️ Database Query")
        
        with st.sidebar.form("db_form"):
            connection_string = st.text_input("Connection String", 
                value="sqlite:///dell_operations.db",
                help="SQLite connection string")
            
            query = st.text_area("SQL Query", 
                value="SELECT * FROM operations WHERE fiscal_year IN ('FY2022', 'FY2023', 'FY2024')",
                height=100)
            
            submitted = st.form_submit_button("Execute Query")
            
            if submitted and connection_string and query:
                try:
                    if st.session_state.data_manager.add_data_from_database(connection_string, query):
                        st.sidebar.success("Data loaded from database!")
                        st.session_state.data_loaded = True
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to load data")
                except Exception as e:
                    st.sidebar.error(f"Database error: {e}")
    
    def render_sample_data(self):
        """Render sample data loading"""
        st.sidebar.subheader("🎯 Sample Data")
        
        st.sidebar.write("Load sample Dell operational data for testing:")
        
        if st.sidebar.button("Load Sample Data"):
            sample_data = DataTemplates.get_manual_data_example()
            
            # Add more sample data
            additional_samples = [
                {
                    "content": "FY2024 Q1 manufacturing automation initiative increased production efficiency by 18% while reducing human errors by 30%",
                    "department": "Manufacturing",
                    "fiscal_year": "FY2024",
                    "quarter": "Q1",
                    "impact_level": "High",
                    "financial_impact": "4500000",
                    "error_reduction": "30%"
                },
                {
                    "content": "Sales team restructuring in FY2023 Q3 improved customer response time by 40% and increased closure rate by 15%",
                    "department": "Sales",
                    "fiscal_year": "FY2023",
                    "quarter": "Q3",
                    "impact_level": "High",
                    "financial_impact": "2200000",
                    "efficiency_gain": "40% response time improvement"
                }
            ]
            
            all_samples = sample_data + additional_samples
            
            if st.session_state.data_manager.add_data_manually(all_samples):
                st.sidebar.success(f"Loaded {len(all_samples)} sample records!")
                st.session_state.data_loaded = True
                st.rerun()
            else:
                st.sidebar.error("Failed to load sample data")
    
    def render_data_stats(self):
        """Render data statistics"""
        st.sidebar.subheader("📈 Data Statistics")
        
        stats = st.session_state.data_manager.get_collection_stats()
        
        if stats:
            st.sidebar.metric("Total Records", stats.get("total_documents", 0))
            st.sidebar.write(f"**Collection:** {stats.get('collection_name', 'N/A')}")
            
            if stats.get("fiscal_year_range"):
                fy_range = stats["fiscal_year_range"]
                st.sidebar.write(f"**FY Range:** FY{fy_range[0]} - FY{fy_range[1]}")
        else:
            st.sidebar.write("No data loaded yet")
    
    def render_chat_interface(self):
        """Render main chat interface"""
        st.subheader("💬 Executive Analysis Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            self.display_message(message)
        
        # Query input
        st.subheader("Ask Your Question")
        
        # Sample questions
        sample_questions = [
            "What were the major operational changes in the last 3 fiscal years?",
            "Which departments had the highest error rates?",
            "Show me financial impact of process improvements",
            "What are the trends in operational efficiency?",
            "Which areas need immediate attention for error prevention?"
        ]
        
        selected_question = st.selectbox(
            "Sample questions (optional):",
            [""] + sample_questions
        )
        
        user_query = st.text_area(
            "Your question:",
            value=selected_question,
            height=100,
            placeholder="Ask about operational changes, error patterns, financial impacts, or trends..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Analyze", type="primary", disabled=not st.session_state.data_loaded):
                if user_query.strip():
                    self.process_query(user_query)
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please add data using the sidebar before asking questions.")
    
    def process_query(self, query: str):
        """Process user query and display results"""
        with st.spinner("🤖 Analyzing your question..."):
            try:
                result = st.session_state.chatbot.process_query(query)
                
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
        """Display a chat message"""
        result = message["result"]
        
        # User question
        st.markdown(f"**🤔 Question:** {message['query']}")
        
        # Response card
        st.markdown('<div class="executive-card">', unsafe_allow_html=True)
        
        # Main response
        st.markdown("**🤖 Analysis:**")
        st.write(result["response"])
        
        # Confidence and metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = result["confidence_score"]
            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            st.markdown(f'<div class="metric-card"><div class="{confidence_class}">Confidence</div><div>{confidence:.0%}</div></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><div>Data Sources</div><div>{result["data_sources"]}</div></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><div>FY Context</div><div>{result["fiscal_year_context"].split("(")[0]}</div></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            validation_score = result.get("validation", {}).get("overall_score", 0)
            st.markdown(f'<div class="metric-card"><div>Validation</div><div>{validation_score:.0%}</div></div>', 
                       unsafe_allow_html=True)
        
        # Key insights
        if result["key_insights"]:
            st.markdown("**💡 Key Insights:**")
            for insight in result["key_insights"]:
                st.write(f"• {insight}")
        
        # Recommendations
        if result["recommendations"]:
            st.markdown("**📋 Recommendations:**")
            for rec in result["recommendations"]:
                st.write(f"• {rec}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_dashboard(self):
        """Render executive dashboard"""
        st.subheader("📊 Executive Dashboard")
        
        # Current fiscal year info
        current_fy = DellConfig.get_current_fiscal_year()
        fy_dates = DellConfig.fiscal_year_to_dates(current_fy)
        
        st.info(f"""
        **Current Dell Fiscal Year:** FY{current_fy}  
        **Period:** {fy_dates[0].strftime('%b %Y')} - {fy_dates[1].strftime('%b %Y')}
        """)
        
        # System status
        stats = st.session_state.data_manager.get_collection_stats()
        
        if stats:
            st.markdown("**System Status:**")
            st.success("✅ System Online")
            st.metric("Data Records", stats.get("total_documents", 0))
            
            # Recent activity
            if st.session_state.chat_history:
                st.markdown("**Recent Analysis:**")
                recent = st.session_state.chat_history[-1]
                st.write(f"Last Query: {recent['query'][:50]}...")
                st.write(f"Confidence: {recent['result']['confidence_score']:.0%}")
        else:
            st.warning("⚠️ No data loaded")
        
        # Quick actions
        st.markdown("**Quick Actions:**")
        
        if st.button("📈 Data Overview", use_container_width=True):
            if stats:
                st.json(stats)
        
        if st.button("🔧 System Health", use_container_width=True):
            health = {
                "ChatBot Engine": "✅ Online",
                "Data Manager": "✅ Connected", 
                "ChromaDB": "✅ Available",
                "Fiscal Year": f"✅ FY{current_fy}",
                "Data Records": stats.get("total_documents", 0) if stats else 0
            }
            st.json(health)

# Run the application
if __name__ == "__main__":
    app = DellChatbotApp()
    app.run()