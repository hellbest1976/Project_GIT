"""
Configuration settings for Dell Organizational Operations Chatbot
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Any

class DellConfig:
    """Dell-specific configuration settings"""
    
    # Dell Fiscal Year Configuration (Feb-Jan)
    FISCAL_YEAR_START_MONTH = 2  # February
    FISCAL_YEAR_END_MONTH = 1    # January
    
    # Default analysis timeframe
    DEFAULT_ANALYSIS_YEARS = 3
    
    # Current fiscal year calculation
    @classmethod
    def get_current_fiscal_year(cls) -> int:
        """Calculate current Dell fiscal year (Feb-Jan cycle)"""
        now = datetime.now()
        if now.month >= cls.FISCAL_YEAR_START_MONTH:
            return now.year + 1
        else:
            return now.year
    
    @classmethod
    def get_fiscal_year_range(cls, years_back: int = DEFAULT_ANALYSIS_YEARS) -> tuple:
        """Get fiscal year range for analysis"""
        current_fy = cls.get_current_fiscal_year()
        start_fy = current_fy - years_back
        return (start_fy, current_fy)
    
    @classmethod
    def fiscal_year_to_dates(cls, fiscal_year: int) -> tuple:
        """Convert fiscal year to actual date range"""
        start_date = datetime(fiscal_year - 1, cls.FISCAL_YEAR_START_MONTH, 1)
        end_date = datetime(fiscal_year, cls.FISCAL_YEAR_END_MONTH, 31)
        return (start_date, end_date)

class ChatbotConfig:
    """General chatbot configuration"""
    
    # Database settings
    CHROMA_DB_PATH = "./data/chroma_db"
    COLLECTION_NAME = "dell_operations"
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-4"
    TEMPERATURE = 0.1  # Low temperature for accuracy
    MAX_TOKENS = 1500
    
    # Response settings
    MIN_CONFIDENCE_THRESHOLD = 0.7
    MAX_SEARCH_RESULTS = 10
    
    # Executive reporting settings
    EXECUTIVE_CONTEXT = {
        "target_audience": "CEO and Board of Directors",
        "response_style": "executive_summary",
        "precision_level": "high",
        "include_metrics": True,
        "include_trends": True,
        "include_recommendations": True
    }
    
    # Error prevention settings
    VALIDATION_CHECKS = [
        "data_completeness",
        "temporal_consistency", 
        "metric_validation",
        "trend_analysis"
    ]

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONNECTION_STRING = os.getenv("CHROME_DB_CONNECTION", "sqlite:///dell_operations.db")