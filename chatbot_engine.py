"""
Core Chatbot Engine for Dell Organizational Operations Analysis
Designed for CEO and Board of Directors with high accuracy and precision
"""

import openai
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime
from data_manager import DataManager
from config import DellConfig, ChatbotConfig, OPENAI_API_KEY
import logging

class DellOperationsChatbot:
    """Executive-level chatbot for organizational operations analysis"""
    
    def __init__(self):
        # Initialize OpenAI
        openai.api_key = OPENAI_API_KEY
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Executive context for responses
        self.executive_context = ChatbotConfig.EXECUTIVE_CONTEXT
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query and return comprehensive analysis
        
        Args:
            user_query: Question from CEO/Board member
            
        Returns:
            Dict containing response, data sources, confidence, and recommendations
        """
        try:
            self.logger.info(f"Processing query: {user_query}")
            
            # Step 1: Search relevant data
            relevant_data = self.data_manager.search_relevant_data(user_query)
            
            if not relevant_data:
                return self._create_no_data_response(user_query)
            
            # Step 2: Analyze data quality and confidence
            confidence_score = self._calculate_confidence(relevant_data, user_query)
            
            # Step 3: Generate executive response
            response = self._generate_executive_response(user_query, relevant_data)
            
            # Step 4: Extract key insights and recommendations
            insights = self._extract_insights(relevant_data, user_query)
            
            # Step 5: Validate response accuracy
            validation_results = self._validate_response(response, relevant_data)
            
            return {
                "response": response,
                "confidence_score": confidence_score,
                "data_sources": len(relevant_data),
                "fiscal_year_context": self._get_fiscal_year_context(user_query),
                "key_insights": insights,
                "recommendations": self._generate_recommendations(relevant_data, user_query),
                "validation": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self._create_error_response(str(e))
    
    def _generate_executive_response(self, query: str, data: List[Dict[str, Any]]) -> str:
        """Generate executive-level response using OpenAI"""
        
        # Prepare data context
        data_context = "\n".join([
            f"Data {i+1}: {item['content']}" 
            for i, item in enumerate(data[:5])  # Limit to top 5 most relevant
        ])
        
        # Get fiscal year context
        fy_context = self._get_fiscal_year_context(query)
        
        prompt = f"""
        You are an AI assistant providing analysis for Dell's CEO and Board of Directors.
        
        CONTEXT:
        - Dell follows Feb-Jan fiscal year cycle
        - {fy_context}
        - Focus on operational changes and human error prevention
        - Provide accurate, precise, executive-level insights
        
        QUERY: {query}
        
        RELEVANT DATA:
        {data_context}
        
        INSTRUCTIONS:
        1. Provide a clear, executive summary-style response
        2. Include specific metrics and financial impacts when available
        3. Highlight trends and patterns across the timeframe
        4. Focus on operational efficiency and error prevention insights
        5. Be concise but comprehensive
        6. Include confidence indicators for your analysis
        
        RESPONSE FORMAT:
        - Executive Summary (2-3 sentences)
        - Key Findings (bullet points with metrics)
        - Trend Analysis (if applicable)
        - Risk Assessment (if relevant)
        
        Generate response:
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a strategic analyst for Dell's executive team."},
                    {"role": "user", "content": prompt}
                ],
                temperature=ChatbotConfig.TEMPERATURE,
                max_tokens=ChatbotConfig.MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return self._fallback_response(query, data)
    
    def _fallback_response(self, query: str, data: List[Dict[str, Any]]) -> str:
        """Generate fallback response when OpenAI is unavailable"""
        
        # Extract key information from data
        departments = set()
        financial_impacts = []
        fiscal_years = set()
        
        for item in data:
            metadata = item.get('metadata', {})
            departments.add(metadata.get('department', 'Unknown'))
            
            # Extract financial impact
            content = item.get('content', '')
            financial_match = re.search(r'\$?(\d{1,3}(?:,?\d{3})*(?:\.\d{2})?)[MmKk]?', content)
            if financial_match:
                financial_impacts.append(financial_match.group(1))
            
            # Extract fiscal year
            fy_match = re.search(r'FY(\d{4})', content)
            if fy_match:
                fiscal_years.add(fy_match.group(1))
        
        summary = f"""
        EXECUTIVE SUMMARY:
        Based on analysis of {len(data)} relevant operational records, here are the key findings for your query about {query}.
        
        KEY FINDINGS:
        • Departments involved: {', '.join(departments)}
        • Fiscal years covered: {', '.join(sorted(fiscal_years)) if fiscal_years else 'Multiple years'}
        • Data points analyzed: {len(data)} records
        
        TREND ANALYSIS:
        The available data shows operational activities across {len(departments)} departments with documented impacts and outcomes.
        
        NOTE: This is a summary based on available data. For detailed OpenAI-powered analysis, please ensure API connectivity.
        """
        
        return summary
    
    def _calculate_confidence(self, data: List[Dict[str, Any]], query: str) -> float:
        """Calculate confidence score for the response"""
        if not data:
            return 0.0
        
        # Base confidence on data relevance and completeness
        confidence_factors = []
        
        # Factor 1: Number of relevant documents
        doc_score = min(len(data) / ChatbotConfig.MAX_SEARCH_RESULTS, 1.0)
        confidence_factors.append(doc_score)
        
        # Factor 2: Recency of data (Dell fiscal year context)
        current_fy = DellConfig.get_current_fiscal_year()
        recent_data_count = 0
        
        for item in data:
            content = item.get('content', '')
            fy_match = re.search(r'FY(\d{4})', content)
            if fy_match:
                fy = int(fy_match.group(1))
                if current_fy - fy <= 3:  # Within last 3 fiscal years
                    recent_data_count += 1
        
        recency_score = recent_data_count / len(data) if data else 0
        confidence_factors.append(recency_score)
        
        # Factor 3: Data completeness (metadata richness)
        metadata_scores = []
        for item in data:
            metadata = item.get('metadata', {})
            completeness = len([v for v in metadata.values() if v]) / max(len(metadata), 1)
            metadata_scores.append(completeness)
        
        metadata_score = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0
        confidence_factors.append(metadata_score)
        
        # Calculate weighted average
        weights = [0.4, 0.4, 0.2]  # Prioritize relevance and recency
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return round(confidence, 2)
    
    def _get_fiscal_year_context(self, query: str) -> str:
        """Determine fiscal year context for the query"""
        current_fy = DellConfig.get_current_fiscal_year()
        
        # Check if specific year mentioned
        year_match = re.search(r'FY(\d{4})|(\d{4})', query)
        if year_match:
            mentioned_year = year_match.group(1) or year_match.group(2)
            return f"Analysis focused on FY{mentioned_year}"
        
        # Default to last 3 years
        start_fy, end_fy = DellConfig.get_fiscal_year_range()
        return f"Analysis covers FY{start_fy} to FY{end_fy} (default 3-year period)"
    
    def _extract_insights(self, data: List[Dict[str, Any]], query: str) -> List[str]:
        """Extract key insights from the data"""
        insights = []
        
        # Analyze patterns in data
        departments = {}
        error_types = {}
        financial_impacts = []
        
        for item in data:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            
            # Department analysis
            dept = metadata.get('department', 'Unknown')
            departments[dept] = departments.get(dept, 0) + 1
            
            # Error type analysis
            error_type = metadata.get('error_type', '')
            if error_type and error_type.lower() != 'none':
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Financial impact extraction
            financial_match = re.search(r'\$?(\d{1,3}(?:,?\d{3})*)', content)
            if financial_match:
                try:
                    amount = int(financial_match.group(1).replace(',', ''))
                    financial_impacts.append(amount)
                except:
                    pass
        
        # Generate insights
        if departments:
            top_dept = max(departments, key=departments.get)
            insights.append(f"Most active department: {top_dept} ({departments[top_dept]} operations)")
        
        if error_types:
            common_error = max(error_types, key=error_types.get)
            insights.append(f"Most common error type: {common_error} ({error_types[common_error]} instances)")
        
        if financial_impacts:
            total_impact = sum(financial_impacts)
            avg_impact = total_impact / len(financial_impacts)
            insights.append(f"Financial impact analysis: ${total_impact:,} total, ${avg_impact:,.0f} average")
        
        return insights
    
    def _generate_recommendations(self, data: List[Dict[str, Any]], query: str) -> List[str]:
        """Generate strategic recommendations based on data analysis"""
        recommendations = []
        
        # Analyze error patterns for prevention recommendations
        error_types = {}
        high_impact_areas = []
        
        for item in data:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            
            error_type = metadata.get('error_type', '')
            if error_type and error_type.lower() not in ['none', '']:
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            impact_level = metadata.get('impact_level', '')
            if impact_level.lower() in ['high', 'critical']:
                dept = metadata.get('department', 'Unknown')
                high_impact_areas.append(dept)
        
        # Generate recommendations
        if error_types:
            common_error = max(error_types, key=error_types.get)
            recommendations.append(f"Implement focused training to address {common_error} (most frequent error type)")
        
        if high_impact_areas:
            dept_counts = {}
            for dept in high_impact_areas:
                dept_counts[dept] = dept_counts.get(dept, 0) + 1
            
            critical_dept = max(dept_counts, key=dept_counts.get)
            recommendations.append(f"Prioritize process review in {critical_dept} department (highest impact operations)")
        
        # General Dell-specific recommendations
        recommendations.append("Continue quarterly operational reviews aligned with Dell's Feb-Jan fiscal calendar")
        recommendations.append("Implement predictive analytics for early error detection and prevention")
        
        return recommendations
    
    def _validate_response(self, response: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate response accuracy and completeness"""
        validation = {
            "data_completeness": len(data) >= 3,
            "fiscal_year_alignment": "FY" in response or any("FY" in item.get('content', '') for item in data),
            "executive_appropriate": len(response.split()) >= 50,  # Minimum executive summary length
            "metrics_included": bool(re.search(r'\d+%|\$\d+|(\d{1,3}(?:,\d{3})*)', response))
        }
        
        validation["overall_score"] = sum(validation.values()) / len(validation)
        return validation
    
    def _create_no_data_response(self, query: str) -> Dict[str, Any]:
        """Create response when no relevant data found"""
        return {
            "response": f"No relevant operational data found for your query: '{query}'. Please ensure data has been loaded for the specified timeframe or check query parameters.",
            "confidence_score": 0.0,
            "data_sources": 0,
            "fiscal_year_context": self._get_fiscal_year_context(query),
            "key_insights": [],
            "recommendations": ["Load relevant operational data for analysis", "Verify fiscal year parameters"],
            "validation": {"data_available": False},
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create response for system errors"""
        return {
            "response": f"System error occurred during analysis: {error_message}. Please try again or contact support.",
            "confidence_score": 0.0,
            "data_sources": 0,
            "fiscal_year_context": "Unable to determine",
            "key_insights": [],
            "recommendations": ["Check system configuration", "Verify data connectivity"],
            "validation": {"system_error": True},
            "timestamp": datetime.now().isoformat()
        }