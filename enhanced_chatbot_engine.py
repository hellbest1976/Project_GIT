"""
Enhanced Dell Organizational Operations Chatbot Engine
Integrates with existing ChromaDB system while adding executive-level capabilities
"""

import openai
import pandas as pd
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Import existing modules
try:
    import he_query_executor
    from he_question_parser import extract_he_filters
    from he_query_executor import load_he_chroma_as_dataframe, apply_he_filters
    from question_parser import extract_filters
    from query_executor import load_chroma_as_dataframe, apply_filters
except ImportError as e:
    print(f"Warning: Could not import existing modules: {e}")

class EnhancedDellChatbot:
    """Enhanced executive-level chatbot that integrates with existing Dell system"""
    
    def __init__(self, openai_api_key: str = None):
        # Initialize OpenAI if provided
        if openai_api_key:
            openai.api_key = openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load data
        self.main_df = None
        self.he_df = None
        self._load_data()
        
        # Executive context settings
        self.executive_context = {
            "target_audience": "CEO and Board of Directors",
            "response_style": "executive_summary",
            "precision_level": "high",
            "include_metrics": True,
            "include_trends": True,
            "include_recommendations": True
        }
    
    def _load_data(self):
        """Load both main and human error ChromaDB data"""
        try:
            self.main_df = load_chroma_as_dataframe()
            self.logger.info("✅ Main ChromaDB loaded successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load main ChromaDB: {e}")
        
        try:
            self.he_df = load_he_chroma_as_dataframe()
            self.logger.info("✅ Human Error ChromaDB loaded successfully")
            
            # Ensure FY and quarter columns exist
            if self.he_df is not None and 'fy' not in self.he_df.columns:
                self._assign_dell_fy_quarter()
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load Human Error ChromaDB: {e}")
    
    def _assign_dell_fy_quarter(self):
        """Assign Dell fiscal year and quarter to human error data"""
        if self.he_df is None:
            return
            
        def assign_fy_quarter(row):
            date_obj = pd.to_datetime(row.get('opened'), errors='coerce')
            if pd.isna(date_obj):
                return pd.Series({'fy': None, 'quarter': None})
            
            month = date_obj.month
            year = date_obj.year
            
            if month == 1:
                fy = year
                q = 4
            elif 2 <= month <= 4:
                fy = year + 1
                q = 1
            elif 5 <= month <= 7:
                fy = year + 1
                q = 2
            elif 8 <= month <= 10:
                fy = year + 1
                q = 3
            elif 11 <= month <= 12:
                fy = year + 1
                q = 4
            else:
                fy, q = None, None
            
            return pd.Series({'fy': fy, 'quarter': q})
        
        fy_quarter_data = self.he_df.apply(assign_fy_quarter, axis=1)
        self.he_df['fy'] = fy_quarter_data['fy']
        self.he_df['quarter'] = fy_quarter_data['quarter']
        
        self.logger.info("✅ Dell FY and quarter assigned to HE data")
    
    def process_executive_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process executive query with enhanced analysis capabilities
        
        Args:
            user_query: Question from CEO/Board member
            
        Returns:
            Dict containing comprehensive executive response
        """
        try:
            self.logger.info(f"Processing executive query: {user_query}")
            
            # Determine if this is a human error question
            is_he_question = self._is_human_error_question(user_query)
            
            # Extract appropriate filters
            if is_he_question:
                filters = extract_he_filters(user_query)
                db_type = "Human Error"
                relevant_df = self.he_df
            else:
                filters = extract_filters(user_query)
                db_type = "Main Operations"
                relevant_df = self.main_df
            
            if relevant_df is None or relevant_df.empty:
                return self._create_no_data_response(user_query, db_type)
            
            # Apply filters and get results
            if is_he_question:
                result, filtered_df = apply_he_filters(relevant_df, filters, return_df=True)
            else:
                result = apply_filters(relevant_df, filters)
                filtered_df = relevant_df  # Fallback for main DB
            
            # Calculate confidence and metrics
            confidence_score = self._calculate_confidence(filtered_df, result, filters)
            
            # Generate executive response
            executive_response = self._generate_executive_response(
                user_query, result, filtered_df, filters, db_type
            )
            
            # Extract insights and recommendations
            insights = self._extract_executive_insights(filtered_df, result, filters, db_type)
            recommendations = self._generate_strategic_recommendations(
                filtered_df, result, filters, db_type
            )
            
            # Validate response
            validation = self._validate_executive_response(
                executive_response, filtered_df, result
            )
            
            return {
                "response": executive_response,
                "confidence_score": confidence_score,
                "data_sources": len(filtered_df) if filtered_df is not None else 0,
                "database_type": db_type,
                "fiscal_year_context": self._get_fiscal_year_context(filters),
                "key_insights": insights,
                "recommendations": recommendations,
                "validation": validation,
                "filters_applied": filters,
                "raw_result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing executive query: {str(e)}")
            return self._create_error_response(str(e))
    
    def _is_human_error_question(self, question: str) -> bool:
        """Determine if question is about human errors"""
        q = question.lower()
        return any(kw in q for kw in [
            "human error", "human errors", "hea", "error avoidance", 
            "root cause type", "problem tkt trigger", "problem trigger"
        ])
    
    def _generate_executive_response(self, query: str, result: Dict, 
                                   filtered_df: pd.DataFrame, filters: Dict, 
                                   db_type: str) -> str:
        """Generate executive-level response using AI or structured analysis"""
        
        if self.use_openai:
            return self._generate_ai_response(query, result, filtered_df, filters, db_type)
        else:
            return self._generate_structured_response(query, result, filtered_df, filters, db_type)
    
    def _generate_ai_response(self, query: str, result: Dict, 
                            filtered_df: pd.DataFrame, filters: Dict, 
                            db_type: str) -> str:
        """Generate response using OpenAI GPT"""
        
        # Prepare context for AI
        context = self._prepare_ai_context(result, filtered_df, filters, db_type)
        
        prompt = f"""You are an executive analyst for Dell Technologies, providing insights to the CEO and Board of Directors.

CONTEXT:
- Database: {db_type}
- Dell Fiscal Year: Feb-Jan cycle
- Analysis Focus: Last 3 fiscal years unless specified
- Target Audience: C-suite executives

QUERY: {query}

DATA ANALYSIS:
{context}

INSTRUCTIONS:
1. Provide a clear executive summary (2-3 sentences)
2. Include specific metrics and financial impacts
3. Highlight key trends and patterns
4. Focus on operational efficiency and error prevention
5. Use board-level language and format
6. Include confidence indicators

RESPONSE FORMAT:
- Executive Summary
- Key Findings (with metrics)
- Trend Analysis
- Strategic Implications

Generate executive response:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a strategic analyst for Dell's executive team."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return self._generate_structured_response(query, result, filtered_df, filters, db_type)
    
    def _generate_structured_response(self, query: str, result: Dict, 
                                    filtered_df: pd.DataFrame, filters: Dict, 
                                    db_type: str) -> str:
        """Generate structured response without AI"""
        
        if result is None:
            return "No relevant data found for the specified query parameters."
        
        response_parts = []
        
        # Executive Summary
        if result.get("result_type") == "summary":
            total = result.get("value", {}).get("total", 0)
            response_parts.append(f"**EXECUTIVE SUMMARY**")
            response_parts.append(f"Analysis of {total} operational records from {db_type} database reveals key patterns in organizational performance.")
        
        elif result.get("result_type") == "company_counts":
            total = result.get("total", 0)
            companies = len(result.get("value", {}))
            response_parts.append(f"**EXECUTIVE SUMMARY**")
            response_parts.append(f"Cross-company analysis shows {total} operational events across {companies} business units, with detailed quarterly breakdown available.")
        
        else:
            count = result.get("value", 0) if result.get("result_type") == "count" else len(filtered_df) if filtered_df is not None else 0
            response_parts.append(f"**EXECUTIVE SUMMARY**")
            response_parts.append(f"Targeted analysis identified {count} operational records matching specified criteria.")
        
        # Key Findings
        response_parts.append(f"\n**KEY FINDINGS**")
        
        if filtered_df is not None and not filtered_df.empty:
            # Financial impact analysis
            if self._has_financial_data(filtered_df):
                financial_insights = self._analyze_financial_impact(filtered_df)
                response_parts.extend(financial_insights)
            
            # Department analysis
            dept_analysis = self._analyze_departments(filtered_df)
            response_parts.extend(dept_analysis)
            
            # Temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(filtered_df, filters)
            response_parts.extend(temporal_analysis)
        
        # Strategic Implications
        response_parts.append(f"\n**STRATEGIC IMPLICATIONS**")
        
        if result.get("result_type") == "summary" and result.get("value", {}).get("problem_trigger_breakdown"):
            triggers = result["value"]["problem_trigger_breakdown"]
            top_trigger = max(triggers.keys(), key=lambda k: triggers[k].get("total", 0)) if triggers else None
            if top_trigger:
                response_parts.append(f"• Primary operational focus area: {top_trigger}")
        
        response_parts.append("• Recommend continued quarterly operational reviews aligned with Dell's fiscal calendar")
        response_parts.append("• Consider implementing predictive analytics for early issue detection")
        
        return "\n".join(response_parts)
    
    def _prepare_ai_context(self, result: Dict, filtered_df: pd.DataFrame, 
                          filters: Dict, db_type: str) -> str:
        """Prepare context for AI analysis"""
        context_parts = []
        
        # Data summary
        if filtered_df is not None:
            context_parts.append(f"Records analyzed: {len(filtered_df)}")
            
            if 'fy' in filtered_df.columns:
                fys = sorted(filtered_df['fy'].dropna().unique())
                context_parts.append(f"Fiscal years covered: {', '.join(map(str, fys))}")
        
        # Result summary
        if result:
            context_parts.append(f"Result type: {result.get('result_type', 'Unknown')}")
            
            if result.get("result_type") == "summary":
                summary = result.get("value", {})
                context_parts.append(f"Total incidents: {summary.get('total', 0)}")
                
                if summary.get("problem_trigger_breakdown"):
                    triggers = summary["problem_trigger_breakdown"]
                    context_parts.append("Trigger breakdown:")
                    for trigger, info in list(triggers.items())[:3]:  # Top 3
                        context_parts.append(f"  - {trigger}: {info.get('total', 0)}")
        
        # Filter context
        active_filters = [f"{k}: {v}" for k, v in filters.items() if v and k != "intent"]
        if active_filters:
            context_parts.append(f"Filters applied: {', '.join(active_filters)}")
        
        return "\n".join(context_parts)
    
    def _has_financial_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains financial impact data"""
        financial_columns = ['financial_impact', 'cost_impact', 'savings', 'revenue_impact']
        return any(col in df.columns for col in financial_columns)
    
    def _analyze_financial_impact(self, df: pd.DataFrame) -> List[str]:
        """Analyze financial impact from the data"""
        insights = []
        
        # Look for financial data in various columns
        for col in ['financial_impact', 'cost_impact', 'savings', 'revenue_impact']:
            if col in df.columns:
                financial_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if not financial_data.empty:
                    total_impact = financial_data.sum()
                    avg_impact = financial_data.mean()
                    insights.append(f"• Total {col.replace('_', ' ')}: ${total_impact:,.0f}")
                    insights.append(f"• Average {col.replace('_', ' ')}: ${avg_impact:,.0f}")
                    break
        
        return insights
    
    def _analyze_departments(self, df: pd.DataFrame) -> List[str]:
        """Analyze department-level patterns"""
        insights = []
        
        dept_columns = ['department', 'company_display_value', 'company']
        for col in dept_columns:
            if col in df.columns:
                dept_counts = df[col].value_counts()
                if not dept_counts.empty:
                    top_dept = dept_counts.index[0]
                    top_count = dept_counts.iloc[0]
                    insights.append(f"• Most active unit: {top_dept} ({top_count} records)")
                    
                    if len(dept_counts) > 1:
                        insights.append(f"• Total units involved: {len(dept_counts)}")
                    break
        
        return insights
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, filters: Dict) -> List[str]:
        """Analyze temporal patterns in the data"""
        insights = []
        
        if 'fy' in df.columns:
            fy_counts = df['fy'].value_counts().sort_index()
            if len(fy_counts) > 1:
                trend = "increasing" if fy_counts.iloc[-1] > fy_counts.iloc[0] else "decreasing"
                insights.append(f"• Trend analysis: {trend} pattern over time")
        
        if 'quarter' in df.columns:
            quarter_counts = df['quarter'].value_counts().sort_index()
            if not quarter_counts.empty:
                peak_quarter = quarter_counts.idxmax()
                insights.append(f"• Peak activity: Q{peak_quarter} ({quarter_counts.max()} records)")
        
        return insights
    
    def _calculate_confidence(self, filtered_df: pd.DataFrame, result: Dict, 
                            filters: Dict) -> float:
        """Calculate confidence score for the analysis"""
        confidence_factors = []
        
        # Data availability factor
        if filtered_df is not None:
            data_count = len(filtered_df)
            data_factor = min(data_count / 10, 1.0)  # Normalize to max 1.0
            confidence_factors.append(data_factor)
        else:
            confidence_factors.append(0.0)
        
        # Filter specificity factor
        specific_filters = sum(1 for k, v in filters.items() if v and k != "intent")
        specificity_factor = min(specific_filters / 3, 1.0)  # Normalize to max 1.0
        confidence_factors.append(specificity_factor)
        
        # Result completeness factor
        if result and result.get("result_type"):
            completeness_factor = 0.8  # Base confidence for having results
            if result.get("value"):
                completeness_factor = 1.0
        else:
            completeness_factor = 0.0
        confidence_factors.append(completeness_factor)
        
        # Calculate weighted average
        weights = [0.5, 0.3, 0.2]  # Prioritize data availability
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return round(confidence, 2)
    
    def _extract_executive_insights(self, filtered_df: pd.DataFrame, result: Dict, 
                                  filters: Dict, db_type: str) -> List[str]:
        """Extract executive-level insights from the analysis"""
        insights = []
        
        # Database-specific insights
        insights.append(f"Analysis performed on {db_type} database")
        
        if filtered_df is not None and not filtered_df.empty:
            insights.append(f"Data points analyzed: {len(filtered_df)}")
            
            # Fiscal year insights
            if 'fy' in filtered_df.columns:
                fys = sorted(filtered_df['fy'].dropna().unique())
                if fys:
                    fy_range = f"FY{min(fys)} to FY{max(fys)}" if len(fys) > 1 else f"FY{fys[0]}"
                    insights.append(f"Timeframe coverage: {fy_range}")
        
        # Result-specific insights
        if result and result.get("result_type") == "summary":
            summary = result.get("value", {})
            if summary.get("problem_trigger_breakdown"):
                trigger_count = len(summary["problem_trigger_breakdown"])
                insights.append(f"Operational categories identified: {trigger_count}")
        
        return insights
    
    def _generate_strategic_recommendations(self, filtered_df: pd.DataFrame, 
                                          result: Dict, filters: Dict, 
                                          db_type: str) -> List[str]:
        """Generate strategic recommendations for executives"""
        recommendations = []
        
        # Dell-specific recommendations
        recommendations.append("Maintain quarterly operational reviews aligned with Dell's Feb-Jan fiscal calendar")
        
        # Database-specific recommendations
        if db_type == "Human Error":
            recommendations.append("Implement focused training programs to address recurring error patterns")
            recommendations.append("Establish predictive analytics for early error detection and prevention")
        else:
            recommendations.append("Enhance process monitoring for operational efficiency improvements")
        
        # Data-driven recommendations
        if filtered_df is not None and not filtered_df.empty:
            if 'company_display_value' in filtered_df.columns:
                companies = filtered_df['company_display_value'].nunique()
                if companies > 1:
                    recommendations.append(f"Consider cross-company best practice sharing ({companies} units involved)")
            
            if 'fy' in filtered_df.columns:
                recent_fy = filtered_df['fy'].max()
                if recent_fy:
                    recommendations.append(f"Focus immediate attention on FY{int(recent_fy)} operational outcomes")
        
        return recommendations
    
    def _validate_executive_response(self, response: str, filtered_df: pd.DataFrame, 
                                   result: Dict) -> Dict[str, Any]:
        """Validate the executive response for accuracy and completeness"""
        validation = {}
        
        # Response length validation
        validation["adequate_length"] = len(response.split()) >= 50
        
        # Data alignment validation
        validation["data_referenced"] = filtered_df is not None and not filtered_df.empty
        
        # Metrics inclusion validation
        validation["metrics_included"] = bool(re.search(r'\d+', response))
        
        # Executive format validation
        validation["executive_format"] = any(keyword in response.upper() for keyword in [
            "EXECUTIVE", "SUMMARY", "KEY", "STRATEGIC", "RECOMMENDATION"
        ])
        
        # Calculate overall validation score
        validation["overall_score"] = sum(validation.values()) / len(validation)
        
        return validation
    
    def _get_fiscal_year_context(self, filters: Dict) -> str:
        """Get fiscal year context for the analysis"""
        if filters.get("fy_year"):
            return f"Analysis focused on FY{filters['fy_year']}"
        
        # Default to current context
        current_fy = self._get_current_dell_fy()
        return f"Analysis covers recent fiscal years (current: FY{current_fy})"
    
    def _get_current_dell_fy(self) -> int:
        """Get current Dell fiscal year"""
        now = datetime.now()
        return now.year + 1 if now.month >= 2 else now.year
    
    def _create_no_data_response(self, query: str, db_type: str) -> Dict[str, Any]:
        """Create response when no data is available"""
        return {
            "response": f"No relevant data found in {db_type} database for query: '{query}'. Please verify data availability and query parameters.",
            "confidence_score": 0.0,
            "data_sources": 0,
            "database_type": db_type,
            "fiscal_year_context": "Unable to determine",
            "key_insights": [],
            "recommendations": ["Verify data connectivity", "Check query parameters"],
            "validation": {"data_available": False},
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create response for system errors"""
        return {
            "response": f"System error during analysis: {error_message}. Please try again or contact support.",
            "confidence_score": 0.0,
            "data_sources": 0,
            "database_type": "Unknown",
            "fiscal_year_context": "Unable to determine",
            "key_insights": [],
            "recommendations": ["Check system configuration", "Verify database connectivity"],
            "validation": {"system_error": True},
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "main_db_loaded": self.main_df is not None,
            "he_db_loaded": self.he_df is not None,
            "main_db_records": len(self.main_df) if self.main_df is not None else 0,
            "he_db_records": len(self.he_df) if self.he_df is not None else 0,
            "openai_enabled": self.use_openai,
            "current_dell_fy": self._get_current_dell_fy(),
            "system_ready": (self.main_df is not None) or (self.he_df is not None)
        }