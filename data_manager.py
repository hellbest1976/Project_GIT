"""
Data Management System for Dell Organizational Operations Chatbot
Supports multiple data input methods and ChromaDB integration
"""

import pandas as pd
import sqlite3
import chromadb
import json
import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from config import DellConfig, ChatbotConfig

class DataManager:
    """Manages data ingestion and retrieval for the chatbot"""
    
    def __init__(self):
        # Setup ChromaDB
        os.makedirs(os.path.dirname(ChatbotConfig.CHROMA_DB_PATH), exist_ok=True)
        self.client = chromadb.PersistentClient(path=ChatbotConfig.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=ChatbotConfig.COLLECTION_NAME,
            metadata={"description": "Dell organizational operations data"}
        )
        
        # Setup embedding model
        self.embedding_model = SentenceTransformer(ChatbotConfig.EMBEDDING_MODEL)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_data_from_csv(self, file_path: str) -> bool:
        """Add data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loading CSV with {len(df)} rows")
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                # Create document text
                doc_text = self._create_document_text(row)
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "source": "csv",
                    "file_path": file_path,
                    "row_index": idx,
                    "ingestion_date": datetime.now().isoformat(),
                    **{str(k): str(v) for k, v in row.items() if pd.notna(v)}
                }
                
                metadatas.append(metadata)
                ids.append(f"csv_{os.path.basename(file_path)}_{idx}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding CSV data: {str(e)}")
            return False
    
    def add_data_manually(self, data_entries: List[Dict[str, Any]]) -> bool:
        """Add data manually provided by user"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for idx, entry in enumerate(data_entries):
                if "content" not in entry:
                    continue
                
                documents.append(entry["content"])
                
                metadata = {
                    "source": "manual",
                    "entry_index": idx,
                    "ingestion_date": datetime.now().isoformat(),
                    **{k: str(v) for k, v in entry.items() if k != "content"}
                }
                
                metadatas.append(metadata)
                ids.append(f"manual_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully added {len(documents)} manual entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding manual data: {str(e)}")
            return False
    
    def add_data_from_database(self, connection_string: str, query: str) -> bool:
        """Add data from database query"""
        try:
            conn = sqlite3.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert to manual format and add
            data_entries = []
            for _, row in df.iterrows():
                entry = {"content": " | ".join([f"{k}: {v}" for k, v in row.items()])}
                entry.update(row.to_dict())
                data_entries.append(entry)
            
            return self.add_data_manually(data_entries)
            
        except Exception as e:
            self.logger.error(f"Error adding database data: {str(e)}")
            return False
    
    def search_relevant_data(self, query: str, n_results: int = None) -> List[Dict[str, Any]]:
        """Search for relevant data based on query"""
        if n_results is None:
            n_results = ChatbotConfig.MAX_SEARCH_RESULTS
        
        try:
            # Apply fiscal year filtering if no specific year mentioned
            filtered_query = self._apply_fiscal_year_filter(query)
            
            results = self.collection.query(
                query_texts=[filtered_query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching data: {str(e)}")
            return []
    
    def _create_document_text(self, row: pd.Series) -> str:
        """Create document text from pandas row"""
        text_parts = []
        for col, value in row.items():
            if pd.notna(value):
                text_parts.append(f"{col}: {value}")
        return " | ".join(text_parts)
    
    def _apply_fiscal_year_filter(self, query: str) -> str:
        """Apply default 3-year fiscal year filter if no year mentioned"""
        # Check if query contains year mentions
        year_keywords = ['FY', 'fiscal', 'year', '202', '201']
        has_year = any(keyword in query.lower() for keyword in year_keywords)
        
        if not has_year:
            # Add fiscal year context
            start_fy, end_fy = DellConfig.get_fiscal_year_range()
            fy_context = f" (Focus on Dell fiscal years FY{start_fy} to FY{end_fy})"
            return query + fy_context
        
        return query
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the data collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "last_updated": datetime.now().isoformat(),
                "fiscal_year_range": DellConfig.get_fiscal_year_range()
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def clear_all_data(self) -> bool:
        """Clear all data from collection"""
        try:
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            self.logger.info("All data cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            return False

# Data input templates and examples
class DataTemplates:
    """Provides templates for different data input formats"""
    
    @staticmethod
    def get_csv_template() -> str:
        """CSV template for organizational data"""
        return """date,department,operation_type,description,fiscal_year,impact_level,financial_impact,error_type,status
2024-01-15,Manufacturing,Process Change,Updated assembly line workflow for efficiency,FY2024,High,500000,None,Completed
2024-01-20,IT,System Upgrade,Migrated to new ERP system,FY2024,Critical,2000000,Configuration Error,In Progress
2023-12-10,Sales,Territory Restructure,Reorganized sales territories,FY2024,Medium,150000,Communication Gap,Resolved"""
    
    @staticmethod
    def get_manual_data_example() -> List[Dict[str, Any]]:
        """Example manual data entries"""
        return [
            {
                "content": "Dell FY2023 Q4 supply chain optimization reduced delivery times by 25% and costs by $3M across all regions",
                "department": "Supply Chain",
                "fiscal_year": "FY2023",
                "quarter": "Q4",
                "impact_type": "Efficiency",
                "financial_impact": "3000000",
                "key_metrics": "25% delivery time reduction"
            },
            {
                "content": "IT infrastructure modernization in FY2022 improved system uptime to 99.9% preventing operational disruptions",
                "department": "IT", 
                "fiscal_year": "FY2022",
                "impact_type": "Reliability",
                "uptime_improvement": "99.9%",
                "error_prevention": "System downtime reduction"
            }
        ]