"""
Data Ingestion System for Dell Organizational Operations Chatbot
Supports multiple data input methods and formats
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

class DataIngestionManager:
    """Manages data ingestion from various sources into ChromaDB"""
    
    def __init__(self, chroma_db_path: str = "./data/chroma_db"):
        self.chroma_db_path = chroma_db_path
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name="dell_operations",
            metadata={"description": "Dell organizational operations data"}
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def ingest_from_csv(self, file_path: str, metadata_columns: List[str] = None) -> bool:
        """
        Ingest data from CSV file
        
        Args:
            file_path: Path to CSV file
            metadata_columns: Columns to use as metadata (optional)
        
        Returns:
            bool: Success status
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Process each row
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                # Create document text from all columns
                doc_text = self._create_document_text(row)
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "source": "csv",
                    "file_path": file_path,
                    "row_index": idx,
                    "ingestion_date": datetime.now().isoformat()
                }
                
                # Add specified metadata columns
                if metadata_columns:
                    for col in metadata_columns:
                        if col in row:
                            metadata[col] = str(row[col])
                
                metadatas.append(metadata)
                ids.append(f"csv_{os.path.basename(file_path)}_{idx}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully ingested {len(documents)} documents from CSV")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting CSV: {str(e)}")
            return False
    
    def ingest_from_json(self, file_path: str, text_field: str = "content") -> bool:
        """
        Ingest data from JSON file
        
        Args:
            file_path: Path to JSON file
            text_field: Field containing main text content
        
        Returns:
            bool: Success status
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                records = data
            else:
                records = [data]
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, record in enumerate(records):
                # Extract text content
                if text_field in record:
                    doc_text = str(record[text_field])
                else:
                    doc_text = json.dumps(record)
                
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "source": "json",
                    "file_path": file_path,
                    "record_index": idx,
                    "ingestion_date": datetime.now().isoformat()
                }
                
                # Add all other fields as metadata
                for key, value in record.items():
                    if key != text_field:
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
                ids.append(f"json_{os.path.basename(file_path)}_{idx}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully ingested {len(documents)} documents from JSON")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting JSON: {str(e)}")
            return False
    
    def ingest_from_database(self, connection_string: str, query: str, 
                           text_columns: List[str]) -> bool:
        """
        Ingest data from database query
        
        Args:
            connection_string: Database connection string
            query: SQL query to fetch data
            text_columns: Columns to combine for document text
        
        Returns:
            bool: Success status
        """
        try:
            # Connect to database
            conn = sqlite3.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            self.logger.info(f"Loaded {len(df)} rows from database")
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                # Combine specified text columns
                doc_text = " ".join([str(row[col]) for col in text_columns if col in row])
                documents.append(doc_text)
                
                # Create metadata from all columns
                metadata = {
                    "source": "database",
                    "query": query,
                    "row_index": idx,
                    "ingestion_date": datetime.now().isoformat()
                }
                
                for col, value in row.items():
                    if col not in text_columns:
                        metadata[col] = str(value)
                
                metadatas.append(metadata)
                ids.append(f"db_{idx}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully ingested {len(documents)} documents from database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting from database: {str(e)}")
            return False
    
    def ingest_manual_data(self, data_entries: List[Dict[str, Any]]) -> bool:
        """
        Ingest manually provided data entries
        
        Args:
            data_entries: List of data dictionaries with 'content' and metadata
        
        Returns:
            bool: Success status
        """
        try:
            documents = []
            metadatas = []
            ids = []
            
            for idx, entry in enumerate(data_entries):
                if "content" not in entry:
                    self.logger.warning(f"Entry {idx} missing 'content' field")
                    continue
                
                documents.append(entry["content"])
                
                # Create metadata
                metadata = {
                    "source": "manual",
                    "entry_index": idx,
                    "ingestion_date": datetime.now().isoformat()
                }
                
                # Add provided metadata
                for key, value in entry.items():
                    if key != "content":
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
                ids.append(f"manual_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully ingested {len(documents)} manual entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting manual data: {str(e)}")
            return False
    
    def _create_document_text(self, row: pd.Series) -> str:
        """Create document text from pandas row"""
        text_parts = []
        for col, value in row.items():
            if pd.notna(value):
                text_parts.append(f"{col}: {value}")
        return " | ".join(text_parts)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all data from collection"""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            self.logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            return False

# Example usage and data templates
class DataTemplates:
    """Templates and examples for different data formats"""
    
    @staticmethod
    def get_csv_template() -> str:
        """Return CSV template for organizational data"""
        return """date,department,operation_type,description,impact_level,financial_impact,error_type,resolution_status
2024-01-15,Manufacturing,Process Change,Updated assembly line workflow,High,500000,None,Completed
2024-01-20,IT,System Upgrade,Migrated to new ERP system,Critical,2000000,Configuration Error,In Progress
2023-12-10,Sales,Territory Restructure,Reorganized sales territories,Medium,150000,Communication Gap,Resolved"""
    
    @staticmethod
    def get_json_template() -> Dict[str, Any]:
        """Return JSON template for organizational data"""
        return {
            "operations": [
                {
                    "id": "OP001",
                    "content": "Q3 FY2024 manufacturing process optimization resulted in 15% efficiency gain",
                    "date": "2024-01-15",
                    "department": "Manufacturing",
                    "fiscal_year": "FY2024",
                    "impact_level": "High",
                    "financial_impact": 500000,
                    "metrics": {
                        "efficiency_gain": 0.15,
                        "cost_reduction": 200000
                    }
                }
            ]
        }
    
    @staticmethod
    def get_manual_data_example() -> List[Dict[str, Any]]:
        """Return example manual data entries"""
        return [
            {
                "content": "Dell FY2023 Q4 supply chain optimization reduced delivery times by 25% and costs by $3M",
                "department": "Supply Chain",
                "fiscal_year": "FY2023",
                "quarter": "Q4",
                "impact_type": "Efficiency",
                "financial_impact": 3000000
            },
            {
                "content": "IT infrastructure modernization in FY2022 improved system uptime to 99.9%",
                "department": "IT",
                "fiscal_year": "FY2022",
                "impact_type": "Reliability",
                "uptime_improvement": 0.999
            }
        ]