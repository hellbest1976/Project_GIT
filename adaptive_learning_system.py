"""
Adaptive Learning System for Dell Organizational Operations Chatbot
Learns from Q&A pairs in JSON format and continuously improves responses
"""

import json
import os
import pandas as pd
import chromadb
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import hashlib
import pickle

class AdaptiveLearningSystem:
    """System that learns from Q&A pairs and improves chatbot responses"""
    
    def __init__(self, knowledge_db_path: str = "./data/adaptive_knowledge"):
        self.knowledge_db_path = knowledge_db_path
        self.knowledge_file = os.path.join(knowledge_db_path, "learned_knowledge.json")
        self.metadata_file = os.path.join(knowledge_db_path, "learning_metadata.json")
        self.chroma_path = os.path.join(knowledge_db_path, "chroma_learned")
        
        # Setup directories
        os.makedirs(knowledge_db_path, exist_ok=True)
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB for learned knowledge
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.learned_collection = self.client.get_or_create_collection(
            name="learned_qa",
            metadata={"description": "Learned Q&A pairs for adaptive responses"}
        )
        
        # Load existing knowledge
        self.learned_qa_pairs = []
        self.load_existing_knowledge()
        
        self.logger.info(f"Adaptive Learning System initialized with {len(self.learned_qa_pairs)} learned Q&A pairs")
    
    def load_existing_knowledge(self):
        """Load existing learned knowledge from storage"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    self.learned_qa_pairs = json.load(f)
                self.logger.info(f"Loaded {len(self.learned_qa_pairs)} existing Q&A pairs")
            else:
                self.learned_qa_pairs = []
                self.logger.info("No existing knowledge file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading existing knowledge: {e}")
            self.learned_qa_pairs = []
    
    def ingest_qa_json(self, json_data: str | dict | list) -> bool:
        """
        Ingest Q&A pairs from JSON format
        
        Expected formats:
        1. Single Q&A: {"question": "...", "answer": "...", "metadata": {...}}
        2. List of Q&As: [{"question": "...", "answer": "..."}, ...]
        3. Structured format: {"qa_pairs": [...], "session_info": {...}}
        
        Args:
            json_data: JSON string, dict, or list containing Q&A pairs
            
        Returns:
            bool: Success status
        """
        try:
            # Parse JSON if string
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            new_pairs = []
            
            # Handle different formats
            if isinstance(data, dict):
                if "qa_pairs" in data:
                    # Structured format
                    new_pairs = data["qa_pairs"]
                    session_info = data.get("session_info", {})
                elif "question" in data and "answer" in data:
                    # Single Q&A
                    new_pairs = [data]
                else:
                    self.logger.warning("Unrecognized dict format for Q&A data")
                    return False
                    
            elif isinstance(data, list):
                # List of Q&As
                new_pairs = data
            else:
                self.logger.error("Invalid data format for Q&A ingestion")
                return False
            
            # Process and validate Q&A pairs
            valid_pairs = []
            for pair in new_pairs:
                if self._validate_qa_pair(pair):
                    enriched_pair = self._enrich_qa_pair(pair)
                    valid_pairs.append(enriched_pair)
                else:
                    self.logger.warning(f"Invalid Q&A pair skipped: {pair}")
            
            if valid_pairs:
                # Add to learned knowledge
                self.learned_qa_pairs.extend(valid_pairs)
                
                # Save to persistent storage
                self._save_knowledge()
                
                # Add to ChromaDB for similarity search
                self._add_to_chroma(valid_pairs)
                
                self.logger.info(f"Successfully ingested {len(valid_pairs)} new Q&A pairs")
                return True
            else:
                self.logger.warning("No valid Q&A pairs found in provided data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error ingesting Q&A JSON: {e}")
            return False
    
    def _validate_qa_pair(self, pair: dict) -> bool:
        """Validate a Q&A pair"""
        required_fields = ["question", "answer"]
        
        if not isinstance(pair, dict):
            return False
        
        for field in required_fields:
            if field not in pair or not pair[field] or not isinstance(pair[field], str):
                return False
        
        # Check for minimum content length
        if len(pair["question"].strip()) < 5 or len(pair["answer"].strip()) < 10:
            return False
        
        return True
    
    def _enrich_qa_pair(self, pair: dict) -> dict:
        """Enrich Q&A pair with metadata and processing info"""
        enriched = pair.copy()
        
        # Add learning metadata
        enriched["learned_timestamp"] = datetime.now().isoformat()
        enriched["question_hash"] = hashlib.md5(pair["question"].encode()).hexdigest()
        enriched["answer_length"] = len(pair["answer"])
        enriched["question_length"] = len(pair["question"])
        
        # Add Dell-specific categorization
        enriched["categories"] = self._categorize_qa_pair(pair)
        
        # Add confidence and usage tracking
        enriched["usage_count"] = 0
        enriched["confidence_score"] = 1.0  # Start with high confidence
        enriched["source"] = pair.get("source", "user_provided")
        
        # Extract Dell fiscal year if mentioned
        enriched["fiscal_years"] = self._extract_fiscal_years(pair["question"] + " " + pair["answer"])
        
        return enriched
    
    def _categorize_qa_pair(self, pair: dict) -> List[str]:
        """Categorize Q&A pair for better organization"""
        categories = []
        
        question_lower = pair["question"].lower()
        answer_lower = pair["answer"].lower()
        combined = question_lower + " " + answer_lower
        
        # Human error related
        if any(term in combined for term in ["human error", "hea", "error avoidance", "mistake"]):
            categories.append("human_error")
        
        # Operational categories
        if any(term in combined for term in ["operational", "process", "procedure", "workflow"]):
            categories.append("operations")
        
        # Executive/strategic
        if any(term in combined for term in ["strategic", "executive", "board", "ceo", "leadership"]):
            categories.append("executive")
        
        # Financial
        if any(term in combined for term in ["financial", "cost", "savings", "revenue", "budget"]):
            categories.append("financial")
        
        # Dell-specific
        if any(term in combined for term in ["dell", "fiscal year", "fy20", "fy21", "fy22", "fy23", "fy24", "fy25"]):
            categories.append("dell_specific")
        
        # Trend analysis
        if any(term in combined for term in ["trend", "pattern", "analysis", "insight"]):
            categories.append("analysis")
        
        return categories if categories else ["general"]
    
    def _extract_fiscal_years(self, text: str) -> List[str]:
        """Extract Dell fiscal years mentioned in text"""
        import re
        
        fy_pattern = r"FY\s*(\d{4})|fy\s*(\d{4})|fiscal\s+year\s+(\d{4})"
        matches = re.findall(fy_pattern, text, re.IGNORECASE)
        
        fiscal_years = []
        for match in matches:
            for group in match:
                if group:
                    fiscal_years.append(f"FY{group}")
        
        return list(set(fiscal_years))
    
    def _save_knowledge(self):
        """Save learned knowledge to persistent storage"""
        try:
            # Save Q&A pairs
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_qa_pairs, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata = {
                "total_pairs": len(self.learned_qa_pairs),
                "last_updated": datetime.now().isoformat(),
                "categories": self._get_category_stats(),
                "fiscal_years": self._get_fy_stats()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("Knowledge saved to persistent storage")
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge: {e}")
    
    def _add_to_chroma(self, qa_pairs: List[dict]):
        """Add Q&A pairs to ChromaDB for similarity search"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for pair in qa_pairs:
                # Use question as document for similarity search
                documents.append(pair["question"])
                
                # Create metadata
                metadata = {
                    "answer": pair["answer"],
                    "learned_timestamp": pair["learned_timestamp"],
                    "categories": ",".join(pair["categories"]),
                    "fiscal_years": ",".join(pair.get("fiscal_years", [])),
                    "source": pair.get("source", "unknown"),
                    "confidence_score": pair.get("confidence_score", 1.0)
                }
                
                metadatas.append(metadata)
                ids.append(pair["question_hash"])
            
            # Add to collection
            self.learned_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(qa_pairs)} Q&A pairs to ChromaDB")
            
        except Exception as e:
            self.logger.error(f"Error adding to ChromaDB: {e}")
    
    def search_learned_knowledge(self, question: str, n_results: int = 5, 
                                confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search learned knowledge for relevant Q&A pairs
        
        Args:
            question: User question to search for
            n_results: Maximum number of results
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of relevant Q&A pairs with confidence scores
        """
        try:
            results = self.learned_collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            relevant_pairs = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0] if results.get("distances") else [0] * len(results["documents"][0])
                )):
                    # Convert distance to confidence (lower distance = higher confidence)
                    confidence = max(0, 1 - distance)
                    
                    if confidence >= confidence_threshold:
                        relevant_pairs.append({
                            "question": doc,
                            "answer": metadata["answer"],
                            "confidence": confidence,
                            "categories": metadata.get("categories", "").split(","),
                            "fiscal_years": metadata.get("fiscal_years", "").split(","),
                            "source": metadata.get("source", "unknown"),
                            "learned_timestamp": metadata.get("learned_timestamp")
                        })
            
            # Sort by confidence
            relevant_pairs.sort(key=lambda x: x["confidence"], reverse=True)
            
            return relevant_pairs
            
        except Exception as e:
            self.logger.error(f"Error searching learned knowledge: {e}")
            return []
    
    def get_learned_answer(self, question: str, confidence_threshold: float = 0.8) -> Optional[str]:
        """
        Get a learned answer for a specific question if confidence is high enough
        
        Args:
            question: User question
            confidence_threshold: Minimum confidence for returning learned answer
            
        Returns:
            Learned answer if found with high confidence, None otherwise
        """
        relevant_pairs = self.search_learned_knowledge(question, n_results=1, 
                                                      confidence_threshold=confidence_threshold)
        
        if relevant_pairs and relevant_pairs[0]["confidence"] >= confidence_threshold:
            # Update usage count
            self._update_usage_count(relevant_pairs[0]["question"])
            
            learned_answer = relevant_pairs[0]["answer"]
            confidence = relevant_pairs[0]["confidence"]
            
            # Add confidence indicator to the answer
            answer_with_confidence = f"{learned_answer}\n\n*[Learned Response - Confidence: {confidence:.0%}]*"
            
            self.logger.info(f"Returning learned answer with {confidence:.0%} confidence")
            return answer_with_confidence
        
        return None
    
    def _update_usage_count(self, question: str):
        """Update usage count for a learned Q&A pair"""
        try:
            question_hash = hashlib.md5(question.encode()).hexdigest()
            
            for pair in self.learned_qa_pairs:
                if pair.get("question_hash") == question_hash:
                    pair["usage_count"] = pair.get("usage_count", 0) + 1
                    pair["last_used"] = datetime.now().isoformat()
                    break
            
            # Save updated knowledge
            self._save_knowledge()
            
        except Exception as e:
            self.logger.error(f"Error updating usage count: {e}")
    
    def _get_category_stats(self) -> Dict[str, int]:
        """Get statistics about learned categories"""
        category_counts = {}
        
        for pair in self.learned_qa_pairs:
            for category in pair.get("categories", ["general"]):
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts
    
    def _get_fy_stats(self) -> Dict[str, int]:
        """Get statistics about fiscal years in learned knowledge"""
        fy_counts = {}
        
        for pair in self.learned_qa_pairs:
            for fy in pair.get("fiscal_years", []):
                if fy:  # Skip empty strings
                    fy_counts[fy] = fy_counts.get(fy, 0) + 1
        
        return fy_counts
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            "total_learned_pairs": len(self.learned_qa_pairs),
            "categories": self._get_category_stats(),
            "fiscal_years": self._get_fy_stats(),
            "recent_pairs": len([p for p in self.learned_qa_pairs 
                               if (datetime.now() - datetime.fromisoformat(p["learned_timestamp"])).days < 7]),
            "most_used": sorted([p for p in self.learned_qa_pairs if p.get("usage_count", 0) > 0],
                              key=lambda x: x.get("usage_count", 0), reverse=True)[:5],
            "last_updated": max([p["learned_timestamp"] for p in self.learned_qa_pairs], default="Never")
        }
    
    def export_learned_knowledge(self, format: str = "json") -> str:
        """Export learned knowledge in specified format"""
        if format == "json":
            return json.dumps({
                "qa_pairs": self.learned_qa_pairs,
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_pairs": len(self.learned_qa_pairs),
                    "categories": self._get_category_stats()
                }
            }, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_learned_knowledge(self) -> bool:
        """Clear all learned knowledge (use with caution)"""
        try:
            self.learned_qa_pairs = []
            
            # Clear ChromaDB collection
            all_ids = self.learned_collection.get()["ids"]
            if all_ids:
                self.learned_collection.delete(ids=all_ids)
            
            # Remove files
            if os.path.exists(self.knowledge_file):
                os.remove(self.knowledge_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            self.logger.info("All learned knowledge cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing learned knowledge: {e}")
            return False

# JSON Format Examples and Documentation
class QAJsonFormats:
    """Documentation and examples for Q&A JSON formats"""
    
    @staticmethod
    def get_single_qa_format() -> dict:
        """Example format for single Q&A pair"""
        return {
            "question": "What were the major human error incidents in Dell FY2024?",
            "answer": "Based on analysis of FY2024 data, the major human error incidents included: 1) Configuration errors in Q1 affecting 15 customers, 2) Process deviations in manufacturing during Q2 resulting in $2M impact, 3) Communication gaps in Q3 leading to delayed deployments. Key recommendations include enhanced training programs and automated validation checks.",
            "metadata": {
                "source": "executive_review",
                "fiscal_year": "FY2024",
                "department": "Operations",
                "confidence": 0.95,
                "tags": ["human_error", "executive", "dell_specific"]
            }
        }
    
    @staticmethod
    def get_multiple_qa_format() -> list:
        """Example format for multiple Q&A pairs"""
        return [
            {
                "question": "How do we prevent human errors in Dell operations?",
                "answer": "Implement these strategies: 1) Automated validation systems, 2) Enhanced training programs, 3) Regular process reviews, 4) Error pattern analysis, 5) Cross-team knowledge sharing.",
                "source": "best_practices"
            },
            {
                "question": "What's the trend in Dell error rates over last 3 FYs?",
                "answer": "Error rates have decreased 25% from FY2022 to FY2024: FY2022: 3.2%, FY2023: 2.8%, FY2024: 2.4%. This improvement is attributed to enhanced training and process automation.",
                "source": "trend_analysis"
            }
        ]
    
    @staticmethod
    def get_structured_format() -> dict:
        """Example format for structured Q&A data"""
        return {
            "session_info": {
                "session_id": "exec_review_2024_q1",
                "timestamp": "2024-01-15T10:00:00Z",
                "source": "board_meeting",
                "participants": ["CEO", "COO", "Board"]
            },
            "qa_pairs": [
                {
                    "question": "What are our error prevention priorities for FY2025?",
                    "answer": "Top priorities: 1) AI-powered error detection, 2) Enhanced automation in critical processes, 3) Cross-functional training programs, 4) Real-time monitoring dashboards.",
                    "priority": "high",
                    "fiscal_year": "FY2025"
                }
            ]
        }
    
    @staticmethod
    def get_format_documentation() -> str:
        """Get complete format documentation"""
        return """
Dell Organizational Operations Chatbot - Q&A JSON Format Documentation

SUPPORTED FORMATS:

1. SINGLE Q&A PAIR:
{
    "question": "Your question here",
    "answer": "Detailed answer here",
    "metadata": {
        "source": "source_identifier",
        "fiscal_year": "FY2024",
        "department": "Operations",
        "confidence": 0.95,
        "tags": ["tag1", "tag2"]
    }
}

2. MULTIPLE Q&A PAIRS:
[
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
]

3. STRUCTURED FORMAT:
{
    "session_info": {
        "session_id": "unique_id",
        "timestamp": "ISO_timestamp",
        "source": "meeting_type"
    },
    "qa_pairs": [
        {"question": "...", "answer": "..."}
    ]
}

REQUIRED FIELDS:
- question: String (minimum 5 characters)
- answer: String (minimum 10 characters)

OPTIONAL FIELDS:
- source: Data source identifier
- fiscal_year: Dell fiscal year (FY2024, etc.)
- department: Relevant department
- confidence: Confidence score (0.0-1.0)
- tags/categories: List of relevant tags
- priority: Priority level (high/medium/low)

AUTOMATIC ENRICHMENT:
The system automatically adds:
- learned_timestamp: When the Q&A was learned
- question_hash: Unique identifier
- categories: Auto-detected categories
- fiscal_years: Extracted fiscal years
- usage_count: How often it's been used
- confidence_score: System confidence
"""