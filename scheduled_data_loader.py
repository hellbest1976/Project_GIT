"""
Scheduled Data Loader for Dell Organizational Operations Chatbot
Loads Excel data into ChromaDB daily at 4:00 AM IST and manages automatic learning
"""

import schedule
import time
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import pytz
import json
import glob
from typing import List, Dict, Any
import threading
from pathlib import Path

# Import existing modules
try:
    import he_query_executor
    from adaptive_learning_system import AdaptiveLearningSystem
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")

class ScheduledDataLoader:
    """Manages scheduled data loading and automatic learning from interactions"""
    
    def __init__(self, 
                 excel_folder: str = "./data/excel_imports",
                 backup_folder: str = "./data/backups",
                 chroma_path: str = None):
        
        self.excel_folder = excel_folder
        self.backup_folder = backup_folder
        self.chroma_path = chroma_path or he_query_executor.CHROMA_PATH
        
        # Setup directories
        os.makedirs(excel_folder, exist_ok=True)
        os.makedirs(backup_folder, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./data/scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize learning system
        try:
            self.learning_system = AdaptiveLearningSystem()
        except Exception as e:
            self.logger.error(f"Failed to initialize learning system: {e}")
            self.learning_system = None
        
        # Chat interaction storage
        self.chat_interactions = []
        self.interaction_file = "./data/chat_interactions.json"
        self.load_existing_interactions()
        
        # IST timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # ChromaDB setup
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.logger.info("Scheduled Data Loader initialized")
    
    def start_scheduler(self):
        """Start the scheduled data loading system"""
        # Schedule daily data loading at 4:00 AM IST
        schedule.every().day.at("04:00").do(self.daily_data_load_job)
        
        # Schedule chat learning every hour
        schedule.every().hour.do(self.process_chat_interactions)
        
        # Schedule weekly backup
        schedule.every().sunday.at("02:00").do(self.weekly_backup_job)
        
        self.logger.info("Scheduler started - Daily data loading at 4:00 AM IST")
        
        # Run scheduler in background thread
        def run_scheduler():
            while True:
                # Get current time in IST
                current_ist = datetime.now(self.ist)
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        return scheduler_thread
    
    def daily_data_load_job(self):
        """Daily job to load Excel data into ChromaDB"""
        self.logger.info("Starting daily data loading job...")
        
        try:
            # Get current IST time
            current_ist = datetime.now(self.ist)
            self.logger.info(f"Daily load starting at {current_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")
            
            # Find Excel files
            excel_files = self.find_excel_files()
            
            if not excel_files:
                self.logger.warning("No Excel files found for loading")
                return
            
            # Backup existing ChromaDB
            self.backup_chroma_db()
            
            # Load each Excel file
            total_loaded = 0
            for excel_file in excel_files:
                loaded_count = self.load_excel_to_chroma(excel_file)
                total_loaded += loaded_count
                
                # Move processed file to backup
                self.archive_processed_file(excel_file)
            
            self.logger.info(f"Daily data loading completed. Total records loaded: {total_loaded}")
            
            # Process any pending chat interactions
            self.process_chat_interactions()
            
        except Exception as e:
            self.logger.error(f"Error in daily data loading: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def find_excel_files(self) -> List[str]:
        """Find Excel files to process"""
        patterns = [
            "*.xlsx", "*.xls", "*.xlsm"
        ]
        
        excel_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(self.excel_folder, pattern))
            excel_files.extend(files)
        
        # Sort by modification time (newest first)
        excel_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        self.logger.info(f"Found {len(excel_files)} Excel files to process")
        return excel_files
    
    def load_excel_to_chroma(self, excel_file: str) -> int:
        """Load Excel file data into ChromaDB"""
        try:
            self.logger.info(f"Loading Excel file: {excel_file}")
            
            # Read Excel file
            df = pd.read_excel(excel_file)
            self.logger.info(f"Read {len(df)} rows from {excel_file}")
            
            # Initialize ChromaDB connection
            client = chromadb.PersistentClient(path=self.chroma_path)
            collection = client.get_or_create_collection(
                name=he_query_executor.COLLECTION_NAME,
                metadata={"description": "Human error records loaded from Excel"}
            )
            
            # Process data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                # Create document text from all columns
                doc_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                documents.append(doc_text)
                
                # Create metadata
                metadata = {}
                for col in df.columns:
                    if pd.notna(row[col]):
                        metadata[str(col).lower().replace(' ', '_')] = str(row[col])
                
                # Add loading metadata
                metadata['source'] = 'excel_import'
                metadata['file_name'] = os.path.basename(excel_file)
                metadata['loaded_timestamp'] = datetime.now().isoformat()
                
                metadatas.append(metadata)
                
                # Create unique ID
                file_id = os.path.basename(excel_file).replace('.', '_')
                ids.append(f"excel_{file_id}_{idx}")
            
            # Add to ChromaDB in batches
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                total_added += len(batch_docs)
                self.logger.info(f"Added batch {i//batch_size + 1}, total records: {total_added}")
            
            self.logger.info(f"Successfully loaded {total_added} records from {excel_file}")
            return total_added
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file {excel_file}: {e}")
            return 0
    
    def backup_chroma_db(self):
        """Backup existing ChromaDB before loading new data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_folder, f"chroma_backup_{timestamp}")
            
            # Copy ChromaDB directory
            import shutil
            if os.path.exists(self.chroma_path):
                shutil.copytree(self.chroma_path, backup_path)
                self.logger.info(f"ChromaDB backed up to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error backing up ChromaDB: {e}")
    
    def archive_processed_file(self, excel_file: str):
        """Move processed Excel file to archive"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(excel_file)
            name, ext = os.path.splitext(filename)
            
            archived_name = f"{name}_processed_{timestamp}{ext}"
            archived_path = os.path.join(self.backup_folder, archived_name)
            
            import shutil
            shutil.move(excel_file, archived_path)
            self.logger.info(f"Archived processed file to {archived_path}")
            
        except Exception as e:
            self.logger.error(f"Error archiving file {excel_file}: {e}")
    
    def weekly_backup_job(self):
        """Weekly backup job"""
        try:
            self.logger.info("Starting weekly backup job...")
            
            # Backup learned knowledge
            if self.learning_system:
                knowledge_backup = self.learning_system.export_learned_knowledge()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(self.backup_folder, f"learned_knowledge_backup_{timestamp}.json")
                
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(knowledge_backup)
                
                self.logger.info(f"Learned knowledge backed up to {backup_file}")
            
            # Clean old backups (keep last 30 days)
            self.cleanup_old_backups(days=30)
            
        except Exception as e:
            self.logger.error(f"Error in weekly backup: {e}")
    
    def cleanup_old_backups(self, days: int = 30):
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for file_path in glob.glob(os.path.join(self.backup_folder, "*")):
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        self.logger.info(f"Removed old backup: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning old backups: {e}")
    
    # === CHAT INTERACTION LEARNING ===
    
    def capture_chat_interaction(self, user_query: str, bot_response: str, 
                                metadata: Dict[str, Any] = None):
        """Capture chat interaction for learning"""
        try:
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "bot_response": bot_response,
                "metadata": metadata or {},
                "processed": False
            }
            
            self.chat_interactions.append(interaction)
            
            # Save immediately
            self.save_chat_interactions()
            
            self.logger.info(f"Captured chat interaction: {user_query[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error capturing chat interaction: {e}")
    
    def load_existing_interactions(self):
        """Load existing chat interactions"""
        try:
            if os.path.exists(self.interaction_file):
                with open(self.interaction_file, 'r', encoding='utf-8') as f:
                    self.chat_interactions = json.load(f)
                self.logger.info(f"Loaded {len(self.chat_interactions)} existing interactions")
            else:
                self.chat_interactions = []
        except Exception as e:
            self.logger.error(f"Error loading interactions: {e}")
            self.chat_interactions = []
    
    def save_chat_interactions(self):
        """Save chat interactions to file"""
        try:
            with open(self.interaction_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_interactions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving interactions: {e}")
    
    def process_chat_interactions(self):
        """Process captured chat interactions for learning"""
        try:
            if not self.learning_system:
                return
            
            unprocessed = [i for i in self.chat_interactions if not i.get("processed", False)]
            
            if not unprocessed:
                return
            
            self.logger.info(f"Processing {len(unprocessed)} chat interactions for learning")
            
            for interaction in unprocessed:
                # Create Q&A pair for learning
                qa_pair = {
                    "question": interaction["user_query"],
                    "answer": interaction["bot_response"],
                    "source": "chat_interaction",
                    "timestamp": interaction["timestamp"],
                    "metadata": interaction.get("metadata", {})
                }
                
                # Learn the Q&A pair
                success = self.learning_system.ingest_qa_json(qa_pair)
                
                if success:
                    interaction["processed"] = True
                    interaction["learned_timestamp"] = datetime.now().isoformat()
            
            # Save updated interactions
            self.save_chat_interactions()
            
            processed_count = len([i for i in unprocessed if i.get("processed", False)])
            self.logger.info(f"Successfully processed {processed_count} interactions for learning")
            
        except Exception as e:
            self.logger.error(f"Error processing chat interactions: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        try:
            current_ist = datetime.now(self.ist)
            
            # Count files in excel folder
            excel_files = self.find_excel_files()
            
            # Count unprocessed interactions
            unprocessed_interactions = len([i for i in self.chat_interactions if not i.get("processed", False)])
            
            # Get next scheduled runs
            next_jobs = []
            for job in schedule.jobs:
                next_jobs.append({
                    "job": str(job.job_func.__name__),
                    "next_run": job.next_run.strftime("%Y-%m-%d %H:%M:%S") if job.next_run else "Not scheduled"
                })
            
            return {
                "current_ist_time": current_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                "excel_files_pending": len(excel_files),
                "chat_interactions_total": len(self.chat_interactions),
                "unprocessed_interactions": unprocessed_interactions,
                "learning_system_active": self.learning_system is not None,
                "next_scheduled_jobs": next_jobs,
                "excel_folder": self.excel_folder,
                "backup_folder": self.backup_folder
            }
            
        except Exception as e:
            self.logger.error(f"Error getting scheduler status: {e}")
            return {"error": str(e)}
    
    def manual_data_load(self) -> Dict[str, Any]:
        """Manually trigger data loading"""
        try:
            self.logger.info("Manual data loading triggered")
            result = self.daily_data_load_job()
            return {"success": True, "message": "Manual data loading completed"}
        except Exception as e:
            self.logger.error(f"Manual data loading failed: {e}")
            return {"success": False, "error": str(e)}

# Global scheduler instance
_scheduler_instance = None

def get_scheduler_instance(excel_folder: str = "./data/excel_imports") -> ScheduledDataLoader:
    """Get or create scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ScheduledDataLoader(excel_folder=excel_folder)
    return _scheduler_instance

def start_scheduled_system(excel_folder: str = "./data/excel_imports"):
    """Start the scheduled data loading system"""
    scheduler = get_scheduler_instance(excel_folder)
    scheduler_thread = scheduler.start_scheduler()
    return scheduler, scheduler_thread

# Usage example and setup instructions
if __name__ == "__main__":
    print("Dell Organizational Operations Chatbot - Scheduled Data Loader")
    print("=" * 60)
    
    # Setup instructions
    setup_instructions = """
    SETUP INSTRUCTIONS:
    
    1. Excel File Setup:
       - Place Excel files in ./data/excel_imports/ folder
       - Files will be automatically loaded at 4:00 AM IST daily
       - Supported formats: .xlsx, .xls, .xlsm
    
    2. Required Excel Columns (suggested):
       - problem_number, opened, company_display_value
       - problem_trigger, root_cause_type, resource_region
       - fiscal_year, quarter, etc.
    
    3. Chat Learning:
       - All chat interactions are automatically captured
       - Q&A pairs are learned hourly
       - Manual learning also available via API
    
    4. Backups:
       - Daily ChromaDB backups before loading
       - Weekly learned knowledge backups
       - Old backups cleaned after 30 days
    
    5. Monitoring:
       - Check ./data/scheduler.log for logs
       - Use get_scheduler_status() for current status
    """
    
    print(setup_instructions)
    
    # Start scheduler
    scheduler, thread = start_scheduled_system()
    
    print(f"\n✅ Scheduler started successfully!")
    print(f"📁 Excel folder: {scheduler.excel_folder}")
    print(f"💾 Backup folder: {scheduler.backup_folder}")
    print(f"🕐 Next data load: Daily at 4:00 AM IST")
    print(f"🧠 Learning system: {'✅ Active' if scheduler.learning_system else '❌ Inactive'}")
    
    # Keep running
    try:
        while True:
            time.sleep(10)
            status = scheduler.get_scheduler_status()
            print(f"\r📊 Status: {status['current_ist_time']} | Excel files: {status['excel_files_pending']} | Interactions: {status['unprocessed_interactions']}", end="")
    except KeyboardInterrupt:
        print("\n\n👋 Scheduler stopped by user")