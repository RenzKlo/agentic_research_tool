"""
Enhanced Memory System for the Agentic AI.
"""

import json
import os
import sqlite3
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.llms.base import BaseLLM
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class EnhancedMemory:
    """Enhanced memory system with multiple storage backends and retrieval methods."""
    
    def __init__(self, 
                 memory_type: str = "conversation_buffer",
                 max_token_limit: int = 2000,
                 db_path: str = "memory.db",
                 llm: Optional[BaseLLM] = None):
        """
        Initialize the enhanced memory system.
        
        Args:
            memory_type: Type of memory ("conversation_buffer", "summary_buffer", "vector", "hybrid")
            max_token_limit: Maximum token limit for memory
            db_path: Path to SQLite database for persistent storage
            llm: Language model for summarization
        """
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.db_path = db_path
        self.llm = llm
        
        # Initialize different memory components
        self.conversation_memory = None
        self.persistent_memory = None
        self.semantic_memory = {}
        self.episodic_memory = []
        
        # Initialize storage
        self._init_memory()
        self._init_database()
    
    def _init_memory(self):
        """Initialize the appropriate memory type."""
        if self.memory_type == "conversation_buffer":
            self.conversation_memory = ConversationBufferWindowMemory(
                k=10,  # Remember last 10 exchanges
                return_messages=True,
                memory_key="chat_history"
            )
        elif self.memory_type == "summary_buffer" and self.llm:
            self.conversation_memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                return_messages=True,
                memory_key="chat_history"
            )
        else:
            # Default to buffer memory
            self.conversation_memory = ConversationBufferWindowMemory(
                k=10,
                return_messages=True,
                memory_key="chat_history"
            )
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    message_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_hash TEXT UNIQUE,
                    fact_content TEXT,
                    source TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    description TEXT,
                    status TEXT,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preference_key TEXT,
                    preference_value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            logger.info(f"Database initialized: {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self.conn = None
            self.cursor = None
    
    def add_message(self, message: BaseMessage, session_id: str = "default") -> None:
        """Add a message to memory."""
        try:
            # Add to conversation memory
            if self.conversation_memory:
                if isinstance(message, HumanMessage):
                    self.conversation_memory.chat_memory.add_user_message(message.content)
                elif isinstance(message, AIMessage):
                    self.conversation_memory.chat_memory.add_ai_message(message.content)
            
            # Add to persistent storage
            if self.cursor:
                message_type = type(message).__name__
                metadata = json.dumps(message.additional_kwargs) if hasattr(message, 'additional_kwargs') else "{}"
                
                self.cursor.execute('''
                    INSERT INTO conversations (session_id, message_type, content, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, message_type, message.content, metadata))
                self.conn.commit()
            
            # Add to episodic memory
            self.episodic_memory.append({
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "message": message,
                "type": type(message).__name__
            })
            
            # Keep episodic memory size manageable
            if len(self.episodic_memory) > 100:
                self.episodic_memory = self.episodic_memory[-100:]
        
        except Exception as e:
            logger.error(f"Error adding message to memory: {str(e)}")
    
    def get_conversation_history(self, session_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history from persistent storage."""
        try:
            if not self.cursor:
                return []
            
            self.cursor.execute('''
                SELECT message_type, content, metadata, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            
            rows = self.cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "type": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "timestamp": row[3]
                })
            
            return list(reversed(history))  # Return in chronological order
        
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def add_fact(self, fact: str, source: str = "user", confidence: float = 1.0) -> None:
        """Add a fact to semantic memory."""
        try:
            # Create hash for deduplication
            fact_hash = hashlib.md5(fact.encode()).hexdigest()
            
            # Add to semantic memory
            self.semantic_memory[fact_hash] = {
                "content": fact,
                "source": source,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to persistent storage
            if self.cursor:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO facts (fact_hash, fact_content, source, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (fact_hash, fact, source, confidence))
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Error adding fact to memory: {str(e)}")
    
    def get_facts(self, query: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get facts from semantic memory."""
        try:
            if not self.cursor:
                return []
            
            if query:
                # Search for facts containing the query
                self.cursor.execute('''
                    SELECT fact_content, source, confidence, timestamp
                    FROM facts
                    WHERE fact_content LIKE ?
                    ORDER BY confidence DESC, timestamp DESC
                    LIMIT ?
                ''', (f"%{query}%", limit))
            else:
                # Get all facts
                self.cursor.execute('''
                    SELECT fact_content, source, confidence, timestamp
                    FROM facts
                    ORDER BY confidence DESC, timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            rows = self.cursor.fetchall()
            
            facts = []
            for row in rows:
                facts.append({
                    "content": row[0],
                    "source": row[1],
                    "confidence": row[2],
                    "timestamp": row[3]
                })
            
            return facts
        
        except Exception as e:
            logger.error(f"Error getting facts: {str(e)}")
            return []
    
    def add_task(self, task_id: str, description: str, status: str = "pending") -> None:
        """Add a task to memory."""
        try:
            if self.cursor:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO tasks (task_id, description, status)
                    VALUES (?, ?, ?)
                ''', (task_id, description, status))
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Error adding task to memory: {str(e)}")
    
    def update_task(self, task_id: str, status: str = None, result: str = None) -> None:
        """Update a task in memory."""
        try:
            if self.cursor:
                if status:
                    self.cursor.execute('''
                        UPDATE tasks SET status = ? WHERE task_id = ?
                    ''', (status, task_id))
                
                if result:
                    self.cursor.execute('''
                        UPDATE tasks SET result = ? WHERE task_id = ?
                    ''', (result, task_id))
                
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Error updating task in memory: {str(e)}")
    
    def get_tasks(self, status: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get tasks from memory."""
        try:
            if not self.cursor:
                return []
            
            if status:
                self.cursor.execute('''
                    SELECT task_id, description, status, result, timestamp
                    FROM tasks
                    WHERE status = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (status, limit))
            else:
                self.cursor.execute('''
                    SELECT task_id, description, status, result, timestamp
                    FROM tasks
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            rows = self.cursor.fetchall()
            
            tasks = []
            for row in rows:
                tasks.append({
                    "task_id": row[0],
                    "description": row[1],
                    "status": row[2],
                    "result": row[3],
                    "timestamp": row[4]
                })
            
            return tasks
        
        except Exception as e:
            logger.error(f"Error getting tasks: {str(e)}")
            return []
    
    def set_preference(self, user_id: str, key: str, value: str) -> None:
        """Set a user preference."""
        try:
            if self.cursor:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO preferences (user_id, preference_key, preference_value)
                    VALUES (?, ?, ?)
                ''', (user_id, key, value))
                self.conn.commit()
        
        except Exception as e:
            logger.error(f"Error setting preference: {str(e)}")
    
    def get_preference(self, user_id: str, key: str) -> Optional[str]:
        """Get a user preference."""
        try:
            if self.cursor:
                self.cursor.execute('''
                    SELECT preference_value FROM preferences
                    WHERE user_id = ? AND preference_key = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (user_id, key))
                
                row = self.cursor.fetchone()
                return row[0] if row else None
        
        except Exception as e:
            logger.error(f"Error getting preference: {str(e)}")
            return None
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for the agent."""
        if self.conversation_memory:
            return self.conversation_memory.load_memory_variables({})
        return {}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to memory."""
        if self.conversation_memory:
            self.conversation_memory.save_context(inputs, outputs)
    
    def clear_memory(self, session_id: str = "default") -> None:
        """Clear memory for a session."""
        try:
            # Clear conversation memory
            if self.conversation_memory:
                self.conversation_memory.clear()
            
            # Clear persistent storage
            if self.cursor:
                self.cursor.execute('''
                    DELETE FROM conversations WHERE session_id = ?
                ''', (session_id,))
                self.conn.commit()
            
            # Clear episodic memory
            self.episodic_memory = [
                episode for episode in self.episodic_memory
                if episode.get("session_id") != session_id
            ]
            
            logger.info(f"Memory cleared for session: {session_id}")
        
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of memory usage."""
        try:
            summary = {
                "memory_type": self.memory_type,
                "max_token_limit": self.max_token_limit,
                "episodic_memory_count": len(self.episodic_memory),
                "semantic_memory_count": len(self.semantic_memory),
                "database_path": self.db_path,
                "database_connected": self.conn is not None
            }
            
            # Get database statistics
            if self.cursor:
                self.cursor.execute("SELECT COUNT(*) FROM conversations")
                summary["total_conversations"] = self.cursor.fetchone()[0]
                
                self.cursor.execute("SELECT COUNT(*) FROM facts")
                summary["total_facts"] = self.cursor.fetchone()[0]
                
                self.cursor.execute("SELECT COUNT(*) FROM tasks")
                summary["total_tasks"] = self.cursor.fetchone()[0]
                
                self.cursor.execute("SELECT COUNT(*) FROM preferences")
                summary["total_preferences"] = self.cursor.fetchone()[0]
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting memory summary: {str(e)}")
            return {}
    
    def export_memory(self, export_path: str, format: str = "json") -> bool:
        """Export memory to file."""
        try:
            memory_data = {
                "conversations": self.get_conversation_history(limit=1000),
                "facts": self.get_facts(limit=1000),
                "tasks": self.get_tasks(limit=1000),
                "episodic_memory": [
                    {
                        "timestamp": episode["timestamp"],
                        "session_id": episode["session_id"],
                        "message_type": episode["type"],
                        "content": episode["message"].content if hasattr(episode["message"], "content") else str(episode["message"])
                    }
                    for episode in self.episodic_memory
                ],
                "semantic_memory": dict(self.semantic_memory),
                "export_timestamp": datetime.now().isoformat()
            }
            
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(memory_data, f, indent=2)
            elif format == "pickle":
                with open(export_path, 'wb') as f:
                    pickle.dump(memory_data, f)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Memory exported to: {export_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting memory: {str(e)}")
            return False
    
    def import_memory(self, import_path: str, format: str = "json") -> bool:
        """Import memory from file."""
        try:
            if format == "json":
                with open(import_path, 'r') as f:
                    memory_data = json.load(f)
            elif format == "pickle":
                with open(import_path, 'rb') as f:
                    memory_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # Import facts
            if "facts" in memory_data:
                for fact in memory_data["facts"]:
                    self.add_fact(
                        fact["content"],
                        fact.get("source", "imported"),
                        fact.get("confidence", 1.0)
                    )
            
            # Import tasks
            if "tasks" in memory_data:
                for task in memory_data["tasks"]:
                    self.add_task(
                        task["task_id"],
                        task["description"],
                        task.get("status", "pending")
                    )
                    if task.get("result"):
                        self.update_task(task["task_id"], result=task["result"])
            
            logger.info(f"Memory imported from: {import_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error importing memory: {str(e)}")
            return False
    
    def search_memory(self, query: str, memory_type: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """Search across different memory types."""
        try:
            results = []
            
            if memory_type in ["all", "facts"]:
                facts = self.get_facts(query, limit)
                for fact in facts:
                    results.append({
                        "type": "fact",
                        "content": fact["content"],
                        "source": fact["source"],
                        "confidence": fact["confidence"],
                        "timestamp": fact["timestamp"]
                    })
            
            if memory_type in ["all", "conversations"] and self.cursor:
                self.cursor.execute('''
                    SELECT message_type, content, timestamp
                    FROM conversations
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (f"%{query}%", limit))
                
                rows = self.cursor.fetchall()
                for row in rows:
                    results.append({
                        "type": "conversation",
                        "message_type": row[0],
                        "content": row[1],
                        "timestamp": row[2]
                    })
            
            if memory_type in ["all", "tasks"]:
                tasks = self.get_tasks(limit=limit)
                for task in tasks:
                    if query.lower() in task["description"].lower():
                        results.append({
                            "type": "task",
                            "task_id": task["task_id"],
                            "description": task["description"],
                            "status": task["status"],
                            "timestamp": task["timestamp"]
                        })
            
            return results[:limit]
        
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return []
    
    def __del__(self):
        """Clean up database connection."""
        if self.conn:
            self.conn.close()
