import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import hashlib
from enum import Enum
import faiss
from sentence_transformers import SentenceTransformer
import asyncio
from abc import ABC, abstractmethod

class MemoryType(Enum):
    """Types of memory entries"""
    EPISODIC = "episodic"      # Specific experiences/conversations
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge and patterns
    EMOTIONAL = "emotional"    # Affective associations
    CONTEXTUAL = "contextual"  # Environmental/situational context

class MemoryPriority(Enum):
    """Memory importance levels"""
    CRITICAL = 4    # Core identity, critical decisions
    HIGH = 3        # Important preferences, key relationships
    MEDIUM = 2      # Regular interactions, learned patterns  
    LOW = 1         # Background context, casual mentions
    MINIMAL = 0     # Temporary, candidate for pruning

@dataclass
class MemoryEntry:
    """Individual memory record"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    memory_type: MemoryType = MemoryType.EPISODIC
    priority: MemoryPriority = MemoryPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    decay_factor: float = 1.0
    associations: List[str] = field(default_factory=list)  # Links to other memories
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID based on content and timestamp"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        time_hash = str(int(self.timestamp.timestamp()))[-6:]
        return f"{self.memory_type.value}_{content_hash}_{time_hash}"
    
    def access(self):
        """Mark memory as accessed and update statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_relevance_score(self, query_embedding: np.ndarray, time_weight: float = 0.2) -> float:
        """Calculate relevance score for retrieval"""
        if self.embedding is None:
            return 0.0
            
        # Semantic similarity
        similarity = np.dot(self.embedding, query_embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(query_embedding)
        )
        
        # Time decay
        time_diff = (datetime.now() - self.timestamp).total_seconds() / (24 * 3600)  # days
        time_score = np.exp(-time_diff / 30)  # 30-day half-life
        
        # Priority boost
        priority_score = self.priority.value / 4.0
        
        # Access frequency
        frequency_score = min(np.log1p(self.access_count) / 10, 1.0)
        
        # Combined score
        relevance = (
            similarity * 0.5 + 
            time_score * time_weight + 
            priority_score * 0.2 + 
            frequency_score * 0.1
        ) * self.decay_factor
        
        return float(relevance)

class MemoryIndex:
    """High-performance memory indexing using FAISS"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        
    def add_memory(self, memory: MemoryEntry):
        """Add memory to index"""
        if memory.embedding is None:
            return
            
        # Normalize embedding for cosine similarity
        embedding = memory.embedding / np.linalg.norm(memory.embedding)
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        self.index.add(embedding)
        self.id_to_idx[memory.id] = self.next_idx
        self.idx_to_id[self.next_idx] = memory.id
        self.next_idx += 1
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar memories"""
        if self.index.ntotal == 0:
            return []
            
        # Normalize query
        query = query_embedding / np.linalg.norm(query_embedding)
        query = query.astype(np.float32).reshape(1, -1)
        
        scores, indices = self.index.search(query, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.idx_to_id:
                memory_id = self.idx_to_id[idx]
                results.append((memory_id, float(score)))
                
        return results
    
    def remove_memory(self, memory_id: str):
        """Remove memory from index (marks as inactive)"""
        if memory_id in self.id_to_idx:
            # FAISS doesn't support removal, so we mark as inactive
            idx = self.id_to_idx[memory_id]
            del self.id_to_idx[memory_id]
            if idx in self.idx_to_id:
                del self.idx_to_id[idx]

class MemoryConsolidator:
    """Handles memory consolidation, compression, and pruning"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        
    def consolidate_similar_memories(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Merge similar memories to reduce redundancy"""
        if len(memories) < 2:
            return memories
            
        consolidated = []
        processed = set()
        
        for i, memory in enumerate(memories):
            if memory.id in processed:
                continue
                
            similar_group = [memory]
            processed.add(memory.id)
            
            # Find similar memories
            for j, other in enumerate(memories[i+1:], i+1):
                if other.id in processed or other.embedding is None or memory.embedding is None:
                    continue
                    
                similarity = np.dot(memory.embedding, other.embedding) / (
                    np.linalg.norm(memory.embedding) * np.linalg.norm(other.embedding)
                )
                
                if similarity > self.similarity_threshold:
                    similar_group.append(other)
                    processed.add(other.id)
            
            # Consolidate the group
            if len(similar_group) > 1:
                consolidated_memory = self._merge_memories(similar_group)
                consolidated.append(consolidated_memory)
            else:
                consolidated.append(memory)
                
        return consolidated
    
    def _merge_memories(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """Merge multiple similar memories into one"""
        # Use the most important/recent memory as base
        base = max(memories, key=lambda m: (m.priority.value, m.timestamp))
        
        # Combine content
        contents = [m.content for m in memories]
        merged_content = f"{base.content}\n[Consolidated from {len(memories)} similar memories]"
        
        # Average embeddings
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        if embeddings:
            merged_embedding = np.mean(embeddings, axis=0)
        else:
            merged_embedding = base.embedding
            
        # Combine metadata
        merged_tags = list(set().union(*[m.tags for m in memories]))
        merged_associations = list(set().union(*[m.associations for m in memories]))
        
        # Create consolidated memory
        consolidated = MemoryEntry(
            id=f"consolidated_{base.id}",
            content=merged_content,
            embedding=merged_embedding,
            memory_type=base.memory_type,
            priority=base.priority,
            timestamp=base.timestamp,
            tags=merged_tags,
            metadata={**base.metadata, 'consolidated_from': [m.id for m in memories]},
            access_count=sum(m.access_count for m in memories),
            last_accessed=max(m.last_accessed for m in memories),
            associations=merged_associations
        )
        
        return consolidated
    
    def prune_low_value_memories(self, memories: List[MemoryEntry], max_size: int) -> List[MemoryEntry]:
        """Remove least valuable memories when approaching capacity"""
        if len(memories) <= max_size:
            return memories
            
        # Score memories for retention
        scored_memories = []
        for memory in memories:
            # Retention score based on priority, recency, access patterns
            age_days = (datetime.now() - memory.timestamp).total_seconds() / (24 * 3600)
            recency_score = np.exp(-age_days / 30)
            
            access_recency = (datetime.now() - memory.last_accessed).total_seconds() / (24 * 3600)
            access_score = np.exp(-access_recency / 7)
            
            retention_score = (
                memory.priority.value * 0.4 +
                recency_score * 0.3 +
                access_score * 0.2 +
                min(np.log1p(memory.access_count) / 10, 1.0) * 0.1
            )
            
            scored_memories.append((retention_score, memory))
        
        # Keep top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:max_size]]

class MemoryStore:
    """Main memory storage and retrieval system"""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        max_memories: int = 10000,
        consolidation_interval: int = 1000
    ):
        self.memories: Dict[str, MemoryEntry] = {}
        self.index = MemoryIndex()
        self.consolidator = MemoryConsolidator()
        self.encoder = SentenceTransformer(embedding_model)
        self.max_memories = max_memories
        self.consolidation_interval = consolidation_interval
        self.operation_count = 0
        
        # Memory statistics
        self.stats = {
            'total_memories': 0,
            'retrievals': 0,
            'consolidations': 0,
            'type_distribution': defaultdict(int)
        }
    
    def add_memory(
        self, 
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> MemoryEntry:
        """Add new memory entry"""
        
        # Generate embedding
        embedding = self.encoder.encode(content)
        
        # Create memory entry
        memory = MemoryEntry(
            id="",  # Will be auto-generated
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            priority=priority,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store memory
        self.memories[memory.id] = memory
        self.index.add_memory(memory)
        
        # Update statistics
        self.stats['total_memories'] += 1
        self.stats['type_distribution'][memory_type.value] += 1
        self.operation_count += 1
        
        # Periodic maintenance
        if self.operation_count % self.consolidation_interval == 0:
            self._perform_maintenance()
            
        return memory
    
    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        memory_types: List[MemoryType] = None,
        time_range: Tuple[datetime, datetime] = None,
        min_priority: MemoryPriority = MemoryPriority.MINIMAL
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        
        # Generate query embedding
        query_embedding = self.encoder.encode(query)
        
        # Get candidates from index
        candidates = self.index.search(query_embedding, k * 3)  # Get more candidates for filtering
        
        # Filter and score candidates
        relevant_memories = []
        for memory_id, _ in candidates:
            if memory_id not in self.memories:
                continue
                
            memory = self.memories[memory_id]
            
            # Apply filters
            if memory_types and memory.memory_type not in memory_types:
                continue
            if memory.priority.value < min_priority.value:
                continue
            if time_range:
                if not (time_range[0] <= memory.timestamp <= time_range[1]):
                    continue
            
            # Calculate relevance score
            relevance = memory.calculate_relevance_score(query_embedding)
            relevant_memories.append((relevance, memory))
        
        # Sort by relevance and return top k
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        result_memories = [memory for _, memory in relevant_memories[:k]]
        
        # Update access statistics
        for memory in result_memories:
            memory.access()
        
        self.stats['retrievals'] += 1
        return result_memories
    
    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve specific memory by ID"""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing memory"""
        if memory_id not in self.memories:
            return False
            
        memory = self.memories[memory_id]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(memory, field):
                setattr(memory, field, value)
                
        # Regenerate embedding if content changed
        if 'content' in updates:
            memory.embedding = self.encoder.encode(memory.content)
            # Note: FAISS index would need rebuilding for embedding updates
            
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory entry"""
        if memory_id not in self.memories:
            return False
            
        # Remove from store and index
        memory = self.memories.pop(memory_id)
        self.index.remove_memory(memory_id)
        
        # Update statistics
        self.stats['total_memories'] -= 1
        self.stats['type_distribution'][memory.memory_type.value] -= 1
        
        return True
    
    def _perform_maintenance(self):
        """Periodic memory maintenance"""
        memory_list = list(self.memories.values())
        
        # Consolidate similar memories
        if len(memory_list) > self.max_memories * 0.8:
            consolidated = self.consolidator.consolidate_similar_memories(memory_list)
            
            # Replace memories with consolidated versions
            self.memories.clear()
            for memory in consolidated:
                self.memories[memory.id] = memory
            
            # Rebuild index
            self.index = MemoryIndex(self.index.embedding_dim)
            for memory in consolidated:
                self.index.add_memory(memory)
                
            self.stats['consolidations'] += 1
        
        # Prune if still over capacity
        if len(self.memories) > self.max_memories:
            pruned = self.consolidator.prune_low_value_memories(
                list(self.memories.values()), 
                self.max_memories
            )
            
            # Keep only pruned memories
            self.memories.clear()
            for memory in pruned:
                self.memories[memory.id] = memory
                
            # Rebuild index
            self.index = MemoryIndex(self.index.embedding_dim)
            for memory in pruned:
                self.index.add_memory(memory)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            **self.stats,
            'current_memory_count': len(self.memories),
            'memory_types': dict(self.stats['type_distribution'])
        }
    
    def export_memories(self, file_path: str):
        """Export memories to JSON file"""
        export_data = []
        for memory in self.memories.values():
            export_data.append({
                'id': memory.id,
                'content': memory.content,
                'memory_type': memory.memory_type.value,
                'priority': memory.priority.value,
                'timestamp': memory.timestamp.isoformat(),
                'tags': memory.tags,
                'metadata': memory.metadata,
                'access_count': memory.access_count,
                'associations': memory.associations
            })
            
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_memories(self, file_path: str):
        """Import memories from JSON file"""
        with open(file_path, 'r') as f:
            import_data = json.load(f)
            
        for data in import_data:
            memory = MemoryEntry(
                id=data['id'],
                content=data['content'],
                memory_type=MemoryType(data['memory_type']),
                priority=MemoryPriority(data['priority']),
                timestamp=datetime.fromisoformat(data['timestamp']),
                tags=data['tags'],
                metadata=data['metadata'],
                access_count=data['access_count'],
                associations=data.get('associations', [])
            )
            
            # Generate embedding
            memory.embedding = self.encoder.encode(memory.content)
            
            # Store memory
            self.memories[memory.id] = memory
            self.index.add_memory(memory)

# Example usage and utilities
class ORACMemoryManager:
    """High-level interface for ORAC memory operations"""
    
    def __init__(self):
        self.store = MemoryStore()
        self.context_cache = {}
        
    def remember(self, content: str, importance: str = "medium", context: str = None) -> str:
        """Simple interface to add memories"""
        priority_map = {
            "critical": MemoryPriority.CRITICAL,
            "high": MemoryPriority.HIGH,
            "medium": MemoryPriority.MEDIUM,
            "low": MemoryPriority.LOW
        }
        
        priority = priority_map.get(importance.lower(), MemoryPriority.MEDIUM)
        
        # Determine memory type based on content
        memory_type = self._classify_memory_type(content)
        
        memory = self.store.add_memory(
            content=content,
            memory_type=memory_type,
            priority=priority,
            metadata={'context': context} if context else {}
        )
        
        return memory.id
    
    def recall(self, query: str, count: int = 3) -> List[str]:
        """Simple interface to retrieve memories"""
        memories = self.store.retrieve_memories(query, k=count)
        return [memory.content for memory in memories]
    
    def _classify_memory_type(self, content: str) -> MemoryType:
        """Simple heuristic to classify memory type"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['feel', 'emotion', 'love', 'hate', 'happy', 'sad']):
            return MemoryType.EMOTIONAL
        elif any(word in content_lower for word in ['how to', 'procedure', 'step', 'process']):
            return MemoryType.PROCEDURAL
        elif any(word in content_lower for word in ['fact', 'define', 'is', 'means']):
            return MemoryType.SEMANTIC
        else:
            return MemoryType.EPISODIC

if __name__ == "__main__":
    # Example usage
    memory_manager = ORACMemoryManager()
    
    # Add some memories
    memory_manager.remember("User prefers Python over JavaScript", "high")
    memory_manager.remember("Had a great conversation about AI ethics yesterday", "medium")
    memory_manager.remember("User is working on a transformer architecture project", "high")
    
    # Retrieve memories
    relevant = memory_manager.recall("programming preferences", count=2)
    print("Relevant memories:", relevant)
    
    # Get statistics
    stats = memory_manager.store.get_stats()
    print("Memory stats:", stats)
