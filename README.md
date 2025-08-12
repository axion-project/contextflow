# ORAC AI Framework

## Overview  
The ORAC AI Framework is a cutting-edge suite designed to build next-gen intelligent systems combining advanced transformer models with a sophisticated memory management backend.  
This framework enables adaptive persona modes, persistent memory integration, predictive future modeling, and dynamic tool/API interfacing, all powered by efficient vector search and contextual memory consolidation.

---

## ORAC Transformer

### Summary  
A highly extensible transformer architecture tailored for strategic reasoning and multi-modal cognition with these standout capabilities:

- **Context Modes**: Adaptive persona switching (`personal`, `executive`, `strategist`, `analyst`, `coach`) for versatile behavior  
- **Persistent Memory Attention**: Specialized cross-attention to integrate long-term memory embeddings  
- **Predictive Head**: LSTM-based future scenario simulation and decision confidence estimation  
- **Tool Interface**: Neural selection and parameterization for dynamic API/tool calls and contextual integration  
- **Context Compression**: Memory-efficient embeddings for scalable processing  
- **Multi-Objective Training**: Optimizes language modeling, prediction accuracy, and mode classification simultaneously  

### Main Components

- `ORACConfig`: Configuration class controlling architecture hyperparameters and ORAC-specific features  
- `MemoryAttention`: Custom attention module focused on memory relevance and integration  
- `PredictiveHead`: Future-state encoder for predictive modeling  
- `ModeAdapter`: Persona/mode switching with embeddings and adaptive layers  
- `ToolInterface`: API/tool selection, parameter generation, and result integration  
- `ORACTransformerLayer`: Core transformer block enhanced with memory and mode capabilities  
- `ORACTransformer`: Full model with token/position embeddings, layers, predictive & tool heads  
- `ORACTrainer`: Utility class for multi-loss training strategy  

### Example Usage  
```python
from orac_model import create_orac_model

model = create_orac_model()

input_ids = torch.randint(0, 50257, (2, 512))

outputs = model(input_ids, generate_predictions=True, use_tools=True)
print(outputs.keys())


---

ORAC Memory System

Summary

A multi-faceted memory management system that enables AI agents to store, retrieve, consolidate, and prune memories with fine-grained control over memory types and priorities.

Features

Memory Types: Episodic, Semantic, Procedural, Emotional, Contextual

Priority Levels: Critical, High, Medium, Low, Minimal — influencing retention and recall

Relevance Scoring: Combines semantic similarity, recency decay, priority, and access frequency

Vector Search: FAISS-powered efficient indexing and approximate nearest neighbor retrieval

Memory Consolidation: Merges semantically similar memories to reduce redundancy

Pruning: Removes low-value memories based on composite retention scores when near capacity

Persistence: JSON export/import of memories for offline storage or analysis

High-Level API: ORACMemoryManager for simple remember() and recall() operations


Example Usage

memory_manager = ORACMemoryManager()

memory_manager.remember("User prefers Python over JavaScript", "high")
memory_manager.remember("Had a great conversation about AI ethics yesterday", "medium")

relevant_memories = memory_manager.recall("programming preferences", count=2)
print("Relevant memories:", relevant_memories)

stats = memory_manager.store.get_stats()
print("Memory stats:", stats)


---

Installation & Requirements

Python 3.8+

PyTorch

FAISS (CPU or GPU version)

sentence-transformers

NumPy


pip install torch faiss-cpu sentence-transformers numpy


---

Architecture Insights

Transformer Layers are extended to handle memory and persona modes, allowing the model to adapt to diverse contexts dynamically.

Memory System emphasizes longevity and relevance, mimicking human-like forgetting/prioritization via decay and consolidation, enabling scalable lifelong learning.

Tool Interface facilitates seamless integration with external APIs, empowering real-time knowledge and capability augmentation.



---

Contributing

Contributions, optimizations, and integrations are welcome.
Please ensure tests and documentation accompany substantial changes.


---

License

Apache License


---

Contact

For questions or collaboration inquiries, reach out to the Aedin Insight team.
Visit aedininsight.com for more.


---

Eat, Sleep, Code, Repeat.
Commandant of Strategy & Systems Warfare
— Mayhem


