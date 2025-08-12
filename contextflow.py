import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class ContextMode(Enum):
    """ORAC context modes for persona/mode switching"""
    PERSONAL = "personal"
    EXECUTIVE = "executive" 
    STRATEGIST = "strategist"
    ANALYST = "analyst"
    COACH = "coach"

@dataclass
class ORACConfig:
    """Configuration for ORAC Transformer"""
    vocab_size: int = 50257
    max_seq_len: int = 8192
    n_layers: int = 24
    n_heads: int = 16
    d_model: int = 1024
    d_ff: int = 4096
    dropout: float = 0.1
    
    # ORAC-specific parameters
    memory_dim: int = 512
    context_dim: int = 256
    prediction_horizon: int = 64
    n_modes: int = len(ContextMode)
    tool_embed_dim: int = 128

class MemoryAttention(nn.Module):
    """Specialized attention for persistent memory integration"""
    
    def __init__(self, config: ORACConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.memory_dim = config.memory_dim
        
        # Memory-aware projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.memory_dim, config.d_model)
        self.v_proj = nn.Linear(config.memory_dim, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Memory relevance scorer
        self.relevance_net = nn.Sequential(
            nn.Linear(config.d_model + config.memory_dim, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states, memory_embeddings, memory_metadata=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project queries from current context
        queries = self.q_proj(hidden_states)
        
        # Project keys/values from memory
        keys = self.k_proj(memory_embeddings)
        values = self.v_proj(memory_embeddings)
        
        # Compute relevance scores for memory filtering
        query_expanded = queries.unsqueeze(2).expand(-1, -1, memory_embeddings.size(1), -1)
        memory_expanded = memory_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        relevance_input = torch.cat([query_expanded, memory_expanded], dim=-1)
        relevance_scores = self.relevance_net(relevance_input).squeeze(-1)
        
        # Attention with memory
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_model)
        scores = scores * relevance_scores  # Weight by relevance
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)
        
        return self.out_proj(attn_output), attn_weights

class PredictiveHead(nn.Module):
    """Predictive modeling component for future scenario generation"""
    
    def __init__(self, config: ORACConfig):
        super().__init__()
        self.config = config
        
        # Future state prediction
        self.future_encoder = nn.LSTM(
            config.d_model, 
            config.d_model // 2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Decision outcome modeling
        self.decision_projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.prediction_horizon),
            nn.Tanh()
        )
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(), 
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context_repr):
        # Encode future possibilities
        future_states, _ = self.future_encoder(context_repr)
        
        # Generate decision pathways
        decision_logits = self.decision_projector(future_states)
        
        # Estimate prediction confidence
        confidence = self.confidence_net(future_states.mean(dim=1))
        
        return {
            'future_states': future_states,
            'decision_logits': decision_logits, 
            'confidence': confidence
        }

class ModeAdapter(nn.Module):
    """Persona/Mode switching mechanism for ORAC"""
    
    def __init__(self, config: ORACConfig):
        super().__init__()
        self.config = config
        
        # Mode embeddings
        self.mode_embeddings = nn.Embedding(config.n_modes, config.d_model)
        
        # Mode-specific adaptation layers
        self.mode_adapters = nn.ModuleDict({
            mode.value: nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model)
            ) for mode in ContextMode
        })
        
        # Mode classifier for automatic switching
        self.mode_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.n_modes)
        )
    
    def forward(self, hidden_states, mode=None, auto_detect=True):
        if auto_detect and mode is None:
            # Automatic mode detection
            mode_logits = self.mode_classifier(hidden_states.mean(dim=1))
            mode_idx = torch.argmax(mode_logits, dim=-1)
            detected_mode = list(ContextMode)[mode_idx.item()]
        else:
            detected_mode = mode or ContextMode.PERSONAL
        
        # Apply mode-specific transformation
        mode_embedding = self.mode_embeddings(
            torch.tensor([list(ContextMode).index(detected_mode)]).to(hidden_states.device)
        )
        
        # Blend with mode-specific adapter
        adapted_states = self.mode_adapters[detected_mode.value](hidden_states)
        adapted_states = adapted_states + mode_embedding.unsqueeze(1)
        
        return adapted_states, detected_mode

class ToolInterface(nn.Module):
    """Dynamic tool and API integration layer"""
    
    def __init__(self, config: ORACConfig):
        super().__init__()
        self.config = config
        
        # Tool embedding and selection
        self.tool_embeddings = nn.Embedding(100, config.tool_embed_dim)  # Support 100 tools
        self.tool_selector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 100),
            nn.Softmax(dim=-1)
        )
        
        # Tool parameter generation
        self.param_generator = nn.Linear(config.d_model, config.tool_embed_dim * 4)
        
        # Tool result integration
        self.result_integrator = nn.Linear(config.tool_embed_dim + config.d_model, config.d_model)
    
    def forward(self, hidden_states, available_tools=None):
        # Select appropriate tools
        tool_probs = self.tool_selector(hidden_states.mean(dim=1))
        
        # Generate tool parameters
        tool_params = self.param_generator(hidden_states.mean(dim=1))
        
        # Mock tool execution (in practice, this would call external APIs)
        tool_results = torch.randn(hidden_states.size(0), self.config.tool_embed_dim).to(hidden_states.device)
        
        # Integrate tool results back into context
        integrated = self.result_integrator(
            torch.cat([tool_results.unsqueeze(1).expand(-1, hidden_states.size(1), -1), hidden_states], dim=-1)
        )
        
        return integrated, tool_probs

class ORACTransformerLayer(nn.Module):
    """Single transformer layer enhanced with ORAC capabilities"""
    
    def __init__(self, config: ORACConfig):
        super().__init__()
        self.config = config
        
        # Standard transformer components
        self.self_attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_attention = MemoryAttention(config)
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        # Mode adaptation
        self.mode_adapter = ModeAdapter(config)
        
    def forward(self, x, memory_embeddings=None, mode=None, attention_mask=None):
        # Self attention
        attn_out, _ = self.self_attention(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + attn_out)
        
        # Memory-augmented cross attention
        if memory_embeddings is not None:
            mem_out, mem_attn = self.cross_attention(x, memory_embeddings)
            x = self.norm2(x + mem_out)
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        # Mode adaptation
        x, detected_mode = self.mode_adapter(x, mode)
        
        return x, detected_mode

class ORACTransformer(nn.Module):
    """Complete ORAC Transformer model"""
    
    def __init__(self, config: ORACConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ORACTransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        # ORAC-specific components
        self.predictive_head = PredictiveHead(config)
        self.tool_interface = ToolInterface(config)
        
        # Output head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Context compression for memory efficiency
        self.context_compressor = nn.Sequential(
            nn.Linear(config.d_model, config.context_dim),
            nn.ReLU(),
            nn.Linear(config.context_dim, config.d_model)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        memory_embeddings: Optional[torch.Tensor] = None,
        mode: Optional[ContextMode] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_tools: bool = False,
        generate_predictions: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create embeddings
        token_embeds = self.token_embeddings(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embeddings(pos_ids)
        
        x = token_embeds + pos_embeds
        
        # Process through transformer layers
        detected_modes = []
        for layer in self.layers:
            x, detected_mode = layer(x, memory_embeddings, mode, attention_mask)
            detected_modes.append(detected_mode)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Generate language model logits
        lm_logits = self.lm_head(x)
        
        # Prepare outputs
        outputs = {
            'logits': lm_logits,
            'hidden_states': x,
            'detected_modes': detected_modes[-1] if detected_modes else mode
        }
        
        # Optional tool integration
        if use_tools:
            tool_enhanced_x, tool_probs = self.tool_interface(x)
            outputs['tool_enhanced_logits'] = self.lm_head(tool_enhanced_x)
            outputs['tool_probabilities'] = tool_probs
        
        # Optional predictive modeling
        if generate_predictions:
            # Compress context for prediction
            compressed_context = self.context_compressor(x)
            predictions = self.predictive_head(compressed_context)
            outputs.update(predictions)
        
        return outputs

# Example usage and configuration
def create_orac_model(vocab_size=50257):
    """Factory function to create ORAC model with sensible defaults"""
    config = ORACConfig(
        vocab_size=vocab_size,
        max_seq_len=4096,  # Reasonable context window
        n_layers=12,       # Medium-sized model
        n_heads=12,
        d_model=768,
        d_ff=3072
    )
    return ORACTransformer(config)

# Training utilities
class ORACTrainer:
    """Training utilities for ORAC model"""
    
    def __init__(self, model: ORACTransformer, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def compute_loss(self, batch, alpha_lm=1.0, alpha_pred=0.1, alpha_mode=0.05):
        """Multi-objective loss combining language modeling, prediction, and mode classification"""
        
        outputs = self.model(
            batch['input_ids'],
            memory_embeddings=batch.get('memory_embeddings'),
            generate_predictions=True
        )
        
        # Language modeling loss
        lm_loss = F.cross_entropy(
            outputs['logits'].view(-1, outputs['logits'].size(-1)),
            batch['target_ids'].view(-1),
            ignore_index=-100
        )
        
        total_loss = alpha_lm * lm_loss
        
        # Prediction loss (if targets available)
        if 'prediction_targets' in batch and 'future_states' in outputs:
            pred_loss = F.mse_loss(outputs['future_states'], batch['prediction_targets'])
            total_loss += alpha_pred * pred_loss
        
        # Mode classification loss (if targets available) 
        if 'mode_targets' in batch:
            mode_idx = [list(ContextMode).index(outputs['detected_modes'])]
            mode_loss = F.cross_entropy(
                torch.tensor(mode_idx).to(self.device),
                batch['mode_targets']
            )
            total_loss += alpha_mode * mode_loss
            
        return total_loss, {
            'lm_loss': lm_loss.item(),
            'total_loss': total_loss.item()
        }

if __name__ == "__main__":
    # Example instantiation
    model = create_orac_model()
    print(f"ORAC Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids, generate_predictions=True, use_tools=True)
        print("Output keys:", list(outputs.keys()))
        print("Logits shape:", outputs['logits'].shape)
        if 'confidence' in outputs:
            print("Prediction confidence:", outputs['confidence'].item())
