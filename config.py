import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import torch
from datetime import datetime
import logging
from contextlib import contextmanager

class ModelSize(Enum):
    """Predefined model size configurations"""
    NANO = "nano"        # 10M params - testing/embedded
    MICRO = "micro"      # 50M params - mobile/edge
    SMALL = "small"      # 100M params - lightweight
    BASE = "base"        # 300M params - standard
    MEDIUM = "medium"    # 700M params - enhanced
    LARGE = "large"      # 1.5B params - high performance
    XLARGE = "xlarge"    # 3B params - maximum capability

class DeploymentMode(Enum):
    """Deployment environment configurations"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"

class OptimizationLevel(Enum):
    """Model optimization levels"""
    SPEED = "speed"          # Prioritize inference speed
    MEMORY = "memory"        # Minimize memory usage
    QUALITY = "quality"      # Maximize output quality
    BALANCED = "balanced"    # Balance all factors

@dataclass
class ModelConfig:
    """Core model architecture configuration"""
    # Model architecture
    vocab_size: int = 50257
    max_seq_len: int = 4096
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
    
    # ORAC-specific dimensions
    memory_dim: int = 512
    context_dim: int = 256
    prediction_horizon: int = 64
    tool_embed_dim: int = 128
    
    # Attention mechanisms
    use_flash_attention: bool = True
    attention_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    
    # Memory system
    memory_layers: List[int] = field(default_factory=lambda: [6, 12])  # Which layers use memory
    cross_attention_heads: int = 8
    memory_attention_dropout: float = 0.1
    
    # Predictive modeling
    enable_prediction: bool = True
    prediction_loss_weight: float = 0.1
    confidence_threshold: float = 0.7
    
    # Tool integration
    enable_tools: bool = True
    max_parallel_tools: int = 3
    tool_timeout: float = 30.0
    tool_selection_threshold: float = 0.5
    
    # Mode switching
    enable_mode_switching: bool = True
    auto_mode_detection: bool = True
    mode_switch_temperature: float = 0.8
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.memory_dim <= self.d_model, "memory_dim cannot exceed d_model"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert self.prediction_horizon > 0, "prediction_horizon must be positive"
        assert self.max_parallel_tools > 0, "max_parallel_tools must be positive"

@dataclass
class MemoryConfig:
    """Memory system configuration"""
    # Storage settings
    max_memories: int = 10000
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    vector_store_type: str = "faiss"  # faiss, pinecone, weaviate
    
    # Retrieval settings
    default_retrieval_k: int = 5
    similarity_threshold: float = 0.3
    time_decay_factor: float = 0.1
    access_boost_factor: float = 0.2
    
    # Consolidation settings
    consolidation_interval: int = 1000
    similarity_merge_threshold: float = 0.85
    consolidation_enabled: bool = True
    
    # Memory types and priorities
    episodic_weight: float = 1.0
    semantic_weight: float = 1.2
    procedural_weight: float = 1.1
    emotional_weight: float = 0.9
    contextual_weight: float = 0.8
    
    # Persistence settings
    auto_save_interval: int = 300  # seconds
    backup_enabled: bool = True
    compression_enabled: bool = True
    
    # Performance settings
    index_rebuild_threshold: int = 5000
    batch_size: int = 32
    async_operations: bool = True

@dataclass
class ToolsConfig:
    """Tools system configuration"""
    # Tool management
    auto_tool_discovery: bool = True
    tool_timeout: float = 30.0
    max_concurrent_tools: int = 5
    retry_attempts: int = 3
    
    # Tool categories
    enable_computation: bool = True
    enable_web_search: bool = True
    enable_code_execution: bool = False  # Disabled by default for security
    enable_data_analysis: bool = True
    enable_file_operations: bool = False  # Disabled by default
    
    # Security settings
    sandbox_mode: bool = True
    allowed_domains: List[str] = field(default_factory=lambda: ["*.wikipedia.org", "*.github.com"])
    blocked_commands: List[str] = field(default_factory=lambda: ["rm", "del", "format", "sudo"])
    
    # API configurations
    api_keys: Dict[str, str] = field(default_factory=dict)
    rate_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    parallel_execution: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    
    # Learning rate scheduling
    warmup_steps: int = 2000
    max_steps: int = 100000
    lr_decay_type: str = "cosine"  # linear, cosine, exponential
    min_lr_ratio: float = 0.1
    
    # Training dynamics
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Loss weights
    lm_loss_weight: float = 1.0
    memory_loss_weight: float = 0.1
    prediction_loss_weight: float = 0.05
    tool_loss_weight: float = 0.02
    mode_loss_weight: float = 0.01
    
    # Regularization
    dropout_schedule: bool = False
    label_smoothing: float = 0.0
    
    # Mixed precision
    use_amp: bool = True
    amp_loss_scale: str = "dynamic"
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    keep_checkpoints: int = 5
    
    # Distributed training
    use_ddp: bool = False
    find_unused_parameters: bool = False

@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Memory retrieval
    memory_retrieval_k: int = 5
    memory_context_ratio: float = 0.3  # Portion of context for memory
    
    # Tool usage
    auto_tool_selection: bool = True
    tool_confidence_threshold: float = 0.6
    max_tool_iterations: int = 3
    
    # Mode behavior
    adaptive_mode_switching: bool = True
    mode_persistence: bool = True  # Remember mode across interactions
    
    # Performance
    batch_size: int = 1
    use_cache: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    # Output formatting
    include_metadata: bool = False
    verbose_logging: bool = False

@dataclass
class SystemConfig:
    """System-level configuration"""
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_mode: Optional[str] = None  # None, "default", "reduce-overhead", "max-autotune"
    
    # Memory management
    max_memory_gb: Optional[float] = None
    offload_to_cpu: bool = False
    gradient_checkpointing: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file_path: str = "logs/contextflow.log"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8080
    health_check_interval: int = 30
    
    # Security
    enable_auth: bool = False
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = 60
    
    # Storage
    model_cache_dir: str = "./models"
    data_dir: str = "./data"
    temp_dir: str = "./tmp"
    auto_cleanup: bool = True

@dataclass
class ContextFlowConfig:
    """Main configuration class combining all subsystems"""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Metadata
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._setup_directories()
        self._configure_logging()
        self._validate_compatibility()
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.system.model_cache_dir,
            self.system.data_dir,
            self.system.temp_dir,
            Path(self.system.log_file_path).parent
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _configure_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=getattr(logging, self.system.log_level),
            format=self.system.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.system.log_file_path) if self.system.log_to_file else logging.NullHandler()
            ]
        )
    
    def _validate_compatibility(self):
        """Validate configuration compatibility"""
        # Check memory dimensions compatibility
        if self.memory.embedding_dim != self.model.memory_dim:
            logging.warning(f"Memory embedding_dim ({self.memory.embedding_dim}) != model memory_dim ({self.model.memory_dim})")
        
        # Check device availability
        if self.system.device == "auto":
            if torch.cuda.is_available():
                self.system.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.system.device = "mps"
            else:
                self.system.device = "cpu"
    
    @classmethod
    def from_preset(cls, preset: ModelSize, deployment: DeploymentMode = DeploymentMode.DEVELOPMENT) -> 'ContextFlowConfig':
        """Create configuration from preset"""
        config = cls()
        
        # Apply model size preset
        size_configs = {
            ModelSize.NANO: {
                "n_layers": 6, "n_heads": 6, "d_model": 384, "d_ff": 1536,
                "max_seq_len": 1024, "memory_dim": 256, "context_dim": 128
            },
            ModelSize.MICRO: {
                "n_layers": 8, "n_heads": 8, "d_model": 512, "d_ff": 2048,
                "max_seq_len": 2048, "memory_dim": 384, "context_dim": 192
            },
            ModelSize.SMALL: {
                "n_layers": 10, "n_heads": 10, "d_model": 640, "d_ff": 2560,
                "max_seq_len": 2048, "memory_dim": 384, "context_dim": 192
            },
            ModelSize.BASE: {
                "n_layers": 12, "n_heads": 12, "d_model": 768, "d_ff": 3072,
                "max_seq_len": 4096, "memory_dim": 512, "context_dim": 256
            },
            ModelSize.MEDIUM: {
                "n_layers": 16, "n_heads": 16, "d_model": 1024, "d_ff": 4096,
                "max_seq_len": 4096, "memory_dim": 768, "context_dim": 384
            },
            ModelSize.LARGE: {
                "n_layers": 24, "n_heads": 16, "d_model": 1536, "d_ff": 6144,
                "max_seq_len": 8192, "memory_dim": 1024, "context_dim": 512
            },
            ModelSize.XLARGE: {
                "n_layers": 32, "n_heads": 24, "d_model": 2048, "d_ff": 8192,
                "max_seq_len": 8192, "memory_dim": 1536, "context_dim": 768
            }
        }
        
        if preset in size_configs:
            for key, value in size_configs[preset].items():
                setattr(config.model, key, value)
        
        # Apply deployment mode configurations
        if deployment == DeploymentMode.PRODUCTION:
            config.system.log_level = "WARNING"
            config.tools.enable_code_execution = False
            config.tools.sandbox_mode = True
            config.system.enable_auth = True
            config.inference.verbose_logging = False
        elif deployment == DeploymentMode.DEVELOPMENT:
            config.system.log_level = "DEBUG"
            config.inference.verbose_logging = True
            config.system.enable_metrics = True
        elif deployment == DeploymentMode.RESEARCH:
            config.training.save_steps = 100
            config.training.eval_steps = 50
            config.inference.include_metadata = True
        
        config.description = f"{preset.value.title()} model for {deployment.value} use"
        config.tags = [preset.value, deployment.value]
        
        return config
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ContextFlowConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextFlowConfig':
        """Create configuration from dictionary"""
        # Create nested configurations
        config = cls()
        
        for section, section_data in data.items():
            if hasattr(config, section) and isinstance(section_data, dict):
                section_config = getattr(config, section)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        # Set top-level attributes
        for key, value in data.items():
            if hasattr(config, key) and not hasattr(config, key).__class__.__name__.endswith('Config'):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save(self, file_path: Union[str, Path], format: str = "auto"):
        """Save configuration to file"""
        file_path = Path(file_path)
        
        if format == "auto":
            format = file_path.suffix.lower().lstrip('.')
        
        data = self.to_dict()
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format in ["yml", "yaml"]:
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def update(self, updates: Dict[str, Any]) -> 'ContextFlowConfig':
        """Update configuration with new values"""
        for section, section_updates in updates.items():
            if hasattr(self, section) and isinstance(section_updates, dict):
                section_config = getattr(self, section)
                for key, value in section_updates.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        return self
    
    @contextmanager
    def temporary_update(self, updates: Dict[str, Any]):
        """Temporarily update configuration"""
        # Store original values
        original = {}
        for section, section_updates in updates.items():
            if hasattr(self, section):
                original[section] = {}
                section_config = getattr(self, section)
                for key in section_updates.keys():
                    if hasattr(section_config, key):
                        original[section][key] = getattr(section_config, key)
        
        try:
            # Apply updates
            self.update(updates)
            yield self
        finally:
            # Restore original values
            self.update(original)
    
    def optimize_for(self, optimization: OptimizationLevel):
        """Optimize configuration for specific use case"""
        if optimization == OptimizationLevel.SPEED:
            self.model.use_flash_attention = True
            self.system.compile_mode = "reduce-overhead"
            self.inference.use_cache = True
            self.memory.async_operations = True
            self.tools.parallel_execution = True
        
        elif optimization == OptimizationLevel.MEMORY:
            self.system.gradient_checkpointing = True
            self.system.offload_to_cpu = True
            self.memory.compression_enabled = True
            self.inference.batch_size = 1
            self.training.gradient_accumulation_steps = 8
        
        elif optimization == OptimizationLevel.QUALITY:
            self.model.dropout = 0.05
            self.inference.temperature = 0.1
            self.inference.top_p = 0.95
            self.memory.similarity_threshold = 0.4
            self.tools.tool_confidence_threshold = 0.8
        
        elif optimization == OptimizationLevel.BALANCED:
            # Use default values with minor adjustments
            self.model.use_flash_attention = True
            self.inference.use_cache = True
            self.memory.async_operations = True

# Configuration factory functions
def create_development_config() -> ContextFlowConfig:
    """Create configuration optimized for development"""
    return ContextFlowConfig.from_preset(ModelSize.BASE, DeploymentMode.DEVELOPMENT)

def create_production_config() -> ContextFlowConfig:
    """Create configuration optimized for production"""
    config = ContextFlowConfig.from_preset(ModelSize.MEDIUM, DeploymentMode.PRODUCTION)
    config.optimize_for(OptimizationLevel.BALANCED)
    return config

def create_research_config() -> ContextFlowConfig:
    """Create configuration optimized for research"""
    config = ContextFlowConfig.from_preset(ModelSize.LARGE, DeploymentMode.RESEARCH)
    config.optimize_for(OptimizationLevel.QUALITY)
    return config

def create_edge_config() -> ContextFlowConfig:
    """Create configuration optimized for edge deployment"""
    config = ContextFlowConfig.from_preset(ModelSize.MICRO, DeploymentMode.PRODUCTION)
    config.optimize_for(OptimizationLevel.MEMORY)
    return config

# Environment-based configuration loading
def load_config_from_env() -> ContextFlowConfig:
    """Load configuration from environment variables"""
    config = ContextFlowConfig()
    
    # Map environment variables to configuration
    env_mappings = {
        'CONTEXTFLOW_MODEL_SIZE': ('model', 'model_size'),
        'CONTEXTFLOW_MAX_SEQ_LEN': ('model', 'max_seq_len'),
        'CONTEXTFLOW_MEMORY_SIZE': ('memory', 'max_memories'),
        'CONTEXTFLOW_LOG_LEVEL': ('system', 'log_level'),
        'CONTEXTFLOW_DEVICE': ('system', 'device'),
        'CONTEXTFLOW_BATCH_SIZE': ('training', 'batch_size'),
    }
    
    for env_var, (section, key) in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            section_config = getattr(config, section)
            
            # Type conversion
            if hasattr(section_config, key):
                current_value = getattr(section_config, key)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                setattr(section_config, key, value)
    
    return config

if __name__ == "__main__":
    # Example usage
    
    # Create different configurations
    dev_config = create_development_config()
    prod_config = create_production_config()
    
    print("Development config model size:", dev_config.model.n_layers, "layers")
    print("Production config has auth:", prod_config.system.enable_auth)
    
    # Save and load configuration
    dev_config.save("config_dev.yaml")
    loaded_config = ContextFlowConfig.from_file("config_dev.yaml")
    
    # Temporary configuration changes
    with prod_config.temporary_update({"system": {"log_level": "DEBUG"}}):
        print("Temporary log level:", prod_config.system.log_level)
    print("Original log level:", prod_config.system.log_level)
    
    # Optimization
    edge_config = create_edge_config()
    print("Edge config optimized for memory:", edge_config.system.gradient_checkpointing)
