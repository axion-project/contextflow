import torch
import torch.nn as nn
import asyncio
import aiohttp
import json
import inspect
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from datetime import datetime
import ast
import re
from functools import wraps
import traceback
import time

class ToolCategory(Enum):
    """Categories of available tools"""
    COMPUTATION = "computation"
    DATA_ACCESS = "data_access"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    CREATION = "creation"
    AUTOMATION = "automation"
    SEARCH = "search"
    PERSONAL = "personal"

class ToolPriority(Enum):
    """Tool execution priority levels"""
    CRITICAL = 4    # System-critical operations
    HIGH = 3        # User-requested actions
    MEDIUM = 2      # Contextual enhancements
    LOW = 1         # Background optimizations
    OPTIONAL = 0    # Nice-to-have additions

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'tool_name': self.tool_name
        }

@dataclass
class ToolSpec:
    """Tool specification and metadata"""
    name: str
    description: str
    category: ToolCategory
    priority: ToolPriority = ToolPriority.MEDIUM
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    cost_estimate: float = 0.0  # Estimated execution cost/time
    rate_limit: Optional[Dict[str, Any]] = None
    
class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, name: str, spec: ToolSpec):
        self.name = name
        self.spec = spec
        self.usage_count = 0
        self.total_execution_time = 0.0
        self.last_used = None
        self.error_count = 0
        
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate input parameters against spec"""
        for param_name, param_spec in self.spec.parameters.items():
            if param_spec.get('required', False) and param_name not in kwargs:
                return False
        return True
    
    def update_stats(self, execution_time: float, success: bool):
        """Update tool usage statistics"""
        self.usage_count += 1
        self.total_execution_time += execution_time
        self.last_used = datetime.now()
        if not success:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool performance statistics"""
        avg_time = self.total_execution_time / max(self.usage_count, 1)
        error_rate = self.error_count / max(self.usage_count, 1)
        
        return {
            'usage_count': self.usage_count,
            'average_execution_time': avg_time,
            'error_rate': error_rate,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

class CalculatorTool(BaseTool):
    """Mathematical computation tool"""
    
    def __init__(self):
        spec = ToolSpec(
            name="calculator",
            description="Perform mathematical calculations and evaluations",
            category=ToolCategory.COMPUTATION,
            parameters={
                "expression": {"type": "string", "required": True, "description": "Mathematical expression to evaluate"},
                "precision": {"type": "integer", "default": 10, "description": "Decimal precision for results"}
            },
            returns={"type": "number", "description": "Result of mathematical computation"},
            examples=[
                {"input": {"expression": "2 + 3 * 4"}, "output": 14},
                {"input": {"expression": "sqrt(16)"}, "output": 4.0}
            ]
        )
        super().__init__("calculator", spec)
    
    async def execute(self, expression: str, precision: int = 10, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            # Sanitize expression for safety
            if not self._is_safe_expression(expression):
                raise ValueError("Unsafe mathematical expression")
            
            # Evaluate using safe eval with math functions
            import math
            safe_dict = {
                '__builtins__': {},
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e
            }
            
            result = eval(expression, safe_dict)
            
            if isinstance(result, float):
                result = round(result, precision)
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return ToolResult(
                success=True,
                data=result,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"expression": expression, "precision": precision}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )
    
    def _is_safe_expression(self, expression: str) -> bool:
        """Check if mathematical expression is safe to evaluate"""
        forbidden = ['import', '__', 'exec', 'eval', 'open', 'file']
        return not any(word in expression.lower() for word in forbidden)

class WebSearchTool(BaseTool):
    """Web search and information retrieval tool"""
    
    def __init__(self, api_key: Optional[str] = None):
        spec = ToolSpec(
            name="web_search",
            description="Search the web for current information",
            category=ToolCategory.SEARCH,
            priority=ToolPriority.HIGH,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "num_results": {"type": "integer", "default": 5, "description": "Number of results to return"},
                "time_filter": {"type": "string", "default": "any", "description": "Time filter: any, day, week, month, year"}
            },
            returns={"type": "array", "description": "List of search results with titles, URLs, and snippets"},
            cost_estimate=0.01,
            rate_limit={"requests_per_minute": 60}
        )
        super().__init__("web_search", spec)
        self.api_key = api_key
    
    async def execute(self, query: str, num_results: int = 5, time_filter: str = "any", **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            # Simulate web search (replace with actual API)
            results = await self._perform_search(query, num_results, time_filter)
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return ToolResult(
                success=True,
                data=results,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"query": query, "num_results": len(results)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            
            return ToolResult(
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )
    
    async def _perform_search(self, query: str, num_results: int, time_filter: str) -> List[Dict[str, Any]]:
        """Perform actual web search (mock implementation)"""
        # Mock search results - replace with real search API
        mock_results = [
            {
                "title": f"Search result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a relevant snippet for query '{query}' from result {i+1}",
                "timestamp": datetime.now().isoformat()
            }
            for i in range(min(num_results, 5))
        ]
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        return mock_results

class CodeExecutorTool(BaseTool):
    """Safe code execution tool"""
    
    def __init__(self):
        spec = ToolSpec(
            name="code_executor",
            description="Execute code snippets safely in isolated environment",
            category=ToolCategory.COMPUTATION,
            priority=ToolPriority.MEDIUM,
            parameters={
                "code": {"type": "string", "required": True, "description": "Code to execute"},
                "language": {"type": "string", "default": "python", "description": "Programming language"},
                "timeout": {"type": "number", "default": 10.0, "description": "Execution timeout in seconds"}
            },
            returns={"type": "object", "description": "Execution result with output and any errors"}
        )
        super().__init__("code_executor", spec)
    
    async def execute(self, code: str, language: str = "python", timeout: float = 10.0, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            if language.lower() != "python":
                raise ValueError(f"Language '{language}' not supported")
            
            # Safety checks
            if not self._is_safe_code(code):
                raise ValueError("Code contains unsafe operations")
            
            # Execute in restricted environment
            result = await self._execute_python_code(code, timeout)
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return ToolResult(
                success=True,
                data=result,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"language": language, "code_length": len(code)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if code is safe to execute"""
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            'open(', 'file(', 'exec(', 'eval(',
            '__import__', 'globals(', 'locals(',
            'input(', 'raw_input('
        ]
        
        code_lower = code.lower()
        return not any(pattern in code_lower for pattern in dangerous_patterns)
    
    async def _execute_python_code(self, code: str, timeout: float) -> Dict[str, Any]:
        """Execute Python code safely"""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Restricted globals
        safe_globals = {
            '__builtins__': {
                'print': print, 'len': len, 'range': range, 'str': str,
                'int': int, 'float': float, 'bool': bool, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set,
                'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum
            }
        }
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Use exec with restricted environment
                exec(code, safe_globals)
            
            return {
                'output': stdout_capture.getvalue(),
                'error': stderr_capture.getvalue(),
                'success': True
            }
            
        except Exception as e:
            return {
                'output': stdout_capture.getvalue(),
                'error': f"{stderr_capture.getvalue()}\n{str(e)}",
                'success': False
            }

class DataAnalysisTool(BaseTool):
    """Data analysis and visualization tool"""
    
    def __init__(self):
        spec = ToolSpec(
            name="data_analysis",
            description="Analyze datasets and generate insights",
            category=ToolCategory.ANALYSIS,
            parameters={
                "data": {"type": "array", "required": True, "description": "Dataset to analyze"},
                "analysis_type": {"type": "string", "default": "summary", "description": "Type of analysis: summary, correlation, trend"},
                "columns": {"type": "array", "default": [], "description": "Specific columns to analyze"}
            },
            returns={"type": "object", "description": "Analysis results and insights"}
        )
        super().__init__("data_analysis", spec)
    
    async def execute(self, data: List[Dict[str, Any]], analysis_type: str = "summary", columns: List[str] = None, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            if not data:
                raise ValueError("No data provided for analysis")
            
            result = {}
            
            if analysis_type == "summary":
                result = self._generate_summary(data, columns)
            elif analysis_type == "correlation":
                result = self._analyze_correlations(data, columns)
            elif analysis_type == "trend":
                result = self._analyze_trends(data, columns)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return ToolResult(
                success=True,
                data=result,
                execution_time=execution_time,
                tool_name=self.name,
                metadata={"analysis_type": analysis_type, "data_size": len(data)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_stats(execution_time, False)
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.name
            )
    
    def _generate_summary(self, data: List[Dict[str, Any]], columns: List[str] = None) -> Dict[str, Any]:
        """Generate statistical summary of data"""
        if not data:
            return {}
        
        # Get all keys if columns not specified
        if not columns:
            columns = list(data[0].keys())
        
        summary = {
            "total_records": len(data),
            "columns_analyzed": columns,
            "column_stats": {}
        }
        
        for col in columns:
            values = [row.get(col) for row in data if col in row and row[col] is not None]
            
            if not values:
                continue
            
            # Determine if numeric
            numeric_values = []
            for val in values:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    pass
            
            col_stats = {
                "count": len(values),
                "unique_count": len(set(str(v) for v in values))
            }
            
            if numeric_values:
                col_stats.update({
                    "mean": sum(numeric_values) / len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "std": np.std(numeric_values) if len(numeric_values) > 1 else 0
                })
            
            summary["column_stats"][col] = col_stats
        
        return summary
    
    def _analyze_correlations(self, data: List[Dict[str, Any]], columns: List[str] = None) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        # Simplified correlation analysis
        return {"correlations": "Correlation analysis would be implemented here"}
    
    def _analyze_trends(self, data: List[Dict[str, Any]], columns: List[str] = None) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        # Simplified trend analysis
        return {"trends": "Trend analysis would be implemented here"}

class ToolOrchestrator:
    """Manages tool selection, execution, and coordination"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_embeddings: Dict[str, np.ndarray] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.parallel_executor = None
        
        # Initialize default tools
        self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """Initialize commonly used tools"""
        self.register_tool(CalculatorTool())
        self.register_tool(WebSearchTool())
        self.register_tool(CodeExecutorTool())
        self.register_tool(DataAnalysisTool())
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        
        # Generate embedding for tool selection
        tool_description = f"{tool.spec.description} {' '.join(tool.spec.parameters.keys())}"
        # In practice, use a real embedding model
        self.tool_embeddings[tool.name] = np.random.random(384)  # Mock embedding
    
    def get_available_tools(self, category: ToolCategory = None) -> List[str]:
        """Get list of available tools, optionally filtered by category"""
        if category is None:
            return list(self.tools.keys())
        
        return [
            name for name, tool in self.tools.items()
            if tool.spec.category == category
        ]
    
    async def select_tools(self, context: str, max_tools: int = 3) -> List[str]:
        """Intelligently select tools based on context"""
        if not self.tools:
            return []
        
        # Generate context embedding (mock)
        context_embedding = np.random.random(384)
        
        # Calculate similarity scores
        tool_scores = []
        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = np.dot(context_embedding, tool_embedding) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(tool_embedding)
            )
            
            # Boost score based on tool priority and success rate
            tool = self.tools[tool_name]
            stats = tool.get_stats()
            success_rate = 1.0 - stats['error_rate']
            priority_boost = tool.spec.priority.value / 4.0
            
            final_score = similarity * 0.6 + success_rate * 0.3 + priority_boost * 0.1
            tool_scores.append((final_score, tool_name))
        
        # Sort by score and return top tools
        tool_scores.sort(reverse=True)
        selected_tools = [tool_name for _, tool_name in tool_scores[:max_tools]]
        
        return selected_tools
    
    async def execute_tool(self, tool_name: str, **parameters) -> ToolResult:
        """Execute a specific tool with given parameters"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name
            )
        
        tool = self.tools[tool_name]
        
        # Validate parameters
        if not tool.validate_parameters(**parameters):
            return ToolResult(
                success=False,
                data=None,
                error="Invalid parameters provided",
                tool_name=tool_name
            )
        
        # Execute tool
        result = await tool.execute(**parameters)
        
        # Log execution
        self.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'tool_name': tool_name,
            'parameters': parameters,
            'success': result.success,
            'execution_time': result.execution_time
        })
        
        return result
    
    async def execute_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tools in parallel"""
        tasks = []
        for call in tool_calls:
            tool_name = call['tool']
            parameters = call.get('parameters', {})
            task = self.execute_tool(tool_name, **parameters)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ToolResult(
                    success=False,
                    data=None,
                    error=str(result),
                    tool_name=tool_calls[i]['tool']
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics for all tools"""
        stats = {}
        for tool_name, tool in self.tools.items():
            stats[tool_name] = tool.get_stats()
        
        return stats
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]

class ContextualToolSelector(nn.Module):
    """Neural tool selection based on context"""
    
    def __init__(self, context_dim: int = 768, tool_embed_dim: int = 128, max_tools: int = 50):
        super().__init__()
        self.context_dim = context_dim
        self.tool_embed_dim = tool_embed_dim
        self.max_tools = max_tools
        
        # Tool embeddings
        self.tool_embeddings = nn.Embedding(max_tools, tool_embed_dim)
        
        # Context-tool attention
        self.context_projector = nn.Linear(context_dim, tool_embed_dim)
        self.tool_scorer = nn.Sequential(
            nn.Linear(tool_embed_dim * 2, tool_embed_dim),
            nn.ReLU(),
            nn.Linear(tool_embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Multi-tool selection
        self.selector = nn.Sequential(
            nn.Linear(context_dim + tool_embed_dim, context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, max_tools),
            nn.Sigmoid()
        )
    
    def forward(self, context_embedding: torch.Tensor, available_tools: torch.Tensor) -> torch.Tensor:
        """Select tools based on context"""
        batch_size = context_embedding.size(0)
        
        # Project context to tool space
        context_proj = self.context_projector(context_embedding)
        
        # Get tool embeddings
        tool_embeds = self.tool_embeddings(available_tools)
        
        # Calculate tool relevance scores
        context_expanded = context_proj.unsqueeze(1).expand(-1, available_tools.size(1), -1)
        combined = torch.cat([context_expanded, tool_embeds], dim=-1)
        tool_scores = self.tool_scorer(combined).squeeze(-1)
        
        # Global tool selection
        pooled_tools = torch.mean(tool_embeds, dim=1)
        selection_input = torch.cat([context_embedding, pooled_tools], dim=-1)
        tool_probabilities = self.selector(selection_input)
        
        return tool_scores, tool_probabilities

# Example usage and integration
def create_orac_tools() -> ToolOrchestrator:
    """Factory function to create ORAC tool system"""
    orchestrator = ToolOrchestrator()
    
    # Add any custom tools here
    # orchestrator.register_tool(CustomTool())
    
    return orchestrator

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create tool orchestrator
        tools = create_orac_tools()
        
        # Select tools for a context
        selected = await tools.select_tools("I need to calculate the fibonacci sequence and search for recent AI news")
        print("Selected tools:", selected)
        
        # Execute a calculation
        calc_result = await tools.execute_tool("calculator", expression="2**10")
        print("Calculation result:", calc_result.data)
        
        # Execute multiple tools in parallel
        parallel_calls = [
            {"tool": "calculator", "parameters": {"expression": "sqrt(144)"}},
            {"tool": "web_search", "parameters": {"query": "machine learning news", "num_results": 3}}
        ]
        
        parallel_results = await tools.execute_parallel(parallel_calls)
        print("Parallel execution results:", [r.success for r in parallel_results])
        
        # Get tool statistics
        stats = tools.get_tool_stats()
        print("Tool stats:", stats)
    
    # Run example
    asyncio.run(main())
