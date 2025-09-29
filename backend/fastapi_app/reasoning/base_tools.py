"""
Base classes for ReAct tools and tool results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any
    message: str
    tool_name: str
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time
        }

    def to_observation_string(self) -> str:
        """Convert to string for ReAct observation."""
        if self.success:
            return f"Tool '{self.tool_name}' executed successfully: {self.message}"
        else:
            return f"Tool '{self.tool_name}' failed: {self.message}"


class BaseTool(ABC):
    """Base class for all ReAct tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM understanding."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema for this tool."""
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """Validate tool parameters before execution."""
        return True