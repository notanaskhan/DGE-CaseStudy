"""
Tool registry for managing ReAct tools.
Centralized registry for all available tools and their capabilities.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from .base_tools import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing all available ReAct tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool: BaseTool, category: str = "general") -> bool:
        """Register a tool in the registry."""
        try:
            if tool.name in self._tools:
                logger.warning(f"Tool '{tool.name}' already registered, overwriting")

            self._tools[tool.name] = tool
            self._tool_metadata[tool.name] = {
                "category": category,
                "schema": tool.get_schema(),
                "registered_at": __import__('datetime').datetime.utcnow()
            }

            logger.info(f"Registered tool: {tool.name} in category: {category}")
            return True
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
            return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with parameters."""
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool '{name}' not found",
                tool_name=name
            )

        try:
            if not tool.validate_parameters(**kwargs):
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Invalid parameters for tool '{name}'",
                    tool_name=name
                )

            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool execution failed: {str(e)}",
                tool_name=name
            )

    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List all registered tools, optionally filtered by category."""
        if category is None:
            return list(self._tools.keys())

        return [
            name for name, metadata in self._tool_metadata.items()
            if metadata.get("category") == category
        ]

    def get_tool_schemas(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get schemas for all tools, optionally filtered by category."""
        tools = self.list_tools(category)
        return [self._tool_metadata[name]["schema"] for name in tools]

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        if name not in self._tools:
            return None

        return {
            "name": name,
            "tool": self._tools[name],
            "metadata": self._tool_metadata[name]
        }

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            del self._tool_metadata[name]
            logger.info(f"Removed tool: {name}")
            return True
        return False

    def get_available_tools_description(self) -> str:
        """Get a formatted description of all available tools for LLM."""
        if not self._tools:
            return "No tools available."

        descriptions = []
        for name, tool in self._tools.items():
            category = self._tool_metadata[name].get("category", "general")
            descriptions.append(f"- {name} ({category}): {tool.description}")

        return "Available tools:\\n" + "\\n".join(descriptions)


# Global tool registry instance
_global_registry = ToolRegistry()

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry

def register_tool(tool: BaseTool, category: str = "general") -> bool:
    """Register a tool in the global registry."""
    return _global_registry.register_tool(tool, category)

def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool from the global registry."""
    return _global_registry.execute_tool(name, **kwargs)