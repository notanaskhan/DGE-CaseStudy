"""
Tool initializer for ReAct framework.
Registers all available tools with the global tool registry.
"""

import logging
from typing import List, Dict, Any

from ..tool_registry import get_tool_registry, register_tool
from .ml_scoring_tool import MLScoringTool
from .conflict_detection_tool import ConflictDetectionTool
from .rag_tool import RAGTool
from .policy_lookup_tool import PolicyLookupTool

logger = logging.getLogger(__name__)


def initialize_tools() -> bool:
    """
    Initialize and register all ReAct tools.

    Returns:
        bool: True if all tools were registered successfully
    """
    tools_to_register = [
        (MLScoringTool(), "ml_analysis"),
        (ConflictDetectionTool(), "validation"),
        (RAGTool(), "knowledge"),
        (PolicyLookupTool(), "policy"),
    ]

    success_count = 0
    total_tools = len(tools_to_register)

    for tool, category in tools_to_register:
        try:
            if register_tool(tool, category):
                logger.info(f"Successfully registered tool: {tool.name}")
                success_count += 1
            else:
                logger.warning(f"Failed to register tool: {tool.name}")
        except Exception as e:
            logger.error(f"Error registering tool {tool.name}: {e}")

    logger.info(f"Tool registration complete: {success_count}/{total_tools} tools registered")
    return success_count == total_tools


def get_available_tools() -> List[str]:
    """Get list of all available tool names."""
    registry = get_tool_registry()
    return registry.list_tools()


def get_tools_by_category(category: str) -> List[str]:
    """Get list of tools in a specific category."""
    registry = get_tool_registry()
    return registry.list_tools(category)


def get_tool_summary() -> Dict[str, Any]:
    """Get summary of all registered tools."""
    registry = get_tool_registry()

    summary = {
        "total_tools": len(registry.list_tools()),
        "tools_by_category": {},
        "tool_descriptions": {}
    }

    for category in ["ml_analysis", "validation", "knowledge", "policy"]:
        tools = registry.list_tools(category)
        summary["tools_by_category"][category] = tools

        for tool_name in tools:
            tool = registry.get_tool(tool_name)
            if tool:
                summary["tool_descriptions"][tool_name] = tool.description

    return summary


# Auto-initialize tools when module is imported
try:
    if not initialize_tools():
        logger.warning("Some tools failed to initialize")
except Exception as e:
    logger.error(f"Tool initialization failed: {e}")