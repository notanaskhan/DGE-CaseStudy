"""
Configuration for ReAct reasoning framework.
Provides settings and controls for ReAct integration.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ReActConfig:
    """Configuration settings for ReAct reasoning."""
    enabled: bool = True
    max_iterations: int = 5
    timeout_seconds: int = 120
    use_for_complex_cases_only: bool = False
    debug_mode: bool = False
    save_traces: bool = True


# Global configuration instance
_react_config = ReActConfig()


def get_react_config() -> ReActConfig:
    """Get current ReAct configuration."""
    return _react_config


def update_react_config(**kwargs) -> ReActConfig:
    """
    Update ReAct configuration settings.

    Args:
        **kwargs: Configuration parameters to update

    Returns:
        Updated configuration
    """
    global _react_config

    for key, value in kwargs.items():
        if hasattr(_react_config, key):
            setattr(_react_config, key, value)

    return _react_config


def is_react_enabled() -> bool:
    """Check if ReAct reasoning is enabled."""
    return _react_config.enabled


def load_config_from_env():
    """Load configuration from environment variables."""
    config_updates = {}

    # Check environment variables
    if os.getenv("REACT_ENABLED") is not None:
        config_updates["enabled"] = os.getenv("REACT_ENABLED", "true").lower() == "true"

    if os.getenv("REACT_MAX_ITERATIONS") is not None:
        try:
            config_updates["max_iterations"] = int(os.getenv("REACT_MAX_ITERATIONS", "5"))
        except ValueError:
            pass

    if os.getenv("REACT_TIMEOUT") is not None:
        try:
            config_updates["timeout_seconds"] = int(os.getenv("REACT_TIMEOUT", "120"))
        except ValueError:
            pass

    if os.getenv("REACT_COMPLEX_ONLY") is not None:
        config_updates["use_for_complex_cases_only"] = os.getenv("REACT_COMPLEX_ONLY", "false").lower() == "true"

    if os.getenv("REACT_DEBUG") is not None:
        config_updates["debug_mode"] = os.getenv("REACT_DEBUG", "false").lower() == "true"

    if config_updates:
        update_react_config(**config_updates)


def get_config_dict() -> Dict[str, Any]:
    """Get configuration as dictionary."""
    return {
        "enabled": _react_config.enabled,
        "max_iterations": _react_config.max_iterations,
        "timeout_seconds": _react_config.timeout_seconds,
        "use_for_complex_cases_only": _react_config.use_for_complex_cases_only,
        "debug_mode": _react_config.debug_mode,
        "save_traces": _react_config.save_traces
    }


# Load configuration from environment on import
load_config_from_env()