"""
ReAct integration with the existing LangGraph orchestrator.
Provides optional ReAct reasoning for enhanced decision transparency.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from ..main import SessionLocal, Application
from .react_engine import ReActEngine, ReActResult
from .tools.tool_initializer import initialize_tools
from .react_config import is_react_enabled

logger = logging.getLogger(__name__)


def react_eligibility_node(state: Dict[str, Any], use_react: bool = True) -> Dict[str, Any]:
    """
    Enhanced eligibility node with optional ReAct reasoning.

    Args:
        state: LangGraph orchestrator state
        use_react: Whether to use ReAct reasoning (default: True)

    Returns:
        Updated state with eligibility decision
    """
    app_id = state["app_id"]

    # Initialize tools if not already done
    try:
        initialize_tools()
    except Exception as e:
        logger.warning(f"Tool initialization failed: {e}")

    # Check if ReAct is globally enabled and should be used for this case
    should_use_react = (use_react and
                       is_react_enabled() and
                       _should_use_react(state))

    if should_use_react:
        return _process_with_react(state)
    else:
        # Fallback to existing ML-only approach
        return _process_with_ml_only(state)


def _should_use_react(state: Dict[str, Any]) -> bool:
    """
    Determine if ReAct reasoning should be used based on application complexity.

    Args:
        state: Current orchestrator state

    Returns:
        bool: True if ReAct should be used
    """
    # Use ReAct for complex cases that would benefit from step-by-step reasoning
    validation = state.get("validation", {})

    # Check for conflict flags or complex validation issues
    flags = validation.get("flags", {})
    if isinstance(flags, dict):
        # Use ReAct if there are any validation conflicts
        has_conflicts = any(not flag_value for flag_value in flags.values())
        if has_conflicts:
            return True

    # Use ReAct for high-income cases requiring detailed analysis
    app_id = state["app_id"]
    db = SessionLocal()
    try:
        app_row = db.query(Application).filter(Application.id == app_id).first()
        if app_row and app_row.declared_monthly_income > 8000:  # High income cases
            return True
    except Exception:
        pass
    finally:
        db.close()

    # Default: use ReAct for enhanced transparency
    return True


def _process_with_react(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process eligibility using ReAct reasoning framework.

    Args:
        state: Current orchestrator state

    Returns:
        Updated state with ReAct reasoning results
    """
    app_id = state["app_id"]
    db = SessionLocal()

    try:
        # Get application data
        app_row = db.query(Application).filter(Application.id == app_id).first()

        # Prepare context for ReAct reasoning
        context = {
            "application_id": app_id,
            "application_data": {
                "app_row": {
                    "declared_monthly_income": app_row.declared_monthly_income,
                    "household_size": app_row.household_size,
                    "name": app_row.name,
                    "emirates_id": app_row.emirates_id
                },
                "validation": state.get("validation", {}),
                "extraction_summary": state.get("extraction_summary", {}),
                "ocr_results": state.get("ocr", {}),
                "parsed_results": state.get("parsed", {})
            }
        }

        # Define reasoning task
        task = (
            f"Determine eligibility for social support application {app_id}. "
            f"Analyze the application thoroughly using available tools and provide "
            f"a well-reasoned decision with supporting evidence."
        )

        # Initialize ReAct engine
        react_engine = ReActEngine(max_iterations=5, timeout_seconds=120)

        # Run ReAct reasoning
        react_result = react_engine.run(
            task=task,
            context=context,
            application_id=app_id,
            task_type="eligibility_determination"
        )

        # Extract decision from ReAct result
        decision, confidence, reason = _extract_decision_from_react(react_result)

        # Persist to database
        app_row.decision = decision
        app_row.decision_reason = reason
        app_row.status = "DECIDED"
        db.add(app_row)
        db.commit()

        # Save ReAct trace for audit
        _save_react_trace(app_id, react_result)

        # Update state
        state["eligibility"] = {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "reasoning_method": "react",
            "react_result": react_result.reasoning_trace,
            "success": react_result.success,
            "iterations": react_result.total_iterations
        }
        state["status"] = "DECIDED"

        # Add trace entry
        from ..agents.orchestrator import append_trace
        append_trace(app_id, {
            "step": "react_eligibility",
            "status": "ok" if react_result.success else "partial",
            "decision": state["eligibility"]
        })

        return state

    except Exception as e:
        logger.error(f"ReAct eligibility processing failed for {app_id}: {e}")
        # Fallback to ML-only approach
        return _process_with_ml_only(state)

    finally:
        db.close()


def _process_with_ml_only(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback to original ML-only processing.

    Args:
        state: Current orchestrator state

    Returns:
        Updated state with ML decision
    """
    # Import and call original eligibility logic
    from ..agents.orchestrator import eligibility_node
    return eligibility_node(state)


def _extract_decision_from_react(react_result: ReActResult) -> tuple[str, float, str]:
    """
    Extract decision, confidence, and reason from ReAct result.

    Args:
        react_result: Result from ReAct engine

    Returns:
        Tuple of (decision, confidence, reason)
    """
    if not react_result.success:
        return "REJECTED", 0.1, f"ReAct reasoning failed: {react_result.error_message}"

    final_answer = react_result.final_answer.lower()

    # Parse decision from final answer
    if "approved" in final_answer or "eligible" in final_answer or "accept" in final_answer:
        decision = "APPROVED"
        confidence = 0.8
    elif "rejected" in final_answer or "not eligible" in final_answer or "deny" in final_answer:
        decision = "REJECTED"
        confidence = 0.8
    else:
        # Default to rejected if unclear
        decision = "REJECTED"
        confidence = 0.3

    reason = (
        f"ReAct reasoning ({react_result.total_iterations} steps): "
        f"{react_result.final_answer[:200]}..."
    )

    return decision, confidence, reason


def _save_react_trace(app_id: str, react_result: ReActResult) -> None:
    """
    Save ReAct reasoning trace for audit purposes.

    Args:
        app_id: Application ID
        react_result: Result from ReAct engine
    """
    try:
        from ..main import APPS_DIR

        dest_dir = os.path.join(APPS_DIR, app_id)
        os.makedirs(dest_dir, exist_ok=True)

        trace_file = os.path.join(dest_dir, "react_reasoning_trace.json")
        with open(trace_file, "w") as f:
            json.dump({
                "success": react_result.success,
                "final_answer": react_result.final_answer,
                "total_iterations": react_result.total_iterations,
                "execution_time": react_result.execution_time,
                "reasoning_trace": react_result.reasoning_trace,
                "error_message": react_result.error_message
            }, f, indent=2)

        logger.info(f"ReAct trace saved for application {app_id}")

    except Exception as e:
        logger.warning(f"Failed to save ReAct trace for {app_id}: {e}")


def get_react_summary(app_id: str) -> Optional[Dict[str, Any]]:
    """
    Get ReAct reasoning summary for an application.

    Args:
        app_id: Application ID

    Returns:
        ReAct reasoning summary or None if not available
    """
    try:
        from ..main import APPS_DIR

        trace_file = os.path.join(APPS_DIR, app_id, "react_reasoning_trace.json")
        if os.path.exists(trace_file):
            with open(trace_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load ReAct trace for {app_id}: {e}")

    return None