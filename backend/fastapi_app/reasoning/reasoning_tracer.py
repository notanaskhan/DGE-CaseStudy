"""
Reasoning tracer for audit trails and transparency.
Records all ReAct reasoning steps for government accountability.
"""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from ..database.mongo_service import get_mongo_service


@dataclass
class ReasoningStep:
    """Single step in ReAct reasoning process."""
    step_id: str
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.step_id is None:
            self.step_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ReasoningTracer:
    """Traces and logs ReAct reasoning for audit trails."""

    def __init__(self, application_id: str, task_type: str):
        self.application_id = application_id
        self.task_type = task_type
        self.session_id = str(uuid.uuid4())
        self.steps: List[ReasoningStep] = []
        self.start_time = datetime.utcnow()
        self.mongo_service = get_mongo_service()

    def add_step(self, iteration: int, thought: str, action: str = None,
                 action_input: Dict[str, Any] = None, observation: str = None) -> ReasoningStep:
        """Add a reasoning step."""
        step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            iteration=iteration,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation
        )
        self.steps.append(step)
        return step

    def update_last_step(self, observation: str):
        """Update the last step with observation."""
        if self.steps:
            self.steps[-1].observation = observation

    def get_reasoning_trace(self) -> Dict[str, Any]:
        """Get complete reasoning trace."""
        return {
            "session_id": self.session_id,
            "application_id": self.application_id,
            "task_type": self.task_type,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "total_steps": len(self.steps),
            "steps": [step.to_dict() for step in self.steps]
        }

    def save_to_mongo(self) -> bool:
        """Save reasoning trace to MongoDB for audit."""
        try:
            trace_data = self.get_reasoning_trace()
            return self.mongo_service.store_analytics_data("reasoning_trace", trace_data)
        except Exception:
            # Fail silently if MongoDB is not available
            return False

    def get_summary(self) -> str:
        """Get human-readable summary of reasoning."""
        if not self.steps:
            return "No reasoning steps recorded."

        summary_lines = [
            f"ReAct Reasoning Summary for {self.task_type}",
            f"Application: {self.application_id}",
            f"Total Steps: {len(self.steps)}",
            f"Duration: {(datetime.utcnow() - self.start_time).total_seconds():.2f} seconds",
            "",
            "Reasoning Chain:"
        ]

        for i, step in enumerate(self.steps, 1):
            summary_lines.append(f"  Step {i}: {step.thought}")
            if step.action:
                summary_lines.append(f"    Action: {step.action}")
            if step.observation:
                summary_lines.append(f"    Result: {step.observation}")
            summary_lines.append("")

        return "\\n".join(summary_lines)

    def get_audit_log(self) -> Dict[str, Any]:
        """Get audit-ready log format for government compliance."""
        return {
            "audit_type": "reasoning_trace",
            "application_id": self.application_id,
            "session_id": self.session_id,
            "task_type": self.task_type,
            "timestamp": self.start_time.isoformat(),
            "reasoning_steps": len(self.steps),
            "trace_data": self.get_reasoning_trace(),
            "compliance_version": "1.0"
        }