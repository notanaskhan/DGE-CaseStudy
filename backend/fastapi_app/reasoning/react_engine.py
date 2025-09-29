"""
ReAct reasoning engine implementation.
Core ReAct loop with Thought-Action-Observation pattern.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

from .tool_registry import get_tool_registry, ToolResult
from .reasoning_tracer import ReasoningTracer

logger = logging.getLogger(__name__)


@dataclass
class ReActResult:
    """Result of ReAct reasoning process."""
    success: bool
    final_answer: str
    reasoning_trace: Dict[str, Any]
    total_iterations: int
    execution_time: float
    error_message: Optional[str] = None


class ReActEngine:
    """Core ReAct reasoning engine using Thought-Action-Observation pattern."""

    def __init__(self, max_iterations: int = 10, timeout_seconds: int = 300):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.tool_registry = get_tool_registry()

    def run(self, task: str, context: Dict[str, Any], application_id: str,
            task_type: str = "decision_making") -> ReActResult:
        """
        Run ReAct reasoning loop.

        Args:
            task: The task to reason about
            context: Context information for the task
            application_id: Application ID for audit trails
            task_type: Type of task for categorization
        """
        start_time = time.time()
        tracer = ReasoningTracer(application_id, task_type)

        try:
            # Initialize reasoning
            initial_prompt = self._build_initial_prompt(task, context)

            for iteration in range(self.max_iterations):
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    error_msg = f"ReAct reasoning timed out after {self.timeout_seconds} seconds"
                    logger.warning(error_msg)
                    return ReActResult(
                        success=False,
                        final_answer="Reasoning timed out",
                        reasoning_trace=tracer.get_reasoning_trace(),
                        total_iterations=iteration,
                        execution_time=time.time() - start_time,
                        error_message=error_msg
                    )

                # Generate thought (simulate LLM reasoning)
                thought = self._generate_thought(iteration, context, tracer.steps)

                # Check if this is a final answer
                if self._is_final_answer(thought):
                    final_answer = self._extract_final_answer(thought)
                    tracer.add_step(iteration, thought)
                    tracer.save_to_mongo()

                    return ReActResult(
                        success=True,
                        final_answer=final_answer,
                        reasoning_trace=tracer.get_reasoning_trace(),
                        total_iterations=iteration + 1,
                        execution_time=time.time() - start_time
                    )

                # Parse action from thought
                action, action_input = self._parse_action(thought)

                if action is None:
                    # No action found, add thought and continue
                    tracer.add_step(iteration, thought)
                    continue

                # Execute action
                tool_result = self._execute_action(action, action_input)
                observation = tool_result.to_observation_string()

                # Add step to trace
                tracer.add_step(iteration, thought, action, action_input, observation)

                # Update context with observation
                context = self._update_context(context, tool_result)

            # Max iterations reached
            error_msg = f"ReAct reasoning reached maximum iterations ({self.max_iterations})"
            logger.warning(error_msg)
            tracer.save_to_mongo()

            return ReActResult(
                success=False,
                final_answer="Unable to reach conclusion within iteration limit",
                reasoning_trace=tracer.get_reasoning_trace(),
                total_iterations=self.max_iterations,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"ReAct reasoning failed with error: {str(e)}"
            logger.error(error_msg)
            tracer.save_to_mongo()

            return ReActResult(
                success=False,
                final_answer="Reasoning failed due to error",
                reasoning_trace=tracer.get_reasoning_trace(),
                total_iterations=0,
                execution_time=time.time() - start_time,
                error_message=error_msg
            )

    def _build_initial_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Build initial prompt for reasoning."""
        tools_description = self.tool_registry.get_available_tools_description()

        return f'''You are a ReAct agent helping with social support decision making.

Task: {task}

Context: {json.dumps(context, indent=2)}

Available tools:
{tools_description}

Instructions:
1. Think step by step about the task
2. Use available tools to gather information and make decisions
3. Provide clear reasoning for each step
4. When you have enough information, provide a final answer

Format your response as:
Thought: [your reasoning]
Action: [tool_name]
Action Input: {{"parameter": "value"}}

Or when ready to conclude:
Thought: [final reasoning]
Final Answer: [your conclusion]
'''

    def _generate_thought(self, iteration: int, context: Dict[str, Any],
                         previous_steps: List) -> str:
        """
        Generate thought for current iteration.
        This is a simplified version - in a full implementation,
        this would use an LLM like Ollama.
        """
        if iteration == 0:
            return "I need to analyze this application for social support eligibility. Let me start by checking the basic eligibility criteria."

        # For demo purposes, provide rule-based reasoning
        # In production, this would be replaced with LLM calls
        return self._get_demo_thought(iteration, context, previous_steps)

    def _get_demo_thought(self, iteration: int, context: Dict[str, Any],
                         previous_steps: List) -> str:
        """Generate demo thoughts for testing (replace with LLM in production)."""
        if iteration == 1:
            return "Let me check the ML scoring for this application to get a baseline assessment."
        elif iteration == 2:
            return "Now I should verify if there are any conflicts or red flags in the application data."
        elif iteration == 3:
            return "Based on the ML score and conflict analysis, I can now make a final eligibility determination."
        else:
            return "Final Answer: Based on my analysis, I have reached a conclusion."

    def _is_final_answer(self, thought: str) -> bool:
        """Check if thought contains a final answer."""
        return "final answer:" in thought.lower()

    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from thought."""
        match = re.search(r'final answer:(.+?)(?:\\n|$)', thought, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return thought

    def _parse_action(self, thought: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse action and action input from thought."""
        action_match = re.search(r'action:\\s*(.+?)(?:\\n|$)', thought, re.IGNORECASE)
        if not action_match:
            return None, None

        action = action_match.group(1).strip()

        # Parse action input
        input_match = re.search(r'action input:\\s*({.+?})', thought, re.IGNORECASE | re.DOTALL)
        action_input = {}
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse action input: {input_match.group(1)}")

        return action, action_input

    def _execute_action(self, action: str, action_input: Dict[str, Any]) -> ToolResult:
        """Execute action using tool registry."""
        start_time = time.time()
        result = self.tool_registry.execute_tool(action, **action_input)
        result.execution_time = time.time() - start_time
        return result

    def _update_context(self, context: Dict[str, Any], tool_result: ToolResult) -> Dict[str, Any]:
        """Update context with tool result."""
        # Create new context dict to avoid mutations
        new_context = context.copy()

        # Add tool results to context
        if 'tool_results' not in new_context:
            new_context['tool_results'] = []

        new_context['tool_results'].append(tool_result.to_dict())

        # Add specific tool data to context
        if tool_result.success and tool_result.data:
            tool_key = f"{tool_result.tool_name}_result"
            new_context[tool_key] = tool_result.data

        return new_context