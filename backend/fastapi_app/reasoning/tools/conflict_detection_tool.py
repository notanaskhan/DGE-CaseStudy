"""
Conflict Detection Tool for ReAct framework.
Wraps existing validation service for document-form consistency checking.
"""

import json
from typing import Dict, Any, List

from ..base_tools import BaseTool, ToolResult
from ...services.validation_service import validate_application


class ConflictDetectionTool(BaseTool):
    """Tool for detecting conflicts between form data and uploaded documents."""

    def __init__(self):
        super().__init__(
            name="conflict_detection",
            description="Detect conflicts between declared form data and document analysis"
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_row": {
                    "type": "object",
                    "description": "Application form data",
                    "properties": {
                        "declared_monthly_income": {"type": "number"},
                        "household_size": {"type": "integer"}
                    },
                    "required": ["declared_monthly_income", "household_size"]
                },
                "doc_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document file paths to analyze"
                }
            },
            "required": ["app_row", "doc_paths"]
        }

    def validate_parameters(self, **kwargs) -> bool:
        app_row = kwargs.get("app_row")
        doc_paths = kwargs.get("doc_paths")

        if not app_row or not isinstance(app_row, dict):
            return False

        if not doc_paths or not isinstance(doc_paths, list):
            return False

        required_fields = ["declared_monthly_income", "household_size"]
        return all(field in app_row for field in required_fields)

    def execute(self, **kwargs) -> ToolResult:
        try:
            app_row = kwargs["app_row"]
            doc_paths = kwargs["doc_paths"]

            # Call existing validation service
            validation_result = validate_application(app_row, doc_paths)

            # Extract conflict information
            conflicts = []
            flags = validation_result.get("flags", {})

            if isinstance(flags, dict):
                for flag_name, flag_value in flags.items():
                    if not flag_value:  # False flags indicate conflicts
                        conflicts.append(flag_name)

            # Extract key metrics
            declared_income = app_row.get("declared_monthly_income", 0)
            analyzed_income = validation_result.get("analyzed_monthly_income", 0)
            income_discrepancy = abs(declared_income - analyzed_income)

            # Create summary
            conflict_summary = {
                "has_conflicts": len(conflicts) > 0,
                "conflict_types": conflicts,
                "income_discrepancy": income_discrepancy,
                "discrepancy_percentage": (
                    (income_discrepancy / declared_income * 100)
                    if declared_income > 0 else 0
                ),
                "validation_details": validation_result
            }

            # Generate human-readable message
            if conflicts:
                conflict_list = ", ".join(conflicts)
                message = (
                    f"Conflicts detected: {conflict_list}. "
                    f"Income discrepancy: AED {income_discrepancy:.2f} "
                    f"({conflict_summary['discrepancy_percentage']:.1f}%)"
                )
            else:
                message = "No major conflicts detected between form data and documents"

            return ToolResult(
                success=True,
                data=conflict_summary,
                message=message,
                tool_name=self.name
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Conflict detection failed: {str(e)}",
                tool_name=self.name
            )