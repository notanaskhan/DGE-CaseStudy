"""
ML Scoring Tool for ReAct framework.
Wraps existing ML scoring service for transparent decision making.
"""

import json
from typing import Dict, Any

from ..base_tools import BaseTool, ToolResult
from ...ml.scoring_service import score_application


class MLScoringTool(BaseTool):
    """Tool for ML-based eligibility scoring with SHAP explanations."""

    def __init__(self):
        super().__init__(
            name="ml_scoring",
            description="Score application eligibility using ML model with SHAP explanations"
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "application_data": {
                    "type": "object",
                    "description": "Application data including app_row and validation results",
                    "properties": {
                        "app_row": {
                            "type": "object",
                            "properties": {
                                "declared_monthly_income": {"type": "number"},
                                "household_size": {"type": "integer"}
                            },
                            "required": ["declared_monthly_income", "household_size"]
                        },
                        "validation": {
                            "type": "object",
                            "description": "Validation results from document analysis"
                        }
                    },
                    "required": ["app_row"]
                }
            },
            "required": ["application_data"]
        }

    def validate_parameters(self, **kwargs) -> bool:
        application_data = kwargs.get("application_data")
        if not application_data:
            return False

        app_row = application_data.get("app_row")
        if not app_row:
            return False

        required_fields = ["declared_monthly_income", "household_size"]
        return all(field in app_row for field in required_fields)

    def execute(self, **kwargs) -> ToolResult:
        try:
            application_data = kwargs["application_data"]

            # Call existing ML scoring service
            ml_result = score_application(application_data)

            # Extract key information for reasoning
            decision = ml_result.get("decision", "unknown")
            confidence = ml_result.get("confidence", 0.0)
            top_factors = ml_result.get("top_factors", [])

            # Create human-readable explanation
            factor_explanations = []
            for factor in top_factors[:3]:  # Top 3 factors
                feature = factor.get("feature", "unknown")
                value = factor.get("value", "unknown")
                impact = factor.get("impact", 0.0)
                impact_type = "positive" if impact > 0 else "negative"

                factor_explanations.append(
                    f"{feature}: {value} ({impact_type} impact: {abs(impact):.3f})"
                )

            explanation = (
                f"ML model predicts: {decision} with {confidence:.2f} confidence. "
                f"Key factors: {'; '.join(factor_explanations)}"
            )

            return ToolResult(
                success=True,
                data={
                    "decision": decision,
                    "confidence": confidence,
                    "explanation": explanation,
                    "top_factors": top_factors,
                    "full_result": ml_result
                },
                message=explanation,
                tool_name=self.name
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"ML scoring failed: {str(e)}",
                tool_name=self.name
            )