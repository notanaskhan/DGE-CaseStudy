"""
Policy Lookup Tool for ReAct framework.
Simple policy rule lookup for eligibility determination.
"""

from typing import Dict, Any

from ..base_tools import BaseTool, ToolResult


class PolicyLookupTool(BaseTool):
    """Tool for looking up specific policy rules and thresholds."""

    def __init__(self):
        super().__init__(
            name="policy_rules",
            description="Look up specific UAE social support policy rules and thresholds"
        )

        # Basic policy rules (in production, this would come from a database or policy service)
        self.policy_rules = {
            "income_thresholds": {
                "individual_max": 10000,  # AED per month
                "family_max": 15000,      # AED per month
                "household_multiplier": 1.2,  # Per additional member
            },
            "eligibility_criteria": {
                "min_age": 18,
                "valid_emirates_id": True,
                "residency_months_required": 12,
                "employment_consideration": True
            },
            "benefit_amounts": {
                "basic_assistance": 1500,   # AED per month
                "family_supplement": 500,   # AED per additional member
                "max_benefit": 5000,        # AED per month
            },
            "documentation_requirements": {
                "emirates_id": "mandatory",
                "bank_statements": "3_months_minimum",
                "employment_proof": "required_if_employed",
                "housing_proof": "utility_bills_or_lease"
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rule_category": {
                    "type": "string",
                    "enum": ["income_thresholds", "eligibility_criteria", "benefit_amounts", "documentation_requirements"],
                    "description": "Category of policy rule to lookup"
                },
                "specific_rule": {
                    "type": "string",
                    "description": "Specific rule within the category (optional)",
                    "default": None
                }
            },
            "required": ["rule_category"]
        }

    def validate_parameters(self, **kwargs) -> bool:
        rule_category = kwargs.get("rule_category")
        return rule_category in self.policy_rules

    def execute(self, **kwargs) -> ToolResult:
        try:
            rule_category = kwargs["rule_category"]
            specific_rule = kwargs.get("specific_rule")

            category_rules = self.policy_rules[rule_category]

            if specific_rule:
                if specific_rule in category_rules:
                    result = {specific_rule: category_rules[specific_rule]}
                    message = f"Policy rule {rule_category}.{specific_rule}: {category_rules[specific_rule]}"
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        message=f"Rule '{specific_rule}' not found in category '{rule_category}'",
                        tool_name=self.name
                    )
            else:
                result = category_rules
                rule_count = len(category_rules)
                message = f"Retrieved {rule_count} rules from category '{rule_category}'"

            return ToolResult(
                success=True,
                data={
                    "category": rule_category,
                    "rules": result,
                    "lookup_type": "specific" if specific_rule else "category"
                },
                message=message,
                tool_name=self.name
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Policy lookup failed: {str(e)}",
                tool_name=self.name
            )