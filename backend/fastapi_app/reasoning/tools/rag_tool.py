"""
RAG Tool for ReAct framework.
Wraps existing RAG service for policy knowledge retrieval.
"""

import json
from typing import Dict, Any, List

from ..base_tools import BaseTool, ToolResult


class RAGTool(BaseTool):
    """Tool for retrieving relevant policy information using RAG."""

    def __init__(self):
        super().__init__(
            name="policy_lookup",
            description="Retrieve relevant UAE social support policies and guidelines"
        )
        self.rag_service = None

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search for in the policy knowledge base"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (e.g., 'eligibility', 'documentation', 'benefits')",
                    "default": None
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }

    def validate_parameters(self, **kwargs) -> bool:
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return False

        limit = kwargs.get("limit", 3)
        if not isinstance(limit, int) or limit < 1 or limit > 10:
            return False

        return True

    def _initialize_rag_service(self):
        """Initialize RAG service if not already done."""
        if self.rag_service is None:
            try:
                from ...rag.rag_service import RAGService
                self.rag_service = RAGService()

                # Initialize with basic UAE social support knowledge if empty
                if not self.rag_service.is_initialized():
                    self._populate_basic_knowledge()
            except Exception:
                # RAG service not available, create mock service
                self.rag_service = self._create_mock_rag_service()

    def _create_mock_rag_service(self):
        """Create mock RAG service for basic policy lookup."""
        class MockRAGService:
            def __init__(self):
                self.knowledge_base = {
                    "eligibility": [
                        "UAE nationals and residents with valid Emirates ID are eligible",
                        "Monthly income must be below AED 10,000 for individual applicants",
                        "Household size affects eligibility thresholds",
                        "Employment status and family circumstances are considered"
                    ],
                    "documentation": [
                        "Valid Emirates ID required for all applicants",
                        "Bank statements for last 3-6 months needed",
                        "Salary certificates or employment documentation",
                        "Utility bills and housing documents for verification"
                    ],
                    "benefits": [
                        "Monthly financial assistance ranging from AED 1,000 to AED 5,000",
                        "Healthcare support and insurance coverage",
                        "Educational assistance for children",
                        "Job training and skill development programs"
                    ]
                }

            def search(self, query: str, category: str = None, limit: int = 3):
                results = []
                search_term = query.lower()

                categories_to_search = [category] if category else self.knowledge_base.keys()

                for cat in categories_to_search:
                    if cat in self.knowledge_base:
                        for item in self.knowledge_base[cat]:
                            if any(word in item.lower() for word in search_term.split()):
                                results.append({
                                    "content": item,
                                    "category": cat,
                                    "score": 0.8,
                                    "metadata": {"source": "basic_knowledge"}
                                })

                return results[:limit] if results else [{
                    "content": f"No specific policy information found for: {query}",
                    "category": "general",
                    "score": 0.1,
                    "metadata": {"source": "fallback"}
                }]

        return MockRAGService()

    def _populate_basic_knowledge(self):
        """Populate basic UAE social support knowledge."""
        basic_knowledge = [
            {
                "content": "UAE nationals and residents with valid Emirates ID are eligible for social support",
                "category": "eligibility",
                "metadata": {"source": "uae_guidelines", "priority": "high"}
            },
            {
                "content": "Monthly income threshold for eligibility is typically below AED 10,000",
                "category": "eligibility",
                "metadata": {"source": "income_guidelines", "priority": "high"}
            },
            {
                "content": "Required documents include Emirates ID, bank statements, and employment certificates",
                "category": "documentation",
                "metadata": {"source": "documentation_requirements", "priority": "high"}
            }
        ]

        try:
            for item in basic_knowledge:
                self.rag_service.add_knowledge_item(
                    content=item["content"],
                    category=item["category"],
                    metadata=item["metadata"]
                )
        except Exception:
            # Fail silently if knowledge population fails
            pass

    def execute(self, **kwargs) -> ToolResult:
        try:
            self._initialize_rag_service()

            query = kwargs["query"]
            category = kwargs.get("category")
            limit = kwargs.get("limit", 3)

            # Search knowledge base
            results = self.rag_service.search(query, category=category, limit=limit)

            if not results:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"No relevant policy information found for: {query}",
                    tool_name=self.name
                )

            # Format results for reasoning
            policy_info = []
            for result in results:
                policy_info.append({
                    "content": result.get("content", ""),
                    "category": result.get("category", "general"),
                    "relevance_score": result.get("score", 0.0)
                })

            # Create summary message
            top_result = results[0]
            message = (
                f"Found {len(results)} relevant policy items. "
                f"Most relevant: {top_result.get('content', '')[:100]}..."
            )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": policy_info,
                    "total_found": len(results)
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