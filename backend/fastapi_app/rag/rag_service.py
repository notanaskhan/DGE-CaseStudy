"""
Handles knowledge base creation, embedding, and retrieval for UAE social support policies.
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import ollama
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge in the knowledge base."""
    id: str
    content: str
    category: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RAGResult:
    """Represents a RAG retrieval result."""
    content: str
    score: float
    category: str
    metadata: Dict[str, Any]


class RAGService:
    """Service for managing knowledge base and RAG operations."""

    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "social_support_kb"):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("RAG dependencies not available. Install with: pip install qdrant-client ollama")

        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.embedding_model = "nomic-embed-text"
        self.vector_size = 768  # nomic-embed-text dimension

        # Initialize collection if not exists
        self._init_collection()

    def _init_collection(self):
        """Initialize Qdrant collection for knowledge base."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                print(f"✓ Created Qdrant collection: {self.collection_name}")
            else:
                print(f"✓ Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            print(f"Failed to initialize Qdrant collection: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Failed to get embedding: {e}")
            raise

    def add_knowledge_item(self, content: str, category: str, metadata: Dict[str, Any] = None) -> str:
        """Add a knowledge item to the knowledge base."""
        metadata = metadata or {}
        item_id = str(uuid.uuid4())

        # Get embedding
        embedding = self._get_embedding(content)

        # Create point
        point = PointStruct(
            id=item_id,
            vector=embedding,
            payload={
                "content": content,
                "category": category,
                "metadata": metadata
            }
        )

        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        return item_id

    def search_knowledge(self, query: str, limit: int = 5, score_threshold: float = 0.5) -> List[RAGResult]:
        """Search knowledge base for relevant content."""
        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )

        # Convert to RAGResult objects
        results = []
        for result in search_results:
            rag_result = RAGResult(
                content=result.payload["content"],
                score=result.score,
                category=result.payload["category"],
                metadata=result.payload["metadata"]
            )
            results.append(rag_result)

        return results

    def get_contextual_response(self, query: str, context_limit: int = 3) -> Dict[str, Any]:
        """Get contextual response for a query with RAG."""
        # Search knowledge base
        knowledge_results = self.search_knowledge(query, limit=context_limit)

        if not knowledge_results:
            return {
                "query": query,
                "context": [],
                "response": "I don't have specific information about that. Could you please provide more details or rephrase your question?",
                "confidence": 0.0
            }

        # Build context
        context = []
        for result in knowledge_results:
            context.append({
                "content": result.content,
                "category": result.category,
                "score": result.score
            })

        # Calculate average confidence
        avg_confidence = sum(r.score for r in knowledge_results) / len(knowledge_results)

        # For now, return the most relevant content as response
        # In a full implementation, this would use an LLM to generate a response
        best_match = knowledge_results[0]
        response = best_match.content

        return {
            "query": query,
            "context": context,
            "response": response,
            "confidence": avg_confidence,
            "category": best_match.category
        }

    def initialize_uae_knowledge_base(self):
        """Initialize knowledge base with UAE social support information."""
        uae_knowledge = [
            # Eligibility Criteria
            {
                "content": "UAE citizens and GCC nationals are eligible for social support. Household monthly income should be below 15,000 AED for a family of 4. Documentation required includes Emirates ID, salary certificate, bank statements, and utility bills.",
                "category": "eligibility",
                "metadata": {"priority": "high", "source": "policy_doc_1"}
            },
            {
                "content": "Income thresholds: Single person: 5,000 AED, Family of 2: 8,000 AED, Family of 3: 12,000 AED, Family of 4+: 15,000 AED. These are maximum monthly household income limits.",
                "category": "eligibility",
                "metadata": {"priority": "high", "source": "income_guidelines"}
            },

            # Required Documents
            {
                "content": "Required documents for social support application: 1) Emirates ID copy, 2) Salary certificate or employment letter, 3) Bank statements (last 3 months), 4) Utility bills, 5) Tenancy contract or property ownership documents, 6) Family book for dependents.",
                "category": "documents",
                "metadata": {"priority": "high", "source": "application_requirements"}
            },
            {
                "content": "Bank statements must show clear income and expense patterns. Statements should be official with bank stamps or digital statements with verification codes. Cash deposits without source documentation may require additional verification.",
                "category": "documents",
                "metadata": {"priority": "medium", "source": "verification_guidelines"}
            },

            # Common Issues & FAQs
            {
                "content": "Common application issues: Income discrepancies between declared and bank statements, missing employment documentation, expired Emirates ID, unclear address verification. These issues can delay processing by 5-10 business days.",
                "category": "troubleshooting",
                "metadata": {"priority": "medium", "source": "common_issues"}
            },
            {
                "content": "If your declared income differs significantly from bank statements, you'll need to provide additional documentation: employment contract, HR confirmation letter, explanation letter for any discrepancies, and recent pay slips.",
                "category": "troubleshooting",
                "metadata": {"priority": "high", "source": "income_verification"}
            },

            # Decision Factors
            {
                "content": "Application approval factors: Household income below threshold, consistent employment history, clear financial documentation, genuine need demonstrated, all required documents provided. Higher household sizes receive priority consideration.",
                "category": "decision_factors",
                "metadata": {"priority": "high", "source": "approval_criteria"}
            },
            {
                "content": "Common rejection reasons: Income exceeds limits, incomplete documentation, suspected fraud or duplicate applications, employment history gaps, insufficient proof of need. Appeals can be submitted within 30 days.",
                "category": "decision_factors",
                "metadata": {"priority": "medium", "source": "rejection_criteria"}
            },

            # Process Information
            {
                "content": "Application processing timeline: Initial review (1-2 days), document verification (3-5 days), income assessment (1-2 days), final decision (1-2 days). Total processing time: 6-11 business days for complete applications.",
                "category": "process",
                "metadata": {"priority": "medium", "source": "processing_timeline"}
            },
            {
                "content": "After approval, social support payments are processed monthly via bank transfer. First payment typically arrives within 10 business days of approval. Recipients must provide quarterly updates on income and household changes.",
                "category": "process",
                "metadata": {"priority": "medium", "source": "payment_process"}
            }
        ]

        # Add all knowledge items
        for item in uae_knowledge:
            try:
                item_id = self.add_knowledge_item(
                    content=item["content"],
                    category=item["category"],
                    metadata=item["metadata"]
                )
                print(f"✓ Added knowledge item: {item['category']} - {item_id}")
            except Exception as e:
                print(f"Failed to add knowledge item: {e}")

        print(f"✓ Initialized UAE knowledge base with {len(uae_knowledge)} items")

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "total_items": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name
            }
        except Exception as e:
            return {"error": str(e)}


# Global RAG service instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def initialize_knowledge_base() -> bool:
    """Initialize the knowledge base with UAE social support information."""
    try:
        rag_service = get_rag_service()
        rag_service.initialize_uae_knowledge_base()
        return True
    except Exception as e:
        print(f"Failed to initialize knowledge base: {e}")
        return False