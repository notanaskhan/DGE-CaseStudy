"""
MongoDB service for document storage, analytics, and caching.
Provides document storage, application analytics, and performance optimization.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .database_config import get_mongo_collection

logger = logging.getLogger(__name__)

class MongoService:
    """Service for MongoDB operations."""

    def __init__(self):
        self.applications_collection = "applications"
        self.documents_collection = "documents"
        self.analytics_collection = "analytics"
        self.cache_collection = "cache"
        self.ocr_results_collection = "ocr_results"
        self.parsed_results_collection = "parsed_results"

    def _get_collection(self, collection_name: str):
        """Get MongoDB collection with fallback handling."""
        collection = get_mongo_collection(collection_name)
        if collection is None:
            logger.warning(f"MongoDB collection {collection_name} not available")
        return collection

    def store_application_data(self, application_id: str, application_data: Dict[str, Any]) -> bool:
        """Store complete application data in MongoDB."""
        collection = self._get_collection(self.applications_collection)
        if collection is None:
            return False

        try:
            document = {
                "_id": application_id,
                "application_id": application_id,
                "data": application_data,
                "stored_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Upsert operation
            collection.replace_one(
                {"_id": application_id},
                document,
                upsert=True
            )

            logger.info(f"Application data stored in MongoDB: {application_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store application data: {e}")
            return False

    def get_application_data(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve application data from MongoDB."""
        collection = self._get_collection(self.applications_collection)
        if collection is None:
            return None

        try:
            document = collection.find_one({"_id": application_id})
            if document:
                return document.get("data")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve application data: {e}")
            return None

    def store_document_result(self, application_id: str, filename: str,
                             result_type: str, result_data: Dict[str, Any]) -> bool:
        """Store document processing results (OCR, parsing, etc.)."""
        collection_name = f"{result_type}_results"
        collection = self._get_collection(collection_name)
        if collection is None:
            return False

        try:
            document = {
                "_id": f"{application_id}_{filename}_{result_type}",
                "application_id": application_id,
                "filename": filename,
                "result_type": result_type,
                "result_data": result_data,
                "processed_at": datetime.utcnow()
            }

            collection.replace_one(
                {"_id": document["_id"]},
                document,
                upsert=True
            )

            logger.debug(f"Document result stored: {result_type} for {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to store document result: {e}")
            return False

    def get_document_result(self, application_id: str, filename: str,
                          result_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve document processing results."""
        collection_name = f"{result_type}_results"
        collection = self._get_collection(collection_name)
        if collection is None:
            return None

        try:
            document_id = f"{application_id}_{filename}_{result_type}"
            document = collection.find_one({"_id": document_id})
            if document:
                return document.get("result_data")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve document result: {e}")
            return None

    def store_analytics_data(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Store analytics and performance data."""
        collection = self._get_collection(self.analytics_collection)
        if collection is None:
            return False

        try:
            document = {
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.utcnow()
            }

            collection.insert_one(document)
            logger.debug(f"Analytics data stored: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to store analytics data: {e}")
            return False

    def get_application_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get application statistics for the last N days."""
        collection = self._get_collection(self.applications_collection)
        if collection is None:
            return {}

        try:
            since_date = datetime.utcnow() - timedelta(days=days)

            pipeline = [
                {
                    "$match": {
                        "stored_at": {"$gte": since_date}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_applications": {"$sum": 1},
                        "avg_income": {"$avg": "$data.app_row.declared_monthly_income"},
                        "avg_household_size": {"$avg": "$data.app_row.household_size"},
                        "channels": {"$addToSet": "$data.app_row.channel"}
                    }
                }
            ]

            result = list(collection.aggregate(pipeline))
            if result:
                stats = result[0]
                stats.pop("_id", None)
                return stats
            else:
                return {"total_applications": 0}

        except Exception as e:
            logger.error(f"Failed to get application statistics: {e}")
            return {}

    def cache_set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set cache value with TTL."""
        collection = self._get_collection(self.cache_collection)
        if collection is None:
            return False

        try:
            expire_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            document = {
                "_id": key,
                "value": value,
                "expire_at": expire_at,
                "created_at": datetime.utcnow()
            }

            collection.replace_one(
                {"_id": key},
                document,
                upsert=True
            )

            # Create TTL index if it doesn't exist
            collection.create_index("expire_at", expireAfterSeconds=0)

            return True

        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value if not expired."""
        collection = self._get_collection(self.cache_collection)
        if collection is None:
            return None

        try:
            document = collection.find_one({"_id": key})
            if document and document.get("expire_at", datetime.utcnow()) > datetime.utcnow():
                return document.get("value")
            return None

        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None

    def find_similar_applications(self, emirates_id: str, income: float,
                                household_size: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar applications for duplicate detection."""
        collection = self._get_collection(self.applications_collection)
        if collection is None:
            return []

        try:
            # Income range (±20%)
            income_lower = income * 0.8
            income_upper = income * 1.2

            query = {
                "$or": [
                    {"data.app_row.emirates_id": emirates_id},
                    {
                        "$and": [
                            {"data.app_row.declared_monthly_income": {
                                "$gte": income_lower, "$lte": income_upper}},
                            {"data.app_row.household_size": household_size}
                        ]
                    }
                ]
            }

            results = list(collection.find(query).limit(limit))
            return [{"application_id": r["application_id"], "data": r["data"]} for r in results]

        except Exception as e:
            logger.error(f"Failed to find similar applications: {e}")
            return []

    def get_processing_performance(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        collection = self._get_collection(self.analytics_collection)
        if collection is None:
            return {}

        try:
            pipeline = [
                {
                    "$match": {
                        "event_type": {"$in": ["ocr_processing", "document_parsing", "eligibility_check"]},
                        "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=24)}
                    }
                },
                {
                    "$group": {
                        "_id": "$event_type",
                        "avg_duration": {"$avg": "$data.duration_seconds"},
                        "total_processed": {"$sum": 1},
                        "max_duration": {"$max": "$data.duration_seconds"},
                        "min_duration": {"$min": "$data.duration_seconds"}
                    }
                }
            ]

            results = list(collection.aggregate(pipeline))
            return {r["_id"]: {
                "avg_duration": r["avg_duration"],
                "total_processed": r["total_processed"],
                "max_duration": r["max_duration"],
                "min_duration": r["min_duration"]
            } for r in results}

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}

    def cleanup_expired_data(self, days_old: int = 90) -> int:
        """Clean up old data to manage storage."""
        cleaned_count = 0

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Clean old analytics data
            analytics_collection = self._get_collection(self.analytics_collection)
            if analytics_collection:
                result = analytics_collection.delete_many({
                    "timestamp": {"$lt": cutoff_date}
                })
                cleaned_count += result.deleted_count

            # Clean old processing results
            for result_type in ["ocr", "parsed"]:
                collection = self._get_collection(f"{result_type}_results")
                if collection:
                    result = collection.delete_many({
                        "processed_at": {"$lt": cutoff_date}
                    })
                    cleaned_count += result.deleted_count

            logger.info(f"Cleaned up {cleaned_count} old documents")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
            return 0

# Global service instance
_mongo_service = None

def get_mongo_service() -> MongoService:
    """Get or create the global MongoDB service instance."""
    global _mongo_service
    if _mongo_service is None:
        _mongo_service = MongoService()
    return _mongo_service