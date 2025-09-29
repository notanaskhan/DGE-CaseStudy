"""
Database configuration and initialization module.
Provides PostgreSQL, MongoDB, and SQLite database connections with fallback support.
"""

from .database_config import (
    get_postgres_engine,
    get_mongo_client,
    get_sqlite_engine,
    get_current_session,
    get_mongo_collection,
    init_databases,
    close_databases,
    DatabaseSession
)

from .models import Base, Application, Document, AuditLog

__all__ = [
    'get_postgres_engine',
    'get_mongo_client',
    'get_sqlite_engine',
    'get_current_session',
    'get_mongo_collection',
    'init_databases',
    'close_databases',
    'DatabaseSession',
    'Base',
    'Application',
    'Document',
    'AuditLog'
]