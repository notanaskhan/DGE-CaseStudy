"""
Database configuration with PostgreSQL, MongoDB, and SQLite support.
Provides graceful fallback from PostgreSQL -> SQLite if PostgreSQL is unavailable.
"""

import os
import logging
from typing import Optional, Any, Dict
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB imports with graceful fallback
try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    MongoClient = None
    Collection = None
    Database = None

from .models import Base

logger = logging.getLogger(__name__)

# Global database instances
_postgres_engine: Optional[Engine] = None
_sqlite_engine: Optional[Engine] = None
_mongo_client: Optional[MongoClient] = None
_mongo_db: Optional[Database] = None
_session_maker: Optional[sessionmaker] = None
_current_db_type: str = "unknown"

# Paths
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(REPO_DIR, "data")
DEFAULT_SQLITE_DB = os.path.join(DATA_DIR, "app.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_postgres_engine() -> Optional[Engine]:
    """Get PostgreSQL engine if available."""
    global _postgres_engine

    if _postgres_engine is not None:
        return _postgres_engine

    # Get PostgreSQL connection details from environment
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        # Construct from individual components using pg8000 driver
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        db_name = os.getenv("POSTGRES_DB", "social_support")
        postgres_url = f"postgresql+pg8000://{user}:{password}@{host}:{port}/{db_name}"

    try:
        _postgres_engine = create_engine(
            postgres_url,
            echo=os.getenv("DATABASE_DEBUG", "").lower() == "true",
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800
        )

        # Test connection
        from sqlalchemy import text
        with _postgres_engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("PostgreSQL connection established")
        return _postgres_engine

    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e}")
        _postgres_engine = None
        return None

def get_sqlite_engine() -> Engine:
    """Get SQLite engine as fallback."""
    global _sqlite_engine

    if _sqlite_engine is not None:
        return _sqlite_engine

    sqlite_path = os.getenv("SQLITE_PATH", DEFAULT_SQLITE_DB)
    sqlite_url = f"sqlite:///{sqlite_path}"

    _sqlite_engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        echo=os.getenv("DATABASE_DEBUG", "").lower() == "true"
    )

    logger.info(f"SQLite connection established: {sqlite_path}")
    return _sqlite_engine

def get_current_engine() -> Engine:
    """Get the current database engine (PostgreSQL with SQLite fallback)."""
    global _current_db_type

    # Try PostgreSQL first
    postgres_engine = get_postgres_engine()
    if postgres_engine is not None:
        _current_db_type = "postgresql"
        return postgres_engine

    # Fallback to SQLite
    _current_db_type = "sqlite"
    sqlite_engine = get_sqlite_engine()
    return sqlite_engine

def get_current_session() -> sessionmaker:
    """Get session maker for current database."""
    global _session_maker

    if _session_maker is None:
        engine = get_current_engine()
        _session_maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)

    return _session_maker

def get_mongo_client() -> Optional[MongoClient]:
    """Get MongoDB client if available."""
    global _mongo_client

    if not MONGO_AVAILABLE:
        logger.warning("MongoDB not available - pymongo not installed")
        return None

    if _mongo_client is not None:
        return _mongo_client

    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")

    try:
        _mongo_client = MongoClient(
            mongo_url,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )

        # Test connection
        _mongo_client.admin.command('ping')
        logger.info("MongoDB connection established")
        return _mongo_client

    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        _mongo_client = None
        return None

def get_mongo_database() -> Optional[Database]:
    """Get MongoDB database instance."""
    global _mongo_db

    if _mongo_db is not None:
        return _mongo_db

    client = get_mongo_client()
    if client is None:
        return None

    db_name = os.getenv("MONGO_DB", "social_support")
    _mongo_db = client[db_name]
    return _mongo_db

def get_mongo_collection(collection_name: str) -> Optional[Collection]:
    """Get MongoDB collection."""
    db = get_mongo_database()
    if db is None:
        return None
    return db[collection_name]

def get_database_info() -> Dict[str, Any]:
    """Get information about current database connections."""
    info = {
        "primary_db": _current_db_type,
        "postgres_available": get_postgres_engine() is not None,
        "sqlite_available": True,  # SQLite is always available
        "mongo_available": get_mongo_client() is not None,
        "mongo_installed": MONGO_AVAILABLE
    }
    return info

def init_databases():
    """Initialize all database connections."""
    logger.info("Initializing database connections...")

    # Initialize primary database (PostgreSQL with SQLite fallback)
    engine = get_current_engine()
    Base.metadata.create_all(bind=engine)
    logger.info(f"Primary database initialized: {_current_db_type}")

    # Initialize MongoDB if available
    mongo_client = get_mongo_client()
    if mongo_client:
        logger.info("MongoDB initialized")
    else:
        logger.info("MongoDB not available - continuing without")

    # Log database status
    info = get_database_info()
    logger.info(f"Database status: {info}")

def close_databases():
    """Close all database connections."""
    global _postgres_engine, _sqlite_engine, _mongo_client, _session_maker

    if _postgres_engine:
        _postgres_engine.dispose()
        _postgres_engine = None

    if _sqlite_engine:
        _sqlite_engine.dispose()
        _sqlite_engine = None

    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None

    _session_maker = None
    logger.info("All database connections closed")

# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions with automatic cleanup."""

    def __init__(self):
        self.session: Optional[Session] = None

    def __enter__(self) -> Session:
        session_maker = get_current_session()
        self.session = session_maker()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()

# Compatibility layer for existing code
def get_session_local():
    """Get session maker for backward compatibility."""
    return get_current_session()

def get_engine():
    """Get current engine for backward compatibility."""
    return get_current_engine()