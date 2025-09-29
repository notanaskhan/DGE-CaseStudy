"""
Database models for PostgreSQL/SQLite using SQLAlchemy.
"""

import datetime as dt
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Application(Base):
    __tablename__ = "applications"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    emirates_id = Column(String, nullable=True, index=True)
    submitted_at = Column(DateTime, default=dt.datetime.utcnow)
    status = Column(String, default="NEW")
    channel = Column(String, default="Online")
    declared_monthly_income = Column(Float, default=0.0)
    household_size = Column(Integer, default=1)
    decision = Column(String, nullable=True)
    decision_reason = Column(Text, nullable=True)
    documents = relationship("Document", back_populates="application")

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    application_id = Column(String, ForeignKey("applications.id"))
    kind = Column(String, default="unknown")
    path = Column(Text, nullable=False)
    filename = Column(String, nullable=False)
    sha256 = Column(String, nullable=True)
    parsed_ok = Column(Integer, default=0)
    ocr_ok = Column(Integer, default=0)
    application = relationship("Application", back_populates="documents")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, index=True)
    application_id = Column(String, index=True)
    actor = Column(String, default="system")
    action = Column(String)
    payload_json = Column(Text)
    at = Column(DateTime, default=dt.datetime.utcnow)