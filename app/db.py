# app/db.py
"""
Módulo de base de datos — Delowyss Trading AI
CEO: Eduardo Solis — © 2025
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./delowyss.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --------------------
# MODELOS DE TABLAS
# --------------------

class Tick(Base):
    __tablename__ = "ticks"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, default="EUR/USD")
    mid = Column(Float, nullable=False)
    t_ms = Column(BigInteger, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    symbol = Column(String, nullable=False)
    time_now_ms = Column(BigInteger, nullable=False)
    p_up = Column(Float, nullable=False)
    p_down = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    signal = Column(String, nullable=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    password_hash = Column(String)
    api_key = Column(String, unique=True)
    plan = Column(String, default="weekly")
    active = Column(Boolean, default=True)
    created_at = Column(BigInteger, default=0)


# --------------------
# Funciones públicas
# --------------------

def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
