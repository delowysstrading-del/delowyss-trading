# app/schemas.py
"""
Esquemas Pydantic — Delowyss Trading AI
CEO: Eduardo Solis — © 2025
"""

from pydantic import BaseModel
from typing import List


class TickItem(BaseModel):
    mid: float
    t_ms: int


class InferPayload(BaseModel):
    symbol: str
    time_now: int
    ticks: List[TickItem]


class PredictionOut(BaseModel):
    signal: str
    p_up: float
    p_down: float
    confidence: float
    time: int
