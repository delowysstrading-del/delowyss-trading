# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional


# ---- Tick recibido desde el cliente ----
class TickIn(BaseModel):
    mid: float = Field(..., description="Precio medio del tick")


# ---- Payload para /api/infer ----
class InferPayload(BaseModel):
    symbol: str
    time_now: int
    ticks: List[TickIn]


# ---- Respuesta de inferencia ----
class PredictionOut(BaseModel):
    signal: str
    p_up: float
    p_down: float
    confidence: float
    time: int


# ---- Endpoints de usuario admin ----
class UserCreate(BaseModel):
    username: str
    password: str
    plan: str = "weekly"


class UserOut(BaseModel):
    username: str
    plan: str
    active: bool
    api_key: Optional[str] = None
