# app/auth.py
"""
Autenticación y seguridad — Delowyss Trading AI
CEO: Eduardo Solis — © 2025
"""

import os
import hashlib
import secrets
from fastapi import Header, HTTPException, Depends
from app.db import User, get_session

MASTER_KEY = os.getenv("MASTER_KEY", "MASTER1234")


# --------------------------------
# Generación de API KEYS
# --------------------------------
def gen_api_key() -> str:
    return secrets.token_hex(32)


# --------------------------------
# Hash de contraseñas
# --------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# --------------------------------
# Verificar API KEY Master
# --------------------------------
def verify_master(x_master: str = Header(None)):
    if x_master != MASTER_KEY:
        raise HTTPException(status_code=401, detail="Master key inválida")
    return True


# --------------------------------
# Verificar API KEY de usuario
# --------------------------------
def require_api_key(x_api_key: str = Header(None), session=Depends(get_session)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key requerida")
    user = session.query(User).filter(User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="API key inválida")
    if not user.active:
        raise HTTPException(status_code=403, detail="Usuario inactivo")
    return user
