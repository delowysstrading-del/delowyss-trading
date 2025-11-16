#!/bin/bash
# start.sh — Inicia la API de Delowyss Trading AI

# Activar entorno virtual (opcional, Render hace esto automáticamente)
# source .venv/bin/activate

# Ejecutar servidor Uvicorn
exec uvicorn app.main:app --host 0.0.0.0 --port 10000 --reload
