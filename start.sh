#!/bin/bash
# start.sh — Inicia la API de Delowyss Trading AI

exec uvicorn app.main:app --host 0.0.0.0 --port 10000 --reload
