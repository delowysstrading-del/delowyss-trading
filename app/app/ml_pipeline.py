# app/ml_pipeline.py
"""
ML pipeline minimalista para Delowyss Trading AI.
Funciones exposadas:
- load_model()
- save_model(model)
- extract_features_from_ticks(ticks)
- predict_from_features(model, features)
- train_from_dataset(dataset, labels)

Diseñado para ser sencillo, auditable y fácil de mejorar con tus features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
RANDOM_STATE = 42

def load_model():
    """Carga el modelo desde disco si existe, si no devuelve None."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            # si falla la carga, loggear y devolver None
            print(f"[ml_pipeline] Error loading model: {e}")
            return None
    return None

def save_model(model):
    """Persiste el modelo en disco con joblib."""
    try:
        joblib.dump(model, MODEL_PATH)
        print("[ml_pipeline] Modelo guardado en:", MODEL_PATH)
    except Exception as e:
        print(f"[ml_pipeline] Error saving model: {e}")
        raise

# ---------------------------
# Feature extraction
# ---------------------------
def extract_features_from_ticks(ticks):
    """
    ticks: lista de dicts con al menos {'mid': float, 't_ms': int} o similar.
    Devuelve: numpy array 1D de features o None si no hay suficientes ticks.
    Implementación conservadora: estadísticas básicas (mean, std, min, max, slope, last-return).
    """
    try:
        if not ticks or len(ticks) < 5:
            return None
        mids = np.array([float(x.get("mid", x.get("price", 0.0))) for x in ticks])
        # estadísticas simples
        mean = mids.mean()
        std = mids.std()
        mn = mids.min()
        mx = mids.max()
        last = mids[-1]
        # returns
        returns = np.diff(mids) / mids[:-1]
        mean_ret = returns.mean() if len(returns) > 0 else 0.0
        std_ret = returns.std() if len(returns) > 0 else 0.0
        # slope (linear fit)
        x = np.arange(len(mids))
        if len(mids) >= 2:
            slope = np.polyfit(x, mids, 1)[0]
        else:
            slope = 0.0

        features = np.array([mean, std, mn, mx, last, mean_ret, std_ret, slope], dtype=float)
        return features.reshape(1, -1)  # return 2D for sklearn
    except Exception as e:
        print(f"[ml_pipeline] Error extracting features: {e}")
        return None

# ---------------------------
# Prediction helpers
# ---------------------------
def predict_from_features(model, features):
    """
    Dado un modelo y features (1xN), devuelve (p_up, p_down).
    Asume que la clase '1' corresponde a UP y '0' a DOWN.
    Si el modelo no expone predict_proba, intenta usar predict.
    """
    try:
        if model is None:
            raise ValueError("Modelo no cargado")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            # buscar índice de clase 1 y 0 (puede variar)
            classes = list(model.classes_)
            if 1 in classes and 0 in classes:
                p_up = float(probs[classes.index(1)])
                p_down = float(probs[classes.index(0)])
            else:
                # fallback: si solo hay una clase conocida, devolver confianza 1 en esa clase
                if classes[0] == 1:
                    p_up, p_down = 1.0, 0.0
                else:
                    p_up, p_down = 0.0, 1.0
        else:
            # fallback: usar predict, devolver 0.9/0.1 confianza artificial
            pred = model.predict(features)[0]
            if pred == 1:
                p_up, p_down = 0.9, 0.1
            else:
                p_up, p_down = 0.1, 0.9
        return p_up, p_down
    except Exception as e:
        print(f"[ml_pipeline] Error predict_from_features: {e}")
        # si hay problema, devolver probabilidades neutrales
        return 0.5, 0.5

# ---------------------------
# Training
# ---------------------------
def train_from_dataset(dataset, labels):
    """
    dataset: list/array of feature vectors (each 1D)
    labels: list/array of 0/1
    Devuelve: modelo entrenado (RandomForestClassifier).
    """
    try:
        X = np.vstack(dataset)
        y = np.array(labels, dtype=int)

        # Validación mínima
        if X.shape[0] < 10 or len(set(y)) < 2:
            raise ValueError("Datos insuficientes para entrenar o solo una clase presente")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        # evaluar (opcional)
        try:
            acc = model.score(X_val, y_val)
            print(f"[ml_pipeline] Entrenamiento finalizado. Val acc: {acc:.4f}")
        except Exception:
            acc = None

        # persistir inmediatamente
        try:
            save_model(model)
        except Exception:
            pass

        return model
    except Exception as e:
        print(f"[ml_pipeline] Error train_from_dataset: {e}")
        raise
