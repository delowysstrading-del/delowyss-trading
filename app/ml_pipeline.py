# app/ml_pipeline.py
"""
ML pipeline para Delowyss Trading AI
Funciones públicas:
- load_model()
- save_model(model)
- extract_features_from_ticks(ticks)
- predict_from_features(model, features)
- train_from_dataset(dataset, labels)

Implementación conservadora basada en scikit-learn (RandomForest).
Persistencia con joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))
MIN_FEATURE_TICKS = int(os.environ.get("MIN_FEATURE_TICKS", 5))


def load_model():
    """Carga modelo desde disco si existe, si no devuelve None."""
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            return m
        except Exception as e:
            print(f"[ml_pipeline] Error cargando modelo: {e}")
            return None
    return None


def save_model(model):
    """Guarda modelo en disco con joblib."""
    try:
        joblib.dump(model, MODEL_PATH)
        print(f"[ml_pipeline] Modelo guardado en {MODEL_PATH}")
    except Exception as e:
        print(f"[ml_pipeline] Error guardando modelo: {e}")
        raise


# ---------------------------
# Feature extraction
# ---------------------------
def extract_features_from_ticks(ticks):
    """
    ticks: lista de dicts con al menos {'mid': float} y opcional 't_ms'
    Devuelve: features (2D numpy array) o None si insuficientes ticks.
    Features simples: mean, std, min, max, last, mean_ret, std_ret, slope
    """
    try:
        if not ticks or len(ticks) < MIN_FEATURE_TICKS:
            return None

        # Extraer mids
        mids = np.array([float(t.get("mid", t.get("price", 0.0))) for t in ticks], dtype=float)

        mean = mids.mean()
        std = mids.std()
        mn = mids.min()
        mx = mids.max()
        last = mids[-1]

        # returns relativos
        if len(mids) >= 2:
            returns = np.diff(mids) / mids[:-1]
            mean_ret = returns.mean()
            std_ret = returns.std()
        else:
            mean_ret = 0.0
            std_ret = 0.0

        # slope linea recta
        if len(mids) >= 2:
            x = np.arange(len(mids))
            slope = np.polyfit(x, mids, 1)[0]
        else:
            slope = 0.0

        features = np.array([mean, std, mn, mx, last, mean_ret, std_ret, slope], dtype=float)
        return features.reshape(1, -1)
    except Exception as e:
        print(f"[ml_pipeline] Error en extract_features_from_ticks: {e}")
        return None


# ---------------------------
# Prediction helpers
# ---------------------------
def predict_from_features(model, features):
    """
    Dado modelo y features (1xN), devuelve (p_up, p_down).
    Si model tiene predict_proba, se usan probabilidades; si no, fallback.
    """
    try:
        if model is None:
            raise ValueError("Modelo no cargado")
        # predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            # mapear indices a clases
            classes = list(model.classes_)
            p_up = 0.0
            p_down = 0.0
            # buscar índice de la clase 1 y 0
            if 1 in classes:
                p_up = float(probs[classes.index(1)])
            if 0 in classes:
                p_down = float(probs[classes.index(0)])
            # si no hay ambas clases, repartir probabilidades
            if (p_up + p_down) == 0:
                # si solo una clase presente, asignar 1.0 a la presente
                if len(classes) == 1:
                    if classes[0] == 1:
                        p_up, p_down = 1.0, 0.0
                    else:
                        p_up, p_down = 0.0, 1.0
                else:
                    p_up, p_down = 0.5, 0.5
        else:
            pred = model.predict(features)[0]
            if pred == 1:
                p_up, p_down = 0.9, 0.1
            else:
                p_up, p_down = 0.1, 0.9
        return p_up, p_down
    except Exception as e:
        print(f"[ml_pipeline] Error en predict_from_features: {e}")
        return 0.5, 0.5


# ---------------------------
# Training
# ---------------------------
def train_from_dataset(dataset, labels):
    """
    dataset: lista o array de vectores de features (1D o 2D)
    labels: lista de 0/1
    Devuelve: modelo entrenado (RandomForestClassifier)
    """
    try:
        X = None
        if isinstance(dataset, list):
            X = np.vstack(dataset)
        else:
            X = np.array(dataset)

        y = np.array(labels, dtype=int)

        if X.shape[0] < 10:
            raise ValueError("Datos insuficientes para entrenar (menos de 10 muestras)")

        if len(set(y)) < 2:
            raise ValueError("Se necesita al menos 2 clases para entrenar")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        try:
            acc = model.score(X_val, y_val)
            print(f"[ml_pipeline] Entrenamiento completado. Val acc: {acc:.4f}")
        except Exception:
            acc = None

        # persistir
        try:
            save_model(model)
        except Exception:
            pass

        return model
    except Exception as e:
        print(f"[ml_pipeline] Error en train_from_dataset: {e}")
        raise
