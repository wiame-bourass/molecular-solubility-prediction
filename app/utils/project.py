from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data

def read_csv(relative):
    p = ROOT / relative
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource

def read_model(path):
    return joblib.load(str(path))

def model_options():
    candidates = [
        ROOT/'results/optimisation/best_scaffold.joblib',
        ROOT/'results/optimisation/best_random.joblib',
        ROOT/'models/representations/best_scaffold.joblib',
        ROOT/'models/representations/best_random.joblib',
    ]
    return [p for p in candidates if p.is_file()]

def model_details(payload, filename):
    """Explicitly support the known comparison-model payload; avoid guessing optimization format."""
    if not isinstance(payload, dict) or 'model' not in payload or 'representation' not in payload:
        raise ValueError('Format du fichier modèle non reconnu. Vérifier le contenu du .joblib.')
    return payload
