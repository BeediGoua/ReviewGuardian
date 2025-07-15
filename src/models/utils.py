# src/models/utils.py

import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Any, Optional

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Sauvegarde & chargement de modèles =========

def save_model(obj: Any, output_path: str) -> None:
    """
    Sauvegarde un modèle ou pipeline avec joblib.
    """
    joblib.dump(obj, output_path)
    logger.info(f"Modèle sauvegardé : {output_path}")


def load_model(model_path: str) -> Any:
    """
    Charge un modèle ou pipeline depuis un fichier .pkl.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    logger.info(f"Chargement modèle depuis : {model_path}")
    return joblib.load(model_path)


def save_model_with_version(obj: Any, base_dir: str, prefix: str, ext: str = ".pkl") -> str:
    """
    Sauvegarde un modèle avec un nom de version unique : ex. logreg_v2.pkl
    """
    path = get_next_version_name(base_dir, prefix, ext)
    save_model(obj, path)
    return path


def get_next_version_name(base_dir: str, prefix: str, extension: str = ".pkl") -> str:
    """
    Génère un nom de fichier versionné automatiquement.
    """
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}_v{i}{extension}")):
        i += 1
    return os.path.join(base_dir, f"{prefix}_v{i}{extension}")


# ========= Chargement Dataset =========

def load_preprocessed_dataset(path: str,
                              text_col: str = "text_clean",
                              label_col: str = "label_toxic") -> pd.DataFrame:
    """
    Charge un DataFrame nettoyé et vérifie les colonnes clés.
    """
    df = pd.read_csv(path)
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")
    logger.info(f"Dataset chargé depuis : {path} ({len(df)} lignes)")
    return df


# ========= Diagnostic classification =========

def show_misclassified_examples(X_text, y_true, y_pred, n: int = 5) -> None:
    """
    Affiche quelques exemples d’erreurs de prédiction.
    """
    errors = np.where(y_true != y_pred)[0]
    logger.info(f"{len(errors)} erreurs détectées")
    for idx in errors[:n]:
        print(f"\n[Exemple {idx}]")
        print(f"Prédit : {y_pred[idx]} | Réel : {y_true[idx]}")
        print("Texte :", X_text.iloc[idx][:300], "...")


def show_class_distribution(df: pd.DataFrame, label_col: str = "label_toxic") -> None:
    """
    Affiche la répartition des classes dans un DataFrame.
    """
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    print("\nDistribution des classes :")
    for label, count in counts.items():
        pct = (count / total) * 100
        print(f" - Classe {label} : {count} exemples ({pct:.2f}%)")


def compute_class_weights(df: pd.DataFrame, label_col: str = "label_toxic") -> dict:
    """
    Calcule les poids inverses pour chaque classe (utile si pas de class_weight='balanced').
    """
    counts = df[label_col].value_counts(normalize=True)
    return {cls: 1 / prop for cls, prop in counts.items()}
