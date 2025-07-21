# src/models/utils.py

"""
Utilitaires pour le pipeline de modération de contenu :
- Sauvegarde/chargement de modèles
- Chargement de datasets
- Diagnostic des performances
"""

import os
import joblib
import logging
from typing import Any, Optional, List

import pandas as pd
import numpy as np
from typing import Union
# CONFIGURATION LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SAUVEGARDE & CHARGEMENT DE MODÈLES (joblib + versioning)


def save_model(obj: Any, output_path: str) -> None:
    """
    Sauvegarde un objet (modèle, pipeline, etc.) au format .pkl avec joblib.
    """
    joblib.dump(obj, output_path)
    logger.info(f"Modèle sauvegardé → {output_path}")


def load_model(model_path: str) -> Any:
    """
    Charge un modèle enregistré avec joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    logger.info(f"Chargement modèle → {model_path}")
    return joblib.load(model_path)


def get_next_version_name(base_dir: str, prefix: str, extension: str = ".pkl") -> str:
    """
    Génère un nom de fichier unique pour la version suivante.
    Exemple : logreg_v1.pkl, logreg_v2.pkl...
    """
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}_v{i}{extension}")):
        i += 1
    return os.path.join(base_dir, f"{prefix}_v{i}{extension}")


def save_model_with_version(obj: Any, base_dir: str, prefix: str, ext: str = ".pkl") -> str:
    """
    Combine versioning + sauvegarde.
    """
    path = get_next_version_name(base_dir, prefix, ext)
    save_model(obj, path)
    return path


# CHARGEMENT & VALIDATION DE DONNÉES


def load_dataset(path: str) -> pd.DataFrame:
    """
    Charge un DataFrame depuis un fichier CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier non trouvé : {path}")
    df = pd.read_csv(path)
    logger.info(f"{len(df)} lignes chargées depuis {path}")
    return df


def load_preprocessed_dataset(path: str,
                              required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Charge un dataset nettoyé et valide la présence des colonnes nécessaires.
    Par défaut : ["text_clean", "label_toxic"]
    """
    df = load_dataset(path)
    required_cols = required_cols or ["text_clean", "label_toxic"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le fichier : {col}")
    return df


# DIAGNOSTIC DES CLASSIFICATIONS


def show_misclassified_examples(X_text: pd.Series,
                                y_true: Union[pd.Series, np.ndarray],
                                y_pred: Union[pd.Series, np.ndarray],
                                n: int = 5,
                                save_path: Optional[str] = None) -> None:
    """
    Affiche (et sauvegarde optionnellement) les exemples mal classés.
    """
    errors = np.where(y_true != y_pred)[0]
    logger.info(f"{len(errors)} erreurs de classification détectées")

    rows = []
    for idx in errors[:n]:
        print(f"\n[Exemple {idx}]")
        print(f"Prédit : {y_pred[idx]} | Réel : {y_true[idx]}")
        print("Texte :", X_text.iloc[idx][:300], "...")
        rows.append({
            "index": idx,
            "predicted": y_pred[idx],
            "true": y_true[idx],
            "text": X_text.iloc[idx]
        })

    if save_path:
        pd.DataFrame(rows).to_csv(save_path, index=False)
        logger.info(f"Erreurs sauvegardées dans : {save_path}")


def show_class_distribution(df: pd.DataFrame, label_col: str = "label_toxic") -> None:
    """
    Affiche la répartition des classes 0/1.
    """
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    logger.info("\nRépartition des classes :")
    for label, count in counts.items():
        pct = (count / total) * 100
        print(f" - Classe {label} : {count} exemples ({pct:.2f}%)")


def compute_class_weights(df: pd.DataFrame, label_col: str = "label_toxic") -> dict:
    """
    Calcule les poids inverses pour chaque classe.
    Utile pour gérer les classes déséquilibrées.
    """
    counts = df[label_col].value_counts(normalize=True)
    weights = {cls: round(1 / prop, 2) for cls, prop in counts.items()}
    logger.info(f"Poids de classe calculés : {weights}")
    return weights

def plot_top_features(vectorizer, model, top_n=20):
    """
    Affiche les n mots les plus influents d’un modèle linéaire (ex: LogisticRegression).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not hasattr(model, "coef_"):
        raise ValueError("Le modèle ne possède pas d'attribut 'coef_'. Ce n'est pas un modèle linéaire.")

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    # Vérification de cohérence
    if len(coefs) != len(feature_names):
        print(f"Attention : nombre de coefficients ({len(coefs)}) ≠ nombre de features ({len(feature_names)}).")
        min_len = min(len(coefs), len(feature_names))
        coefs = coefs[:min_len]
        feature_names = feature_names[:min_len]

    # Sélection des indices des top_n features
    top_indices = np.argsort(np.abs(coefs))[-top_n:]
    top_features = [(feature_names[i], coefs[i]) for i in reversed(top_indices)]

    print("Mots les plus influents :")
    for word, coef in top_features:
        print(f"{word:<20} {coef:.4f}")

    # Optionnel : affichage graphique
    words, weights = zip(*top_features)
    plt.figure(figsize=(10, 5))
    plt.barh(words, weights)
    plt.xlabel("Poids")
    plt.title("Top features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

