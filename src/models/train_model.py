# src/models/train_model.py

import os
import json
import joblib
import logging
from typing import Tuple, Union, List, Optional

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import Bunch

from models.utils import get_next_version_name, save_model  

# LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONSTRUCTION DES BLOCS


def build_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int = 3
) -> TfidfVectorizer:
    """
    Initialise un TF-IDF vectorizer avec des paramètres optimaux pour la détection de toxicité.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode",
        stop_words="english"
    )


def build_model(
    class_weight: Union[str, dict] = "balanced",
    max_iter: int = 500
) -> ClassifierMixin:
    """
    Initialise un modèle de classification supervisée.
    Par défaut : Régression Logistique avec équilibrage des classes.
    """
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        solver="lbfgs"
    )

# FONCTION D’ENTRAÎNEMENT

def train_model_pipeline(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Pipeline, Bunch, dict]:
    """
    Entraîne un pipeline complet NLP : TF-IDF + (features) + modèle.
    Retourne :
    - pipeline (vectorizer + modèle)
    - bundle (X/y train/test)
    - scores (accuracy, f1, etc.)
    """

    logger.info("Démarrage de l'entraînement...")

    # Vérification
    required_cols = [text_col, label_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne {col} manquante dans le DataFrame.")

    # Target et texte
    y = df[label_col]
    X_text = df[text_col]
    
    # Features additionnelles
    X_structured = df[feature_cols] if feature_cols else None

    # Vectorisation
    logger.info("TF-IDF vectorisation en cours...")
    vectorizer = build_vectorizer()
    X_vec = vectorizer.fit_transform(X_text)

    # Fusion (texte + features)
    if X_structured is not None:
        logger.info("Fusion avec variables enrichies")
        X_final = hstack([X_vec, X_structured.values])
    else:
        X_final = X_vec

    # Split train/test
    logger.info("Split train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    # Modèle
    logger.info("Entraînement du modèle supervisé...")
    model = build_model()
    model.fit(X_train, y_train)

    # Évaluation rapide
    logger.info("Évaluation rapide...")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    logger.info(f"Scores de validation : {metrics}")

    # Pipeline scikit-learn complet (vectorizer est déjà "fit")
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])

    # Bundle pour analyse ou évaluation externe
    data_bundle = Bunch(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_pred=y_pred
    )

    return pipeline, data_bundle, metrics


# SAUVEGARDE DU MODÈLE


def save_artifacts(
    model: Pipeline,
    metrics: dict,
    output_dir: str = "models",
    model_name: str = "logreg_toxic",
    versioning: bool = True
) -> None:
    """
    Sauvegarde le modèle entraîné et les métriques associées.
    Crée automatiquement un nom unique si `versioning=True`.
    """

    os.makedirs(output_dir, exist_ok=True)

    if versioning:
        model_path = get_next_version_name(output_dir, model_name, ".pkl")
        metrics_path = model_path.replace(".pkl", "_metrics.json")
    else:
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")

    # Sauvegarde
    save_model(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Modèle sauvegardé → {model_path}")
    logger.info(f"Métriques sauvegardées → {metrics_path}")
