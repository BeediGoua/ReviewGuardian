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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from models.utils import get_next_version_name, save_model
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from typing import Tuple, Union, List, Optional
from scipy.sparse import hstack

# LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CHOIX DU MODÈLE


def get_model(name: str) -> ClassifierMixin:
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(class_weight='balanced', max_iter=500, solver="lbfgs")
    elif name == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight='balanced')
    elif name == "nb":
        return MultinomialNB()
    else:
        raise ValueError(f"Modèle inconnu : {name}. Essayez 'logreg', 'rf', ou 'nb'.")

# TF-IDF VECTORIZER


def build_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int = 3
) -> TfidfVectorizer:
    """
    Initialise un TF-IDF vectorizer avec des paramètres adaptés au NLP.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode",
        stop_words="english"
    )

# ENTRAÎNEMENT PRINCIPAL


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils import Bunch

def build_vectorizer(model_name: str = "logreg"):
    """
    Retourne un vectorizer adapté au modèle (TF-IDF ou CountVectorizer).
    """
    if model_name == "nb":
        return CountVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=3,
            stop_words="english",
            strip_accents="unicode"
        )
    else:
        return TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=3,
            stop_words="english",
            strip_accents="unicode"
        )

def train_model_pipeline(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str = "logreg",
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Pipeline, Bunch, dict]:
    """
    Entraîne un pipeline NLP supervisé (Vectorizer + modèle + features optionnelles).
    """

    logger.info(f"Entraînement pipeline pour modèle `{model_name}`")

    # Vérification colonnes
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    y = df[label_col]
    X_text = df[text_col]
    X_structured = df[feature_cols].astype(float) if feature_cols else None

    # Vectorisation
    vectorizer = build_vectorizer(model_name)
    logger.info(f"Vectorisation avec {'CountVectorizer' if model_name == 'nb' else 'TF-IDF'}")
    X_vec = vectorizer.fit_transform(X_text)

    # Fusion texte + features structurées
    if X_structured is not None:
        logger.info("Fusion avec variables enrichies...")
        X_combined = hstack([X_vec, X_structured.values])
    else:
        X_combined = X_vec

    # 
    if model_name == "nb":
        logger.info("Forçage des valeurs positives (MultinomialNB)")
        X_combined.data = np.maximum(X_combined.data, 0)

    # Split
    logger.info("Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Taille du train : {X_train.shape}, test : {X_test.shape}")

    # Modèle
    try:
        model = get_model(model_name)
    except ValueError as e:
        logger.error(f"Erreur dans le nom du modèle : {e}")
        raise

    logger.info("Entraînement en cours...")
    model.fit(X_train, y_train)

    # Évaluation rapide
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            logger.warning("Impossible de calculer l'AUC")

    logger.info(f"Scores : {metrics}")

    # Pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])

    # Données debug
    data_bundle = Bunch(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_pred=y_pred
    )

    return pipeline, data_bundle, metrics


# SAUVEGARDE ARTIFACTS


def save_artifacts(
    model: Pipeline,
    metrics: dict,
    output_dir: str = "models",
    model_name: str = "logreg_toxic",
    versioning: bool = True
) -> None:
    """
    Sauvegarde le pipeline + métriques associées (en JSON).
    """

    os.makedirs(output_dir, exist_ok=True)

    if versioning:
        model_path = get_next_version_name(output_dir, model_name, ".pkl")
        metrics_path = model_path.replace(".pkl", "_metrics.json")
    else:
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")

    save_model(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Modèle sauvegardé : {model_path}")
    logger.info(f"Métriques sauvegardées : {metrics_path}")

#  OPTIONNEL : pipeline direct


def train_and_save(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    model_name: str = "logreg",
    output_dir: str = "models"
):
    """
    Version compacte pour entraîner et sauvegarder en un seul appel.
    """
    pipe, data, metrics = train_model_pipeline(
        df=df,
        text_col=text_col,
        label_col=label_col,
        feature_cols=feature_cols,
        model_name=model_name
    )
    save_artifacts(pipe, metrics, output_dir=output_dir, model_name=model_name)
    return pipe, data, metrics
