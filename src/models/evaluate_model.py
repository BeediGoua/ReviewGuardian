# src/models/evaluate_model.py

import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score
)

from models.utils import load_model

# LOGGING 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UTILS DE VISUALISATION 

def plot_confusion_matrix(y_true, y_pred, path, display=False):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non toxique", "Toxique"],
                yticklabels=["Non toxique", "Toxique"])
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(path)
    if display:
        plt.show()
    plt.close()


def plot_roc_curve(y_true, y_proba, path, display=False):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    if display:
        plt.show()
    plt.close()


def plot_pr_curve(y_true, y_proba, path, display=False):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Précision-Recall")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    if display:
        plt.show()
    plt.close()

# ÉVALUATION PRINCIPALE 

def evaluate_model_with_features(model_path, df_test, text_col, label_col, feature_cols, model_name):
    from models.utils import load_model
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score, classification_report
    )
    from scipy.sparse import hstack

    model = load_model(model_path)
    
    # Données d'entrée
    X_text = df_test[text_col]
    y_true = df_test[label_col]
    
    # CORRECTION: Utiliser directement le pipeline complet
    # Le pipeline gère automatiquement la vectorisation + features
    X_combined = X_text
    
    # Si features structurées, créer un DataFrame combiné
    if feature_cols:
        # Créer un DataFrame temporaire avec texte + features
        temp_df = df_test[[text_col] + feature_cols].copy()
        
        # Vectorisation manuelle pour reproduire l'entraînement
        vectorizer = model.named_steps["vectorizer"]
        classifier = model.named_steps["classifier"]
        
        X_text_vec = vectorizer.transform(X_text)
        X_feat = df_test[feature_cols].astype(float)
        X_combined = hstack([X_text_vec, X_feat.values])
        
        # Pour MultinomialNB, forcer les valeurs positives
        if hasattr(classifier, 'alpha') and X_combined.min() < 0:
            X_combined.data = np.maximum(X_combined.data, 0)
            
        # Prédiction directe avec le classifier
        y_pred = classifier.predict(X_combined)
    else:
        # Cas simple: utiliser le pipeline complet
        y_pred = model.predict(X_text)

    # Gestion des probabilités
    if feature_cols:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            y_proba = classifier.predict_proba(X_combined)[:, 1]
        else:
            raise AttributeError("Le modèle ne supporte pas `predict_proba`.")
    else:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_text)[:, 1]
        else:
            raise AttributeError("Le modèle ne supporte pas `predict_proba`.")

    # Scores
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba)
    }

    # Sauvegardes
    report_prefix = os.path.join("reports", model_name)
    os.makedirs("reports", exist_ok=True)

    # Rapport JSON
    with open(f"{report_prefix}_report.json", "w") as f:
        json.dump(classification_report(y_true, y_pred, output_dict=True), f, indent=2)

    # Visualisations
    plot_confusion_matrix(y_true, y_pred, f"{report_prefix}_confusion_matrix.png")
    plot_roc_curve(y_true, y_proba, f"{report_prefix}_roc_curve.png")
    plot_pr_curve(y_true, y_proba, f"{report_prefix}_precision_recall_curve.png")

    # Markdown
    with open(f"{report_prefix}_summary.md", "w") as md:
        md.write(f"# Évaluation du modèle `{model_name}`\n\n")
        md.write("## Scores globaux\n")
        for metric, val in scores.items():
            md.write(f"- **{metric}** : `{val:.4f}`\n")
        md.write("\n## Visualisations\n")
        md.write(f"![]({model_name}_confusion_matrix.png)\n")
        md.write(f"![]({model_name}_roc_curve.png)\n")
        md.write(f"![]({model_name}_precision_recall_curve.png)\n")
        md.write("\n---\n_Généré avec `evaluate_model_with_features()`_\n")

    logger.info(f"Rapport complet généré pour {model_name}")
    return scores



def evaluate_classifier(
    model_path: str,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    output_dir: str = "reports",
    model_name: str = "logreg_toxic",
    display: bool = False,
    return_scores: bool = False
) -> Optional[dict]:
    """
    Évalue un modèle de classification binaire.
    Génère :
    - rapport JSON
    - courbes PNG
    - markdown résumé
    """

    os.makedirs(output_dir, exist_ok=True)
    report_prefix = os.path.join(output_dir, model_name)

    logger.info(f"Chargement du modèle depuis : {model_path}")
    model = load_model(model_path)

    logger.info("Prédiction...")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError("Le modèle ne supporte pas `predict_proba`.")

    logger.info("Calcul des scores...")
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba)
    }

    # Rapport JSON
    report_path = f"{report_prefix}_report.json"
    with open(report_path, "w") as f:
        json.dump(classification_report(y_test, y_pred, output_dict=True), f, indent=2)
    logger.info(f"Rapport sauvegardé → {report_path}")

    # Graphiques
    cm_path = f"{report_prefix}_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path, display)

    roc_path = f"{report_prefix}_roc_curve.png"
    plot_roc_curve(y_test, y_proba, roc_path, display)

    pr_path = f"{report_prefix}_precision_recall_curve.png"
    plot_pr_curve(y_test, y_proba, pr_path, display)

    # Markdown résumé
    md_path = f"{report_prefix}_summary.md"
    with open(md_path, "w") as md:
        md.write(f"# Évaluation du modèle `{model_name}`\n\n")
        md.write("## Scores globaux\n")
        for metric, val in scores.items():
            md.write(f"- **{metric}** : `{val:.4f}`\n")
        md.write("\n## Visualisations\n")
        md.write(f"![]({os.path.basename(cm_path)})\n")
        md.write(f"![]({os.path.basename(roc_path)})\n")
        md.write(f"![]({os.path.basename(pr_path)})\n")
        md.write("\n---\n_Généré avec `evaluate_classifier()`_\n")

    logger.info(f"Rapport Markdown → {md_path}")

    return scores if return_scores else None
