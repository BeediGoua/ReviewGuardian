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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction principale


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
    Évalue un modèle binaire :
    Génère :
    - classification_report.json
    - confusion_matrix.png
    - ROC.png
    - Precision-Recall.png
    - summary.md

    :param model_path: chemin vers le modèle .pkl
    :param X_test: matrice de features test
    :param y_test: vecteur des vraies classes
    :param output_dir: dossier de sortie
    :param model_name: nom du modèle pour le préfixe
    :param display: si True, affiche les graphes à l'écran
    :param return_scores: si True, retourne le dict des métriques
    """

    # Sécurité
    os.makedirs(output_dir, exist_ok=True)
    report_prefix = os.path.join(output_dir, model_name)

    # Chargement modèle
    logger.info(f"📥 Chargement du modèle depuis : {model_path}")
    model = load_model(model_path)

    # Prédictions
    logger.info("🔮 Prédiction...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métriques globales
    
    logger.info("Calcul des scores")
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba)
    }

    # Rapport détaillé JSON
    report_path = f"{report_prefix}_report.json"
    classif_report = classification_report(y_test, y_pred, output_dict=True)
    with open(report_path, "w") as f:
        json.dump(classif_report, f, indent=2)
    logger.info(f"Rapport de classification sauvegardé → {report_path}")

    # Matrice de confusion
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non toxique", "Toxique"],
                yticklabels=["Non toxique", "Toxique"])
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.tight_layout()
    cm_path = f"{report_prefix}_confusion_matrix.png"
    plt.savefig(cm_path)
    if display: plt.show()
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée → {cm_path}")

    #  ROC Curve
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = scores["roc_auc"]
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    roc_path = f"{report_prefix}_roc_curve.png"
    plt.savefig(roc_path)
    if display: plt.show()
    plt.close()
    logger.info(f"Courbe ROC sauvegardée → {roc_path}")

    # Courbe Précision-Rappel
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = scores["average_precision"]
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Précision-Recall")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pr_path = f"{report_prefix}_precision_recall_curve.png"
    plt.savefig(pr_path)
    if display: plt.show()
    plt.close()
    logger.info(f"Courbe Précision-Recall sauvegardée → {pr_path}")

    # Rapport Markdown (summary)
    
    md_path = f"{report_prefix}_summary.md"
    with open(md_path, "w") as md:
        md.write(f"# Évaluation du modèle `{model_name}`\n\n")
        md.write("## Scores globaux\n")
        for metric, val in scores.items():
            md.write(f"- **{metric}** : `{val:.4f}`\n")
        md.write("\n## Matrice de confusion\n")
        md.write(f"![]({os.path.basename(cm_path)})\n")
        md.write("\n## ROC Curve\n")
        md.write(f"![]({os.path.basename(roc_path)})\n")
        md.write("\n## Precision-Recall\n")
        md.write(f"![]({os.path.basename(pr_path)})\n")
        md.write("\n---\n_Généré automatiquement avec `evaluate_classifier()`_\n")

    logger.info(f"Rapport Markdown généré → {md_path}")

    if return_scores:
        return scores
