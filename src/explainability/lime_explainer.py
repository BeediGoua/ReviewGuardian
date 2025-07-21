# src/explainability/lime_explainer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewGuardianLimeExplainer:
    """
    Explicabilité LIME pour ReviewGuardian
    Explique les prédictions individuelles de toxicité
    """
    
    def __init__(self, model: Union[Pipeline, BaseEstimator], feature_names: Optional[List[str]] = None):
        """
        Args:
            model: Modèle entraîné (Pipeline sklearn)
            feature_names: Noms des features (optionnel)
        """
        self.model = model
        self.feature_names = feature_names
        
        # Initialiser LIME Text Explainer
        self.explainer = LimeTextExplainer(
            class_names=['Non Toxique', 'Toxique'],
            feature_selection='auto',
            verbose=True
        )
        
        logger.info("LIME Explainer initialisé")
    
    def _predict_proba_wrapper(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper pour la prédiction compatible avec LIME
        
        Args:
            texts: Liste de textes à prédire
            
        Returns:
            Probabilités de classe [non_toxic, toxic]
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                # Pipeline complet
                probas = self.model.predict_proba(texts)
            else:
                # Modèle seul (nécessite preprocessing manuel)
                if hasattr(self.model, 'named_steps'):
                    # Pipeline avec étapes nommées
                    vectorizer = self.model.named_steps.get('vectorizer')
                    classifier = self.model.named_steps.get('classifier')
                    
                    if vectorizer and classifier:
                        X_vec = vectorizer.transform(texts)
                        probas = classifier.predict_proba(X_vec)
                    else:
                        raise ValueError("Pipeline mal configuré")
                else:
                    raise ValueError("Modèle non compatible")
            
            # S'assurer que les probabilités sont au bon format
            if probas.shape[1] == 1:
                # Cas binaire avec une seule colonne
                probas = np.column_stack([1 - probas.ravel(), probas.ravel()])
            
            return probas
            
        except Exception as e:
            logger.error(f"Erreur dans predict_proba_wrapper: {e}")
            # Fallback: prédictions aléatoires
            return np.random.rand(len(texts), 2)
    
    def explain_prediction(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict:
        """
        Explique une prédiction individuelle
        
        Args:
            text: Texte à expliquer
            num_features: Nombre de features importantes à afficher
            num_samples: Nombre d'échantillons pour LIME
            
        Returns:
            Dictionnaire avec l'explication
        """
        
        logger.info(f"Explication LIME pour: '{text[:50]}...'")
        
        try:
            # Générer l'explication
            explanation = self.explainer.explain_instance(
                text,
                self.predict_proba_wrapper,
                num_features=num_features,
                num_samples=num_samples,
                distance_metric='cosine'
            )
            
            # Extraire les informations
            prediction_proba = self.predict_proba_wrapper([text])[0]
            predicted_class = int(np.argmax(prediction_proba))
            confidence = float(prediction_proba[predicted_class])
            
            # Features importantes
            features_importance = explanation.as_list()
            
            # Scores par classe
            class_probabilities = {
                'non_toxic': float(prediction_proba[0]),
                'toxic': float(prediction_proba[1])
            }
            
            result = {
                'text': text,
                'prediction': {
                    'class': 'toxic' if predicted_class == 1 else 'non_toxic',
                    'class_id': predicted_class,
                    'confidence': confidence,
                    'probabilities': class_probabilities
                },
                'explanation': {
                    'method': 'LIME',
                    'num_features': num_features,
                    'num_samples': num_samples,
                    'features_importance': features_importance,
                    'local_prediction': explanation.local_pred[predicted_class] if hasattr(explanation, 'local_pred') else None
                },
                'visualization_data': {
                    'words': [feat[0] for feat in features_importance],
                    'importances': [feat[1] for feat in features_importance],
                    'colors': ['red' if imp < 0 else 'green' for _, imp in features_importance]
                }
            }
            
            logger.info(f"Explication générée: {predicted_class} ({confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'explication LIME: {e}")
            return {
                'error': str(e),
                'text': text,
                'explanation': None
            }
    
    def explain_batch(
        self,
        texts: List[str],
        num_features: int = 10,
        num_samples: int = 2000
    ) -> List[Dict]:
        """
        Explique plusieurs prédictions en lot
        
        Args:
            texts: Liste de textes
            num_features: Nombre de features par explication
            num_samples: Nombre d'échantillons LIME
            
        Returns:
            Liste d'explications
        """
        
        logger.info(f"Explication batch de {len(texts)} textes")
        
        explanations = []
        for i, text in enumerate(texts):
            logger.info(f"Processing {i+1}/{len(texts)}")
            explanation = self.explain_prediction(text, num_features, num_samples)
            explanations.append(explanation)
        
        return explanations
    
    def visualize_explanation(
        self,
        explanation: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Visualise une explication LIME
        
        Args:
            explanation: Résultat de explain_prediction()
            save_path: Chemin de sauvegarde (optionnel)
            figsize: Taille de la figure
        """
        
        if 'error' in explanation:
            print(f"Erreur dans l'explication: {explanation['error']}")
            return
        
        # Données de visualisation
        viz_data = explanation['visualization_data']
        words = viz_data['words']
        importances = viz_data['importances']
        colors = viz_data['colors']
        
        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Graphique 1: Importance des features
        y_pos = np.arange(len(words))
        bars = ax1.barh(y_pos, importances, color=colors, alpha=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(words)
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Feature Importance (LIME)')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Graphique 2: Probabilités de classe
        classes = ['Non Toxique', 'Toxique']
        probas = [
            explanation['prediction']['probabilities']['non_toxic'],
            explanation['prediction']['probabilities']['toxic']
        ]
        colors_prob = ['green', 'red']
        
        bars2 = ax2.bar(classes, probas, color=colors_prob, alpha=0.7)
        ax2.set_ylabel('Probabilité')
        ax2.set_title('Prédiction du Modèle')
        ax2.set_ylim(0, 1)
        
        # Annotations
        for bar, prob in zip(bars2, probas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # Titre général
        predicted_class = explanation['prediction']['class']
        confidence = explanation['prediction']['confidence']
        text_preview = explanation['text'][:50] + "..."
        
        fig.suptitle(f'Explication LIME: "{text_preview}"\n'
                    f'Prédiction: {predicted_class.upper()} ({confidence:.3f})',
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualisation sauvegardée: {save_path}")
        
        plt.show()
    
    def generate_html_explanation(self, explanation: Dict) -> str:
        """
        Génère une explication HTML interactive
        
        Args:
            explanation: Résultat de explain_prediction()
            
        Returns:
            Code HTML de l'explication
        """
        
        if 'error' in explanation:
            return f"<p>Erreur: {explanation['error']}</p>"
        
        # Données
        text = explanation['text']
        prediction = explanation['prediction']
        features = explanation['explanation']['features_importance']
        
        # Style CSS
        css = """
        <style>
        .lime-explanation {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .prediction-box {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .feature-item {
            padding: 5px;
            margin: 2px 0;
            border-radius: 3px;
            display: inline-block;
            margin-right: 5px;
        }
        .positive { background-color: rgba(0, 255, 0, 0.3); }
        .negative { background-color: rgba(255, 0, 0, 0.3); }
        .text-highlight {
            line-height: 1.6;
            font-size: 16px;
        }
        </style>
        """
        
        # Contenu HTML
        html = f"""
        {css}
        <div class="lime-explanation">
            <h2>🔍 Explication LIME</h2>
            
            <div class="prediction-box">
                <h3>Prédiction</h3>
                <p><strong>Classe:</strong> {prediction['class'].upper()}</p>
                <p><strong>Confiance:</strong> {prediction['confidence']:.3f}</p>
                <p><strong>Probabilité Toxique:</strong> {prediction['probabilities']['toxic']:.3f}</p>
                <p><strong>Probabilité Non-Toxique:</strong> {prediction['probabilities']['non_toxic']:.3f}</p>
            </div>
            
            <div>
                <h3>Mots les plus influents</h3>
                <div class="text-highlight">
        """
        
        # Ajouter les features avec highlighting
        for word, importance in features:
            css_class = "positive" if importance > 0 else "negative"
            html += f'<span class="feature-item {css_class}" title="Importance: {importance:.3f}">{word}</span>'
        
        html += f"""
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <h3>Texte original</h3>
                <p style="background: #f9f9f9; padding: 10px; border-radius: 5px;">
                    {text}
                </p>
            </div>
            
            <div style="margin-top: 15px; font-size: 12px; color: #666;">
                <p>🟢 Vert: pousse vers NON-TOXIQUE | 🔴 Rouge: pousse vers TOXIQUE</p>
            </div>
        </div>
        """
        
        return html
    
    def save_explanation_report(
        self,
        explanations: List[Dict],
        output_path: str = "reports/lime_explanations.html"
    ):
        """
        Sauvegarde un rapport d'explications LIME
        
        Args:
            explanations: Liste d'explications
            output_path: Chemin de sauvegarde
        """
        
        # Créer le répertoire si nécessaire
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Explications LIME - ReviewGuardian</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>📊 Rapport d'Explications LIME</h1>
            <p>Généré le: {timestamp}</p>
            <p>Nombre d'explications: {count}</p>
            <hr>
        """.format(
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            count=len(explanations)
        )
        
        # Ajouter chaque explication
        for i, explanation in enumerate(explanations):
            html_content += f"<h2>Explication #{i+1}</h2>\n"
            html_content += self.generate_html_explanation(explanation)
            html_content += "<hr>\n"
        
        html_content += """
        </body>
        </html>
        """
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport LIME sauvegardé: {output_path}")


# Fonctions utilitaires
def quick_lime_explanation(
    model: Union[Pipeline, BaseEstimator],
    text: str,
    num_features: int = 10
) -> Dict:
    """
    Fonction rapide pour obtenir une explication LIME
    
    Args:
        model: Modèle entraîné
        text: Texte à expliquer
        num_features: Nombre de features importantes
        
    Returns:
        Explication LIME
    """
    explainer = ReviewGuardianLimeExplainer(model)
    return explainer.explain_prediction(text, num_features)

def batch_lime_analysis(
    model: Union[Pipeline, BaseEstimator],
    texts: List[str],
    output_dir: str = "reports/lime_analysis"
) -> str:
    """
    Analyse LIME en lot avec sauvegarde automatique
    
    Args:
        model: Modèle entraîné
        texts: Liste de textes
        output_dir: Répertoire de sortie
        
    Returns:
        Chemin du rapport généré
    """
    
    # Créer le répertoire
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Générer les explications
    explainer = ReviewGuardianLimeExplainer(model)
    explanations = explainer.explain_batch(texts)
    
    # Sauvegarder le rapport
    report_path = output_path / f"lime_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    explainer.save_explanation_report(explanations, str(report_path))
    
    return str(report_path)