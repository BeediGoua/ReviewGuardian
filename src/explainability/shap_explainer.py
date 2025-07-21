# src/explainability/shap_explainer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

import shap
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewGuardianShapExplainer:
    """
    Explicabilité SHAP pour ReviewGuardian
    Analyse globale et locale des contributions des features
    """
    
    def __init__(
        self, 
        model: Union[Pipeline, BaseEstimator], 
        vectorizer=None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Modèle entraîné
            vectorizer: Vectorizer pour le texte (optionnel si dans pipeline)
            feature_names: Noms des features (optionnel)
        """
        self.model = model
        self.feature_names = feature_names
        
        # Extraire les composants du pipeline
        if hasattr(model, 'named_steps'):
            self.vectorizer = model.named_steps.get('vectorizer', vectorizer)
            self.classifier = model.named_steps.get('classifier')
        else:
            self.vectorizer = vectorizer
            self.classifier = model
        
        if not self.vectorizer:
            raise ValueError("Vectorizer requis pour SHAP text analysis")
        
        # Initialiser SHAP explainer (différent selon le type de modèle)
        self.explainer = None
        self._setup_explainer()
        
        logger.info(f"SHAP Explainer initialisé pour {type(self.classifier).__name__}")
    
    def _setup_explainer(self):
        """Configure l'explainer SHAP selon le type de modèle"""
        
        classifier_name = type(self.classifier).__name__
        
        if classifier_name in ['LogisticRegression', 'LinearSVC']:
            # Modèles linéaires: utiliser LinearExplainer (plus rapide)
            self.explainer_type = 'linear'
            logger.info("Utilisation de SHAP LinearExplainer")
            
        elif classifier_name in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            # Modèles tree-based: utiliser TreeExplainer
            self.explainer_type = 'tree'
            logger.info("Utilisation de SHAP TreeExplainer")
            
        else:
            # Autres modèles: utiliser KernelExplainer (plus lent mais universel)
            self.explainer_type = 'kernel'
            logger.info("Utilisation de SHAP KernelExplainer")
    
    def _create_explainer(self, X_background: np.ndarray):
        """
        Crée l'explainer SHAP avec données de background
        
        Args:
            X_background: Données de référence (échantillon du training set)
        """
        
        if self.explainer_type == 'linear':
            # Pour modèles linéaires
            self.explainer = shap.LinearExplainer(
                self.classifier, 
                X_background,
                feature_perturbation='interventional'
            )
            
        elif self.explainer_type == 'tree':
            # Pour modèles tree-based
            self.explainer = shap.TreeExplainer(self.classifier)
            
        else:
            # KernelExplainer universel
            def predict_fn(X):
                return self.classifier.predict_proba(X)[:, 1]
            
            self.explainer = shap.KernelExplainer(
                predict_fn, 
                X_background,
                link="logit"
            )
    
    def prepare_data(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Prépare les données pour SHAP (vectorisation)
        
        Args:
            texts: Liste de textes
            
        Returns:
            (X_vectorized, feature_names)
        """
        
        # Vectorisation
        X_vec = self.vectorizer.transform(texts)
        
        # Noms des features
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            feature_names = self.vectorizer.get_feature_names_out().tolist()
        elif hasattr(self.vectorizer, 'get_feature_names'):
            feature_names = self.vectorizer.get_feature_names().tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_vec.shape[1])]
        
        return X_vec, feature_names
    
    def global_feature_importance(
        self,
        texts: List[str],
        max_display: int = 20,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Analyse globale de l'importance des features avec SHAP
        
        Args:
            texts: Textes d'entraînement ou d'évaluation
            max_display: Nombre max de features à afficher
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Dictionnaire avec importance globale
        """
        
        logger.info(f"Analyse SHAP globale sur {len(texts)} textes")
        
        # Préparer les données
        X_vec, feature_names = self.prepare_data(texts)
        
        # Créer l'explainer si pas encore fait
        if self.explainer is None:
            # Utiliser un sous-échantillon comme background
            n_background = min(100, len(texts) // 2)
            background_indices = np.random.choice(len(texts), n_background, replace=False)
            X_background = X_vec[background_indices]
            
            self._create_explainer(X_background)
        
        # Calculer les valeurs SHAP
        logger.info("Calcul des valeurs SHAP...")
        try:
            if self.explainer_type == 'linear':
                shap_values = self.explainer.shap_values(X_vec)
            elif self.explainer_type == 'tree':
                shap_values = self.explainer.shap_values(X_vec)
                # Pour modèles binaires, prendre la classe positive
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Classe toxique
            else:
                # KernelExplainer
                shap_values = self.explainer.shap_values(X_vec, nsamples=100)
        
        except Exception as e:
            logger.error(f"Erreur calcul SHAP: {e}")
            return {"error": str(e)}
        
        # Calculer l'importance moyenne absolue
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Créer le DataFrame des résultats
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Top features
        top_features = importance_df.head(max_display)
        
        # Visualisation
        if len(feature_names) > 0:
            plt.figure(figsize=(12, 8))
            
            # Summary plot
            plt.subplot(2, 1, 1)
            shap.summary_plot(
                shap_values[:100],  # Limiter pour performance
                X_vec[:100],
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            plt.title("SHAP Summary Plot - Impact des Features")
            
            # Bar plot
            plt.subplot(2, 1, 2)
            y_pos = np.arange(len(top_features))
            plt.barh(y_pos, top_features['importance'], alpha=0.7)
            plt.yticks(y_pos, top_features['feature'])
            plt.xlabel('Importance SHAP Moyenne (valeur absolue)')
            plt.title('Top Features - Importance Globale')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé: {save_path}")
            
            plt.show()
        
        # Résultats
        result = {
            'method': 'SHAP',
            'explainer_type': self.explainer_type,
            'num_samples': len(texts),
            'num_features': len(feature_names),
            'global_importance': top_features.to_dict('records'),
            'statistics': {
                'mean_abs_importance': float(mean_abs_shap.mean()),
                'std_importance': float(mean_abs_shap.std()),
                'max_importance': float(mean_abs_shap.max()),
                'min_importance': float(mean_abs_shap.min())
            }
        }
        
        logger.info("Analyse SHAP globale terminée")
        return result
    
    def explain_prediction(
        self,
        text: str,
        background_texts: Optional[List[str]] = None,
        max_features: int = 15
    ) -> Dict:
        """
        Explication SHAP d'une prédiction individuelle
        
        Args:
            text: Texte à expliquer
            background_texts: Textes de référence (optionnel)
            max_features: Nombre max de features à retourner
            
        Returns:
            Dictionnaire avec l'explication SHAP
        """
        
        logger.info(f"Explication SHAP individuelle: '{text[:50]}...'")
        
        # Préparer les données
        X_vec, feature_names = self.prepare_data([text])
        
        # Créer background si nécessaire
        if self.explainer is None:
            if background_texts is None:
                # Créer un background simple (texte vide)
                X_background = self.vectorizer.transform([""])
            else:
                X_background, _ = self.prepare_data(background_texts[:100])  # Limiter pour performance
            
            self._create_explainer(X_background)
        
        # Calculer les valeurs SHAP pour ce texte
        try:
            if self.explainer_type == 'linear':
                shap_values = self.explainer.shap_values(X_vec)[0]
            elif self.explainer_type == 'tree':
                shap_values = self.explainer.shap_values(X_vec)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1][0]  # Classe toxique, premier échantillon
                else:
                    shap_values = shap_values[0]
            else:
                shap_values = self.explainer.shap_values(X_vec, nsamples=50)[0]
        
        except Exception as e:
            logger.error(f"Erreur SHAP individuelle: {e}")
            return {"error": str(e)}
        
        # Prédiction du modèle
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([text])[0]
        else:
            proba = self.classifier.predict_proba(X_vec)[0]
        
        prediction = {
            'class': 'toxic' if np.argmax(proba) == 1 else 'non_toxic',
            'confidence': float(proba.max()),
            'probabilities': {
                'non_toxic': float(proba[0]),
                'toxic': float(proba[1])
            }
        }
        
        # Features les plus importantes
        feature_importance = [
            (feature_names[i], float(shap_values[i]))
            for i in range(len(feature_names))
            if shap_values[i] != 0
        ]
        
        # Trier par importance absolue
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_importance[:max_features]
        
        result = {
            'text': text,
            'prediction': prediction,
            'explanation': {
                'method': 'SHAP',
                'explainer_type': self.explainer_type,
                'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
                'shap_sum': float(np.sum(shap_values)),
                'features_importance': top_features,
                'num_active_features': len([x for x in shap_values if x != 0])
            },
            'visualization_data': {
                'words': [feat[0] for feat in top_features],
                'shap_values': [feat[1] for feat in top_features],
                'colors': ['red' if val > 0 else 'blue' for _, val in top_features]
            }
        }
        
        return result
    
    def waterfall_plot(
        self,
        explanation: Dict,
        save_path: Optional[str] = None,
        max_display: int = 10
    ):
        """
        Crée un waterfall plot SHAP pour une explication
        
        Args:
            explanation: Résultat de explain_prediction()
            save_path: Chemin de sauvegarde
            max_display: Nombre max de features à afficher
        """
        
        if 'error' in explanation:
            print(f"Erreur: {explanation['error']}")
            return
        
        # Extraire les données
        features = explanation['explanation']['features_importance'][:max_display]
        base_value = explanation['explanation']['base_value']
        
        if not features:
            print("Aucune feature importante trouvée")
            return
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Données pour le waterfall
        feature_names = [f[0] for f in features]
        shap_values = [f[1] for f in features]
        
        # Position de départ
        y_pos = np.arange(len(feature_names))
        cumulative = base_value
        
        # Barres horizontales
        colors = ['red' if val > 0 else 'blue' for val in shap_values]
        bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7)
        
        # Annotations
        for i, (name, value) in enumerate(features):
            ax.text(value + 0.01 if value > 0 else value - 0.01, i, 
                   f'{value:.3f}', va='center', 
                   ha='left' if value > 0 else 'right')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Contribution SHAP')
        ax.set_title(f'Waterfall Plot SHAP\nTexte: "{explanation["text"][:50]}..."')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Ajouter la prédiction
        prediction = explanation['prediction']
        ax.text(0.02, 0.98, f'Prédiction: {prediction["class"].upper()}\n'
                           f'Confiance: {prediction["confidence"]:.3f}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot sauvegardé: {save_path}")
        
        plt.show()
    
    def generate_dashboard(
        self,
        texts: List[str],
        output_dir: str = "reports/shap_analysis"
    ) -> str:
        """
        Génère un dashboard SHAP complet
        
        Args:
            texts: Textes à analyser
            output_dir: Répertoire de sortie
            
        Returns:
            Chemin du dashboard HTML
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Analyse globale
        global_analysis = self.global_feature_importance(
            texts,
            save_path=str(output_path / "shap_global.png")
        )
        
        # Analyses individuelles (échantillon)
        sample_texts = texts[:5]  # Prendre quelques exemples
        individual_analyses = []
        
        for i, text in enumerate(sample_texts):
            analysis = self.explain_prediction(text, background_texts=texts)
            individual_analyses.append(analysis)
            
            # Waterfall plot
            self.waterfall_plot(
                analysis,
                save_path=str(output_path / f"shap_waterfall_{i}.png")
            )
        
        # Générer HTML
        html_content = self._generate_html_dashboard(
            global_analysis,
            individual_analyses,
            output_path
        )
        
        dashboard_path = output_path / f"shap_dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard SHAP généré: {dashboard_path}")
        return str(dashboard_path)
    
    def _generate_html_dashboard(
        self,
        global_analysis: Dict,
        individual_analyses: List[Dict],
        output_path: Path
    ) -> str:
        """Génère le HTML du dashboard SHAP"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard SHAP - ReviewGuardian</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
                .explanation {{ margin: 15px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dashboard SHAP - ReviewGuardian</h1>
            <p>Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Analyse Globale</h2>
                <div class="metric">Méthode: {global_analysis.get('explainer_type', 'N/A')}</div>
                <div class="metric">Échantillons: {global_analysis.get('num_samples', 'N/A')}</div>
                <div class="metric">Features: {global_analysis.get('num_features', 'N/A')}</div>
                
                <h3>Top Features Globales</h3>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
        """
        
        # Ajouter les top features
        for feature in global_analysis.get('global_importance', [])[:10]:
            html += f"<tr><td>{feature['feature']}</td><td>{feature['importance']:.4f}</td></tr>"
        
        html += """
                </table>
                <img src="shap_global.png" alt="Analyse Globale SHAP">
            </div>
        """
        
        # Analyses individuelles
        for i, analysis in enumerate(individual_analyses):
            prediction = analysis['prediction']
            html += f"""
            <div class="section">
                <h2>Exemple #{i+1}</h2>
                <div class="explanation">
                    <strong>Texte:</strong> {analysis['text'][:100]}...<br>
                    <strong>Prédiction:</strong> {prediction['class'].upper()}<br>
                    <strong>Confiance:</strong> {prediction['confidence']:.3f}
                </div>
                
                <h3>Features Importantes</h3>
                <table>
                    <tr><th>Feature</th><th>Valeur SHAP</th></tr>
            """
            
            for feature, value in analysis['explanation']['features_importance'][:10]:
                color = 'red' if value > 0 else 'blue'
                html += f"<tr><td>{feature}</td><td style='color:{color}'>{value:.4f}</td></tr>"
            
            html += f"""
                </table>
                <img src="shap_waterfall_{i}.png" alt="Waterfall Plot {i}">
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


# Fonctions utilitaires
def quick_shap_analysis(
    model: Union[Pipeline, BaseEstimator],
    texts: List[str],
    individual_text: Optional[str] = None
) -> Dict:
    """
    Analyse SHAP rapide
    
    Args:
        model: Modèle entraîné
        texts: Textes pour analyse globale
        individual_text: Texte pour analyse individuelle
        
    Returns:
        Résultats SHAP
    """
    
    explainer = ReviewGuardianShapExplainer(model)
    
    result = {
        'global_analysis': explainer.global_feature_importance(texts)
    }
    
    if individual_text:
        result['individual_analysis'] = explainer.explain_prediction(
            individual_text, 
            background_texts=texts
        )
    
    return result