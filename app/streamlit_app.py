#!/usr/bin/env python3
"""
ReviewGuardian - App Streamlit
Interface web moderne pour la modération de texte avec fonctionnalités enterprise
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io
import base64

# Imports ReviewGuardian
from models.utils import load_model
from models.train_model import train_model_pipeline
from mlops.model_registry import ModelRegistry
from mlops.ab_testing import ABTestingEngine
from explainability.lime_explainer import ReviewGuardianLimeExplainer
from explainability.shap_explainer import ReviewGuardianShapExplainer
from optimization.cache_manager import CacheManager, PredictionCache

# Configuration Streamlit
st.set_page_config(
    page_title="ReviewGuardian",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .toxic-text {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    
    .clean-text {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model_registry():
    """Cache du model registry"""
    return ModelRegistry("models/registry")

@st.cache_resource
def get_cache_manager():
    """Cache manager avec fallback local"""
    return CacheManager(default_ttl=3600)

@st.cache_data
def load_sample_data():
    """Charger des données d'exemple"""
    try:
        return pd.read_csv("data/processed/balanced_reviews.csv").sample(100)
    except:
        # Données factices si fichier indisponible
        return pd.DataFrame({
            'text_clean': [
                "This product is amazing!",
                "Terrible quality, worst purchase ever",
                "Good value for money",
                "Complete waste of time and money",
                "Excellent service and fast delivery"
            ],
            'label_toxic': [0, 1, 0, 1, 0]
        })

def predict_toxicity(text, model):
    """Prédiction de toxicité avec gestion d'erreur"""
    try:
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba([text])[0]
            is_toxic = prediction[1] > 0.5
            confidence = max(prediction)
        else:
            prediction = model.predict([text])[0]
            is_toxic = bool(prediction)
            confidence = 0.8  # Valeur par défaut
        
        return {
            'is_toxic': is_toxic,
            'confidence': float(confidence),
            'label': 'TOXIQUE' if is_toxic else 'PROPRE'
        }
    except Exception as e:
        st.error(f"Erreur de prédiction: {e}")
        return {'is_toxic': False, 'confidence': 0.0, 'label': 'ERREUR'}

def main():
    # Header principal
    st.markdown('<h1 class="main-header">ReviewGuardian</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligence Artificielle pour la Modération de Texte</p>', unsafe_allow_html=True)
    
    # Sidebar - Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une fonctionnalité",
        [
            "Accueil & Demo",
            "Modération Temps Réel", 
            "Model Registry",
            "A/B Testing",
            "Explicabilité",
            "Analytics & Monitoring",
            "Performance & Cache"
        ]
    )
    
    # Pages
    if page == "Accueil & Demo":
        page_home()
    elif page == "Modération Temps Réel":
        page_moderation()
    elif page == "Model Registry":
        page_model_registry()
    elif page == "A/B Testing":
        page_ab_testing()
    elif page == "Explicabilité":
        page_explainability()
    elif page == "Analytics & Monitoring":
        page_analytics()
    elif page == "Performance & Cache":
        page_performance()

def page_home():
    """Page d'accueil avec overview du système"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Vue d'ensemble")
        st.write("""
        **ReviewGuardian** est un système de modération de texte basé sur l'IA, 
        conçu pour détecter automatiquement les contenus toxiques dans les avis clients.
        
        **Fonctionnalités clés :**
        - **Modération en temps réel** avec IA
        - **Gestion avancée des modèles** (versioning, déploiement)
        - **Tests A/B** pour comparer les performances
        - **Explicabilité** avec LIME et SHAP
        - **Monitoring** et métriques en temps réel
        - **Optimisations** avec cache intelligent
        """)
    
    with col2:
        st.header("Métriques Système")
        
        # Métriques factices pour la démo
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Modèles Actifs", "3", "+1")
            st.metric("Précision", "87.5%", "+2.1%")
        with col_b:
            st.metric("Tests A/B", "2", "0")
            st.metric("Cache Hit Rate", "89%", "+5%")
    
    # Demo rapide
    st.header("Démo Rapide")
    
    sample_texts = [
        "Ce produit est fantastique, je le recommande !",
        "Service client horrible, j'ai perdu mon temps",
        "Très bonne qualité, livraison rapide",
        "Arnaque totale, vendeur malhonnête, à éviter absolument"
    ]
    
    demo_text = st.selectbox("Choisir un texte d'exemple :", sample_texts)
    custom_text = st.text_input("Ou saisir votre propre texte :")
    
    text_to_analyze = custom_text if custom_text else demo_text
    
    if st.button("Analyser"):
        with st.spinner("Analyse en cours..."):
            # Simulation d'analyse
            time.sleep(1)
            
            # Logique simple de détection pour la démo
            toxic_keywords = ['horrible', 'arnaque', 'malhonnête', 'éviter']
            is_toxic = any(word in text_to_analyze.lower() for word in toxic_keywords)
            confidence = 0.85 if is_toxic else 0.92
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_toxic:
                    st.markdown(f'<div class="toxic-text"><strong>TOXIQUE</strong><br>Confiance: {confidence:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="clean-text"><strong>PROPRE</strong><br>Confiance: {confidence:.1%}</div>', unsafe_allow_html=True)
            
            with col2:
                # Graphique de confiance
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confiance"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if is_toxic else "green"},
                        'steps': [{'range': [0, 50], 'color': "lightgray"},
                                  {'range': [50, 100], 'color': "gray"}],
                        'threshold': {'line': {'color': "black", 'width': 4},
                                      'thickness': 0.75, 'value': 80}
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

def page_moderation():
    """Page de modération en temps réel"""
    st.header("Modération de Texte en Temps Réel")
    
    # Options de modèle
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Configuration")
        model_choice = st.selectbox(
            "Modèle à utiliser:",
            ["LogisticRegression (Prod)", "RandomForest (Test)", "NaiveBayes (Backup)"]
        )
        
        threshold = st.slider("Seuil de toxicité", 0.1, 0.9, 0.5, 0.05)
        
        batch_mode = st.checkbox("Mode batch (multiple textes)")
    
    with col1:
        st.subheader("Saisie de texte")
        
        if batch_mode:
            text_input = st.text_area(
                "Entrez plusieurs textes (un par ligne):",
                height=200,
                placeholder="Texte 1\nTexte 2\nTexte 3..."
            )
            texts = [t.strip() for t in text_input.split('\n') if t.strip()]
        else:
            text_input = st.text_area(
                "Entrez le texte à analyser:",
                height=150,
                placeholder="Tapez votre texte ici..."
            )
            texts = [text_input] if text_input.strip() else []
    
    if texts and st.button("Analyser", type="primary"):
        with st.spinner(f"Analyse de {len(texts)} texte(s)..."):
            time.sleep(0.5)  # Simulation
            
            results = []
            for i, text in enumerate(texts):
                # Simulation d'analyse
                toxic_score = np.random.random()
                is_toxic = toxic_score > threshold
                
                results.append({
                    'Texte': text[:100] + "..." if len(text) > 100 else text,
                    'Résultat': 'TOXIQUE' if is_toxic else 'PROPRE',
                    'Score': f"{toxic_score:.3f}",
                    'Confiance': f"{np.random.uniform(0.7, 0.95):.1%}"
                })
            
            # Affichage des résultats
            st.subheader("Résultats d'analyse")
            
            if len(results) == 1:
                # Affichage détaillé pour un texte
                result = results[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Statut", result['Résultat'])
                with col2:
                    st.metric("Score de toxicité", result['Score'])
                with col3:
                    st.metric("Confiance", result['Confiance'])
                
                # Visualisation
                score_val = float(result['Score'])
                fig = px.bar(
                    x=['Propre', 'Toxique'], 
                    y=[1-score_val, score_val],
                    color=['Propre', 'Toxique'],
                    color_discrete_map={'Propre': 'green', 'Toxique': 'red'}
                )
                fig.update_layout(title="Distribution des scores", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Table pour multiple textes
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Statistiques
                toxic_count = sum(1 for r in results if 'TOXIQUE' in r['Résultat'])
                st.metric("Textes toxiques détectés", f"{toxic_count}/{len(results)}")

def page_model_registry():
    """Page de gestion des modèles"""
    st.header(" Model Registry & MLOps")
    
    registry = get_model_registry()
    
    tab1, tab2, tab3 = st.tabs([" Modèles Disponibles", " Déploiement", " Entraînement"])
    
    with tab1:
        st.subheader("Modèles enregistrés")
        
        # Simulation de modèles
        models_data = [
            {"Nom": "logreg_toxic_v1", "Version": "1.0.0", "Algorithme": "LogisticRegression", 
             "Précision": "87.5%", "Statut": " Production", "Date": "2025-01-15"},
            {"Nom": "rf_toxic_v2", "Version": "2.1.0", "Algorithme": "RandomForest", 
             "Précision": "89.2%", "Statut": " Test", "Date": "2025-01-18"},
            {"Nom": "nb_toxic_v1", "Version": "1.0.0", "Algorithme": "NaiveBayes", 
             "Précision": "82.1%", "Statut": " Deprecated", "Date": "2025-01-10"}
        ]
        
        df_models = pd.DataFrame(models_data)
        st.dataframe(df_models, use_container_width=True)
        
        # Actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(" Comparer Modèles"):
                fig = px.bar(
                    x=[m["Nom"] for m in models_data],
                    y=[float(m["Précision"].replace('%', '')) for m in models_data],
                    title="Comparaison des performances"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            selected_model = st.selectbox("Modèle à promouvoir:", [m["Nom"] for m in models_data])
            if st.button(" Promouvoir en Production"):
                st.success(f"Modèle {selected_model} promu en production!")
        
        with col3:
            if st.button(" Nettoyer Anciens Modèles"):
                st.info("Suppression des modèles obsolètes...")
    
    with tab2:
        st.subheader("Déploiement de modèles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Configuration de déploiement**")
            
            deployment_target = st.selectbox(
                "Environnement cible:",
                ["Production", "Staging", "Development"]
            )
            
            scaling_config = st.selectbox(
                "Configuration scaling:",
                ["Auto-scaling", "Fixed 2 instances", "Fixed 5 instances"]
            )
            
            health_check = st.checkbox("Activer health checks", True)
            monitoring = st.checkbox("Activer monitoring", True)
            
            if st.button(" Déployer"):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)
                st.success(f" Déploiement réussi sur {deployment_target}!")
        
        with col2:
            st.write("**Statut des déploiements**")
            
            deployments = [
                {"Env": "Production", "Modèle": "logreg_v1", "Statut": " Healthy", "CPU": "45%", "RAM": "2.1GB"},
                {"Env": "Staging", "Modèle": "rf_v2", "Statut": " Testing", "CPU": "23%", "RAM": "1.8GB"},
                {"Env": "Development", "Modèle": "nb_v1", "Statut": " Stopped", "CPU": "0%", "RAM": "0GB"}
            ]
            
            for dep in deployments:
                with st.container():
                    st.write(f"**{dep['Env']}** - {dep['Modèle']}")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.write(dep['Statut'])
                    with col_b:
                        st.write(f"CPU: {dep['CPU']}")
                    with col_c:
                        st.write(f"RAM: {dep['RAM']}")
                    st.divider()
    
    with tab3:
        st.subheader("Entraînement de nouveaux modèles")
        
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Nom du modèle:", "nouveau_modele_v1")
                algorithm = st.selectbox("Algorithme:", ["LogisticRegression", "RandomForest", "NaiveBayes", "SVM"])
                dataset = st.selectbox("Dataset:", ["balanced_reviews.csv", "enriched_reviews.csv", "full_dataset.csv"])
            
            with col2:
                train_split = st.slider("Split train/test", 0.6, 0.9, 0.8, 0.05)
                cv_folds = st.number_input("Cross-validation folds", 3, 10, 5)
                auto_tune = st.checkbox("Auto-hyperparameter tuning", True)
            
            submitted = st.form_submit_button(" Lancer l'entraînement")
            
            if submitted:
                with st.spinner("Entraînement en cours..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        "Chargement des données...",
                        "Préprocessing...", 
                        "Split train/test...",
                        "Entraînement du modèle...",
                        "Cross-validation...",
                        "Évaluation finale...",
                        "Sauvegarde du modèle..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        time.sleep(0.5)
                        progress_bar.progress((i + 1) / len(steps))
                    
                    # Résultats simulés
                    metrics = {
                        "Accuracy": np.random.uniform(0.82, 0.91),
                        "Precision": np.random.uniform(0.80, 0.89),
                        "Recall": np.random.uniform(0.78, 0.88),
                        "F1-Score": np.random.uniform(0.79, 0.87)
                    }
                    
                    st.success(" Entraînement terminé avec succès!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for metric, value in metrics.items():
                            st.metric(metric, f"{value:.3f}")
                    
                    with col2:
                        fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), 
                                     title="Métriques du modèle")
                        st.plotly_chart(fig, use_container_width=True)

def page_ab_testing():
    """Page de tests A/B"""
    st.header(" Tests A/B & Expérimentation")
    
    tab1, tab2, tab3 = st.tabs([" Nouveau Test", " Tests Actifs", " Résultats"])
    
    with tab1:
        st.subheader("Créer un nouveau test A/B")
        
        with st.form("ab_test_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                test_name = st.text_input("Nom du test:", "Comparaison_LogReg_vs_RF")
                description = st.text_area("Description:", "Test comparant LogisticRegression vs RandomForest")
                
                model_a = st.selectbox("Modèle A (Contrôle):", ["logreg_toxic_v1", "rf_toxic_v1", "nb_toxic_v1"])
                model_b = st.selectbox("Modèle B (Challenge):", ["logreg_toxic_v2", "rf_toxic_v2", "nb_toxic_v2"])
            
            with col2:
                traffic_split = st.slider("% trafic vers B", 10, 50, 30)
                duration = st.number_input("Durée (jours)", 1, 30, 7)
                
                metrics = st.multiselect(
                    "Métriques de succès:",
                    ["accuracy", "precision", "recall", "f1_score", "response_time"],
                    default=["accuracy", "precision"]
                )
                
                min_sample = st.number_input("Échantillon minimum", 100, 10000, 1000)
            
            if st.form_submit_button(" Lancer le test A/B"):
                test_data = {
                    "name": test_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "traffic_split": traffic_split,
                    "duration": duration,
                    "metrics": metrics,
                    "status": "Active",
                    "start_date": datetime.now().strftime("%Y-%m-%d"),
                    "samples_a": 0,
                    "samples_b": 0
                }
                
                st.success(f" Test A/B '{test_name}' créé avec succès!")
                st.json(test_data)
    
    with tab2:
        st.subheader("Tests A/B en cours")
        
        # Simulation de tests actifs
        active_tests = [
            {
                "Test": "LogReg_vs_RF_v2",
                "Modèle A": "logreg_v1 (70%)",
                "Modèle B": "rf_v2 (30%)", 
                "Échantillons": "2,450 / 5,000",
                "Durée restante": "3 jours",
                "Statut": " En cours"
            },
            {
                "Test": "Threshold_Optimization",
                "Modèle A": "rf_v2 (seuil=0.5)",
                "Modèle B": "rf_v2 (seuil=0.7)",
                "Échantillons": "1,200 / 3,000", 
                "Durée restante": "5 jours",
                "Statut": " Démarrage"
            }
        ]
        
        df_tests = pd.DataFrame(active_tests)
        st.dataframe(df_tests, use_container_width=True)
        
        # Contrôles
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Suspendre Test"):
                st.warning("Test suspendu temporairement")
        
        with col2:
            if st.button(" Arrêter Test"):
                st.error("Test arrêté - données sauvegardées")
        
        with col3:
            if st.button(" Résultats Intermédiaires"):
                st.info("Analyse des résultats partiels...")
    
    with tab3:
        st.subheader("Résultats des tests terminés")
        
        # Simulation de résultats
        test_results = {
            "Test": "LogReg_vs_NB_Jan2025",
            "Échantillons": {"A": 3500, "B": 1500},
            "Métriques": {
                "Accuracy": {"A": 0.875, "B": 0.821},
                "Precision": {"A": 0.883, "B": 0.798},
                "Recall": {"A": 0.867, "B": 0.845},
                "Response Time": {"A": "45ms", "B": "23ms"}
            },
            "Recommandation": "Garder Modèle A (LogReg)",
            "Significance": "p < 0.01 "
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Résumé du test**")
            st.write(f"Test: {test_results['Test']}")
            st.write(f"Échantillons A: {test_results['Échantillons']['A']:,}")
            st.write(f"Échantillons B: {test_results['Échantillons']['B']:,}")
            st.write(f"Recommandation: {test_results['Recommandation']}")
            st.write(f"Significativité: {test_results['Significance']}")
        
        with col2:
            # Graphique comparatif
            metrics_names = list(test_results["Métriques"].keys())[:-1]  # Sans Response Time
            values_a = [test_results["Métriques"][m]["A"] for m in metrics_names]
            values_b = [test_results["Métriques"][m]["B"] for m in metrics_names]
            
            fig = go.Figure(data=[
                go.Bar(name='Modèle A', x=metrics_names, y=values_a),
                go.Bar(name='Modèle B', x=metrics_names, y=values_b)
            ])
            fig.update_layout(title="Comparaison des performances", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Détails
        st.subheader("Analyse détaillée")
        
        insights = [
            " Modèle A (LogReg) surpasse B sur accuracy (+5.4 points)",
            " Precision significativement meilleure pour A (+8.5 points)",
            " Modèle B plus rapide (-22ms) mais moins précis",
            " Différences statistiquement significatives (p < 0.01)",
            " Recommandation: Conserver Modèle A en production"
        ]
        
        for insight in insights:
            st.write(insight)

def page_explainability():
    """Page d'explicabilité"""
    st.header(" Explicabilité & Interprétation")
    
    tab1, tab2, tab3 = st.tabs([" LIME", " SHAP", " Feature Importance"])
    
    with tab1:
        st.subheader("LIME - Explications Locales")
        st.write("Comprenez pourquoi un texte spécifique a été classé comme toxique ou propre.")
        
        # Texte d'exemple
        example_texts = [
            "Ce produit est absolument terrible et une perte d'argent complète!",
            "Excellente qualité et livraison rapide, je recommande ce vendeur.",
            "Service client horrible, j'ai perdu mon temps avec ces idiots."
        ]
        
        selected_text = st.selectbox("Choisir un exemple:", example_texts)
        custom_text = st.text_area("Ou saisir votre texte:", height=100)
        
        text_to_explain = custom_text if custom_text.strip() else selected_text
        
        if st.button(" Expliquer avec LIME"):
            with st.spinner("Génération de l'explication LIME..."):
                time.sleep(2)  # Simulation
                
                # Simulation d'explication LIME
                words = text_to_explain.split()
                explanations = []
                
                for word in words:
                    # Score simulé basé sur des mots-clés
                    if word.lower() in ['terrible', 'horrible', 'idiots', 'perte']:
                        score = -np.random.uniform(0.3, 0.8)
                    elif word.lower() in ['excellente', 'recommande', 'qualité', 'rapide']:
                        score = np.random.uniform(0.2, 0.6)
                    else:
                        score = np.random.uniform(-0.1, 0.1)
                    
                    explanations.append((word, score))
                
                # Prédiction
                avg_score = np.mean([exp[1] for exp in explanations])
                is_toxic = avg_score < -0.1
                confidence = abs(avg_score) + 0.5
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Mots contribuant à la classification:**")
                    
                    for word, score in sorted(explanations, key=lambda x: abs(x[1]), reverse=True)[:10]:
                        if abs(score) > 0.05:  # Seulement les mots significatifs
                            color = "red" if score < 0 else "green"
                            intensity = min(abs(score) * 2, 1.0)
                            st.markdown(
                                f'<span style="background-color: {color}; opacity: {intensity}; '
                                f'color: white; padding: 2px 5px; margin: 2px; border-radius: 3px;">'
                                f'{word} ({score:+.2f})</span>',
                                unsafe_allow_html=True
                            )
                
                with col2:
                    st.metric(
                        "Prédiction",
                        "TOXIQUE" if is_toxic else "PROPRE",
                        f"Confiance: {confidence:.1%}"
                    )
                    
                    # Graphique de contribution
                    top_words = sorted(explanations, key=lambda x: abs(x[1]), reverse=True)[:5]
                    fig = px.bar(
                        x=[w[1] for w in top_words],
                        y=[w[0] for w in top_words],
                        orientation='h',
                        title="Top 5 contributions",
                        color=[w[1] for w in top_words],
                        color_continuous_scale="RdYlGn"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("SHAP - Analyse Globale")
        st.write("Comprenez l'importance des features sur l'ensemble du dataset.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_size = st.slider("Taille de l'échantillon", 100, 1000, 500)
            analysis_type = st.selectbox(
                "Type d'analyse:",
                ["Feature Importance", "Summary Plot", "Dependence Plot"]
            )
        
        with col2:
            if st.button(" Lancer l'analyse SHAP"):
                with st.spinner("Calcul des valeurs SHAP..."):
                    time.sleep(3)  # Simulation
                    
                    # Simulation de features importantes
                    features = [
                        "sentiment_polarity", "flesch_score", "capital_ratio",
                        "nb_exclamations", "has_url", "word_count", "toxicity_words"
                    ]
                    
                    importance_values = np.random.exponential(0.5, len(features))
                    importance_values = sorted(importance_values, reverse=True)
                    
                    fig = px.bar(
                        x=importance_values,
                        y=features,
                        orientation='h',
                        title=f"Feature Importance (SHAP) - {dataset_size} échantillons"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(" Analyse SHAP terminée!")
                    
                    # Insights
                    st.write("**Insights clés:**")
                    st.write("• La polarité de sentiment est le facteur le plus important")
                    st.write("• Le score de lisibilité (Flesch) influence fortement la classification")
                    st.write("• Le ratio de majuscules est un indicateur de toxicité")
                    st.write("• Le nombre d'exclamations corrèle avec l'agressivité")
    
    with tab3:
        st.subheader(" Importance des Features")
        
        # Simulation de l'importance des features pour différents modèles
        models = ["LogisticRegression", "RandomForest", "NaiveBayes"]
        features = [
            "sentiment_polarity", "flesch_score", "capital_ratio", 
            "nb_exclamations", "has_url", "word_count", "has_emoji", "toxicity_keywords"
        ]
        
        # Données simulées
        importance_data = {}
        for model in models:
            importance_data[model] = np.random.exponential(0.3, len(features))
            importance_data[model] = importance_data[model] / np.sum(importance_data[model])
        
        # Graphique comparatif
        fig = go.Figure()
        
        for model in models:
            fig.add_trace(go.Bar(
                name=model,
                x=features,
                y=importance_data[model],
                text=[f"{val:.2%}" for val in importance_data[model]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Comparaison de l'importance des features par modèle",
            xaxis_title="Features",
            yaxis_title="Importance relative",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table détaillée
        st.subheader("Tableau détaillé")
        
        df_importance = pd.DataFrame(importance_data, index=features)
        df_importance = df_importance.round(4)
        
        # Formatter en pourcentages
        styled_df = df_importance.style.format("{:.2%}").background_gradient(cmap='RdYlGn')
        st.dataframe(styled_df, use_container_width=True)

def page_analytics():
    """Page d'analytics et monitoring"""
    st.header(" Analytics & Monitoring")
    
    tab1, tab2, tab3 = st.tabs([" Métriques Temps Réel", " Rapports", " Alertes"])
    
    with tab1:
        st.subheader("Tableau de bord en temps réel")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Prédictions/jour",
                "12,450",
                delta="1,250",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Taux de toxicité",
                "15.2%",
                delta="-2.1%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Temps de réponse",
                "45ms",
                delta="-5ms",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Disponibilité",
                "99.9%",
                delta="0.1%",
                delta_color="normal"
            )
        
        # Graphiques temps réel
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique de prédictions par heure
            hours = list(range(24))
            predictions = [np.random.poisson(500) for _ in hours]
            
            fig = px.line(
                x=hours,
                y=predictions,
                title="Prédictions par heure (24h)",
                labels={"x": "Heure", "y": "Nombre de prédictions"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution toxique/propre
            toxic_data = pd.DataFrame({
                'Type': ['Propre', 'Toxique'],
                'Count': [84.8, 15.2]
            })
            
            fig = px.pie(
                toxic_data,
                values='Count',
                names='Type',
                title="Distribution Toxique/Propre",
                color_discrete_map={'Propre': 'green', 'Toxique': 'red'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Métriques par modèle
        st.subheader("Performance par modèle")
        
        model_metrics = pd.DataFrame({
            'Modèle': ['LogReg_v1', 'RandomForest_v2', 'NaiveBayes_v1'],
            'Requests': [8500, 2800, 1150],
            'Accuracy': [87.5, 89.2, 82.1],
            'Avg_Response_Time': [45, 78, 23],
            'Error_Rate': [0.2, 0.1, 0.4]
        })
        
        st.dataframe(model_metrics, use_container_width=True)
        
        # Auto-refresh
        if st.checkbox("Auto-refresh (10s)"):
            time.sleep(1)
            st.rerun()
    
    with tab2:
        st.subheader("Rapports d'analyse")
        
        # Sélection de période
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Période d'analyse:",
                value=[datetime.now() - timedelta(days=7), datetime.now()],
                max_value=datetime.now()
            )
        
        with col2:
            report_type = st.selectbox(
                "Type de rapport:",
                ["Performance", "Usage", "Erreurs", "Business Impact"]
            )
        
        with col3:
            granularity = st.selectbox(
                "Granularité:",
                ["Hourly", "Daily", "Weekly"]
            )
        
        if st.button(" Générer Rapport"):
            with st.spinner("Génération du rapport..."):
                time.sleep(2)
                
                # Rapport simulé basé sur le type
                if report_type == "Performance":
                    st.subheader("Rapport de Performance")
                    
                    # Métriques de performance
                    perf_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Response Time'],
                        'Current': [87.5, 85.2, 89.1, 87.1, 45],
                        'Previous': [85.8, 83.7, 87.4, 85.5, 52],
                        'Change': ['+1.7', '+1.5', '+1.7', '+1.6', '-7ms']
                    }
                    
                    df_perf = pd.DataFrame(perf_data)
                    st.dataframe(df_perf, use_container_width=True)
                    
                    # Évolution temporelle
                    days = list(range(1, 8))
                    accuracy_trend = [85.2, 85.8, 86.1, 86.7, 87.0, 87.2, 87.5]
                    
                    fig = px.line(
                        x=days,
                        y=accuracy_trend,
                        title="Évolution de l'accuracy (7 derniers jours)",
                        labels={"x": "Jour", "y": "Accuracy (%)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif report_type == "Usage":
                    st.subheader("Rapport d'Usage")
                    
                    # Volume par jour
                    days = [f"Jour {i}" for i in range(1, 8)]
                    volumes = [np.random.poisson(12000) for _ in days]
                    
                    fig = px.bar(
                        x=days,
                        y=volumes,
                        title="Volume de prédictions par jour"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top sources
                    sources_data = pd.DataFrame({
                        'Source': ['Web App', 'API Direct', 'Batch Process', 'Mobile App', 'Webhook'],
                        'Requests': [45000, 23000, 12000, 8500, 3200],
                        'Percentage': [49.2, 25.1, 13.1, 9.3, 3.5]
                    })
                    
                    st.subheader("Répartition par source")
                    st.dataframe(sources_data, use_container_width=True)
        
        # Export options
        st.subheader("Options d'export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Export PDF"):
                st.success("Rapport PDF généré!")
        
        with col2:
            if st.button(" Export Excel"):
                st.success("Fichier Excel téléchargé!")
        
        with col3:
            if st.button(" Envoyer par email"):
                st.success("Rapport envoyé!")
    
    with tab3:
        st.subheader(" Système d'alertes")
        
        # Alertes actives
        st.write("**Alertes actives**")
        
        alerts = [
            {"Type": " Critical", "Message": "Accuracy dropped below 85%", "Time": "Il y a 2h"},
            {"Type": " Warning", "Message": "Response time > 100ms", "Time": "Il y a 30min"},
            {"Type": " Info", "Message": "New model deployed successfully", "Time": "Il y a 1h"}
        ]
        
        for alert in alerts:
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.write(alert["Type"])
                with col2:
                    st.write(alert["Message"])
                with col3:
                    st.write(alert["Time"])
                st.divider()
        
        # Configuration des alertes
        st.subheader(" Configuration des alertes")
        
        with st.form("alert_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Seuils de performance**")
                accuracy_threshold = st.slider("Accuracy minimum (%)", 70, 95, 85)
                response_time_threshold = st.slider("Temps de réponse max (ms)", 50, 500, 100)
                error_rate_threshold = st.slider("Taux d'erreur max (%)", 0.1, 5.0, 1.0)
            
            with col2:
                st.write("**Notifications**")
                email_alerts = st.checkbox("Alertes email", True)
                slack_alerts = st.checkbox("Notifications Slack", False)
                sms_alerts = st.checkbox("SMS urgents", False)
                
                alert_frequency = st.selectbox(
                    "Fréquence de vérification:",
                    ["1 minute", "5 minutes", "15 minutes", "1 heure"]
                )
            
            if st.form_submit_button(" Sauvegarder"):
                st.success("Configuration des alertes sauvegardée!")

def page_performance():
    """Page de performance et cache"""
    st.header(" Performance & Optimisation")
    
    tab1, tab2, tab3 = st.tabs([" Métriques Performance", " Cache Management", " Optimisations"])
    
    with tab1:
        st.subheader("Métriques de performance")
        
        # Métriques système
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CPU Usage", "45%", "-5%")
        with col2:
            st.metric("Memory", "2.8GB", "+0.2GB")
        with col3:
            st.metric("Requests/sec", "124", "+12")
        with col4:
            st.metric("Avg Latency", "45ms", "-8ms")
        
        # Benchmark par taille de batch
        st.subheader("Benchmark par taille de batch")
        
        batch_data = pd.DataFrame({
            'Batch Size': [1, 10, 25, 50, 100, 200],
            'Throughput (req/s)': [25, 89, 165, 234, 298, 345],
            'Latency (ms)': [45, 52, 48, 43, 38, 35],
            'Memory (MB)': [120, 145, 180, 230, 340, 520]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                batch_data,
                x='Batch Size',
                y='Throughput (req/s)',
                title="Throughput vs Batch Size"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                batch_data,
                x='Batch Size', 
                y='Latency (ms)',
                title="Latency vs Batch Size",
                color_discrete_sequence=['red']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(batch_data, use_container_width=True)
        
        # Recommandations
        st.subheader(" Recommandations")
        st.write("• **Batch size optimal**: 100 (meilleur compromise throughput/latency)")
        st.write("• **Memory scaling**: Linéaire avec la taille de batch")
        st.write("• **Latency**: Diminue avec des batches plus importants")
        st.write("• **Throughput**: Croissance logarithmique, saturation vers 200")
    
    with tab2:
        st.subheader(" Gestion du cache")
        
        cache_manager = get_cache_manager()
        
        # Statistiques du cache
        col1, col2, col3, col4 = st.columns(4)
        
        # Simulation de stats
        cache_stats = {
            'hits': np.random.randint(5000, 8000),
            'misses': np.random.randint(1000, 2000),
            'writes': np.random.randint(1500, 2500),
            'errors': np.random.randint(0, 50)
        }
        
        total_requests = cache_stats['hits'] + cache_stats['misses']
        hit_rate = cache_stats['hits'] / total_requests * 100 if total_requests > 0 else 0
        
        with col1:
            st.metric("Cache Hits", f"{cache_stats['hits']:,}")
        with col2:
            st.metric("Cache Misses", f"{cache_stats['misses']:,}")
        with col3:
            st.metric("Hit Rate", f"{hit_rate:.1f}%")
        with col4:
            st.metric("Errors", cache_stats['errors'])
        
        # Graphique hit rate dans le temps
        hours = list(range(24))
        hit_rates = [np.random.uniform(70, 95) for _ in hours]
        
        fig = px.line(
            x=hours,
            y=hit_rates,
            title="Hit Rate par heure (24h)",
            labels={"x": "Heure", "y": "Hit Rate (%)"}
        )
        fig.add_hline(y=85, line_dash="dash", annotation_text="Objectif: 85%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Configuration du cache
        st.subheader(" Configuration du cache")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ttl = st.slider("TTL par défaut (minutes)", 5, 120, 60)
            max_size = st.number_input("Taille max (MB)", 100, 2000, 500)
            
            if st.button(" Vider le cache"):
                st.success("Cache vidé avec succès!")
        
        with col2:
            st.write("**Statistiques par type de cache:**")
            
            cache_types = pd.DataFrame({
                'Type': ['Prédictions', 'Modèles', 'Features', 'Metadata'],
                'Taille (MB)': [234, 145, 67, 23],
                'Hit Rate (%)': [89.2, 95.1, 76.4, 92.3],
                'TTL (min)': [60, 1440, 30, 120]
            })
            
            st.dataframe(cache_types, use_container_width=True)
        
        # Top clés
        st.subheader(" Top des clés cachées")
        
        top_keys = pd.DataFrame({
            'Clé': [
                'model:logreg_v1:predictions',
                'features:tfidf_vectorizer', 
                'metadata:model_registry',
                'batch:processed_texts_1000',
                'shap:explanations_cache'
            ],
            'Hits': [2450, 1890, 1234, 987, 654],
            'Taille (KB)': [1234, 5678, 234, 8901, 3456],
            'Dernière utilisation': [
                "Il y a 2min", "Il y a 15min", "Il y a 1h", 
                "Il y a 3h", "Il y a 6h"
            ]
        })
        
        st.dataframe(top_keys, use_container_width=True)
    
    with tab3:
        st.subheader(" Optimisations système")
        
        # Paramètres d'optimisation
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Configuration des workers**")
            
            num_workers = st.slider("Nombre de workers", 1, 8, 4)
            worker_memory = st.slider("Mémoire par worker (GB)", 1, 8, 2)
            batch_size = st.slider("Taille de batch", 10, 200, 50)
            
            if st.button(" Appliquer config"):
                with st.spinner("Redémarrage des workers..."):
                    time.sleep(2)
                st.success("Configuration appliquée!")
        
        with col2:
            st.write("**Optimisations activées**")
            
            optimizations = {
                "Cache intelligent": True,
                "Compression des réponses": True,
                "Pool de connexions": True,
                "Batch processing": True,
                "Load balancing": False,
                "Auto-scaling": False
            }
            
            for opt, enabled in optimizations.items():
                status = "" if enabled else ""
                st.write(f"{status} {opt}")
        
        # Monitoring des ressources
        st.subheader(" Monitoring des ressources")
        
        # Simulation de données de monitoring
        times = list(range(60))  # 60 dernières minutes
        cpu_usage = [np.random.uniform(30, 70) for _ in times]
        memory_usage = [np.random.uniform(40, 80) for _ in times]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=cpu_usage, name='CPU %', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=times, y=memory_usage, name='Memory %', line=dict(color='red')))
        
        fig.update_layout(
            title="Usage des ressources (60 dernières minutes)",
            xaxis_title="Minutes",
            yaxis_title="Usage (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Profiling
        st.subheader(" Profiling & Debugging")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Memory Profiling"):
                with st.spinner("Analyse mémoire..."):
                    time.sleep(1)
                st.success("Rapport de profiling généré!")
        
        with col2:
            if st.button("Performance Profiling"):
                with st.spinner("Analyse performance..."):
                    time.sleep(1)
                st.success("Rapport de performance généré!")
        
        with col3:
            if st.button(" Debug Mode"):
                st.info("Mode debug activé - logs détaillés")

if __name__ == "__main__":
    main()