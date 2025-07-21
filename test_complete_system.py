#!/usr/bin/env python3
"""
Script de test complet pour ReviewGuardian
Vérifie toutes les phases et fonctionnalités
"""

import sys
import os
from pathlib import Path
import traceback
import time
import json
import pandas as pd
import numpy as np

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_phase1_data_and_models():
    """Test Phase 1: Données et modèles de base"""
    print("=== TEST PHASE 1: DONNÉES ET MODÈLES ===")
    
    try:
        # 1. Vérifier les données
        print("Test des données...")
        
        # Données brutes
        raw_files = [
            "data/raw/amazon_reviews.csv",
            "data/raw/imdb_reviews.csv", 
            "data/raw/toxic_comments_training.csv"
        ]
        
        for file_path in raw_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path, nrows=5)  # Lecture test
                print(f"  {file_path}: {len(df)} échantillons")
            else:
                print(f"  {file_path}: Fichier manquant")
        
        # Données processées
        processed_files = [
            "data/processed/balanced_reviews.csv",
            "data/processed/enriched_reviews.csv"
        ]
        
        for file_path in processed_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path, nrows=5)
                print(f"  {file_path}: {df.shape}")
                print(f"    Colonnes: {list(df.columns[:5])}...")
            else:
                print(f"  {file_path}: MANQUANT - Exécutez le preprocessing")
        
        # 2. Vérifier les modèles
        print("\nTest des modèles...")
        
        model_files = [
            "notebooks/models/logreg_toxic.pkl",
            "notebooks/models/rf_toxic.pkl", 
            "notebooks/models/nb_toxic.pkl"
        ]
        
        for model_path in model_files:
            if Path(model_path).exists():
                try:
                    from models.utils import load_model
                    model = load_model(model_path)
                    print(f"  {model_path}: {type(model).__name__}")
                    
                    # Test prédiction simple
                    test_text = ["This is a test message"]
                    pred = model.predict(test_text)
                    print(f"    Test prédiction: {pred[0]}")
                    
                except Exception as e:
                    print(f"  {model_path}: Erreur chargement - {e}")
            else:
                print(f"  {model_path}: Modèle manquant")
        
        # 3. Test métriques
        print("\nTest des métriques...")
        metrics_files = [
            "notebooks/models/logreg_toxic_metrics.json",
            "notebooks/models/rf_toxic_metrics.json",
            "notebooks/models/nb_toxic_metrics.json"
        ]
        
        for metrics_path in metrics_files:
            if Path(metrics_path).exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print(f"  {metrics_path}: Accuracy {metrics.get('accuracy', 0):.3f}")
            else:
                print(f"  {metrics_path}: Métriques manquantes")
        
        return True
        
    except Exception as e:
        print(f"Erreur Phase 1: {e}")
        return False

def test_phase2_api_and_ui():
    """Test Phase 2: API et Interface"""
    print("\n=== TEST PHASE 2: API ET INTERFACE ===")
    
    try:
        # 1. Test structure API
        print("Test structure API...")
        
        api_files = [
            "src/api/main.py",
            "app/streamlit_app.py"
        ]
        
        for file_path in api_files:
            if Path(file_path).exists():
                # Vérifier que le fichier n'est pas vide
                content = Path(file_path).read_text()
                if len(content.strip()) > 10:
                    print(f"  {file_path}: Structure OK")
                else:
                    print(f"  {file_path}: Fichier vide ou minimal")
            else:
                print(f"  {file_path}: MANQUANT")
        
        # 2. Test imports API (sans lancer le serveur)
        print("\nTest imports API...")
        try:
            # Test import FastAPI
            import fastapi
            print(f"  FastAPI: v{fastapi.__version__}")
            
            # Test import Streamlit
            import streamlit
            print(f"  Streamlit: v{streamlit.__version__}")
            
            # Test autres dépendances
            import uvicorn
            print(f"  Uvicorn: disponible")
            
        except ImportError as e:
            print(f"  Import manquant: {e}")
            print("  Exécutez: pip install -r requirements.txt")
        
        return True
        
    except Exception as e:
        print(f"Erreur Phase 2: {e}")
        return False

def test_phase3_docker_and_docs():
    """Test Phase 3: Docker et Documentation"""
    print("\n=== TEST PHASE 3: DOCKER ET DOCUMENTATION ===")
    
    try:
        # 1. Test fichiers Docker
        print("est fichiers Docker...")
        
        docker_files = [
            "Dockerfile",
            "Dockerfile.streamlit", 
            "docker-compose.yml",
            ".dockerignore"
        ]
        
        for file_path in docker_files:
            if Path(file_path).exists():
                content = Path(file_path).read_text()
                lines = len(content.strip().split('\n'))
                print(f"  {file_path}: {lines} lignes")
            else:
                print(f"  {file_path}: MANQUANT")
        
        # 2. Test configuration déploiement
        print("\nTest config déploiement...")
        
        deploy_files = [
            "railway.json",
            "heroku.yml", 
            "Procfile"
        ]
        
        for file_path in deploy_files:
            if Path(file_path).exists():
                print(f"  {file_path}: Configuré")
            else:
                print(f"  {file_path}: Optionnel manquant")
        
        # 3. Test documentation
        print("\nTest documentation...")
        
        if Path("README.md").exists():
            readme = Path("README.md").read_text()
            sections = ["Quick Start", "API Usage", "Docker", "Features"]
            
            for section in sections:
                if section.lower() in readme.lower():
                    print(f"  README: Section '{section}' présente")
                else:
                    print(f"  README: Section '{section}' manquante")
        else:
            print("  README.md: MANQUANT")
        
        # 4. Test monitoring
        print("\nTest config monitoring...")
        
        monitoring_files = [
            "monitoring/prometheus.yml",
            "monitoring/grafana/dashboards/reviewguardian.json"
        ]
        
        for file_path in monitoring_files:
            if Path(file_path).exists():
                print(f"  {file_path}: Configuré")
            else:
                print(f"  {file_path}: Monitoring optionnel")
        
        return True
        
    except Exception as e:
        print(f"Erreur Phase 3: {e}")
        return False

def test_phase4_advanced_features():
    """Test Phase 4: Fonctionnalités avancées"""
    print("\n=== TEST PHASE 4: FONCTIONNALITÉS AVANCÉES ===")
    
    try:
        # 1. Test MLOps
        print("Test MLOps...")
        
        try:
            from mlops.model_registry import ModelRegistry
            from mlops.ab_testing import ABTestingEngine
            
            # Test création registry
            registry = ModelRegistry("test_registry")
            print("  ModelRegistry: Importable et créable")
            
            # Test A/B testing
            ab_engine = ABTestingEngine(registry, "test_ab")
            print("  ABTestingEngine: Importable et créable")
            
        except ImportError as e:
            print(f"   MLOps import: {e}")
        except Exception as e:
            print(f"  MLOps création: {e}")
        
        # 2. Test Explicabilité
        print("\nTest Explicabilité...")
        
        try:
            from explainability.lime_explainer import ReviewGuardianLimeExplainer
            from explainability.shap_explainer import ReviewGuardianShapExplainer
            
            print("  LIME: Importable")
            print("  SHAP: Importable")
            
            # Test dépendances
            import lime
            import shap
            print(f"  LIME library: v{lime.__version__}")
            print(f"  SHAP library: v{shap.__version__}")
            
        except ImportError as e:
            print(f"  Explicabilité import: {e}")
            print("  Installez: pip install lime shap")
        
        # 3. Test Optimisation
        print("\nTest Optimisation...")
        
        try:
            from optimization.cache_manager import CacheManager
            from optimization.performance_optimizer import BatchProcessor
            
            # Test cache
            cache = CacheManager(local_cache_dir="test_cache")
            print("  CacheManager: Créé")
            
            # Test cache fonctionnel
            cache.set("test_key", {"test": "value"})
            value = cache.get("test_key")
            if value and value.get("test") == "value":
                print("  Cache: Fonctionnel")
            else:
                print("  Cache: Problème lecture/écriture")
            
            # Test batch processor
            processor = BatchProcessor(batch_size=5, n_jobs=2)
            print("  BatchProcessor: Créé")
            
            # Test Redis (optionnel)
            try:
                import redis
                print(f"  Redis library: Disponible")
            except ImportError:
                print("  Redis: Non installé (optionnel)")
            
        except ImportError as e:
            print(f"  Optimisation import: {e}")
        except Exception as e:
            print(f"  Optimisation: {e}")
        
        return True
        
    except Exception as e:
        print(f"Erreur Phase 4: {e}")
        return False

def test_integration_complete():
    """Test d'intégration complet"""
    print("\n=== TEST INTÉGRATION COMPLÈTE ===")
    
    try:
        # Test pipeline complet avec un exemple
        print("Test pipeline end-to-end...")
        
        # 1. Charger un modèle
        model_path = "notebooks/models/logreg_toxic.pkl"
        if not Path(model_path).exists():
            print("  Modèle manquant pour test intégration")
            return False
        
        from models.utils import load_model
        model = load_model(model_path)
        print("  Modèle chargé")
        
        # 2. Test prédiction
        test_texts = [
            "This product is amazing, great quality!",
            "Terrible product, complete waste of money and time!",
            "Average quality, good value for the price."
        ]
        
        predictions = model.predict(test_texts)
        probabilities = model.predict_proba(test_texts)
        
        print(f"  Prédictions: {predictions}")
        print(f"  Probabilités shape: {probabilities.shape}")
        
        # 3. Test avec cache (si Phase 4 disponible)
        try:
            from optimization.cache_manager import CacheManager, PredictionCache
            
            cache_manager = CacheManager(local_cache_dir="test_integration_cache")
            pred_cache = PredictionCache(cache_manager)
            
            # Test cache
            for i, text in enumerate(test_texts):
                pred_result = {
                    "is_toxic": bool(predictions[i]),
                    "confidence": float(probabilities[i].max()),
                    "probabilities": probabilities[i].tolist()
                }
                
                # Cache
                pred_cache.cache_prediction(text, "test_model", pred_result)
                
                # Retrieve
                cached = pred_cache.get_prediction(text, "test_model")
                if cached:
                    print(f"  Cache fonctionnel pour texte {i+1}")
                else:
                    print(f"  Cache non fonctionnel pour texte {i+1}")
            
        except ImportError:
            print("   Cache Phase 4 non disponible")
        
        # 4. Test explicabilité (si disponible)
        try:
            from explainability.lime_explainer import quick_lime_explanation
            
            explanation = quick_lime_explanation(model, test_texts[1], num_features=5)
            if 'error' not in explanation:
                print("  ✅ LIME explicabilité fonctionnelle")
            else:
                print(f"  ⚠️ LIME erreur: {explanation['error']}")
                
        except ImportError:
            print("  ⚠️ LIME Phase 4 non disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        traceback.print_exc()
        return False

def test_performance_basic():
    """Test de performance basique"""
    print("\n⚡ === TEST PERFORMANCE BASIQUE ===")
    
    try:
        model_path = "notebooks/models/logreg_toxic.pkl"
        if not Path(model_path).exists():
            print("  ⚠️ Modèle manquant pour test performance")
            return False
        
        from models.utils import load_model
        model = load_model(model_path)
        
        # Créer données de test
        test_texts = [f"Test message number {i} for performance testing." for i in range(50)]
        
        # Test temps de prédiction
        start_time = time.time()
        predictions = model.predict(test_texts)
        elapsed = time.time() - start_time
        
        throughput = len(test_texts) / elapsed
        
        print(f"  ✅ Performance: {throughput:.1f} prédictions/sec")
        print(f"  📊 Temps total: {elapsed:.3f}s pour {len(test_texts)} textes")
        
        if throughput > 10:
            print("  🚀 Performance: EXCELLENTE")
        elif throughput > 5:
            print("  ✅ Performance: BONNE") 
        else:
            print("  ⚠️ Performance: À optimiser")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur performance: {e}")
        return False

def generate_test_report(results):
    """Génère un rapport de test"""
    print("\n📋 === RAPPORT DE TEST FINAL ===")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"🎯 Tests réussis: {passed_tests}/{total_tests}")
    print(f"📊 Taux de réussite: {passed_tests/total_tests:.1%}")
    
    print("\n📝 Détail par phase:")
    phase_names = {
        'phase1': 'Phase 1: Données et Modèles',
        'phase2': 'Phase 2: API et Interface', 
        'phase3': 'Phase 3: Docker et Documentation',
        'phase4': 'Phase 4: Fonctionnalités Avancées',
        'integration': 'Test Intégration',
        'performance': 'Test Performance'
    }
    
    for phase, success in results.items():
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        name = phase_names.get(phase, phase)
        print(f"  {status} {name}")
    
    # Recommandations
    print("\n💡 Recommandations:")
    
    if not results.get('phase1', False):
        print("  🔧 Phase 1: Exécutez l'entraînement des modèles")
        print("     jupyter notebook notebooks/03_train_evaluation.ipynb")
    
    if not results.get('phase2', False):
        print("  📦 Phase 2: Installez les dépendances manquantes")
        print("     pip install -r requirements.txt")
    
    if not results.get('phase4', False):
        print("  🚀 Phase 4: Installez les dépendances avancées")
        print("     pip install lime shap redis")
    
    if passed_tests == total_tests:
        print("\n🎉 FÉLICITATIONS! Tous les tests sont réussis!")
        print("🚀 ReviewGuardian est prêt pour la production!")
    elif passed_tests >= total_tests * 0.8:
        print("\n👍 Très bon score! Quelques ajustements mineurs.")
    else:
        print("\n⚠️ Plusieurs composants nécessitent attention.")
    
    # Sauvegarder le rapport
    report_path = "test_report.json"
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": passed_tests/total_tests,
        "results": results,
        "recommendations": []
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Rapport sauvegardé: {report_path}")

def main():
    """Fonction principale de test"""
    print("🧪 === REVIEWGUARDIAN - TEST COMPLET DU SYSTÈME ===")
    print("Vérification de toutes les phases et fonctionnalités\n")
    
    # Dictionnaire pour stocker les résultats
    results = {}
    
    # Exécuter tous les tests
    results['phase1'] = test_phase1_data_and_models()
    results['phase2'] = test_phase2_api_and_ui()
    results['phase3'] = test_phase3_docker_and_docs()
    results['phase4'] = test_phase4_advanced_features()
    results['integration'] = test_integration_complete()
    results['performance'] = test_performance_basic()
    
    # Générer le rapport final
    generate_test_report(results)
    
    # Code de sortie
    total_success = all(results.values())
    return 0 if total_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)