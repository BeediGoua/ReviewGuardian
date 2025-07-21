#!/usr/bin/env python3
"""
Exemples d'utilisation des fonctionnalités avancées de la Phase 4
ReviewGuardian - Features Enterprise
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Imports Phase 4
from mlops.model_registry import ModelRegistry, create_model_metadata
from mlops.ab_testing import ABTestingEngine
from explainability.lime_explainer import ReviewGuardianLimeExplainer, quick_lime_explanation
from explainability.shap_explainer import ReviewGuardianShapExplainer, quick_shap_analysis
from optimization.cache_manager import CacheManager, PredictionCache, cached_prediction
from optimization.performance_optimizer import (
    BatchProcessor, PredictionOptimizer, SparseMatrixOptimizer,
    benchmark_prediction_performance
)

# Imports existants
from models.utils import load_model
from models.train_model import train_model_pipeline

def demo_model_registry():
    """Démonstration du Model Registry"""
    print("=== DEMO MODEL REGISTRY ===")
    
    # Initialiser le registry
    registry = ModelRegistry("models/registry_demo")
    
    # Charger des données pour l'exemple
    df = pd.read_csv("data/processed/balanced_reviews.csv")
    feature_cols = ["flesch_score", "sentiment_polarity", "nb_exclamations"]
    
    # Entraîner un modèle pour la démo
    pipeline, bundle, metrics = train_model_pipeline(
        df=df.sample(1000),  # Échantillon pour rapidité
        text_col="text_clean",
        label_col="label_toxic",
        feature_cols=feature_cols,
        model_name="logreg"
    )
    
    # Créer les métadonnées
    metadata = create_model_metadata(
        name="demo_toxicity_classifier",
        algorithm="LogisticRegression",
        training_data=df,
        feature_columns=feature_cols,
        hyperparameters={"C": 1.0, "max_iter": 500},
        metrics=metrics,
        training_time=10.5,
        version="1.0.0",
        tags=["demo", "toxicity", "reviews"]
    )
    
    # Enregistrer le modèle
    model_id = registry.register_model(pipeline, metadata)
    print(f"Modèle enregistré: {model_id}")
    
    # Lister les modèles
    models = registry.list_models()
    print(f"Modèles disponibles: {len(models)}")
    for model in models:
        print(f"  - {model['name']} v{model['version']} ({model['algorithm']})")
    
    # Promouvoir en production
    registry.promote_model(model_id, "production")
    print(f"Modèle promu en production")
    
    # Récupérer le modèle en production
    prod_model, prod_metadata = registry.get_production_model("demo_toxicity_classifier")
    print(f"Modèle en production récupéré: {prod_metadata.model_id}")
    
    return registry, model_id

def demo_ab_testing(registry, model_id):
    """Démonstration du système A/B Testing"""
    print("\n=== DEMO A/B TESTING ===")
    
    # Initialiser l'engine A/B
    ab_engine = ABTestingEngine(registry)
    
    # Créer un deuxième modèle pour le test (copie pour la démo)
    model_b_id = model_id + "_v2"
    
    # Simuler un test A/B
    test_id = ab_engine.create_test(
        name="Toxicity Model Comparison",
        description="Test between v1 and v2 of toxicity classifier",
        model_a_id=model_id,
        model_b_id=model_id,  # Même modèle pour la démo
        traffic_split=0.3,  # 30% vers B
        duration_days=7,
        success_metrics=["accuracy", "precision", "recall"]
    )
    
    print(f"Test A/B créé: {test_id}")
    
    # Simuler des prédictions avec différents utilisateurs
    test_texts = [
        "This product is amazing!",
        "Terrible quality, worst purchase ever",
        "Good value for money",
        "Complete waste of time and money"
    ]
    
    # Simuler l'utilisation
    for i, text in enumerate(test_texts):
        user_id = f"user_{i}"
        
        # Déterminer quel modèle utiliser
        model_id_selected, model, group = ab_engine.get_model_for_prediction(test_id, user_id)
        
        # Simuler une prédiction
        prediction = {
            "is_toxic": bool(np.random.choice([0, 1])),
            "confidence": np.random.uniform(0.6, 0.95),
            "processing_time": np.random.uniform(0.01, 0.05)
        }
        
        # Logger le résultat
        ab_engine.log_prediction(test_id, user_id, model_id_selected, group, prediction, prediction["processing_time"])
        
        print(f"User {user_id}: Groupe {group}, Prédiction: {prediction['is_toxic']}")
    
    # Analyser les résultats
    results = ab_engine.analyze_test_results(test_id)
    print(f"Analyse: {results.get('recommendation', 'Données insuffisantes')}")
    
    return ab_engine, test_id

def demo_explainability(registry=None, model_id=None):
    """Démonstration des fonctionnalités d'explicabilité"""
    print("\n=== DEMO EXPLICABILITÉ ===")
    
    # Utiliser le modèle fraîchement entraîné si disponible
    model = None
    if registry and model_id:
        try:
            model, _ = registry.get_model(model_id)
            print(f"Utilisation du modèle fraîchement entraîné: {model_id}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle registry: {e}")
    
    # Fallback sur un modèle existant
    if model is None:
        try:
            model = load_model("models/logreg_toxic.pkl")
            print("Utilisation d'un modèle existant")
        except:
            print("Aucun modèle disponible, passez d'abord l'entraînement")
            return
    
    # Textes d'exemple
    test_texts = [
        "This product is absolutely terrible and a complete waste of money!",
        "Great quality and fast shipping, highly recommend this seller.",
        "The item broke after one day, terrible quality control."
    ]
    
    # LIME Explanation
    print("\nLIME Explanations:")
    lime_explainer = ReviewGuardianLimeExplainer(model)
    
    for i, text in enumerate(test_texts[:2]):  # Limiter pour la démo
        explanation = lime_explainer.explain_prediction(text, num_features=8)
        
        if 'error' not in explanation:
            pred = explanation['prediction']
            features = explanation['explanation']['features_importance']
            
            print(f"\nTexte {i+1}: {text[:50]}...")
            print(f"Prédiction: {pred['class'].upper()} ({pred['confidence']:.3f})")
            print("Features importantes:")
            for word, importance in features[:5]:
                sign = "[-]" if importance < 0 else "[+]"
                print(f"  {sign} {word}: {importance:.3f}")
    
    # SHAP Explanation
    print("\nSHAP Analysis:")
    try:
        shap_explainer = ReviewGuardianShapExplainer(model)
        
        # Analyse globale (échantillon réduit pour rapidité)
        sample_texts = test_texts * 10  # Répéter pour avoir plus de données
        global_analysis = shap_explainer.global_feature_importance(
            sample_texts, 
            max_display=10,
            save_path="reports/demo_shap_global.png"
        )
        
        print(f"Analyse globale SHAP terminée")
        print(f"Features analysées: {global_analysis.get('num_features', 0)}")
        
        # Analyse individuelle
        individual = shap_explainer.explain_prediction(
            test_texts[0],
            background_texts=test_texts
        )
        
        if 'error' not in individual:
            print(f"Analyse individuelle: {individual['prediction']['class'].upper()}")
    
    except Exception as e:
        print(f"SHAP non disponible: {e}")

def demo_caching_optimization():
    """Démonstration du cache et optimisations"""
    print("\n=== DEMO CACHE & OPTIMISATION ===")
    
    # Initialiser le cache
    cache_manager = CacheManager(
        redis_url="redis://localhost:6379",  # Optionnel
        default_ttl=1800
    )
    
    prediction_cache = PredictionCache(cache_manager)
    
    # Simuler des prédictions avec cache
    test_texts = [
        "This is a test message for caching",
        "Another test message",
        "This is a test message for caching"  # Répétition pour test cache
    ]
    
    model_id = "demo_model"
    
    for i, text in enumerate(test_texts):
        # Vérifier le cache
        cached_pred = prediction_cache.get_prediction(text, model_id)
        
        if cached_pred:
            print(f"Cache HIT pour texte {i+1}")
        else:
            print(f"Cache MISS pour texte {i+1}")
            
            # Simuler une prédiction
            prediction = {
                "is_toxic": bool(np.random.choice([0, 1])),
                "confidence": np.random.uniform(0.6, 0.95),
                "processing_time": np.random.uniform(0.01, 0.05)
            }
            
            # Mettre en cache
            prediction_cache.cache_prediction(text, model_id, prediction)
    
    # Statistiques du cache
    stats = cache_manager.get_stats()
    print(f"Stats cache: {stats['hits']} hits, {stats['misses']} misses")
    print(f"Taux de réussite: {stats['hit_rate']:.2%}")
    
    # Démonstration des optimisations
    print("\nOptimisations de performance:")
    
    # Batch processor
    batch_processor = BatchProcessor(batch_size=10, n_jobs=2)
    
    def dummy_process(text):
        # Simulation d'un traitement
        return len(text.split())
    
    results = batch_processor.process_texts_parallel(test_texts * 20, dummy_process)
    print(f"Traitement parallèle: {len(results)} résultats")
    
    # Optimization matricielle (simulation)
    from scipy import sparse
    
    # Créer une matrice sparse d'exemple
    data = np.random.rand(100, 1000)
    data[data < 0.9] = 0  # Rendre sparse
    sparse_matrix = sparse.csr_matrix(data)
    
    print(f"Matrice originale: {sparse_matrix.shape}, densité: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.3f}")
    
    # Optimiser
    optimized = SparseMatrixOptimizer.optimize_sparse_matrix(sparse_matrix)
    print(f"Matrice optimisée: même forme, format optimisé")

def demo_performance_benchmark(registry=None, model_id=None):
    """Démonstration du benchmark de performance"""
    print("\n=== DEMO BENCHMARK PERFORMANCE ===")
    
    # Utiliser le modèle fraîchement entraîné si disponible
    model = None
    if registry and model_id:
        try:
            model, _ = registry.get_model(model_id)
            print(f"Utilisation du modèle fraîchement entraîné: {model_id}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle registry: {e}")
    
    if model is None:
        try:
            model = load_model("models/logreg_toxic.pkl")
            print("Utilisation d'un modèle existant")
        except Exception as e:
            print(f"Aucun modèle disponible: {e}")
            return
    
    try:
        # Créer des textes de test
        test_texts = [
            f"This is test message number {i} for performance testing."
            for i in range(100)
        ]
        
        # Benchmark différentes tailles de batch
        print("Benchmark des tailles de batch...")
        results = benchmark_prediction_performance(
            model, 
            test_texts,
            batch_sizes=[1, 10, 25, 50]
        )
        
        print("Résultats du benchmark:")
        for batch_size, metrics in results.items():
            print(f"  Batch {batch_size:2d}: {metrics['throughput']:6.1f} textes/sec, "
                  f"{metrics['elapsed_time']:5.3f}s")
        
        # Trouver la taille optimale
        best_batch = max(results.keys(), key=lambda k: results[k]['throughput'])
        print(f"Meilleure performance: batch_size={best_batch}")
        
    except Exception as e:
        print(f"Benchmark non disponible: {e}")

def main():
    """Fonction principale de démonstration"""
    print("=== REVIEWGUARDIAN - PHASE 4 DEMO ===")
    print("Fonctionnalités avancées enterprise\n")
    
    try:
        # 1. Model Registry & MLOps
        registry, model_id = demo_model_registry()
        
        # 2. A/B Testing
        ab_engine, test_id = demo_ab_testing(registry, model_id)
        
        # 3. Explicabilité
        demo_explainability(registry, model_id)
        
        # 4. Cache & Optimisation
        demo_caching_optimization()
        
        # 5. Benchmark Performance
        demo_performance_benchmark(registry, model_id)
        
        print("\n=== DÉMONSTRATION TERMINÉE ===")
        print("Toutes les fonctionnalités Phase 4 ont été testées!")
        
        # Résumé des fichiers générés
        print("\nFichiers générés:")
        print("  - models/registry_demo/: Registry des modèles")
        print("  - models/ab_tests/: Résultats A/B testing")
        print("  - reports/demo_shap_global.png: Analyse SHAP")
        print("  - cache/: Cache local des prédictions")
        
    except Exception as e:
        print(f"Erreur pendant la démonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()