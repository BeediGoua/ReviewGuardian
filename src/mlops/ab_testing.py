# src/mlops/ab_testing.py

import json
import random
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

from .model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ABTestConfig:
    """Configuration d'un test A/B"""
    test_id: str
    name: str
    description: str
    model_a_id: str  # Modèle de contrôle
    model_b_id: str  # Modèle challenger
    traffic_split: float  # % trafic vers B (0.1 = 10%)
    start_date: str
    end_date: str
    success_metrics: List[str]  # ['accuracy', 'precision', 'recall']
    min_sample_size: int = 1000
    significance_level: float = 0.05
    status: str = "active"  # active, paused, completed, stopped

@dataclass
class ABTestResult:
    """Résultat d'une prédiction A/B"""
    test_id: str
    user_id: str
    model_used: str  # 'A' ou 'B'
    prediction: Dict
    timestamp: str
    processing_time: float

class ABTestingEngine:
    """Moteur de test A/B pour modèles ML"""
    
    def __init__(self, registry: ModelRegistry, results_path: str = "models/ab_tests"):
        self.registry = registry
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.config_file = self.results_path / "ab_configs.json"
        self.results_file = self.results_path / "ab_results.jsonl"
        
        self._load_configs()
    
    def _load_configs(self):
        """Charge les configurations des tests A/B"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.configs = {
                    test_id: ABTestConfig(**config) 
                    for test_id, config in data.items()
                }
        else:
            self.configs = {}
            self._save_configs()
    
    def _save_configs(self):
        """Sauvegarde les configurations"""
        data = {test_id: asdict(config) for test_id, config in self.configs.items()}
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _log_result(self, result: ABTestResult):
        """Enregistre un résultat de test"""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')
    
    def _hash_user_id(self, user_id: str, test_id: str) -> float:
        """Hash consistant pour assigner les utilisateurs aux groupes"""
        content = f"{user_id}_{test_id}"
        hash_obj = hashlib.md5(content.encode())
        return int(hash_obj.hexdigest(), 16) / (16**32)
    
    def create_test(
        self,
        name: str,
        description: str,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.1,
        duration_days: int = 14,
        success_metrics: List[str] = None,
        **kwargs
    ) -> str:
        """
        Crée un nouveau test A/B
        
        Args:
            name: Nom du test
            description: Description du test
            model_a_id: ID du modèle de contrôle
            model_b_id: ID du modèle challenger
            traffic_split: % de trafic vers le modèle B
            duration_days: Durée du test en jours
            success_metrics: Métriques à optimiser
            
        Returns:
            test_id: Identifiant du test créé
        """
        
        if success_metrics is None:
            success_metrics = ['accuracy', 'precision', 'recall']
        
        # Vérifier que les modèles existent
        if model_a_id not in self.registry.registry:
            raise ValueError(f"Modèle A {model_a_id} non trouvé")
        if model_b_id not in self.registry.registry:
            raise ValueError(f"Modèle B {model_b_id} non trouvé")
        
        # Générer ID du test
        test_id = f"ab_{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Dates
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        config = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split=traffic_split,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            success_metrics=success_metrics,
            **kwargs
        )
        
        self.configs[test_id] = config
        self._save_configs()
        
        logger.info(f"Test A/B créé: {test_id}")
        logger.info(f"Modèle A (contrôle): {model_a_id}")
        logger.info(f"Modèle B (challenger): {model_b_id}")
        logger.info(f"Split trafic: {(1-traffic_split)*100:.1f}% / {traffic_split*100:.1f}%")
        
        return test_id
    
    def get_model_for_prediction(
        self,
        test_id: str,
        user_id: str
    ) -> Tuple[str, object, str]:
        """
        Détermine quel modèle utiliser pour un utilisateur
        
        Returns:
            (model_id, model_object, group) où group est 'A' ou 'B'
        """
        
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} non trouvé")
        
        config = self.configs[test_id]
        
        # Vérifier si le test est actif
        now = datetime.now()
        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)
        
        if config.status != "active" or now < start_date or now > end_date:
            # Test inactif, utiliser le modèle A par défaut
            model, _ = self.registry.get_model(config.model_a_id)
            return config.model_a_id, model, 'A'
        
        # Assignment basé sur hash consistant
        user_hash = self._hash_user_id(user_id, test_id)
        
        if user_hash < config.traffic_split:
            # Groupe B (challenger)
            model, _ = self.registry.get_model(config.model_b_id)
            return config.model_b_id, model, 'B'
        else:
            # Groupe A (contrôle)
            model, _ = self.registry.get_model(config.model_a_id)
            return config.model_a_id, model, 'A'
    
    def log_prediction(
        self,
        test_id: str,
        user_id: str,
        model_id: str,
        group: str,
        prediction: Dict,
        processing_time: float
    ):
        """Enregistre le résultat d'une prédiction"""
        
        result = ABTestResult(
            test_id=test_id,
            user_id=user_id,
            model_used=group,
            prediction=prediction,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        self._log_result(result)
    
    def get_test_results(self, test_id: str) -> pd.DataFrame:
        """Récupère les résultats d'un test A/B"""
        if not self.results_file.exists():
            return pd.DataFrame()
        
        # Charger tous les résultats
        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                if result['test_id'] == test_id:
                    results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extraire les métriques de prédiction
        prediction_cols = ['is_toxic', 'confidence', 'processing_time']
        for col in prediction_cols:
            if 'prediction' in df.columns:
                df[col] = df['prediction'].apply(lambda x: x.get(col, None))
        
        return df
    
    def analyze_test_results(self, test_id: str) -> Dict:
        """Analyse statistique des résultats d'un test A/B"""
        
        df = self.get_test_results(test_id)
        
        if df.empty:
            return {"error": "Aucune donnée trouvée pour ce test"}
        
        config = self.configs[test_id]
        
        # Grouper par modèle
        group_a = df[df['model_used'] == 'A']
        group_b = df[df['model_used'] == 'B']
        
        if len(group_a) == 0 or len(group_b) == 0:
            return {"error": "Données insuffisantes pour les deux groupes"}
        
        results = {
            "test_id": test_id,
            "test_name": config.name,
            "sample_sizes": {
                "group_a": len(group_a),
                "group_b": len(group_b),
                "total": len(df)
            },
            "metrics_comparison": {},
            "statistical_tests": {},
            "recommendation": ""
        }
        
        # Comparer les métriques
        numeric_cols = ['confidence', 'processing_time']
        
        for col in numeric_cols:
            if col in df.columns:
                a_values = group_a[col].dropna()
                b_values = group_b[col].dropna()
                
                if len(a_values) > 0 and len(b_values) > 0:
                    # Statistiques descriptives
                    results["metrics_comparison"][col] = {
                        "group_a": {
                            "mean": float(a_values.mean()),
                            "std": float(a_values.std()),
                            "count": len(a_values)
                        },
                        "group_b": {
                            "mean": float(b_values.mean()),
                            "std": float(b_values.std()),
                            "count": len(b_values)
                        }
                    }
                    
                    # Test statistique (t-test)
                    t_stat, p_value = stats.ttest_ind(a_values, b_values)
                    
                    results["statistical_tests"][col] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < config.significance_level,
                        "improvement": "B > A" if b_values.mean() > a_values.mean() else "A > B"
                    }
        
        # Taux de toxicité détectée
        if 'is_toxic' in df.columns:
            toxic_rate_a = group_a['is_toxic'].mean()
            toxic_rate_b = group_b['is_toxic'].mean()
            
            results["metrics_comparison"]["toxicity_rate"] = {
                "group_a": {"rate": float(toxic_rate_a)},
                "group_b": {"rate": float(toxic_rate_b)}
            }
        
        # Recommandation
        significant_improvements = [
            metric for metric, test in results["statistical_tests"].items()
            if test["significant"] and test["improvement"] == "B > A"
        ]
        
        if len(significant_improvements) >= len(config.success_metrics) // 2:
            results["recommendation"] = "PROMOTE_B"
        elif any(test["significant"] and test["improvement"] == "A > B" 
                for test in results["statistical_tests"].values()):
            results["recommendation"] = "KEEP_A"
        else:
            results["recommendation"] = "CONTINUE_TEST"
        
        return results
    
    def stop_test(self, test_id: str, reason: str = ""):
        """Arrête un test A/B en cours"""
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} non trouvé")
        
        self.configs[test_id].status = "stopped"
        self._save_configs()
        
        logger.info(f"Test A/B {test_id} arrêté. Raison: {reason}")
    
    def list_tests(self, status: Optional[str] = None) -> List[Dict]:
        """Liste les tests A/B avec filtrage optionnel"""
        tests = []
        
        for test_id, config in self.configs.items():
            if status and config.status != status:
                continue
            
            # Récupérer quelques stats
            df = self.get_test_results(test_id)
            
            test_info = {
                "test_id": test_id,
                "name": config.name,
                "status": config.status,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "traffic_split": config.traffic_split,
                "sample_size": len(df) if not df.empty else 0,
                "model_a": config.model_a_id,
                "model_b": config.model_b_id
            }
            
            tests.append(test_info)
        
        return sorted(tests, key=lambda x: x["start_date"], reverse=True)


# Fonction utilitaire pour intégration FastAPI
def get_model_for_user(
    ab_engine: ABTestingEngine,
    test_id: Optional[str],
    user_id: str,
    fallback_model_id: str
) -> Tuple[str, object, str]:
    """
    Wrapper pour obtenir le bon modèle selon le contexte A/B
    
    Args:
        ab_engine: Engine de test A/B
        test_id: ID du test (None si pas de test)
        user_id: Identifiant utilisateur
        fallback_model_id: Modèle par défaut si pas de test
        
    Returns:
        (model_id, model_object, group_info)
    """
    
    if test_id and test_id in ab_engine.configs:
        # Mode A/B testing
        return ab_engine.get_model_for_prediction(test_id, user_id)
    else:
        # Mode normal
        model, _ = ab_engine.registry.get_model(fallback_model_id)
        return fallback_model_id, model, 'default'