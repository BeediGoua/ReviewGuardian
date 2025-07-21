# src/mlops/model_registry.py

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Métadonnées complètes d'un modèle"""
    model_id: str
    name: str
    version: str
    algorithm: str
    training_date: str
    training_data_hash: str
    feature_columns: List[str]
    hyperparameters: Dict
    metrics: Dict[str, float]
    data_size: int
    training_time: float
    tags: List[str] = None
    description: str = ""
    author: str = "ReviewGuardian"
    status: str = "trained"  # trained, validated, production, deprecated
    status_updated_at: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ModelRegistry:
    """Registry centralisé pour la gestion des modèles ML"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "model_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Charge le registre des modèles"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
            self._save_registry()
    
    def _save_registry(self):
        """Sauvegarde le registre"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _generate_model_id(self, name: str, algorithm: str, data_hash: str) -> str:
        """Génère un ID unique pour le modèle"""
        content = f"{name}_{algorithm}_{data_hash}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calcule hash des données d'entraînement"""
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()[:16]
    
    def register_model(
        self,
        model: BaseEstimator,
        metadata: ModelMetadata,
        model_path: Optional[str] = None
    ) -> str:
        """
        Enregistre un nouveau modèle dans le registry
        
        Args:
            model: Modèle scikit-learn entraîné
            metadata: Métadonnées du modèle
            model_path: Chemin personnalisé (optionnel)
            
        Returns:
            model_id: Identifiant unique du modèle
        """
        
        # Chemin de sauvegarde
        if model_path is None:
            model_path = self.registry_path / f"{metadata.model_id}.pkl"
        
        # Sauvegarde du modèle
        joblib.dump(model, model_path)
        logger.info(f"Modèle sauvegardé: {model_path}")
        
        # Ajout au registre
        self.registry[metadata.model_id] = {
            **asdict(metadata),
            "model_path": str(model_path),
            "registered_at": datetime.now().isoformat()
        }
        
        self._save_registry()
        logger.info(f"Modèle enregistré avec ID: {metadata.model_id}")
        
        return metadata.model_id
    
    def get_model(self, model_id: str) -> tuple[BaseEstimator, ModelMetadata]:
        """Charge un modèle et ses métadonnées"""
        if model_id not in self.registry:
            raise ValueError(f"Modèle {model_id} non trouvé dans le registre")
        
        metadata_dict = self.registry[model_id]
        model_path = metadata_dict["model_path"]
        
        # Charger le modèle
        model = joblib.load(model_path)
        
        # Reconstruire les métadonnées
        metadata = ModelMetadata(**{k: v for k, v in metadata_dict.items() 
                                  if k not in ["model_path", "registered_at"]})
        
        return model, metadata
    
    def list_models(
        self, 
        status: Optional[str] = None,
        algorithm: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict]:
        """Liste les modèles avec filtrage optionnel"""
        models = []
        
        for model_id, metadata in self.registry.items():
            # Filtres
            if status and metadata.get("status") != status:
                continue
            if algorithm and metadata.get("algorithm") != algorithm:
                continue
            if tag and tag not in metadata.get("tags", []):
                continue
                
            models.append({
                "model_id": model_id,
                "name": metadata["name"],
                "version": metadata["version"],
                "algorithm": metadata["algorithm"],
                "status": metadata["status"],
                "training_date": metadata["training_date"],
                "metrics": metadata["metrics"]
            })
        
        return sorted(models, key=lambda x: x["training_date"], reverse=True)
    
    def promote_model(self, model_id: str, status: str):
        """Change le statut d'un modèle (ex: trained -> production)"""
        if model_id not in self.registry:
            raise ValueError(f"Modèle {model_id} non trouvé")
        
        valid_statuses = ["trained", "validated", "production", "deprecated"]
        if status not in valid_statuses:
            raise ValueError(f"Statut invalide. Utilisez: {valid_statuses}")
        
        old_status = self.registry[model_id]["status"]
        self.registry[model_id]["status"] = status
        self.registry[model_id]["status_updated_at"] = datetime.now().isoformat()
        
        self._save_registry()
        logger.info(f"Modèle {model_id}: {old_status} -> {status}")
    
    def get_production_model(self, name: str) -> Optional[tuple[BaseEstimator, ModelMetadata]]:
        """Récupère le modèle en production pour un nom donné"""
        production_models = [
            (mid, meta) for mid, meta in self.registry.items()
            if meta["name"] == name and meta["status"] == "production"
        ]
        
        if not production_models:
            return None
        
        # Prendre le plus récent
        latest = max(production_models, key=lambda x: x[1]["training_date"])
        return self.get_model(latest[0])
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Compare les métriques de plusieurs modèles"""
        comparison_data = []
        
        for model_id in model_ids:
            if model_id in self.registry:
                meta = self.registry[model_id]
                row = {
                    "model_id": model_id,
                    "name": meta["name"],
                    "algorithm": meta["algorithm"],
                    "version": meta["version"],
                    "status": meta["status"],
                    **meta["metrics"]
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def delete_model(self, model_id: str, force: bool = False):
        """Supprime un modèle du registre"""
        if model_id not in self.registry:
            raise ValueError(f"Modèle {model_id} non trouvé")
        
        metadata = self.registry[model_id]
        
        # Vérification sécurité
        if metadata["status"] == "production" and not force:
            raise ValueError("Impossible de supprimer un modèle en production. Utilisez force=True")
        
        # Supprimer le fichier
        model_path = Path(metadata["model_path"])
        if model_path.exists():
            model_path.unlink()
        
        # Retirer du registre
        del self.registry[model_id]
        self._save_registry()
        
        logger.info(f"Modèle {model_id} supprimé")


# Fonctions utilitaires
def create_model_metadata(
    name: str,
    algorithm: str,
    training_data: pd.DataFrame,
    feature_columns: List[str],
    hyperparameters: Dict,
    metrics: Dict[str, float],
    training_time: float,
    version: str = "1.0.0",
    **kwargs
) -> ModelMetadata:
    """Factory pour créer des métadonnées de modèle"""
    
    data_hash = hashlib.md5(pd.util.hash_pandas_object(training_data).values).hexdigest()[:16]
    model_id = f"{name}_{algorithm}_{data_hash}"[:20]
    
    return ModelMetadata(
        model_id=model_id,
        name=name,
        version=version,
        algorithm=algorithm,
        training_date=datetime.now().isoformat(),
        training_data_hash=data_hash,
        feature_columns=feature_columns,
        hyperparameters=hyperparameters,
        metrics=metrics,
        data_size=len(training_data),
        training_time=training_time,
        **kwargs
    )