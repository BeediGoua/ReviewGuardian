# src/optimization/cache_manager.py

import json
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
import pickle
from pathlib import Path

import redis
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Gestionnaire de cache intelligent pour ReviewGuardian
    Support Redis et cache local avec TTL et invalidation
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        local_cache_dir: str = "cache",
        default_ttl: int = 3600,  # 1 heure
        max_local_size: int = 1000  # Nombre max d'entrées en cache local
    ):
        """
        Args:
            redis_url: URL Redis (optionnel)
            local_cache_dir: Répertoire cache local
            default_ttl: TTL par défaut en secondes
            max_local_size: Taille max du cache local
        """
        
        self.default_ttl = default_ttl
        self.max_local_size = max_local_size
        
        # Cache local (fallback)
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_cache = {}
        
        # Redis (si disponible)
        self.redis_client = None
        self.redis_available = False
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()  # Test de connexion
                self.redis_available = True
                logger.info("Cache Redis connecté")
            except Exception as e:
                logger.warning(f"Redis non disponible, utilisation cache local: {e}")
                self.redis_available = False
        
        # Statistiques
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0
        }
    
    def _generate_key(self, prefix: str, data: Union[str, Dict, List]) -> str:
        """Génère une clé de cache basée sur le contenu"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Sérialise une valeur pour le stockage"""
        if isinstance(value, (dict, list, str, int, float, bool)):
            return json.dumps(value).encode()
        else:
            # Utiliser pickle pour objets complexes
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Désérialise une valeur depuis le stockage"""
        try:
            # Essayer JSON d'abord
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Utiliser pickle en fallback
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur cachée ou None si pas trouvée
        """
        
        try:
            # Essayer Redis d'abord (seulement si disponible)
            if self.redis_available and self.redis_client:
                try:
                    value = self.redis_client.get(key)
                    if value:
                        self.stats['hits'] += 1
                        return self._deserialize_value(value.encode())
                except Exception as e:
                    logger.debug(f"Erreur Redis get: {e}")  # Réduire le niveau de log
                    self.redis_available = False  # Marquer comme indisponible
            
            # Fallback cache local
            if key in self.local_cache:
                entry = self.local_cache[key]
                
                # Vérifier expiration
                if datetime.now() < entry['expires']:
                    self.stats['hits'] += 1
                    return entry['value']
                else:
                    # Expirée, supprimer
                    del self.local_cache[key]
            
            # Essayer cache fichier
            cache_file = self.local_cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if datetime.now() < data['expires']:
                        self.stats['hits'] += 1
                        return data['value']
                    else:
                        cache_file.unlink()  # Supprimer fichier expiré
                
                except Exception as e:
                    logger.warning(f"Erreur lecture cache fichier: {e}")
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Erreur cache get: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        persist_to_file: bool = False
    ) -> bool:
        """
        Stocke une valeur dans le cache
        
        Args:
            key: Clé de cache
            value: Valeur à cacher
            ttl: Time-to-live en secondes
            persist_to_file: Sauvegarder aussi sur disque
            
        Returns:
            True si succès
        """
        
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            serialized = self._serialize_value(value)
            
            # Redis (seulement si disponible)
            if self.redis_available and self.redis_client:
                try:
                    self.redis_client.setex(key, ttl, serialized)
                except Exception as e:
                    logger.debug(f"Erreur Redis set: {e}")  # Réduire le niveau de log
                    self.redis_available = False  # Marquer comme indisponible
            
            # Cache local en mémoire
            expires = datetime.now() + timedelta(seconds=ttl)
            self.local_cache[key] = {
                'value': value,
                'expires': expires
            }
            
            # Nettoyer si trop grand
            if len(self.local_cache) > self.max_local_size:
                # Supprimer les plus anciens
                oldest_keys = sorted(
                    self.local_cache.keys(),
                    key=lambda k: self.local_cache[k]['expires']
                )[:100]
                
                for old_key in oldest_keys:
                    del self.local_cache[old_key]
            
            # Cache fichier (optionnel)
            if persist_to_file:
                cache_file = self.local_cache_dir / f"{key}.cache"
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'expires': expires
                    }, f)
            
            self.stats['writes'] += 1
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Erreur cache set: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        try:
            # Redis
            if self.redis_client:
                self.redis_client.delete(key)
            
            # Cache local
            if key in self.local_cache:
                del self.local_cache[key]
            
            # Cache fichier
            cache_file = self.local_cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur cache delete: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Vide le cache (avec pattern optionnel)
        
        Args:
            pattern: Pattern pour filtrer les clés (ex: "predictions:*")
            
        Returns:
            Nombre de clés supprimées
        """
        
        count = 0
        
        try:
            # Redis
            if self.redis_client:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        count += self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
                    count += len(self.local_cache)
            
            # Cache local
            if pattern:
                import fnmatch
                keys_to_delete = [
                    key for key in self.local_cache.keys()
                    if fnmatch.fnmatch(key, pattern)
                ]
                for key in keys_to_delete:
                    del self.local_cache[key]
                count += len(keys_to_delete)
            else:
                count += len(self.local_cache)
                self.local_cache.clear()
            
            logger.info(f"Cache vidé: {count} entrées supprimées")
            return count
            
        except Exception as e:
            logger.error(f"Erreur cache clear: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'local_cache_size': len(self.local_cache),
            'redis_connected': self.redis_client is not None
        }


class PredictionCache:
    """Cache spécialisé pour les prédictions de modèle"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "prediction"
    
    def get_prediction(self, text: str, model_id: str) -> Optional[Dict]:
        """Récupère une prédiction cachée"""
        key = self.cache._generate_key(
            f"{self.prefix}:{model_id}",
            text.lower().strip()
        )
        return self.cache.get(key)
    
    def cache_prediction(
        self,
        text: str,
        model_id: str,
        prediction: Dict,
        ttl: int = 3600
    ) -> bool:
        """Cache une prédiction"""
        key = self.cache._generate_key(
            f"{self.prefix}:{model_id}",
            text.lower().strip()
        )
        
        # Ajouter métadonnées
        cached_data = {
            **prediction,
            'cached_at': datetime.now().isoformat(),
            'model_id': model_id,
            'text_hash': hashlib.md5(text.encode()).hexdigest()
        }
        
        return self.cache.set(key, cached_data, ttl)
    
    def invalidate_model(self, model_id: str) -> int:
        """Invalide toutes les prédictions d'un modèle"""
        pattern = f"{self.prefix}:{model_id}:*"
        return self.cache.clear(pattern)


class ModelCache:
    """Cache pour les modèles ML"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "model"
    
    def get_model(self, model_id: str) -> Optional[BaseEstimator]:
        """Récupère un modèle caché"""
        key = f"{self.prefix}:{model_id}"
        return self.cache.get(key)
    
    def cache_model(
        self,
        model_id: str,
        model: BaseEstimator,
        ttl: int = 86400  # 24h
    ) -> bool:
        """Cache un modèle"""
        key = f"{self.prefix}:{model_id}"
        return self.cache.set(key, model, ttl, persist_to_file=True)


class VectorizationCache:
    """Cache pour les vectorisations de texte"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.prefix = "vectorization"
    
    def get_vectors(self, texts: List[str], vectorizer_id: str) -> Optional[np.ndarray]:
        """Récupère des vecteurs cachés"""
        # Créer une clé basée sur le hash des textes
        text_hash = hashlib.md5('|'.join(texts).encode()).hexdigest()
        key = f"{self.prefix}:{vectorizer_id}:{text_hash}"
        
        cached = self.cache.get(key)
        if cached and isinstance(cached, dict):
            return np.array(cached['vectors'])
        return None
    
    def cache_vectors(
        self,
        texts: List[str],
        vectorizer_id: str,
        vectors: np.ndarray,
        ttl: int = 7200  # 2h
    ) -> bool:
        """Cache des vecteurs"""
        text_hash = hashlib.md5('|'.join(texts).encode()).hexdigest()
        key = f"{self.prefix}:{vectorizer_id}:{text_hash}"
        
        cached_data = {
            'vectors': vectors.tolist(),
            'shape': vectors.shape,
            'vectorizer_id': vectorizer_id,
            'num_texts': len(texts)
        }
        
        return self.cache.set(key, cached_data, ttl)


# Décorateur pour cache automatique
def cached_prediction(cache_manager: CacheManager, ttl: int = 3600):
    """
    Décorateur pour mettre en cache les prédictions automatiquement
    
    Usage:
        @cached_prediction(cache_manager, ttl=1800)
        def predict_toxicity(text: str, model_id: str) -> Dict:
            # logique de prédiction
            return prediction
    """
    
    def decorator(func):
        prediction_cache = PredictionCache(cache_manager)
        
        def wrapper(text: str, model_id: str, *args, **kwargs):
            # Essayer de récupérer du cache
            cached = prediction_cache.get_prediction(text, model_id)
            if cached:
                logger.debug(f"Cache hit pour prédiction: {text[:30]}...")
                return cached
            
            # Calculer et cacher
            result = func(text, model_id, *args, **kwargs)
            prediction_cache.cache_prediction(text, model_id, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Fonctions utilitaires
def setup_cache(redis_url: Optional[str] = None) -> CacheManager:
    """Configure et retourne un gestionnaire de cache"""
    return CacheManager(redis_url=redis_url)

def get_cache_stats(cache_manager: CacheManager) -> Dict:
    """Retourne les statistiques détaillées du cache"""
    stats = cache_manager.get_stats()
    
    # Ajouter des métriques spécifiques
    if cache_manager.redis_client:
        try:
            info = cache_manager.redis_client.info()
            stats['redis_memory'] = info.get('used_memory_human', 'N/A')
            stats['redis_keys'] = info.get('db0', {}).get('keys', 0)
        except:
            pass
    
    return stats