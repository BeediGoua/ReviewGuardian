# src/optimization/performance_optimizer.py

import numpy as np
import pandas as pd
from scipy import sparse
import logging
from typing import List, Tuple, Optional, Union, Dict
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseMatrixOptimizer:
    """
    Optimisations pour matrices sparse et vectorisation
    """
    
    @staticmethod
    def optimize_sparse_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Optimise une matrice sparse pour la mémoire et les performances
        
        Args:
            matrix: Matrice sparse d'entrée
            
        Returns:
            Matrice optimisée
        """
        
        # Éliminer les données nulles explicites
        matrix.eliminate_zeros()
        
        # Compresser la matrice
        matrix.sort_indices()
        
        # Convertir en format le plus efficace selon la densité
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        
        if density < 0.1:  # Très sparse
            if not isinstance(matrix, sparse.csr_matrix):
                matrix = matrix.tocsr()
        elif density < 0.5:  # Moyennement sparse
            if not isinstance(matrix, sparse.csc_matrix):
                matrix = matrix.tocsc()
        else:  # Dense
            logger.warning(f"Matrice dense détectée (densité: {density:.3f}), "
                          "considérer conversion en dense")
        
        return matrix
    
    @staticmethod
    def batch_sparse_operations(
        matrix: sparse.csr_matrix,
        operation: str,
        batch_size: int = 1000
    ) -> sparse.csr_matrix:
        """
        Effectue des opérations sur matrices sparse par batch
        
        Args:
            matrix: Matrice d'entrée
            operation: Type d'opération ('normalize', 'sqrt', etc.)
            batch_size: Taille des batches
            
        Returns:
            Matrice transformée
        """
        
        logger.info(f"Opération sparse par batch: {operation}")
        
        n_batches = (matrix.shape[0] + batch_size - 1) // batch_size
        results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, matrix.shape[0])
            
            batch = matrix[start_idx:end_idx]
            
            if operation == 'normalize':
                # Normalisation L2 par ligne
                norms = sparse.linalg.norm(batch, axis=1)
                norms[norms == 0] = 1  # Éviter division par zéro
                batch = batch.multiply(1 / norms[:, np.newaxis])
            
            elif operation == 'sqrt':
                # Racine carrée des valeurs
                batch.data = np.sqrt(batch.data)
            
            elif operation == 'log1p':
                # Log(1 + x)
                batch.data = np.log1p(batch.data)
            
            results.append(batch)
        
        return sparse.vstack(results)
    
    @staticmethod
    def reduce_feature_dimensions(
        matrix: sparse.csr_matrix,
        feature_names: List[str],
        min_frequency: int = 2,
        max_features: Optional[int] = None
    ) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Réduit les dimensions en filtrant les features peu fréquentes
        
        Args:
            matrix: Matrice de features
            feature_names: Noms des features
            min_frequency: Fréquence minimale
            max_features: Nombre max de features à garder
            
        Returns:
            (matrice_réduite, nouveaux_noms_features)
        """
        
        # Calculer les fréquences par feature
        feature_counts = np.array(matrix.sum(axis=0)).flatten()
        
        # Filtrer par fréquence
        valid_features = feature_counts >= min_frequency
        
        # Limiter le nombre total si spécifié
        if max_features and valid_features.sum() > max_features:
            # Garder les features les plus fréquentes
            top_indices = np.argsort(feature_counts)[::-1][:max_features]
            valid_features = np.zeros_like(valid_features, dtype=bool)
            valid_features[top_indices] = True
        
        # Filtrer la matrice et les noms
        matrix_reduced = matrix[:, valid_features]
        feature_names_reduced = [
            name for i, name in enumerate(feature_names) 
            if valid_features[i]
        ]
        
        logger.info(f"Features réduites: {matrix.shape[1]} -> {matrix_reduced.shape[1]}")
        
        return matrix_reduced, feature_names_reduced


class BatchProcessor:
    """
    Traitement par batch pour optimiser les performances
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        n_jobs: int = -1,
        use_multiprocessing: bool = False
    ):
        """
        Args:
            batch_size: Taille des batches
            n_jobs: Nombre de workers (-1 = tous les CPUs)
            use_multiprocessing: Utiliser multiprocessing vs threading
        """
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.use_multiprocessing = use_multiprocessing
        
        logger.info(f"BatchProcessor: batch_size={batch_size}, n_jobs={self.n_jobs}")
    
    def process_texts_parallel(
        self,
        texts: List[str],
        process_func,
        **kwargs
    ) -> List:
        """
        Traite une liste de textes en parallèle
        
        Args:
            texts: Liste de textes
            process_func: Fonction de traitement (text -> result)
            **kwargs: Arguments additionnels pour process_func
            
        Returns:
            Liste des résultats
        """
        
        logger.info(f"Traitement parallèle de {len(texts)} textes")
        
        # Créer les batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Choisir l'executor
        if self.use_multiprocessing:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        
        with executor_class(max_workers=self.n_jobs) as executor:
            # Soumettre tous les batches
            futures = [
                executor.submit(self._process_batch, batch, process_func, **kwargs)
                for batch in batches
            ]
            
            # Collecter les résultats
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[str], process_func, **kwargs) -> List:
        """Traite un batch de textes"""
        return [process_func(text, **kwargs) for text in batch]
    
    def vectorize_parallel(
        self,
        texts: List[str],
        vectorizer: Union[TfidfVectorizer, CountVectorizer]
    ) -> sparse.csr_matrix:
        """
        Vectorisation parallèle par batch
        
        Args:
            texts: Textes à vectoriser
            vectorizer: Vectorizer pré-entraîné
            
        Returns:
            Matrice sparse vectorisée
        """
        
        logger.info(f"Vectorisation parallèle de {len(texts)} textes")
        
        # Créer les batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        # Vectoriser chaque batch
        batch_matrices = []
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(vectorizer.transform, batch)
                for batch in batches
            ]
            
            for future in futures:
                batch_matrix = future.result()
                batch_matrices.append(batch_matrix)
        
        # Combiner les matrices
        combined_matrix = sparse.vstack(batch_matrices)
        
        return SparseMatrixOptimizer.optimize_sparse_matrix(combined_matrix)


class PredictionOptimizer:
    """
    Optimisations spécifiques aux prédictions
    """
    
    def __init__(self, model: Union[Pipeline, BaseEstimator]):
        self.model = model
        
        # Extraire les composants si pipeline
        if hasattr(model, 'named_steps'):
            self.vectorizer = model.named_steps.get('vectorizer')
            self.classifier = model.named_steps.get('classifier')
        else:
            self.vectorizer = None
            self.classifier = model
    
    def predict_batch_optimized(
        self,
        texts: List[str],
        batch_size: int = 100,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prédiction optimisée par batch
        
        Args:
            texts: Textes à prédire
            batch_size: Taille des batches
            return_probabilities: Retourner aussi les probabilités
            
        Returns:
            Prédictions (et probabilités si demandées)
        """
        
        logger.info(f"Prédiction optimisée pour {len(texts)} textes")
        start_time = time.time()
        
        # Vectorisation en une fois (plus efficace)
        if self.vectorizer:
            X = self.vectorizer.transform(texts)
        else:
            # Modèle déjà vectorisé
            X = texts
        
        # Optimiser la matrice
        if sparse.issparse(X):
            X = SparseMatrixOptimizer.optimize_sparse_matrix(X)
        
        # Prédictions par batch
        n_batches = (len(texts) + batch_size - 1) // batch_size
        predictions = []
        probabilities = [] if return_probabilities else None
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            
            X_batch = X[start_idx:end_idx]
            
            # Prédiction
            if hasattr(self.classifier, 'predict'):
                pred_batch = self.classifier.predict(X_batch)
            else:
                pred_batch = self.model.predict(texts[start_idx:end_idx])
            
            predictions.extend(pred_batch)
            
            # Probabilités si demandées
            if return_probabilities:
                if hasattr(self.classifier, 'predict_proba'):
                    prob_batch = self.classifier.predict_proba(X_batch)
                else:
                    prob_batch = self.model.predict_proba(texts[start_idx:end_idx])
                probabilities.extend(prob_batch)
        
        predictions = np.array(predictions)
        
        # Statistiques
        elapsed = time.time() - start_time
        throughput = len(texts) / elapsed
        
        logger.info(f"Prédiction terminée: {elapsed:.2f}s, "
                   f"{throughput:.1f} textes/sec")
        
        if return_probabilities:
            return predictions, np.array(probabilities)
        return predictions
    
    def warm_up_model(self, sample_texts: List[str] = None):
        """
        Réchauffe le modèle avec des prédictions d'exemple
        
        Args:
            sample_texts: Textes d'exemple (optionnel)
        """
        
        if sample_texts is None:
            sample_texts = [
                "This is a test message",
                "Another sample text for warmup",
                "One more example to initialize the model"
            ]
        
        logger.info("Réchauffage du modèle...")
        start_time = time.time()
        
        # Faire quelques prédictions pour initialiser
        for _ in range(3):
            self.predict_batch_optimized(sample_texts[:2])
        
        elapsed = time.time() - start_time
        logger.info(f"Modèle réchauffé en {elapsed:.2f}s")


class MemoryOptimizer:
    """
    Optimisations mémoire pour gros datasets
    """
    
    @staticmethod
    def chunked_processing(
        data: Union[List, pd.DataFrame],
        process_func,
        chunk_size: int = 1000,
        **kwargs
    ):
        """
        Traite des données par chunks pour économiser la mémoire
        
        Args:
            data: Données à traiter
            process_func: Fonction de traitement
            chunk_size: Taille des chunks
            **kwargs: Arguments pour process_func
            
        Yields:
            Résultats par chunk
        """
        
        logger.info(f"Traitement par chunks de {len(data)} éléments")
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            yield process_func(chunk, **kwargs)
    
    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Réduit l'usage mémoire d'un DataFrame
        
        Args:
            df: DataFrame à optimiser
            
        Returns:
            DataFrame optimisé
        """
        
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            
            else:
                # Optimiser les strings
                df[col] = df[col].astype('category')
        
        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (start_memory - end_memory) / start_memory * 100
        
        logger.info(f"Mémoire réduite de {reduction:.1f}% "
                   f"({start_memory:.2f}MB -> {end_memory:.2f}MB)")
        
        return df


# Fonctions utilitaires
def benchmark_prediction_performance(
    model: Union[Pipeline, BaseEstimator],
    texts: List[str],
    batch_sizes: List[int] = [1, 10, 50, 100, 500]
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark des performances de prédiction selon la taille de batch
    
    Args:
        model: Modèle à tester
        texts: Textes de test
        batch_sizes: Tailles de batch à tester
        
    Returns:
        Dictionnaire des résultats {batch_size: {metrics}}
    """
    
    results = {}
    optimizer = PredictionOptimizer(model)
    
    for batch_size in batch_sizes:
        logger.info(f"Test batch_size={batch_size}")
        
        start_time = time.time()
        predictions = optimizer.predict_batch_optimized(texts, batch_size)
        elapsed = time.time() - start_time
        
        results[batch_size] = {
            'elapsed_time': elapsed,
            'throughput': len(texts) / elapsed,
            'predictions_count': len(predictions)
        }
    
    return results

def optimize_vectorizer_params(
    texts: List[str],
    labels: np.ndarray,
    param_grid: Dict = None
) -> Dict:
    """
    Optimise les paramètres du vectorizer pour performance/mémoire
    
    Args:
        texts: Textes d'entraînement
        labels: Labels correspondants
        param_grid: Grille de paramètres à tester
        
    Returns:
        Meilleurs paramètres trouvés
    """
    
    if param_grid is None:
        param_grid = {
            'max_features': [5000, 10000, 20000],
            'ngram_range': [(1, 1), (1, 2), (1, 3)],
            'min_df': [1, 2, 3],
            'max_df': [0.95, 0.98, 1.0]
        }
    
    best_score = -1
    best_params = {}
    
    # Test simple (sans cross-validation pour la rapidité)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Tester quelques combinaisons importantes
    test_combinations = [
        {'max_features': 10000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.95},
        {'max_features': 5000, 'ngram_range': (1, 1), 'min_df': 3, 'max_df': 0.98},
        {'max_features': 20000, 'ngram_range': (1, 3), 'min_df': 1, 'max_df': 1.0},
    ]
    
    for params in test_combinations:
        logger.info(f"Test params: {params}")
        
        vectorizer = TfidfVectorizer(**params)
        
        start_time = time.time()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        vectorization_time = time.time() - start_time
        
        # Test avec modèle simple
        model = LogisticRegression(max_iter=100)
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, predictions)
        
        # Score composite (accuracy + vitesse)
        score = accuracy - (vectorization_time / 100)  # Pénaliser la lenteur
        
        if score > best_score:
            best_score = score
            best_params = {
                **params,
                'accuracy': accuracy,
                'vectorization_time': vectorization_time,
                'matrix_shape': X_train_vec.shape,
                'memory_usage': X_train_vec.data.nbytes / 1024**2  # MB
            }
    
    logger.info(f"Meilleurs paramètres: {best_params}")
    return best_params