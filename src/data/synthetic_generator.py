# src/data/synthetic_generator.py

import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
import uuid


class SyntheticReviewGenerator:
    """
    Générateur d'avis synthétiques pour NLP local.
    - Crée des avis positifs, négatifs, neutres
    - Génère un ID unique, calcule longueur, shuffle
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        # Patterns diversifiés
        self.positive_patterns = [
            "Le produit est {adj_pos}, je {verb_pos}",
            "Service {adj_pos}, livraison {adj_pos}",
            "{adj_pos} qualité, {adj_pos} prix",
            "Je trouve ce produit vraiment {adj_pos} et je le {verb_pos}",
            "Livraison {adj_pos}, emballage {adj_pos}, tout est parfait",
            "Très {adj_pos}, je {verb_pos} sans hésiter",
            "Produit {adj_pos} : je le {verb_pos} à mes proches",
            "Pourquoi hésiter ? Produit {adj_pos}, je {verb_pos}"
        ]

        self.negative_patterns = [
            "Très {adj_neg}, je {verb_neg}",
            "Produit {adj_neg}, service {adj_neg}",
            "Qualité {adj_neg}, ne {verb_neg} pas",
            "Ce produit est vraiment {adj_neg}, je {verb_neg} fortement",
            "Livraison {adj_neg}, emballage {adj_neg}",
            "Service client {adj_neg}, je {verb_neg} la marque",
            "Expérience {adj_neg}, je {verb_neg} cet achat",
            "Prix {adj_neg}, qualité {adj_neg}, je {verb_neg}"
        ]

        self.neutral_patterns = [
            "Produit {adj_neutral}, je {verb_neutral}",
            "Service {adj_neutral}, livraison {adj_neutral}",
            "Qualité {adj_neutral}, prix {adj_neutral}",
            "Livraison {adj_neutral}, expérience {adj_neutral}",
            "Globalement {adj_neutral}, je {verb_neutral}",
            "Expérience {adj_neutral}, service {adj_neutral}"
        ]

        # Vocabulaire étendu
        self.adj_pos = [
            "excellent", "parfait", "super", "génial", "formidable",
            "remarquable", "incroyable", "impeccable", "satisfaisant", "optimal"
        ]

        self.adj_neg = [
            "décevant", "médiocre", "mauvais", "horrible", "nul",
            "lamentable", "pitoyable", "désastreux", "exécrable", "navrant"
        ]

        self.adj_neutral = [
            "correct", "moyen", "acceptable", "standard", "normal", "basique"
        ]

        self.verb_pos = [
            "recommande", "suis satisfait", "adore", "apprécie", "valide",
            "applaudis", "encourage", "félicite", "salue"
        ]

        self.verb_neg = [
            "déconseille", "regrette", "suis déçu", "évite", "blâme",
            "critique", "rejette", "fuis"
        ]

        self.verb_neutral = [
            "note", "observe", "considère", "remarque", "mentionne",
            "souligne", "indique"
        ]

    def generate_reviews(
        self,
        n_positive: int,
        n_negative: int,
        n_neutral: int = 0,
        save_path: str = None,
        shuffle: bool = True,
        as_dict: bool = False
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Génère des avis synthétiques complets.
        - ID unique, longueur
        - Option : sauvegarde CSV
        """
        reviews = []

        for _ in range(n_positive):
            text = random.choice(self.positive_patterns).format(
                adj_pos=random.choice(self.adj_pos),
                verb_pos=random.choice(self.verb_pos)
            )
            reviews.append({
                'id': f"SYN_{uuid.uuid4().hex[:8]}",
                'text': text,
                'label': 'positive',
                'synthetic': True,
                'length': len(text.split())
            })

        for _ in range(n_negative):
            text = random.choice(self.negative_patterns).format(
                adj_neg=random.choice(self.adj_neg),
                verb_neg=random.choice(self.verb_neg)
            )
            reviews.append({
                'id': f"SYN_{uuid.uuid4().hex[:8]}",
                'text': text,
                'label': 'negative',
                'synthetic': True,
                'length': len(text.split())
            })

        for _ in range(n_neutral):
            text = random.choice(self.neutral_patterns).format(
                adj_neutral=random.choice(self.adj_neutral),
                verb_neutral=random.choice(self.verb_neutral)
            )
            reviews.append({
                'id': f"SYN_{uuid.uuid4().hex[:8]}",
                'text': text,
                'label': 'neutral',
                'synthetic': True,
                'length': len(text.split())
            })

        if shuffle:
            random.shuffle(reviews)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(reviews).to_csv(save_path, index=False)
            print(f"[INFO] Saved {len(reviews)} synthetic reviews → {save_path}")

        print(f"[INFO] Generated {n_positive} positive | {n_negative} negative | {n_neutral} neutral")

        return reviews if as_dict else pd.DataFrame(reviews)
