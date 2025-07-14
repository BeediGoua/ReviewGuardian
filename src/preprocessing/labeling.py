# src/preprocessing/labeling.py

import pandas as pd
import logging
from sklearn.utils import resample
# Configuration logging (affichage console)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_toxic_labels(df: pd.DataFrame,
                          toxic_col: str = "potential_toxic",
                          badword_col: str = "flag_badwords",
                          verbose: bool = True) -> pd.DataFrame:
    """
    Génère une étiquette binaire `label_toxic` à partir des colonnes existantes :
    - 1 si le texte est potentiellement toxique OU contient des injures
    - 0 sinon
    """
    df = df.copy()

    if toxic_col not in df.columns or badword_col not in df.columns:
        raise ValueError(f"Colonnes {toxic_col} ou {badword_col} manquantes dans le DataFrame.")

    df["label_toxic"] = ((df[toxic_col] == True) | (df[badword_col] == True)).astype(int)

    if verbose:
        n_pos = df["label_toxic"].sum()
        logger.info(f"Étiquetage terminé : {n_pos} textes toxiques sur {len(df)}")

    return df


def analyze_label_distribution(df: pd.DataFrame,
                               label_col: str = "label_toxic",
                               return_stats: bool = False) -> dict:
    """
    Affiche la distribution des étiquettes et retourne les statistiques optionnellement.
    """
    if label_col not in df.columns:
        raise ValueError(f"Colonne {label_col} introuvable dans le DataFrame.")

    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    stats = {}

    print("\nDistribution des étiquettes :")
    for label, count in counts.items():
        perc = (count / total) * 100
        stats[label] = {"count": count, "percentage": perc}
        print(f" - Classe {label} : {count} avis ({perc:.2f}%)")

    print("\nVérifie s’il y a un déséquilibre significatif avant la modélisation (phase 2.4).")

    return stats if return_stats else None


def preview_labeled_samples(df: pd.DataFrame,
                            label_col: str = "label_toxic",
                            n: int = 5,
                            random_state: int = 42) -> None:
    """
    Affiche un échantillon de textes pour chaque classe d'étiquette.
    """
    if label_col not in df.columns:
        raise ValueError(f"Colonne {label_col} introuvable.")

    print(f"\nExemples de textes étiquetés ({n} par classe) :")
    for label in sorted(df[label_col].dropna().unique()):
        print(f"\n--- Classe {label} ---")
        samples = df[df[label_col] == label].sample(
            n=min(n, df[label_col].value_counts()[label]),
            random_state=random_state
        )
        for idx, row in samples.iterrows():
            print(f"→ {row['text'][:150]}...")
def add_toxic_source(df: pd.DataFrame,
                     toxic_col: str = "potential_toxic",
                     badword_col: str = "flag_badwords") -> pd.DataFrame:
    df["toxic_source"] = "none"
    df.loc[df[toxic_col], "toxic_source"] = "potential"
    df.loc[df[badword_col], "toxic_source"] = "badword"
    df.loc[df[toxic_col] & df[badword_col], "toxic_source"] = "both"
    return df



def create_balanced_subset(df: pd.DataFrame,
                           label_col: str = "label_toxic",
                           random_state: int = 42,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Rééquilibre le DataFrame en sous-échantillonnant la classe majoritaire.
    Renvoie un DataFrame équilibré (df_balanced).
    """
    df = df.copy()

    # Vérification colonne
    if label_col not in df.columns:
        raise ValueError(f"Colonne {label_col} introuvable.")

    # Séparation
    df_majority = df[df[label_col] == 0]
    df_minority = df[df[label_col] == 1]

    if len(df_minority) == 0 or len(df_majority) == 0:
        raise ValueError("Une des deux classes est vide. Impossible de rééquilibrer.")

    # Downsampling
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=random_state)

    # Fusion & shuffle
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if verbose:
        logger.info(f"Jeu équilibré créé : {len(df_balanced)} lignes "
                    f"(classe 0: {len(df_majority_downsampled)}, classe 1: {len(df_minority)})")

    return df_balanced
