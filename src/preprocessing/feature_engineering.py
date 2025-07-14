# src/preprocessing/feature_engineering.py

import pandas as pd
import re
import string
from textblob import TextBlob
from textstat import flesch_reading_ease, flesch_kincaid_grade
from urllib.parse import urlparse
import spacy

# Chargement du modèle SpaCy 
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def compute_readability_scores(text: str) -> tuple:
    """
    Calcule deux scores de lisibilité :
    - Flesch Reading Ease : plus c’est haut, plus c’est facile.
    - Flesch-Kincaid Grade : niveau scolaire approximatif (US).
    """
    try:
        return flesch_reading_ease(text), flesch_kincaid_grade(text)
    except Exception:
        return 0.0, 0.0


def compute_capital_ratio(text: str) -> float:
    """
    Calcule le ratio de lettres majuscules par rapport au texte total.
    Indice possible d’agressivité ou emphase dans les messages toxiques.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    return sum(1 for c in text if c.isupper()) / len(text)


def count_exclamations(text: str) -> int:
    """Compte le nombre de points d’exclamation."""
    return text.count("!")


def count_questions(text: str) -> int:
    """Compte le nombre de points d’interrogation."""
    return text.count("?")


def has_url(text: str) -> bool:
    """Détecte la présence d’un lien URL dans le texte."""
    return bool(re.search(r"http[s]?://|www\.", text))


def has_email(text: str) -> bool:
    """Détecte la présence d’un email."""
    return bool(re.search(r"\b[\w\.-]+?@\w+?\.\w+?\b", text))


def has_phone_number(text: str) -> bool:
    """Détecte un numéro de téléphone générique."""
    return bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", text))


def compute_sentiment_scores(text: str) -> tuple:
    """
    Calcule deux scores :
    - Polarity : [-1, +1] (positif ou négatif)
    - Subjectivity : [0, 1] (objectif ou subjectif)
    """
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0


def detect_repeated_chars(text: str, threshold: int = 3) -> bool:
    """
    Détecte la répétition abusive de caractères (e.g. "looooser", "noooo").
    """
    return bool(re.search(r'(.)\1{' + str(threshold) + ',}', text))


def detect_emoji(text: str) -> bool:
    """
    Détecte la présence d’emojis via des ranges Unicode.
    """
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"  # emoticônes
                               "\U0001F300-\U0001F5FF"  # symboles & pictos
                               "\U0001F680-\U0001F6FF"  # transport & maps
                               "\U0001F1E0-\U0001F1FF"  # drapeaux
                               "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))

def avg_sentence_length(text: str) -> float:
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    total_words = len(re.findall(r'\w+', text))
    return total_words / len(sentences)

def pos_ratios(text: str, pos_tag: str = "ADJ") -> float:
    doc = nlp(text)
    if len(doc) == 0:
        return 0.0
    return sum(1 for token in doc if token.pos_ == pos_tag) / len(doc)
from langdetect import detect

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def enrich_text_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Applique tous les enrichissements linguistiques et structurels sur une colonne texte.
    Renvoie un nouveau DataFrame enrichi.
    """
    df = df.copy()

    # Lire scores de lisibilité
    df[["flesch_score", "fk_grade"]] = df[text_col].apply(
        lambda x: pd.Series(compute_readability_scores(x))
    )

    # Punctuation et structure
    df["capital_ratio"] = df[text_col].apply(compute_capital_ratio)
    df["nb_exclamations"] = df[text_col].apply(count_exclamations)
    df["nb_questions"] = df[text_col].apply(count_questions)

    # Informations techniques
    df["has_url"] = df[text_col].apply(has_url)
    df["has_email"] = df[text_col].apply(has_email)
    df["has_phone"] = df[text_col].apply(has_phone_number)

    # Scores sentiment
    df[["sentiment_polarity", "sentiment_subjectivity"]] = df[text_col].apply(
        lambda x: pd.Series(compute_sentiment_scores(x))
    )

    # Extras
    df["has_repeated_chars"] = df[text_col].apply(detect_repeated_chars)
    df["has_emoji"] = df[text_col].apply(detect_emoji)

    df["avg_sentence_length"] = df[text_col].apply(avg_sentence_length)
    df["adj_ratio"] = df[text_col].apply(lambda x: pos_ratios(x, "ADJ"))
    df["verb_ratio"] = df[text_col].apply(lambda x: pos_ratios(x, "VERB"))

    return df
