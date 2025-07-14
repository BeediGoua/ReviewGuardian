import re
import string
import pandas as pd
import spacy
from nltk.corpus import stopwords

# Chargement des stopwords anglais
EN_STOPWORDS = set(stopwords.words('english'))

# Chargement du modèle SpaCy (désactive parser + ner pour la vitesse)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Expressions régulières utiles
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_PATTERN = re.compile(r'<[^>]+>')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
NUMERIC_PATTERN = re.compile(r'\b\d+\b')
EMOJI_PATTERN = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def clean_text(text: str,
               lowercase: bool = True,
               remove_punct: bool = True,
               remove_stop: bool = True,
               lemmatize: bool = True,
               remove_urls: bool = True,
               remove_emails: bool = True,
               remove_numbers: bool = True,
               remove_emojis: bool = True) -> str:
    """
    Nettoyage complet d’un texte brut.
    """
    if not isinstance(text, str):
        return ""

    # Minuscule
    if lowercase:
        text = text.lower()

    # Nettoyage HTML
    text = HTML_PATTERN.sub(" ", text)

    # Supprimer les URLs
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)

    # Supprimer les emails
    if remove_emails:
        text = EMAIL_PATTERN.sub(" ", text)

    # Supprimer les emojis
    if remove_emojis:
        text = EMOJI_PATTERN.sub(" ", text)

    # Supprimer les nombres isolés
    if remove_numbers:
        text = NUMERIC_PATTERN.sub(" ", text)

    # Supprimer ponctuation
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenisation rudimentaire
    tokens = re.findall(r'\w+', text)

    # Stopwords
    if remove_stop:
        tokens = [t for t in tokens if t not in EN_STOPWORDS]

    # Lemmatisation
    if lemmatize and tokens:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]

    return " ".join(tokens)

def clean_pipeline(df: pd.DataFrame, text_col: str = "text",
                   min_words: int = 5, max_words: int = 500) -> pd.DataFrame:
    """
    Pipeline : Applique clean_text() à une colonne d’un DataFrame,
    puis filtre les textes trop courts ou trop longs.
    """
    df = df.copy()
    df["text_clean"] = df[text_col].apply(clean_text)
    
    # Nombre de mots dans le texte nettoyé
    df["text_clean_n_words"] = df["text_clean"].str.split().str.len()

    # Filtrage
    df = df[df["text_clean_n_words"].between(min_words, max_words)]

    # Optionnel : on supprime la colonne temporaire
    df.drop(columns=["text_clean_n_words"], inplace=True)

    return df

