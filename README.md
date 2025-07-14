
# SafeText Modération — Pipeline local de détection de toxicité

Projet de modération de texte automatisée (Trustpilot, Amazon, IMDB) avec extraction, enrichissement, visualisation et export local. Tout est conçu pour rester **RGPD-friendly**, **reproductible**, et **sans API externe**.

---

## Objectifs du projet

- Collecter des avis (réels + synthétiques)
- Analyser et enrichir automatiquement les textes
- Détecter les contenus **toxiques**, **injurieux** ou **suspects**
- Explorer visuellement les corrélations et comportements anormaux
- Produire un **CSV enrichi** exploitable pour entraînement ou monitoring

---

## Structure du projet

```bash
project/
│
├── data/
│   ├── raw/               # Données brutes (amazon, imdb, scraping)
│   ├── processed/         # Données nettoyées + enrichies
│
├── notebooks/
│   ├── 01_EDA_reviews.ipynb     # Analyse exploratoire complète
│
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py      # Générateur d'avis synthétiques
│   │   └── scrapers/
│   │       └── review_scraper.py       # Scraper Trustpilot (Scrapy)
│
├── README.md
└── .gitignore



## Installation

```bash
git clone https://github.com/ton_profil/safetext-moderation.git
cd safetext-moderation
pip install -r requirements.txt
```

---

## Démarrage rapide

### 1. Générer ou scraper des données

* **Scraper Trustpilot** :

```python
from src.data.scrapers.review_scraper import scrape_reviews
scrape_reviews(["https://fr.trustpilot.com/review/example.com"], "data/raw/trustpilot_reviews.csv")
```

* **Générer des avis synthétiques** :

```python
from src.data.synthetic_generator import SyntheticReviewGenerator

gen = SyntheticReviewGenerator(seed=42)
df = gen.generate_reviews(n_positive=200, n_negative=200, n_neutral=100, save_path="data/raw/synthetic.csv")
```

### 2. Lancer l’analyse exploratoire (`notebooks/01_EDA_reviews.ipynb`)

Contenu du notebook :

* Fusion des jeux de données
* Extraction de features (n\_words, n\_sentences, lexical\_density…)
* WordCloud, top mots, bigrams, TF-IDF
* Analyse de sentiment (IMDB)
* Détection de toxicité (`potential_toxic`) et injures (`flag_badwords`)
* Corrélations, heatmaps, outliers, scatterplots
* Enregistrement du CSV enrichi

---

## 🔎 Exemple de détection (regex + score TF-IDF)

| Champ             | Exemple                                                     |
| ----------------- | ----------------------------------------------------------- |
| `potential_toxic` | `True` si mots comme `terrible`, `stupid`, `awful`, etc.    |
| `flag_badwords`   | `True` si mots injurieux comme `fuck`, `bitch`, `retard`... |
| `lexical_density` | Ratio mots uniques / total — détecte répétitions ou spam    |

---

## 🧠 Prochaines idées

* Ajouter un **modèle ML local** (ex : LogisticRegression) pour prédire la toxicité
* Convertir en API FastAPI ou interface Streamlit
* Intégrer des embeddings + RAG pour modération contextuelle
* Exploitation avec SHAP/LIME pour interpréter les décisions

---

## 👨‍💻 Auteur

Projet réalisé dans un cadre personnel de montée en compétences en **modération NLP**, **scraping éthique** et **analyse textuelle**.
Par \[Ton Nom / Ton GitHub].

---

## 📄 Licence

Ce projet est libre et open-source sous licence MIT.

```

---

Souhaites-tu aussi que je t’écrive le `requirements.txt` pour ce projet ? Ou une version avec badge GitHub, liens, ou version anglaise ?
```
