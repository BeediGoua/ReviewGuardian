
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

## Auteur

Projet réalisé dans un cadre personnel de montée en compétences en **modération NLP**, **scraping éthique** et **analyse textuelle**.
Par \[Beedi Goua].

---

## Licence

Ce projet est libre et open-source sous licence MIT.

```


