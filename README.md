
# ReviewGuardian

Pipeline de modération de texte pour la détection de toxicité dans les avis clients, avec explicabilité locale et respect de la vie privée.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Streamlit_&_FastAPI-green.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ReviewGuardian** est une solution complète et 100% locale pour la modération de contenu. Conçue pour être conforme au RGPD, elle détecte les commentaires toxiques sans dépendre d'API externes, garantissant ainsi que les données de vos utilisateurs ne quittent jamais votre infrastructure.

Le projet intègre des outils d'explicabilité (XAI) comme LIME et SHAP pour rendre chaque prédiction transparente et compréhensible.

---

### Sommaire
- [ReviewGuardian](#reviewguardian)
    - [Sommaire](#sommaire)
  - [Fonctionnalités Principales](#fonctionnalités-principales)
  - [Architecture du Projet](#architecture-du-projet)
  - [Démarrage Rapide](#démarrage-rapide)
    - [Docker (Recommandé)](#docker-recommandé)
    - [Installation Locale](#installation-locale)
  - [Interface de Démonstration](#interface-de-démonstration)
  - [Détails du Modèle et Performances](#détails-du-modèle-et-performances)
    - [Entraînement](#entraînement)
    - [Features d'ingénierie](#features-dingénierie)
    - [Performances des modèles](#performances-des-modèles)
  - [Explicabilité (XAI) : LIME \& SHAP](#explicabilité-xai--lime--shap)
  - [Utilisation de l'API](#utilisation-de-lapi)
    - [Prédire un seul avis](#prédire-un-seul-avis)
    - [Prédire en batch](#prédire-en-batch)
  - [Structure du Projet](#structure-du-projet)
  - [Développement](#développement)
    - [Entraînement de nouveaux modèles](#entraînement-de-nouveaux-modèles)
    - [Tests et Qualité de Code](#tests-et-qualité-de-code)
  - [Déploiement](#déploiement)
    - [Variables d'environnement](#variables-denvironnement)
  - [Monitoring \& Observabilité](#monitoring--observabilité)
  - [Contribuer](#contribuer)
  - [Licence](#licence)
  - [Auteur](#auteur)

---

## Fonctionnalités Principales

| Composant | Description |
| :--- | :--- |
| **Détection de toxicité** | Classifie automatiquement un avis en *toxique* ou *non toxique*. |
| **Explicabilité locale** | Affiche les mots qui influencent la décision du modèle (via LIME & SHAP). |
| **Synthèse de données** | Génère des textes d'entraînement pour augmenter et équilibrer le corpus. |
| **Interface Streamlit** | Application web interactive pour démos, tests en batch et monitoring. |
| **Model Registry** | Gestion des modèles, comparaison et promotion en production. |
| **A/B Testing** | Comparaison automatique de modèles sur des métriques de performance clés. |
| **Monitoring & Cache** | Suivi du système (latence, erreurs, ressources) et gestion du cache pour la performance. |

---

## Architecture du Projet

ReviewGuardian est structuré comme un pipeline MLOps complet, de la donnée brute à l'interface utilisateur.

```mermaid
graph TD
    subgraph "1. Préparation des Données"
        A[Sources: Amazon, IMDb, Yelp] --> B(00_download_public_datasets.ipynb);
        B --> C(01_EDA_reviews.ipynb);
        C --> D{Pipeline de Prétraitement<br/>(02_preprocessing.ipynb)};
        SG[Générateur Synthétique<br/>(synthetic_generator.py)] --> D;
    end

    subgraph "2. Modélisation & 3. Explicabilité"
        D --> E[Entraînement & Évaluation<br/>(03_train_evaluation.ipynb)];
        E --> F{Modèles Sauvegardés<br/>(LogisticRegression, RandomForest, ...)<br/>(.pkl)};
        F --> G[Explainers<br/>(lime_explainer.py, shap_explainer.py)];
    end
    
    subgraph "4. Interface & Déploiement"
        F & G --> H[API Backend<br/>(FastAPI)];
        H --> I[Interface Web<br/>(streamlit_app.py)];
        H --> J[Endpoints REST<br/>(/predict, /batch, ...)];
        K[Docker Compose] --> H;
        K --> I;
    end
````

-----

## Démarrage Rapide

### Docker (Recommandé)

```bash
# 1. Cloner le dépôt
git clone https://github.com/BeediGoua/ReviewGuardian.git
cd ReviewGuardian

# 2. Lancer la stack complète
docker-compose up -d --build

# 3. Accéder aux services
#    - Interface Streamlit : http://localhost:8501
#    - API Backend : http://localhost:8000
#    - Documentation API (Swagger) : http://localhost:8000/docs
```

### Installation Locale

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le serveur API (backend)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Lancer l'interface Streamlit (frontend)
streamlit run streamlit_app.py
```

-----

## Interface de Démonstration

L'application Streamlit fournit une interface complète pour interagir avec le projet :

  * **Modération en temps réel** : Analysez un texte et obtenez un score de toxicité instantané.
  * **Analyse en batch** : Uploadez un fichier CSV pour traiter plusieurs avis en une seule fois.
  * **Visualisation de l'explicabilité** : Explorez les rapports LIME et SHAP pour comprendre *pourquoi* un avis a été classifié d'une certaine manière.
  * **Outils MLOps** : Accédez aux dashboards de monitoring, comparez les modèles (A/B testing) et gérez le cycle de vie des modèles.

* **Demo**: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reviewguardian-2rkgvek43wzaqqaniuzdsq.streamlit.app/)


-----

## Détails du Modèle et Performances

### Entraînement

Les modèles sont entraînés sur un jeu de données équilibré de **7 296 avis** (50% toxiques, 50% non toxiques) provenant de diverses sources (Amazon, IMDb, Yelp, etc.).

### Features d'ingénierie

Le pipeline extrait plus de 30 features pour capturer le ton, la complexité et les signaux non-textuels :

  * **Features Textuelles** : Vecteurs TF-IDF et CountVectorizer (N-grams de 1 et 2 mots).
  * **Features Structurées** :
      * `sentiment_polarity` : Score de sentiment (-1 à 1).
      * `flesch_score` : Score de lisibilité du texte.
      * `capital_ratio` : Proportion de lettres majuscules.
      * `nb_exclamations`, `nb_interrogations` : Utilisation de la ponctuation.
      * `has_url`, `has_email`, `has_emoji` : Détection de contenus spécifiques.



-----

## Explicabilité (XAI) : LIME & SHAP

Comprendre les décisions du modèle est essentiel. ReviewGuardian intègre deux bibliothèques de pointe :

  - **LIME (`lime_explainer.py`)** : Fournit une **explication locale** en surlignant les mots qui ont le plus contribué à la prédiction pour un texte donné. Idéal pour les utilisateurs finaux.
  - **SHAP (`shap_explainer.py`)** : Offre des **explications locales et globales**. Il peut montrer l'impact de chaque feature pour une prédiction (waterfall plot) ou l'importance moyenne des features sur l'ensemble du jeu de données.

-----

## Utilisation de l'API

L'API FastAPI expose plusieurs endpoints pour l'intégration.

### Prédire un seul avis

```bash
curl -X POST "http://localhost:8000/predict" \
 -H "Content-Type: application/json" \
 -d '{
   "text": "This product is absolutely terrible!",
   "model": "rf"
 }'
```

**Réponse :**

```json
{
 "is_toxic": true,
 "confidence": 0.87,
 "model_used": "rf_toxic_v1.2",
 "processing_time_ms": 45
}
```

### Prédire en batch

```bash
curl -X POST "http://localhost:8000/predict/batch" \
 -H "Content-Type: application/json" \
 -d '{
   "texts": [
     "Great product, highly recommend!",
     "Worst purchase ever, total garbage!"
   ],
   "model": "rf"
 }'
```

*(La réponse sera une liste de prédictions)*

-----

## Structure du Projet

```
ReviewGuardian/
├── .github/                # Workflows CI/CD
├── data/
│   ├── 00_raw/             # Données brutes
│   └── 01_processed/       # Données nettoyées et features
├── models/                 # Modèles entraînés (.pkl)
├── notebooks/              # Notebooks d'exploration et d'entraînement (00, 01, 02, 03)
├── src/                    # Code source du projet
│   ├── api/                # Code de l'API FastAPI
│   ├── data/               # Scripts de traitement de données (synthetic_generator.py)
│   ├── explainability/     # Modules LIME et SHAP (lime_explainer.py, shap_explainer.py)
│   └── models/             # Scripts d'entraînement et de prédiction
├── tests/                  # Tests unitaires et d'intégration
├── .dockerignore
├── docker-compose.yml      # Orchestration des conteneurs
├── Dockerfile              # Fichier de configuration Docker
├── requirements.txt        # Dépendances Python
└── streamlit_app.py        # Application web Streamlit
```

-----

## Développement

### Entraînement de nouveaux modèles

Le notebook `03_train_evaluation.ipynb` est le point central pour l'entraînement.

```bash
# Lancer Jupyter pour accéder aux notebooks
jupyter notebook notebooks/
```

### Tests et Qualité de Code

```bash
# Exécuter les tests unitaires
pytest tests/ -v

# Vérifier la couverture de code
pytest --cov=src

# Formater le code avec Black
black .

# Linter le code avec flake8/ruff
ruff check .
```

-----

## Déploiement

Le projet est conçu pour être déployé facilement via Docker. Des instructions pour des plateformes comme Railway, Heroku ou des fournisseurs Cloud (AWS, GCP, Azure) peuvent être développées.

### Variables d'environnement

Créez un fichier `.env` à la racine pour configurer le projet :

```bash
ENVIRONMENT=production
LOG_LEVEL=info
DEFAULT_MODEL_PATH=models/RandomForest_v1.2.pkl
API_KEY=votre-cle-secrete-ici
REDIS_URL=redis://redis:6379 # URL pour le cache en production
```

-----

## Monitoring & Observabilité

La stack Docker Compose inclut Prometheus et Grafana pour un monitoring clé en main.

  * **Prometheus** : `http://localhost:9090`
  * **Grafana** : `http://localhost:3000` (login/pass: admin/admin)

Les dashboards pré-configurés permettent de suivre :

  * Latence des requêtes (P50, P95, P99)
  * Débit (requêtes/seconde)
  * Taux d'erreur (4xx, 5xx)
  * Utilisation des ressources (CPU, mémoire)

-----

## Contribuer

Les contributions sont les bienvenues \! Merci de consulter `CONTRIBUTING.md` pour les bonnes pratiques.

1.  Forker le dépôt.
2.  Créer une branche pour votre fonctionnalité (`git checkout -b feature/ma-super-feature`).
3.  Commit vos changements (`git commit -m 'feat: Ajout de ma super feature'`).
4.  Push vers la branche (`git push origin feature/ma-super-feature`).
5.  Ouvrir une Pull Request.

-----

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](https://www.google.com/search?q=LICENSE) pour plus de détails.

-----

## Auteur

**Beedi Goua** - *ML Engineer & Data Scientist*

  * **GitHub** : [@BeediGoua](https://www.google.com/search?q=https://github.com/BeediGoua)
  * **LinkedIn** : [@BeediGoua](https://www.linkedin.com/in/goua-beedi-henri-a152bb1b2/)
  


