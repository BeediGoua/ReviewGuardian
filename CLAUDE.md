# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **ReviewGuardian** (SafeText Modération), a local text moderation pipeline for detecting toxicity in reviews from Trustpilot, Amazon, and IMDB. The project is designed to be GDPR-friendly, reproducible, and works without external APIs.

## Architecture

The project follows a modular ML pipeline structure:

- **Data Pipeline**: `src/data/` - Raw data collection, scraping, and synthetic generation
- **Preprocessing**: `src/preprocessing/` - Text cleaning, feature engineering, and labeling
- **Models**: `src/models/` - Training, evaluation, and model utilities with support for LogReg, RandomForest, and Naive Bayes
- **Pipeline**: `src/pipeline/` - End-to-end review processing pipeline
- **API**: `src/api/` - FastAPI endpoints for model serving
- **App**: `app/` - Streamlit frontend application

### Key Data Flow

1. Raw reviews from `data/raw/` (amazon_reviews.csv, imdb_reviews.csv, toxic_comments_training.csv)
2. Processing creates balanced (`balanced_reviews.csv`) and enriched (`enriched_reviews.csv`) datasets in `data/processed/`
3. Models are trained and saved to `models/` with versioning support
4. Evaluation reports and visualizations are generated in `reports/`

## Development Commands

### Training Models
```bash
# Train models via notebook
jupyter notebook notebooks/03_train_evaluation.ipynb

# Direct training (if script exists)
python src/models/train_model.py
```

### Data Processing
```bash
# Run preprocessing pipeline
python src/preprocessing/text_cleaner.py
python src/preprocessing/feature_engineering.py
```

### Testing
Since no test framework configuration was found, check for:
```bash
# Look for test files
python -m pytest tests/ 
# or
python -m unittest discover tests/
```

### Running Applications
```bash
# Streamlit app
streamlit run app/streamlit_app.py

# API server
python src/api/main.py
```

## Model Training Workflow

The project supports training multiple toxicity classifiers:

1. **LogisticRegression** (`logreg_toxic`) - Primary model with TF-IDF vectorization
2. **RandomForest** (`rf_toxic`) - Ensemble model 
3. **MultinomialNB** (`nb_toxic`) - Naive Bayes with CountVectorizer

### Feature Engineering
Models use both text features (TF-IDF/Count vectors) and structured features:
- `flesch_score` - Text readability
- `sentiment_polarity` - Sentiment analysis
- `nb_exclamations` - Punctuation patterns
- `has_url`, `has_email`, `has_phone` - Content type flags
- `has_emoji` - Emoji presence
- `capital_ratio` - Capitalization ratio

### Model Persistence
- Models saved as `.pkl` files in `models/` directory
- Metrics saved as `*_metrics.json` 
- Evaluation reports generated in `reports/` with visualizations (confusion matrix, ROC curves, PR curves)

## Key Files and Their Purpose

- `src/models/train_model.py` - Main training pipeline with multi-model support
- `src/models/evaluate_model.py` - Comprehensive evaluation with visualization
- `src/models/utils.py` - Model loading, versioning, and utility functions
- `notebooks/03_train_evaluation.ipynb` - Interactive training and comparison workflow
- `data/processed/balanced_reviews.csv` - Balanced training dataset
- `data/processed/enriched_reviews.csv` - Full evaluation dataset with feature engineering

## Dependencies

Key dependencies include:
- scikit-learn (ML models and evaluation)
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- nltk, spacy (NLP preprocessing)
- streamlit (web app)
- fastapi (API)

Install via: `pip install -r requirements.txt`

## Known Issues

- Feature dimension mismatch between training and evaluation pipelines needs fixing
- Missing test framework configuration
- Some scripts in `app/` and `src/pipeline/` appear to be placeholder files