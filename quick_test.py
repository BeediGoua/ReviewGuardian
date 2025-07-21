#!/usr/bin/env python3
"""
Test rapide pour vérifier l'état du système ReviewGuardian
"""

import sys
from pathlib import Path
import pandas as pd

def quick_check():
    """Vérification rapide du système"""
    print("=== VÉRIFICATION RAPIDE REVIEWGUARDIAN ===\n")
    
    # 1. Données
    print("Données:")
    data_files = {
        "data/processed/balanced_reviews.csv": "Dataset équilibré",
        "data/processed/enriched_reviews.csv": "Dataset enrichi"
    }
    
    for file_path, description in data_files.items():
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path, nrows=1)
                print(f"  {description}: {file_path}")
            except:
                print(f"  {description}: Erreur lecture")
        else:
            print(f"  {description}: MANQUANT")
    
    # 2. Modèles
    print("\nModèles:")
    model_files = {
        "notebooks/models/logreg_toxic.pkl": "LogisticRegression",
        "notebooks/models/rf_toxic.pkl": "RandomForest", 
        "notebooks/models/nb_toxic.pkl": "MultinomialNB"
    }
    
    working_models = 0
    for file_path, name in model_files.items():
        if Path(file_path).exists():
            print(f"  {name}: {file_path}")
            working_models += 1
        else:
            print(f"  {name}: MANQUANT")
    
    # 3. Configuration
    print("\n🔧 Configuration:")
    config_files = {
        "requirements.txt": "Dépendances",
        "Dockerfile": "Docker", 
        "docker-compose.yml": "Docker Compose",
        "README.md": "Documentation"
    }
    
    for file_path, description in config_files.items():
        if Path(file_path).exists():
            print(f"  {description}: {file_path}")
        else:
            print(f"  {description}: MANQUANT")
    
    # 4. Test import simple
    print("\nTest imports:")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test imports de base
        import pandas as pd
        import numpy as np
        import sklearn
        print(f"  ML stack: pandas {pd.__version__}, sklearn {sklearn.__version__}")
        
        # Test modèles utils
        try:
            from models.utils import load_model
            print("  Models utils: Importable")
        except ImportError as e:
            print(f"  Models utils: {e}")
        
        # Test API
        try:
            import fastapi
            import streamlit
            print(f"  Web stack: FastAPI {fastapi.__version__}, Streamlit {streamlit.__version__}")
        except ImportError:
            print("  Web stack: Partiellement installé")
        
    except Exception as e:
        print(f"   Erreur imports: {e}")
    
    # 5. État général
    print("\nÉTAT GÉNÉRAL:")
    
    if working_models >= 2:
        print("  Modèles: PRÊTS")
    elif working_models >= 1:
        print("  Modèles: PARTIELS")
    else:
        print("  Modèles: MANQUANTS")
    
    # Recommandations
    print("\nPROCHAINES ÉTAPES:")
    
    if working_models == 0:
        print("  1. Entraîner les modèles:")
        print("     jupyter notebook notebooks/03_train_evaluation.ipynb")
    
    print("  2. Test complet:")
    print("     python test_complete_system.py")
    
    print("  3. Lancer l'API:")
    print("     python -m uvicorn src.api.main:app --reload")
    
    print("  4. Lancer l'interface:")
    print("     streamlit run app/streamlit_app.py")
    
    print("  5. Docker (optionnel):")
    print("     docker-compose up -d")

if __name__ == "__main__":
    quick_check()