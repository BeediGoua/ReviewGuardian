# ReviewGuardian - Multi-stage Docker build for production
FROM python:3.11-slim as base

# Métadonnées
LABEL maintainer="ReviewGuardian Team"
LABEL description="Local text moderation pipeline for review toxicity detection"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 1: Dependencies installation
FROM base as deps

WORKDIR /app

# Copier requirements d'abord (cache Docker)
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base as app

WORKDIR /app

# Copier les dépendances depuis le stage précédent
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Créer utilisateur non-root pour sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copier le code source
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY CLAUDE.md README.md ./

# Créer répertoires nécessaires
RUN mkdir -p logs reports && \
    chown -R appuser:appuser /app

# Changer vers utilisateur non-root
USER appuser

# Port d'exposition
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]