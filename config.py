"""
Configuration centrale du système multi-agents
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement (optionnel maintenant)
load_dotenv()

# Chemins du projet
BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "Corpus"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
OUTPUT_DIR = BASE_DIR / "output"

# Créer les dossiers s'ils n'existent pas
VECTORSTORE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration Ollama (remplace Gemini)
OLLAMA_MODEL = "gemma3:4b"  # modèle initial
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.5
OLLAMA_MAX_TOKENS = 4096 # Plus rapide, fiche plus courte
OLLAMA_CONTEXT_SIZE = 32768  # Contexte de 32K tokens

# Configuration de l'embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Seuils de validation par cycle
VALIDATION_THRESHOLDS = {
    "Primaire": 50,
    "Secondaire": 50,
    "Universitaire": 50
}

# Seuil de similarité pour réutilisation
SIMILARITY_THRESHOLD = 0.60

# Limite de boucles de correction
MAX_CORRECTION_LOOPS = 3

# Configuration des gabarits par durée
DURATION_TEMPLATES = {
    "1-2h": "court",
    "3-4h": "moyen",
    "5h+": "etendu"
}

# Matières supportées dans le Corpus
SUPPORTED_SUBJECTS = ["Informatique", "Mathematiques"]

# Niveaux d'enseignement
EDUCATION_LEVELS = {
    "Primaire": ["CP", "CE1", "CE2", "CM1", "CM2"],
    "Secondaire": ["6ème", "5ème", "4ème", "3ème", "2nde", "1ère", "Terminale"],
    "Universitaire": ["Licence 1", "Licence 2", "Licence 3", "Master 1", "Master 2"]
}
# Gabarits selon durée
DURATION_TEMPLATES = {
    "court": {
        "sections": ["Introduction", "Contenu principal", "Exercice", "Conclusion"],
        "activites_min": 1,
        "activites_max": 2,
        "evaluation": "Exercice d'application"
    },
    "moyen": {
        "sections": ["Introduction", "Cours magistral", "Activités", "Exercices", "Évaluation"],
        "activites_min": 2,
        "activites_max": 4,
        "evaluation": "QCM + Exercice"
    },
    "etendu": {
        "sections": ["Introduction", "Cours détaillé", "Activités pratiques", 
                    "Travaux dirigés", "Évaluation formative", "Conclusion"],
        "activites_min": 4,
        "activites_max": 6,
        "evaluation": "Évaluation complète"
    }
}