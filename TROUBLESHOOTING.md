# 🔧 Guide de Dépannage

## Problèmes Courants et Solutions

### 1. Erreur : Module Not Found

#### Symptôme
```
ModuleNotFoundError: No module named 'streamlit'
ModuleNotFoundError: No module named 'langgraph'
```

#### Solution
```bash
# Activer l'environnement virtuel
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
pip list | grep streamlit
pip list | grep langgraph
```

---

### 2. Erreur : API Key invalide

#### Symptôme
```
google.api_core.exceptions.Unauthenticated: 401 Invalid API key
```

#### Solution
```bash
# Vérifier le fichier .env
cat .env

# Devrait contenir:
GOOGLE_API_KEY=AIzaSyDYOBkjZyt3ehIToK_KyC5i5VC7TpzzBF4

# Si absent ou incorrect, éditer .env
nano .env  # ou vim, code, etc.

# Redémarrer l'application
```

#### Tester la clé API
```python
import google.generativeai as genai
from config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Test")
print(response.text)  # Devrait afficher une réponse
```

---

### 3. Port 8501 déjà utilisé

#### Symptôme
```
OSError: [Errno 98] Address already in use
```

#### Solution

**Option 1 : Changer de port**
```bash
streamlit run app.py --server.port 8502
```

**Option 2 : Tuer le processus existant**
```bash
# Linux/Mac
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

### 4. Corpus vide / Documents non chargés

#### Symptôme
```
Corpus chargé: 0 documents pour Mathématiques - Secondaire
```

#### Diagnostic
```bash
# Vérifier la structure
ls -R Corpus/

# Devrait afficher:
Corpus/
├── Informatique/
│   └── (vos PDFs)
└── Mathématiques/
    └── (vos PDFs)
```

#### Solution
1. **Placer les documents**
```bash
# Copier vos PDFs
cp ~/Documents/programme_maths.pdf Corpus/Mathématiques/
cp ~/Documents/programme_info.pdf Corpus/Informatique/
```

2. **Vérifier les formats supportés**
   - ✅ PDF (.pdf)
   - ✅ Texte (.txt)
   - ❌ Word (.docx) - Non supporté
   - ❌ Images (.jpg, .png) - Non supporté

3. **Vérifier les permissions**
```bash
chmod -R 755 Corpus/
```

---

### 5. Erreur de génération Gemini

#### Symptôme
```
Error lors de la génération: rate limit exceeded
Error lors de la génération: quota exceeded
```

#### Solution

**Rate Limit :**
- Attendre 60 secondes entre les générations
- Utiliser un modèle avec quota plus élevé

**Quota Exceeded :**
```python
# Dans config.py, changer le modèle
GEMINI_MODEL = "gemini-1.5-pro"  # Plus de quota
# ou
GEMINI_MODEL = "gemini-1.5-flash"  # Moins gourmand
```

#### Vérifier les quotas
https://aistudio.google.com/app/apikey

---

### 6. Score de validation toujours < seuil

#### Symptôme
```
⚠️ Limite d'itérations atteinte (3)
Score final: 78%
```

#### Diagnostic
```python
# Afficher les commentaires de validation
python -c "
from test_system import test_generation_complete
test_generation_complete()
"
```

#### Solutions possibles

**1. Baisser les seuils (temporaire)**
```python
# Dans config.py
VALIDATION_THRESHOLDS = {
    "Primaire": 85,      # au lieu de 90
    "Secondaire": 80,    # au lieu de 85
    "Universitaire": 75  # au lieu de 80
}
```

**2. Enrichir le Corpus**
- Ajouter plus de documents officiels
- Inclure des exemples de fiches

**3. Améliorer le prompt**
```python
# Dans agent_writer.py, ajuster le prompt
# Être plus explicite sur les attentes
```

---

### 7. Situation-problème absente (Secondaire)

#### Symptôme
```
CRITIQUE: Situation-problème OBLIGATOIRE pour le Secondaire mais absente
```

#### Diagnostic
```python
# Vérifier le flag
state.necessite_situation_probleme  # Doit être True pour Secondaire
```

#### Solution
Le problème vient généralement de l'Agent Writer. Vérifier :

```python
# Dans agent_writer.py
if state.necessite_situation_probleme:
    prompt += """
    SITUATION-PROBLÈME (OBLIGATOIRE)...
    """
```

---

### 8. Fichiers non générés

#### Symptôme
```
✅ Génération terminée
Mais aucun fichier dans output/
```

#### Diagnostic
```bash
# Vérifier les permissions
ls -la output/

# Vérifier l'espace disque
df -h
```

#### Solution
```bash
# Recréer le dossier
rm -rf output/
mkdir output/
chmod 777 output/

# Relancer la génération
```

---

### 9. Erreur JSON Parser

#### Symptôme
```
Erreur JSON: Expecting property name enclosed in double quotes
```

#### Cause
Gemini a retourné du texte non-JSON ou mal formaté

#### Solution

**Automatique** (déjà implémenté) :
```python
# Dans agent_writer.py
def _parser_json_response(self, response_text):
    # Nettoie les balises markdown
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    # ...
```

**Manuel** :
Augmenter la clarté du prompt :
```python
prompt += """
FORMAT DE SORTIE (JSON STRICT):
Réponds UNIQUEMENT avec un objet JSON.
PAS de texte avant ou après.
PAS de ```json ou ```.
JUSTE le JSON brut.
"""
```

---

### 10. VectorStore corrompue

#### Symptôme
```
Error: Database is locked
Error: Collection not found
```

#### Solution
```bash
# Supprimer et recréer
rm -rf vectorstore/
mkdir vectorstore/

# Ou via Python
python -c "
import shutil
from pathlib import Path
shutil.rmtree('vectorstore', ignore_errors=True)
Path('vectorstore').mkdir(exist_ok=True)
"
```

---

### 11. Lenteur excessive

#### Symptôme
Génération > 2 minutes

#### Diagnostic
```python
import time

# Chronométrer chaque agent
start = time.time()
# ... agent.process()
print(f"Temps: {time.time() - start}s")
```

#### Optimisations

**1. Cache des embeddings**
```python
# Déjà implémenté dans VectorStoreManager
# Vérifier que le cache existe
ls vectorstore/embeddings_cache.json
```

**2. Réduire le corpus**
```bash
# Ne garder que les documents essentiels
# Documents officiels uniquement
```

**3. Limiter les recherches**
```python
# Dans agent_similarite.py
top_k=3  # au lieu de 5 ou 10
```

---

### 12. Import Error avec LangGraph

#### Symptôme
```
ImportError: cannot import name 'StateGraph' from 'langgraph.graph'
```

#### Solution
```bash
# Mettre à jour LangGraph
pip install --upgrade langgraph langchain langchain-core

# Vérifier la version
pip show langgraph
# Version minimum: 0.2.0
```

---

### 13. Streamlit ne démarre pas

#### Symptôme
```
streamlit: command not found
```

#### Solution
```bash
# Vérifier l'environnement virtuel
which python
# Doit pointer vers .venv/bin/python

# Réactiver
source .venv/bin/activate

# Réinstaller Streamlit
pip install --force-reinstall streamlit
```

---

### 14. Erreur de dépendances conflictuelles

#### Symptôme
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

#### Solution
```bash
# Créer un nouvel environnement propre
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Installer étape par étape
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Tests de Diagnostic

### Test Complet du Système
```bash
python test_system.py
```

### Test Individuel des Composants

**1. Test VectorStore**
```python
from utils.vectorstore import VectorStoreManager

vs = VectorStoreManager()
print(f"Collection: {vs.collection.name}")
print(f"Documents: {vs.collection.count()}")
```

**2. Test Gemini**
```python
import google.generativeai as genai
from config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Bonjour")
print(response.text)
```

**3. Test Agent Context**
```python
from agents.agent_context import AgentContext
from state import GraphState, InputData

input_data = InputData(
    etablissement="Test",
    ville="Paris",
    annee_scolaire="2024-2025",
    classe="3ème",
    volume_horaire=2.0,
    matiere="Maths",
    nom_professeur="Test",
    theme_chapitre="Test",
    sequence_ou_date="1"
)

state = GraphState(input_data=input_data)
agent = AgentContext()
result = agent.process(state)
print(f"Cycle: {result.contexte.cycle}")
```

---

## Logs et Debugging

### Activer les logs détaillés

**Dans Streamlit:**
```bash
streamlit run app.py --logger.level debug
```

**Dans Python:**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Consulter les logs

```bash
# Logs Streamlit
~/.streamlit/logs/

# Logs Python
tail -f app.log
```

---

## Commandes Utiles

### Vérification Environnement
```bash
python --version        # Python 3.10+
pip --version           # pip récent
pip list | grep lang    # LangChain/LangGraph
pip list | grep streamlit
```

### Nettoyage
```bash
# Cache Python
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Cache Streamlit
rm -rf ~/.streamlit/cache

# Réinitialisation complète
rm -rf vectorstore/ output/ .streamlit/cache
```

---

## Support et Ressources

### Documentation Officielle
- LangGraph: https://langchain-ai.github.io/langgraph/
- Gemini API: https://ai.google.dev/docs
- Streamlit: https://docs.streamlit.io/

### Logs Utiles à Fournir
```bash
# Versions
python --version
pip list > packages.txt

# Structure
tree -L 2 > structure.txt

# Logs d'erreur
streamlit run app.py 2>&1 | tee error.log
```

---

**Si le problème persiste après ces solutions, vérifiez les logs détaillés et consultez la documentation des packages concernés.**
