# 📚 Générateur de Fiches de Cours Multi-Agents

Système intelligent de génération automatique de fiches de cours basé sur une architecture multi-agents utilisant LangGraph et Ollama.

## 🎯 Fonctionnalités

- ✅ **Génération automatique** de fiches de cours conformes aux programmes officiels
- ✅ **Architecture multi-agents** avec 6 agents spécialisés
- ✅ **Ancrage local** - Les situations-problèmes sont contextualisées selon la ville
- ✅ **Validation automatique** avec seuils adaptatifs par cycle
- ✅ **Corpus de référence** pour Informatique et Mathématiques (Secondaire)
- ✅ **Export multi-formats** (Markdown, JSON, HTML)
- ✅ **Interface Streamlit** intuitive pour les professeurs
- ✅ **Recherche vectorielle** pour réutilisation de fiches existantes
- ✅ **Boucle de correction** intelligente avec maximum 3 itérations

## 🏗️ Architecture Multi-Agents

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATEUR (LangGraph)                 │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐        ┌─────────┐        ┌──────────┐
   │ Context │──────▶│ Program │──────▶│Similarité│
   └─────────┘        └─────────┘        └──────────┘
        │                   │                   │
        │     Validation    │    Extraction     │  Recherche
        │     & Contexte    │    Référentiel    │  Vectorielle
        ▼                   ▼                   ▼
   ┌─────────┐        ┌─────────┐        ┌──────────┐
   │  Writer │◀───────│Validation│────────│  Export  │
   │ (Gemma)│        │         │        │          │
   └─────────┘        └─────────┘        └──────────┘
        │                   │                   │
        │   Génération      │  Contrôle Qualité │  Fichiers
        │   Contenu         │  Seuils adaptatifs│  (MD/JSON/HTML)
        └───────────────────┴───────────────────┘
```

### Les 6 Agents Spécialisés

1. **Agent Context** 🎯
   - Validation des données d'entrée
   - Identification du cycle d'enseignement
   - Enrichissement du contexte local
   - Détection de la nécessité d'une situation-problème

2. **Agent Program** 📖
   - Extraction des référentiels officiels
   - Accès au corpus (Informatique & Mathématiques)
   - Détermination du gabarit (court/moyen/étendu)
   - Gestion des objectifs pédagogiques

3. **Agent Similarité** 🔍
   - Recherche vectorielle de fiches existantes
   - Cache des embeddings pour performance
   - Décision adaptation vs création complète
   - Seuil de similarité : 90%

4. **Agent Writer** ✍️
   - Génération de contenu via Gemini 1.5 Flash
   - Création de situations-problèmes ancrées localement
   - Adaptation de fiches existantes
   - Corrections itératives basées sur la validation

5. **Agent Validation** ✅
   - Contrôle de conformité aux objectifs
   - Vérification de la structure
   - Seuils adaptatifs par cycle :
     - Primaire : 90%
     - Secondaire : 85%
     - Universitaire : 80%
   - Génération de rapports de correction

6. **Agent Export** 📄
   - Export Markdown (édition facile)
   - Export JSON (intégration système)
   - Export HTML (visualisation/impression)
   - Horodatage et métadonnées

## 📋 Prérequis

- Python 3.10+

- Environnement virtuel `.venv` (déjà créé dans votre projet)

## 🚀 Installation

### 1. Installation des dépendances

```bash
# Activer l'environnement virtuel
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration

Ollama doit etre actif (Gemma3:4b)

### 3. Structure du Corpus

Placez vos documents de référence dans le dossier `Corpus` :

```
Corpus/
├── Informatique/
│   ├── programme_officiel.pdf
│   ├── referentiel_competences.pdf
│   └── ...
└── Mathématiques/
    ├── programme_college.pdf
    ├── programme_lycee.pdf
    └── ...
```

**Formats supportés :** PDF, TXT

## 🎮 Utilisation

### Méthode 1 : Script de démarrage (Recommandé)

```bash
chmod +x run.sh
./run.sh
```

### Méthode 2 : Démarrage manuel

```bash
source .venv/bin/activate
streamlit run app.py
```

L'application sera accessible sur : **http://localhost:8501**

## 📝 Workflow de Génération

1. **Saisie des informations**
   - Établissement, ville, année scolaire
   - Classe, matière, professeur
   - Thème/chapitre, volume horaire, séquence

2. **Traitement multi-agents**
   - Validation et enrichissement du contexte
   - Extraction du référentiel officiel
   - Recherche de fiches similaires
   - Génération du contenu avec Gemini
   - Validation automatique
   - Correction si nécessaire (max 3 itérations)

3. **Export des résultats**
   - Téléchargement des fichiers (MD, JSON, HTML)
   - Aperçu de la fiche générée
   - Rapport de validation détaillé

## 🎓 Cas d'Usage

### Exemple 1 : Fiche avec Corpus (Secondaire - Mathématiques)

```
Établissement: Lycée Victor Hugo
Ville: Paris
Classe: 3ème
Matière: Mathématiques
Thème: Les fonctions affines
Volume horaire: 3h
```

→ Le système :
1. Charge le référentiel Mathématiques du Corpus
2. Génère une situation-problème ancrée à Paris
3. Structure le cours sur 3h (gabarit moyen)
4. Valide avec seuil 85% (Secondaire)

### Exemple 2 : Fiche sans Corpus (Universitaire - Physique)

```
Établissement: Université de Lyon
Classe: Licence 2
Matière: Physique
Thème: Thermodynamique
Volume horaire: 5h
```

→ Le système :
1. Utilise des objectifs génériques
2. Pas de situation-problème (Universitaire)
3. Gabarit étendu (5h+)
4. Valide avec seuil 80%

## ⚙️ Configuration Avancée

### Seuils de Validation

Modifiez dans `config.py` :

```python
VALIDATION_THRESHOLDS = {
    "Primaire": 90,      # 90% minimum
    "Secondaire": 85,    # 85% minimum
    "Universitaire": 80  # 80% minimum
}
```

### Nombre de Corrections

```python
MAX_CORRECTION_LOOPS = 3  # Nombre maximum d'itérations
```



## 📊 Optimisations Implémentées

✅ **Cache des embeddings** - Accélère les recherches vectorielles
✅ **Hiérarchie des sources** - Priorise les documents officiels
✅ **Validation croisée** - Vérifie les objectifs pédagogiques
✅ **Templates adaptatifs** - Gabarits selon niveau et matière
✅ **Système de citations** - Référence les pages sources

## 📁 Structure du Projet

```
multi-agents-vickson/
├── .venv/                      # Environnement virtuel (existant)
├── Corpus/                     # Documents de référence (existant)
│   ├── Informatique/
│   └── Mathématiques/
├── agents/                     # Agents IA
│   ├── __init__.py
│   ├── agent_context.py
│   ├── agent_program.py
│   ├── agent_similarite.py
│   ├── agent_writer.py
│   ├── agent_validation.py
│   └── agent_export.py
├── utils/                      # Utilitaires
│   ├── __init__.py
│   └── vectorstore.py
├── vectorstore/                # Base vectorielle (créé auto)
├── output/                     # Fiches générées (créé auto)
├── app.py                      # Application Streamlit
├── orchestrator.py             # Orchestrateur LangGraph
├── state.py                    # Modèles de données
├── config.py                   # Configuration
├── requirements.txt            # Dépendances
├── .env                        # Variables d'environnement
├── run.sh                      # Script de démarrage
└── README.md                   # Ce fichier
```

## 🐛 Dépannage

### Erreur : Module not found

```bash
pip install -r requirements.txt
```

### Corpus vide

Placez vos PDFs dans `Corpus/Informatique/` ou `Corpus/Mathématiques/`

### Port 8501 occupé

```bash
streamlit run app.py --server.port 8502
```

## 📈 Évolutions Futures

- [ ] Support de plus de matières dans le Corpus
- [ ] Génération d'exercices interactifs
- [ ] Export au format SCORM
- [ ] Intégration avec LMS (Moodle, Canvas)
- [ ] API REST pour intégration externe
- [ ] Traduction multi-langues
- [ ] Génération de QCM automatiques
- [ ] Analyse de progression élèves

## 📄 Licence

Projet académique - Master 2024/2025

## 👤 Auteur

**Bekoa et Njembe**

Projet Tutoré - Master 2024/2025

