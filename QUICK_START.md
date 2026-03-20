# 🚀 Guide de Démarrage Rapide

## Installation en 3 étapes

### 1️⃣ Installer les dépendances

```bash
# Activer l'environnement virtuel existant
source .venv/bin/activate

# Installer les packages
pip install -r requirements.txt
```

### 2️⃣ Vérifier la configuration

```bash
# Votre clé API est déjà configurée dans .env
cat .env
# Devrait afficher: GOOGLE_API_KEY=AIzaSy...
```

### 3️⃣ Lancer l'application

```bash
# Méthode simple
./run.sh

# Ou manuellement
streamlit run app.py
```

🌐 Ouvrez votre navigateur sur : **http://localhost:8501**

## 📚 Premier Usage

1. **Remplissez le formulaire** avec les informations de votre cours :
   - Établissement : "Lycée Victor Hugo"
   - Ville : "Paris"
   - Classe : "3ème"
   - Matière : "Mathématiques"
   - Thème : "Les fonctions affines"
   - Volume horaire : 2h

2. **Cliquez sur "Générer"** 🚀

3. **Attendez** (30-60 secondes) que les 6 agents fassent leur travail

4. **Téléchargez** vos fichiers (MD, JSON, HTML)

## 🎓 Corpus de Référence

Pour utiliser les programmes officiels :

```bash
# Placez vos PDFs dans :
Corpus/
├── Informatique/
│   └── programme_informatique_college.pdf
└── Mathématiques/
    └── programme_maths_lycee.pdf
```

Le système chargera automatiquement ces documents pour enrichir les fiches.

## ✅ Test Rapide

```bash
# Tester le système
python test_system.py
```

## 🆘 Aide Rapide

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Port 8501 occupé | `streamlit run app.py --server.port 8502` |
| Erreur API Key | Vérifier `.env` |
| Corpus vide | Ajouter PDFs dans `Corpus/` |

## 📖 Documentation Complète

Voir [README.md](README.md) pour la documentation détaillée.

---

**Bon cours ! 📚✨**
