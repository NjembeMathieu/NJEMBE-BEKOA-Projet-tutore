# 📝 Exemples d'Utilisation

## Exemple 1 : Mathématiques - Collège (avec Corpus)

### Données d'Entrée

```
Établissement: Collège Jean Moulin
Ville: Lyon
Année scolaire: 2024-2025
Classe: 4ème
Professeur: M. Dubois
Matière: Mathématiques
Thème/Chapitre: Les fractions et proportionnalité
Séquence/Date: Séquence 2
Volume horaire: 3h
```

### Résultat Attendu

- ✅ Cycle détecté : **Secondaire**
- ✅ Gabarit : **Moyen** (3h)
- ✅ Situation-problème : **OUI** (ancrée à Lyon)
- ✅ Référentiel : Chargé depuis `Corpus/Mathématiques/`
- ✅ Seuil validation : **85%**
- ✅ Export : 3 fichiers (MD, JSON, HTML)

### Extrait de la Fiche Générée

```markdown
# Les Fractions et la Proportionnalité - Classe de 4ème

## Informations Générales
- Établissement: Collège Jean Moulin
- Ville: Lyon
- Classe: 4ème
- Professeur: M. Dubois

## Situation-Problème
Le marché de la Croix-Rousse à Lyon propose différents fruits...
[Situation ancrée dans le contexte lyonnais]

## Objectifs Pédagogiques
1. Comprendre la notion de fraction
2. Effectuer des opérations sur les fractions
3. Résoudre des problèmes de proportionnalité
...
```

---

## Exemple 2 : Informatique - Lycée (avec Corpus)

### Données d'Entrée

```
Établissement: Lycée Victor Hugo
Ville: Paris
Année scolaire: 2024-2025
Classe: Terminale
Professeur: Mme Leclerc
Matière: Informatique
Thème/Chapitre: Bases de données relationnelles
Séquence/Date: Séquence 4
Volume horaire: 5h
```

### Résultat Attendu

- ✅ Cycle détecté : **Secondaire**
- ✅ Gabarit : **Étendu** (5h+)
- ✅ Situation-problème : **OUI** (Paris)
- ✅ Référentiel : Chargé depuis `Corpus/Informatique/`
- ✅ Seuil validation : **85%**
- ✅ Activités : 4-6 activités pratiques

### Extrait de la Fiche

```markdown
# Bases de Données Relationnelles - Terminale

## Situation-Problème
La bibliothèque François Mitterrand à Paris souhaite moderniser 
son système de gestion des emprunts...

## Activités Pédagogiques

### Activité 1: Modélisation d'une base de données
Durée: 60 min
Description: Analyser et modéliser une base de données pour la 
bibliothèque en utilisant le modèle entité-association...

### Activité 2: Création de tables SQL
Durée: 45 min
...
```

---

## Exemple 3 : Physique - Université (sans Corpus)

### Données d'Entrée

```
Établissement: Université de Bordeaux
Ville: Bordeaux
Année scolaire: 2024-2025
Classe: Licence 2
Professeur: Dr. Martin
Matière: Physique
Thème/Chapitre: Mécanique des fluides
Séquence/Date: 12/02/2025
Volume horaire: 4h
```

### Résultat Attendu

- ✅ Cycle détecté : **Universitaire**
- ✅ Gabarit : **Moyen** (4h)
- ✅ Situation-problème : **NON** (pas au niveau universitaire)
- ✅ Référentiel : **Objectifs génériques** (pas de Corpus Physique)
- ✅ Seuil validation : **80%**
- ✅ Approche plus théorique

### Extrait de la Fiche

```markdown
# Mécanique des Fluides - Licence 2

## Informations Générales
- Établissement: Université de Bordeaux
- Ville: Bordeaux
- Classe: Licence 2
- Professeur: Dr. Martin

## Objectifs Pédagogiques
1. Comprendre les concepts fondamentaux de Mécanique des fluides
2. Appliquer les connaissances dans des situations concrètes
3. Développer un raisonnement logique
4. Maîtriser les techniques et méthodes

## Introduction
Ce cours aborde les principes fondamentaux de la mécanique des fluides,
en mettant l'accent sur les équations de conservation...

[Pas de situation-problème pour l'universitaire]
```

---

## Exemple 4 : Histoire - Primaire (sans Corpus)

### Données d'Entrée

```
Établissement: École Primaire Les Lilas
Ville: Toulouse
Année scolaire: 2024-2025
Classe: CM2
Professeur: Mme Rousseau
Matière: Histoire
Thème/Chapitre: La Révolution Française
Séquence/Date: Séquence 1
Volume horaire: 1.5h
```

### Résultat Attendu

- ✅ Cycle détecté : **Primaire**
- ✅ Gabarit : **Court** (1.5h)
- ✅ Situation-problème : **NON** (réservé au Secondaire)
- ✅ Référentiel : **Objectifs génériques**
- ✅ Seuil validation : **90%** (plus strict pour le Primaire)
- ✅ Langage adapté aux CM2

### Extrait de la Fiche

```markdown
# La Révolution Française - CM2

## Objectifs Pédagogiques
1. Découvrir les événements marquants de la Révolution Française
2. Comprendre les causes et conséquences
3. Situer la Révolution dans le temps
4. Développer l'esprit critique

## Activité Principale

### Activité 1: Frise chronologique de 1789
Durée: 30 min
Description: Les élèves créent une frise chronologique illustrant
les événements clés de l'année 1789...
```

---

## Exemple 5 : Correction Itérative

### Scénario

Génération d'une fiche de Mathématiques Terminale, mais validation échouée à la 1ère tentative.

### Itération 1
```
Score: 78% (< seuil 85%)
Problèmes détectés:
- Objectif #2 "Résoudre des équations du second degré" non traité
- Manque d'exercices d'application
- Évaluation trop courte
```

### Correction Automatique
L'Agent Writer reçoit le feedback et régénère avec focus sur :
- Ajouter section sur équations du second degré
- Créer 3 exercices progressifs
- Étoffer l'évaluation

### Itération 2
```
Score: 87% (> seuil 85%)
✅ Validation réussie !
- Tous les objectifs traités
- Structure conforme
- Évaluation complète
```

---

## Comparaison des Modes de Génération

### Mode "Adaptation" (Similarité > 90%)

```
Fiche existante trouvée: "Fonctions linéaires - 3ème - Marseille"
Similarité: 92%

Action:
- Réutilise la structure
- Adapte les exemples pour Lyon
- Change établissement et professeur
- Modifie situation-problème (contexte local)

Temps: ~20 secondes ⚡
```

### Mode "Création Complète" (Aucune fiche similaire)

```
Aucune fiche similaire

Action:
- Génère 100% du contenu
- Utilise le référentiel du Corpus
- Crée situation-problème de zéro
- Structure complète selon gabarit

Temps: ~45 secondes
```

---

## Formats d'Export Générés

### 1. Markdown (.md)
```markdown
# Titre de la fiche

## Objectifs
- Objectif 1
- Objectif 2

[Contenu formaté en Markdown]
```
**Usage:** Édition facile, version control (Git)

### 2. JSON (.json)
```json
{
  "metadata": {
    "etablissement": "Lycée...",
    "classe": "Terminale",
    ...
  },
  "contenu": {
    "titre": "...",
    "objectifs": [...],
    ...
  },
  "validation": {
    "score_conformite": 87.5,
    ...
  }
}
```
**Usage:** Intégration système, API, base de données

### 3. HTML (.html)
```html
<!DOCTYPE html>
<html>
<head>
    <style>
        /* Design professionnel */
    </style>
</head>
<body>
    <div class="container">
        <h1>Titre</h1>
        [Contenu formaté HTML]
    </div>
</body>
</html>
```
**Usage:** Visualisation, impression, partage web

---

## Conseils pour de Meilleurs Résultats

### ✅ Bonnes Pratiques

1. **Thème/Chapitre précis**
   - ✅ "Les fonctions affines et leur représentation graphique"
   - ❌ "Maths"

2. **Volume horaire réaliste**
   - 1-2h : Cours court, 1-2 activités
   - 3-4h : Cours moyen, 2-4 activités
   - 5h+ : Cours étendu, 4-6 activités

3. **Ville spécifique**
   - Permet un meilleur ancrage local
   - ✅ "Lyon" → "Le marché de la Croix-Rousse"
   - ✅ "Paris" → "La Tour Eiffel"

4. **Corpus bien fourni**
   - Plus de documents → Meilleure qualité
   - Documents officiels prioritaires
   - Nommer clairement : `programme_officiel_*.pdf`

### ⚠️ À Éviter

- Volume horaire > 20h (trop long)
- Thème trop vague
- Champs vides
- Matière absente du Corpus (pour Secondaire Maths/Info)

---

## Temps de Génération Moyens

| Configuration | Temps | Facteurs |
|--------------|-------|----------|
| Primaire, 1-2h, sans Corpus | 25-35s | Génération simple |
| Secondaire, 3-4h, avec Corpus | 40-55s | Recherche + Génération |
| Secondaire, 5h+, avec Situation-Problème | 50-70s | Contenu étendu |
| Universitaire, 4h, sans Corpus | 35-50s | Pas de Situation-Problème |
| Avec correction (2 itérations) | +30s | Régénération partielle |

---

**Tous ces exemples sont testables directement dans l'interface Streamlit !**
