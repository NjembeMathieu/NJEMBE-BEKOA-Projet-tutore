# 🏗️ Architecture du Système Multi-Agents

## Vue d'Ensemble

```mermaid
graph TB
    subgraph "Interface Utilisateur"
        UI[Streamlit App<br/>app.py]
    end
    
    subgraph "Orchestration"
        ORCH[Orchestrateur LangGraph<br/>orchestrator.py]
        STATE[Graph State<br/>state.py]
    end
    
    subgraph "Agents IA"
        A1[Agent Context<br/>Validation & Contexte]
        A2[Agent Program<br/>Référentiel]
        A3[Agent Similarité<br/>Recherche Vectorielle]
        A4[Agent Writer<br/>Génération Gemini]
        A5[Agent Validation<br/>Contrôle Qualité]
        A6[Agent Export<br/>Multi-formats]
    end
    
    subgraph "Services"
        VS[VectorStore<br/>ChromaDB]
        GEMINI[Gemini 1.5 Flash<br/>Google AI]
        CORPUS[Corpus<br/>Programmes Officiels]
    end
    
    subgraph "Stockage"
        OUT[Output<br/>Fiches générées]
        CACHE[Cache<br/>Embeddings]
    end
    
    UI -->|Input Data| ORCH
    ORCH -->|GraphState| STATE
    
    STATE -->|1| A1
    A1 -->|Contexte enrichi| STATE
    
    STATE -->|2| A2
    A2 -->|Référentiel| STATE
    A2 -.->|Charge| CORPUS
    
    STATE -->|3| A3
    A3 -->|Similarité| STATE
    A3 <-->|Recherche| VS
    VS <-->|Cache| CACHE
    
    STATE -->|4| A4
    A4 -->|Fiche| STATE
    A4 <-->|Génération| GEMINI
    
    STATE -->|5| A5
    A5 -->|Validation| STATE
    A5 -.->|Si échec| A4
    
    STATE -->|6| A6
    A6 -->|Fichiers| OUT
    A6 -.->|Sauvegarde| VS
    
    OUT -->|Téléchargement| UI
    
    style UI fill:#3498db,color:#fff
    style ORCH fill:#2ecc71,color:#fff
    style GEMINI fill:#e74c3c,color:#fff
    style OUT fill:#f39c12,color:#fff
```

## Flux de Données Détaillé

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant App as Streamlit App
    participant Orch as Orchestrateur
    participant Ctx as Agent Context
    participant Prog as Agent Program
    participant Sim as Agent Similarité
    participant Wrt as Agent Writer
    participant Val as Agent Validation
    participant Exp as Agent Export
    participant Gem as Gemini API
    participant VS as VectorStore
    
    U->>App: Saisie formulaire
    App->>Orch: GraphState initial
    
    Note over Orch: Étape 1: Contexte
    Orch->>Ctx: process(state)
    Ctx->>Ctx: Identifier cycle
    Ctx->>Ctx: Enrichir contexte local
    Ctx-->>Orch: State + Contexte
    
    Note over Orch: Étape 2: Programme
    Orch->>Prog: process(state)
    Prog->>VS: load_corpus(matière)
    VS-->>Prog: Documents
    Prog->>VS: search_similar(objectifs)
    VS-->>Prog: Objectifs officiels
    Prog-->>Orch: State + Référentiel
    
    Note over Orch: Étape 3: Similarité
    Orch->>Sim: process(state)
    Sim->>VS: search_similar(query)
    VS-->>Sim: Fiches existantes
    Sim->>Sim: Décider mode génération
    Sim-->>Orch: State + Similarité
    
    Note over Orch: Étape 4: Génération
    Orch->>Wrt: process(state)
    Wrt->>Wrt: Construire prompt
    Wrt->>Gem: generate_content()
    Gem-->>Wrt: Contenu JSON
    Wrt->>Wrt: Parser et valider
    Wrt-->>Orch: State + Fiche
    
    Note over Orch: Étape 5: Validation
    Orch->>Val: process(state)
    Val->>Val: Vérifier conformité
    Val->>Val: Calculer score
    Val-->>Orch: State + Validation
    
    alt Score < Seuil & Iterations < 3
        Note over Orch: Boucle de correction
        Orch->>Wrt: process(state) [correction]
        Wrt->>Gem: generate_content() [corrigé]
        Gem-->>Wrt: Contenu corrigé
        Wrt-->>Orch: State + Fiche v2
        Orch->>Val: process(state)
        Val-->>Orch: State + Validation v2
    end
    
    Note over Orch: Étape 6: Export
    Orch->>Exp: process(state)
    Exp->>Exp: Formater MD/JSON/HTML
    Exp->>VS: add_validated_fiche()
    Exp-->>Orch: State final
    
    Orch-->>App: State final
    App-->>U: Fichiers + Aperçu
```

## Architecture des Agents

### Agent Context
```
┌─────────────────────────────┐
│     AGENT CONTEXT           │
├─────────────────────────────┤
│ Entrées:                    │
│  • InputData                │
├─────────────────────────────┤
│ Traitements:                │
│  • Identifier cycle         │
│  • Catégoriser durée        │
│  • Valider cohérence        │
│  • Enrichir ancrage local   │
├─────────────────────────────┤
│ Sorties:                    │
│  • ContexteEnrichi          │
│  • Flags (situation-pb)     │
└─────────────────────────────┘
```

### Agent Program
```
┌─────────────────────────────┐
│     AGENT PROGRAM           │
├─────────────────────────────┤
│ Entrées:                    │
│  • Matière, Thème, Niveau   │
├─────────────────────────────┤
│ Services:                   │
│  • VectorStore              │
│  • Corpus (PDFs/TXT)        │
├─────────────────────────────┤
│ Traitements:                │
│  • Charger corpus           │
│  • Extraire objectifs       │
│  • Déterminer gabarit       │
├─────────────────────────────┤
│ Sorties:                    │
│  • ReferentielData          │
└─────────────────────────────┘
```

### Agent Similarité
```
┌─────────────────────────────┐
│     AGENT SIMILARITÉ        │
├─────────────────────────────┤
│ Entrées:                    │
│  • Query (thème+objectifs)  │
├─────────────────────────────┤
│ Services:                   │
│  • VectorStore + Cache      │
│  • Embeddings               │
├─────────────────────────────┤
│ Traitements:                │
│  • Recherche vectorielle    │
│  • Calcul similarité        │
│  • Décision adaptation/création │
├─────────────────────────────┤
│ Sorties:                    │
│  • SimilariteResult         │
│  • Mode génération          │
└─────────────────────────────┘
```

### Agent Writer
```
┌─────────────────────────────┐
│     AGENT WRITER            │
├─────────────────────────────┤
│ Entrées:                    │
│  • Contexte + Référentiel   │
│  • Mode (création/adaptation) │
│  • Feedback (si correction) │
├─────────────────────────────┤
│ Services:                   │
│  • Gemini 1.5 Flash API     │
├─────────────────────────────┤
│ Traitements:                │
│  • Construire prompt        │
│  • Générer contenu          │
│  • Parser JSON              │
│  • Ancrer localement        │
├─────────────────────────────┤
│ Sorties:                    │
│  • FicheContent             │
└─────────────────────────────┘
```

### Agent Validation
```
┌─────────────────────────────┐
│     AGENT VALIDATION        │
├─────────────────────────────┤
│ Entrées:                    │
│  • FicheContent             │
│  • Référentiel              │
├─────────────────────────────┤
│ Vérifications:              │
│  • Champs obligatoires      │
│  • Situation-problème       │
│  • Objectifs pédagogiques   │
│  • Structure gabarit        │
├─────────────────────────────┤
│ Calculs:                    │
│  • Score conformité         │
│  • Comparaison seuil        │
├─────────────────────────────┤
│ Sorties:                    │
│  • ValidationResult         │
│  • Corrections requises     │
└─────────────────────────────┘
```

### Agent Export
```
┌─────────────────────────────┐
│     AGENT EXPORT            │
├─────────────────────────────┤
│ Entrées:                    │
│  • FicheContent validée     │
├─────────────────────────────┤
│ Traitements:                │
│  • Formater Markdown        │
│  • Formater JSON            │
│  • Formater HTML            │
│  • Ajouter métadonnées      │
├─────────────────────────────┤
│ Actions:                    │
│  • Sauvegarder fichiers     │
│  • Indexer dans VectorStore │
├─────────────────────────────┤
│ Sorties:                    │
│  • 3 fichiers (MD/JSON/HTML)│
└─────────────────────────────┘
```

## Modèle de Données (State)

```mermaid
classDiagram
    class GraphState {
        +InputData input_data
        +ContexteEnrichi contexte
        +ReferentielData referentiel
        +SimilariteResult similarite
        +FicheContent fiche
        +ValidationResult validation
        +int compteur_boucles
        +datetime timestamp_debut
        +list historique_corrections
        +bool necessite_situation_probleme
        +str mode_generation
    }
    
    class InputData {
        +str etablissement
        +str ville
        +str annee_scolaire
        +str classe
        +float volume_horaire
        +str matiere
        +str nom_professeur
        +str theme_chapitre
        +str sequence_ou_date
    }
    
    class ContexteEnrichi {
        +str cycle
        +str niveau_exact
        +str duree_categorisee
        +bool validation_coherence
        +list erreurs_coherence
        +dict ancrage_local
    }
    
    class ReferentielData {
        +list objectifs_officiels
        +list competences
        +str gabarit
        +str source_document
        +list pages_references
    }
    
    class SimilariteResult {
        +bool fiche_trouvee
        +float score_similarite
        +str contenu_existant
        +str mode_generation
    }
    
    class FicheContent {
        +str titre
        +str etablissement
        +str ville
        +str classe
        +list objectifs
        +str situation_probleme
        +str introduction
        +str developpement
        +list activites
        +str evaluation
        +str conclusion
        +list references
    }
    
    class ValidationResult {
        +bool valide
        +float score_conformite
        +list commentaires
        +list elements_manquants
        +list corrections_requises
    }
    
    GraphState --> InputData
    GraphState --> ContexteEnrichi
    GraphState --> ReferentielData
    GraphState --> SimilariteResult
    GraphState --> FicheContent
    GraphState --> ValidationResult
```

## Technologies Utilisées

```mermaid
graph LR
    subgraph "Frontend"
        ST[Streamlit]
    end
    
    subgraph "Orchestration"
        LG[LangGraph]
        LC[LangChain]
    end
    
    subgraph "IA/ML"
        GEMINI[Gemini 1.5 Flash]
        EMB[Sentence Transformers]
    end
    
    subgraph "Base de Données"
        CHROMA[ChromaDB]
    end
    
    subgraph "Validation"
        PYD[Pydantic]
    end
    
    ST --> LG
    LG --> LC
    LC --> GEMINI
    LC --> EMB
    EMB --> CHROMA
    LG --> PYD
```

## Décisions de Routage

```mermaid
graph TD
    START[Début] --> CTX[Agent Context]
    CTX --> PROG[Agent Program]
    PROG --> SIM[Agent Similarité]
    
    SIM --> DEC1{Similarité<br/>> 90% ?}
    DEC1 -->|Oui| MODE_ADAPT[Mode: Adaptation]
    DEC1 -->|Non| MODE_CREATE[Mode: Création]
    
    MODE_ADAPT --> WRITER[Agent Writer]
    MODE_CREATE --> WRITER
    
    WRITER --> VAL[Agent Validation]
    
    VAL --> DEC2{Score >=<br/>Seuil ?}
    
    DEC2 -->|Oui| EXPORT[Agent Export]
    DEC2 -->|Non| DEC3{Iterations<br/>< 3 ?}
    
    DEC3 -->|Oui| CORRECT[Correction]
    DEC3 -->|Non| EXPORT_ANYWAY[Export avec réserves]
    
    CORRECT --> WRITER
    EXPORT --> END[Fin]
    EXPORT_ANYWAY --> END
    
    style START fill:#2ecc71
    style END fill:#e74c3c
    style EXPORT fill:#3498db
    style WRITER fill:#f39c12
```

## Optimisations Techniques

### 1. Cache des Embeddings
```
Embedding(texte) 
    ↓
Hash(texte) → Cache JSON
    ↓
Si présent: return cache[hash]
Sinon: calcule + sauvegarde
```

### 2. Hiérarchie des Sources
```
Documents Corpus:
  Priority 1: type="officiel"
  Priority 2: type="complement"
  
Tri: (priority, -score_similarité)
```

### 3. Boucle de Correction
```
MAX_LOOPS = 3

for i in range(MAX_LOOPS):
    fiche = generate()
    score = validate(fiche)
    
    if score >= threshold:
        break
    
    feedback = create_feedback(score)
    
return fiche  # Meilleure version
```

---

**Cette architecture garantit:**
- ✅ Séparation des responsabilités
- ✅ Modularité et extensibilité
- ✅ Performance optimisée
- ✅ Qualité pédagogique
- ✅ Traçabilité complète
