"""
Définition de l'état global partagé entre les agents (State)
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class InputData(BaseModel):
    """Données d'entrée de l'utilisateur"""
    etablissement: str
    ville: str
    annee_scolaire: str
    classe: str
    volume_horaire: float
    matiere: str
    nom_professeur: str
    theme_chapitre: str
    sequence_ou_date: str
    

class ContexteEnrichi(BaseModel):
    """Contexte enrichi par l'Agent Context"""
    cycle: str  # Primaire, Secondaire, Universitaire
    niveau_exact: str
    duree_categorisee: str  # court, moyen, etendu
    validation_coherence: bool
    erreurs_coherence: List[str] = Field(default_factory=list)
    ancrage_local: Dict[str, str] = Field(default_factory=dict)


class ReferentielData(BaseModel):
    """Données du référentiel extraites par l'Agent Program"""
    objectifs_officiels: List[str] = Field(default_factory=list)
    competences: List[str] = Field(default_factory=list)
    gabarit: str  # court, moyen, etendu
    source_document: Optional[str] = None
    pages_references: List[int] = Field(default_factory=list)


class SimilariteResult(BaseModel):
    """Résultat de la recherche de similarité"""
    fiche_trouvee: bool
    score_similarite: float = 0.0
    contenu_existant: Optional[str] = None
    mode_generation: str  # "adaptation" ou "creation_complete"


class FicheContent(BaseModel):
    """Contenu de la fiche générée"""
    titre: str = ""
    etablissement: str = ""
    ville: str = ""
    classe: str = ""
    objectifs: List[str] = Field(default_factory=list)
    situation_probleme: Optional[str] = None
    introduction: str = ""
    developpement: str = ""
    activites: List[Dict[str, str]] = Field(default_factory=list)
    evaluation: str = ""
    conclusion: str = ""
    references: List[str] = Field(default_factory=list)

        # Nouveaux champs pour le format camerounais
    informations_generales: Dict[str, str] = Field(default_factory=dict)
    preambule: Dict[str, Any] = Field(default_factory=dict)
    deroulement: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
    evaluation_cloture: Dict[str, Any] = Field(default_factory=dict)

class ValidationResult(BaseModel):
    """Résultat de la validation"""
    valide: bool
    score_conformite: float
    commentaires: List[str] = Field(default_factory=list)
    elements_manquants: List[str] = Field(default_factory=list)
    corrections_requises: List[str] = Field(default_factory=list)


class GraphState(BaseModel):
    """État global partagé entre tous les agents"""
    # Données d'entrée
    input_data: InputData
    
    # Contexte enrichi
    contexte: Optional[ContexteEnrichi] = None
    
    # Référentiel
    referentiel: Optional[ReferentielData] = None
    
    # Similarité
    similarite: Optional[SimilariteResult] = None
    
    # Fiche générée
    fiche: Optional[FicheContent] = None
    
    # Validation
    validation: Optional[ValidationResult] = None
    
    # Métadonnées de suivi
    compteur_boucles: int = 0
    timestamp_debut: datetime = Field(default_factory=datetime.now)
    historique_corrections: List[str] = Field(default_factory=list)
    
    # Ajouter ce champ pour les métriques RAG
    rag_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Flags de contrôle
    necessite_situation_probleme: bool = False
    mode_generation: str = "creation_complete"  # ou "adaptation"
    
    class Config:
        arbitrary_types_allowed = True
