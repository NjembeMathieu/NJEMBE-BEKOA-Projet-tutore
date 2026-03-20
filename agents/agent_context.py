"""
Agent Context - Collecteur & Structureur de Données
Point d'entrée unique et validateur de données front-end
"""
from typing import Dict
from state import GraphState, ContexteEnrichi
from config import EDUCATION_LEVELS, DURATION_TEMPLATES


class AgentContext:
    """Agent de contextualisation et validation des données d'entrée"""
    
    def __init__(self):
        self.education_levels = EDUCATION_LEVELS
        
    def _identifier_cycle(self, classe: str) -> str:
        """Identifie le cycle d'enseignement"""
        for cycle, classes in self.education_levels.items():
            if classe in classes:
                return cycle
        return "Secondaire"  # Par défaut
    
    def _categoriser_duree(self, volume_horaire: float) -> str:
        """Catégorise la durée du cours"""
        if volume_horaire <= 2:
            return "court"
        elif volume_horaire <= 4:
            return "moyen"
        else:
            return "etendu"
    
    def _valider_coherence(self, state: GraphState) -> tuple[bool, list[str]]:
        """Valide la cohérence des données d'entrée"""
        erreurs = []
        
        # Vérifier que la classe existe
        classe = state.input_data.classe
        cycle = self._identifier_cycle(classe)
        
        if cycle not in self.education_levels:
            erreurs.append(f"Cycle non reconnu pour la classe {classe}")
        
        # Vérifier le volume horaire
        if state.input_data.volume_horaire <= 0:
            erreurs.append("Le volume horaire doit être positif")
        
        # Vérifier les champs obligatoires
        champs_obligatoires = [
            "etablissement", "ville", "annee_scolaire", 
            "matiere", "nom_professeur", "theme_chapitre"
        ]
        
        for champ in champs_obligatoires:
            valeur = getattr(state.input_data, champ)
            if not valeur or valeur.strip() == "":
                erreurs.append(f"Le champ '{champ}' est obligatoire")
        
        return len(erreurs) == 0, erreurs
    
    def _enrichir_ancrage_local(self, ville: str, etablissement: str) -> Dict[str, str]:
        """
        Crée un dictionnaire d'ancrage local pour contextualiser la fiche
        """
        ancrage = {
            "ville": ville,
            "etablissement": etablissement,
            "description": f"Fiche de cours pour l'établissement {etablissement} situé à {ville}"
        }
        
        # Suggestions d'éléments locaux (à enrichir selon les besoins)
        suggestions = []
        
        # Exemples génériques d'ancrage
        if any(mot in ville.lower() for mot in ["paris", "lyon", "marseille"]):
            suggestions.append("Faire référence à des lieux connus de la ville")
        
        ancrage["suggestions"] = ", ".join(suggestions) if suggestions else "Utiliser le contexte local dans les exemples"
        
        return ancrage
    
    def process(self, state: GraphState) -> GraphState:
        """
        Traite les données d'entrée et enrichit le contexte
        """
        # Identifier le cycle
        cycle = self._identifier_cycle(state.input_data.classe)
        
        # Catégoriser la durée
        duree = self._categoriser_duree(state.input_data.volume_horaire)
        
        # Valider la cohérence
        coherence_ok, erreurs = self._valider_coherence(state)
        
        # Enrichir l'ancrage local
        ancrage = self._enrichir_ancrage_local(
            state.input_data.ville,
            state.input_data.etablissement
        )
        
        # Créer le contexte enrichi
        contexte = ContexteEnrichi(
            cycle=cycle,
            niveau_exact=state.input_data.classe,
            duree_categorisee=duree,
            validation_coherence=coherence_ok,
            erreurs_coherence=erreurs,
            ancrage_local=ancrage
        )
        
        # Mettre à jour l'état
        state.contexte = contexte
        
        # Déterminer si une situation-problème est nécessaire
        state.necessite_situation_probleme = (cycle == "Secondaire")
        
        return state


def agent_context_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Context"""
    agent = AgentContext()
    return agent.process(state)
