"""
Agent Program - Architecte Documentaire
Gestionnaire de référentiels et garant du volume horaire
"""
from typing import List, Dict
from pathlib import Path
from state import GraphState, ReferentielData
from config import CORPUS_DIR, SUPPORTED_SUBJECTS
from utils.vectorstore import VectorStoreManager


class AgentProgram:
    """Agent d'extraction et structuration du référentiel officiel"""
    
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.gabarits = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Charge les gabarits de fiches selon la durée"""
        return {
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
                "evaluation": "Évaluation complète (QCM + Exercices + Mini-projet)"
            }
        }
    
    def _extraire_objectifs_corpus(
        self, 
        matiere: str, 
        theme: str, 
        niveau: str
    ) -> tuple[List[str], List[str], List[int], str]:
        """
        Extrait les objectifs et compétences du corpus
        Retourne: (objectifs, competences, pages_ref, source_doc)
        """
        if matiere not in SUPPORTED_SUBJECTS:
            return self._objectifs_generiques(theme, niveau)
        
        # Rechercher dans le corpus
        query = f"Objectifs pédagogiques {theme} niveau {niveau}"
        results = self.vector_store.search_similar(
            query=query,
            matiere=matiere,
            niveau=niveau,
            top_k=3
        )
        
        objectifs = []
        competences = []
        pages_ref = []
        source_doc = None
        
        if results:
            # Prioriser les documents officiels
            for content, score, metadata in results:
                if metadata.get('type') == 'officiel':
                    source_doc = metadata.get('source', 'Programme officiel')
                    
                    # Extraire page si disponible
                    if 'page' in metadata:
                        pages_ref.append(metadata['page'])
                    
                    # Parser le contenu pour extraire objectifs
                    objectifs.extend(self._parser_objectifs(content))
                    competences.extend(self._parser_competences(content))
        
        # Si rien trouvé, utiliser des objectifs génériques
        if not objectifs:
            return self._objectifs_generiques(theme, niveau)
        
        # Dédupliquer
        objectifs = list(set(objectifs))[:5]  # Max 5 objectifs
        competences = list(set(competences))[:5]
        
        return objectifs, competences, pages_ref, source_doc
    
    def _parser_objectifs(self, texte: str) -> List[str]:
        """Parse le texte pour extraire les objectifs"""
        objectifs = []
        lignes = texte.split('\n')
        
        for ligne in lignes:
            ligne = ligne.strip()
            # Chercher des patterns d'objectifs
            if any(mot in ligne.lower() for mot in ['objectif', 'apprendre', 'comprendre', 'maîtriser']):
                if len(ligne) > 20 and len(ligne) < 200:
                    objectifs.append(ligne)
        
        return objectifs[:5]
    
    def _parser_competences(self, texte: str) -> List[str]:
        """Parse le texte pour extraire les compétences"""
        competences = []
        lignes = texte.split('\n')
        
        for ligne in lignes:
            ligne = ligne.strip()
            if any(mot in ligne.lower() for mot in ['compétence', 'savoir', 'capacité', 'être capable']):
                if len(ligne) > 20 and len(ligne) < 200:
                    competences.append(ligne)
        
        return competences[:5]
    
    def _objectifs_generiques(self, theme: str, niveau: str) -> tuple[List[str], List[str], List[int], str]:
        """Génère des objectifs génériques si aucun corpus disponible"""
        objectifs = [
            f"Comprendre les concepts fondamentaux de {theme}",
            f"Appliquer les connaissances de {theme} dans des situations concrètes",
            f"Développer un raisonnement logique sur {theme}",
            f"Maîtriser les techniques et méthodes liées à {theme}"
        ]
        
        competences = [
            "Analyser et résoudre des problèmes",
            "Communiquer ses résultats de manière claire",
            "Travailler de manière autonome et en groupe",
            "Utiliser les outils appropriés"
        ]
        
        return objectifs, competences, [], "Référentiel générique"
    
    def _appliquer_proportionnalite(
        self, 
        duree: str, 
        cycle: str
    ) -> str:
        """Détermine le gabarit selon la durée et le cycle"""
        # Le gabarit est déjà déterminé par la durée catégorisée
        return duree
    
    def process(self, state: GraphState) -> GraphState:
        """
        Extrait le référentiel et détermine le gabarit
        """
        # Extraire les informations du contexte
        matiere = state.input_data.matiere
        theme = state.input_data.theme_chapitre
        niveau = state.contexte.niveau_exact
        duree = state.contexte.duree_categorisee
        cycle = state.contexte.cycle
        
        # Charger le corpus si nécessaire
        if matiere in SUPPORTED_SUBJECTS:
            nb_docs = self.vector_store.load_corpus(matiere, cycle)
            print(f"Corpus chargé: {nb_docs} documents pour {matiere} - {cycle}")
        
        # Extraire objectifs et compétences
        objectifs, competences, pages_ref, source_doc = self._extraire_objectifs_corpus(
            matiere, theme, niveau
        )
        
        # Déterminer le gabarit
        gabarit = self._appliquer_proportionnalite(duree, cycle)
        
        # Créer le référentiel
        referentiel = ReferentielData(
            objectifs_officiels=objectifs,
            competences=competences,
            gabarit=gabarit,
            source_document=source_doc,
            pages_references=pages_ref
        )
        
        # Mettre à jour l'état
        state.referentiel = referentiel
        
        return state


def agent_program_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Program"""
    agent = AgentProgram()
    return agent.process(state)
