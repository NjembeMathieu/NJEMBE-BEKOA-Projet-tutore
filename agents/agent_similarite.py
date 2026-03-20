"""
Agent Similarité - Analyste de Précédents & Gestionnaire de Flux
Optimiseur de ressources via recherche vectorielle
"""
from state import GraphState, SimilariteResult
from config import SIMILARITY_THRESHOLD, SUPPORTED_SUBJECTS
from utils.vectorstore import VectorStoreManager


class AgentSimilarite:
    """Agent de recherche de fiches similaires pour optimisation"""
    
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.threshold = SIMILARITY_THRESHOLD
    
    def _construire_query(self, state: GraphState) -> str:
        """Construit la requête de recherche"""
        matiere = state.input_data.matiere
        theme = state.input_data.theme_chapitre
        niveau = state.contexte.niveau_exact
        
        query = f"{matiere} {theme} niveau {niveau}"
        
        # Ajouter les objectifs si disponibles
        if state.referentiel and state.referentiel.objectifs_officiels:
            objectifs_str = " ".join(state.referentiel.objectifs_officiels[:2])
            query += f" {objectifs_str}"
        
        return query
    
    def process(self, state: GraphState) -> GraphState:
        """
        Recherche des fiches similaires validées
        """
        matiere = state.input_data.matiere
        
        # Si la matière n'est pas dans le corpus supporté, passer en création complète
        if matiere not in SUPPORTED_SUBJECTS:
            state.similarite = SimilariteResult(
                fiche_trouvee=False,
                score_similarite=0.0,
                contenu_existant=None,
                mode_generation="creation_complete"
            )
            state.mode_generation = "creation_complete"
            return state
        
        # Construire la requête
        query = self._construire_query(state)
        
        # Rechercher dans le vector store
        results = self.vector_store.search_similar(
            query=query,
            matiere=matiere,
            niveau=state.contexte.cycle,
            top_k=3
        )
        
        # Analyser les résultats
        fiche_trouvee = False
        meilleur_score = 0.0
        meilleur_contenu = None
        
        if results:
            # Prendre le meilleur résultat de type "fiche_validee"
            for content, score, metadata in results:
                if metadata.get('type') == 'fiche_validee' and score > meilleur_score:
                    meilleur_score = score
                    meilleur_contenu = content
            
            # Décider si on utilise l'adaptation ou création complète
            if meilleur_score >= self.threshold:
                fiche_trouvee = True
        
        # Déterminer le mode de génération
        mode = "adaptation" if fiche_trouvee else "creation_complete"
        
        # Créer le résultat de similarité
        similarite_result = SimilariteResult(
            fiche_trouvee=fiche_trouvee,
            score_similarite=meilleur_score,
            contenu_existant=meilleur_contenu,
            mode_generation=mode
        )
        
        # Mettre à jour l'état
        state.similarite = similarite_result
        state.mode_generation = mode

        # Tracking des métriques RAG
        state.rag_metrics = {
            'documents_retrieved': len(results),
            'avg_similarity': sum(score for _, score, _ in results) / len(results) if results else 0,
            'sources': list(set(meta.get('source') for _, _, meta in results))
        }
        
        return state


def agent_similarite_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Similarité"""
    agent = AgentSimilarite()
    return agent.process(state)