"""
Orchestrateur - Gestion du flux de travail multi-agents avec LangGraph
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from state import GraphState
from config import MAX_CORRECTION_LOOPS 
import time
from datetime import datetime

# Import des agents
from agents.agent_context import agent_context_node
from agents.agent_program import agent_program_node
from agents.agent_similarite import agent_similarite_node
from agents.agent_writer import agent_writer_node
from agents.agent_validation import agent_validation_node
from agents.agent_export import agent_export_node


class PerformanceTracker:
    """Tracker minimal des performances"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.agent_times = {}
        self.agent_calls = {}
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def track_agent(self, name, duration):
        if name not in self.agent_times:
            self.agent_times[name] = []
            self.agent_calls[name] = 0
        self.agent_times[name].append(duration)
        self.agent_calls[name] += 1
    
    def get_summary(self):
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        summary = {
            'total_time': round(total_time, 2),
            'agents': {}
        }
        for name in self.agent_times:
            avg_time = sum(self.agent_times[name]) / len(self.agent_times[name])
            summary['agents'][name] = {
                'avg_time': round(avg_time, 2),
                'calls': self.agent_calls[name]
            }
        return summary


class Orchestrateur:
    """
    Orchestrateur du système multi-agents
    Gère le flux de travail et les décisions de routage
    """
    
    def __init__(self):
        self.performance = PerformanceTracker()
        self.graph = self._build_graph()
    
    def _get_validation_info(self, final_state):
        """
        Extrait les informations de validation quel que soit le type de retour
        (objet GraphState ou dictionnaire)
        """
        try:
            # Cas 1: final_state est un objet GraphState (ou similaire)
            if hasattr(final_state, 'validation') and hasattr(final_state, 'compteur_boucles'):
                if final_state.validation:
                    return {
                        'score': final_state.validation.score_conformite,
                        'valide': final_state.validation.valide,
                        'compteur': final_state.compteur_boucles
                    }
                else:
                    return {'score': 'N/A', 'valide': False, 'compteur': 0}
            
            # Cas 2: final_state est un dictionnaire
            elif isinstance(final_state, dict):
                validation = final_state.get('validation')
                if validation:
                    # validation peut être un objet ou un dict
                    if hasattr(validation, 'score_conformite'):
                        score = validation.score_conformite
                        valide = validation.valide
                    else:
                        score = validation.get('score_conformite', 'N/A')
                        valide = validation.get('valide', False)
                else:
                    score, valide = 'N/A', False
                compteur = final_state.get('compteur_boucles', 0)
                return {'score': score, 'valide': valide, 'compteur': compteur}
            
            # Cas 3: autre structure
            else:
                return {'score': 'N/A', 'valide': False, 'compteur': 0}
                
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération des métriques: {e}")
            return {'score': 'N/A', 'valide': False, 'compteur': 0}
    
    def _should_continue_correction(self, state: GraphState) -> Literal["writer", "export"]:
        """
        Décide si on continue les corrections ou si on exporte
        """
        validation = state.validation
        
        # Si la fiche est validée, on exporte
        if validation.valide:
            return "export"
        
        # Si on a atteint le maximum d'itérations
        if state.compteur_boucles >= MAX_CORRECTION_LOOPS:
            print(f"⚠️ Limite d'itérations atteinte ({MAX_CORRECTION_LOOPS})")
            print(f"   Score final: {validation.score_conformite}%")
            return "export"
        
        # Sinon, on retourne au writer pour correction
        print(f"🔄 Correction nécessaire (tentative {state.compteur_boucles + 1}/{MAX_CORRECTION_LOOPS})")
        print(f"   Score actuel: {validation.score_conformite}%")
        
        # Incrémenter le compteur de boucles
        state.compteur_boucles += 1
        
        return "writer"
    
    def _wrap_agent(self, name: str, agent_func):
        """Wrapper pour mesurer le temps des agents"""
        def wrapped(state):
            start = time.time()
            try:
                result = agent_func(state)
                duration = time.time() - start
                self.performance.track_agent(name, duration)
                return result
            except Exception as e:
                duration = time.time() - start
                self.performance.track_agent(name, duration)
                raise e
        return wrapped
    
    def _build_graph(self) -> StateGraph:
        """
        Construit le graphe de workflow
        """
        workflow = StateGraph(GraphState)
        
        # Ajouter les nœuds (agents) avec wrapper
        workflow.add_node("context", self._wrap_agent("context", agent_context_node))
        workflow.add_node("program", self._wrap_agent("program", agent_program_node))
        workflow.add_node("similarite", self._wrap_agent("similarite", agent_similarite_node))
        workflow.add_node("writer", self._wrap_agent("writer", agent_writer_node))
        workflow.add_node("validation", self._wrap_agent("validation", agent_validation_node))
        workflow.add_node("export", self._wrap_agent("export", agent_export_node))
        
        # Définir le point d'entrée
        workflow.set_entry_point("context")
        
        # Définir les arêtes (flux séquentiel)
        workflow.add_edge("context", "program")
        workflow.add_edge("program", "similarite")
        workflow.add_edge("similarite", "writer")
        workflow.add_edge("writer", "validation")
        
        # Branchement conditionnel après validation
        workflow.add_conditional_edges(
            "validation",
            self._should_continue_correction,
            {
                "writer": "writer",
                "export": "export"
            }
        )
        
        # Fin du workflow après export
        workflow.add_edge("export", END)
        
        return workflow.compile()
    
    def run(self, state: GraphState) -> GraphState:
        """
        Exécute le workflow complet
        """
        print("="*60)
        print("🚀 DÉMARRAGE DE LA GÉNÉRATION DE FICHE DE COURS")
        print("="*60)
        
        self.performance.start()
        
        # Exécuter le graphe
        final_state = self.graph.invoke(state)
        
        self.performance.stop()
        summary = self.performance.get_summary()
        
        # Récupérer les informations de validation
        info = self._get_validation_info(final_state)
        
        print("="*60)
        print("✅ GÉNÉRATION TERMINÉE")
        print(f"   Score final: {info['score']}%")
        print(f"   Statut: {'✅ Validée' if info['valide'] else '⚠️ Exportée avec réserves'}")
        print(f"   Itérations: {info['compteur'] + 1}")
        print(f"   ⏱️  Temps total: {summary['total_time']}s")
        print("="*60)
        
        return final_state
    
    def visualize(self, output_path: str = "workflow_graph.png"):
        """
        Visualise le graphe de workflow
        Nécessite graphviz installé
        """
        try:
            from langchain_core.runnables.graph import MermaidDrawMethod
            
            mermaid_png = self.graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API
            )
            
            with open(output_path, 'wb') as f:
                f.write(mermaid_png)
            
            print(f"📊 Graphe de workflow sauvegardé: {output_path}")
        except Exception as e:
            print(f"⚠️ Impossible de générer le graphe: {e}")


def create_orchestrator() -> Orchestrateur:
    """Fonction factory pour créer l'orchestrateur"""
    return Orchestrateur()