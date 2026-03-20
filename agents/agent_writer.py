"""
Agent Writer - Producteur Adaptatif & Rédacteur
Génère le contenu de la fiche avec Ollama
"""
import streamlit as st
import time
from typing import Dict, List
import json
from state import GraphState, FicheContent
from config import (
    OLLAMA_MODEL, 
    OLLAMA_TEMPERATURE, 
    OLLAMA_MAX_TOKENS,
    DURATION_TEMPLATES
)
from utils.ollama_client import create_ollama_client

# Créer le client Ollama (pas besoin de clé API)
client = create_ollama_client(model=OLLAMA_MODEL)


class AgentWriter:
    """Agent de génération de contenu via Ollama"""
    
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.generation_config = {
            "temperature": OLLAMA_TEMPERATURE,
            "max_output_tokens": OLLAMA_MAX_TOKENS,
        }
        self.gabarits = DURATION_TEMPLATES
    
    def _construire_prompt_creation_complete(self, state: GraphState) -> tuple[str, str]:
        """
        Construit le prompt pour une création complète
        Retourne: (system_prompt, user_prompt)
        """
        input_data = state.input_data
        contexte = state.contexte
        referentiel = state.referentiel
        
        # Déterminer le type de cours
        duree_cat = contexte.duree_categorisee
        gabarit = {
            "court": {"min": 1, "max": 2, "type": "court"},
            "moyen": {"min": 2, "max": 4, "type": "moyen"},
            "etendu": {"min": 4, "max": 6, "type": "étendu"}
        }.get(duree_cat, {"min": 2, "max": 4, "type": "moyen"})
        
        # System prompt (rôle de l'agent)
        system_prompt = """Tu es un expert pédagogue français, spécialiste de la création de fiches de cours.
Tu respectes scrupuleusement les programmes officiels et tu adaptes ton langage au niveau des élèves.
Tu réponds TOUJOURS en français avec un format JSON strict.
Tes fiches sont détaillées, structurées et adaptées au contexte local de l'établissement."""
        
        # User prompt (la tâche)
        user_prompt = f"""Crée une fiche de cours professionnelle et complète.

INFORMATIONS OBLIGATOIRES:
- Établissement: {input_data.etablissement}
- Ville: {input_data.ville}
- Année scolaire: {input_data.annee_scolaire}
- Classe: {input_data.classe}
- Professeur: {input_data.nom_professeur}
- Matière: {input_data.matiere}
- Thème: {input_data.theme_chapitre}
- Volume horaire: {input_data.volume_horaire}h

CONTEXTE PÉDAGOGIQUE:
- Cycle: {contexte.cycle}
- Niveau exact: {contexte.niveau_exact}
- Type de cours: {duree_cat} ({gabarit['min']}-{gabarit['max']} activités)

OBJECTIFS PÉDAGOGIQUES:
{chr(10).join(f"- {obj}" for obj in referentiel.objectifs_officiels)}

COMPÉTENCES À DÉVELOPPER:
{chr(10).join(f"- {comp}" for comp in referentiel.competences)}

ANCRAGE LOCAL:
- Ville: {contexte.ancrage_local['ville']}
- Établissement: {contexte.ancrage_local['etablissement']}
- {contexte.ancrage_local.get('suggestions', 'Utilise des exemples locaux')}
"""
        
        if state.necessite_situation_probleme:
            user_prompt += f"""

SITUATION-PROBLÈME OBLIGATOIRE:
Crée une situation-problème concrète et engageante liée à {input_data.ville} ou {input_data.etablissement}.
La situation doit:
1. Partir d'un contexte réel et authentique
2. Poser un défi intellectuel en lien avec {input_data.theme_chapitre}
3. Mobiliser les objectifs pédagogiques
4. Être adaptée aux élèves de {input_data.classe}

La situation-problème doit être détaillée (150-250 mots).
"""
        
        user_prompt += """

FORMAT DE SORTIE (JSON UNIQUEMENT):
{
    "titre": "Titre de la fiche (incluant le thème et la classe)",
    "etablissement": "nom",
    "ville": "ville",
    "classe": "classe",
    "objectifs": ["objectif 1", "objectif 2", ...],
    "situation_probleme": "texte ou null",
    "introduction": "Introduction du cours (contextualisation)",
    "developpement": "Contenu principal structuré en plusieurs parties",
    "activites": [
        {"titre": "Activité 1", "description": "...", "duree": "XXmin"},
        {"titre": "Activité 2", "description": "...", "duree": "XXmin"}
    ],
    "evaluation": "Description de l'évaluation (exercices, QCM, etc.)",
    "conclusion": "Conclusion et ouverture",
    "references": ["référence 1", "référence 2"]
}

IMPORTANT: 
- Le développement doit être détaillé et structuré (minimum 500 mots pour un cours moyen/étendu)
- Les activités doivent être variées et adaptées à la durée
- Inclus des exemples concrets liés à {input_data.ville}

Génère maintenant la fiche complète en JSON."""
        
        return system_prompt, user_prompt
    
    def _construire_prompt_adaptation(self, state: GraphState) -> tuple[str, str]:
        """Construit le prompt pour une adaptation"""
        input_data = state.input_data
        contexte = state.contexte
        fiche_existante = state.similarite.contenu_existant
        
        system_prompt = "Tu es un expert pédagogue qui adapte des fiches de cours existantes à de nouveaux contextes."
        
        user_prompt = f"""Adapte cette fiche de cours existante.

NOUVEAU CONTEXTE:
- Établissement: {input_data.etablissement}
- Ville: {input_data.ville}
- Classe: {input_data.classe}
- Professeur: {input_data.nom_professeur}
- Thème: {input_data.theme_chapitre}
- Volume horaire: {input_data.volume_horaire}h

FICHE EXISTANTE:
{fiche_existante}

CONSIGNES D'ADAPTATION:
1. Conserve la structure et les objectifs principaux
2. ADAPTE tous les exemples et situations à {input_data.ville}
3. Mets à jour les références à l'établissement et au professeur
4. Ajuste la complexité pour le niveau {contexte.niveau_exact}
5. Si une situation-problème existe, adapte-la à {input_data.ville}

Format JSON identique à la création.

Génère la fiche adaptée en JSON."""
        
        return system_prompt, user_prompt
    
    def _construire_prompt_correction(self, state: GraphState) -> tuple[str, str]:
        """Construit le prompt pour corriger une fiche"""
        validation = state.validation
        fiche_actuelle = state.fiche.model_dump_json(indent=2, ensure_ascii=False)
        
        system_prompt = "Tu es un expert pédagogue qui corrige des fiches de cours pour les rendre conformes."
        
        user_prompt = f"""Corrige cette fiche de cours qui a été rejetée par le système de validation.

FICHE ACTUELLE:
{fiche_actuelle}

PROBLÈMES IDENTIFIÉS:
{chr(10).join(f"- {commentaire}" for commentaire in validation.commentaires)}

ÉLÉMENTS MANQUANTS:
{chr(10).join(f"- {elem}" for elem in validation.elements_manquants)}

CORRECTIONS REQUISES:
{chr(10).join(f"- {correction}" for correction in validation.corrections_requises)}

CONSIGNES:
1. Conserve ce qui fonctionne déjà
2. Corrige UNIQUEMENT les problèmes identifiés
3. Assure-toi que TOUS les objectifs pédagogiques sont traités
4. Vérifie que la structure est complète
5. Si une situation-problème est requise, ajoute-la ou améliore-la

Format JSON identique.

Génère la fiche corrigée en JSON."""
        
        return system_prompt, user_prompt
    
    def _parser_json_response(self, response_text: str, state: GraphState) -> Dict:
        """Parse la réponse JSON d'Ollama"""
        text = response_text.strip()
        
        # Nettoyer les éventuels markdown
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Essayer de parser le JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"❌ Erreur JSON: {e}")
            print(f"📝 Texte reçu: {text[:500]}")
            
            # Tentative de récupération : chercher le premier { et dernier }
            try:
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = text[start:end+1]
                    return json.loads(json_str)
            except:
                pass
            
            # Fallback en cas d'erreur
            return {
                "titre": f"Fiche - {state.input_data.theme_chapitre}",
                "etablissement": state.input_data.etablissement,
                "ville": state.input_data.ville,
                "classe": state.input_data.classe,
                "objectifs": state.referentiel.objectifs_officiels if state.referentiel else [],
                "situation_probleme": None,
                "introduction": "Introduction en cours de génération...",
                "developpement": "Le contenu détaillé sera disponible prochainement.",
                "activites": [],
                "evaluation": "À définir",
                "conclusion": "À compléter",
                "references": []
            }
    
    def process(self, state: GraphState) -> GraphState:
        """
        Génère ou adapte le contenu de la fiche
        """
        # Construire les prompts selon le contexte
        if state.compteur_boucles > 0 and state.validation and not state.validation.valide:
            system_prompt, user_prompt = self._construire_prompt_correction(state)
            if not hasattr(state, 'historique_corrections'):
                state.historique_corrections = []
            state.historique_corrections.append(f"Itération {state.compteur_boucles}: Correction après rejet")
        elif hasattr(state, 'mode_generation') and state.mode_generation == "adaptation":
            system_prompt, user_prompt = self._construire_prompt_adaptation(state)
        else:
            system_prompt, user_prompt = self._construire_prompt_creation_complete(state)
        
        try:
            # Génération avec Ollama
            response = client.models.generate_content(
                contents=user_prompt,
                temperature=self.generation_config["temperature"],
                max_output_tokens=self.generation_config["max_output_tokens"],
                response_mime_type="application/json",
                system_prompt=system_prompt
            )
            
            # Petit délai pour éviter de surcharger Ollama
            time.sleep(2)
            
            # Parser la réponse
            fiche_dict = self._parser_json_response(response.text, state)
            
            # S'assurer que tous les champs requis sont présents
            required_fields = ["titre", "etablissement", "ville", "classe", "objectifs", 
                              "introduction", "developpement", "activites", "evaluation", "conclusion"]
            
            for field in required_fields:
                if field not in fiche_dict:
                    if field == "objectifs":
                        fiche_dict[field] = state.referentiel.objectifs_officiels if state.referentiel else []
                    elif field == "activites":
                        fiche_dict[field] = []
                    else:
                        fiche_dict[field] = ""
            
            fiche = FicheContent(**fiche_dict)
            state.fiche = fiche
            print(f"✅ Fiche générée: {fiche.titre}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback en cas d'erreur
            state.fiche = FicheContent(
                titre=f"Fiche de cours - {state.input_data.theme_chapitre}",
                etablissement=state.input_data.etablissement,
                ville=state.input_data.ville,
                classe=state.input_data.classe,
                objectifs=state.referentiel.objectifs_officiels if state.referentiel else [],
                introduction="Contenu en cours de génération...",
                developpement="Le contenu détaillé sera disponible prochainement.",
                evaluation="À définir",
                conclusion="À compléter"
            )
        
        return state


def agent_writer_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Writer"""
    agent = AgentWriter()
    return agent.process(state)