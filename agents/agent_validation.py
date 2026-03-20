"""
Agent Validation - Auditeur & Juge de Conformité
Contrôleur qualité final avec seuils de tolérance discriminants
"""
from typing import List, Tuple
from state import GraphState, ValidationResult
from config import VALIDATION_THRESHOLDS


class AgentValidation:
    """Agent de validation de la conformité de la fiche"""
    
    def __init__(self):
        self.thresholds = VALIDATION_THRESHOLDS
    
    def _verifier_champs_obligatoires(self, state: GraphState) -> Tuple[List[str], List[str]]:
        """Vérifie la présence des champs obligatoires"""
        manquants = []
        commentaires = []
        fiche = state.fiche
        
        # Vérifier les métadonnées
        if not fiche.etablissement or fiche.etablissement.strip() == "":
            manquants.append("Nom de l'établissement")
        
        if not fiche.ville or fiche.ville.strip() == "":
            manquants.append("Ville")
        
        if not fiche.classe or fiche.classe.strip() == "":
            manquants.append("Classe")
        
        # Vérifier le contenu pédagogique
        if not fiche.objectifs or len(fiche.objectifs) == 0:
            manquants.append("Objectifs pédagogiques")
            commentaires.append("Aucun objectif pédagogique défini")
        
        if not fiche.introduction or len(fiche.introduction) < 50:
            manquants.append("Introduction suffisamment développée")
            commentaires.append("L'introduction est trop courte (min 50 caractères)")
        
        if not fiche.developpement or len(fiche.developpement) < 200:
            manquants.append("Développement suffisamment détaillé")
            commentaires.append("Le développement est insuffisant (min 200 caractères)")
        
        if not fiche.activites or len(fiche.activites) == 0:
            manquants.append("Activités pédagogiques")
            commentaires.append("Aucune activité proposée")
        
        if not fiche.evaluation or len(fiche.evaluation) < 30:
            manquants.append("Évaluation définie")
            commentaires.append("L'évaluation est absente ou trop courte")
        
        return manquants, commentaires
    
    def _verifier_situation_probleme(self, state: GraphState) -> Tuple[bool, List[str]]:
        """Vérifie la situation-problème si nécessaire"""
        commentaires = []
        
        if state.necessite_situation_probleme:
            fiche = state.fiche
            
            if not fiche.situation_probleme:
                commentaires.append("CRITIQUE: Situation-problème OBLIGATOIRE pour le Secondaire mais absente")
                return False, commentaires
            
            if len(fiche.situation_probleme) < 100:
                commentaires.append("La situation-problème est trop courte (min 100 caractères)")
                return False, commentaires
            
            # Vérifier l'ancrage local
            ville = state.input_data.ville.lower()
            if ville not in fiche.situation_probleme.lower():
                commentaires.append(f"La situation-problème devrait mentionner la ville ({state.input_data.ville})")
        
        return True, commentaires
    
    def _verifier_objectifs_pedagogiques(self, state: GraphState) -> Tuple[float, List[str]]:
        """Vérifie que les objectifs du référentiel sont traités"""
        if not state.referentiel or not state.referentiel.objectifs_officiels:
            return 100.0, []  # Pas d'objectifs officiels à vérifier
        
        objectifs_officiels = state.referentiel.objectifs_officiels
        fiche = state.fiche
        
        # Convertir le contenu de la fiche en texte pour recherche
        contenu_complet = " ".join([
            fiche.introduction,
            fiche.developpement,
            " ".join([act.get('description', '') for act in fiche.activites]),
            fiche.evaluation,
            fiche.conclusion
        ]).lower()
        
        objectifs_traites = 0
        commentaires = []
        
        for i, objectif in enumerate(objectifs_officiels, 1):
            # Extraire les mots-clés de l'objectif
            mots_cles = [mot.lower() for mot in objectif.split() if len(mot) > 4]
            
            # Vérifier si au moins 50% des mots-clés sont présents
            mots_trouves = sum(1 for mot in mots_cles if mot in contenu_complet)
            
            if len(mots_cles) > 0:
                ratio = mots_trouves / len(mots_cles)
                if ratio >= 0.5:
                    objectifs_traites += 1
                else:
                    commentaires.append(f"Objectif #{i} insuffisamment traité: {objectif[:60]}...")
        
        # Calculer le pourcentage
        if len(objectifs_officiels) > 0:
            pourcentage = (objectifs_traites / len(objectifs_officiels)) * 100
        else:
            pourcentage = 100.0
        
        return pourcentage, commentaires
    
    def _verifier_structure_gabarit(self, state: GraphState) -> Tuple[bool, List[str]]:
        """Vérifie la conformité avec le gabarit"""
        if not state.referentiel:
            return True, []
        
        gabarit = state.referentiel.gabarit
        fiche = state.fiche
        commentaires = []
        
        # Vérifier le nombre d'activités
        nb_activites = len(fiche.activites)
        
        gabarit_config = {
            "court": {"min": 1, "max": 2},
            "moyen": {"min": 2, "max": 4},
            "etendu": {"min": 4, "max": 6}
        }
        
        if gabarit in gabarit_config:
            config = gabarit_config[gabarit]
            if nb_activites < config["min"]:
                commentaires.append(
                    f"Nombre d'activités insuffisant ({nb_activites}) pour un cours {gabarit}. "
                    f"Minimum attendu: {config['min']}"
                )
                return False, commentaires
            elif nb_activites > config["max"]:
                commentaires.append(
                    f"Trop d'activités ({nb_activites}) pour un cours {gabarit}. "
                    f"Maximum recommandé: {config['max']}"
                )
        
        return True, commentaires
    
    def _calculer_score_global(
        self, 
        manquants: List[str],
        score_objectifs: float,
        situation_probleme_ok: bool,
        structure_ok: bool
    ) -> float:
        """Calcule le score de conformité global"""
        # Score de base
        score = 100.0
        
        # Pénalités pour champs manquants (5% par champ)
        score -= len(manquants) * 5
        
        # Score des objectifs pédagogiques (poids 40%)
        score = score * 0.6 + score_objectifs * 0.4
        
        # Pénalité si situation-problème manquante (critique pour Secondaire)
        if not situation_probleme_ok:
            score -= 20
        
        # Pénalité si structure non conforme
        if not structure_ok:
            score -= 10
        
        return max(0.0, min(100.0, score))
    
    def _generer_corrections(
        self,
        manquants: List[str],
        commentaires: List[str],
        score_objectifs: float
    ) -> List[str]:
        """Génère les corrections à apporter"""
        corrections = []
        
        if manquants:
            corrections.append(f"Compléter les éléments manquants: {', '.join(manquants)}")
        
        if score_objectifs < 100:
            corrections.append(
                "Approfondir le traitement des objectifs pédagogiques officiels. "
                "Assurez-vous que chaque objectif est explicitement abordé dans le cours."
            )
        
        return corrections
    
    def process(self, state: GraphState) -> GraphState:
        """
        Valide la fiche générée
        """
        # Vérifications
        manquants, commentaires_champs = self._verifier_champs_obligatoires(state)
        situation_ok, commentaires_situation = self._verifier_situation_probleme(state)
        score_objectifs, commentaires_objectifs = self._verifier_objectifs_pedagogiques(state)
        structure_ok, commentaires_structure = self._verifier_structure_gabarit(state)
        
        # Combiner les commentaires
        tous_commentaires = (
            commentaires_champs + 
            commentaires_situation + 
            commentaires_objectifs + 
            commentaires_structure
        )
        
        # Calculer le score global
        score_global = self._calculer_score_global(
            manquants,
            score_objectifs,
            situation_ok,
            structure_ok
        )
        
        # Déterminer le seuil de validation selon le cycle
        cycle = state.contexte.cycle
        seuil = self.thresholds.get(cycle, 50)
        
        # Décision de validation
        valide = score_global >= seuil
        
        # Générer les corrections si nécessaire
        corrections = [] if valide else self._generer_corrections(
            manquants,
            tous_commentaires,
            score_objectifs
        )
        
        # Créer le résultat de validation
        validation = ValidationResult(
            valide=valide,
            score_conformite=round(score_global, 2),
            commentaires=tous_commentaires,
            elements_manquants=manquants,
            corrections_requises=corrections
        )
        
        # Mettre à jour l'état
        state.validation = validation
        
        return state


def agent_validation_node(state: GraphState) -> GraphState:
    """Node LangGraph pour l'Agent Validation"""
    agent = AgentValidation()
    return agent.process(state)