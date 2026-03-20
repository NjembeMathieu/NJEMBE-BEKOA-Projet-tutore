# test_orchestrateur.py
from orchestrator import create_orchestrator
from state import GraphState, InputData, ContexteEnrichi, ReferentielData

# Créer un état minimal
state = GraphState(
    input_data=InputData(
        etablissement="Test",
        ville="Yaoundé",
        annee_scolaire="2026",
        classe="6ème",
        volume_horaire=2.0,
        matiere="Mathématiques",
        nom_professeur="M. Test",
        theme_chapitre="Test",
        sequence_ou_date="Seq1"
    )
)

state.contexte = ContexteEnrichi(
    cycle="Secondaire",
    niveau_exact="6ème",
    duree_categorisee="moyen",
    validation_coherence=True,
    ancrage_local={"ville": "Yaoundé", "etablissement": "Test"}
)

state.referentiel = ReferentielData(
    objectifs_officiels=["Objectif 1", "Objectif 2"],
    competences=["Comp 1"],
    gabarit="moyen"
)

# Lancer l'orchestrateur
orchestrator = create_orchestrator()
final_state = orchestrator.run(state)
print("✅ Test terminé")