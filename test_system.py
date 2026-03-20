"""
Script de test du système multi-agents
"""
import sys
from pathlib import Path

# Ajouter le répertoire au path
sys.path.append(str(Path(__file__).parent))

from state import GraphState, InputData
from orchestrator import create_orchestrator
from datetime import datetime


def test_generation_complete():
    """Test de génération complète d'une fiche"""
    print("="*60)
    print("TEST: Génération d'une fiche de cours")
    print("="*60)
    
    # Créer les données d'entrée
    input_data = InputData(
        etablissement="Lycée de Test",
        ville="Paris",
        annee_scolaire="2024-2025",
        classe="3ème",
        volume_horaire=2.0,
        matiere="Mathématiques",
        nom_professeur="M. Dupont",
        theme_chapitre="Les fonctions affines",
        sequence_ou_date="Séquence 3"
    )
    
    # Créer l'état initial
    initial_state = GraphState(input_data=input_data)
    
    # Créer et exécuter l'orchestrateur
    print("\n🚀 Lancement de l'orchestrateur...")
    orchestrator = create_orchestrator()
    
    try:
        final_state = orchestrator.run(initial_state)
        
        print("\n" + "="*60)
        print("RÉSULTATS DU TEST")
        print("="*60)
        
        print(f"\n✓ Cycle identifié: {final_state.contexte.cycle}")
        print(f"✓ Gabarit utilisé: {final_state.referentiel.gabarit}")
        print(f"✓ Mode de génération: {final_state.mode_generation}")
        print(f"✓ Score de conformité: {final_state.validation.score_conformite}%")
        print(f"✓ Statut: {'Validée ✓' if final_state.validation.valide else 'Non validée ✗'}")
        print(f"✓ Nombre d'itérations: {final_state.compteur_boucles + 1}")
        
        if final_state.fiche:
            print(f"\n✓ Titre de la fiche: {final_state.fiche.titre}")
            print(f"✓ Nombre d'objectifs: {len(final_state.fiche.objectifs)}")
            print(f"✓ Nombre d'activités: {len(final_state.fiche.activites)}")
            print(f"✓ Situation-problème: {'Oui' if final_state.fiche.situation_probleme else 'Non'}")
        
        if final_state.validation.commentaires:
            print(f"\n⚠️ Commentaires de validation:")
            for comment in final_state.validation.commentaires[:3]:
                print(f"   - {comment}")
        
        print("\n" + "="*60)
        print("✅ TEST RÉUSSI")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_context_agent():
    """Test de l'Agent Context"""
    print("\n" + "="*60)
    print("TEST: Agent Context")
    print("="*60)
    
    from agents.agent_context import AgentContext
    
    input_data = InputData(
        etablissement="Lycée Victor Hugo",
        ville="Lyon",
        annee_scolaire="2024-2025",
        classe="Terminale",
        volume_horaire=5.0,
        matiere="Informatique",
        nom_professeur="Mme Martin",
        theme_chapitre="Bases de données",
        sequence_ou_date="Séquence 1"
    )
    
    state = GraphState(input_data=input_data)
    agent = AgentContext()
    
    result = agent.process(state)
    
    print(f"\n✓ Cycle: {result.contexte.cycle}")
    print(f"✓ Niveau exact: {result.contexte.niveau_exact}")
    print(f"✓ Durée catégorisée: {result.contexte.duree_categorisee}")
    print(f"✓ Validation: {result.contexte.validation_coherence}")
    print(f"✓ Situation-problème requise: {result.necessite_situation_probleme}")
    
    print("\n✅ Agent Context OK")
    return True


def test_vectorstore():
    """Test du VectorStore"""
    print("\n" + "="*60)
    print("TEST: VectorStore")
    print("="*60)
    
    from utils.vectorstore import VectorStoreManager
    
    vs = VectorStoreManager()
    
    # Test d'ajout
    print("\n✓ VectorStore initialisé")
    print(f"✓ Collection: {vs.collection.name}")
    
    # Test de recherche
    results = vs.search_similar(
        query="fonctions mathématiques",
        matiere="Mathématiques",
        niveau="Secondaire",
        top_k=3
    )
    
    print(f"✓ Résultats de recherche: {len(results)} documents trouvés")
    
    print("\n✅ VectorStore OK")
    return True


if __name__ == "__main__":
    print("\n" + "🧪 SUITE DE TESTS DU SYSTÈME MULTI-AGENTS")
    print("="*60)
    
    tests = [
        ("Agent Context", test_context_agent),
        ("VectorStore", test_vectorstore),
        ("Génération Complète", test_generation_complete),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Erreur dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\nRésultats: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
    else:
        print(f"\n⚠️ {total - passed} test(s) échoué(s)")
