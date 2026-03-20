"""
Script de test pour vérifier qu'Ollama fonctionne correctement
"""
from utils.ollama_client import OllamaClient

def test_ollama():
    print("🔍 Test de connexion à Ollama...")
    
    client = OllamaClient(model="gemma3:4b")
    
    # Test simple
    print("\n📝 Test de génération simple...")
    response = client.generate_content(
        contents="Explique le théorème de Pythagore en une phrase.",
        temperature=0.7,
        max_output_tokens=100
    )
    print(f"✅ Réponse: {response.text}")
    
    # Test JSON
    print("\n📊 Test de génération JSON...")
    response = client.generate_content(
        contents="Donne-moi un objet JSON avec les champs: nom, age, ville",
        temperature=0.7,
        max_output_tokens=200,
        response_mime_type="application/json"
    )
    print(f"✅ Réponse: {response.text}")
    
    print("\n✅ Tous les tests sont passés !")

if __name__ == "__main__":
    test_ollama()