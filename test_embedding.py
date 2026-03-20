# test_embedding.py
from sentence_transformers import SentenceTransformer
from pathlib import Path

model_path = Path("models") / "paraphrase-multilingual-MiniLM-L12-v2"
print(f"📁 Chargement depuis: {model_path.absolute()}")

try:
    model = SentenceTransformer(str(model_path), local_files_only=True)
    print("✅ Modèle chargé!")
    
    # Test simple
    embedding = model.encode("Test")
    print(f"✅ Embedding réussi! Dimension: {len(embedding)}")
    
except Exception as e:
    print(f"❌ Erreur: {e}")