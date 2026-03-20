"""
Gestion du Vector Store pour la recherche de similarité avec FAISS
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import hashlib
from datetime import datetime
# Configuration pour le mode hors ligne
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Définir le chemin ABSOLU du modèle
BASE_DIR = Path(__file__).parent.parent  # remonte de utils/ vers la racine
MODEL_PATH = BASE_DIR / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


# Afficher le chemin pour vérification
print(f"📁 Chemin du modèle: {MODEL_PATH.absolute()}")
print(f"📁 Le dossier existe: {MODEL_PATH.exists()}")

if MODEL_PATH.exists():
    print(f"📋 Fichiers trouvés:")
    for f in MODEL_PATH.glob("*"):
        if f.is_file():
            size = f.stat().st_size / (1024*1024)
            print(f"   - {f.name} ({size:.1f} MB)")
    if (MODEL_PATH / "1_Pooling").exists():
        print(f"   - 1_Pooling/ (dossier)")
else:
    print(f"❌ Dossier NON trouvé!")
    print(f"   Chemn complet: {MODEL_PATH.absolute()}")

# FAISS pour la recherche vectorielle
import faiss
from sentence_transformers import SentenceTransformer
# Chargement de documents
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CORPUS_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, SUPPORTED_SUBJECTS
from state import GraphState





class VectorStoreManager:
    """Gestionnaire du Vector Store avec FAISS et cache"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialise le vector store avec FAISS
        
        Args:
            dimension: Dimension des embeddings (384 pour MiniLM)
        """
        print("🔧 Initialisation du Vector Store avec FAISS")
        
        # Charger le modèle local
        self._load_local_model()
        self.dimension = dimension
        
        # Index FAISS (L2 distance - cosine similarity via normalisation)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Stockage des documents et métadonnées
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.document_ids: List[str] = []
        
        # Cache pour éviter de recalculer les mêmes embeddings
        self.cache_file = VECTORSTORE_DIR / "embeddings_cache.json"
        self.cache = self._load_cache()
        
        # Fichier de sauvegarde de l'index FAISS
        self.index_file = VECTORSTORE_DIR / "faiss_index.bin"
        self.metadata_file = VECTORSTORE_DIR / "metadata.json"
        
        # Charge l'index existant s'il existe
        self._load_existing_index()
    
    def _load_local_model(self):
        """Charge le modèle SentenceTransformer depuis le dossier local"""
        try:
            print(f"📁 Chargement du modèle depuis: {MODEL_PATH}")
            
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Dossier modèle non trouvé: {MODEL_PATH}")
            
            # Vérifier les fichiers essentiels
            fichiers_requis = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            fichiers_manquants = [f for f in fichiers_requis if not (MODEL_PATH / f).exists()]
            
            if fichiers_manquants:
                print(f"⚠️ Fichiers manquants: {fichiers_manquants}")
                print("   Tentative de chargement quand même...")
            
            # Charger le modèle en mode hors ligne
            self.embedding_model = SentenceTransformer(
                str(MODEL_PATH),
                local_files_only=True
            )
            print("✅ Modèle chargé avec succès depuis le dossier local")
            
        except Exception as e:
            print(f"❌ Erreur de chargement du modèle local: {e}")
            print("\n📋 Vérifiez que le dossier contient:")
            print("   - config.json")
            print("   - pytorch_model.bin (le plus important)")
            print("   - tokenizer.json")
            print("   - tokenizer_config.json")
            print("   - modules.json")
            print("   - 1_Pooling/config.json")
            raise
    
    def _load_cache(self) -> Dict:
        """Charge le cache des embeddings depuis le disque"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f" Erreur lors du chargement du cache: {e}")
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache des embeddings sur le disque"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" Erreur lors de la sauvegarde du cache: {e}")

    def _formater_markdown_ancien(self, state: GraphState) -> str:
        """Format markdown pour l'ancien format (compatibilité)"""
        fiche = state.fiche
        input_data = state.input_data
        validation = state.validation
        
        md = f"""# {fiche.titre}

    ---

    ## Informations Générales

    - **Établissement:** {fiche.etablissement}
    - **Ville:** {fiche.ville}
    - **Année scolaire:** {input_data.annee_scolaire}
    - **Classe:** {fiche.classe}
    - **Professeur:** {input_data.nom_professeur}
    - **Matière:** {input_data.matiere}
    - **Thème/Chapitre:** {input_data.theme_chapitre}
    - **Séquence/Date:** {input_data.sequence_ou_date}
    - **Volume horaire:** {input_data.volume_horaire}h

    ---

    ## Objectifs Pédagogiques

    """
        
        for i, objectif in enumerate(fiche.objectifs, 1):
            md += f"{i}. {objectif}\n"
        
        if fiche.situation_probleme:
            md += f"""
    ---

    ## Situation-Problème

    {fiche.situation_probleme}

    """
        
        md += f"""---

    ## Introduction

    {fiche.introduction}

    ---

    ## Développement du Cours

    {fiche.developpement}

    ---

    ## Activités Pédagogiques

    """
        
        for i, activite in enumerate(fiche.activites, 1):
            md += f"""
    ### Activité {i}: {activite.get('titre', f'Activité {i}')}

    **Durée:** {activite.get('duree', 'Non spécifiée')}

    {activite.get('description', '')}

    """
        
        md += f"""---

    ## Évaluation

    {fiche.evaluation}

    ---

    ## Conclusion

    {fiche.conclusion}

    """
        
        if fiche.references:
            md += """---

    ## Références

    """
            for ref in fiche.references:
                md += f"- {ref}\n"
        
        # Ajouter les métadonnées de validation
        md += f"""
    ---

    ## Métadonnées de Validation

    - **Score de conformité:** {validation.score_conformite}%
    - **Statut:** {'✅ Validée' if validation.valide else '⚠️ Nécessite corrections'}
    - **Mode de génération:** {state.mode_generation}
    - **Nombre d'itérations:** {state.compteur_boucles + 1}

    """
        
        return md


    def _load_existing_index(self):
        """Charge un index FAISS existant depuis le disque"""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                # Charge l'index FAISS
                self.index = faiss.read_index(str(self.index_file))
                
                # Charge les métadonnées
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                    self.document_ids = data.get('document_ids', [])
                
                print(f"✅ Index FAISS chargé: {len(self.documents)} documents")
                
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de l'index: {e}")
                # Réinitialise en cas d'erreur
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
                self.metadatas = []
                self.document_ids = []
        else:
            print("📭 Aucun index existant trouvé, création d'un nouvel index")
    
    def _save_index(self):
        """Sauvegarde l'index FAISS et les métadonnées sur le disque"""
        try:
            # Sauvegarde l'index FAISS
            faiss.write_index(self.index, str(self.index_file))
            
            # Sauvegarde les métadonnées
            data = {
                'documents': self.documents,
                'metadatas': self.metadatas,
                'document_ids': self.document_ids,
                'timestamp': datetime.now().isoformat(),
                'count': len(self.documents)
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Index sauvegardé: {len(self.documents)} documents")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde de l'index: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Génère un hash MD5 du texte pour le cache"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Récupère l'embedding d'un texte avec cache
        
        Args:
            text: Texte à encoder
            
        Returns:
            np.ndarray: Embedding normalisé
        """
        text_hash = self._get_text_hash(text)
        
        # Vérifie le cache
        if text_hash in self.cache:
            embedding = np.array(self.cache[text_hash], dtype='float32')
        else:
            # Encode le texte
            embedding = self.embedding_model.encode(text)
            
            # Normalise pour la similarité cosinus (plus pertinent avec L2)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Met en cache
            self.cache[text_hash] = embedding.tolist()
            self._save_cache()
        
        return embedding
    
    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Récupère les embeddings d'un batch de textes
        
        Args:
            texts: Liste de textes
            
        Returns:
            np.ndarray: Matrice d'embeddings normalisés
        """
        embeddings = []
        for text in texts:
            embeddings.append(self._get_embedding(text))
        
        return np.array(embeddings, dtype='float32')
    
    def load_corpus(self, matiere: str, niveau: str) -> int:
        """
        Charge les documents du corpus selon la matière et le niveau
        
        Args:
            matiere: Matière à charger
            niveau: Niveau scolaire
            
        Returns:
            int: Nombre de documents chargés
        """
        if matiere not in SUPPORTED_SUBJECTS:
            print(f"⚠️ Matière non supportée: {matiere}")
            return 0
        
        # Dossier spécifique à la matière
        matiere_dir = CORPUS_DIR / matiere
        if not matiere_dir.exists():
            print(f"📁 Dossier non trouvé: {matiere_dir}")
            return 0
        
        print(f"📚 Chargement du corpus: {matiere} - {niveau}")
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Charger les PDFs
        pdf_files = list(matiere_dir.glob("**/*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata.update({
                        "source": str(pdf_file),
                        "matiere": matiere,
                        "type": "officiel" if "programme" in pdf_file.stem.lower() else "complement",
                        "niveau": niveau,
                        "format": "pdf"
                    })
                
                documents.extend(docs)
                print(f"  ✅ PDF: {pdf_file.name} ({len(docs)} chunks)")
                
            except Exception as e:
                print(f"  ❌ Erreur PDF {pdf_file.name}: {e}")
        
        # Charger les fichiers texte
        txt_files = list(matiere_dir.glob("**/*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata.update({
                        "source": str(txt_file),
                        "matiere": matiere,
                        "type": "complement",
                        "niveau": niveau,
                        "format": "txt"
                    })
                
                documents.extend(docs)
                print(f"  ✅ TXT: {txt_file.name} ({len(docs)} chunks)")
                
            except Exception as e:
                print(f"  ❌ Erreur TXT {txt_file.name}: {e}")
        
        if not documents:
            print("📭 Aucun document trouvé")
            return 0
        
        # Découper les documents
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Total: {len(chunks)} chunks de texte")
        
        # Préparer les données pour FAISS
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Générer les IDs uniques
        ids = [f"{matiere}_{niveau}_{i}" for i in range(len(texts))]
        
        # Ajouter au vector store
        self.add_documents(texts, metadatas, ids)
        
        # Sauvegarder l'index
        self._save_index()
        
        return len(texts)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: Optional[List[str]] = None):
        """
        Ajoute des documents au vector store
        
        Args:
            texts: Liste des textes
            metadatas: Liste des métadonnées
            ids: Liste des IDs (optionnel)
        """
        if not texts:
            return
        
        print(f"➕ Ajout de {len(texts)} documents")
        
        # Générer les IDs si non fournis
        if ids is None:
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
        
        # Calculer les embeddings
        embeddings = self._get_embeddings_batch(texts)
        
        # Ajouter à l'index FAISS
        self.index.add(embeddings)
        
        # Stocker les documents et métadonnées
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.document_ids.extend(ids)
        
        print(f"✅ Documents ajoutés, total: {len(self.documents)}")
    
    def search_similar(
        self, 
        query: str, 
        matiere: Optional[str] = None, 
        niveau: Optional[str] = None, 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict]]:
        """
        Recherche des documents similaires
        
        Args:
            query: Requête de recherche
            matiere: Filtre par matière (optionnel)
            niveau: Filtre par niveau (optionnel)
            top_k: Nombre de résultats
            similarity_threshold: Seuil minimal de similarité
            
        Returns:
            List[Tuple[str, float, Dict]]: (contenu, score, metadata)
        """
        if len(self.documents) == 0:
            print("📭 Vector store vide")
            return []
        
        # Embedding de la requête
        query_embedding = self._get_embedding(query).reshape(1, -1)
        
        # Recherche dans FAISS
        k = min(top_k * 2, len(self.documents))  # Cherche plus large pour filtrer après
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            # Convertir distance L2 en similarité cosinus (car embeddings normalisés)
            # Pour des vecteurs normalisés: cosine_sim = 1 - (distance^2)/2
            distance = distances[0][i]
            similarity = 1.0 - (distance * distance) / 2.0
            
            # Filtrer par seuil
            if similarity < similarity_threshold:
                continue
            
            # Récupérer le document
            document = self.documents[idx]
            metadata = self.metadatas[idx]
            
            # Filtrer par matière et niveau si spécifiés
            if matiere and metadata.get('matiere') != matiere:
                continue
            if niveau and metadata.get('niveau') != niveau:
                continue
            
            results.append((document, similarity, metadata))
            
            # Arrêter si on a assez de résultats
            if len(results) >= top_k:
                break
        
        # Trier par similarité décroissante
        results.sort(key=lambda x: x[1], reverse=True)
        
        if results:
            print(f"🔍 Recherche: '{query[:50]}...' → {len(results)} résultats (max: {results[0][1]:.3f})")
        else:
            print(f"🔍 Recherche: '{query[:50]}...' → 0 résultat")
        
        return results
    
    def add_validated_fiche(
        self,
        fiche_id: str,
        content: str,
        metadata: Dict
    ):
        """
        Ajoute une fiche validée au vector store
        
        Args:
            fiche_id: ID unique de la fiche
            content: Contenu de la fiche
            metadata: Métadonnées de la fiche
        """
        print(f"💾 Sauvegarde de la fiche: {fiche_id}")
        
        # Préparer les métadonnées complètes
        full_metadata = {
            **metadata,
            "type": "fiche_validee",
            "fiche_id": fiche_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Ajouter au vector store
        self.add_documents([content], [full_metadata], [fiche_id])
        
        # Sauvegarder l'index
        self._save_index()
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques du vector store
        
        Returns:
            Dict: Statistiques
        """
        stats = {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'cache_size': len(self.cache),
            'materials': {}
        }
        
        # Compter par matière
        for metadata in self.metadatas:
            matiere = metadata.get('matiere', 'inconnu')
            if matiere not in stats['materials']:
                stats['materials'][matiere] = 0
            stats['materials'][matiere] += 1
        
        return stats
    
    def clear(self):
        """Vide complètement le vector store"""
        print("🗑️  Vidage du vector store")
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.document_ids = []
        
        # Supprimer les fichiers
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        print("✅ Vector store vidé")


# Singleton pour faciliter l'utilisation
_vectorstore_instance = None

def get_vectorstore() -> VectorStoreManager:
    """Retourne l'instance singleton du VectorStoreManager"""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStoreManager()
    return _vectorstore_instance


if __name__ == "__main__":
    # Test du vector store
    vs = VectorStoreManager()
    print("🧪 Test du Vector Store FAISS")
    
    # Ajout de quelques documents de test
    test_docs = [
        "Les fonctions affines sont de la forme f(x) = ax + b",
        "Une équation du second degré a la forme ax² + bx + c = 0",
        "Python est un langage de programmation interprété"
    ]
    
    test_metadatas = [
        {"matiere": "Mathématiques", "niveau": "Secondaire", "type": "test"},
        {"matiere": "Mathématiques", "niveau": "Secondaire", "type": "test"},
        {"matiere": "Informatique", "niveau": "Secondaire", "type": "test"}
    ]
    
    vs.add_documents(test_docs, test_metadatas)
    
    # Test de recherche
    results = vs.search_similar("fonctions linéaires", matiere="Mathématiques")
    for doc, score, meta in results:
        print(f"Score: {score:.3f} | {doc[:50]}...")
    
    # Afficher les stats
    stats = vs.get_stats()
    print(f"\n📊 Statistiques: {stats}")