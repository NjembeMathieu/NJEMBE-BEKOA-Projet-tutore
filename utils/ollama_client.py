"""
Client Ollama pour remplacer Gemini
"""
import requests
import json
from typing import Dict, Any, Optional
import time

class OllamaClient:
    """Client pour interagir avec Ollama en local"""
    
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._check_connection()
    
    def _check_connection(self):
        """Vérifie la connexion à Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [m["name"] for m in models["models"]]
                if self.model not in available_models:
                    print(f"⚠️ Modèle {self.model} non trouvé. Modèles disponibles: {available_models}")
                    print(f"💡 Utilisez: ollama pull {self.model}")
                else:
                    print(f"✅ Connecté à Ollama avec {self.model}")
            else:
                print("❌ Erreur de connexion à Ollama")
        except Exception as e:
            print(f"❌ Impossible de se connecter à Ollama: {e}")
            print("💡 Vérifiez qu'Ollama est lancé (ollama serve)")
    
    def generate_content(
        self, 
        contents: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        response_mime_type: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Any:
        """
        Génère du contenu avec Ollama (compatible avec l'API Gemini)
        """
        # Construction du message
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": contents})
        
        # Construction de la requête
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
                "num_ctx": 32768,  # Contexte de 32K tokens
            }
        }
        
        # Si on demande du JSON, on force le format
        if response_mime_type == "application/json":
            payload["format"] = "json"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Créer un objet compatible avec l'API Gemini
            class Response:
                def __init__(self, text):
                    self.text = text
            
            return Response(result["message"]["content"])
            
        except Exception as e:
            print(f"Erreur Ollama: {e}")
            raise
    
    def generate_with_retry(self, **kwargs):
        """
        Version avec retry pour gérer les erreurs temporaires
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.generate_content(**kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Tentative {attempt + 1} échouée, nouvel essai dans {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Échec après {max_retries} tentatives")
                    raise


class Models:
    """Classe pour simuler genai.models"""
    def __init__(self, client):
        self.client = client
    
    def generate_content(self, **kwargs):
        return self.client.generate_content(**kwargs)


class Client:
    """Wrapper pour simuler genai.Client"""
    def __init__(self, model: str = "gemma3:4b"):
        self.models = Models(OllamaClient(model=model))


# Pour une compatibilité maximale avec votre code existant
def create_ollama_client(api_key=None, model: str = "gemma3:4b"):
    """
    Crée un client compatible avec l'API Gemini
    Note: api_key est ignoré (Ollama est local)
    """
    return Client(model=model)