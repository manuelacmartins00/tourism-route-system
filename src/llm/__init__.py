# src/llm/__init__.py
#from .llm_orchestrator import LlamaOrchestrator, UserPreferences

#__all__ = ['LlamaOrchestrator', 'UserPreferences']

def __init__(self, api_key: str = None):  # api_key deixa de ser necessário
    self.model = "llama3.1:8b-instruct-q4_K_M"
    self.ollama_url = "http://localhost:11434/api/chat"