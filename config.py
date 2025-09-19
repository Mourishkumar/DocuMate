#import os
#import torch
#MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
#OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
#OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
#user_local_embeddings = False
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
import torch

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# LLM for chatting
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

# Embedding model
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

user_local_embeddings = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
