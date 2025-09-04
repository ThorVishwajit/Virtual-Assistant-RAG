import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "roberta-base-go_emotions")

# Ensure folders exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rhubarb path (set via env or PATH)
RHUBARB_PATH = os.environ.get("RHUBARB_PATH", "rhubarb")

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:7b"
