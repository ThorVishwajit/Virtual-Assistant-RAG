import requests
from config import OLLAMA_URL, OLLAMA_MODEL

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "message" in data:
        return data["message"]["content"].strip()
    elif "response" in data:
        return data["response"].strip()
    return str(data)
