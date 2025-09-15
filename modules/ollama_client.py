import requests
from config import OLLAMA_URL, OLLAMA_MODEL


def query_ollama(prompt: str, context_docs: list = None, model: str = OLLAMA_MODEL) -> str:
    """
    Queries the Ollama server with a prompt and optional context from the RAG dataset.
    """
    # Define the core personality of the assistant
    system_message = (
        "You are a witty, slightly silly, and supportive anime friend. "
        "Your replies are short, casual, and sound like a real human. "
        "Never say you are an AI. If you don't know an answer, just be playful about it."
    )

    # If we have context documents, format them into the system message
    if context_docs:
        system_message += "\n\nHere are some examples of how you talk:\n"
        for doc in context_docs:
            instruction = doc.get("instruction", "")
            output = doc.get("output", "")
            system_message += f"- User: '{instruction}'\nYou: '{output}'\n"

    # Assemble the message payload for the API
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        # Handle different response structures from Ollama
        if "message" in data:
            return data["message"]["content"].strip()
        elif "response" in data:
            return data["response"].strip()

        return str(data)  # Fallback for unexpected response format

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error: The Ollama server responded with an error.")
        print(f"   Status Code: {http_err.response.status_code}")
        print(f"   Response: {http_err.response.text}")
        return "Sorry, the AI server is having trouble right now. Please check the Ollama logs."
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Connection Error: Could not connect to the Ollama server.")
        print(f"   Is Ollama running?")
        return "Sorry, I can't connect to the AI server. Please make sure Ollama is running."