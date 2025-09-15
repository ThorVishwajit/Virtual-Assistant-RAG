from modules.rag_engine import build_rag_engine
from modules.emotion_engine import EmotionEngine
from modules.stt import speech_to_text
from modules.ollama_client import query_ollama
from modules.tts import synthesize_tts
from modules.lip_sync import run_rhubarb_on_wav
from modules.memory_manager import save_memory, load_memory
import os, json, wave
import re
from config import OUTPUT_DIR

def get_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def main():
    # --- Init Engines ---
    rag = build_rag_engine()
    memory = load_memory()
    engine = EmotionEngine()

    # --- Step 1: Speech-to-Text ---
    user_text = speech_to_text()
    print(f"🎤 User said: {user_text}")

    # --- Step 2: RAG Retrieval ---
    best_doc, score, context_docs = rag.query(user_text)
    if score >= 0.7:
        llm_text = best_doc.get("output", "")
        print(f"🎯 RAG matched with high confidence! score={score:.2f}")
    else:
        print(f"🤖 RAG score is low ({score:.2f}). Augmenting prompt with context...")
        llm_text = query_ollama(user_text, context_docs=context_docs)

    # --- Step 3: Text Cleanup and Emotion Analysis ---
    clean_text = re.sub(r"<emotion:[a-zA-Z_]+>", "", llm_text).strip()
    emotion_plan = engine.analyze(llm_text)
    if isinstance(emotion_plan, dict) and "primary" in emotion_plan:
        print(f"🎭 Primary Emotion Detected: '{emotion_plan['primary']}'")

    # --- Step 4: Text-to-Speech ---
    print("🎙 Generating speech...")
    wav_path = synthesize_tts(clean_text)
    duration = get_wav_duration(wav_path)
    print(f"✅ Audio generated: {wav_path} ({duration:.2f}s)")

    # --- Step 5: Lip Sync ---
    print("👄 Running lip sync analysis...")
    lip_sync = run_rhubarb_on_wav(wav_path)
    print(f"✅ Lip sync cues: {len(lip_sync)} phonemes")

    # --- Step 6: Save packet ---
    packet = {
        "text": llm_text,
        "audio_file": os.path.basename(wav_path),
        "timestamps": {"start": 0.0, "end": duration},
        "emotion_plan": emotion_plan,
        "lip_sync": lip_sync
    }
    json_path = wav_path.replace(".wav", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2, ensure_ascii=False)
    print(f"✅ Packet saved: {json_path}")

    # --- Step 7: Save memory ---
    memory["last_response"] = {"text": llm_text}
    save_memory(memory)

if __name__ == "__main__":
    main()