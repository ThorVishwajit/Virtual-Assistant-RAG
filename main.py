import os, json, wave
from config import OUTPUT_DIR
from modules.stt import speech_to_text
from modules.ollama_client import query_ollama
from modules.tts import synthesize_tts
from modules.emotion_engine import EmotionEngine
from modules.lip_sync import run_rhubarb_on_wav
from modules.memory_manager import save_memory, load_memory

def get_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def main():
    memory = load_memory()
    engine = EmotionEngine()

    # --- Step 1: STT
    user_text = speech_to_text()

    # --- Step 2: LLM
    llm_text = query_ollama(user_text)

    # --- Step 3: TTS
    wav_path = synthesize_tts(llm_text)
    duration = get_wav_duration(wav_path)

    # --- Step 4: Emotion Analysis
    emotion_plan = engine.analyze(llm_text)

    # --- Step 5: Lip Sync
    lip_sync = run_rhubarb_on_wav(wav_path)

    # --- Step 6: Combine into ONE JSON packet
    packet = {
        "text": llm_text,
        "audio_file": os.path.basename(wav_path),
        "timestamps": {"start": 0.0, "end": duration},
        "emotion_plan": emotion_plan,
        "lip_sync": lip_sync
    }

    # Save JSON next to audio
    json_path = wav_path.replace(".wav", ".json")
    with open(json_path, "w") as f:
        json.dump(packet, f, indent=2)

    print(f"✅ Packet saved: {json_path}")

    # --- Step 7: Save memory separately (context only)
    memory["last_response"] = {"text": llm_text}
    save_memory(memory)

if __name__ == "__main__":
    main()
