import time
import os
import wave
import struct
import re
from TTS.api import TTS
import soundfile as sf
from config import OUTPUT_DIR

# Initialize the High-Quality single-speaker TTS model.
print("🎙️ Initializing High-Quality Female Voice (LJSpeech VITS)...")
print("   (The model will be downloaded on the first run)")
try:
    # --- THIS IS THE FIX ---
    # We are now using the VITS model trained on the LJSpeech dataset.
    # This provides one consistent, high-quality, pleasant female voice.
    tts_model = TTS(model_name="tts_models/en/ljspeech/vits",
                    progress_bar=False, gpu=True)
    print("✅ TTS model loaded successfully on GPU.")
except Exception as e:
    print(f"❌ Failed to load TTS model: {e}")
    tts_model = None

def clean_text_for_tts(text: str) -> str:
    """Clean text by removing emojis and other non-ASCII characters."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def make_silent_wav(path: str, duration_s: float = 2.0, sr: int = 16000):
    """Creates a silent WAV file as a fallback."""
    nframes = int(duration_s * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        silence = struct.pack("<h", 0)
        wf.writeframes(silence * nframes)

def synthesize_tts(text: str) -> str:
    """
    Synthesizes text to speech using the single LJSpeech female voice.
    """
    ts = int(time.time() * 1000)
    wav_path = os.path.join(OUTPUT_DIR, f"utterance_{ts}.wav")
    text = clean_text_for_tts(text)

    if not tts_model:
        print("TTS model not available, creating silent WAV.")
        make_silent_wav(wav_path, duration_s=max(1.0, len(text.split()) * 0.35))
        return wav_path

    try:
        # Use the standard tts_to_file method. No pitch or speaker is needed.
        tts_model.tts_to_file(text=text, file_path=wav_path)

        # Resample to 16kHz PCM16 for Rhubarb Lip Sync compatibility
        data, sr = sf.read(wav_path)
        sf.write(wav_path, data, 16000, subtype="PCM_16")

    except Exception as e:
        print(f"TTS synthesis failed: {e}. Creating silent WAV instead.")
        make_silent_wav(wav_path, duration_s=max(1.0, len(text.split()) * 0.35))

    return wav_path