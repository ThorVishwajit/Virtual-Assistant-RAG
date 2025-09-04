import time
import wave, struct
from TTS.api import TTS
from config import OUTPUT_DIR

def make_silent_wav(path: str, duration_s: float = 2.0, sr: int = 22050):
    nframes = int(duration_s * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        silence = struct.pack("<h", 0)
        wf.writeframes(silence * nframes)

def synthesize_tts(text: str) -> str:
    ts = int(time.time() * 1000)
    wav_path = f"{OUTPUT_DIR}/utterance_{ts}.wav"
    try:
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        tts.tts_to_file(text=text, file_path=wav_path)
    except Exception:
        make_silent_wav(wav_path, duration_s=max(1.0, len(text.split()) * 0.35))
    return wav_path
