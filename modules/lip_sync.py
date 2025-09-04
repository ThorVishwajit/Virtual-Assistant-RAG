import subprocess, shutil, json
from config import RHUBARB_PATH

def run_rhubarb_on_wav(wav_path: str):
    rh = shutil.which(RHUBARB_PATH)
    if not rh:
        return [{"time": 0.0, "phoneme": "X"}]
    try:
        proc = subprocess.run([rh, "-f", "json", wav_path], capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        cues = data.get("mouthCues", [])
        return [{"time": float(c["start"]), "phoneme": c["value"]} for c in cues]
    except Exception:
        return [{"time": 0.0, "phoneme": "X"}]
