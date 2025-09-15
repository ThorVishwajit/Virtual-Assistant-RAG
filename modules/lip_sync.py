import subprocess, shutil, json, os
from config import RHUBARB_PATH

def run_rhubarb_on_wav(wav_path: str):
    rh = shutil.which(RHUBARB_PATH) if os.path.isfile(RHUBARB_PATH) == False else RHUBARB_PATH
    if not rh:
        print("⚠️ Rhubarb not found! Returning dummy phoneme.")
        return [{"time": 0.0, "phoneme": "X"}]

    try:
        # Run rhubarb with json output
        proc = subprocess.run(
            [rh, "-f", "json", wav_path],
            capture_output=True, text=True, check=True
        )

        # Parse JSON output
        data = json.loads(proc.stdout.strip())
        cues = data.get("mouthCues", [])

        if not cues:
            print(f"⚠️ No phoneme cues detected in {wav_path}")
            return [{"time": 0.0, "phoneme": "X"}]

        return [{"time": float(c["start"]), "phoneme": c["value"]} for c in cues]

    except Exception as e:
        print(f"❌ Rhubarb failed on {wav_path}: {e}")
        return [{"time": 0.0, "phoneme": "X"}]
