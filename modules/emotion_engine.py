import os, json
from transformers import pipeline
from config import MODEL_DIR

EMO_LABELS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

VA_MAP = {
    'joy': (0.9,0.5),'love': (0.9,0.4),'admiration':(0.7,0.3),'amusement':(0.7,0.6),
    'excitement':(0.8,0.8),'optimism':(0.7,0.4),'gratitude':(0.8,0.3),'pride':(0.7,0.5),
    'relief':(0.6,0.2),'surprise':(0.2,0.9),'neutral':(0.0,0.0),'confusion':(-0.2,0.5),
    'nervousness':(-0.4,0.7),'fear':(-0.8,0.8),'anger':(-0.8,0.7),'sadness':(-0.8,0.4),
    'disgust':(-0.9,0.6)
}

class EmotionEngine:
    def __init__(self):
        self.classifier = pipeline("text-classification", model=MODEL_DIR, top_k=None, device=-1)
        self.ema = {k: 0.0 for k in EMO_LABELS}
        self.alpha = 0.35

    def analyze(self, text: str):
        raw = self.classifier(text)[0]
        inst = {d["label"]: float(d["score"]) for d in raw}
        for k in EMO_LABELS:
            s = inst.get(k, 0.0)
            self.ema[k] = self.alpha * s + (1 - self.alpha) * self.ema[k]
        primary = max(self.ema.items(), key=lambda t: t[1])[0]
        return self._plan(primary)

    def _plan(self, primary: str):
        v_sum = a_sum = w_sum = 0.0
        for k, w in self.ema.items():
            v, a = VA_MAP.get(k, (0.0,0.0))
            v_sum += v * w; a_sum += a * w; w_sum += w
        valence = v_sum / w_sum if w_sum else 0.0
        arousal = a_sum / w_sum if w_sum else 0.0
        intensity = min(1.0, w_sum)
        return {
            "primary": primary,
            "valence": valence,
            "arousal": arousal,
            "intensity": intensity,
            "top_emotions": sorted(
                [{"label": k, "score": float(v)} for k,v in self.ema.items()],
                key=lambda x: x["score"], reverse=True)[:3],
            "face_blendshapes": {},
            "gesture": None,
            "fade_in": 0.12,
            "hold_min": 0.35,
            "fade_out": 0.18
        }
