
try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = ImageChops = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except (ImportError, OSError):
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None


class SenseTransform:
    def __init__(self):
        self.device = "cpu"
        self.object_model = None
        self.face_analyzer = None
        self.midas = None
        self.transform = None

        if torch:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if YOLO:
            try:
                self.object_model = YOLO("yolov8n.pt")
            except Exception as _e:
                print(f"[SenseTransform] YOLO init error: {_e}", flush=True)
        if FaceAnalysis:
            try:
                self.face_analyzer = FaceAnalysis(
                    name="buffalo_l", providers=["CPUExecutionProvider"]
                )
                self.face_analyzer.prepare(ctx_id=-1)
            except Exception as _e:
                print(f"[SenseTransform] FaceAnalysis init error: {_e}", flush=True)
        if torch:
            try:
                self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(self.device).eval()
                self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
            except Exception as _e:
                print(f"[SenseTransform] MiDaS init error: {_e}", flush=True)

    # ---------- VISION ----------
    def image_tag_all(self, img_rgb: np.ndarray):
        if np is None:
            return {
                "people": [],
                "objects": [],
                "features": [],
                "distance_map": [],
                "confidence": 0.0,
            }
        results = {"people": [], "objects": [], "features": [], "distance_map": []}

        if Image is not None and cv2 is not None:
            img_pil = Image.fromarray(img_rgb.astype(np.uint8))
            img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            img_pil, img_cv = None, None

        if self.face_analyzer is not None and img_cv is not None:
            faces = self.face_analyzer.get(img_cv)
            for face in faces:
                results["people"].append(
                    {
                        "box": face.bbox.tolist(),
                        "gender": face.sex,
                        "age": face.age,
                        "embedding": face.embedding.tolist()[:5],
                    }
                )

        if self.object_model is not None and img_cv is not None:
            detect_results = self.object_model(img_cv)[0]
            for det in detect_results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                results["objects"].append(
                    {
                        "label": self.object_model.names[int(cls)],
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 2),
                    }
                )

        if (
            self.midas is not None
            and self.transform is not None
            and img_pil is not None
            and torch is not None
        ):
            img_midas = self.transform(img_pil).to(self.device)
            with torch.no_grad():
                depth = self.midas(img_midas.unsqueeze(0)).squeeze().cpu().numpy()
            if cv2 is not None:
                results["distance_map"] = cv2.resize(
                    depth, (img_rgb.shape[1], img_rgb.shape[0])
                ).tolist()

        results["features"] += ["bright" if np.mean(img_rgb) > 127 else "dim"]
        if cv2 is not None and img_cv is not None:
            results["features"].append("blurred" if self._detect_blur(img_cv) else "sharp")
        results["confidence"] = (
            round(
                np.mean(
                    [x["confidence"] for x in results["objects"]] + [1.0] * len(results["people"])
                ),
                2,
            )
            if results["objects"] or results["people"]
            else 0.0
        )
        return results

    def _detect_blur(self, image_cv):
        return cv2.Laplacian(cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 100

    # ---------- SOUND ----------
    def sound_tag_all(self, waveform: np.ndarray, sample_rate: int = 44100):
        if librosa is None or np is None:
            return {
                "duration_sec": 0.0,
                "rms": 0.0,
                "mean_pitch": 0.0,
                "tags": [],
                "confidence": 0.0,
            }
        y = waveform.astype(np.float32)
        sr = sample_rate
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).mean()
        pitch = librosa.piptrack(y=y, sr=sr)[0]
        mean_pitch = pitch[pitch > 0].mean() if np.any(pitch > 0) else 0.0
        conf = sum([rms > 0.01, mean_pitch > 80, duration > 0.2]) / 3.0
        return {
            "duration_sec": round(duration, 2),
            "volume_level": "loud" if rms > 0.05 else "quiet",
            "mean_pitch_hz": round(float(mean_pitch), 2),
            "sample_rate": sr,
            "confidence": round(conf, 2),
        }

    # ---------- SMELL ----------
    def smell_tag_all(self, arr: np.ndarray):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return {
                "smell_profile": "unknown",
                "intensity": 0.0,
                "familiarity": "unknown",
                "confidence": 0.0,
            }
        patterns = {
            "smoke": np.array([0.4, 0.5, 0.1, 0.9, 0.3, 0.2, 0.5, 0.1]),
            "coffee": np.array([0.2, 0.6, 0.1, 0.5, 0.4, 0.3, 0.6, 0.2]),
        }
        best, score = "unknown", 0.0
        for name, ref in patterns.items():
            if len(arr) == len(ref):
                sim = 1 - np.linalg.norm(arr - ref) / np.linalg.norm(ref)
                if sim > score:
                    best, score = name, sim
        intensity = float(np.mean(arr))
        return {
            "smell_profile": best,
            "intensity": round(intensity, 2),
            "familiarity": "known" if best != "unknown" else "unknown",
            "confidence": round(min(score * 0.7 + intensity * 0.3, 1.0), 2),
        }

    # ---------- TASTE ----------
    def taste_tag_all(self, arr: np.ndarray):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return {
                "taste_profile": "unknown",
                "complexity": 0,
                "familiarity": "unknown",
                "confidence": 0.0,
            }

        profiles = {
            "sweet": np.array([0.2, 0.3, 0.5, 0.6]),
            "bitter": np.array([0.8, 0.6, 0.1, 0.1]),
        }

        best, score = "unknown", 0.0
        for name, ref in profiles.items():
            if len(arr) >= len(ref):
                sim = 1 - np.linalg.norm(arr[: len(ref)] - ref) / np.linalg.norm(ref)
                if sim > score:
                    best, score = name, sim

        complexity = len(set(np.round(arr * 100)))
        return {
            "taste_profile": best,
            "complexity": complexity,
            "familiarity": "known" if best != "unknown" else "unknown",
            "confidence": round(min(score * 0.7 + (complexity / 10.0) * 0.3, 1.0), 2),
        }

    # ---------- TOUCH ----------
    def touch_tag_all(self, d: dict):
        if not d or not isinstance(d, dict):
            return {
                "pressure_level": "unknown",
                "temperature": "unknown",
                "texture": "unknown",
                "confidence": 0.0,
            }

        return {
            "pressure_level": "firm" if d.get("pressure", 0) > 0.7 else "light",
            "temperature": f"{round(d.get('temperature', 0), 1)} °C",
            "texture": "rough" if d.get("vibration", 0) > 0.5 else "smooth",
            "confidence": round(
                sum(1 for k in ["pressure", "temperature", "vibration"] if d.get(k) is not None)
                / 3.0,
                2,
            ),
        }
