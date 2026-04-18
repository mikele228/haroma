"""
Sensor Adapters -- pull real hardware data into the SensorBuffer.

Two integration patterns coexist:

  PULL (server polls hardware)
    A background thread calls adapter.read() at a fixed interval and
    pushes the result into the SensorBuffer.  Used for devices with
    Python SDKs: USB cameras, microphones, serial-port sensors, etc.

  PUSH (device posts to server)
    The device (or its driver) sends HTTP POST to /sensor on the server.
    Already handled by the Flask endpoint.  No adapter needed -- just
    configure the device to POST JSON to http://<host>:8193/sensor.

Each pull-adapter is a small class with:
    channel: str        -- name like "vision", "audio", "lidar"
    interval: float     -- seconds between polls
    available: bool     -- whether the hardware was detected
    read() -> dict      -- grab one reading; returns {} if nothing new

All adapters degrade gracefully: if the library or device is missing,
available=False and the adapter is silently skipped.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Callable, Dict, List, Optional

# =====================================================================
# Base class
# =====================================================================


class SensorAdapter:
    channel: str = "unknown"
    interval: float = 1.0
    available: bool = False

    def read(self) -> Dict[str, Any]:
        return {}

    def release(self):
        pass


# =====================================================================
# NEURAL PERCEPTION PIPELINE (U12)
# =====================================================================


class _VisionEncoder:
    """CLIP for image embeddings — model variant selected by resource tier.

    Supports ViT-B/32 (lighter, ~150M) and ViT-L/14 (heavier, ~428M).
    Falls back through: open_clip → transformers → unavailable.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.available = False
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.embed_dim = 768
        self._lock = threading.Lock()

        if model_name is None:
            try:
                from engine.ResourceAdaptiveConfig import get_resource_config

                model_name = get_resource_config().sensor.get("vision_model")
            except Exception as _e:
                print(f"[VisionEncoder] resource config error: {_e}", flush=True)

        clip_name = model_name or "openai/clip-vit-large-patch14"
        oc_name = "ViT-L-14" if "large" in clip_name else "ViT-B-32"

        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                oc_name, pretrained="openai"
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(oc_name)
            self.embed_dim = 768 if "L" in oc_name else 512
            self.available = True
        except Exception:
            try:
                from transformers import CLIPModel, CLIPProcessor

                self._model = CLIPModel.from_pretrained(clip_name)
                self._preprocess = CLIPProcessor.from_pretrained(clip_name)
                self._model.eval()
                self.embed_dim = self._model.config.projection_dim
                self.available = True
            except Exception as _e:
                print(f"[VisionEncoder] CLIP fallback error: {_e}", flush=True)

    def encode_image(self, frame) -> Optional[List[float]]:
        if not self.available or self._model is None:
            return None
        with self._lock:
            return self._encode_image_impl(frame)

    def _encode_image_impl(self, frame) -> Optional[List[float]]:
        try:
            import torch
            from PIL import Image
            import numpy as np

            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    img = Image.fromarray(frame, mode="L").convert("RGB")
                elif frame.shape[2] == 1:
                    img = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
                else:
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    img = Image.fromarray(frame[:, :, ::-1])
            else:
                img = frame
            if hasattr(self._model, "encode_image"):
                image_input = self._preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    features = self._model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)
                return features.squeeze(0).cpu().tolist()
            else:
                inputs = self._preprocess(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._model.get_image_features(**inputs)
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                return outputs.squeeze(0).cpu().tolist()
        except Exception:
            return None

    def classify_scene(self, frame, labels: List[str]) -> Optional[Dict[str, float]]:
        if not self.available or self._model is None or not labels:
            return None
        with self._lock:
            return self._classify_scene_impl(frame, labels)

    def _classify_scene_impl(self, frame, labels: List[str]) -> Optional[Dict[str, float]]:
        try:
            import torch
            from PIL import Image
            import numpy as np

            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    img = Image.fromarray(frame, mode="L").convert("RGB")
                elif frame.shape[2] == 1:
                    img = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
                else:
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    img = Image.fromarray(frame[:, :, ::-1])
            else:
                img = frame
            if hasattr(self._model, "encode_image") and self._tokenizer:
                image_input = self._preprocess(img).unsqueeze(0)
                text_tokens = self._tokenizer(labels)
                with torch.no_grad():
                    img_feat = self._model.encode_image(image_input)
                    txt_feat = self._model.encode_text(text_tokens)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                    sim = (img_feat @ txt_feat.T).squeeze(0).softmax(dim=0)
                return {l: round(float(s), 4) for l, s in zip(labels, sim)}
            else:
                inputs = self._preprocess(
                    images=img, text=labels, return_tensors="pt", padding=True
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits_per_image.squeeze(0).softmax(dim=0)
                return {l: round(float(s), 4) for l, s in zip(labels, logits)}
        except Exception:
            return None


class _AudioTranscriber:
    """Whisper-small for speech-to-text transcription."""

    def __init__(self):
        self.available = False
        self._model = None
        self._processor = None
        self._is_hf = False
        self._lock = threading.Lock()
        try:
            import whisper

            self._model = whisper.load_model("small")
            self.available = True
        except Exception:
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration

                self._processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-small"
                )
                self._model.eval()
                self._is_hf = True
                self.available = True
            except Exception:
                self._is_hf = False

    def transcribe(self, audio_array, sample_rate: int = 16000) -> Optional[Dict[str, Any]]:
        if not self.available or self._model is None:
            return None
        with self._lock:
            return self._transcribe_impl(audio_array, sample_rate)

    def _transcribe_impl(self, audio_array, sample_rate: int) -> Optional[Dict[str, Any]]:
        try:
            import numpy as np

            audio_np = np.array(audio_array, dtype=np.float32).flatten()
            if hasattr(self._model, "transcribe"):
                result = self._model.transcribe(audio_np, fp16=False)
                return {
                    "text": result.get("text", "").strip(),
                    "language": result.get("language", "unknown"),
                    "segments": len(result.get("segments", [])),
                }
            else:
                import torch

                inputs = self._processor(audio_np, sampling_rate=sample_rate, return_tensors="pt")
                with torch.no_grad():
                    predicted_ids = self._model.generate(inputs.input_features, max_new_tokens=128)
                text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return {"text": text.strip(), "language": "unknown", "segments": 1}
        except Exception:
            return None


class _AudioEventClassifier:
    """Classifies environmental audio events using a pretrained model."""

    def __init__(self):
        self.available = False
        self._pipeline = None
        self._lock = threading.Lock()
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", top_k=5
            )
            self.available = True
        except Exception as _e:
            print(f"[AudioEventClassifier] pipeline init error: {_e}", flush=True)

    def classify(self, audio_array, sample_rate: int = 16000) -> Optional[List[Dict[str, Any]]]:
        if not self.available or self._pipeline is None:
            return None
        with self._lock:
            return self._classify_impl(audio_array, sample_rate)

    def _classify_impl(self, audio_array, sample_rate: int) -> Optional[List[Dict[str, Any]]]:
        try:
            import numpy as np

            audio_np = np.array(audio_array, dtype=np.float32).flatten()
            results = self._pipeline({"array": audio_np, "sampling_rate": sample_rate})
            return [{"label": r["label"], "score": round(r["score"], 4)} for r in results[:5]]
        except Exception:
            return None


_vision_encoder: Optional[_VisionEncoder] = None
_audio_transcriber: Optional[_AudioTranscriber] = None
_audio_classifier: Optional[_AudioEventClassifier] = None
_vision_init_lock = threading.Lock()
_audio_neural_init_lock = threading.Lock()
_vision_initialized = False
_audio_neural_initialized = False
_warmup_started = False

# Set by mind.elarion_server_v2 after BootAgent.shared exists so heavy audio models
# (Whisper / AST) do not load during an active HTTP /chat turn.
_http_chat_busy_fn: Optional[Callable[[], bool]] = None


def register_http_chat_busy_checker(fn: Optional[Callable[[], bool]]) -> None:
    """When set, :func:`_ensure_audio_neural_models` skips loading while the callback is true."""
    global _http_chat_busy_fn
    _http_chat_busy_fn = fn


def _http_chat_busy() -> bool:
    if _http_chat_busy_fn is None:
        return False
    try:
        return bool(_http_chat_busy_fn())
    except Exception:
        return False


def warmup_neural_models():
    """Load the vision encoder in a background thread. Audio models load lazily."""
    global _warmup_started
    if _warmup_started or _vision_initialized:
        return
    _warmup_started = True
    t = threading.Thread(target=_ensure_vision_models, daemon=True)
    t.start()


def _ensure_vision_models() -> None:
    """Lazy-init CLIP / vision encoder only (does not load Whisper)."""
    global _vision_encoder, _vision_initialized
    if _vision_initialized:
        return
    with _vision_init_lock:
        if _vision_initialized:
            return
        _vision_encoder = _VisionEncoder()
        _vision_initialized = True


def _ensure_audio_neural_models() -> None:
    """Lazy-init Whisper + audio event classifier. Deferred while HTTP /chat is in flight."""
    global _audio_transcriber, _audio_classifier, _audio_neural_initialized
    if _audio_neural_initialized:
        return
    if _http_chat_busy():
        return
    with _audio_neural_init_lock:
        if _audio_neural_initialized:
            return
        if _http_chat_busy():
            return
        _audio_transcriber = _AudioTranscriber()
        _audio_classifier = _AudioEventClassifier()
        _audio_neural_initialized = True


def _ensure_neural_models() -> None:
    """Full perception stack (vision + audio). Prefer :func:`_ensure_vision_models` / :func:`_ensure_audio_neural_models` when splitting work."""
    _ensure_vision_models()
    _ensure_audio_neural_models()


def get_neural_perception_stats() -> Dict[str, Any]:
    _ensure_vision_models()
    return {
        "vision_encoder": _vision_encoder.available if _vision_encoder else False,
        "audio_transcriber": _audio_transcriber.available if _audio_transcriber else False,
        "audio_classifier": _audio_classifier.available if _audio_classifier else False,
    }


# =====================================================================
# VISION -- USB / built-in camera via OpenCV + CLIP (U12)
# =====================================================================


class VisionAdapter(SensorAdapter):
    """Grabs frames from a webcam or USB camera.

    Hardware: any UVC-compatible camera, Raspberry Pi camera, etc.
    Library:  pip install opencv-python
    """

    channel = "vision"
    interval = 0.5

    def __init__(self, device_index: int = 0):
        self._cap = None
        try:
            import cv2

            self._cv2 = cv2
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                self._cap = cap
                self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available or self._cap is None:
            return {}
        ret, frame = self._cap.read()
        if not ret:
            return {}
        h, w = frame.shape[:2]
        # Downsample to small region stats (not raw pixels)
        import numpy as np

        small = self._cv2.resize(frame, (8, 8))
        brightness = float(np.mean(small))
        color_avg = [float(x) for x in np.mean(small, axis=(0, 1))]
        # Edge density as a proxy for visual complexity
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        edges = self._cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges) / 255.0)
        result = {
            "resolution": [w, h],
            "brightness": round(brightness, 2),
            "color_bgr_avg": [round(c, 2) for c in color_avg],
            "edge_density": round(edge_density, 4),
            "has_motion": False,
            "timestamp": time.time(),
        }
        _ensure_vision_models()
        if _vision_encoder and _vision_encoder.available:
            embedding = _vision_encoder.encode_image(frame)
            if embedding is not None:
                result["clip_embedding"] = embedding
            scene_labels = [
                "a person",
                "an empty room",
                "outdoors nature",
                "a computer screen",
                "text or writing",
                "an animal",
                "food",
                "a vehicle",
                "darkness",
                "bright light",
            ]
            scene = _vision_encoder.classify_scene(frame, scene_labels)
            if scene:
                result["scene_classification"] = scene
        return result

    def release(self):
        if self._cap:
            self._cap.release()


# =====================================================================
# AUDIO / HEARING -- microphone via sounddevice or pyaudio
# =====================================================================


class AudioAdapter(SensorAdapter):
    """Captures audio level and basic spectral features from a microphone.

    Hardware: any system microphone, USB mic, I2S MEMS mic
    Library:  pip install sounddevice numpy
    """

    channel = "audio"
    interval = 0.3

    def __init__(self):
        try:
            import sounddevice as sd
            import numpy as np

            self._sd = sd
            self._np = np
            devs = sd.query_devices()
            self._sample_rate = 16000
            self._block_size = 4096
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available:
            return {}
        try:
            audio = self._sd.rec(
                self._block_size,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            )
            rms = float(self._np.sqrt(self._np.mean(audio**2)))
            peak = float(self._np.max(self._np.abs(audio)))
            fft = self._np.abs(self._np.fft.rfft(audio[:, 0]))
            spectral_centroid = float(
                self._np.sum(self._np.arange(len(fft)) * fft) / (self._np.sum(fft) + 1e-10)
            )
            result = {
                "rms_level": round(rms, 5),
                "peak_level": round(peak, 5),
                "spectral_centroid": round(spectral_centroid, 2),
                "is_speech_likely": rms > 0.01 and spectral_centroid > 50,
                "timestamp": time.time(),
            }
            if rms > 0.005:
                _ensure_audio_neural_models()
                if (
                    _audio_transcriber
                    and _audio_transcriber.available
                    and result["is_speech_likely"]
                ):
                    transcription = _audio_transcriber.transcribe(audio, self._sample_rate)
                    if transcription and transcription.get("text"):
                        result["transcription"] = transcription
                if _audio_classifier and _audio_classifier.available:
                    events = _audio_classifier.classify(audio, self._sample_rate)
                    if events:
                        result["audio_events"] = events
            return result
        except Exception:
            return {}


# =====================================================================
# TOUCH -- serial pressure / capacitive sensors
# =====================================================================


class TouchAdapter(SensorAdapter):
    """Reads pressure or capacitive touch from a serial device (Arduino, etc.).

    Hardware: FSR (force-sensitive resistor), capacitive touch breakout
    Protocol: serial line sending JSON or CSV values
    Library:  pip install pyserial
    """

    channel = "touch"
    interval = 0.2

    def __init__(self, port: str = "COM3", baud: int = 9600):
        self._ser = None
        try:
            import serial

            self._ser = serial.Serial(port, baud, timeout=0.1)
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available or self._ser is None:
            return {}
        try:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return {}
            import json as _json

            try:
                data = _json.loads(line)
            except ValueError:
                vals = [float(x) for x in line.split(",") if x.strip()]
                data = {"pressure": vals[0] if vals else 0.0}
            data["timestamp"] = time.time()
            return data
        except Exception:
            return {}

    def release(self):
        if self._ser:
            self._ser.close()


# =====================================================================
# SMELL -- gas / VOC sensors via serial or I2C
# =====================================================================


class SmellAdapter(SensorAdapter):
    """Reads gas / VOC levels from sensors like BME680, MQ-series, SGP30.

    Hardware: BME680 (gas + humidity + temp), MQ-135 (air quality)
    Protocol: I2C via smbus2 or serial via Arduino bridge
    Library:  pip install smbus2  (for direct I2C on Raspberry Pi)
              or serial bridge (same as TouchAdapter)
    """

    channel = "smell"
    interval = 2.0

    def __init__(self, port: str = "COM4", baud: int = 9600, i2c_addr: int = 0x76):
        self._source = None
        self._type = None
        # Try I2C first (Raspberry Pi / Linux)
        try:
            import smbus2

            bus = smbus2.SMBus(1)
            bus.read_byte(i2c_addr)
            self._source = bus
            self._i2c_addr = i2c_addr
            self._type = "i2c"
            self.available = True
            return
        except Exception as _e:
            print(f"[SmellAdapter] I2C probe error: {_e}", flush=True)
        # Fall back to serial
        try:
            import serial

            self._source = serial.Serial(port, baud, timeout=0.5)
            self._type = "serial"
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available:
            return {}
        try:
            if self._type == "serial":
                line = self._source.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    return {}
                import json as _json

                try:
                    return _json.loads(line)
                except ValueError:
                    vals = [float(x) for x in line.split(",") if x.strip()]
                    return {
                        "gas_resistance": vals[0] if vals else 0,
                        "voc_index": vals[1] if len(vals) > 1 else 0,
                        "timestamp": time.time(),
                    }
            return {}
        except Exception:
            return {}

    def release(self):
        if self._source and self._type == "serial":
            self._source.close()


# =====================================================================
# TASTE -- pH / conductivity sensors (liquid analysis)
# =====================================================================


class TasteAdapter(SensorAdapter):
    """Reads pH and conductivity from liquid sensors.

    Hardware: Atlas Scientific pH probe, TDS (total dissolved solids) probe
    Protocol: serial / UART from the sensor board
    Library:  pip install pyserial
    """

    channel = "taste"
    interval = 5.0

    def __init__(self, port: str = "COM5", baud: int = 9600):
        self._ser = None
        try:
            import serial

            self._ser = serial.Serial(port, baud, timeout=1.0)
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available or self._ser is None:
            return {}
        try:
            self._ser.write(b"R\r")
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return {}
            return {"ph": float(line), "timestamp": time.time()}
        except Exception:
            return {}

    def release(self):
        if self._ser:
            self._ser.close()


# =====================================================================
# LIDAR -- depth / distance scanning
# =====================================================================


class LidarAdapter(SensorAdapter):
    """Reads distance data from a LiDAR unit.

    Hardware examples:
      - RPLiDAR A1/A2/S1 (USB, 2D scan, ~$100-300)
      - TFmini / TFmini Plus (serial, single-point, ~$30)
      - Intel RealSense L515 (USB, 3D depth, ~$350)
      - Livox Mid-40 (USB/Ethernet, 3D, ~$600)

    Libraries:
      - rplidar-roboticia   (pip install rplidar-roboticia)
      - pyrealsense2        (pip install pyrealsense2)
      - serial for TFmini
    """

    channel = "lidar"
    interval = 0.2

    def __init__(self, port: str = "COM6", variant: str = "auto"):
        self._source = None
        self._variant = None

        # Try RPLiDAR
        if variant in ("auto", "rplidar"):
            try:
                from rplidar import RPLidar

                lidar = RPLidar(port)
                lidar.get_info()
                self._source = lidar
                self._variant = "rplidar"
                self._scan_iter = lidar.iter_scans()
                self.available = True
                return
            except Exception as _e:
                print(f"[LidarAdapter] RPLiDAR probe error: {_e}", flush=True)

        # Try Intel RealSense
        if variant in ("auto", "realsense"):
            try:
                import pyrealsense2 as rs

                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pipe.start(cfg)
                self._source = pipe
                self._variant = "realsense"
                self.available = True
                return
            except Exception as _e:
                print(f"[LidarAdapter] RealSense probe error: {_e}", flush=True)

        # Try TFmini (serial single-point)
        if variant in ("auto", "tfmini"):
            try:
                import serial

                ser = serial.Serial(port, 115200, timeout=0.1)
                self._source = ser
                self._variant = "tfmini"
                self.available = True
                return
            except Exception as _e:
                print(f"[LidarAdapter] TFmini probe error: {_e}", flush=True)

        self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available:
            return {}
        try:
            if self._variant == "rplidar":
                scan = next(self._scan_iter)
                points = [(angle, dist) for _, angle, dist in scan if dist > 0]
                distances = [d for _, d in points]
                return {
                    "point_count": len(points),
                    "min_distance_mm": min(distances) if distances else 0,
                    "max_distance_mm": max(distances) if distances else 0,
                    "mean_distance_mm": round(sum(distances) / len(distances), 1)
                    if distances
                    else 0,
                    "scan_degrees": 360,
                    "timestamp": time.time(),
                }

            elif self._variant == "realsense":
                import numpy as np

                frames = self._source.wait_for_frames(timeout_ms=500)
                depth = frames.get_depth_frame()
                if not depth:
                    return {}
                data = np.asanyarray(depth.get_data())
                valid = data[data > 0]
                return {
                    "resolution": [depth.get_width(), depth.get_height()],
                    "min_depth_mm": int(valid.min()) if valid.size else 0,
                    "max_depth_mm": int(valid.max()) if valid.size else 0,
                    "mean_depth_mm": round(float(valid.mean()), 1) if valid.size else 0,
                    "coverage_pct": round(float(valid.size / data.size * 100), 1),
                    "timestamp": time.time(),
                }

            elif self._variant == "tfmini":
                raw = self._source.read(9)
                if len(raw) == 9 and raw[0] == 0x59 and raw[1] == 0x59:
                    dist = raw[2] + raw[3] * 256
                    strength = raw[4] + raw[5] * 256
                    return {
                        "distance_cm": dist,
                        "signal_strength": strength,
                        "timestamp": time.time(),
                    }
        except Exception as _e:
            print(f"[LidarAdapter] read error: {_e}", flush=True)
        return {}

    def release(self):
        if self._source:
            try:
                if self._variant == "rplidar":
                    self._source.stop()
                    self._source.disconnect()
                elif self._variant == "realsense":
                    self._source.stop()
                elif self._variant == "tfmini":
                    self._source.close()
            except Exception as _e:
                print(f"[LidarAdapter] release error: {_e}", flush=True)


# =====================================================================
# INFRARED / THERMAL -- thermal camera or IR array
# =====================================================================


class InfraredAdapter(SensorAdapter):
    """Reads thermal / IR data.

    Hardware examples:
      - FLIR Lepton (SPI/I2C, 80x60 thermal, ~$200 with breakout)
      - MLX90640 (I2C 32x24 thermal array, ~$60)
      - AMG8833 Grid-EYE (I2C 8x8 thermal, ~$40)
      - Generic IR temperature sensor (serial)

    Libraries:
      - adafruit-circuitpython-amg88xx  (pip install)
      - adafruit-circuitpython-mlx90640
      - flirpy  (pip install flirpy)
    """

    channel = "infrared"
    interval = 1.0

    def __init__(self, variant: str = "auto"):
        self._source = None
        self._variant = None

        # Try MLX90640 (32x24 thermal array via I2C)
        if variant in ("auto", "mlx90640"):
            try:
                import adafruit_mlx90640
                import board, busio

                i2c = busio.I2C(board.SCL, board.SDA)
                mlx = adafruit_mlx90640.MLX90640(i2c)
                mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
                self._source = mlx
                self._variant = "mlx90640"
                self._frame = [0.0] * 768
                self.available = True
                return
            except Exception as _e:
                print(f"[InfraredAdapter] MLX90640 probe error: {_e}", flush=True)

        # Try AMG8833 (8x8 thermal array)
        if variant in ("auto", "amg8833"):
            try:
                import adafruit_amg88xx
                import board, busio

                i2c = busio.I2C(board.SCL, board.SDA)
                self._source = adafruit_amg88xx.AMG88XX(i2c)
                self._variant = "amg8833"
                self.available = True
                return
            except Exception as _e:
                print(f"[InfraredAdapter] AMG8833 probe error: {_e}", flush=True)

        # Try FLIR via flirpy
        if variant in ("auto", "flir"):
            try:
                from flirpy.camera.lepton import Lepton

                cam = Lepton()
                self._source = cam
                self._variant = "flir"
                self.available = True
                return
            except Exception as _e:
                print(f"[InfraredAdapter] FLIR probe error: {_e}", flush=True)

        self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available:
            return {}
        try:
            if self._variant == "mlx90640":
                self._source.getFrame(self._frame)
                temps = self._frame
                return {
                    "resolution": [32, 24],
                    "min_temp_c": round(min(temps), 1),
                    "max_temp_c": round(max(temps), 1),
                    "mean_temp_c": round(sum(temps) / len(temps), 1),
                    "hot_spot_detected": max(temps) > 35.0,
                    "timestamp": time.time(),
                }

            elif self._variant == "amg8833":
                pixels = self._source.pixels
                flat = [p for row in pixels for p in row]
                return {
                    "resolution": [8, 8],
                    "min_temp_c": round(min(flat), 1),
                    "max_temp_c": round(max(flat), 1),
                    "mean_temp_c": round(sum(flat) / len(flat), 1),
                    "hot_spot_detected": max(flat) > 35.0,
                    "timestamp": time.time(),
                }

            elif self._variant == "flir":
                import numpy as np

                frame = self._source.grab()
                if frame is None:
                    return {}
                return {
                    "resolution": list(frame.shape[:2]),
                    "min_temp_c": round(float(frame.min()), 1),
                    "max_temp_c": round(float(frame.max()), 1),
                    "mean_temp_c": round(float(frame.mean()), 1),
                    "hot_spot_detected": float(frame.max()) > 35.0,
                    "timestamp": time.time(),
                }
        except Exception as _e:
            print(f"[InfraredAdapter] read error: {_e}", flush=True)
        return {}

    def release(self):
        if self._source and self._variant == "flir":
            try:
                self._source.close()
            except Exception as _e:
                print(f"[InfraredAdapter] release error: {_e}", flush=True)


# =====================================================================
# IMU / PROPRIOCEPTION -- accelerometer, gyroscope, magnetometer
# =====================================================================


class ImuAdapter(SensorAdapter):
    """Reads body orientation and motion from an IMU.

    Hardware: MPU6050, BNO055, ICM-20948 (I2C or serial)
    Library:  pip install adafruit-circuitpython-bno055
              or serial bridge from Arduino
    """

    channel = "proprioception"
    interval = 0.1

    def __init__(self, port: str = "COM7", baud: int = 115200):
        self._source = None
        self._variant = None
        # Try BNO055 via I2C
        try:
            import adafruit_bno055
            import board

            i2c = board.I2C()
            self._source = adafruit_bno055.BNO055_I2C(i2c)
            self._variant = "bno055"
            self.available = True
            return
        except Exception as _e:
            print(f"[ImuAdapter] BNO055 probe error: {_e}", flush=True)
        # Try serial bridge
        try:
            import serial

            self._source = serial.Serial(port, baud, timeout=0.05)
            self._variant = "serial"
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available:
            return {}
        try:
            if self._variant == "bno055":
                euler = self._source.euler
                accel = self._source.acceleration
                gyro = self._source.gyro
                return {
                    "euler_deg": list(euler) if euler[0] is not None else [0, 0, 0],
                    "accel_ms2": list(accel) if accel[0] is not None else [0, 0, 0],
                    "gyro_rads": list(gyro) if gyro[0] is not None else [0, 0, 0],
                    "timestamp": time.time(),
                }
            elif self._variant == "serial":
                import json as _json

                line = self._source.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    return _json.loads(line)
        except Exception as _e:
            print(f"[ImuAdapter] read error: {_e}", flush=True)
        return {}

    def release(self):
        if self._source and self._variant == "serial":
            self._source.close()


# =====================================================================
# GPS / SPATIAL -- location awareness
# =====================================================================


class GpsAdapter(SensorAdapter):
    """Reads GPS coordinates from a GNSS module.

    Hardware: u-blox NEO-6M/7M/8M, Adafruit Ultimate GPS (serial/USB)
    Library:  pip install pynmea2 pyserial
    """

    channel = "gps"
    interval = 2.0

    def __init__(self, port: str = "COM8", baud: int = 9600):
        self._ser = None
        try:
            import serial
            import pynmea2

            self._pynmea2 = pynmea2
            self._ser = serial.Serial(port, baud, timeout=1.0)
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> Dict[str, Any]:
        if not self.available or self._ser is None:
            return {}
        try:
            line = self._ser.readline().decode("ascii", errors="ignore")
            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                msg = self._pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    return {
                        "latitude": round(msg.latitude, 6),
                        "longitude": round(msg.longitude, 6),
                        "altitude_m": round(msg.altitude, 1) if msg.altitude else 0,
                        "satellites": int(msg.num_sats) if msg.num_sats else 0,
                        "timestamp": time.time(),
                    }
        except Exception as _e:
            print(f"[GpsAdapter] read error: {_e}", flush=True)
        return {}

    def release(self):
        if self._ser:
            self._ser.close()


# =====================================================================
# Sensor Poller -- background thread that pulls from all adapters
# =====================================================================


class SensorPoller:
    """Runs a background thread per adapter, pushing readings into a
    SensorBuffer at each adapter's own interval."""

    def __init__(self, buffer, adapters: Optional[List[SensorAdapter]] = None):
        self._buffer = buffer
        self._adapters: List[SensorAdapter] = adapters or []
        self._threads: List[threading.Thread] = []
        self._running = False

    def add(self, adapter: SensorAdapter):
        self._adapters.append(adapter)

    @property
    def active_adapters(self) -> List[SensorAdapter]:
        return [a for a in self._adapters if a.available]

    def start(self):
        self._running = True
        for adapter in self._adapters:
            if adapter.available:
                t = threading.Thread(
                    target=self._poll_loop,
                    args=(adapter,),
                    daemon=True,
                    name=f"Sensor-{adapter.channel}",
                )
                t.start()
                self._threads.append(t)
                print(
                    f"[Sensors] {adapter.channel} adapter: ONLINE (poll every {adapter.interval}s)",
                    flush=True,
                )
            else:
                print(f"[Sensors] {adapter.channel} adapter: not available", flush=True)

    def stop(self):
        self._running = False
        for adapter in self._adapters:
            adapter.release()

    def _poll_loop(self, adapter: SensorAdapter):
        while self._running:
            try:
                reading = adapter.read()
                if reading:
                    self._buffer.push_sensor(adapter.channel, reading)
            except Exception as _e:
                print(f"[SensorPoller] poll error: {_e}", flush=True)
            time.sleep(adapter.interval)

    def stats(self) -> dict:
        return {
            "adapters": {
                a.channel: {
                    "available": a.available,
                    "interval": a.interval,
                }
                for a in self._adapters
            },
            "active_count": len(self.active_adapters),
        }
