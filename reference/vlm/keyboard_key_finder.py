#!/usr/bin/env python
"""Reference: locate ONE key in a camera frame by asking a vision-language model.

First perception experiment of the project (May 2026). The VLM is prompted
for the pixel centre and a confidence for a single key; backends include a
local Qwen2.5-VL via transformers, Ollama, LM Studio, Gemini and Claude.
Superseded in the final system by OCR anchors + homography (and by the
multi-key VLM backends in ``cybertyping/perception/detect/vlm_anchors.py``),
but kept as a compact, dependency-light reference of the VLM approach.

    python reference/vlm/keyboard_key_finder.py --backend google --letter r
    python reference/vlm/keyboard_key_finder.py --backend claude          # interactive

API keys are read from GEMINI_API_KEY / ANTHROPIC_API_KEY.
"""
import argparse
import base64
import json
import os
import platform
import re
import time
from abc import ABC, abstractmethod
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

# ──── USER EDITABLE ────
CAM_INDEX           = 0
DEFAULT_BACKEND     = "huggingface"
HF_MODEL_ID         = "Qwen/Qwen2.5-VL-3B-Instruct"
OLLAMA_URL          = "http://localhost:11434"
OLLAMA_MODEL        = "llava:7b"
LMSTUDIO_URL        = "http://localhost:1234"
GOOGLE_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_MODEL        = "gemini-2.5-pro"
CLAUDE_API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL        = "claude-opus-4-7"
JPEG_QUALITY        = 85
CONFIDENCE_THRESHOLD = 0.3


# ──── VLM BACKENDS ────

class VLMBackend(ABC):
    @abstractmethod
    def query(self, image_bgr: np.ndarray, prompt: str) -> str: ...

    def _encode_image_base64(self, image_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


class OllamaBackend(VLMBackend):
    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.url   = url.rstrip("/")
        self.model = model

    def query(self, image_bgr: np.ndarray, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "images": [self._encode_image_base64(image_bgr)],
            "stream": False,
        }
        r = requests.post(f"{self.url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "")


class HuggingFaceBackend(VLMBackend):
    def __init__(self, model_id: str = HF_MODEL_ID, device: str | None = None):
        try:
            import torch
        except ImportError:
            raise SystemExit("[hf] torch not found — pip install torch")

        from transformers import AutoProcessor

        # AutoModelForVision2Seq was removed in transformers 5.x; use model-specific classes.
        model_id_lower = model_id.lower()
        if "qwen2.5-vl" in model_id_lower or "qwen2_5_vl" in model_id_lower:
            try:
                import qwen_vl_utils  # noqa: F401
            except ImportError:
                raise SystemExit("[hf] qwen-vl-utils not found — pip install qwen-vl-utils")
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        elif "qwen2-vl" in model_id_lower or "qwen2_vl" in model_id_lower:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
        elif "smolvlm" in model_id_lower:
            from transformers import SmolVLMForConditionalGeneration as ModelClass
        else:
            # Generic fallback: try the old auto class first, then conditional generation
            try:
                from transformers import AutoModelForVision2Seq as ModelClass
            except ImportError:
                from transformers import AutoModelForConditionalGeneration as ModelClass

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[hf] loading {model_id} on {self.device} ...")
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = ModelClass.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[hf] model ready")

    def query(self, image_bgr: np.ndarray, prompt: str) -> str:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[pil_image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128)
        generated = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


class LMStudioBackend(VLMBackend):
    def __init__(self, url: str = LMSTUDIO_URL, model: str = "local-model"):
        self.url   = url.rstrip("/")
        self.model = model

    def query(self, image_bgr: np.ndarray, prompt: str) -> str:
        b64 = self._encode_image_base64(image_bgr)
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": 200,
            "temperature": 0.1,
        }
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class GoogleBackend(VLMBackend):
    def __init__(self, model_name: str = GOOGLE_MODEL, api_key: str | None = None):
        try:
            import google.generativeai as genai
        except ImportError:
            raise SystemExit("[google] google-generativeai not found — pip install google-generativeai")

        key = api_key or GOOGLE_API_KEY or os.environ.get("GEMINI_API_KEY", "") \
            or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise SystemExit("[google] No API key — set GEMINI_API_KEY env var")
        genai.configure(api_key=key)
        self._genai = genai
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        print(f"[google] Using model: {model_name}")

    def query(self, image_bgr: np.ndarray, prompt: str) -> str:
        h, w = image_bgr.shape[:2]
        # Gemini grounds in a normalized 0–1000 space, not raw pixels.
        # Override the coordinate instructions and rescale on return.
        augmented = prompt + (
            f"\n\nIMPORTANT: Override any earlier coordinate-system instructions. "
            f"Return x and y as NORMALIZED integers in the range 0 to 1000, "
            f"where (0,0) is top-left and (1000,1000) is bottom-right. "
            f"Do NOT return raw pixel values. "
            f"For the 'not visible' case, still use x=-1, y=-1."
        )
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        response = self.model.generate_content([augmented, pil_image])
        raw = response.text or ""

        parsed = parse_vlm_response(raw)
        if parsed is None:
            return raw
        nx, ny = parsed["x"], parsed["y"]
        if nx < 0 or ny < 0:
            return raw
        px = int(round(nx * (w - 1) / 1000))
        py = int(round(ny * (h - 1) / 1000))
        return json.dumps({"x": px, "y": py, "confidence": parsed["confidence"]})


class ClaudeBackend(VLMBackend):
    def __init__(self, model: str = CLAUDE_MODEL, api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("[claude] anthropic not found — pip install anthropic")

        key = api_key or CLAUDE_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise SystemExit("[claude] No API key — set ANTHROPIC_API_KEY env var")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        print(f"[claude] Using model: {model}")

    def query(self, image_bgr: np.ndarray, prompt: str) -> str:
        b64 = self._encode_image_base64(image_bgr)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""


# ──── CAMERA ────

def open_camera(cam_index: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"[cam] index {cam_index} did not open")
    w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[cam] {int(w)}x{int(h)} @ {fps:.1f} FPS")
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ret, frame = cap.read()
    return frame if ret else None


# ──── PROMPT ────

def build_prompt(letter: str) -> str:
    return (
        f'You are a vision assistant analyzing a camera image of a keyboard.\n'
        f'Find the key with the letter "{letter.upper()}" on the keyboard.\n'
        f'The image uses standard pixel coordinates: (0,0) is top-left, '
        f'x increases to the right, y increases downward.\n'
        f'Report the CENTER pixel of that key.\n'
        f'Reply ONLY with a valid JSON object — no other text:\n'
        f'{{"x": <integer pixel column>, "y": <integer pixel row>, "confidence": <float 0.0-1.0>}}\n'
        f'If the key "{letter.upper()}" is not visible in the image, reply exactly:\n'
        f'{{"x": -1, "y": -1, "confidence": 0.0}}'
    )


# ──── JSON PARSING ────

def parse_vlm_response(text: str) -> dict | None:
    if not text:
        return None

    # Stage 1: strip markdown code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None

    # Stage 2: first {...} block in raw text
    if candidate is None:
        brace = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text.strip()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Stage 3: field-by-field regex
        x_m = re.search(r'"x"\s*:\s*(-?\d+)', text)
        y_m = re.search(r'"y"\s*:\s*(-?\d+)', text)
        c_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if x_m and y_m:
            return {
                "x": int(x_m.group(1)),
                "y": int(y_m.group(1)),
                "confidence": float(c_m.group(1)) if c_m else 0.5,
            }
        return None

    if not isinstance(data.get("x"), (int, float)) or \
       not isinstance(data.get("y"), (int, float)):
        return None

    return {
        "x": int(data["x"]),
        "y": int(data["y"]),
        "confidence": float(data.get("confidence", 0.5)),
    }


# ──── OVERLAY ────

def draw_detection(
    frame: np.ndarray, x: int, y: int, confidence: float, letter: str
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if x < 0 or y < 0 or x >= w or y >= h:
        cv2.putText(out, f"Key '{letter.upper()}' not found",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out

    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    half = 20
    cv2.circle(out, (x, y), 8, color, -1)
    cv2.circle(out, (x, y), 12, color, 2)
    cv2.line(out, (x - 20, y), (x + 20, y), color, 2)
    cv2.line(out, (x, y - 20), (x, y + 20), color, 2)
    cv2.rectangle(out, (x - half, y - half), (x + half, y + half), color, 2)
    label = f"'{letter.upper()}' ({x},{y}) conf={confidence:.2f}"
    cv2.putText(out, label, (max(x - 40, 0), max(y - 25, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


# ──── PUBLIC API ────

def find_key_pixel(
    letter: str,
    frame: np.ndarray | None = None,
    backend: VLMBackend | None = None,
    cam_index: int = CAM_INDEX,
) -> tuple[int, int] | None:
    """Return (x, y) pixel center of keyboard key `letter`, or None on failure."""
    if backend is None:
        backend = HuggingFaceBackend()

    cap = None
    if frame is None:
        cap = open_camera(cam_index)
        for _ in range(5):     # discard stale frames so AE/AWB settles
            capture_frame(cap)
        frame = capture_frame(cap)
        cap.release()
        if frame is None:
            print("[find_key_pixel] could not capture frame")
            return None

    raw = backend.query(frame, build_prompt(letter))
    result = parse_vlm_response(raw)

    if result is None:
        print(f"[find_key_pixel] parse failed. raw={raw!r}")
        return None

    x, y, conf = result["x"], result["y"], result["confidence"]

    if x < 0 or y < 0:
        print(f"[find_key_pixel] key '{letter}' not found in frame")
        return None

    if conf < CONFIDENCE_THRESHOLD:
        print(f"[find_key_pixel] low confidence {conf:.2f} for key '{letter}'")
        return None

    return (x, y)


# ──── INTERACTIVE DEMO ────

def _build_backend(args) -> VLMBackend:
    if args.backend == "ollama":
        return OllamaBackend(url=args.url or OLLAMA_URL, model=args.model or OLLAMA_MODEL)
    elif args.backend == "huggingface":
        return HuggingFaceBackend(model_id=args.model or HF_MODEL_ID)
    elif args.backend == "google":
        return GoogleBackend(model_name=args.model or GOOGLE_MODEL)
    elif args.backend == "claude":
        return ClaudeBackend(model=args.model or CLAUDE_MODEL)
    else:
        return LMStudioBackend(url=args.url or LMSTUDIO_URL, model=args.model or "local-model")


def main() -> None:
    global CONFIDENCE_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Detect keyboard key pixel positions using a local VLM"
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        choices=["huggingface", "ollama", "lmstudio", "google", "claude"])
    parser.add_argument("--model",  default=None, help="Model name or HF model ID")
    parser.add_argument("--url",    default=None, help="Base URL for ollama or lmstudio")
    parser.add_argument("--cam",    type=int, default=CAM_INDEX)
    parser.add_argument("--letter", default=None,
                        help="Detect this letter once and exit (one-shot mode)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Minimum confidence threshold")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.confidence

    backend = _build_backend(args)
    cap = open_camera(args.cam)

    # One-shot mode
    if args.letter:
        frame = capture_frame(cap)
        cap.release()
        if frame is None:
            raise SystemExit("[main] no frame captured")
        result = find_key_pixel(args.letter, frame=frame, backend=backend)
        print(f"[result] '{args.letter}' -> {result}")
        return

    # Interactive loop
    print("[demo] Press any letter/digit key to detect it. Press ESC or q to quit.")
    last_letter: str | None = None
    last_detection: tuple[int, int, float] | None = None
    detecting = False

    while True:
        frame = capture_frame(cap)
        if frame is None:
            print("[demo] no frame")
            break

        display = frame.copy()
        if last_detection and last_letter:
            display = draw_detection(display, *last_detection, last_letter)

        if detecting:
            cv2.putText(display, "Querying VLM...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("keyboard_key_finder", display)
        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord("q")):
            break

        if 32 <= k <= 126 and not detecting:
            letter = chr(k)
            detecting = True
            print(f"[demo] querying VLM for '{letter}' ...")
            t0 = time.perf_counter()
            raw = backend.query(frame, build_prompt(letter))
            elapsed = time.perf_counter() - t0
            parsed = parse_vlm_response(raw)
            detecting = False

            if parsed and parsed["x"] >= 0:
                x, y, conf = parsed["x"], parsed["y"], parsed["confidence"]
                last_detection = (x, y, conf)
                last_letter = letter
                print(f"[demo] '{letter}' -> ({x}, {y}) conf={conf:.2f}  [{elapsed:.1f}s]")
            else:
                print(f"[demo] '{letter}' not found or parse error  [{elapsed:.1f}s]  raw={raw!r}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
