"""VLM backends for anchor-key detection.

Each backend returns {label: (x1,y1,x2,y2)} for the requested anchors.
GeminiBackend / ClaudeBackend hit APIs; QwenBackend / MolmoBackend run
locally. get_default_backend() picks one based on which env vars and
optional deps are present. Lazy imports so installing one backend does
not require installing the others.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Iterable

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional


# Prompt, short, schema-strict. Returning fewer keys reliably is better
# than returning all keys flakily.

PROMPT_TEMPLATE = (
    "You are looking at a photo of a US-layout computer keyboard. "
    "Detect the following keys and return their bounding boxes in image-pixel coordinates "
    "(origin top-left, +x right, +y down):\n"
    "{labels}\n"
    "Return STRICT JSON with this exact schema and nothing else:\n"
    '{{"keys": {{"<label>": [x1, y1, x2, y2], ...}}}}\n'
    "Rules:\n"
    "- Use the exact lowercase label strings from the list above.\n"
    "- Each bounding box must tightly enclose the key cap (not just the printed letter).\n"
    "- If a key is occluded or not visible, OMIT it (do not guess).\n"
    "- Do not output any text other than the JSON object."
)


def _parse_strict_json(text: str) -> dict:
    """Tolerant JSON extraction: VLMs occasionally wrap output in ```json blocks."""
    text = text.strip()
    # strip code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back: find the outermost {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in VLM response: {text[:200]!r}")
    return json.loads(m.group(0))


def _validate_bboxes(raw: dict, requested: Iterable[str],
                     frame_w: int, frame_h: int) -> dict[str, tuple[int, int, int, int]]:
    """Normalise, clip, and sanity-check the VLM output."""
    out: dict[str, tuple[int, int, int, int]] = {}
    keys_dict = raw.get("keys", raw)  # accept both wrapped and bare forms
    requested = set(requested)
    for label, bbox in keys_dict.items():
        if label not in requested:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # clip to frame
        x1 = max(0, min(frame_w - 1, x1))
        x2 = max(0, min(frame_w - 1, x2))
        y1 = max(0, min(frame_h - 1, y1))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out[label] = (x1, y1, x2, y2)
    return out


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)



_POINT_PROMPT_TEMPLATE = (
    "You are looking at a photo of a US-layout computer keyboard. "
    "Return the pixel coordinates of the CENTER of {label_phrase}. "
    "Use the convention origin top-left, +x right, +y down. "
    "The image is {w} pixels wide and {h} pixels tall.\n"
    "Return STRICT JSON with this exact schema and nothing else:\n"
    '{{"point": [x, y]}}\n'
    "Do not output any text other than the JSON object."
)

_BBOX_PROMPT_TEMPLATE = (
    "You are looking at a photo of a US-layout computer keyboard. "
    "Return a tight bounding box around {label_phrase}. "
    "Use the convention origin top-left, +x right, +y down. "
    "The image is {w} pixels wide and {h} pixels tall.\n"
    "Return STRICT JSON with this exact schema and nothing else:\n"
    '{{"bbox": [x1, y1, x2, y2]}}\n'
    "Do not output any text other than the JSON object."
)


def _label_phrase(label: str) -> str:
    """Friendly description of a key label for use in VLM prompts."""
    if label == "space": return "the space bar"
    if label == "enter": return "the Enter key"
    if label == "tab":   return "the Tab key"
    if label == "shift_l": return "the left Shift key"
    if label == "shift_r": return "the right Shift key"
    if label == "backspace": return "the Backspace key"
    if label == "caps_lock": return "the Caps Lock key"
    if len(label) == 1 and label.isalpha():
        return f'the "{label.upper()}" key'
    return f"the {label} key"


def _parse_point(text: str, frame_w: int, frame_h: int
                 ) -> tuple[float, float] | None:
    """Parse a single (x, y) point from a JSON response or Molmo point tag."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
        pt = data.get("point") or data.get("coordinates") or data.get("center")
        if isinstance(pt, list) and len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
            if 0 <= x < frame_w and 0 <= y < frame_h:
                return (x, y)
    except Exception:
        pass
    # Molmo's native <point x="..." y="..." ...> tag
    m = re.search(r'<point\s+([^>]*)\bx="([\d.]+)"\s+y="([\d.]+)"',
                  text, re.IGNORECASE)
    if m:
        try:
            x_pct, y_pct = float(m.group(2)), float(m.group(3))
            return (x_pct / 100.0 * frame_w, y_pct / 100.0 * frame_h)
        except ValueError:
            pass
    # Loose extraction: first two numbers in the response
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        try:
            x, y = float(nums[0]), float(nums[1])
            if 0 <= x < frame_w and 0 <= y < frame_h:
                return (x, y)
        except ValueError:
            pass
    return None


class VLMBackend(ABC):
    @abstractmethod
    def _query(self, image_bgr: np.ndarray, prompt: str) -> str: ...

    def detect_bbox(self, image_bgr: np.ndarray, label: str,
                    max_retries: int = 1) -> tuple[int, int, int, int] | None:
        """Single-key bbox query. Returns (x1,y1,x2,y2) or None on failure."""
        h, w = image_bgr.shape[:2]
        prompt = _BBOX_PROMPT_TEMPLATE.format(
            label_phrase=_label_phrase(label), w=w, h=h,
        )
        for attempt in range(max_retries + 1):
            try:
                raw = self._query(image_bgr, prompt)
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if m: raw_json_text = m.group(1)
                else:
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    raw_json_text = m.group(0) if m else raw
                data = json.loads(raw_json_text)
                bbox = data.get("bbox") or data.get("box")
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
                    if x2 > x1 and y2 > y1:
                        x1 = max(0, min(w - 1, x1))
                        x2 = max(0, min(w - 1, x2))
                        y1 = max(0, min(h - 1, y1))
                        y2 = max(0, min(h - 1, y2))
                        return (x1, y1, x2, y2)
            except Exception:
                pass
            time.sleep(0.3 * (attempt + 1))
        return None

    def detect_point(self, image_bgr: np.ndarray, label: str,
                     max_retries: int = 1) -> tuple[float, float] | None:
        """Single-key point query. Returns (x, y) or None on failure."""
        h, w = image_bgr.shape[:2]
        prompt = _POINT_PROMPT_TEMPLATE.format(
            label_phrase=_label_phrase(label), w=w, h=h,
        )
        for attempt in range(max_retries + 1):
            try:
                raw = self._query(image_bgr, prompt)
                pt = _parse_point(raw, w, h)
                if pt is not None:
                    return pt
            except Exception:
                pass
            time.sleep(0.3 * (attempt + 1))
        return None

    def detect_anchors(
        self,
        image_bgr: np.ndarray,
        anchor_labels: Iterable[str],
        max_retries: int = 2,
    ) -> dict[str, tuple[int, int, int, int]]:
        labels = list(anchor_labels)
        prompt = PROMPT_TEMPLATE.format(labels=", ".join(labels))
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                raw_text = self._query(image_bgr, prompt)
                raw_json = _parse_strict_json(raw_text)
                bboxes   = _validate_bboxes(
                    raw_json, labels,
                    frame_w=image_bgr.shape[1], frame_h=image_bgr.shape[0]
                )
                if bboxes:
                    return bboxes
                last_err = ValueError(f"No valid keys parsed from: {raw_text[:200]!r}")
            except Exception as e:
                last_err = e
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"VLM failed after {max_retries+1} attempts: {last_err!r}")



class GeminiBackend(VLMBackend):
    """Google Gemini 2.5 Pro (or any vision-capable Gemini model)."""

    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None):
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Gemini backend requires google-generativeai. "
                "Install via: pip install google-generativeai"
            ) from e
        import google.generativeai as genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in your .env or export it. "
                "Never commit the key to source."
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    def _query(self, image_bgr: np.ndarray, prompt: str) -> str:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        resp = self._model.generate_content(
            [prompt, pil],
            generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        return resp.text or ""



class QwenBackend(VLMBackend):
    """Local Qwen2.5-VL. Heavy install; only use if no API key available.

    Note: Qwen2.5-VL has dynamic resolution support. The processor's
    DEFAULT max_pixels is small (~12M pixels with 14x14 patches works out
    to roughly 1.1k tokens), which for a 1280x720 keyboard frame downsizes
    each keycap to a handful of pixels and tanks pointing accuracy.

    We bump min_pixels/max_pixels by default so the model sees the keyboard
    at near-native resolution. This is the single biggest knob for Qwen
    grounding accuracy on small targets.

    See the Qwen2.5-VL docs:
        https://github.com/QwenLM/Qwen2.5-VL#image-resolution-for-performance-boost
    """

    # 1280x720 = 921,600 px. We want to NOT downsize that, and allow some
    # headroom for 1920x1080 frames too. Round numbers below.
    DEFAULT_MIN_PIXELS = 256 * 28 * 28          # ~200k pixels minimum
    DEFAULT_MAX_PIXELS = 2304 * 28 * 28         # ~1.8M pixels maximum (~ 1280x1440)

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            import qwen_vl_utils  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Qwen backend requires extras. Install via: "
                "pip install 'cybertyping[qwen]' and a matching torch."
            ) from e
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self._min_pixels = min_pixels if min_pixels is not None else self.DEFAULT_MIN_PIXELS
        self._max_pixels = max_pixels if max_pixels is not None else self.DEFAULT_MAX_PIXELS
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        # Pass min/max pixels at processor construction so they are baked
        # into the image preprocessor.
        self._proc = AutoProcessor.from_pretrained(
            model_id, min_pixels=self._min_pixels, max_pixels=self._max_pixels,
        )
        self._model_name = model_id

    def _query(self, image_bgr: np.ndarray, prompt: str) -> str:
        import cv2
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # Pass min/max_pixels at content level too (some Qwen versions key
        # off the message content, others off the processor; we set both
        # to be safe).
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil,
                 "min_pixels": self._min_pixels, "max_pixels": self._max_pixels},
                {"type": "text", "text": prompt},
            ],
        }]
        text_in = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._proc(text=[text_in], images=image_inputs, videos=video_inputs,
                            return_tensors="pt", padding=True).to(self._model.device)
        gen = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        out = self._proc.batch_decode(
            [g[inputs.input_ids.shape[1]:] for g in gen],
            skip_special_tokens=True
        )[0]
        return out



class ClaudeBackend(VLMBackend):
    """Anthropic Claude vision (e.g. claude-sonnet-4-6, claude-opus-4-6).

    Claude's vision API is well-calibrated for grounding tasks and is a
    strong alternative to Gemini when ANTHROPIC_API_KEY is available.
    Uses the standard JSON-schema prompt; response is JSON text.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "ClaudeBackend requires the anthropic SDK. Install via: "
                "pip install anthropic"
            ) from e
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Put it in your .env or export it."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model_name = model

    def _query(self, image_bgr: np.ndarray, prompt: str) -> str:
        import cv2
        # Encode the image as a base64 PNG for the Anthropic content block.
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
        resp = self._client.messages.create(
            model=self._model_name,
            max_tokens=2048,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png",
                                                  "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        # `resp.content` is a list of content blocks; concatenate text blocks.
        out_parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                out_parts.append(text)
        return "".join(out_parts)



# Molmo's pointing format:
#   <point x="34.5" y="56.7" alt="Q key">Q</point>
# x and y are normalized percentages (0..100) of image dimensions. There is
# also a `<points x1="..." y1="..." x2="..." y2="..." ...>` form for
# multi-point. We parse both.
_MOLMO_POINT_RE = re.compile(
    r'<point\s+([^>]*)\bx="([\d.]+)"\s+y="([\d.]+)"[^>]*>([^<]*)</point>',
    re.IGNORECASE,
)
_MOLMO_POINTS_RE = re.compile(
    r'<points\s+([^>]+)>([^<]*)</points>',
    re.IGNORECASE,
)


def _parse_molmo_points(text: str, frame_w: int, frame_h: int
                        ) -> dict[str, tuple[float, float]]:
    """Parse Molmo's <point ...>label</point> output into {label: (px, py)}.

    The same label may appear multiple times; we take the first occurrence
    (Molmo emits results in confidence-ranked order).
    """
    out: dict[str, tuple[float, float]] = {}
    for m in _MOLMO_POINT_RE.finditer(text):
        _attrs, x_str, y_str, label = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            x_pct = float(x_str)
            y_pct = float(y_str)
        except ValueError:
            continue
        px = x_pct / 100.0 * frame_w
        py = y_pct / 100.0 * frame_h
        key = label.strip().lower()
        if key and key not in out:
            out[key] = (px, py)
    # Plural <points x1=.. y1=.. x2=.. y2=..>label</points>
    for m in _MOLMO_POINTS_RE.finditer(text):
        attrs, label = m.group(1), m.group(2).strip().lower()
        if not label or label in out:
            continue
        # take the first (x1, y1) for the label center
        x_m = re.search(r'\bx1="([\d.]+)"', attrs)
        y_m = re.search(r'\by1="([\d.]+)"', attrs)
        if not (x_m and y_m):
            continue
        try:
            px = float(x_m.group(1)) / 100.0 * frame_w
            py = float(y_m.group(1)) / 100.0 * frame_h
        except ValueError:
            continue
        out[label] = (px, py)
    return out


class MolmoBackend(VLMBackend):
    """Allen AI Molmo. Pointing-trained -> generally far better pixel
    grounding than Qwen at comparable parameter count. Local model; first
    use will download weights (~8GB for Molmo-7B-D).

    We ask Molmo to point at each anchor label separately (small, focused
    prompts work best for pointing) and convert the returned points into
    bboxes by padding ±`point_pad_px` pixels around each point.
    """

    DEFAULT_MODEL = "allenai/Molmo-7B-D-0924"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        torch_dtype: str = "auto",
        point_pad_px: int = 24,
        per_anchor_prompts: bool = True,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "MolmoBackend requires extras. Install via: "
                "pip install 'cybertyping[molmo]' (transformers + torch)."
            ) from e
        from transformers import AutoModelForCausalLM, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch_dtype,
            device_map=device or "auto",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch_dtype,
            device_map=device or "auto",
        )
        self._model_name = model_id
        self._point_pad_px = int(point_pad_px)
        self._per_anchor_prompts = bool(per_anchor_prompts)

    # We override detect_anchors to use Molmo's native pointing prompt
    # rather than the strict-JSON template. The parent's parser doesn't
    # know about <point> tags.
    def detect_anchors(
        self,
        image_bgr: np.ndarray,
        anchor_labels: Iterable[str],
        max_retries: int = 1,
    ) -> dict[str, tuple[int, int, int, int]]:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        h, w = image_bgr.shape[:2]
        labels = list(anchor_labels)
        out: dict[str, tuple[int, int, int, int]] = {}

        # Friendly verbalisations of special-key labels for the prompt.
        def _verbal(label: str) -> str:
            if label == "space":
                return "the space bar"
            if label == "enter":
                return "the Enter key"
            if label in ("shift_l", "shift_r"):
                return "the left Shift key" if label.endswith("l") else "the right Shift key"
            if len(label) == 1 and label.isalpha():
                return f'the "{label.upper()}" key'
            return f"the {label} key"

        if self._per_anchor_prompts:
            # Per-anchor prompts: Molmo's pointing accuracy is best on
            # focused single-target queries.
            for label in labels:
                prompt = f"Point to {_verbal(label)} on this keyboard."
                try:
                    text = self._query_pil(pil, prompt)
                except Exception:
                    continue
                points = _parse_molmo_points(text, w, h)
                # Take the first point Molmo emitted for ANY label, since
                # it was prompted with a single target.
                if not points:
                    continue
                px, py = next(iter(points.values()))
                out[label] = self._point_to_bbox(px, py, w, h)
            return out

        # Single multi-target prompt fallback.
        target_list = ", ".join(_verbal(l) for l in labels)
        prompt = (f"Point to each of the following keys on this keyboard, "
                  f"emitting one <point> tag per key with the key label as "
                  f"the tag text: {target_list}")
        for _ in range(max_retries + 1):
            try:
                text = self._query_pil(pil, prompt)
            except Exception:
                continue
            points = _parse_molmo_points(text, w, h)
            for label in labels:
                # Try the label itself, then the upper-case form, then a
                # verbalised fallback.
                candidates = [label, label.upper(), _verbal(label).lower()]
                for c in candidates:
                    if c in points and label not in out:
                        px, py = points[c]
                        out[label] = self._point_to_bbox(px, py, w, h)
                        break
            if len(out) >= 4:
                break
        return out

    def _point_to_bbox(self, px: float, py: float, w: int, h: int
                       ) -> tuple[int, int, int, int]:
        pad = self._point_pad_px
        x1 = int(max(0, round(px - pad)))
        y1 = int(max(0, round(py - pad)))
        x2 = int(min(w - 1, round(px + pad)))
        y2 = int(min(h - 1, round(py + pad)))
        return (x1, y1, x2, y2)

    def _query_pil(self, pil_image, prompt: str) -> str:
        import torch
        inputs = self._processor.process(images=[pil_image], text=prompt)
        inputs = {k: (v.to(self._model.device).unsqueeze(0) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        from transformers import GenerationConfig
        with torch.inference_mode():
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=256, stop_strings="<|endoftext|>"),
                tokenizer=self._processor.tokenizer,
            )
        gen_tokens = output[0, inputs["input_ids"].size(1):]
        text = self._processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text

    # Required by the ABC even though we override detect_anchors.
    def _query(self, image_bgr: np.ndarray, prompt: str) -> str:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._query_pil(pil, prompt)

    def detect_point(self, image_bgr: np.ndarray, label: str,
                     max_retries: int = 1) -> tuple[float, float] | None:
        """Use Molmo's native pointing prompt. More accurate than asking
        for JSON because Molmo was specifically trained on pointing."""
        import cv2
        from PIL import Image
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        verbal = _label_phrase(label)
        prompt = f"Point to {verbal} on this keyboard."
        for attempt in range(max_retries + 1):
            try:
                text = self._query_pil(pil, prompt)
                points = _parse_molmo_points(text, w, h)
                if points:
                    # Return the FIRST point; with a per-label prompt Molmo
                    # emits its highest-confidence pick first.
                    return next(iter(points.values()))
            except Exception:
                pass
        return None



def get_default_backend(prefer: list[str] | None = None) -> VLMBackend:
    """Pick a backend automatically based on what's actually available.

    Cascade (first match wins):
      1. `prefer` ordering, if given
      2. ANTHROPIC_API_KEY -> Claude
      3. GEMINI_API_KEY    -> Gemini
      4. Molmo (if transformers + model are installed)
      5. Qwen  (if transformers + model are installed)
    """
    order = prefer or ["claude", "gemini", "molmo", "qwen"]
    for name in order:
        try:
            if name == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
                return ClaudeBackend()
            if name == "gemini" and os.environ.get("GEMINI_API_KEY"):
                return GeminiBackend()
            if name == "molmo":
                return MolmoBackend()
            if name == "qwen":
                return QwenBackend()
        except SystemExit:
            continue
        except RuntimeError:
            continue
    raise RuntimeError(
        "No VLM backend available. Set ANTHROPIC_API_KEY or GEMINI_API_KEY, "
        "or `pip install 'cybertyping[molmo]'` / `[qwen]` for an offline backend."
    )



class StubBackend(VLMBackend):
    """Returns a pre-baked anchor dictionary regardless of input image.

    Used by tests and the benchmark harness when no real VLM is available.
    """

    def __init__(self, baked: dict[str, tuple[int, int, int, int]]):
        self._baked = dict(baked)

    def _query(self, image_bgr: np.ndarray, prompt: str) -> str:
        # Emit the bboxes in the strict-JSON schema so the parent's parser
        # works unchanged.
        return json.dumps({"keys": {k: list(v) for k, v in self._baked.items()}})


if __name__ == "__main__":
    # Smoke test the parsers without making an API call.
    sample = '```json\n{"keys": {"q": [10, 20, 30, 40], "p": [100, 20, 130, 40]}}\n```'
    parsed = _parse_strict_json(sample)
    valid  = _validate_bboxes(parsed, ["q", "p", "z"], frame_w=200, frame_h=200)
    print("strict JSON:", valid)

    molmo_sample = '<point x="50.0" y="25.0" alt="Q key">Q</point>'
    pts = _parse_molmo_points(molmo_sample, 1280, 720)
    print("molmo points:", pts)
