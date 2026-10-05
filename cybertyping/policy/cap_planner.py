"""Code-as-Policies style planner.

Expands a natural-language instruction (e.g. "type hello world") into a
list of key labels via an LLM (Claude / Gemini / Mock). Modes:
  "json"  LLM emits a JSON array of labels (default, safer).
  "code"  LLM emits Python that calls press_sequence() inside a sandbox.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable



class _LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM(_LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "ClaudeLLM requires anthropic SDK: pip install anthropic"
            ) from e
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model, max_tokens=1024, temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        out = []
        for b in resp.content:
            t = getattr(b, "text", None)
            if t: out.append(t)
        return "".join(out)


class GeminiLLM(_LLMBackend):
    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None):
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError as e:
            raise SystemExit("GeminiLLM requires google-generativeai") from e
        import google.generativeai as genai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)

    def complete(self, system: str, user: str) -> str:
        resp = self._model.generate_content(
            [system, user],
            generation_config={"temperature": 0.0,
                               "response_mime_type": "application/json"},
        )
        return resp.text or ""


class MockLLM(_LLMBackend):
    """Deterministic fake LLM for unit tests.

    Behaviour: parses the user prompt for the canonical 'type "..."' pattern
    and emits a JSON array of characters (with " " -> "space", "\\n" -> "enter").
    For anything else, returns an empty list.
    """
    def complete(self, system: str, user: str) -> str:
        # JSON-encoded instructions appear with backslash-escaped quotes
        # in the user prompt, e.g. `type \"hello world\"`. Make the regex
        # tolerant of optional surrounding quote characters (raw or escaped).
        m = re.search(r"type\s+(?:\\?[\"']?)([a-z\s]+?)(?:\\?[\"']?)\s*(?:\\?n|\\|$|[\"'])",
                      user, re.IGNORECASE)
        if not m:
            m = re.search(r"type\s+(?:\\?[\"']?)([a-z\s]+)", user, re.IGNORECASE)
        if not m:
            return "[]"
        text = m.group(1).strip().lower()
        seq = []
        for c in text:
            if c == " ": seq.append("space")
            elif c == "\n": seq.append("enter")
            elif c.isalpha(): seq.append(c)
        return json.dumps(seq)


def _get_default_llm() -> _LLMBackend:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try: return ClaudeLLM()
        except Exception: pass
    if os.environ.get("GEMINI_API_KEY"):
        try: return GeminiLLM()
        except Exception: pass
    raise RuntimeError(
        "No LLM backend available. Set ANTHROPIC_API_KEY or GEMINI_API_KEY, "
        "or pass a MockLLM() explicitly."
    )



SYSTEM_PROMPT = """You are a planner for a robot that types on a US-layout
keyboard. The robot has a keymap of available keys (letters a-z, plus
'space' and 'enter'). Your job: take a natural-language instruction and
emit the press sequence that satisfies it.

OUTPUT RULES (json mode):
  - Return ONLY a JSON array of key labels, no prose.
  - Each label MUST be one of the keys provided in `available_keys`.
  - For a space character use the label "space".
  - For a newline / Enter use the label "enter".
  - Do not include digits, punctuation, or anything outside the available
    keys, silently skip unsupported characters in the instruction.
"""


CODE_SYSTEM_PROMPT = """You are a planner for a robot that types on a
US-layout keyboard. You will be given a natural-language instruction and
a list of available_keys. Write a SHORT Python function `plan(available_keys)`
that returns a list[str] of key labels to press. You may use:
    press_sequence(labels: list[str]) -> None
to append to the sequence; the function should return the full list at the
end. Output ONLY the code block, no prose."""


@dataclass
class PlanResult:
    sequence: list[str]
    raw_llm_output: str
    mode: str
    skipped_unsupported: list[str]


class CaPPlanner:
    """High-level natural-language planner."""

    def __init__(self, llm: _LLMBackend | None = None, mode: str = "json"):
        if mode not in ("json", "code"):
            raise ValueError("mode must be 'json' or 'code'")
        self._llm = llm or _get_default_llm()
        self._mode = mode

    def plan(self, instruction: str,
             available_keys: Iterable[str]) -> PlanResult:
        keys = list(available_keys)
        if self._mode == "json":
            user = (
                f"available_keys = {json.dumps(sorted(keys))}\n"
                f"instruction = {json.dumps(instruction)}\n"
                "Emit the JSON array."
            )
            raw = self._llm.complete(SYSTEM_PROMPT, user)
            seq = self._parse_json(raw)
        else:
            user = (
                f"available_keys = {json.dumps(sorted(keys))}\n"
                f"instruction = {json.dumps(instruction)}\n"
                "Emit a Python function plan(available_keys) returning the list."
            )
            raw = self._llm.complete(CODE_SYSTEM_PROMPT, user)
            seq = self._parse_code(raw, keys)
        valid, skipped = self._filter_to_known(seq, keys)
        return PlanResult(
            sequence=valid, raw_llm_output=raw, mode=self._mode,
            skipped_unsupported=skipped,
        )

    @staticmethod
    def _parse_json(text: str) -> list[str]:
        text = text.strip()
        # strip code fences if present
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if m: text = m.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # fall back: pull the outermost [...] block
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            try: data = json.loads(m.group(0))
            except Exception: return []
        if not isinstance(data, list):
            return []
        return [str(x).lower() for x in data if isinstance(x, (str,))]

    @staticmethod
    def _parse_code(text: str, available_keys: list[str]) -> list[str]:
        # extract the first ```python``` block, else use the raw text
        m = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
        code = m.group(1) if m else text
        # Build a restricted exec namespace.
        sequence: list[str] = []
        def press_sequence(labels):
            sequence.extend([str(x).lower() for x in labels])
        env = {"__builtins__": {"range": range, "len": len, "list": list,
                                "str": str, "int": int, "enumerate": enumerate,
                                "zip": zip, "min": min, "max": max,
                                "sorted": sorted, "set": set, "dict": dict},
               "press_sequence": press_sequence,
               "available_keys": list(available_keys)}
        try:
            exec(compile(code, "<cap_planner>", "exec"), env)
            if "plan" in env and callable(env["plan"]):
                ret = env["plan"](list(available_keys))
                if isinstance(ret, list):
                    return [str(x).lower() for x in ret]
            return sequence
        except Exception:
            return sequence

    @staticmethod
    def _filter_to_known(seq: list[str], keys: list[str]
                         ) -> tuple[list[str], list[str]]:
        kset = set(keys)
        out: list[str] = []
        skipped: list[str] = []
        for s in seq:
            label = s.strip().lower()
            if label == " ": label = "space"
            elif label == "\n": label = "enter"
            if label in kset: out.append(label)
            else: skipped.append(s)
        return out, skipped



if __name__ == "__main__":
    # No API call: uses the MockLLM to exercise the parsing logic.
    planner = CaPPlanner(llm=MockLLM(), mode="json")
    available = ["a","b","c","d","e","f","g","h","i","j","k","l","m",
                 "n","o","p","q","r","s","t","u","v","w","x","y","z",
                 "space","enter"]
    res = planner.plan("Please type 'hello world' on the keyboard.", available)
    print(f"sequence: {res.sequence}")
    print(f"skipped:  {res.skipped_unsupported}")
