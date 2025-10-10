"""
LLM Text-to-CAD Engine
======================

Optional integration with OpenAI or Hugging Face to map natural language
prompts to a JSON geometry schema for the Text-to-CAD compiler workflow.

This module is designed to be safe-by-default:
- Only attempts network calls if the corresponding API keys exist in env vars
- Falls back gracefully if calls fail or responses are invalid

Environment variables:
- OPENAI_API_KEY       (optional) set to enable OpenAI path
- OPENAI_MODEL         (optional) default: gpt-4o-mini (or user-specified)
- HF_API_TOKEN         (optional) set to enable Hugging Face path
- HF_MODEL_ID          (optional) e.g. meta-llama/Meta-Llama-3-8B-Instruct

Author: KelmoidAI Genesis Team
License: MIT
"""
from __future__ import annotations

import os
import json
import re
import time
import typing as t

import requests


def _get_env(name: str, default: str | None = None) -> str | None:
    try:
        return os.environ.get(name, default)
    except Exception:
        return default


LLM_AVAILABLE = bool(_get_env("OPENAI_API_KEY") or _get_env("HF_API_TOKEN") or _get_env("HUGGINGFACE_API_TOKEN"))


class LLMTextToCAD:
    """Wrapper that queries an LLM to produce a geometry JSON schema.

    The LLM is instructed to output a strict JSON object with this shape:
    {
      "shape": "cube|sphere|cylinder|cone|pyramid|torus|plate|bracket|washer|nut|bearing|flange|pipe",
      "dimensions": {
         "length": number?, "width": number?, "height": number?, "radius": number?,
         "diameter": number?, "thickness": number?, "depth": number?
      },
      "color": string?,
      "notes": string?
    }
    All numeric units are millimeters.
    """

    def __init__(self) -> None:
        self._openai_key = _get_env("OPENAI_API_KEY")
        self._openai_model = _get_env("OPENAI_MODEL", "gpt-4o-mini")
        self._hf_token = _get_env("HF_API_TOKEN") or _get_env("HUGGINGFACE_API_TOKEN")
        self._hf_model = _get_env("HF_MODEL_ID", "gpt2")

    def available(self) -> bool:
        return bool(self._openai_key or self._hf_token)

    # ----------------------------- Public API ------------------------------ #

    def compile(self, prompt: str, timeout: float = 20.0) -> dict:
        """Return { success, schema, error } with geometry JSON schema if possible.

        Never raises; returns a dictionary with success flag and error string on failure.
        """
        try:
            text = self._query_llm(prompt, timeout=timeout)
            if not text:
                return {"success": False, "error": "Empty LLM response"}

            schema = self._extract_json(text)
            if not schema:
                return {"success": False, "error": "Could not parse JSON from LLM response"}

            # Validate minimally
            shape = (schema.get("shape") or "").strip().lower()
            if not shape:
                return {"success": False, "error": "Schema missing 'shape'"}

            dims = schema.get("dimensions") or {}
            if not isinstance(dims, dict):
                return {"success": False, "error": "Schema 'dimensions' is not an object"}

            # Coerce numeric fields
            for k in ["length", "width", "height", "radius", "diameter", "thickness", "depth"]:
                v = dims.get(k)
                if v is None:
                    continue
                try:
                    dims[k] = float(v)
                except Exception:
                    return {"success": False, "error": f"Dimension '{k}' is not numeric"}

            # Provide safe defaults if needed
            if shape in ("cube", "box"):
                dims.setdefault("length", 10.0)
                dims.setdefault("width", 10.0)
                dims.setdefault("height", 10.0)
            elif shape == "sphere":
                dims.setdefault("radius", 5.0)
            elif shape in ("cylinder", "tube", "pipe"):
                dims.setdefault("radius", 5.0)
                dims.setdefault("height", 10.0)

            schema["shape"] = shape
            schema["dimensions"] = dims
            return {"success": True, "schema": schema}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---------------------------- Internal -------------------------------- #

    def _query_llm(self, prompt: str, timeout: float = 20.0) -> str | None:
        """Try OpenAI first (if present), else Hugging Face, else None."""
        if self._openai_key:
            text = self._call_openai(prompt, timeout=timeout)
            if text:
                return text
        if self._hf_token:
            text = self._call_hf(prompt, timeout=timeout)
            if text:
                return text
        return None

    def _system_prompt(self) -> str:
        return (
            "You are a CAD geometry compiler. Given a natural language prompt, "
            "output ONLY a strict JSON object describing one geometry. Units are millimeters. "
            "Do not include any additional text. The JSON schema is:\n"
            "{\n  \"shape\": \"cube|box|sphere|cylinder|cone|pyramid|torus|plate|bracket|washer|nut|bearing|flange|pipe\",\n"
            "  \"dimensions\": {\n    \"length\": number?, \"width\": number?, \"height\": number?, \"radius\": number?,\n    \"diameter\": number?, \"thickness\": number?, \"depth\": number?\n  },\n"
            "  \"color\": string?,\n  \"notes\": string?\n}"
        )

    def _call_openai(self, prompt: str, timeout: float = 20.0) -> str | None:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._openai_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": self._openai_model,
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                # Encourage JSON output
                "temperature": 0.2,
            }
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            if resp.status_code >= 400:
                return None
            j = resp.json()
            content = (
                j.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return content or None
        except Exception:
            return None

    def _call_hf(self, prompt: str, timeout: float = 20.0) -> str | None:
        """Call Hugging Face inference API with fallback models"""
        models_to_try = [
            "microsoft/DialoGPT-medium",
            "gpt2",
            "distilgpt2",
            "microsoft/DialoGPT-small"
        ]
        
        for model in models_to_try:
            try:
                # Try text generation with simple prompt formatting
                formatted_prompt = f"Create JSON for CAD geometry: {prompt}. Format: {{\"shape\": \"cube\", \"dimensions\": {{\"length\": 20}}}}"
                
                url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {
                    "Authorization": f"Bearer {self._hf_token}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "inputs": formatted_prompt,
                    "parameters": {
                        "max_new_tokens": 100,
                        "temperature": 0.3,
                        "do_sample": True,
                        "return_full_text": False
                    },
                    "options": {
                        "wait_for_model": True
                    }
                }
                
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if resp.status_code == 200:
                    j = resp.json()
                    
                    # Handle different response formats
                    text = None
                    if isinstance(j, list) and j:
                        if isinstance(j[0], dict):
                            text = j[0].get("generated_text", "")
                        else:
                            text = str(j[0])
                    elif isinstance(j, dict):
                        text = j.get("generated_text", "")
                    
                    if text and text.strip():
                        # Create a simple JSON response if the model doesn't generate proper JSON
                        if not any(x in text.lower() for x in ["{", "shape", "dimensions"]):
                            # Generate a basic JSON based on the prompt
                            if "cube" in prompt.lower():
                                return '{"shape": "cube", "dimensions": {"length": 20, "width": 20, "height": 20}}'
                            elif "sphere" in prompt.lower():
                                return '{"shape": "sphere", "dimensions": {"radius": 10}}'
                            elif "cylinder" in prompt.lower():
                                return '{"shape": "cylinder", "dimensions": {"radius": 10, "height": 20}}'
                            else:
                                return '{"shape": "cube", "dimensions": {"length": 20, "width": 20, "height": 20}}'
                        return text
                        
                elif resp.status_code == 503:
                    # Model is loading, try next one
                    continue
                else:
                    # Try next model
                    continue
                    
            except Exception:
                # Try next model
                continue
        
        # If all models fail, return a basic response based on the prompt
        return self._generate_fallback_json(prompt)
    
    def _generate_fallback_json(self, prompt: str) -> str:
        """Generate basic JSON when LLM fails"""
        prompt_lower = prompt.lower()
        
        # Extract dimensions if mentioned
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', prompt)
        size = float(numbers[0]) if numbers else 20.0
        
        if any(word in prompt_lower for word in ['cube', 'box', 'block']):
            return f'{{"shape": "cube", "dimensions": {{"length": {size}, "width": {size}, "height": {size}}}}}'
        elif any(word in prompt_lower for word in ['sphere', 'ball']):
            return f'{{"shape": "sphere", "dimensions": {{"radius": {size/2}}}}}'
        elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pipe']):
            height = float(numbers[1]) if len(numbers) > 1 else size * 2
            return f'{{"shape": "cylinder", "dimensions": {{"radius": {size/2}, "height": {height}}}}}'
        elif any(word in prompt_lower for word in ['cone', 'pyramid']):
            height = float(numbers[1]) if len(numbers) > 1 else size
            return f'{{"shape": "cone", "dimensions": {{"radius": {size/2}, "height": {height}}}}}'
        else:
            return f'{{"shape": "cube", "dimensions": {{"length": {size}, "width": {size}, "height": {size}}}}}'

    def _extract_json(self, text: str) -> dict | None:
        """Extract the first top-level JSON object from arbitrary text."""
        try:
            # Quick path if it's already clean JSON
            text = text.strip()
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)

            # Otherwise, find the first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                return json.loads(snippet)
        except Exception:
            return None
        return None

