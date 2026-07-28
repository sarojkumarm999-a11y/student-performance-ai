from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()


def _call_openrouter(
    api_key: str,
    prompt: str,
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 400,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Student Performance System",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        choices = res_data.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "").strip()
        return ""


def generate_student_insights(
    *,
    student: Dict[str, Any],
    predictions: Dict[str, Any],
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 400,
) -> str:
    """
    Returns a short, personalized set of actionable insights using OpenRouter LLM API.
    Uses DEFAULT_API_KEY or OPENROUTER_API_KEY / LLM_API_KEY / ANTHROPIC_API_KEY.
    """
    prompt = f"""
You are an academic advisor. Provide personalized, practical, and supportive guidance.

Student profile (features):
{student}

Model predictions:
{predictions}

Write:
- A 3-sentence summary of likely performance and key drivers
- 5 bullet action plan (specific, measurable where possible)
- 3 risk flags to monitor
- 2 encouragement lines in a warm tone
Avoid medical/diagnostic claims. Avoid inventing facts not in the profile.
""".strip()

    api_key = (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
        or DEFAULT_API_KEY
    )

    if api_key.startswith("sk-ant-"):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=max_tokens,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in getattr(msg, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    parts.append(getattr(block, "text", ""))
            return "\n".join([p for p in parts if p]).strip()
        except Exception:
            pass

    return _call_openrouter(api_key, prompt, model=model, max_tokens=max_tokens)
