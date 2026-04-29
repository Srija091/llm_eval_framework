"""
runners/model_runner.py
Unified model interface — supports Ollama (local, free) and Groq (free tier).
"""

import time
from typing import Optional
import ollama


def call_ollama(prompt: str, model: str = "llama3.2", system: Optional[str] = None) -> str:
    """
    Call a local Ollama model. Free, runs offline.
    Install Ollama: https://ollama.ai
    Pull a model: ollama pull llama3.2
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] Ollama call failed: {str(e)}"


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", system: Optional[str] = None) -> str:
    """
    Call Groq free API. Sign up at https://groq.com (no credit card needed).
    Set env var: export GROQ_API_KEY=your_key
    """
    try:
        from groq import Groq
        import os
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Groq call failed: {str(e)}"


def call_model(prompt: str, model: str = "llama3.2", backend: str = "ollama",
               system: Optional[str] = None) -> dict:
    """
    Unified model call — returns response + metadata.
    """
    start = time.time()

    if backend == "ollama":
        response = call_ollama(prompt, model=model, system=system)
    elif backend == "groq":
        response = call_groq(prompt, model=model, system=system)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    latency = round(time.time() - start, 3)
    return {
        "prompt": prompt,
        "response": response,
        "model": model,
        "backend": backend,
        "latency_s": latency
    }
