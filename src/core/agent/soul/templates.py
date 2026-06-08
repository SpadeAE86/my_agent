# core/agent/soul/templates.py — Soul & Persona Templates Loader
import os
from pathlib import Path

SOUL_DIR = Path(__file__).resolve().parent

def load_template(filename: str) -> str:
    """Loads a markdown template from the soul directory."""
    path = SOUL_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def get_identity_template() -> str:
    return load_template("IDENTITY.md")

def get_soul_template() -> str:
    return load_template("SOUL.md")

def get_user_template() -> str:
    return load_template("USER.md")
