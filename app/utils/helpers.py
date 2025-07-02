import logging
from typing import Any, Union

def normalize_text_input(text: Any) -> str:
    if isinstance(text, list):
        return " ".join(map(str, text))
    return str(text)

def format_seconds(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"