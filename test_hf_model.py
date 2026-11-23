"""Minimal live check for MediBot's configured Hugging Face model/provider."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from huggingface_hub import InferenceClient


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"


def main() -> None:
    """Send one tiny request and print only its answer or a sanitized error."""
    token = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8")).get("HF_TOKEN", "")
    if not isinstance(token, str) or not token.strip():
        print("Error: HF_TOKEN is missing from .streamlit/secrets.toml.")
        return

    try:
        # With an hf_ token, huggingface_hub routes Featherless requests through HF.
        client = InferenceClient(provider="featherless-ai", token=token.strip())
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            model=MODEL,
            max_tokens=10,
            temperature=0,
        )
        print(result.choices[0].message.content or "")
    except Exception as exc:
        # Never let an exception accidentally reveal a token in terminal output.
        safe_error = str(exc).replace(token.strip(), "[redacted]")
        safe_error = re.sub(r"hf_[A-Za-z0-9]{10,}", "[redacted]", safe_error)
        print(f"Error: {safe_error}")


if __name__ == "__main__":
    main()
