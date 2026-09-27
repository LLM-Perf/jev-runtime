"""Explicit, hash-bound local chat templates; never alter the host tokenizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from jev_runtime.errors import JevError
from jev_runtime.schema import Contract


class TemplateFile(Contract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["jinja", "json"] = "jinja"


def read_template(spec: TemplateFile) -> str:
    # Read one bounded snapshot and verify those exact bytes before parsing.
    with Path(spec.path).open("rb") as file:
        raw = file.read(1_048_577)
    if len(raw) > 1_048_576:
        raise JevError("template_file_invalid", "Chat template exceeds the 1 MiB limit", 409)
    if hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise JevError("template_file_mismatch", "Chat template file SHA256 differs", 409)
    try:
        value = raw.decode("utf-8")
        if spec.format == "json":
            document = json.loads(value)
            value = document.get("chat_template") if isinstance(document, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected a nonempty template string")
    except (UnicodeError, ValueError) as exc:
        raise JevError(
            "template_file_invalid", "Invalid UTF-8 chat template document", 409
        ) from exc
    return value
