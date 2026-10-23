from __future__ import annotations

import math
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute


def _has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(item) for item in value)
    return False


class JevRoute(APIRoute):
    """Keep nonfinite request inputs from breaking validation-error JSON."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                errors = exc.errors()
                if not _has_nonfinite(errors):
                    # Preserve the host's existing handler for ordinary errors.
                    raise
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": jsonable_encoder(
                            errors,
                            custom_encoder={
                                float: lambda value: value if math.isfinite(value) else str(value)
                            },
                        )
                    },
                )

        return handle
