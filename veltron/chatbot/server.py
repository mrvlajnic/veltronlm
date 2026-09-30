"""Chat UI server.

Mounts the static chat page onto the same FastAPI app that serves the API, so the demo runs
as a single process with no CORS configuration and no second port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX = STATIC_DIR / "index.html"


def mount_chatbot(app: Any) -> Any:
    """Add ``/chat`` and ``/`` routes to the API application."""

    @app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
    def chat_page() -> Any:
        if not INDEX.exists():
            return HTMLResponse(
                "<h1>chat UI missing</h1><p>Expected "
                f"{INDEX}</p>",
                status_code=500,
            )
        return FileResponse(INDEX, media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Any:
        # A 204 keeps browsers from logging a 404 on every page load.
        from fastapi.responses import Response

        return Response(status_code=204)

    return app


__all__ = ["mount_chatbot", "STATIC_DIR", "INDEX"]
