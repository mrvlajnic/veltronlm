"""Serving package: HTTP API and chat UI mounting."""

from .app import STATE, app, rate_limit

__all__ = ["app", "STATE", "rate_limit", "mount_chatbot"]


def mount_chatbot(app_obj=None):
    """Convenience re-export so callers need not import the chatbot package directly."""
    from ..chatbot.server import mount_chatbot as _mount

    return _mount(app_obj if app_obj is not None else app)
