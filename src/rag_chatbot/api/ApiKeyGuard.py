"""Optional API key check for the protected routes."""

import secrets

from fastapi import Header, HTTPException, status


class ApiKeyGuard:
    """With no key configured every call passes. With a key, callers send X-API-Key or Authorization: Bearer."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def __call__(self, x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)):
        if not self.enabled:
            return
        supplied = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
        if not supplied or not secrets.compare_digest(supplied, self.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="A valid API key is required.")
