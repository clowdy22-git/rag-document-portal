"""
Lightweight API-key auth for the Portal's endpoints.

This is a shared-secret check (a single key in an `X-API-Key` header), not
full user auth — appropriate for a small internal tool with one deployment,
not for a multi-tenant public product. If that changes, replace this with
real auth (OAuth/JWT) rather than extending it.

The key itself is never hardcoded: it's read from the PORTAL_API_KEY
environment variable, which in Phase 6 is injected by ECS from AWS Secrets
Manager rather than baked into the image or committed to source control.
"""

import os
import secrets

from fastapi import Header, HTTPException


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("PORTAL_API_KEY")

    if not expected:
        # Fail closed: if the operator forgot to set a key, refuse rather
        # than silently running with no auth at all.
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: PORTAL_API_KEY is not set.",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
