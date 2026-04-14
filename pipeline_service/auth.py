"""Authentication helpers for pipeline service endpoints (HTTP + WebSocket)."""

from __future__ import annotations

from fastapi import HTTPException, status
from jose import JWTError, jwt

from app.config import get_settings

settings = get_settings()


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    value = authorization.strip()
    if not value.lower().startswith("bearer "):
        return None
    token = value[7:].strip()
    return token or None


def _decode_external_jwt(token: str) -> dict | None:
    """Decode a JWT signed by the external mobile auth service.

    Returns the full payload dict on success, or None if JWT_SECRET is
    not configured or the token is invalid.
    """
    secret = settings.JWT_SECRET
    if not secret:
        return None
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError:
        return None


def _decode_internal_jwt(token: str) -> str | None:
    """Decode a JWT signed by our own backend (PWA auth).

    Returns the user_id (sub claim) or None.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


async def authenticate_pipeline_request(
    x_pipeline_token: str | None,
    authorization: str | None,
) -> str | None:
    """Authenticate request via service token OR JWT.

    Returns:
        user_id (str) when authenticated via JWT, or None when authenticated
        via service token.
    """
    # 1. Service-to-service shared token (backend → pipeline)
    expected = settings.PIPELINE_SHARED_TOKEN
    if expected and x_pipeline_token:
        if x_pipeline_token != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid pipeline token")
        return None

    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication (Bearer token or X-Pipeline-Token)",
        )

    # 2. External JWT (mobile app) — try first
    ext_payload = _decode_external_jwt(token)
    if ext_payload:
        user_id = ext_payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing userId")
        return user_id

    # 3. Internal JWT (PWA fallback)
    internal_user_id = _decode_internal_jwt(token)
    if internal_user_id:
        return internal_user_id

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def verify_scan_access(scan_id: str, user_id: str | None) -> None:
    """Check that the authenticated user owns this job/scan.

    Service-token callers (user_id=None) are always allowed.
    """
    if user_id is None:
        return

    from pipeline_service.routes import _load_job

    state = _load_job(scan_id)
    if state is None:
        return  # job not yet created — allow (will 404 later)

    owner_id = state.get("owner_user_id")
    if owner_id and owner_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden for this job",
        )
