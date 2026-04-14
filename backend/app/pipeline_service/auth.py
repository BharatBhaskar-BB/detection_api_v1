"""Authentication helpers for pipeline service endpoints (HTTP + WebSocket)."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import select

from app.config import get_settings
from app.database import async_session
from app.models.models import Scan, User
from app.utils.auth import decode_access_token

settings = get_settings()


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    value = authorization.strip()
    if not value.lower().startswith("bearer "):
        return None
    token = value[7:].strip()
    return token or None


async def authenticate_pipeline_request(
    x_pipeline_token: str | None,
    authorization: str | None,
) -> str | None:
    """Authenticate request via service token OR user JWT.

    Returns:
        user_id when authenticated via JWT, or None when authenticated via
        service token.
    """
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

    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    async with async_session() as db:
        result = await db.execute(
            select(User.id).where(User.id == user_id, User.is_active.is_(True))
        )
        active_user_id = result.scalar_one_or_none()

    if not active_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user_id


async def verify_scan_access(scan_id: str, user_id: str | None) -> None:
    """When authenticated as user, ensure the scan belongs to that user."""
    if user_id is None:
        return

    async with async_session() as db:
        result = await db.execute(
            select(Scan.id).where(Scan.id == scan_id, Scan.user_id == user_id)
        )
        owned_scan = result.scalar_one_or_none()

    if not owned_scan:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this scan")
