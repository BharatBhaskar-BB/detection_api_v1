"""Push notification sender — sends Web Push to subscribed devices."""

import json
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.models import PushSubscription, Scan

settings = get_settings()


async def send_push_to_scan_owner(
    db: AsyncSession,
    scan_id: str,
    title: str,
    body: str,
    url: str | None = None,
) -> int:
    """Send push notification to all devices of the scan owner. Returns count sent."""
    if not settings.VAPID_PRIVATE_KEY or not settings.VAPID_PUBLIC_KEY:
        return 0

    # Get scan owner
    result = await db.execute(select(Scan.user_id).where(Scan.id == scan_id))
    row = result.first()
    if not row:
        return 0
    user_id = row[0]

    # Get all subscriptions for this user
    result = await db.execute(
        select(PushSubscription).where(PushSubscription.user_id == user_id)
    )
    subscriptions = result.scalars().all()
    if not subscriptions:
        return 0

    payload = json.dumps({
        "title": title,
        "body": body,
        "icon": "/icons/icon-192.png",
        "badge": "/icons/icon-64.png",
        "url": url or f"/scans/{scan_id}",
    })

    sent = 0
    for sub in subscriptions:
        try:
            from pywebpush import webpush
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=payload,
                vapid_private_key=settings.VAPID_PRIVATE_KEY,
                vapid_claims={"sub": settings.VAPID_CLAIMS_EMAIL},
            )
            sent += 1
        except Exception as e:
            logger.warning(f"Push failed for subscription {sub.id}: {e}")
            # If subscription is gone (410), remove it
            if "410" in str(e) or "404" in str(e):
                await db.delete(sub)
                await db.flush()

    return sent
