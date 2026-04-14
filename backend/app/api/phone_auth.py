"""Phone auth routes — Twilio WhatsApp OTP send/verify + name capture."""

import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.models import User
from app.models.schemas import (
    PhoneSendOTPRequest,
    PhoneVerifyOTPRequest,
    PhoneSetNameRequest,
    TokenResponse,
    UserResponse,
)
from app.utils.auth import create_access_token, hash_password
from app.api.deps import get_current_user

router = APIRouter(prefix="/auth/phone", tags=["auth"])
settings = get_settings()

# In-memory OTP store: phone -> {code, expires_at, attempts}
# For production: use Redis
_otp_store: dict[str, dict] = {}

MAX_OTP_ATTEMPTS = 5


def _normalize_phone(phone: str) -> str:
    """Ensure phone starts with +."""
    phone = phone.strip()
    if not phone.startswith("+"):
        phone = "+" + phone
    return phone


@router.post("/send-otp")
async def send_otp(body: PhoneSendOTPRequest):
    """Send a 6-digit OTP via SMS or WhatsApp."""
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_API_KEY_SID:
        raise HTTPException(503, "Twilio not configured")

    phone = _normalize_phone(body.phone)
    channel = body.channel  # "sms" or "whatsapp"

    # Rate limit: don't resend if OTP was sent < 60s ago
    existing = _otp_store.get(phone)
    if existing and time.time() - existing.get("sent_at", 0) < 60:
        return {"status": "already_sent", "message": "OTP already sent. Wait 60s to resend."}

    # Generate 6-digit OTP
    code = f"{secrets.randbelow(900000) + 100000}"

    # Send via Twilio
    try:
        from twilio.rest import Client
        from twilio.http.http_client import TwilioHttpClient
        import ssl

        # Bypass SSL verification (macOS corporate cert issue)
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        http_client = TwilioHttpClient()
        http_client.session.verify = False

        client = Client(
            settings.TWILIO_API_KEY_SID,
            settings.TWILIO_API_KEY_SECRET,
            settings.TWILIO_ACCOUNT_SID,
            http_client=http_client,
        )
        if channel == "whatsapp":
            from_number = settings.TWILIO_WHATSAPP_FROM
            to_number = f"whatsapp:{phone}"
        else:
            # SMS: use the number without the whatsapp: prefix
            from_number = settings.TWILIO_WHATSAPP_FROM.replace("whatsapp:", "")
            to_number = phone

        client.messages.create(
            body=f"Your BundleBox verification code is: {code}",
            from_=from_number,
            to=to_number,
        )
    except Exception as e:
        logger.error(f"Twilio send ({channel}) failed for {phone}: {e}")
        raise HTTPException(502, f"Failed to send OTP via {channel}. Check your phone number.")

    _otp_store[phone] = {
        "code": code,
        "expires_at": time.time() + settings.OTP_EXPIRY_SECONDS,
        "sent_at": time.time(),
        "attempts": 0,
    }
    channel_label = "WhatsApp" if channel == "whatsapp" else "SMS"
    logger.info(f"OTP sent to {phone} via {channel_label}")
    return {"status": "sent", "message": f"OTP sent via {channel_label}"}


@router.post("/verify-otp", response_model=TokenResponse)
async def verify_otp(body: PhoneVerifyOTPRequest, db: AsyncSession = Depends(get_db)):
    """Verify OTP and return JWT. Creates user if new."""
    phone = _normalize_phone(body.phone)
    otp_entry = _otp_store.get(phone)

    if not otp_entry:
        raise HTTPException(400, "No OTP sent for this number. Request a new one.")

    if time.time() > otp_entry["expires_at"]:
        del _otp_store[phone]
        raise HTTPException(400, "OTP expired. Request a new one.")

    otp_entry["attempts"] += 1
    if otp_entry["attempts"] > MAX_OTP_ATTEMPTS:
        del _otp_store[phone]
        raise HTTPException(429, "Too many attempts. Request a new OTP.")

    if not secrets.compare_digest(body.code.strip(), otp_entry["code"]):
        remaining = MAX_OTP_ATTEMPTS - otp_entry["attempts"]
        raise HTTPException(401, f"Invalid OTP. {remaining} attempts remaining.")

    # OTP verified — clean up
    del _otp_store[phone]

    # Find or create user by phone
    result = await db.execute(select(User).where(User.phone == phone))
    user = result.scalar_one_or_none()
    is_new = False

    if user is None:
        user = User(phone=phone, name="")
        db.add(user)
        await db.flush()
        is_new = True

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/set-name", response_model=UserResponse)
async def set_name(
    body: PhoneSetNameRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Complete profile after phone verification — set name and optionally email + password."""
    user.name = body.name

    if body.email:
        # Check email not already taken by another user
        result = await db.execute(
            select(User).where(User.email == body.email, User.id != user.id)
        )
        if result.scalar_one_or_none():
            raise HTTPException(409, "Email already registered to another account")
        user.email = body.email

    if body.password:
        user.hashed_password = hash_password(body.password)

    await db.commit()
    await db.refresh(user)
    return user
