"""
MonuScript - Authentication & Token Verification Module
Validates Supabase Auth access tokens (including anonymous sessions)
and binds the authenticated user ID to the request context.
"""

import os
import time
import functools
import requests
import jwt
from flask import request, jsonify, g
from config import Config

# In-memory cache for Supabase Auth API token verification: {token: (user_id, exp_time)}
_AUTH_CACHE = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

def verify_token(token: str) -> dict:
    """
    Verifies a Supabase access token and returns its claims.
    Raises ValueError with description if verification fails.
    """
    if not token or not isinstance(token, str):
        raise ValueError("Missing or invalid token string.")

    now = time.time()

    # 1. Check in-memory cache first to reduce latency
    if token in _AUTH_CACHE:
        cached_user, exp = _AUTH_CACHE[token]
        if now < exp:
            return cached_user
        else:
            del _AUTH_CACHE[token]

    # 2. Local verification using SUPABASE_JWT_SECRET if configured
    if Config.SUPABASE_JWT_SECRET:
        try:
            # Supabase tokens typically use HS256 with project JWT secret
            payload = jwt.decode(
                token,
                Config.SUPABASE_JWT_SECRET,
                algorithms=["HS256", "RS256", "ES256"],
                audience="authenticated",
                options={"verify_aud": True, "verify_exp": True}
            )
            user_id = payload.get("sub")
            if not user_id:
                raise ValueError("Token missing 'sub' (user identity) claim.")

            result = {
                "user_id": str(user_id),
                "email": payload.get("email"),
                "role": payload.get("role", "authenticated"),
                "is_anonymous": payload.get("is_anonymous", False),
                "claims": payload
            }
            exp_time = payload.get("exp", now + _CACHE_TTL_SECONDS)
            _AUTH_CACHE[token] = (result, min(exp_time, now + _CACHE_TTL_SECONDS))
            return result
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired. Please refresh your session.")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token signature: {str(e)}")

    # 3. Remote verification via Supabase Auth API endpoint if URL & KEY configured
    if Config.SUPABASE_URL and Config.SUPABASE_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "apikey": Config.SUPABASE_KEY
            }
            resp = requests.get(
                f"{Config.SUPABASE_URL}/auth/v1/user",
                headers=headers,
                timeout=5
            )
            if resp.status_code == 200:
                user_data = resp.json()
                user_id = user_data.get("id")
                if not user_id:
                    raise ValueError("Supabase user response missing user ID.")

                result = {
                    "user_id": str(user_id),
                    "email": user_data.get("email"),
                    "role": user_data.get("role", "authenticated"),
                    "is_anonymous": user_data.get("is_anonymous", False),
                    "claims": user_data
                }
                _AUTH_CACHE[token] = (result, now + _CACHE_TTL_SECONDS)
                return result
            else:
                raise ValueError("Supabase token rejected by authentication server.")
        except requests.RequestException as e:
            raise ValueError(f"Failed to reach Supabase authentication service: {str(e)}")

    # 4. Fallback for Local Development & Automated Unit Testing
    # Uses Config.SECRET_KEY to verify locally signed HMAC tokens
    try:
        payload = jwt.decode(
            token,
            Config.SECRET_KEY,
            algorithms=["HS256"],
            options={"verify_exp": True}
        )
        user_id = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise ValueError("Token missing user ID claim.")
        return {
            "user_id": str(user_id),
            "email": payload.get("email"),
            "role": "authenticated",
            "is_anonymous": payload.get("is_anonymous", True),
            "claims": payload
        }
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except jwt.InvalidTokenError:
        raise ValueError("Token signature verification failed and no Supabase credentials configured.")

def create_local_token(user_id: str, is_anonymous: bool = True, expires_in_seconds: int = 86400) -> str:
    """
    Generates a signed JWT for local tests or fallback environments using Config.SECRET_KEY or SUPABASE_JWT_SECRET.
    """
    secret = Config.SUPABASE_JWT_SECRET or Config.SECRET_KEY
    payload = {
        "sub": str(user_id),
        "user_id": str(user_id),
        "aud": "authenticated",
        "role": "authenticated",
        "is_anonymous": is_anonymous,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in_seconds
    }
    return jwt.encode(payload, secret, algorithm="HS256")

def require_auth(f):
    """
    Flask route decorator to mandate and verify a valid Bearer token.
    Injects `g.user_id` and `g.auth_user` into request context.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "").strip()
        if not auth_header:
            return jsonify({
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Missing Authorization header. An authenticated session is required."
                }
            }), 401

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_AUTH_HEADER",
                    "message": "Authorization header must be formatted as 'Bearer <token>'."
                }
            }), 401

        token = parts[1]
        try:
            auth_info = verify_token(token)
            g.user_id = auth_info["user_id"]
            g.auth_user = auth_info
        except ValueError as ve:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_TOKEN",
                    "message": str(ve)
                }
            }), 401
        except Exception as e:
            return jsonify({
                "success": False,
                "error": {
                    "code": "AUTH_ERROR",
                    "message": f"Authentication verification error: {str(e)}"
                }
            }), 401

        return f(*args, **kwargs)
    return decorated_function
