"""
MonuScript - Authentication & Token Verification Module
Validates Supabase Auth access tokens (including anonymous sessions)
and persistent anonymous visitor tokens, binding the authenticated user ID
to the request context.
"""

import os
import re
import time
import functools
import requests
import jwt
from flask import request, jsonify, g
from config import Config

# In-memory cache for token verification: {token: (auth_dict, exp_time)}
_AUTH_CACHE = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Safe regex pattern for visitor identifiers (UUID, hex, or anon_ prefixes)
_SAFE_VISITOR_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def verify_token(token: str) -> dict:
    """
    Verifies an authentication token and returns claims dictionary.
    Supports:
    1. Supabase JWT (HS256 with JWT secret or remote API check)
    2. Locally signed JWT
    3. Anonymous visitor tokens (e.g. anon_uuid, visitor_uuid)
    """
    if not token or not isinstance(token, str):
        raise ValueError("Missing or invalid token string.")

    token = token.strip()
    now = time.time()

    # 1. Check in-memory cache first to reduce latency
    if token in _AUTH_CACHE:
        cached_user, exp = _AUTH_CACHE[token]
        if now < exp:
            return cached_user
        else:
            del _AUTH_CACHE[token]

    # 2. Check if token is a standard JWT (3 base64url segments separated by dots)
    is_jwt = len(token.split(".")) == 3

    if is_jwt:
        # Local verification using SUPABASE_JWT_SECRET if configured
        if Config.SUPABASE_JWT_SECRET:
            try:
                payload = jwt.decode(
                    token,
                    Config.SUPABASE_JWT_SECRET,
                    algorithms=["HS256", "RS256", "ES256"],
                    audience="authenticated",
                    options={"verify_aud": True, "verify_exp": True}
                )
                user_id = payload.get("sub")
                if not user_id:
                    raise ValueError("Token missing 'sub' claim.")

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
            except jwt.InvalidTokenError:
                pass  # Fall through to remote check or secret_key check

        # Remote verification via Supabase Auth API endpoint if URL & KEY configured
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
                    if user_id:
                        result = {
                            "user_id": str(user_id),
                            "email": user_data.get("email"),
                            "role": user_data.get("role", "authenticated"),
                            "is_anonymous": user_data.get("is_anonymous", False),
                            "claims": user_data
                        }
                        _AUTH_CACHE[token] = (result, now + _CACHE_TTL_SECONDS)
                        return result
            except requests.RequestException:
                pass

        # Fallback verification with local SECRET_KEY
        try:
            payload = jwt.decode(
                token,
                Config.SECRET_KEY,
                algorithms=["HS256"],
                options={"verify_exp": True}
            )
            user_id = payload.get("sub") or payload.get("user_id")
            if user_id:
                return {
                    "user_id": str(user_id),
                    "email": payload.get("email"),
                    "role": "authenticated",
                    "is_anonymous": payload.get("is_anonymous", True),
                    "claims": payload
                }
        except Exception:
            pass

    # 3. Anonymous visitor token fallback (for browsers where Supabase keys aren't set yet)
    # Allows deterministic per-visitor isolation without breaking if Supabase is offline
    if _SAFE_VISITOR_PATTERN.match(token):
        result = {
            "user_id": token,
            "email": None,
            "role": "authenticated",
            "is_anonymous": True,
            "claims": {"sub": token}
        }
        _AUTH_CACHE[token] = (result, now + _CACHE_TTL_SECONDS)
        return result

    raise ValueError("Invalid authentication token format or signature.")


def create_local_token(user_id: str, is_anonymous: bool = True, expires_in_seconds: int = 86400) -> str:
    """Generates a signed JWT for local tests or fallback environments."""
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
    Flask route decorator to mandate and verify a valid Bearer token or visitor identifier.
    Guarantees that CORS preflight OPTIONS requests pass without authentication,
    and injects `g.user_id` and `g.auth_user` into the Flask request context.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Always allow OPTIONS preflight requests for CORS without checking auth
        if request.method == "OPTIONS":
            return f(*args, **kwargs)

        # 2. Extract token from Authorization header, query parameter, or X-Visitor-Id
        token = None
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            elif len(parts) == 1:
                token = parts[0]

        if not token:
            token = request.headers.get("X-Visitor-Id", "").strip()

        if not token:
            token = request.args.get("token", "").strip()

        # 3. If still missing, check cookie or generate deterministic visitor ID
        if not token:
            # Check visitor cookie if any
            token = request.cookies.get("monuscript_visitor_id")

        if not token:
            return jsonify({
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Missing authentication credentials. An active visitor session is required."
                }
            }), 401

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
