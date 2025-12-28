"""
JWT token validation for FastAPI.
Validates tokens issued by NextAuth.js (Next.js).
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from api.config import settings
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_token_from_request(request: Request) -> Optional[str]:
    """Extract JWT token from request (header or cookie)."""
    # Try Authorization header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    
    # Try cookie (for NextAuth.js cookie-based sessions)
    cookie_token = request.cookies.get("next-auth.session-token")
    if cookie_token:
        return cookie_token
    
    return None


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.
    
    Returns:
        Decoded token payload with user info
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not settings.auth_secret:
        logger.warning("AUTH_SECRET not set, JWT validation will fail")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )
    
    try:
        # Decode and verify token
        payload = jwt.decode(
            token,
            settings.auth_secret,
            algorithms=["HS256"]
        )
        
        # Extract user information
        user_id = payload.get("id")
        user_email = payload.get("email")
        user_type = payload.get("type", "regular")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )
        
        return {
            "id": user_id,
            "email": user_email,
            "type": user_type,
            "payload": payload
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from JWT token.
    
    Usage:
        @router.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            user_id = user["id"]
    """
    token = await get_token_from_request(request)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return verify_jwt_token(token)


async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current user if authenticated, None otherwise.
    Useful for endpoints that work for both authenticated and anonymous users.
    """
    token = await get_token_from_request(request)
    
    if not token:
        return None
    
    try:
        return verify_jwt_token(token)
    except HTTPException:
        return None

