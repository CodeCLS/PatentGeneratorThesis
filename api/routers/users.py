"""
User management endpoints.
"""
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from typing import Optional, List
from api.schemas.users import UserCreate, UserResponse, UserWithPassword, UserEnsure
from api.database.repositories_chat import UserRepository
from api.database.dependencies import get_db_session
from api.auth.jwt import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/users", tags=["users"])


def get_user_repository(session: AsyncSession = Depends(get_db_session)) -> UserRepository:
    """Get user repository."""
    return UserRepository(session)


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    repo: UserRepository = Depends(get_user_repository)
):
    """
    Create a new user.
    Called by Next.js during registration.
    Password is already hashed by Next.js.
    """
    try:
        user = await repo.create(user_data.email, user_data.password)
        return UserResponse(**user)
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=409, detail="User with this email already exists")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[UserResponse])
async def get_users(
    email: Optional[str] = Query(None),
    repo: UserRepository = Depends(get_user_repository),
    current_user: dict = Depends(get_current_user)
):
    """
    Get users.
    - If email query param provided: Get user by email (for Next.js password verification)
    - If no email: Get all users (requires authentication)
    """
    if email:
        # Get user by email (for Next.js login verification)
        user = await repo.get_by_email(email)
        if user:
            # Return with password for Next.js to verify
            return [UserWithPassword(**user)]
        return []
    else:
        # Get all users (requires authentication)
        users = await repo.list_all()
        return [UserResponse(**u) for u in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    repo: UserRepository = Depends(get_user_repository),
    current_user: dict = Depends(get_current_user)
):
    """Get user by ID (requires authentication)."""
    user = await repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(id=user["id"], email=user["email"])


@router.post("/ensure", response_model=UserResponse)
async def ensure_user_exists(
    user_data: UserEnsure,
    repo: UserRepository = Depends(get_user_repository)
):
    """
    Ensure user exists (create if not, return if exists).
    Called by Next.js during OAuth flows.
    """
    try:
        user = await repo.ensure_exists(user_data.email, user_data.password)
        return UserResponse(**user)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

