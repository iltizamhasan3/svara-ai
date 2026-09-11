from typing import Annotated
from uuid import UUID

from fastapi import Header, HTTPException, status


def current_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> UUID:
    """Development auth stub; production will validate a Supabase JWT instead."""
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_authentication", "message": "X-User-ID is required in development"},
        )
    try:
        return UUID(x_user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_authentication", "message": "X-User-ID must be a UUID"},
        ) from exc
