"""
Health check endpoint for the API.
Prefix: /health
"""
from fastapi import APIRouter
from starlette.responses import Response

router = APIRouter()


@router.get("")
def health() -> Response:
    """Returns 200 OK if the API is running."""
    return Response(status_code=200)
