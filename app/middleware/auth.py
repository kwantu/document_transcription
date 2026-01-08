from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.security.api_key import validate_api_key
from app.logging.logger import log_request

# 🔐 Only this endpoint is protected
PROTECTED_ENDPOINTS = {
    ("POST", "/api/v1/documents/extract"),
}

class AuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        requires_auth = (method, path) in PROTECTED_ENDPOINTS

        api_key_id = None

        if requires_auth:
            api_key = request.headers.get("X-API-Key")

            if not api_key:
                log_request(None, path, 401)
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing API key"}
                )

            try:
                api_key_id = validate_api_key(api_key)
                request.state.api_key_id = api_key_id
            except Exception as e:
                log_request(None, path, 401)
                return JSONResponse(
                    status_code=401,
                    content={"detail": str(e)}
                )

        response = await call_next(request)

        if requires_auth:
            log_request(
                api_key_id=api_key_id,
                endpoint=path,
                status_code=response.status_code
            )

        return response
