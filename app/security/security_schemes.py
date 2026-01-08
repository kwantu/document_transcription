from fastapi.security import APIKeyHeader

api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)
