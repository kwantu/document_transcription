from fastapi import FastAPI
from app.api.routes import router
from app.middleware.auth import AuthMiddleware

app = FastAPI(
    title="ID Processing API",
    version="1.1.0"
)

# 🔐 Runtime enforcement (NOT Swagger)
app.add_middleware(AuthMiddleware)

# 📦 API routes
app.include_router(router, prefix="/api/v1")
