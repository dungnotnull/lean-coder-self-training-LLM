"""FastAPI application for model serving."""

from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request for text generation."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2


class GenerateResponse(BaseModel):
    """Response from text generation."""

    text: str
    tokens_generated: int


def create_app(server) -> FastAPI:
    """Create FastAPI application.

    Args:
        server: InferenceServer instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="LeanCoder API",
        description="Lightweight coding-focused LLM API",
        version="0.0.1",
    )

    @app.get("/")
    def read_root():
        """Root endpoint."""
        return {"message": "LeanCoder API", "status": "running"}

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate text completion.

        Args:
            request: Generation request

        Returns:
            Generated response
        """
        response = server.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return GenerateResponse(text=response, tokens_generated=len(response.split()))

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
