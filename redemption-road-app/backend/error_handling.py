from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
from logging.handlers import RotatingFileHandler
import traceback
import sys
from datetime import datetime
from pathlib import Path

from config import CONFIG

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Main application logger
logger = logging.getLogger("redemption-road")
logger.setLevel(getattr(logging, CONFIG["LOG_LEVEL"]))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(CONFIG["LOG_FORMAT"]))
logger.addHandler(console_handler)

# File handler with rotation
file_handler = RotatingFileHandler(
    CONFIG["LOG_FILE"],
    maxBytes=10_000_000,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(CONFIG["LOG_FORMAT"]))
logger.addHandler(file_handler)

class AppError(Exception):
    """Base exception for application-specific errors."""
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class AudioProcessingError(AppError):
    """Raised when audio processing fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

class ValidationError(AppError):
    """Raised when input validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )

class AuthenticationError(AppError):
    """Raised when authentication fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )

class AuthorizationError(AppError):
    """Raised when user lacks required permissions."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )

class ResourceNotFoundError(AppError):
    """Raised when requested resource is not found."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )

class RateLimitError(AppError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )

class DatabaseError(AppError):
    """Raised when database operations fail."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

class DistributionError(AppError):
    """Raised when music distribution fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DISTRIBUTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler for all exceptions."""
    timestamp = datetime.utcnow().isoformat()
    
    if isinstance(exc, AppError):
        logger.error(
            f"Application error: {exc.error_code} - {exc.message}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "details": exc.details,
                "path": request.url.path,
                "timestamp": timestamp
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                    "timestamp": timestamp
                }
            }
        )
    
    # Handle unexpected errors
    error_id = f"ERR-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    logger.error(
        f"Unexpected error: {error_id}",
        exc_info=True,
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "timestamp": timestamp
        }
    )
    
    if CONFIG["DEBUG"]:
        error_details = {
            "traceback": traceback.format_exc(),
            "error_id": error_id
        }
    else:
        error_details = {
            "error_id": error_id
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": error_details,
                "timestamp": timestamp
            }
        }
    )

def setup_error_handlers(app):
    """Configure error handlers for the FastAPI application."""
    app.add_exception_handler(Exception, error_handler)
    app.add_exception_handler(AppError, error_handler)
    app.add_exception_handler(HTTPException, error_handler)

def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error"
) -> None:
    """Centralized error logging function."""
    log_func = getattr(logger, level)
    
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if context:
        error_data["context"] = context
    
    if isinstance(error, AppError):
        error_data.update({
            "error_code": error.error_code,
            "status_code": error.status_code,
            "details": error.details
        })
    
    log_func(
        f"Error occurred: {error_data['error_type']} - {error_data['error_message']}",
        extra=error_data,
        exc_info=True
    )
