import os
from pathlib import Path
from typing import Dict, Any

# Base configuration
BASE_CONFIG: Dict[str, Any] = {
    "APP_NAME": "Redemption Road",
    "DOMAIN": "www.redemption-seven.com",
    "API_VERSION": "v1",
    "DEBUG": False,
    
    # API URLs
    "API_URL": "https://api.redemption-seven.com",
    "FRONTEND_URL": "https://www.redemption-seven.com",
    
    # Security
    "SECRET_KEY": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
    
    # Database
    "DATABASE_URL": "sqlite:///./database/music.db",
    "DB_POOL_SIZE": 20,
    "DB_MAX_OVERFLOW": 10,
    
    # Audio Processing
    "SAMPLE_RATE": 48000,
    "BIT_DEPTH": 32,
    "MAX_DURATION": 600,  # 10 minutes
    "TARGET_LUFS": -14.0,
    "MAX_PEAK": -1.0,
    
    # File Storage
    "AUDIO_DIR": Path("audio"),
    "VOICE_SAMPLES_DIR": Path("audio/voice_samples"),
    "GENERATED_DIR": Path("audio/generated"),
    "MODELS_DIR": Path("models"),
    
    # Streaming Services
    "STREAMING_SERVICES": [
        "Spotify",
        "Apple Music",
        "Amazon Music",
        "YouTube Music",
        "Tidal",
        "Deezer",
        "Pandora"
    ],
    
    # Error Messages
    "ERRORS": {
        "AUTH_FAILED": "Authentication failed",
        "INVALID_TOKEN": "Invalid or expired token",
        "USER_NOT_FOUND": "User not found",
        "LIMIT_EXCEEDED": "Package limit exceeded",
        "INVALID_FILE": "Invalid file format",
        "PROCESSING_ERROR": "Audio processing error",
        "DATABASE_ERROR": "Database operation failed",
        "DISTRIBUTION_ERROR": "Distribution service error"
    },
    
    # Logging
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_FILE": "logs/app.log",
    
    # Cache
    "CACHE_TTL": 3600,  # 1 hour
    "MAX_CACHE_SIZE": 1000,
    
    # Rate Limiting
    "RATE_LIMIT_CALLS": 100,
    "RATE_LIMIT_PERIOD": 60,  # 1 minute
}

# Environment-specific configurations
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

DEVELOPMENT_CONFIG = {
    "DEBUG": True,
    "API_URL": "http://localhost:8000",
    "FRONTEND_URL": "http://localhost:3000",
    "LOG_LEVEL": "DEBUG"
}

PRODUCTION_CONFIG = {
    "DEBUG": False,
    "API_URL": f"https://api.{BASE_CONFIG['DOMAIN']}",
    "FRONTEND_URL": f"https://{BASE_CONFIG['DOMAIN']}",
    "LOG_LEVEL": "WARNING",
    "SECURE_SSL_REDIRECT": True,
    "SESSION_COOKIE_SECURE": True,
    "CSRF_COOKIE_SECURE": True
}

# Load environment-specific config
if ENVIRONMENT == "development":
    CONFIG = {**BASE_CONFIG, **DEVELOPMENT_CONFIG}
else:
    CONFIG = {**BASE_CONFIG, **PRODUCTION_CONFIG}

# Create necessary directories
for directory in [CONFIG["AUDIO_DIR"], CONFIG["VOICE_SAMPLES_DIR"], 
                 CONFIG["GENERATED_DIR"], CONFIG["MODELS_DIR"]]:
    directory.mkdir(parents=True, exist_ok=True)

# Validate configuration
def validate_config():
    """Validate critical configuration settings."""
    required_settings = [
        "SECRET_KEY",
        "DATABASE_URL",
        "API_URL",
        "FRONTEND_URL"
    ]
    
    for setting in required_settings:
        if not CONFIG.get(setting):
            raise ValueError(f"Missing required configuration: {setting}")
        
    # Validate paths
    for path_setting in ["AUDIO_DIR", "VOICE_SAMPLES_DIR", "GENERATED_DIR", "MODELS_DIR"]:
        path = CONFIG.get(path_setting)
        if not path or not isinstance(path, Path):
            raise ValueError(f"Invalid path configuration: {path_setting}")
        
    # Validate numeric values
    if CONFIG["SAMPLE_RATE"] not in [44100, 48000, 96000]:
        raise ValueError("Invalid sample rate configuration")
    
    if CONFIG["BIT_DEPTH"] not in [16, 24, 32]:
        raise ValueError("Invalid bit depth configuration")
    
    if not isinstance(CONFIG["MAX_DURATION"], (int, float)) or CONFIG["MAX_DURATION"] <= 0:
        raise ValueError("Invalid max duration configuration")

# Validate configuration on import
validate_config()
