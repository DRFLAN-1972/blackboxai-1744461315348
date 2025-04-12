from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
import jwt
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

from database import DatabaseConnection, DatabaseError
from enhanced_music_gen import EnhancedMusicGenerator
from licensing import LicenseGenerator, StreamingDistributor
from models.instruments.piano import PianoSynthesizer

from config import CONFIG
from error_handling import setup_error_handlers, log_error, AppError

# Initialize FastAPI app
app = FastAPI(
    title="Redemption Road Music Generation API",
    description="Professional music generation with distribution and licensing",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CONFIG["FRONTEND_URL"]],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup error handlers
setup_error_handlers(app)

# Initialize logger
logger = logging.getLogger("redemption-road")

# Security configuration
SECRET_KEY = CONFIG["SECRET_KEY"]
ALGORITHM = CONFIG["ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = CONFIG["ACCESS_TOKEN_EXPIRE_MINUTES"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str
    package: str

class User(UserBase):
    id: int
    package: str
    songs_generated: int
    clones_created: int
    subscription_start: datetime
    subscription_end: datetime
    status: str

class Token(BaseModel):
    access_token: str
    token_type: str

class MusicGenerationRequest(BaseModel):
    title: str
    lyrics: Optional[str]
    genre: str
    mood: str
    tempo: int
    key: str
    instruments: List[str]
    voice_clone_id: Optional[str]
    distribution_services: Optional[List[str]]

class RadioSubmissionRequest(BaseModel):
    song_id: int
    formats: List[str]
    promotional_materials: Dict

# Authentication functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise AuthenticationError(CONFIG["ERRORS"]["INVALID_TOKEN"])
    except jwt.JWTError as e:
        log_error(e, {"token": token})
        raise AuthenticationError(CONFIG["ERRORS"]["AUTH_FAILED"])
    
    with DatabaseConnection() as db:
        result = db.fetch_one(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return User(**dict(zip(
            ['id', 'email', 'name', 'package', 'songs_generated', 'clones_created',
             'subscription_start', 'subscription_end', 'status'],
            result
        )))

# API endpoints
@app.post("/api/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        with DatabaseConnection() as db:
            result = db.fetch_one(
                "SELECT * FROM users WHERE email = ?",
                (form_data.username,)
            )
            if not result:
                raise AuthenticationError(CONFIG["ERRORS"]["AUTH_FAILED"])
            
            # Verify password using bcrypt
            if not verify_password(form_data.password, result[2]):
                raise AuthenticationError(CONFIG["ERRORS"]["AUTH_FAILED"])
            
            access_token = create_access_token(
                data={"sub": result[0]}  # user_id
            )
            return Token(access_token=access_token, token_type="bearer")
    except DatabaseError as e:
        log_error(e, {"email": form_data.username})
        raise
        
        access_token = create_access_token(
            data={"sub": result[0]}  # user_id
        )
        return Token(access_token=access_token, token_type="bearer")

@app.post("/users")
async def create_user(user: UserCreate):
    with DatabaseConnection() as db:
        try:
            # Check if email already exists
            existing = db.fetch_one(
                "SELECT id FROM users WHERE email = ?",
                (user.email,)
            )
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Create new user
            now = datetime.utcnow()
            db.execute(
                """
                INSERT INTO users (
                    email, name, password, package,
                    subscription_start, subscription_end, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.email,
                    user.name,
                    hash_password(user.password),
                    user.package,
                    now,
                    now + timedelta(days=30),
                    'active'
                )
            )
            return {"message": "User created successfully"}
        except DatabaseError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

@app.post("/api/generate-music")
async def generate_music(
    request: MusicGenerationRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Initialize music generator
        generator = EnhancedMusicGenerator(
            current_user.package,
            {
                "name": current_user.name,
                "email": current_user.email
            }
        )
        
        # Generate music
        result = generator.generate_music(
            title=request.title,
            lyrics=request.lyrics,
            genre=request.genre,
            mood=request.mood,
            tempo=request.tempo,
            key=request.key,
            instruments=request.instruments,
            voice_clone=request.voice_clone_id,
            distribution_services=request.distribution_services
        )
        
        # Update user's song count
        with DatabaseConnection() as db:
            db.execute(
                "UPDATE users SET songs_generated = songs_generated + 1 WHERE id = ?",
                (current_user.id,)
            )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/submit-to-radio")
async def submit_to_radio(
    request: RadioSubmissionRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Check if package allows radio submission
        if current_user.package not in ['silver', 'gold', 'platinum', 'doublePlatinum']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Radio submission not available in current package"
            )
        
        # Get song details
        with DatabaseConnection() as db:
            song = db.fetch_one(
                "SELECT * FROM songs WHERE id = ? AND artist_id = ?",
                (request.song_id, current_user.id)
            )
            if not song:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Song not found"
                )
        
        # Initialize music generator
        generator = EnhancedMusicGenerator(
            current_user.package,
            {
                "name": current_user.name,
                "email": current_user.email
            }
        )
        
        # Prepare radio submission
        submission = generator.prepare_radio_submission({
            "song_id": song[0],
            "title": song[1],
            "formats": request.formats,
            "promotional_materials": request.promotional_materials
        })
        
        # Record submission
        with DatabaseConnection() as db:
            db.execute(
                """
                INSERT INTO radio_submissions (
                    song_id, submission_date, format, status
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    song[0],
                    datetime.utcnow(),
                    ','.join(request.formats),
                    'pending'
                )
            )
        
        return submission
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/upload-voice-sample")
async def upload_voice_sample(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    try:
        # Check if user has reached clone limit
        if current_user.clones_created >= get_clone_limit(current_user.package):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Voice clone limit reached for current package"
            )
        
        # Save voice sample
        file_path = f"audio/voice_samples/{current_user.id}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create voice clone
        generator = EnhancedMusicGenerator(
            current_user.package,
            {
                "name": current_user.name,
                "email": current_user.email
            }
        )
        
        voice_clone = generator.create_voice_clone(
            name=file.filename,
            voice_data=content
        )
        
        # Update user's clone count
        with DatabaseConnection() as db:
            db.execute(
                "UPDATE users SET clones_created = clones_created + 1 WHERE id = ?",
                (current_user.id,)
            )
        
        return {
            "message": "Voice sample uploaded successfully",
            "voice_clone_id": voice_clone.id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def get_clone_limit(package: str) -> int:
    limits = {
        'bronze': 5,
        'silver': 10,
        'gold': 15,
        'platinum': 20,
        'doublePlatinum': 25
    }
    return limits.get(package, 0)

# Helper functions for password hashing (implement proper hashing)
def hash_password(password: str) -> str:
    # Implement proper password hashing (e.g., using bcrypt)
    return password

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Implement proper password verification
    return plain_password == hashed_password

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
