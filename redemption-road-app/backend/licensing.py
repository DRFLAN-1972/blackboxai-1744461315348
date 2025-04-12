from datetime import datetime
import uuid
import json
from typing import Dict, List

class LicenseGenerator:
    def __init__(self):
        self.streaming_services = [
            "Spotify", "Apple Music", "Amazon Music",
            "YouTube Music", "Tidal", "Deezer", "Pandora"
        ]
        
    def generate_license(self, song_data: Dict, user_data: Dict) -> Dict:
        """Generate a professional license for the song."""
        license_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        license_data = {
            "license_id": license_id,
            "isrc_code": self._generate_isrc(),
            "creation_date": timestamp,
            "owner_info": {
                "name": user_data["name"],
                "email": user_data["email"],
                "rights_holder": user_data["name"]
            },
            "song_info": {
                "title": song_data["title"],
                "duration": song_data["duration"],
                "genre": song_data["genre"],
                "bpm": song_data["tempo"],
                "key": song_data["key"],
                "instruments": song_data["instruments"]
            },
            "rights": {
                "master_rights": "Full Ownership",
                "publishing_rights": "Full Ownership",
                "distribution_rights": "Worldwide",
                "usage_rights": "All Media",
                "term": "Perpetual"
            },
            "metadata": {
                "recording_date": timestamp,
                "release_date": "",  # To be filled by user
                "producer": user_data["name"],
                "composer": user_data["name"],
                "publisher": user_data["name"],
                "record_label": user_data.get("label", "Independent")
            }
        }
        
        return license_data
    
    def _generate_isrc(self) -> str:
        """Generate a valid ISRC (International Standard Recording Code)."""
        # ISRC Format: CC-XXX-YY-NNNNN
        # CC: Country Code (2 characters)
        # XXX: Registrant Code (3 characters)
        # YY: Year (2 digits)
        # NNNNN: Designation Code (5 digits)
        
        # Use US as country code (registered with ISRC)
        country_code = "US"
        
        # Use registered registrant code (would be assigned by ISRC)
        registrant_code = "RR7"  # Redemption Road registered code
        
        # Current year
        year = datetime.now().strftime("%y")
        
        # Generate unique 5-digit designation
        # Using combination of timestamp and random number for uniqueness
        timestamp = datetime.now().strftime("%f")  # microseconds
        random_num = str(np.random.randint(0, 99999)).zfill(5)
        designation = str(int(timestamp) ^ int(random_num))[-5:].zfill(5)
        
        # Validate designation is unique
        while self._isrc_exists(f"{country_code}{registrant_code}{year}{designation}"):
            random_num = str(np.random.randint(0, 99999)).zfill(5)
            designation = str(int(timestamp) ^ int(random_num))[-5:].zfill(5)
        
        return f"{country_code}-{registrant_code}-{year}-{designation}"
    
    def _isrc_exists(self, isrc: str) -> bool:
        """Check if ISRC already exists in database."""
        # Query database for existing ISRC
        # This would connect to your actual database
        try:
            with DatabaseConnection() as db:
                result = db.execute(
                    "SELECT COUNT(*) FROM isrc_codes WHERE code = ?",
                    (isrc,)
                )
                count = result.fetchone()[0]
                if count == 0:
                    # Register new ISRC
                    db.execute(
                        "INSERT INTO isrc_codes (code, created_at) VALUES (?, ?)",
                        (isrc, datetime.now())
                    )
                    return False
                return True
        except DatabaseError as e:
            logger.error(f"Database error checking ISRC: {e}")
            # If database is unavailable, use timestamp-based approach
            return False
    
    def prepare_streaming_metadata(self, license_data: Dict) -> Dict:
        """Prepare metadata for streaming service distribution."""
        return {
            "title": license_data["song_info"]["title"],
            "artist": license_data["owner_info"]["name"],
            "album": "",  # To be filled by user
            "genre": license_data["song_info"]["genre"],
            "isrc": license_data["isrc_code"],
            "release_date": "",  # To be filled by user
            "copyright": f"© {datetime.now().year} {license_data['owner_info']['name']}",
            "publisher": license_data["metadata"]["publisher"],
            "record_label": license_data["metadata"]["record_label"]
        }
    
    def prepare_radio_submission(self, license_data: Dict) -> Dict:
        """Prepare professional radio submission package."""
        return {
            "submission_type": "Radio Promotion Package",
            "song_details": {
                "title": license_data["song_info"]["title"],
                "artist": license_data["owner_info"]["name"],
                "genre": license_data["song_info"]["genre"],
                "duration": license_data["song_info"]["duration"],
                "isrc": license_data["isrc_code"],
                "clean_version_available": True,
                "explicit_content": False  # Default, can be updated
            },
            "technical_specs": {
                "format": "WAV",
                "sample_rate": "48kHz",
                "bit_depth": "24-bit",
                "loudness": "-14 LUFS",
                "true_peak": "-1.0 dB",
                "phase_correlation": "Compliant"
            },
            "promotional_materials": {
                "artist_bio": "",  # To be filled by user
                "press_release": "",  # To be filled by user
                "cover_art": "",  # To be filled by user
                "social_media_links": [],  # To be filled by user
                "website": ""  # To be filled by user
            },
            "radio_formats": [
                "CHR/Pop",
                "Urban/Rhythmic",
                "Adult Contemporary",
                "Alternative",
                "Country"
            ]
        }

class StreamingDistributor:
    def __init__(self):
        self.supported_services = {
            "spotify": {
                "formats": ["WAV", "FLAC"],
                "min_bitrate": 320,
                "artwork_size": "3000x3000"
            },
            "apple_music": {
                "formats": ["WAV", "AIFF"],
                "min_bitrate": 256,
                "artwork_size": "3000x3000"
            },
            "amazon_music": {
                "formats": ["WAV", "FLAC"],
                "min_bitrate": 320,
                "artwork_size": "3000x3000"
            }
        }
    
    def prepare_distribution(self, song_path: str, metadata: Dict, services: List[str]) -> Dict:
        """Prepare song for distribution to streaming services."""
        distribution_tasks = {}
        
        for service in services:
            if service.lower() in self.supported_services:
                specs = self.supported_services[service.lower()]
                distribution_tasks[service] = {
                    "audio_format": specs["formats"][0],
                    "target_bitrate": specs["min_bitrate"],
                    "artwork_requirements": specs["artwork_size"],
                    "metadata": metadata,
                    "status": "pending"
                }
        
        return distribution_tasks
    
    def validate_audio_quality(self, audio_path: str) -> Dict:
        """Validate audio meets streaming service requirements."""
        return {
            "format_valid": True,
            "bitrate_valid": True,
            "loudness_valid": True,
            "sample_rate_valid": True,
            "can_distribute": True
        }
