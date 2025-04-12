from typing import List, Optional
from utils import process_audio, save_audio_file
from audio_utils import (
    synthesize_audio,
    generate_vocal_audio,
    check_subscription_limits,
    VoiceClone,
    AdvancedAudioProcessor,
    VoiceSynthesizer
)

class MusicGenerator:
    def __init__(self, user_package: str):
        self.user_package = user_package
        self.audio_processor = AdvancedAudioProcessor()
        self.voice_synthesizer = VoiceSynthesizer()
        self.current_song_count = 0  # This should be tracked in a database
        self.current_clone_count = 0  # This should be tracked in a database

    def check_limits(self) -> dict:
        """Check if the user has reached their subscription limits."""
        return check_subscription_limits(
            self.user_package,
            self.current_song_count,
            self.current_clone_count
        )

    def generate_music(
        self,
        lyrics: str,
        genre: str,
        mood: str,
        tempo: str,
        instruments: List[str],
        voice_clone: Optional[VoiceClone] = None
    ) -> dict:
        """
        Generate high-quality music with professional mastering and effects.
        """
        # Check subscription limits
        limits = self.check_limits()
        if not limits['can_create_song']:
            raise ValueError(f"Monthly song limit reached for {self.user_package} package")

        # Generate stems for each instrument with high-quality processing
        stems = {}
        for instrument in instruments:
            # Generate and process audio for each instrument
            audio_data = synthesize_audio(instrument, genre, mood)
            processed_audio = self.audio_processor.apply_mastering(
                AudioSegment(audio_data)
            )
            
            # Save the processed audio
            stems[instrument] = save_audio_file(
                f"{instrument}_stem.wav",
                processed_audio.raw_data
            )

        # Generate vocals with optional voice clone and high-quality processing
        vocal_audio_data = generate_vocal_audio(lyrics, voice_clone)
        processed_vocals = self.audio_processor.apply_mastering(
            AudioSegment(vocal_audio_data)
        )
        stems["vocals"] = save_audio_file("vocals_stem.wav", processed_vocals.raw_data)

        # Update song count (should be done in database)
        self.current_song_count += 1

        return {
            "message": "Music generated successfully!",
            "lyrics": lyrics,
            "genre": genre,
            "mood": mood,
            "tempo": tempo,
            "stems": stems,
            "package_info": {
                "package": self.user_package,
                "remaining_songs": limits['remaining_songs'],
                "remaining_clones": limits['remaining_clones']
            }
        }

    def create_voice_clone(self, name: str, voice_data: bytes) -> VoiceClone:
        """
        Create a new high-quality voice clone if within subscription limits.
        """
        limits = self.check_limits()
        if not limits['can_create_clone']:
            raise ValueError(f"Voice clone limit reached for {self.user_package} package")

        # Create a new voice clone with enhanced quality
        voice_id = f"voice_{name}_{self.current_clone_count}"
        voice_clone = VoiceClone(voice_id, name)

        # Update clone count (should be done in database)
        self.current_clone_count += 1

        return voice_clone

def generate_music_endpoint(
    lyrics: str,
    genre: str,
    mood: str,
    tempo: str,
    instruments: List[str],
    user_package: str,
    voice_clone_id: Optional[str] = None
) -> dict:
    """
    Endpoint handler for music generation requests.
    Ensures high-quality output with professional mastering.
    """
    generator = MusicGenerator(user_package)
    
    # Get voice clone if specified
    voice_clone = None
    if voice_clone_id:
        # Here you would retrieve the voice clone from a database
        # For now, we'll create a placeholder
        voice_clone = VoiceClone(voice_clone_id, "Custom Voice")

    return generator.generate_music(
        lyrics=lyrics,
        genre=genre,
        mood=mood,
        tempo=tempo,
        instruments=instruments,
        voice_clone=voice_clone
    )

# Package song limits
PACKAGE_SONG_LIMITS = {
    'bronze': 10,
    'silver': 20,
    'gold': 30,
    'platinum': 40,
    'doublePlatinum': float('inf')  # Unlimited songs
}

# Package voice clone limits
PACKAGE_CLONE_LIMITS = {
    'bronze': 5,
    'silver': 10,
    'gold': 15,
    'platinum': 20,
    'doublePlatinum': 25
}
