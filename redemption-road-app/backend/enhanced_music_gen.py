from typing import List, Optional, Dict
from enhanced_audio_utils import AdvancedAudioProcessor, VoiceSynthesizer
from licensing import LicenseGenerator, StreamingDistributor
from pydub import AudioSegment
import numpy as np
import json
import os

class EnhancedMusicGenerator:
    def __init__(self, user_package: str, user_data: Dict):
        self.user_package = user_package
        self.user_data = user_data
        self.audio_processor = AdvancedAudioProcessor()
        self.voice_synthesizer = VoiceSynthesizer()
        self.license_generator = LicenseGenerator()
        self.streaming_distributor = StreamingDistributor()
        self.current_song_count = 0
        self.current_clone_count = 0

        # Updated package limits with distribution costs
        self.package_limits = {
            'bronze': {
                'songs': 10,
                'clones': 5,
                'base_price': 89.99,
                'features': ['Unlimited Distribution']
            },
            'silver': {
                'songs': 20,
                'clones': 10,
                'base_price': 119.99,
                'features': ['Unlimited Distribution', 'Radio Ready']
            },
            'gold': {
                'songs': 30,
                'clones': 15,
                'base_price': 149.99,
                'features': ['Unlimited Distribution', 'Radio Ready', 'Priority Support']
            },
            'platinum': {
                'songs': 40,
                'clones': 20,
                'base_price': 269.99,
                'features': ['Unlimited Distribution', 'Radio Campaign', 'Priority Support']
            },
            'doublePlatinum': {
                'songs': float('inf'),
                'clones': 25,
                'base_price': 699.99,
                'features': ['Unlimited Distribution', 'Radio Campaign', 'Priority Support', 'Label Services']
            }
        }

    def generate_music(
        self,
        title: str,
        lyrics: str,
        genre: str,
        mood: str,
        tempo: int,
        key: str,
        instruments: List[str],
        voice_clone: Optional[Dict] = None,
        distribution_services: List[str] = None
    ) -> Dict:
        """Generate professional-grade music with licensing and distribution."""
        
        # Check subscription limits
        if not self._check_limits():
            raise ValueError(f"Monthly song limit reached for {self.user_package} package")

        # Generate and process stems
        stems = {}
        for instrument in instruments:
            # Generate instrument audio with enhanced quality
            raw_audio = self._generate_instrument(
                instrument=instrument,
                genre=genre,
                mood=mood,
                tempo=tempo,
                key=key
            )
            
            # Apply professional processing
            processed_audio = self.audio_processor.apply_mastering(raw_audio)
            stems[instrument] = processed_audio

        # Generate and process vocals
        if lyrics:
            vocal_audio = self.voice_synthesizer.synthesize_voice(lyrics, voice_clone)
            processed_vocals = self.audio_processor.apply_mastering(
                AudioSegment(vocal_audio)
            )
            stems['vocals'] = processed_vocals

        # Mix stems
        final_mix = self._mix_stems(stems)

        # Generate license and metadata
        song_data = {
            'title': title,
            'duration': len(final_mix) / 1000.0,  # Convert to seconds
            'genre': genre,
            'tempo': tempo,
            'key': key,
            'instruments': instruments
        }
        
        license_data = self.license_generator.generate_license(song_data, self.user_data)
        streaming_metadata = self.license_generator.prepare_streaming_metadata(license_data)
        
        # Prepare radio submission if applicable
        radio_submission = None
        if self.package_limits[self.user_package]['base_price'] >= 89.99:  # Silver and above
            radio_submission = self.license_generator.prepare_radio_submission(license_data)

        # Save final audio with high quality
        output_path = f"audio/final/{title.replace(' ', '_')}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_mix.export(
            output_path,
            format='wav',
            parameters=[
                '-ar', str(self.audio_processor.sample_rate),
                '-bits_per_raw_sample', str(self.audio_processor.bit_depth)
            ]
        )

        # Prepare distribution if requested
        distribution_info = None
        if distribution_services:
            distribution_info = self.streaming_distributor.prepare_distribution(
                output_path,
                streaming_metadata,
                distribution_services
            )

        # Update song count
        self.current_song_count += 1

        return {
            'status': 'success',
            'message': 'Music generated successfully',
            'output_file': output_path,
            'license': license_data,
            'streaming_metadata': streaming_metadata,
            'radio_submission': radio_submission,
            'distribution': distribution_info,
            'package_info': {
                'package': self.user_package,
                'remaining_songs': self._get_remaining_songs(),
                'remaining_clones': self._get_remaining_clones()
            }
        }

    def _generate_instrument(
        self,
        instrument: str,
        genre: str,
        mood: str,
        tempo: int,
        key: str
    ) -> AudioSegment:
        """Generate realistic instrument audio using advanced synthesis."""
        # Initialize synthesis parameters
        sample_rate = self.audio_processor.sample_rate
        duration = tempo * 4  # Calculate duration based on tempo
        
        # Load instrument model based on type
        instrument_model = self._load_instrument_model(instrument)
        
        # Generate MIDI sequence based on genre and mood
        midi_sequence = self._generate_midi_sequence(
            genre=genre,
            mood=mood,
            tempo=tempo,
            key=key
        )
        
        # Synthesize audio using the instrument model
        raw_audio = instrument_model.synthesize(
            midi_sequence,
            sample_rate=sample_rate,
            duration=duration
        )
        
        # Convert to AudioSegment
        return AudioSegment(
            raw_audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=4,  # 32-bit
            channels=2  # Stereo
        )

    def _load_instrument_model(self, instrument: str) -> Any:
        """Load the appropriate instrument synthesis model."""
        model_path = f"models/instruments/{instrument.lower()}"
        return torch.jit.load(model_path)

    def _generate_midi_sequence(
        self,
        genre: str,
        mood: str,
        tempo: int,
        key: str
    ) -> np.ndarray:
        """Generate MIDI sequence using music theory and genre patterns."""
        # Load genre-specific patterns
        patterns = self._load_genre_patterns(genre)
        
        # Generate chord progression based on key and mood
        chord_progression = self._generate_chord_progression(key, mood)
        
        # Create melody based on chord progression and patterns
        melody = self._create_melody(chord_progression, patterns, mood)
        
        # Combine into final MIDI sequence
        sequence = np.concatenate([
            chord_progression,
            melody
        ])
        
        return sequence

    def _load_genre_patterns(self, genre: str) -> Dict:
        """Load pre-defined musical patterns for specific genre."""
        pattern_path = f"models/patterns/{genre.lower()}.json"
        with open(pattern_path, 'r') as f:
            return json.load(f)

    def _generate_chord_progression(self, key: str, mood: str) -> np.ndarray:
        """Generate musically valid chord progression."""
        # Define chord progressions for different moods
        progressions = {
            'happy': ['I', 'IV', 'V', 'I'],
            'sad': ['i', 'VI', 'III', 'VII'],
            'energetic': ['I', 'V', 'vi', 'IV'],
            'relaxed': ['ii', 'V', 'I', 'vi']
        }
        
        # Get base progression for mood
        base_progression = progressions.get(mood.lower(), ['I', 'IV', 'V', 'I'])
        
        # Convert to MIDI notes based on key
        return self._progression_to_midi(base_progression, key)

    def _progression_to_midi(self, progression: List[str], key: str) -> np.ndarray:
        """Convert chord progression to MIDI notes."""
        # Define key mappings
        key_map = {
            'C': 60, 'C#': 61, 'D': 62, 'D#': 63,
            'E': 64, 'F': 65, 'F#': 66, 'G': 67,
            'G#': 68, 'A': 69, 'A#': 70, 'B': 71
        }
        
        # Get root note from key
        root = key_map[key.split('m')[0]]
        is_minor = 'm' in key
        
        # Convert progression to MIDI notes
        midi_sequence = []
        for chord in progression:
            chord_notes = self._chord_to_notes(chord, root, is_minor)
            midi_sequence.extend(chord_notes)
        
        return np.array(midi_sequence)

    def _chord_to_notes(self, chord: str, root: int, is_minor: bool) -> List[int]:
        """Convert chord symbol to MIDI notes."""
        # Define intervals for different chord types
        major_triad = [0, 4, 7]
        minor_triad = [0, 3, 7]
        diminished_triad = [0, 3, 6]
        
        # Convert Roman numeral to scale degree
        scale_degrees = {
            'I': 0, 'II': 2, 'III': 4, 'IV': 5,
            'V': 7, 'VI': 9, 'VII': 11,
            'i': 0, 'ii': 2, 'iii': 4, 'iv': 5,
            'v': 7, 'vi': 9, 'vii': 11
        }
        
        # Get base note from scale degree
        base = root + scale_degrees[chord.upper()]
        
        # Determine chord type and generate notes
        if chord.isupper():
            intervals = major_triad
        elif chord.islower():
            intervals = minor_triad if not is_minor else diminished_triad
        else:
            intervals = minor_triad
        
        return [base + interval for interval in intervals]

    def _create_melody(
        self,
        chord_progression: np.ndarray,
        patterns: Dict,
        mood: str
    ) -> np.ndarray:
        """Create melody based on chord progression and patterns."""
        # Get mood-specific parameters
        params = patterns['melody_params'][mood.lower()]
        
        # Generate melody notes following chord progression
        melody = []
        for chord in chord_progression.reshape(-1, 3):
            # Use pattern rules to generate melody
            phrase = self._generate_melody_phrase(
                chord,
                params['rhythm'],
                params['intervals']
            )
            melody.extend(phrase)
        
        return np.array(melody)

    def _generate_melody_phrase(
        self,
        chord: np.ndarray,
        rhythm: List[float],
        intervals: List[int]
    ) -> List[int]:
        """Generate melodic phrase based on chord and parameters."""
        phrase = []
        for beat in rhythm:
            if beat > 0:  # Note
                # Choose note from chord or passing tone
                if np.random.random() > 0.2:  # 80% chord tones
                    note = np.random.choice(chord)
                else:  # 20% passing tones
                    note = chord[0] + np.random.choice(intervals)
                phrase.append(note)
            else:  # Rest
                phrase.append(0)
        
        return phrase

    def _mix_stems(self, stems: Dict[str, AudioSegment]) -> AudioSegment:
        """Professional mixing of all stems."""
        # Start with the longest stem
        final_mix = max(stems.values(), key=len)
        
        # Mix in other stems with proper levels
        for stem_name, stem in stems.items():
            if stem_name != 'vocals':  # Handle vocals separately
                final_mix = final_mix.overlay(stem, position=0)
        
        # Add vocals last if they exist
        if 'vocals' in stems:
            final_mix = final_mix.overlay(stems['vocals'], position=0)
        
        return final_mix

    def _check_limits(self) -> bool:
        """Check if user has reached their package limits."""
        limits = self.package_limits[self.user_package]
        return (
            self.current_song_count < limits['songs'] or 
            limits['songs'] == float('inf')
        )

    def _get_remaining_songs(self) -> int:
        """Get remaining songs in current package."""
        limits = self.package_limits[self.user_package]
        if limits['songs'] == float('inf'):
            return float('inf')
        return limits['songs'] - self.current_song_count

    def _get_remaining_clones(self) -> int:
        """Get remaining voice clones in current package."""
        limits = self.package_limits[self.user_package]
        return limits['clones'] - self.current_clone_count

    def prepare_radio_submission(self, song_data: Dict) -> Dict:
        """Prepare professional radio submission package."""
        if self.package_limits[self.user_package]['base_price'] < 89.99:
            raise ValueError("Radio submission requires Silver package or higher")
            
        return self.license_generator.prepare_radio_submission(song_data)
