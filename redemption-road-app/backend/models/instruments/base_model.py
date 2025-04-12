import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import torchaudio
import torchaudio.transforms as T

class InstrumentSynthesizer(nn.Module):
    def __init__(
        self,
        sample_rate: int = 48000,
        n_harmonics: int = 16,
        noise_level: float = 0.003,
        attack_time: float = 0.01,
        decay_time: float = 0.1,
        sustain_level: float = 0.7,
        release_time: float = 0.3
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.noise_level = noise_level
        
        # ADSR envelope parameters
        self.attack_time = attack_time
        self.decay_time = decay_time
        self.sustain_level = sustain_level
        self.release_time = release_time
        
        # Initialize harmonic oscillators
        self.harmonic_weights = nn.Parameter(
            torch.tensor([1.0 / (i + 1) for i in range(n_harmonics)])
        )
        
        # Initialize filters
        self.lowpass = T.Lowpass(
            cutoff_freq=8000,
            sample_rate=sample_rate
        )
        
        self.highpass = T.Highpass(
            cutoff_freq=20,
            sample_rate=sample_rate
        )

    def generate_sine(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 1.0
    ) -> torch.Tensor:
        """Generate a sine wave with given frequency and duration."""
        t = torch.linspace(
            0,
            duration,
            int(self.sample_rate * duration)
        )
        
        signal = torch.zeros_like(t)
        for i in range(self.n_harmonics):
            harmonic_freq = frequency * (i + 1)
            if harmonic_freq < self.sample_rate / 2:  # Prevent aliasing
                signal += self.harmonic_weights[i] * torch.sin(
                    2 * np.pi * harmonic_freq * t
                )
        
        return amplitude * signal / torch.max(torch.abs(signal))

    def apply_envelope(
        self,
        signal: torch.Tensor,
        duration: float
    ) -> torch.Tensor:
        """Apply ADSR envelope to the signal."""
        samples = len(signal)
        
        # Convert times to samples
        attack_samples = int(self.attack_time * self.sample_rate)
        decay_samples = int(self.decay_time * self.sample_rate)
        release_samples = int(self.release_time * self.sample_rate)
        
        # Create envelope
        envelope = torch.zeros(samples)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            decay_end = attack_samples + decay_samples
            envelope[attack_samples:decay_end] = torch.linspace(
                1,
                self.sustain_level,
                decay_samples
            )
        
        # Sustain phase
        sustain_samples = samples - release_samples - decay_end
        if sustain_samples > 0:
            envelope[decay_end:-release_samples] = self.sustain_level
        
        # Release phase
        if release_samples > 0:
            envelope[-release_samples:] = torch.linspace(
                self.sustain_level,
                0,
                release_samples
            )
        
        return signal * envelope

    def add_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Add controlled noise to the signal."""
        noise = torch.randn_like(signal) * self.noise_level
        return signal + noise

    def apply_filters(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply audio filters to the signal."""
        # Apply lowpass to remove high frequency artifacts
        signal = self.lowpass(signal)
        
        # Apply highpass to remove DC offset and sub-bass rumble
        signal = self.highpass(signal)
        
        return signal

    def synthesize(
        self,
        midi_sequence: np.ndarray,
        duration: float
    ) -> torch.Tensor:
        """Synthesize audio from MIDI sequence."""
        output_signal = torch.zeros(int(self.sample_rate * duration))
        
        for note in midi_sequence:
            if note > 0:  # Skip rests (note = 0)
                # Convert MIDI note to frequency
                frequency = 440 * (2 ** ((note - 69) / 12))
                
                # Generate basic tone
                signal = self.generate_sine(frequency, duration)
                
                # Apply envelope
                signal = self.apply_envelope(signal, duration)
                
                # Add noise for realism
                signal = self.add_noise(signal)
                
                # Apply filters
                signal = self.apply_filters(signal)
                
                # Mix into output
                output_signal += signal
        
        # Normalize final output
        output_signal = output_signal / torch.max(torch.abs(output_signal))
        
        return output_signal

    def forward(
        self,
        midi_sequence: torch.Tensor,
        duration: float
    ) -> torch.Tensor:
        """Forward pass for the synthesizer."""
        return self.synthesize(midi_sequence, duration)
