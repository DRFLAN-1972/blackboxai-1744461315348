import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
import torchaudio.transforms as T
from .base_model import InstrumentSynthesizer

class PianoSynthesizer(InstrumentSynthesizer):
    def __init__(
        self,
        sample_rate: int = 48000,
        n_harmonics: int = 24,  # More harmonics for richer piano sound
        noise_level: float = 0.001,
        string_resonance: float = 0.3,
        hammer_hardness: float = 0.7
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_harmonics=n_harmonics,
            noise_level=noise_level,
            attack_time=0.002,  # Fast attack for piano
            decay_time=0.15,
            sustain_level=0.6,
            release_time=0.5
        )
        
        self.string_resonance = string_resonance
        self.hammer_hardness = hammer_hardness
        
        # Initialize piano-specific filters
        self.body_resonance = T.Resample(
            orig_freq=sample_rate,
            new_freq=sample_rate,
            lowpass_filter_width=6
        )
        
        # Harmonic structure specific to piano
        self._initialize_piano_harmonics()
        
        # String resonance simulation
        self.string_delay = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=int(sample_rate * 0.01),  # 10ms delay
            padding='same',
            bias=False
        )
        
        # Initialize delay line coefficients
        self._initialize_delay_line()

    def _initialize_piano_harmonics(self):
        """Initialize harmonic weights specific to piano timbre."""
        # Piano harmonics follow a more complex pattern
        weights = []
        for i in range(self.n_harmonics):
            # Fundamental and even harmonics are stronger
            if i == 0:
                weights.append(1.0)
            elif i % 2 == 0:
                weights.append(0.5 / (i + 1))
            else:
                weights.append(0.3 / (i + 1))
        
        # Apply inharmonicity (stretch factor increases with frequency)
        stretch_factor = 1.0013
        for i in range(len(weights)):
            weights[i] *= (stretch_factor ** i)
        
        self.harmonic_weights = nn.Parameter(torch.tensor(weights))

    def _initialize_delay_line(self):
        """Initialize the delay line for string resonance simulation."""
        delay_line = torch.zeros(self.string_delay.kernel_size[0])
        
        # Create feedback pattern for string resonance
        for i in range(delay_line.size(0)):
            delay_line[i] = (-1) ** i * np.exp(-i * 0.001)
        
        self.string_delay.weight.data = (
            delay_line.view(1, 1, -1) * self.string_resonance
        )
        self.string_delay.weight.requires_grad = False

    def apply_hammer_effect(self, signal: torch.Tensor) -> torch.Tensor:
        """Simulate piano hammer strike effect."""
        # Create hammer impact shape
        attack_samples = int(0.002 * self.sample_rate)  # 2ms attack
        hammer_shape = torch.exp(
            -torch.linspace(0, 5, attack_samples) * self.hammer_hardness
        )
        
        # Apply to beginning of each note
        signal[:attack_samples] *= hammer_shape
        return signal

    def apply_string_resonance(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply string resonance effect using delay line."""
        # Reshape for conv1d
        signal_reshaped = signal.view(1, 1, -1)
        
        # Apply delay line
        resonance = self.string_delay(signal_reshaped)
        
        # Mix with original
        return (signal_reshaped + resonance).view(-1)

    def synthesize(
        self,
        midi_sequence: np.ndarray,
        duration: float
    ) -> torch.Tensor:
        """Synthesize piano sound from MIDI sequence."""
        output_signal = torch.zeros(int(self.sample_rate * duration))
        
        for note in midi_sequence:
            if note > 0:  # Skip rests (note = 0)
                # Convert MIDI note to frequency with stretched tuning
                frequency = 440 * (2 ** ((note - 69) / 12))
                # Apply slight stretch to higher notes
                if note > 60:
                    frequency *= 1.0001 ** (note - 60)
                
                # Generate basic tone
                signal = self.generate_sine(frequency, duration)
                
                # Apply piano-specific effects
                signal = self.apply_hammer_effect(signal)
                signal = self.apply_string_resonance(signal)
                
                # Apply envelope
                signal = self.apply_envelope(signal, duration)
                
                # Add subtle noise for realism
                signal = self.add_noise(signal)
                
                # Apply body resonance and filters
                signal = self.body_resonance(signal)
                signal = self.apply_filters(signal)
                
                # Mix into output with velocity scaling
                velocity_scale = 0.8  # Adjust for natural piano dynamics
                output_signal += signal * velocity_scale
        
        # Normalize final output
        output_signal = output_signal / torch.max(torch.abs(output_signal))
        
        # Apply final piano-specific EQ
        output_signal = self._apply_piano_eq(output_signal)
        
        return output_signal

    def _apply_piano_eq(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply piano-specific equalization."""
        # Create frequency bands for EQ
        bands = [
            (20, 100, 0.7),    # Sub-bass reduction
            (100, 500, 1.2),   # Low-mids boost
            (500, 2000, 1.0),  # Mids flat
            (2000, 5000, 1.1), # Upper-mids slight boost
            (5000, 20000, 0.9) # Highs slight cut
        ]
        
        # Apply multi-band EQ
        eq_signal = signal.clone()
        for low_freq, high_freq, gain in bands:
            bandpass = T.Bandpass(
                cutoff_freq=(low_freq, high_freq),
                sample_rate=self.sample_rate
            )
            band = bandpass(signal)
            eq_signal += (band * (gain - 1))
        
        return eq_signal

    def forward(
        self,
        midi_sequence: torch.Tensor,
        duration: float
    ) -> torch.Tensor:
        """Forward pass for the piano synthesizer."""
        return self.synthesize(midi_sequence, duration)
