"""
Audio Signal Processing and Classroom Acoustic Dynamics Module.
Addresses:
- Scenario 2: Audio activity analysis, vocal interaction detection, speech-to-silence ratio.
Privacy-first: In-memory streaming chunk analysis with zero raw audio persistence.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from config import AudioConfig, PrivacyConfig

@dataclass
class AudioIntervalFeatures:
    timestamp: float
    duration_seconds: float
    rms_energy: float
    speech_ratio: float
    is_speech_active: bool
    acoustic_state: str  # 'SILENT', 'CONSTRUCTIVE_DIALOGUE', 'LECTURE_MONOLOGUE', 'HIGH_NOISE'
    interaction_score: float  # [0.0 - 1.0] representing conversational engagement


class AudioProcessor:
    def __init__(self, audio_cfg: AudioConfig, priv_cfg: PrivacyConfig):
        self.cfg = audio_cfg
        self.priv_cfg = priv_cfg
        
        # Adaptive noise floor baseline
        self.ambient_noise_floor = 0.010
        self.recent_rms_buffer: List[float] = []
        self.max_buffer_samples = int(self.cfg.interaction_window_seconds / (self.cfg.chunk_size_ms / 1000.0))

    def compute_chunk_rms(self, pcm_samples: np.ndarray) -> float:
        """Calculates Root Mean Square (RMS) energy of a 1D audio waveform."""
        if len(pcm_samples) == 0:
            return 0.0
        # Normalize if integer PCM
        if np.issubdtype(pcm_samples.dtype, np.integer):
            info = np.iinfo(pcm_samples.dtype)
            samples = pcm_samples.astype(np.float32) / max(abs(info.min), info.max)
        else:
            samples = pcm_samples.astype(np.float32)
            
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms

    def process_audio_chunk(self, current_time: float, pcm_chunk: np.ndarray) -> Dict:
        """
        Processes an incoming audio chunk (e.g. 100ms) in-memory without disk persistence.
        Updates noise floor, detects voice activity, and tracks interaction metrics.
        """
        rms = self.compute_chunk_rms(pcm_chunk)
        
        # Adaptive noise floor updating (slow exponential moving average during quiet periods)
        if rms < self.ambient_noise_floor * 1.5:
            self.ambient_noise_floor = (self.cfg.silence_decay_factor * self.ambient_noise_floor + 
                                       (1.0 - self.cfg.silence_decay_factor) * rms)
            self.ambient_noise_floor = max(0.005, self.ambient_noise_floor)

        effective_thresh = max(self.cfg.voice_activity_energy_thresh, self.ambient_noise_floor * 2.2)
        is_speech = rms > effective_thresh

        self.recent_rms_buffer.append(rms)
        if len(self.recent_rms_buffer) > self.max_buffer_samples:
            self.recent_rms_buffer.pop(0)

        # Calculate speech activity ratio over current sliding window
        active_chunks = sum(1 for val in self.recent_rms_buffer if val > effective_thresh)
        speech_ratio = float(active_chunks / max(1, len(self.recent_rms_buffer)))

        # Characterize acoustic state
        # In classroom pedagogy:
        # - 0.15 to 0.65 speech ratio with moderate RMS = Healthy Q&A / active student-teacher interaction
        # - < 0.05 speech ratio = Long silence / passivity
        # - > 0.85 with very high RMS = Potential chaotic noise / unruly chatter
        if speech_ratio < 0.05:
            acoustic_state = "PROLONGED_SILENCE"
            interaction_score = 0.40  # Low interactivity
        elif speech_ratio > 0.85 and rms > 0.25:
            acoustic_state = "HIGH_CHAOTIC_NOISE"
            interaction_score = 0.50  # Over-saturated chatter
        elif 0.15 <= speech_ratio <= 0.70:
            acoustic_state = "CONSTRUCTIVE_DIALOGUE"
            interaction_score = 0.95  # Ideal active classroom discussion
        else:
            acoustic_state = "LECTURE_MONOLOGUE"
            interaction_score = 0.70  # Standard instructor delivery

        return {
            "timestamp": current_time,
            "instant_rms": round(rms, 4),
            "ambient_noise_floor": round(self.ambient_noise_floor, 4),
            "is_speech_active": is_speech,
            "window_speech_ratio": round(speech_ratio, 3),
            "acoustic_state": acoustic_state,
            "interaction_score": round(interaction_score, 3)
        }

    def generate_synthetic_audio_signal(self, duration_sec: float, scenario_type: str = "balanced") -> np.ndarray:
        """
        Generates realistic synthetic audio waveforms for offline testing and validation
        when physical microphones are not attached.
        """
        num_samples = int(self.cfg.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        
        # Base ambient room noise
        waveform = np.random.normal(0, 0.008, num_samples)

        if scenario_type == "interactive_discussion":
            # Frequent vocal bursts (formants simulation)
            bursts = [(0.1, 0.4), (0.6, 0.9)]
            for start, end in bursts:
                idx_s = int(start * num_samples)
                idx_e = int(end * num_samples)
                segment_t = t[idx_s:idx_e]
                vocal = (0.09 * np.sin(2 * np.pi * 180 * segment_t) +
                         0.05 * np.sin(2 * np.pi * 320 * segment_t) +
                         np.random.normal(0, 0.02, idx_e - idx_s))
                waveform[idx_s:idx_e] += vocal
                
        elif scenario_type == "silence_drop":
            # Minimal background noise only
            pass
            
        elif scenario_type == "lecture_delivery":
            # Regular cadence teacher speech
            cadence = np.sin(2 * np.pi * 1.5 * t) > 0.2
            vocal = 0.08 * np.sin(2 * np.pi * 200 * t) * cadence
            waveform += vocal

        return np.clip(waveform, -1.0, 1.0).astype(np.float32)
