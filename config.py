"""
Configuration and Hyperparameters for Multimodal Attendance & Engagement Agent.
All parameters are tuned for offline, edge-friendly, privacy-preserving execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AttendanceConfig:
    # Sustained presence requirements (Scenario 1)
    min_sustained_seconds: float = 15.0      # Minimum dwell time to mark attendance (prevents false positives from passersby)
    detection_confidence_threshold: float = 0.65
    tracking_max_lost_frames: int = 30        # Debounce filter frames before declaring student absent
    reid_similarity_threshold: float = 0.70   # Cosine similarity threshold for feature-based re-identification
    sampling_fps: int = 5                     # Video sample rate for low-power edge inference


@dataclass
class VisionConfig:
    # Engagement behavioral indicators (Scenario 2)
    frontal_yaw_max_deg: float = 25.0         # Head yaw angle within which face is considered attentive
    frontal_pitch_max_deg: float = 20.0       # Head pitch angle (up/down) within attentive range
    inactivity_warning_seconds: float = 45.0  # Duration of zero motion indicating daydreaming/distraction
    restlessness_threshold: float = 0.75      # Excessive motion threshold (fidgeting/distracted)
    nominal_motion_range: tuple = (0.10, 0.60)# Normal movement range while engaged (note-taking, nodding)


@dataclass
class AudioConfig:
    # Multimodal audio processing (Scenario 2)
    sample_rate: int = 16000
    chunk_size_ms: int = 100                  # 100ms processing chunks
    voice_activity_energy_thresh: float = 0.035
    silence_decay_factor: float = 0.95        # Adaptive noise floor adaptation
    interaction_window_seconds: float = 10.0   # Window to classify lecture vs. active Q&A / interaction


@dataclass
class FusionConfig:
    # Multimodal fusion weights for Engagement Index (Scenario 2)
    # Engagement Score E(t) = w_gaze * GazeScore + w_motion * MotionScore + w_audio * AudioScore
    weight_gaze_orientation: float = 0.45
    weight_motion_dynamics: float = 0.30
    weight_audio_interaction: float = 0.25
    
    # Dynamics intervals
    interval_aggregation_seconds: float = 10.0 # Time bucket for trends analysis
    low_engagement_threshold: float = 0.40     # Below this triggers attention drop alert
    high_engagement_threshold: float = 0.75    # Above this indicates peak flow/interaction


@dataclass
class PrivacyConfig:
    # Ethical and privacy-preserving constraints
    enable_anonymized_ids: bool = True        # Uses SHA-256 hashed pseudonym IDs (e.g., STD-7F2A)
    store_raw_video: bool = False              # STRICT: Zero persistent raw video storage
    store_raw_audio: bool = False              # STRICT: Zero persistent raw audio recording
    feature_retention_mode: str = "ephemeral"  # Only stores aggregate engagement metrics & logs
    local_only_inference: bool = True         # Offline strictly: no cloud APIs required


@dataclass
class AgentSystemConfig:
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
