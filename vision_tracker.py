"""
Vision Tracking and Visual Behavioral Signal Extraction Module.
Addresses:
- Scenario 1: Sustained Presence Attendance Tracking (filtering transient appearances)
- Scenario 2: Head pose estimation, movement dynamics, and inactivity tracking.
Privacy-first: Processes frames in-memory; face features are ephemeral and IDs are pseudonymized.
"""

import time
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import AttendanceConfig, VisionConfig, PrivacyConfig

@dataclass
class VisualObservation:
    track_id: str
    pseudonym_id: str
    timestamp: float
    bbox: Tuple[int, int, int, int]           # (x, y, w, h)
    head_yaw: float                           # Degrees: negative=left, positive=right
    head_pitch: float                         # Degrees: negative=down, positive=up
    motion_energy: float                      # Normalized [0.0, 1.0] movement energy
    detection_confidence: float = 0.90


@dataclass
class StudentTrackRecord:
    track_id: str
    pseudonym_id: str
    first_seen_time: float
    last_seen_time: float
    cumulative_presence_time: float = 0.0
    consecutive_present_frames: int = 0
    consecutive_lost_frames: int = 0
    attendance_confirmed: bool = False
    confirmed_timestamp: Optional[float] = None
    
    # Behavioral state
    last_motion_time: float = 0.0
    current_inactivity_duration: float = 0.0
    recent_head_yaws: List[float] = field(default_factory=list)
    recent_head_pitches: List[float] = field(default_factory=list)
    recent_motion_energies: List[float] = field(default_factory=list)


class VisionTracker:
    def __init__(self, att_cfg: AttendanceConfig, vis_cfg: VisionConfig, priv_cfg: PrivacyConfig, base_wallclock: Optional[float] = None):
        self.att_cfg = att_cfg
        self.vis_cfg = vis_cfg
        self.priv_cfg = priv_cfg
        self.base_wallclock = base_wallclock if base_wallclock is not None else time.time()
        self.tracks: Dict[str, StudentTrackRecord] = {}
        self.attendance_log: List[Dict] = []
        self._pseudonym_cache: Dict[str, str] = {}

    def _generate_pseudonym(self, raw_id: str) -> str:
        """Generates a stable, privacy-preserving pseudonym (e.g. 'STD-9B4F') from raw ID/embedding."""
        if not self.priv_cfg.enable_anonymized_ids:
            return raw_id
        if raw_id not in self._pseudonym_cache:
            digest = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:4].upper()
            self._pseudonym_cache[raw_id] = f"STD-{digest}"
        return self._pseudonym_cache[raw_id]

    def compute_head_pose_attention(self, yaw_deg: float, pitch_deg: float) -> float:
        """
        Quantifies gaze/orientation attention score [0.0 - 1.0].
        Full attention = 1.0 when facing within configured angular tolerances.
        Gaussian/Cosine penalty as angle deviates towards distraction.
        """
        yaw_dev = abs(yaw_deg)
        pitch_dev = abs(pitch_deg)
        
        # Normalized deviations beyond comfort zone
        yaw_penalty = max(0.0, (yaw_dev - self.vis_cfg.frontal_yaw_max_deg) / 60.0)
        pitch_penalty = max(0.0, (pitch_dev - self.vis_cfg.frontal_pitch_max_deg) / 45.0)
        
        combined_penalty = min(1.0, math.sqrt(yaw_penalty**2 + pitch_penalty**2))
        return float(np.clip(1.0 - combined_penalty, 0.0, 1.0))

    def compute_motion_score(self, motion_energy: float, inactivity_sec: float) -> float:
        """
        Converts movement indicators into an engagement quality index [0.0 - 1.0].
        - Extremely low motion over long duration -> Daydreaming / Sleeping (penalty)
        - Moderate motion -> Active engagement (note-taking, nodding, adjusting posture)
        - Extremely high motion -> Restlessness / Disruptive fidgeting (penalty)
        """
        min_nom, max_nom = self.vis_cfg.nominal_motion_range
        
        if inactivity_sec > self.vis_cfg.inactivity_warning_seconds:
            # Prolonged complete freeze or slump
            stagnation_factor = min(1.0, (inactivity_sec - self.vis_cfg.inactivity_warning_seconds) / 30.0)
            return float(np.clip(0.5 - 0.4 * stagnation_factor, 0.05, 1.0))
        
        if motion_energy > self.vis_cfg.restlessness_threshold:
            # Overly agitated / distracted
            restless_factor = min(1.0, (motion_energy - self.vis_cfg.restlessness_threshold) / 0.25)
            return float(np.clip(0.7 - 0.4 * restless_factor, 0.1, 1.0))
            
        if min_nom <= motion_energy <= max_nom:
            return 0.95  # Natural active learning posture
            
        # Mild movement
        return 0.80

    def process_frame_observations(self, current_time: float, observations: List[VisualObservation]) -> Dict[str, Dict]:
        """
        Ingests observations for the current timestamp, updates sustained presence filters,
        and extracts per-student visual engagement metrics.
        """
        seen_track_ids = set()
        frame_metrics: Dict[str, Dict] = {}

        for obs in observations:
            track_id = obs.track_id
            seen_track_ids.add(track_id)
            pseudo_id = self._generate_pseudonym(track_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = StudentTrackRecord(
                    track_id=track_id,
                    pseudonym_id=pseudo_id,
                    first_seen_time=current_time,
                    last_seen_time=current_time,
                    last_motion_time=current_time
                )

            record = self.tracks[track_id]
            time_delta = max(0.0, current_time - record.last_seen_time)
            
            # Update presence tracking
            record.last_seen_time = current_time
            record.cumulative_presence_time += time_delta
            record.consecutive_present_frames += 1
            record.consecutive_lost_frames = 0
            
            # Motion & Inactivity tracking
            if obs.motion_energy > 0.08:
                record.last_motion_time = current_time
                record.current_inactivity_duration = 0.0
            else:
                record.current_inactivity_duration = current_time - record.last_motion_time

            # Update historical buffers (sliding window of last 20 samples)
            record.recent_head_yaws.append(obs.head_yaw)
            record.recent_head_pitches.append(obs.head_pitch)
            record.recent_motion_energies.append(obs.motion_energy)
            if len(record.recent_head_yaws) > 20:
                record.recent_head_yaws.pop(0)
                record.recent_head_pitches.pop(0)
                record.recent_motion_energies.pop(0)

            # Scenario 1: Sustained presence verification (Anti-false-positive rule)
            if not record.attendance_confirmed:
                if record.cumulative_presence_time >= self.att_cfg.min_sustained_seconds:
                    record.attendance_confirmed = True
                    record.confirmed_timestamp = current_time
                    real_time_stamp = self.base_wallclock + current_time
                    self.attendance_log.append({
                        "student_id": record.pseudonym_id,
                        "status": "PRESENT_CONFIRMED",
                        "first_detected": round(record.first_seen_time, 2),
                        "confirmed_at": round(current_time, 2),
                        "sustained_duration": round(record.cumulative_presence_time, 2),
                        "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(real_time_stamp))
                    })

            # Scenario 2: Calculate instant visual engagement indicators
            head_attention = self.compute_head_pose_attention(obs.head_yaw, obs.head_pitch)
            motion_engagement = self.compute_motion_score(obs.motion_energy, record.current_inactivity_duration)
            combined_visual_score = 0.60 * head_attention + 0.40 * motion_engagement

            frame_metrics[record.pseudonym_id] = {
                "track_id": track_id,
                "pseudonym_id": record.pseudonym_id,
                "attendance_confirmed": record.attendance_confirmed,
                "cumulative_presence_sec": round(record.cumulative_presence_time, 2),
                "head_yaw": round(obs.head_yaw, 1),
                "head_pitch": round(obs.head_pitch, 1),
                "head_attention_score": round(head_attention, 3),
                "motion_energy": round(obs.motion_energy, 3),
                "inactivity_seconds": round(record.current_inactivity_duration, 1),
                "motion_engagement_score": round(motion_engagement, 3),
                "visual_engagement_index": round(combined_visual_score, 3)
            }

        # Handle absent/lost tracks (debounce filter)
        for t_id, record in list(self.tracks.items()):
            if t_id not in seen_track_ids:
                record.consecutive_lost_frames += 1
                if record.consecutive_lost_frames > self.att_cfg.tracking_max_lost_frames:
                    # Mark track as expired/exited
                    pass

        return frame_metrics

    def get_attendance_summary(self) -> List[Dict]:
        """Returns verified attendance records."""
        return self.attendance_log
