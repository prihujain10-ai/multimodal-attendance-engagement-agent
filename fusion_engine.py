"""
Multimodal Signal Fusion and Temporal Trend Engine.
Addresses:
- Scenario 2: Synchronizing and fusing visual behavioral indicators and audio dynamics
  into an unified composite engagement index E(t) across session time intervals.
- Detection of attention drops, interaction peaks, and pacing stagnation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from config import FusionConfig

@dataclass
class SessionIntervalMetrics:
    interval_index: int
    start_time: float
    end_time: float
    active_students_count: int
    mean_gaze_score: float
    mean_motion_score: float
    mean_audio_interaction: float
    composite_engagement_score: float
    engagement_trend: str               # 'RISING', 'STABLE', 'FALLING'
    session_dynamic_state: str          # 'BALANCED', 'ATTENTION_DROP', 'INTERACTION_PEAK', 'PACING_STAGNATION'
    anomalies: List[str] = field(default_factory=list)


class MultimodalFusionEngine:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.interval_history: List[SessionIntervalMetrics] = []
        
        # In-interval temporary accumulators
        self.current_interval_idx = 0
        self.interval_start_time = 0.0
        self.acc_gaze_scores: List[float] = []
        self.acc_motion_scores: List[float] = []
        self.acc_audio_scores: List[float] = []
        self.acc_student_counts: List[int] = []

    def fuse_instantaneous_signals(self, 
                                   visual_student_metrics: Dict[str, Dict], 
                                   audio_metrics: Dict) -> Dict:
        """
        Computes the instantaneous multimodal engagement score at timestamp t:
        E(t) = w_g * V_gaze(t) + w_m * V_motion(t) + w_a * A_audio(t)
        """
        if visual_student_metrics:
            gaze_scores = [m["head_attention_score"] for m in visual_student_metrics.values()]
            motion_scores = [m["motion_engagement_score"] for m in visual_student_metrics.values()]
            mean_gaze = float(np.mean(gaze_scores))
            mean_motion = float(np.mean(motion_scores))
            active_count = len(visual_student_metrics)
        else:
            mean_gaze = 0.5
            mean_motion = 0.5
            active_count = 0

        audio_interaction = audio_metrics.get("interaction_score", 0.5)

        # Mathematical Fusion Formulation
        composite_score = (
            self.cfg.weight_gaze_orientation * mean_gaze +
            self.cfg.weight_motion_dynamics * mean_motion +
            self.cfg.weight_audio_interaction * audio_interaction
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # Accumulate for time-interval trend analysis
        self.acc_gaze_scores.append(mean_gaze)
        self.acc_motion_scores.append(mean_motion)
        self.acc_audio_scores.append(audio_interaction)
        self.acc_student_counts.append(active_count)

        return {
            "mean_gaze_attention": round(mean_gaze, 3),
            "mean_motion_engagement": round(mean_motion, 3),
            "audio_interaction": round(audio_interaction, 3),
            "composite_engagement_score": round(composite_score, 3),
            "active_participants": active_count
        }

    def check_interval_completion(self, current_time: float) -> Optional[SessionIntervalMetrics]:
        """
        Checks if the configured time interval (e.g. 10s or 30s) has elapsed.
        If yes, aggregates metrics, determines trend, classifies classroom dynamics,
        and returns the interval summary.
        """
        if self.interval_start_time == 0.0:
            self.interval_start_time = current_time
            return None

        elapsed = current_time - self.interval_start_time
        if elapsed < self.cfg.interval_aggregation_seconds:
            return None

        # Interval boundary reached: aggregate
        mean_gaze = float(np.mean(self.acc_gaze_scores)) if self.acc_gaze_scores else 0.5
        mean_motion = float(np.mean(self.acc_motion_scores)) if self.acc_motion_scores else 0.5
        mean_audio = float(np.mean(self.acc_audio_scores)) if self.acc_audio_scores else 0.5
        avg_students = int(np.round(np.mean(self.acc_student_counts))) if self.acc_student_counts else 0

        interval_composite = (
            self.cfg.weight_gaze_orientation * mean_gaze +
            self.cfg.weight_motion_dynamics * mean_motion +
            self.cfg.weight_audio_interaction * mean_audio
        )
        interval_composite = float(np.clip(interval_composite, 0.0, 1.0))

        # Trend analysis relative to previous interval
        trend = "STABLE"
        anomalies = []
        if self.interval_history:
            prev_composite = self.interval_history[-1].composite_engagement_score
            delta = interval_composite - prev_composite
            if delta > 0.08:
                trend = "RISING"
            elif delta < -0.08:
                trend = "FALLING"

        # Classroom Dynamic State Classification
        if interval_composite < self.cfg.low_engagement_threshold:
            state = "ATTENTION_DROP"
            anomalies.append("Significant collective attention decline detected")
        elif interval_composite >= self.cfg.high_engagement_threshold and mean_audio > 0.75:
            state = "INTERACTION_PEAK"
            anomalies.append("High collective inquiry / peak discussion zone")
        elif mean_motion < 0.40 and mean_audio < 0.50:
            state = "PACING_STAGNATION"
            anomalies.append("Monotonous lecture pacing with low visual/physical interactivity")
        else:
            state = "BALANCED"

        interval_metric = SessionIntervalMetrics(
            interval_index=self.current_interval_idx,
            start_time=round(self.interval_start_time, 2),
            end_time=round(current_time, 2),
            active_students_count=avg_students,
            mean_gaze_score=round(mean_gaze, 3),
            mean_motion_score=round(mean_motion, 3),
            mean_audio_interaction=round(mean_audio, 3),
            composite_engagement_score=round(interval_composite, 3),
            engagement_trend=trend,
            session_dynamic_state=state,
            anomalies=anomalies
        )

        self.interval_history.append(interval_metric)

        # Reset accumulators for next interval
        self.current_interval_idx += 1
        self.interval_start_time = current_time
        self.acc_gaze_scores.clear()
        self.acc_motion_scores.clear()
        self.acc_audio_scores.clear()
        self.acc_student_counts.clear()

        return interval_metric

    def get_full_session_trend_summary(self) -> Dict:
        """Returns overall session trajectory statistics."""
        if not self.interval_history:
            return {"total_intervals": 0, "session_average_engagement": 0.0}

        scores = [m.composite_engagement_score for m in self.interval_history]
        states = [m.session_dynamic_state for m in self.interval_history]
        
        return {
            "total_intervals": len(self.interval_history),
            "session_average_engagement": round(float(np.mean(scores)), 3),
            "peak_engagement": round(float(np.max(scores)), 3),
            "lowest_engagement": round(float(np.min(scores)), 3),
            "state_distribution": {
                "ATTENTION_DROP": states.count("ATTENTION_DROP"),
                "INTERACTION_PEAK": states.count("INTERACTION_PEAK"),
                "PACING_STAGNATION": states.count("PACING_STAGNATION"),
                "BALANCED": states.count("BALANCED")
            }
        }
