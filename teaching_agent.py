"""
AI-Driven Pedagogical Decision-Support and Teaching Suggestion Agent.
Addresses:
- Scenario 3: Context-aware teaching suggestions, real-time pacing adjustments,
  interactive activity prompts, and classroom management insights.
Guiding principle: Non-enforcing, assistive decision support empowering the educator.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class TeachingSuggestion:
    timestamp: float
    trigger_state: str
    urgency: str               # 'LOW', 'MEDIUM', 'HIGH'
    category: str              # 'PACING', 'ACTIVE_LEARNING', 'CHECK_UNDERSTANDING', 'RE-ENGAGEMENT'
    pedagogical_rationale: str
    actionable_prompt: str
    confidence_score: float

class TeachingSuggestionAgent:
    def __init__(self):
        self.suggestions_history: List[TeachingSuggestion] = []
        self._last_suggestion_time = 0.0
        self._cooldown_seconds = 15.0  # Prevents alert fatigue for the educator
        
    def evaluate_dynamics(self, 
                          current_time: float, 
                          interval_metrics, 
                          recent_trend: str) -> Optional[TeachingSuggestion]:
        """
        Evaluates current session dynamic metrics and generates an assistive pedagogical suggestion
        if an actionable pedagogical pattern is recognized.
        """
        # Respect cooldown to prevent overwhelming the instructor
        if (current_time - self._last_suggestion_time) < self._cooldown_seconds:
            return None

        state = interval_metrics.session_dynamic_state
        composite = interval_metrics.composite_engagement_score
        gaze = interval_metrics.mean_gaze_score
        motion = interval_metrics.mean_motion_score
        audio = interval_metrics.mean_audio_interaction
        
        suggestion = None

        # Rule 1: Attention Drop / Gaze Dissipation
        if state == "ATTENTION_DROP" or (composite < 0.42 and recent_trend == "FALLING"):
            if gaze < 0.35 and motion < 0.40:
                suggestion = TeachingSuggestion(
                    timestamp=current_time,
                    trigger_state="DEEP_ATTENTION_DROP",
                    urgency="HIGH",
                    category="ACTIVE_LEARNING",
                    pedagogical_rationale=(
                        f"Collective gaze fell to {int(gaze*100)}% with prolonged stillness. "
                        "Learners exhibit passive cognitive fatigue."
                    ),
                    actionable_prompt=(
                        "💡 Rapid Engagement Pivot: Initiate a 60-second 'Think-Pair-Share' or prompt "
                        "students to jot down one muddiest point on the board."
                    ),
                    confidence_score=0.92
                )
            else:
                suggestion = TeachingSuggestion(
                    timestamp=current_time,
                    trigger_state="MILD_ATTENTION_DROP",
                    urgency="MEDIUM",
                    category="CHECK_UNDERSTANDING",
                    pedagogical_rationale=(
                        f"Engagement dipped to {int(composite*100)}%. "
                        "Gaze divergence indicates potential conceptual confusion or pacing lag."
                    ),
                    actionable_prompt=(
                        "💡 Formative Check: Pause lecture delivery and conduct a quick thumbs-up/thumbs-down "
                        "confidence vote on the last worked example."
                    ),
                    confidence_score=0.85
                )

        # Rule 2: Monotone Pacing / Physical Stagnation
        elif state == "PACING_STAGNATION":
            suggestion = TeachingSuggestion(
                timestamp=current_time,
                trigger_state="PACING_STAGNATION",
                urgency="MEDIUM",
                category="PACING",
                pedagogical_rationale=(
                    f"Minimal physical motion ({int(motion*100)}%) and monologue acoustic profile. "
                    "Session pacing has plateaued."
                ),
                actionable_prompt=(
                    "💡 Pacing Adjustment: Shift from passive delivery to cold-calling a volunteer "
                    "or asking students to stand up for a quick 30-second concept poll."
                ),
                confidence_score=0.88
            )

        # Rule 3: High Interaction Peak
        elif state == "INTERACTION_PEAK" or (composite > 0.80 and audio > 0.80):
            suggestion = TeachingSuggestion(
                timestamp=current_time,
                trigger_state="INTERACTION_PEAK",
                urgency="LOW",
                category="RE-ENGAGEMENT",
                pedagogical_rationale=(
                    f"Engagement at peak ({int(composite*100)}%) with robust verbal exchange. "
                    "Optimal collaborative discovery zone."
                ),
                actionable_prompt=(
                    "⭐ Pedagogical Opportunity: Excellent momentum! Synthesize the student contributions "
                    "onto the slide or transition into small-group synthesis before energy dissipates."
                ),
                confidence_score=0.90
            )

        # Rule 4: Restlessness / Erratic Movement (excessive fidgeting)
        elif motion > 0.82 and gaze < 0.50:
            suggestion = TeachingSuggestion(
                timestamp=current_time,
                trigger_state="RESTLESSNESS_SPIKE",
                urgency="MEDIUM",
                category="RE-ENGAGEMENT",
                pedagogical_rationale=(
                    "Restlessness detected with low visual orientation. Classroom energy is unfocused."
                ),
                actionable_prompt=(
                    "💡 Focus Alignment: Refocus attention by transitioning to an individual problem-solving challenge "
                    "or writing a thought-provoking challenge problem on the center board."
                ),
                confidence_score=0.81
            )

        if suggestion:
            self.suggestions_history.append(suggestion)
            self._last_suggestion_time = current_time

        return suggestion

    def get_session_suggestions_summary(self) -> List[Dict]:
        """Exports all generated suggestions in structured format."""
        return [
            {
                "time_offset_sec": round(s.timestamp, 1),
                "trigger": s.trigger_state,
                "urgency": s.urgency,
                "category": s.category,
                "rationale": s.pedagogical_rationale,
                "prompt": s.actionable_prompt,
                "confidence": s.confidence_score
            }
            for s in self.suggestions_history
        ]
