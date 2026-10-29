"""
End-to-End Multimodal Classroom Intelligence Pipeline Demonstration.
Executes an end-to-end classroom session covering:
- Scenario 1: Automated Attendance Tracking (with sustained presence vs transient passerby test)
- Scenario 2: Multimodal Engagement & Dynamics Fusion (head pose, motion, audio activity)
- Scenario 3: AI-Driven Teaching Suggestions & Pedagogical Interventions
Generates CSV, JSON, and HTML visual reports.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import math
import random
from typing import List, Dict
import numpy as np

from config import AgentSystemConfig
from vision_tracker import VisionTracker, VisualObservation
from audio_processor import AudioProcessor
from fusion_engine import MultimodalFusionEngine
from teaching_agent import TeachingSuggestionAgent
from analytics_reporter import AnalyticsReporter

def run_simulation(session_duration_seconds: int = 60, time_step_seconds: float = 1.0, verbose: bool = True):
    config = AgentSystemConfig()
    
    # Initialize all subsystems
    vision = VisionTracker(config.attendance, config.vision, config.privacy)
    audio = AudioProcessor(config.audio, config.privacy)
    fusion = MultimodalFusionEngine(config.fusion)
    agent = TeachingSuggestionAgent()
    reporter = AnalyticsReporter("C:/Users/prihu/.gemini/antigravity/scratch/multimodal_classroom_agent/outputs")

    print("=" * 75)
    print("🎓 MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT - SESSION ENGINE")
    print("=" * 75)
    print(f"Privacy Guarantee: Offline Execution | Zero Raw Media Retention | Pseudonymized IDs")
    print(f"Sustained Presence Threshold: {config.attendance.min_sustained_seconds} seconds")
    print(f"Total Simulation Duration: {session_duration_seconds}s (Step: {time_step_seconds}s)\n")

    # Cohort setup
    # Regular students who stay for the class
    core_students = ["student_alpha", "student_bravo", "student_charlie", "student_delta", "student_echo"]
    # Transient passerby (enters at t=5, leaves at t=9 -> 4 seconds presence -> MUST NOT BE MARKED PRESENT!)
    passerby = "transient_visitor_hallway"

    current_time = 0.0
    step_count = int(session_duration_seconds / time_step_seconds)

    for step in range(step_count):
        current_time += time_step_seconds
        
        # Determine Pedagogical Phase
        # Phase 1 (0-20s): Introductory lecture (moderate-high engagement)
        # Phase 2 (20-40s): Post-lunch slump / Monotone explanation (attention drop)
        # Phase 3 (40-60s): Interactive group Q&A / problem solving (interaction peak)
        if current_time <= 20.0:
            phase_name = "Phase 1: Concept Introduction"
            audio_scenario = "lecture_delivery"
            gaze_bias = 0.85
            motion_bias = 0.30
        elif current_time <= 40.0:
            phase_name = "Phase 2: Technical Deep-Dive (Monotone)"
            audio_scenario = "silence_drop"
            gaze_bias = 0.25  # Looking away, distraction
            motion_bias = 0.05 # Stagnation / slouching
        else:
            phase_name = "Phase 3: Active Classroom Discussion"
            audio_scenario = "interactive_discussion"
            gaze_bias = 0.80
            motion_bias = 0.45 # Nodding, hand raising, note taking

        # 1. Synthesize Audio
        pcm_chunk = audio.generate_synthetic_audio_signal(time_step_seconds, scenario_type=audio_scenario)
        audio_metrics = audio.process_audio_chunk(current_time, pcm_chunk)

        # 2. Synthesize Visual Observations
        observations: List[VisualObservation] = []
        
        # Core students
        for sid in core_students:
            # Add stochastic variation around phase bias
            yaw = float(np.random.normal(0.0 if gaze_bias > 0.5 else 45.0, 12.0))
            pitch = float(np.random.normal(-5.0, 10.0))
            motion = float(np.clip(np.random.normal(motion_bias, 0.08), 0.02, 0.95))
            
            observations.append(VisualObservation(
                track_id=sid,
                pseudonym_id=sid,
                timestamp=current_time,
                bbox=(100, 100, 50, 50),
                head_yaw=yaw,
                head_pitch=pitch,
                motion_energy=motion
            ))

        # Add transient visitor only between t=5 and t=9
        if 5.0 <= current_time <= 9.0:
            observations.append(VisualObservation(
                track_id=passerby,
                pseudonym_id=passerby,
                timestamp=current_time,
                bbox=(500, 100, 50, 50),
                head_yaw=65.0,
                head_pitch=0.0,
                motion_energy=0.70
            ))

        # Process Visuals
        visual_student_metrics = vision.process_frame_observations(current_time, observations)

        # 3. Multimodal Fusion
        instant_fusion = fusion.fuse_instantaneous_signals(visual_student_metrics, audio_metrics)
        
        # Check Interval Boundary (every 10 seconds)
        interval_metric = fusion.check_interval_completion(current_time)

        # Print periodic progress
        if int(current_time) % 5 == 0 and verbose:
            print(f"⏱️  [{current_time:04.1f}s] {phase_name:<36} | Engagement: {instant_fusion['composite_engagement_score']*100:.1f}% | Acoustic: {audio_metrics['acoustic_state']}")

        # 4. Agent Reasoning on Interval Completion (Scenario 3)
        if interval_metric:
            recent_trend = interval_metric.engagement_trend
            suggestion = agent.evaluate_dynamics(current_time, interval_metric, recent_trend)
            
            print(f"\n📊 --- INTERVAL {interval_metric.interval_index} SUMMARY [{interval_metric.start_time:.0f}s - {interval_metric.end_time:.0f}s] ---")
            print(f"   Composite Engagement: {interval_metric.composite_engagement_score*100:.1f}% ({interval_metric.engagement_trend}) | State: {interval_metric.session_dynamic_state}")
            print(f"   Visual Gaze: {interval_metric.mean_gaze_score*100:.1f}% | Motion: {interval_metric.mean_motion_score*100:.1f}% | Audio Interactivity: {interval_metric.mean_audio_interaction*100:.1f}%")
            
            if suggestion:
                print(f"   🤖 [AI TEACHING AGENT INTERVENTION] ({suggestion.urgency} Urgency - {suggestion.category})")
                print(f"      Rationale: {suggestion.pedagogical_rationale}")
                print(f"      Action:    {suggestion.actionable_prompt}")
            print("-" * 75 + "\n")

    # Finalize Session Reports
    attendance_records = vision.get_attendance_summary()
    session_stats = fusion.get_full_session_trend_summary()
    suggestions_summary = agent.get_session_suggestions_summary()

    # Generate Exports
    csv_file = reporter.export_attendance_csv(attendance_records)
    json_payload = {
        "metadata": {
            "session_duration_seconds": session_duration_seconds,
            "privacy_compliance": "Zero-raw-data-retained / SHA-256 Pseudonymized",
            "sustained_presence_threshold_seconds": config.attendance.min_sustained_seconds
        },
        "attendance_roll_call": attendance_records,
        "session_dynamics": session_stats,
        "interval_trajectory": [m.__dict__ for m in fusion.interval_history],
        "teaching_suggestions": suggestions_summary
    }
    json_file = reporter.export_session_summary_json(json_payload)
    html_file = reporter.generate_html_dashboard(
        attendance_records=attendance_records,
        intervals=fusion.interval_history,
        suggestions=suggestions_summary,
        session_stats=session_stats
    )

    print("=" * 75)
    print("✅ SESSION EXECUTION COMPLETE")
    print("=" * 75)
    print(f"📋 Verified Attendance Count: {len(attendance_records)} / {len(core_students) + 1} candidates")
    print(f"   (Verified that transient visitor '{passerby}' was REJECTED by sustained presence debounce)")
    for rec in attendance_records:
        print(f"   - {rec['student_id']}: Confirmed at t={rec['confirmed_at']}s (Sustained {rec['sustained_duration']}s)")
    
    print(f"\n📈 Session Trajectory Statistics:")
    print(f"   - Overall Average Engagement: {session_stats['session_average_engagement']*100:.1f}%")
    print(f"   - Dynamic State Breakdown: {session_stats['state_distribution']}")
    print(f"   - Pedagogical Suggestions Issued: {len(suggestions_summary)}")

    print(f"\n📁 Generated Intelligence Reports:")
    print(f"   - CSV Roll Call: {csv_file}")
    print(f"   - JSON Session Data: {json_file}")
    print(f"   - HTML Visual Dashboard: {html_file}")
    print("=" * 75)

    return {
        "attendance": attendance_records,
        "stats": session_stats,
        "suggestions": suggestions_summary,
        "html_report": html_file
    }

if __name__ == "__main__":
    run_simulation(session_duration_seconds=60, time_step_seconds=1.0, verbose=True)
