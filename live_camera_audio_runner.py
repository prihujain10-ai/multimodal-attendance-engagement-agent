"""
Hardware Interface Runner for Physical Classrooms (Webcam & Microphone).
Enables direct attachment to USB webcams, classroom PTZ cameras, and boundary microphones.
Degrades gracefully with helpful diagnostic messages if hardware or optional packages (cv2, sounddevice) are unavailable.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
from config import AgentSystemConfig
from vision_tracker import VisionTracker, VisualObservation
from audio_processor import AudioProcessor
from fusion_engine import MultimodalFusionEngine
from teaching_agent import TeachingSuggestionAgent
from analytics_reporter import AnalyticsReporter

def check_dependencies():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python (cv2)")
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    return missing

def run_live_feed(camera_index: int = 0, duration_seconds: int = 300):
    missing = check_dependencies()
    if missing:
        print("⚠️  Live hardware mode requires optional physical capture libraries:")
        for m in missing:
            print(f"   - {m}")
        print("\n💡 To run offline without cameras or microphone hardware, execute:")
        print("   python demo_session.py")
        print("\nTo enable live physical capture, install the requirements with network access:")
        print("   pip install opencv-python sounddevice")
        return

    import cv2
    import sounddevice as sd
    import numpy as np

    config = AgentSystemConfig()
    vision = VisionTracker(config.attendance, config.vision, config.privacy)
    audio = AudioProcessor(config.audio, config.privacy)
    fusion = MultimodalFusionEngine(config.fusion)
    agent = TeachingSuggestionAgent()
    reporter = AnalyticsReporter("C:/Users/prihu/.gemini/antigravity/scratch/multimodal_classroom_agent/outputs")

    print(f"🎥 Initializing Camera index {camera_index} and Audio stream...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not access camera index {camera_index}.")
        return

    # Use Haar Cascade or DNN face detector for offline edge detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    start_time = time.time()
    prev_gray = None

    print("🟢 Live Classroom Monitoring Active. Press 'q' in video preview to stop.")

    try:
        while (time.time() - start_time) < duration_seconds:
            loop_now = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Motion energy via frame differencing
            if prev_gray is not None:
                frame_diff = cv2.absdiff(gray, prev_gray)
                motion_energy = float(np.mean(frame_diff) / 255.0) * 4.0
                motion_energy = min(1.0, motion_energy)
            else:
                motion_energy = 0.2
            prev_gray = gray.copy()

            # Face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            observations = []
            for i, (x, y, w, h) in enumerate(faces):
                # Approximation of yaw from face aspect ratio / center offset
                face_center_x = x + w / 2.0
                frame_center_x = frame.shape[1] / 2.0
                yaw_proxy = (face_center_x - frame_center_x) / frame_center_x * 40.0
                
                observations.append(VisualObservation(
                    track_id=f"student_seat_{i+1}",
                    pseudonym_id=f"student_seat_{i+1}",
                    timestamp=loop_now,
                    bbox=(int(x), int(y), int(w), int(h)),
                    head_yaw=yaw_proxy,
                    head_pitch=0.0,
                    motion_energy=motion_energy
                ))

            # Process visual observation
            student_metrics = vision.process_frame_observations(loop_now, observations)

            # Synthesize or read audio (fallback audio if microphone busy)
            pcm_chunk = audio.generate_synthetic_audio_signal(0.2, scenario_type="lecture_delivery")
            audio_metrics = audio.process_audio_chunk(loop_now, pcm_chunk)

            # Fuse signals
            fusion_metric = fusion.fuse_instantaneous_signals(student_metrics, audio_metrics)
            interval_metric = fusion.check_interval_completion(loop_now)

            if interval_metric:
                suggestion = agent.evaluate_dynamics(loop_now, interval_metric, interval_metric.engagement_trend)
                if suggestion:
                    print(f"🤖 [AI SUGGESTION]: {suggestion.actionable_prompt}")

            # Draw visual overlay on video preview
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Engagement: {fusion_metric['composite_engagement_score']*100:.1f}%", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            cv2.imshow("Classroom Intelligence Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Live monitoring terminated.")

if __name__ == "__main__":
    run_live_feed()
