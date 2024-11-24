"""
project @ SitBlinkSip
created @ 2024-10-17
author  @ github/ishworrsubedii
"""
import time

from scipy.spatial import distance as dist
from imutils import face_utils
import cv2
import dlib


class BlinkDetector:
    def __init__(self, shape_predictor_path, ear_threshold=0.25, ear_consec_frames_min=2, ear_consec_frames_max=5):
        self.EYE_AR_THRESH = ear_threshold
        self.EYE_AR_CONSEC_FRAMES_MIN = ear_consec_frames_min
        self.EYE_AR_CONSEC_FRAMES_MAX = ear_consec_frames_max
        self.counter = 0
        self.ear = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.colors = {
            'healthy': (46, 204, 113),     # Green
            'warning': (231, 76, 60),      # Red
            'text': (236, 240, 241),       # Light text
            'secondary': (189, 195, 199),  # Light gray
            'accent': (52, 152, 219),      # Blue
            'background': (0, 0, 0)        # Black
        }
        self.blink_count = 0
        self.start_time = time.time()
        self.blinks_per_minute = 0
        self.tips = [
            "Take a 20-second break every 20 minutes",
            "Adjust screen brightness to match surroundings",
            "Position screen at arm's length",
            "Keep screen at eye level",
            "Use proper lighting to reduce glare"
        ]
        self.current_tip = 0
        self.tip_update_time = time.time()
        self.tip_duration = 5  # Change tip every 5 seconds
        self.total_blinks = 0
        self.session_start_time = time.time()
        self.blink_history = []  # To store blink timestamps

        # Add blink goal tracking
        self.blink_goal_per_minute = 15  # Recommended healthy blinks per minute
        self.minute_start_time = time.time()
        self.current_minute_blinks = 0
        self.reset_blink_stats()

        self.ear_values = []  # List to store recent EAR values
        self.smoothing_window = 5  # Number of frames to average

    def reset_blink_stats(self):
        """Reset all blink-related statistics"""
        self.minute_start_time = time.time()
        self.current_minute_blinks = 0
        self.blinks_per_minute = 0
        self.blink_history = []

    def eye_aspect_ratio(self, eye):
        """Compute the eye aspect ratio (EAR) for given eye landmarks."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_ear(self, frame, gray):
        """Detect faces and calculate the average eye aspect ratio (EAR)."""
        rects = self.detector(gray, 0)

        if not rects:  # No faces detected
            return 0.0  # Return a default value

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            if len(self.ear_values) >= self.smoothing_window:
                self.ear_values.pop(0)  # Remove the oldest value
            self.ear_values.append((leftEAR + rightEAR) / 2)  # Average EAR

        # Calculate the average EAR
        if self.ear_values:
            self.ear = sum(self.ear_values) / len(self.ear_values)
        return self.ear

    def update_blink_count(self, EAR):
        if EAR < self.EYE_AR_THRESH:
            self.counter += 1
        else:
            if self.counter >= self.EYE_AR_CONSEC_FRAMES_MIN:
                self.counter = 0
                return True
            self.counter = 0
        return False

    def process_frame(self, frame):
        if frame is None:
            return None, None, False
            
        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        EAR = self.calculate_ear(frame, gray)
        blink_occurred = self.update_blink_count(EAR)
        
        # Update blink count and check for minute reset
        current_time = time.time()
        if current_time - self.minute_start_time >= 60:
            self.blinks_per_minute = self.current_minute_blinks
            self.current_minute_blinks = 0
            self.minute_start_time = current_time
        
        if blink_occurred:
            self.current_minute_blinks += 1
            self.blink_history.append(current_time)
            # Keep only last minute's blinks
            self.blink_history = [t for t in self.blink_history if current_time - t <= 60]
        
        self._draw_enhanced_visualization(frame, EAR)
        return frame, EAR, blink_occurred

    def _draw_enhanced_visualization(self, frame, ear):
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_height = 100  # Increased height
        panel_margin = 20
        panel_y = h - panel_height - panel_margin
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (panel_margin, panel_y),
                     (w - panel_margin, h - panel_margin),
                     self.colors['background'],
                     cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Left section: Blinks per Minute
        count_x = panel_margin + 40
        count_y = panel_y + 40
        
        # Calculate color based on blink rate
        rate_color = self.colors['healthy'] if self.current_minute_blinks >= 12 else self.colors['warning']
        
        # Draw blinks per minute (larger)
        cv2.putText(frame,
                   f"{self.current_minute_blinks}",
                   (count_x, count_y),
                   self.font, 1.5, rate_color, 2, cv2.LINE_AA)
        cv2.putText(frame,
                   "blinks/min",
                   (count_x, count_y + 30),
                   self.font, 0.7, self.colors['secondary'], 2, cv2.LINE_AA)
        
        # Right section: EAR Value
        ear_x = w - panel_margin - 150
        if ear is not None:
            ear_text = f"EAR: {ear:.2f}"
            cv2.putText(frame,
                       ear_text,
                       (ear_x, count_y),
                       self.font, 0.8, self.colors['text'], 2, cv2.LINE_AA)
