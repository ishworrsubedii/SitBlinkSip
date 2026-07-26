"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import cv2
import mediapipe as mp
import numpy as np
import os


class PostureDetector:
    def __init__(self, angle_threshold=145.0, displacement_threshold=0.65, draw=True):
        self.mp_pose = mp.solutions.pose    
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )

        self.angle_threshold = angle_threshold
        self.displacement_threshold = displacement_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.display = draw
        self.colors = {
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'good_posture': (46, 204, 113),  # Softer green
            'bad_posture': (231, 76, 60),  # Softer red
            'background': (40, 40, 40),  # Dark gray background
            'text': (236, 240, 241),  # Light gray text
            'secondary_text': (189, 195, 199),  # Lighter gray
            'accent': (52, 152, 219),  # Blue accent
            'background': (0, 0, 0, 0.7)  # Semi-transparent background
        }
        self.posture_score = 100  # Initialize posture score
        self.score_decay = 0.5  # Score decay rate for bad posture
        self.score_gain = 0.3  # Score gain rate for good posture

    def process_frame(self, frame):
        if frame is None:
            return None, None, None, None

        # Keep original resolution for better accuracy
        image = frame.copy()

        # Process frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return frame, None, None, None

        landmarks_dict = self._get_landmarks(results.pose_landmarks.landmark)

        head_tilt = self._calculate_angle(
            landmarks_dict['left_ear'],
            landmarks_dict['nose'],
            landmarks_dict['right_ear']
        )

        ear_midpoint = [
            (landmarks_dict['left_ear'][0] + landmarks_dict['right_ear'][0]) / 2,
            (landmarks_dict['left_ear'][1] + landmarks_dict['right_ear'][1]) / 2
        ]
        shoulder_midpoint = [
            (landmarks_dict['left_shoulder'][0] + landmarks_dict['right_shoulder'][0]) / 2,
            (landmarks_dict['left_shoulder'][1] + landmarks_dict['right_shoulder'][1]) / 2
        ]

        forward_displacement = self._calculate_distance(ear_midpoint, shoulder_midpoint)
        shoulder_width = self._calculate_distance(
            landmarks_dict['left_shoulder'],
            landmarks_dict['right_shoulder']
        )
        displacement_ratio = forward_displacement / shoulder_width

        bad_posture = head_tilt < self.angle_threshold or displacement_ratio < self.displacement_threshold
        posture_status = "Bad Posture" if bad_posture else "Good Posture"
        color = self.colors['bad_posture'] if bad_posture else self.colors['good_posture']

        # Update posture score
        if bad_posture:
            self.posture_score = max(0, self.posture_score - self.score_decay)
        else:
            self.posture_score = min(100, self.posture_score + self.score_gain)

        if self.display:
            self._draw_enhanced_visualization(image, posture_status, color, head_tilt, displacement_ratio)

        return image, head_tilt, displacement_ratio, posture_status

    def _draw_enhanced_visualization(self, image, status, color, head_tilt, displacement_ratio):
        h, w = image.shape[:2]

        # Increased panel dimensions
        panel_height = 100  # Increased from 60
        panel_margin = 20
        panel_y = h - panel_height - panel_margin

        # Create semi-transparent overlay for the bottom panel
        overlay = image.copy()
        cv2.rectangle(overlay,
                      (panel_margin, panel_y),
                      (w - panel_margin, h - panel_margin),
                      (0, 0, 0),
                      cv2.FILLED)

        # Increased transparency
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

        # Left section: Posture Score with larger size
        score_radius = 25  # Increased from 20
        score_center = (panel_margin + score_radius + 20, panel_y + panel_height // 2)

        # Draw score circle
        cv2.circle(image, score_center, score_radius, color, 2, cv2.LINE_AA)

        # Larger score text
        score_text = f"{int(self.posture_score)}"
        score_size = cv2.getTextSize(score_text, self.font, 0.8, 2)[0]  # Increased from 0.6
        cv2.putText(image, score_text,
                    (score_center[0] - score_size[0] // 2, score_center[1] + score_size[1] // 2),
                    self.font, 0.8, self.colors['text'], 2, cv2.LINE_AA)

        # Middle section: Status and Feedback with larger text
        text_x = score_center[0] + score_radius + 40
        status_y = panel_y + 35
        feedback_y = status_y + 30

        cv2.putText(image, status,
                    (text_x, status_y),
                    self.font, 0.8, color, 2, cv2.LINE_AA)  # Increased from 0.6

        # Add detailed feedback
        feedback = self._get_posture_feedback(head_tilt, displacement_ratio)
        cv2.putText(image, feedback,
                    (text_x, feedback_y),
                    self.font, 0.6, self.colors['secondary_text'], 1, cv2.LINE_AA)

    def _calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def _calculate_distance(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def _get_landmarks(self, landmarks):
        return {
            'left_shoulder': [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            'right_shoulder': [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
            'left_ear': [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y],
            'right_ear': [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y],
            'nose': [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
        }

    def __del__(self):
        self.pose.close()

    def _get_posture_feedback(self, head_tilt, displacement_ratio):
        if head_tilt > self.angle_threshold:
            return "Please align your head with shoulders"
        elif displacement_ratio > self.displacement_threshold:
            return "Maintain upright posture"
        else:
            return "Good posture maintained"
