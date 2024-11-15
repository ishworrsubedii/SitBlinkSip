"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import cv2
import mediapipe as mp
import numpy as np


class PostureDetector:
    def __init__(self, angle_threshold=150.0, displacement_threshold=0.7, draw=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.angle_threshold = angle_threshold
        self.displacement_threshold = displacement_threshold
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.display = draw
        self.colors = {
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'good_posture': (0, 255, 127),  # Brighter green
            'bad_posture': (0, 0, 255)  # Red for bad posture
        }

    def process_frame(self, frame):
        # Reduced dimensions
        TARGET_WIDTH = 640   # Half of 1280
        TARGET_HEIGHT = 360  # Half of 720
        
        # Calculate scaling to maintain aspect ratio
        h, w = frame.shape[:2]
        aspect = w / h
        
        if aspect > TARGET_WIDTH / TARGET_HEIGHT:
            new_w = TARGET_WIDTH
            new_h = int(TARGET_WIDTH / aspect)
            pad_top = (TARGET_HEIGHT - new_h) // 2
            pad_bottom = TARGET_HEIGHT - new_h - pad_top
            pad_left = 0
            pad_right = 0
        else:
            new_h = TARGET_HEIGHT
            new_w = int(TARGET_HEIGHT * aspect)
            pad_left = (TARGET_WIDTH - new_w) // 2
            pad_right = TARGET_WIDTH - new_w - pad_left
            pad_top = 0
            pad_bottom = 0
            
        # Resize frame
        image = cv2.resize(frame, (new_w, new_h))
        
        # Add padding
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Process frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return frame, None

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

        if self.display:
            self._draw_custom_visualization(image, landmarks_dict, posture_status, color)

        return image, head_tilt, displacement_ratio, posture_status

    def _draw_custom_visualization(self, image, landmarks, posture_status, color):
        """Draw minimal visualization with just the status indicator."""
        self._draw_status_label(image, posture_status, color)

    def _draw_status_label(self, image, status, color):
        """Draw a professional, minimal status label."""
        # Font settings
        font_scale = 1.0
        thickness = 2

        # Get text size
        label_size = cv2.getTextSize(status, self.font, font_scale, thickness)[0]

        # Calculate position (centered horizontally, lower part of screen)
        center_x = image.shape[1] // 2
        center_y = int(image.shape[0] * 0.85)  # Position at 85% from top

        # Calculate rectangle coordinates with larger padding
        padding_x = 30
        padding_y = 15
        rect_width = label_size[0] + (padding_x * 2)
        rect_height = label_size[1] + (padding_y * 2)

        rect_start = (
            center_x - rect_width // 2,
            center_y - rect_height // 2
        )
        rect_end = (
            center_x + rect_width // 2,
            center_y + rect_height // 2
        )

        # Draw filled rectangle background
        cv2.rectangle(
            image,
            rect_start,
            rect_end,
            color,
            -1,
            lineType=cv2.LINE_AA
        )

        # Draw white border
        border_thickness = 2
        cv2.rectangle(
            image,
            rect_start,
            rect_end,
            self.colors['white'],
            border_thickness,
            lineType=cv2.LINE_AA
        )

        # Draw text
        text_x = center_x - label_size[0] // 2
        text_y = center_y + label_size[1] // 4
        cv2.putText(
            image,
            status,
            (text_x, text_y),
            self.font,
            font_scale,
            self.colors['white'],
            thickness,
            lineType=cv2.LINE_AA
        )

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
