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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.angle_threshold = angle_threshold
        self.displacement_threshold = displacement_threshold

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.display = draw
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'white': (255, 255, 255)
        }

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def _calculate_distance(self, a, b):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def _get_landmarks(self, landmarks):
        """Extract relevant landmarks for posture detection."""
        return {
            'left_shoulder': [
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ],
            'right_shoulder': [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ],
            'left_ear': [
                landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y
            ],
            'right_ear': [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y
            ],
            'nose': [
                landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[self.mp_pose.PoseLandmark.NOSE.value].y
            ]
        }

    def process_frame(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        # Calculate displacement ratios
        forward_displacement = self._calculate_distance(ear_midpoint, shoulder_midpoint)
        shoulder_width = self._calculate_distance(
            landmarks_dict['left_shoulder'],
            landmarks_dict['right_shoulder']
        )
        displacement_ratio = forward_displacement / shoulder_width

        # Determine posture status
        bad_posture = head_tilt < self.angle_threshold or displacement_ratio < self.displacement_threshold
        posture_status = "Bad Posture" if bad_posture else "Good Posture"
        color = self.colors['red'] if bad_posture else self.colors['green']

        # Draw visualizations
        if self.display:
            self._draw_visualization(
                image,
                landmarks_dict,
                head_tilt,
                displacement_ratio,
                posture_status,
                color
            )

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        else:
            pass

        # Prepare posture data
        posture_data = {
            'head_tilt': head_tilt,
            'displacement_ratio': displacement_ratio,
            'posture_status': posture_status,
            'landmarks': landmarks_dict
        }

        return image, posture_data

    def _draw_visualization(self, image, landmarks, head_tilt, displacement_ratio,
                            posture_status, color):
        """Draw visualization elements on the frame."""
        # Draw landmark points
        for point in landmarks.values():
            cv2.circle(
                image,
                (int(point[0] * image.shape[1]), int(point[1] * image.shape[0])),
                5,
                self.colors['green'],
                -1
            )

        cv2.putText(image, f'Head Tilt: {head_tilt:.2f}', (10, 30),
                    self.font, 0.5, self.colors['white'], 2)
        cv2.putText(image, f'Forward Displacement: {displacement_ratio:.2f}',
                    (10, 60), self.font, 0.5, self.colors['white'], 2)
        cv2.putText(image, posture_status, (10, 90),
                    self.font, 0.9, color, 2)

    def __del__(self):
        self.pose.close()
