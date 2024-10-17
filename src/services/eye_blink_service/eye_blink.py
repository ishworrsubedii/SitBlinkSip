"""
project @ SitBlinkSip
created @ 2024-10-17
author  @ github/ishworrsubedii
"""
import threading
from scipy.spatial import distance as dist
from imutils import face_utils
import cv2
import dlib
import datetime


class BlinkDetector:
    def __init__(self, shape_predictor_path, ear_threshold=0.25, ear_consec_frames_min=2, ear_consec_frames_max=5):
        self.EYE_AR_THRESH = ear_threshold
        self.EYE_AR_CONSEC_FRAMES_MIN = ear_consec_frames_min
        self.EYE_AR_CONSEC_FRAMES_MAX = ear_consec_frames_max
        self.counter = 0
        self.totalBlinks = 0
        self.lock = threading.Lock()
        self.timediff = datetime.datetime.now()
        self.ear = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

    def eye_aspect_ratio(self, eye):
        """Compute the eye aspect ratio (EAR) for given eye landmarks."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_ear(self, frame, gray):
        """Detect faces and calculate the average eye aspect ratio (EAR)."""
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # Draw the convex hull around both eyes
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            self.ear = (leftEAR + rightEAR) / 2.0
        return self.ear

    def update_blink_count(self, EAR):
        """Update blink count based on the current EAR."""
        if EAR < self.EYE_AR_THRESH:
            self.counter += 1
        else:
            if self.EYE_AR_CONSEC_FRAMES_MIN <= self.counter <= self.EYE_AR_CONSEC_FRAMES_MAX:
                self.totalBlinks += 1
            self.counter = 0

    def process_frame(self, frame):
        if frame is None:
            return None
        frame = cv2.resize(frame, (700, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        EAR = self.calculate_ear(frame, gray)

        return frame, self.totalBlinks, EAR
