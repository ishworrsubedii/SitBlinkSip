"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import cv2

from src.services.posture_det_service.posture_det import PostureDetector

if __name__ == '__main__':
    posture_system = PostureDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image, head_tilt, displacement_ratio, posture_status = posture_system.process_frame(frame)

        cv2.imshow('Posture Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
