"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import os
import cv2
from datetime import datetime


class FrameCaptureSave:
    def __init__(self, video_source=0):
        self.video_stream = cv2.VideoCapture(video_source)

    def frame_capture(self):
        ret, frame = self.video_stream.read()
        return frame

    def frame_save(self, frame, output_folder):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.png"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, filename), frame)
