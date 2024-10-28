"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import os
import cv2
import time
from datetime import datetime


class FrameCaptureSave:
    def __init__(self, video_source=0, fps=10):
        self.video_stream = cv2.VideoCapture(video_source)
        self.delay = 1 / fps  # Calculate delay based on FPS
        self.last_save_time = 0  # Track last save timestamp

    def frame_capture(self):
        ret, frame = self.video_stream.read()
        return frame

    def frame_save(self, frame, output_folder):
        current_time = time.time()
        # Check if enough time has passed since the last frame save
        if current_time - self.last_save_time >= self.delay:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.png"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            self.last_save_time = current_time  # Update last save time

    def release(self):
        """Release the video capture resource"""
        if self.video_stream.isOpened():
            self.video_stream.release()
