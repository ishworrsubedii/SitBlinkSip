"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import threading
import cv2
import os
import imagehash
from PIL import Image
from src.services.frame_capture.frame_capture_save import FrameCaptureSave
from src.services.eye_blink_service.eye_blink import BlinkDetector
from src.utils.utils import config_reader

config = config_reader()
shape_predictor_path = config['blink_detector']['shape_predictor_path']
ear_threshold = float(config['blink_detector']['ear_threshold'])
ear_consec_frames_min = int(config['blink_detector']['ear_consec_frames_min'])
ear_consec_frames_max = int(config['blink_detector']['ear_consec_frames_max'])


class SitBlinkSipPipeline:
    def __init__(self, display=False, video_source=0, hash_threshold=0.1, output_folder='outputs/frames'):
        self.frame_capture_obj = FrameCaptureSave(video_source=video_source)
        self.blink_detector = BlinkDetector(
            shape_predictor_path=shape_predictor_path,
            ear_threshold=ear_threshold,
            ear_consec_frames_min=ear_consec_frames_min,
            ear_consec_frames_max=ear_consec_frames_max
        )
        self.display = display
        self.hash_threshold = hash_threshold

        # Threading control
        self.stop_event = threading.Event()
        self.frame_capture_thread = None
        self.blink_detection_thread = None

        # Shared state
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_hash = None

        # output folder
        self.output_folder = output_folder

        # posture detection
        self.frame_capture_save_alive = False
        self.eye_blink_detection_alive = False
        self.posture_detection_alive = False

    def compute_frame_hash(self, frame):
        """Compute perceptual hash of the frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return imagehash.average_hash(pil_image)

    def should_process_frame(self, frame):
        current_hash = self.compute_frame_hash(frame)

        if self.last_frame_hash is None:
            self.last_frame_hash = current_hash
            return True

        # Calculate hash difference
        hash_diff = abs(current_hash - self.last_frame_hash)
        self.last_frame_hash = current_hash

        if hash_diff >= self.hash_threshold:
            return True
        return False

    def start_pipeline(self):
        self.frame_capture_thread = threading.Thread(target=self.frame_capture_save)
        self.blink_detection_thread = threading.Thread(target=self.eye_blink_detection)

        self.frame_capture_thread.start()
        self.frame_capture_save_alive = True
        self.blink_detection_thread.start()
        self.eye_blink_detection_alive = True

    def stop_pipeline(self):
        self.stop_event.set()
        if self.frame_capture_thread:
            self.frame_capture_thread.join()
        if self.blink_detection_thread:
            self.blink_detection_thread.join()

    def frame_capture_save(self, output_folder='outputs/frames'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        while not self.stop_event.is_set():
            frame = self.frame_capture_obj.frame_capture()
            if frame is None:
                break

            if self.should_process_frame(frame):
                with self.frame_lock:
                    self.current_frame = frame.copy()

                self.frame_capture_obj.frame_save(frame, output_folder=output_folder)

    def eye_blink_detection(self):
        processed_files = set()

        while not self.stop_event.is_set():
            if os.path.exists(self.output_folder):
                frame_files = sorted(os.listdir(self.output_folder))

                for file in frame_files:
                    if file in processed_files:
                        continue

                    file_path = os.path.join(self.output_folder, file)
                    frame = cv2.imread(file_path)

                    try:
                        processed_frame, ear, blink = self.blink_detector.process_frame(frame)
                        # TODO: database insertion

                        if self.display:
                            cv2.imshow('Frame', processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.stop_event.set()
                                break
                        if not self.posture_detection_alive:

                            processed_files.add(file)
                            os.remove(file_path)
                        else:
                            # TODO : add logic to move the image into the another folder
                            pass

                    except Exception as e:
                        print(f"Error processing frame {file}: {str(e)}")
                        continue
