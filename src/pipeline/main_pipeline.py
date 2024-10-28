"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import threading
import time

import cv2
import os
import imagehash
from PIL import Image

from src.models.models import initialize_database
from src.services.frame_capture.frame_capture_save import FrameCaptureSave
from src.services.eye_blink_service.eye_blink import BlinkDetector
from src.services.posture_det_service.posture_det import PostureDetector
from src.utils.utils import config_reader, move_file
from src.utils.utils import del_directory

config = config_reader()
shape_predictor_path = config['blink_detector']['shape_predictor_path']
ear_threshold = float(config['blink_detector']['ear_threshold'])
ear_consec_frames_min = int(config['blink_detector']['ear_consec_frames_min'])
ear_consec_frames_max = int(config['blink_detector']['ear_consec_frames_max'])

eye_blink_det_dir = "outputs/eye_blink_frames"
posture_det_dir = "outputs/posture_det_frames"


class SitBlinkSipPipeline:
    def __init__(self, display=False, hash_threshold=0.1):
        self.posture = False
        self.eye_blink = False
        self.video_source = 0

        self.frame_capture_obj = FrameCaptureSave(video_source=self.video_source)
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
        self.posture_detection_thread = None

        # Shared state
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_hash = None

        # posture detection
        self.posture_detector = PostureDetector()

        # threading control
        self.frame_capture_save_alive = False
        self.eye_blink_detection_alive = False
        self.posture_detection_alive = False

        # db
        self.db = initialize_database()

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

    def start_pipeline(self, posture=False, eye_blink=False, video_source=0):
        self.posture = posture
        self.eye_blink = eye_blink
        self.video_source = video_source

        if posture and eye_blink:
            if os.path.exists(eye_blink_det_dir):
                del_directory(eye_blink_det_dir)
            if os.path.exists(posture_det_dir):
                del_directory(posture_det_dir)
            os.mkdir(eye_blink_det_dir)
            os.mkdir(posture_det_dir)
            self.output_folder = eye_blink_det_dir



        elif posture:
            if os.path.exists(posture_det_dir):
                del_directory(posture_det_dir)
            os.mkdir(posture_det_dir)
            self.output_folder = posture_det_dir


        elif eye_blink:
            if os.path.exists(eye_blink_det_dir):
                del_directory(eye_blink_det_dir)
            os.mkdir(eye_blink_det_dir)
            self.output_folder = eye_blink_det_dir



        else:
            raise ValueError("At least one of posture or eye_blink should be True")

        self.frame_capture_thread = threading.Thread(target=self.frame_capture_save)
        self.blink_detection_thread = threading.Thread(target=self.eye_blink_detection)
        self.posture_detection_thread = threading.Thread(target=self.posture_detection)
        self.frame_capture_thread.start()
        self.frame_capture_save_alive = True

        if self.eye_blink and self.posture:
            self.eye_blink_detection_alive = True
            self.blink_detection_thread.start()
            self.posture_detection_alive = True
            self.posture_detection_thread.start()

        elif self.posture:
            self.posture_detection_alive = True
            self.posture_detection_thread.start()

        elif self.eye_blink:
            self.eye_blink_detection_alive = True
            self.blink_detection_thread.start()

    def stop_pipeline(self):
        self.stop_event.set()
        if self.frame_capture_thread:
            print("Stopping frame capture thread...")
            self.frame_capture_thread.join()
            self.frame_capture_obj.release()
        if self.blink_detection_thread:
            print("Stopping eye blink detection thread...")
            self.blink_detection_thread.join()
        if self.posture_detection_thread:
            print("Stopping posture detection thread...")
            self.posture_detection_thread.join()

    def frame_capture_save(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        while not self.stop_event.is_set():
            frame = self.frame_capture_obj.frame_capture()
            if frame is None:
                break

            if self.should_process_frame(frame):
                with self.frame_lock:
                    self.current_frame = frame.copy()

                self.frame_capture_obj.frame_save(frame, output_folder=self.output_folder)

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
                        print(f"EAR: {ear}, Blink: {blink}")
                        self.db.insert_eye_data(ear, blink)

                        if self.display:
                            cv2.imshow('Eye-Blink', processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.stop_event.set()
                                break
                        if not self.posture_detection_alive:
                            processed_files.add(file)
                            time.sleep(0.1)
                            os.remove(file_path)
                        else:
                            move_file(file_path, posture_det_dir)

                    except Exception as e:
                        print(f"Error processing frame {file}: {str(e)}")
                        continue

    def posture_detection(self):
        processed_files = set()
        last_frame_hash = None
        posture_hash_threshold = 0.3

        while not self.stop_event.is_set():
            if os.path.exists(posture_det_dir):
                frame_files = sorted(os.listdir(posture_det_dir))

                for file in frame_files:
                    file_path = os.path.join(posture_det_dir, file)
                    frame = cv2.imread(file_path)

                    # Skip already processed files
                    if file in processed_files:
                        os.remove(file_path)  # Remove unprocessed file to save storage
                        continue

                    try:
                        # Compute the hash of the current frame
                        current_hash = self.compute_frame_hash(frame)

                        # Compare with last hash
                        if last_frame_hash is not None:
                            hash_diff = abs(current_hash - last_frame_hash)
                            if hash_diff < posture_hash_threshold:
                                os.remove(file_path)  # Remove unprocessed file to save storage
                                continue

                        # Update last frame hash
                        last_frame_hash = current_hash

                        # Process frame if hash check passed
                        processed_frame, head_tilt, displacement_ratio, posture_status = self.posture_detector.process_frame(
                            frame)
                        self.db.insert_posture_data(head_tilt, displacement_ratio, posture_status)
                        print(
                            f"Head Tilt: {head_tilt}, Displacement Ratio: {displacement_ratio}, Posture Status: {posture_status}")

                        processed_files.add(file)
                        os.remove(file_path)

                    except Exception as e:
                        print(f"Error processing frame {file}: {str(e)}")
                        continue
