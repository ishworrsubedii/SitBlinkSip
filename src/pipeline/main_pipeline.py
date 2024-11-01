"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
import time
import cv2
import os
from src.models.models import initialize_database
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
    def __init__(self):
        self.video_source = 0

        self.blink_detector = BlinkDetector(
            shape_predictor_path=shape_predictor_path,
            ear_threshold=ear_threshold,
            ear_consec_frames_min=ear_consec_frames_min,
            ear_consec_frames_max=ear_consec_frames_max
        )

        self.posture_detector = PostureDetector()

        # db
        self.db = initialize_database()

    def validate(self, posture=False, eye_blink=False):
        self.posture = posture
        self.eye_blink = eye_blink

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

    def eye_blink_detection(self):
        processed_files = set()

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

                    if not self.posture:
                        processed_files.add(file)
                        time.sleep(0.1)
                        os.remove(file_path)
                    else:
                        move_file(file_path, posture_det_dir)

                    return processed_frame

                except Exception as e:
                    print(f"Error processing frame {file}: {str(e)}")
                    continue

    def posture_detection(self):
        processed_files = set()

        if os.path.exists(posture_det_dir):
            frame_files = sorted(os.listdir(posture_det_dir))

            for file in frame_files:
                file_path = os.path.join(posture_det_dir, file)
                frame = cv2.imread(file_path)

                processed_frame, head_tilt, displacement_ratio, posture_status = self.posture_detector.process_frame(
                    frame)
                self.db.insert_posture_data(head_tilt, displacement_ratio, posture_status)
                print(
                    f"Head Tilt: {head_tilt}, Displacement Ratio: {displacement_ratio}, Posture Status: {posture_status}")

                processed_files.add(file)
                os.remove(file_path)

                return processed_frame
