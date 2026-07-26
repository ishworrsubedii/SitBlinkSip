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

eye_blink_det_dir = config['frame_save']['eye_blink_det_dir']
posture_det_dir = config['frame_save']['posture_det_dir']


class SitBlinkSipPipeline:
    def __init__(self):
        self.eye_blink = None
        self.posture = None
        self.video_source = 0

        self.blink_detector = BlinkDetector(
            shape_predictor_path=shape_predictor_path,
            ear_threshold=0.15,
            ear_consec_frames_min=1,
            ear_consec_frames_max=3
        )

        self.posture_detector = PostureDetector()

        # db
        self.db = initialize_database()

        self.active = False

    def start(self):
        self.active = True
        self.blink_detector.reset_blink_stats()

    def stop(self):
        self.active = False
        # Reset statistics when stopping
        self.blink_detector.reset_blink_stats()

    def validate(self, posture=False, eye_blink=False):
        self.posture = posture
        self.eye_blink = eye_blink

        if posture and eye_blink:
            if os.path.exists(eye_blink_det_dir):
                del_directory(eye_blink_det_dir)
            if os.path.exists(posture_det_dir):
                del_directory(posture_det_dir)
            os.makedirs(eye_blink_det_dir, exist_ok=True)
            os.makedirs(posture_det_dir, exist_ok=True)
            self.output_folder = eye_blink_det_dir



        elif posture:
            if os.path.exists(posture_det_dir):
                del_directory(posture_det_dir)
            os.makedirs(posture_det_dir, exist_ok=True)
            self.output_folder = posture_det_dir


        elif eye_blink:
            if os.path.exists(eye_blink_det_dir):
                del_directory(eye_blink_det_dir)
            os.makedirs(eye_blink_det_dir, exist_ok=True)
            self.output_folder = eye_blink_det_dir



        else:
            raise ValueError("At least one of posture or eye_blink should be True")

    def eye_blink_detection(self):
        processed_files = set()
        latest_frame = None

        if os.path.exists(self.output_folder):
            frame_files = sorted(os.listdir(self.output_folder))
            
            # Only process the latest frame
            if frame_files:
                latest_file = frame_files[-1]
                file_path = os.path.join(self.output_folder, latest_file)
                
                try:
                    frame = cv2.imread(file_path)
                    if frame is not None:
                        processed_frame, ear, blink = self.blink_detector.process_frame(frame)
                        print(f"EAR: {ear}, Blink: {blink}")
                        self.db.insert_eye_data(ear, blink)
                        latest_frame = processed_frame

                    # Clean up processed file
                    if not self.posture:
                        os.remove(file_path)
                    else:
                        move_file(file_path, posture_det_dir)
                        
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    
        return latest_frame

    def posture_detection(self):
        latest_frame = None

        if os.path.exists(posture_det_dir):
            frame_files = sorted(os.listdir(posture_det_dir))
            
            # Only process the latest frame
            if frame_files:
                latest_file = frame_files[-1]
                file_path = os.path.join(posture_det_dir, latest_file)
                
                try:
                    frame = cv2.imread(file_path)
                    if frame is not None:
                        processed_frame, head_tilt, displacement_ratio, posture_status = self.posture_detector.process_frame(frame)
                        self.db.insert_posture_data(head_tilt, displacement_ratio, posture_status == "Good Posture")
                        latest_frame = processed_frame
                    
                    # Clean up processed file
                    os.remove(file_path)
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    
        return latest_frame
