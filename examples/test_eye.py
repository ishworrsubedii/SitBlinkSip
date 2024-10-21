"""
project @ SitBlinkSip
created @ 2024-10-17
author  @ github/ishworrsubedii
"""
import configparser
import os
import cv2
import time
import threading
from datetime import datetime

from src.services.database.csv_data_save import CSVDatabase
from src.services.eye_blink_service.eye_blink import BlinkDetector
from src.utils.utils import send_blink_warning_notification

config = configparser.ConfigParser()
config.read('config.ini')

shape_predictor_path = config['blink_detector']['shape_predictor_path']
ear_threshold = float(config['blink_detector']['ear_threshold'])
ear_consec_frames_min = int(config['blink_detector']['ear_consec_frames_min'])
ear_consec_frames_max = int(config['blink_detector']['ear_consec_frames_max'])


class FrameProcessor:
    def __init__(self, blink_detector_model_path, output_folder, video_source=0):
        """
        Initialize the FrameProcessor class.

        :param blink_detector_model_path: Path to the blink detector model file.
        :param output_folder: Folder where frames will be saved.
        :param video_source: Video source (0 for webcam, or file path for a video).
        """

        self.blink_detector = BlinkDetector(
            shape_predictor_path=shape_predictor_path,
            ear_threshold=ear_threshold,
            ear_consec_frames_min=ear_consec_frames_min,
            ear_consec_frames_max=ear_consec_frames_max
        )

        # Clean up output folder if it exists
        if os.path.exists(output_folder):
            for file in os.listdir(output_folder):
                os.remove(os.path.join(output_folder, file))

        self.output_folder = output_folder
        self.video_stream = cv2.VideoCapture(video_source)
        self.stop_event = threading.Event()
        self.csv_db = CSVDatabase('outputs/DB/eye_blink_record.csv')

        # Blink tracking variables
        self.total_blinks = 0
        self.minute_start_time = datetime.now()
        self.blinks_in_current_minute = 0
        self.last_ear = 0.0
        self.blink_detected = False

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)



    def process_saved_frames(self):
        """Thread to process saved frames using the blink detector and delete them after processing."""
        processed_files = set()

        while not self.stop_event.is_set():
            current_time = datetime.now()

            if (current_time - self.minute_start_time).total_seconds() >= 60:
                if self.blinks_in_current_minute < 20:
                    print("\n" + "!" * 50)
                    warning_msg = f"WARNING: Only {self.blinks_in_current_minute} blinks in the last minute! (Recommended: 20+)"
                    print(warning_msg)
                    print("!" * 50 + "\n")
                    send_blink_warning_notification(
                        message=warning_msg)

                self.minute_start_time = current_time
                self.blinks_in_current_minute = 0
                print(f"\nStarting new minute tracking at {current_time.strftime('%H:%M:%S')}")

            frame_files = sorted(os.listdir(self.output_folder))

            for file in frame_files:
                if file in processed_files:
                    continue

                file_path = os.path.join(self.output_folder, file)
                frame = cv2.imread(file_path)

                try:
                    processed_frame, total_blink, ear = self.blink_detector.process_frame(frame)

                    if ear < 0.2 and self.last_ear >= 0.2:  # Blink started
                        self.blink_detected = True
                    elif ear >= 0.2 and self.last_ear < 0.2 and self.blink_detected:
                        current_time = datetime.now()
                        self.csv_db.insert_record({
                            'id': current_time.strftime("%Y%m%d_%H%M%S_%f"),
                            'time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'eyeBlink': total_blink,
                            'ear_value': round(ear, 3)
                        })

                        self.total_blinks = total_blink
                        self.blinks_in_current_minute += 1
                        self.blink_detected = False

                        print(f"\nBlink detected! Count in current minute: {self.blinks_in_current_minute}")

                    self.last_ear = ear


                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue

                # Print current stats
                remaining_time = 60 - (datetime.now() - self.minute_start_time).total_seconds()
                print(f"\rCurrent minute stats - Blinks: {self.blinks_in_current_minute} | "
                      f"Time remaining: {int(remaining_time)}s | EAR: {ear:.2f}", end="")

                processed_files.add(file)
                os.remove(file_path)

            time.sleep(0.1)

    def start(self):
        """Start the threads for saving and processing frames."""
        try:
            # Start the frame-saving thread
            save_thread = threading.Thread(target=self.save_frames)
            save_thread.start()

            # Start the frame-processing thread
            process_thread = threading.Thread(target=self.process_saved_frames)
            process_thread.start()

            # Wait for threads to complete
            save_thread.join()
            process_thread.join()

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
        finally:
            # Cleanup
            self.stop_event.set()
            self.video_stream.release()
            print("\nProgram terminated.")


if __name__ == "__main__":
    processor = FrameProcessor(
        "resources/dlib_models/shape_predictor_68_face_landmarks.dat",
        "outputs/saved_frames"
    )
    processor.start()
