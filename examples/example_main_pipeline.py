"""
project @ SitBlinkSip
created @ 2024-10-22
author  @ github/ishworrsubedii
"""
import cv2

from src.pipeline.main_pipeline import SitBlinkSipPipeline

if __name__ == '__main__':
    pipeline = SitBlinkSipPipeline(display=False, hash_threshold=0)

    try:
        pipeline.start_pipeline()

        while not pipeline.stop_event.is_set():
            pass

    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
    finally:
        pipeline.stop_pipeline()
        cv2.destroyAllWindows()
