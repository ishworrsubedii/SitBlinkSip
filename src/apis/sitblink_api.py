"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""
from typing import Any

import cv2
from fastapi import File
from fastapi.routing import APIRouter

from src.models.models import initialize_database
from src.pipeline.main_pipeline import SitBlinkSipPipeline

pipeline = SitBlinkSipPipeline(display=False, hash_threshold=0)
db = initialize_database()
sitblink = APIRouter(prefix="/sitblink", tags=["SitBlink"])


@sitblink.post("/start_pipeline")
async def start_pipeline(video_source: str = File(...), posture: bool = True, eye_blink: bool = True) -> Any:
    try:
        pipeline.start_pipeline(posture=posture, eye_blink=eye_blink, video_source=video_source)
        pipeline.stop_event.is_set()

        return {"message": "Pipeline started successfully"}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sitblink.post("/stop_pipeline")
async def stop_pipeline():
    try:
        pipeline.stop_pipeline()
        cv2.destroyAllWindows()

        return {"message": "Pipeline stopped successfully"}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sitblink.get("/get_posture_data")
async def get_posture_data(minutes: int = 20):
    try:
        data = db.get_recent_posture_data(minutes=minutes)

        return {"data": data}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sitblink.get("/get_eye_data")
async def get_eye_data(minutes: int = 20):
    try:
        data = db.get_recent_eye_data(minutes=minutes)

        return {"data": data}
    except Exception as e:
        return {"message": f"Error: {e}"}
