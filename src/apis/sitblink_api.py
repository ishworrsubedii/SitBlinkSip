"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""

from fastapi.routing import APIRouter

from src.models.models import initialize_database
from src.pipeline.main_pipeline import SitBlinkSipPipeline

pipeline = SitBlinkSipPipeline()
db = initialize_database()
sitblink = APIRouter(prefix="/sitblink", tags=["SitBlink"])


@sitblink.post("/posture_detection")
async def start_posture_detection():
    pass


@sitblink.post("/eye_blink_detection")
async def start_eye_blink_detection():
    pass


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
