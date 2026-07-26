"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""
import asyncio
import base64
import io
import json
import os
import sqlite3
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from fastapi.routing import APIRouter
from fastapi import WebSocket, WebSocketDisconnect, Body, HTTPException
from pydantic import BaseModel, EmailStr
from starlette.responses import JSONResponse
from typing import Dict, Optional

from src.models.models import initialize_database
from src.pipeline.main_pipeline import SitBlinkSipPipeline
from fastapi.responses import StreamingResponse

from src.utils.utils import config_reader
from src.utils.queue_manager import CameraQueueManager

pipeline = SitBlinkSipPipeline()
db = initialize_database()
sit_blink_router = APIRouter(tags=["SitBlink"])

config = config_reader()

eye_blink_det_dir = config['frame_save']['eye_blink_det_dir']
posture_det_dir = config['frame_save']['posture_det_dir']
output_dir = config['frame_save']['output_dir']

queue_manager = CameraQueueManager()

# Frames arrive ~10/sec; persisting every one of them adds a synchronous
# sqlite commit to the hot per-frame loop. Throttle writes to reduce lag.
DB_WRITE_INTERVAL_SECONDS = 1.0


class StreamRequest(BaseModel):
    settings: Dict[str, bool]
    user_id: str
    person_id: Optional[int] = None


class PersonRequest(BaseModel):
    name: str


class WaterBreakIntervalRequest(BaseModel):
    interval: int


class DetectionThresholdsRequest(BaseModel):
    ear_threshold: float
    posture_angle_threshold: float
    posture_displacement_threshold: float


class WaitlistRequest(BaseModel):
    email: EmailStr
    full_name: str
    profession: str


active_connections = set()


@sit_blink_router.post("/start-camera-session")
async def start_camera_session(request: StreamRequest):
    session = await queue_manager.create_session(
        request.user_id,
        request.settings,
        request.person_id
    )
    return {
        "status": "success",
        "session": session
    }


@sit_blink_router.post("/persons")
async def create_person(request: PersonRequest):
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    person = db.create_person(request.name.strip())
    return {"status": "success", "person": person}


@sit_blink_router.get("/persons")
async def list_persons():
    return {"persons": db.list_persons()}


@sit_blink_router.put("/persons/{person_id}/water-break-interval")
async def update_water_break_interval(person_id: int, request: WaterBreakIntervalRequest):
    if request.interval <= 0:
        raise HTTPException(status_code=400, detail="Interval must be a positive integer")
    person = db.update_water_break_interval(person_id, request.interval)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"status": "success", "person": person}


@sit_blink_router.put("/persons/{person_id}/detection-thresholds")
async def update_detection_thresholds(person_id: int, request: DetectionThresholdsRequest):
    if not (0 < request.ear_threshold < 1):
        raise HTTPException(status_code=400, detail="ear_threshold must be between 0 and 1")
    if not (0 < request.posture_angle_threshold <= 180):
        raise HTTPException(status_code=400, detail="posture_angle_threshold must be between 0 and 180")
    if not (0 < request.posture_displacement_threshold < 2):
        raise HTTPException(status_code=400, detail="posture_displacement_threshold must be between 0 and 2")

    person = db.update_detection_thresholds(
        person_id,
        request.ear_threshold,
        request.posture_angle_threshold,
        request.posture_displacement_threshold,
    )
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"status": "success", "person": person}


def _seconds_until_next_water_break(person: dict) -> float:
    """Compute time until the next reminder from the persisted schedule anchor,
    so a page refresh (which just reconnects) never resets the countdown."""
    interval_seconds = max(person.get("water_break_interval") or 30, 1) * 60
    started_at_str = person.get("water_break_started_at")
    try:
        started_at = datetime.strptime(started_at_str, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        started_at = datetime.utcnow()

    elapsed = max((datetime.utcnow() - started_at).total_seconds(), 0)
    intervals_passed = elapsed // interval_seconds
    next_due_at = (intervals_passed + 1) * interval_seconds
    return max(next_due_at - elapsed, 1.0)


@sit_blink_router.websocket("/ws/water-break/{person_id}")
async def water_break_websocket(websocket: WebSocket, person_id: int):
    person = db.get_person(person_id)
    if not person:
        await websocket.close(code=4004, reason="Person not found")
        return

    await websocket.accept()
    POLL_INTERVAL_SECONDS = 5
    try:
        while True:
            # Re-read on every tick so an interval change (via the settings page)
            # takes effect within a few seconds instead of waiting out a stale sleep.
            person = db.get_person(person_id) or person
            remaining = _seconds_until_next_water_break(person)
            if remaining <= POLL_INTERVAL_SECONDS:
                await asyncio.sleep(remaining)
                await websocket.send_json({"type": "water_break"})
            else:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
    except (WebSocketDisconnect, RuntimeError):
        pass
@sit_blink_router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    posture: bool = False,
    eye_blink: bool = False
):
    session = await queue_manager.get_session(user_id)
    if not session:
        await websocket.close(code=4000, reason="No active session found")
        return

    person_id = session.get('person_id')

    # The detectors are shared, process-wide instances (this app targets a single
    # local user at a time), so apply the connecting person's saved sensitivity
    # thresholds to them for the duration of this session.
    if person_id is not None:
        person_settings = db.get_person(person_id)
        if person_settings:
            pipeline.blink_detector.EYE_AR_THRESH = person_settings["ear_threshold"]
            pipeline.posture_detector.angle_threshold = person_settings["posture_angle_threshold"]
            pipeline.posture_detector.displacement_threshold = person_settings["posture_displacement_threshold"]

    await websocket.accept()
    await queue_manager.update_session(user_id, "active")

    # Frames arrive faster (~10/sec) than a single dlib+mediapipe pass can usually be
    # processed. Reading frames strictly in order behind that processing means the
    # backend falls further and further behind — the "watching a minute-old frame"
    # symptom. Decoupling receive from processing and always working on the latest
    # frame (silently dropping any that piled up while busy) keeps latency bounded
    # to roughly one processing pass, without changing what actually gets analyzed.
    latest_frame = {"data": None}
    new_frame_event = asyncio.Event()
    stop_event = asyncio.Event()

    async def receive_frames():
        try:
            while not stop_event.is_set():
                data = await websocket.receive_bytes()
                latest_frame["data"] = data
                new_frame_event.set()
        except Exception:
            stop_event.set()

    async def process_frames():
        last_db_write = 0.0
        try:
            pipeline.validate(
                session['settings']['posture'],
                session['settings']['eye_blink']
            )
            pipeline.start()

            while not stop_event.is_set():
                await new_frame_event.wait()
                new_frame_event.clear()
                frame_data = latest_frame["data"]
                if frame_data is None:
                    continue
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                frame_response = {}
                now = time.monotonic()
                should_persist = now - last_db_write >= DB_WRITE_INTERVAL_SECONDS

                if session['settings']['eye_blink']:
                    processed_frame_eye_blink, ear, blink = pipeline.blink_detector.process_frame(frame)
                    if processed_frame_eye_blink is not None:
                        frame_response["eye_blink_image"] = await convert_frame_to_webp_base64(processed_frame_eye_blink)
                        frame_response["eye_blink_data"] = {
                            "ear": ear,
                            "blink": blink
                        }
                        if should_persist:
                            db.insert_eye_data(ear, blink, person_id=person_id)

                if session['settings']['posture']:
                    processed_frame_posture, head_tilt, displacement_ratio, posture_status = pipeline.posture_detector.process_frame(frame)
                    if processed_frame_posture is not None:
                        frame_response["posture_image"] = await convert_frame_to_webp_base64(processed_frame_posture)
                        frame_response["posture_data"] = {
                            "head_tilt": head_tilt,
                            "displacement_ratio": displacement_ratio,
                            "status": posture_status
                        }
                        if should_persist:
                            db.insert_posture_data(
                                head_tilt, displacement_ratio,
                                posture_status == "Good Posture",
                                person_id=person_id
                            )

                if should_persist:
                    last_db_write = now

                if frame_response:
                    await websocket.send_json(frame_response)
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            stop_event.set()

    receiver_task = asyncio.create_task(receive_frames())
    processor_task = asyncio.create_task(process_frames())

    try:
        await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        stop_event.set()
        receiver_task.cancel()
        processor_task.cancel()
        await queue_manager.update_session(user_id, "inactive")
        pipeline.stop()


async def eye_blink_detection():
    processed_frame = pipeline.eye_blink_detection()
    return processed_frame


async def posture_detection():
    processed_frame = pipeline.posture_detection()
    return processed_frame


async def convert_frame_to_webp_base64(frame: np.ndarray) -> str:
    if frame is not None:
        # Reduced dimensions
        TARGET_WIDTH = 640
        TARGET_HEIGHT = 360
        
        # Calculate scaling to maintain aspect ratio
        h, w = frame.shape[:2]
        aspect = w / h
        
        if aspect > TARGET_WIDTH / TARGET_HEIGHT:
            new_w = TARGET_WIDTH
            new_h = int(TARGET_WIDTH / aspect)
            pad_top = (TARGET_HEIGHT - new_h) // 2
            pad_bottom = TARGET_HEIGHT - new_h - pad_top
            pad_left = 0
            pad_right = 0
        else:
            new_h = TARGET_HEIGHT
            new_w = int(TARGET_HEIGHT * aspect)
            pad_left = (TARGET_WIDTH - new_w) // 2
            pad_right = TARGET_WIDTH - new_w - pad_left
            pad_top = 0
            pad_bottom = 0
            
        # Resize frame
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Add padding
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="WebP", quality=85, method=4)  # Reduced quality and faster compression
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/webp;base64,{img_base64}"
    return None


@sit_blink_router.get("/get_posture_data")
async def get_posture_data(minutes: int = 20, person_id: Optional[int] = None):
    try:
        data = db.get_recent_posture_data(minutes=minutes, person_id=person_id)

        return {"data": data}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sit_blink_router.get("/get_eye_data")
async def get_eye_data(minutes: int = 20, person_id: Optional[int] = None):
    try:
        data = db.get_recent_eye_data(minutes=minutes, person_id=person_id)

        return {"data": data}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sit_blink_router.get("/session/{session_id}")
async def get_session(session_id: str):
    session = await queue_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": "success",
        "session": session
    }
