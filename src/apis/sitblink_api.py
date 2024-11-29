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
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from fastapi.routing import APIRouter
from fastapi import WebSocket, Body, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from typing import Dict
from redis.exceptions import ConnectionError

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


class StreamRequest(BaseModel):
    settings: Dict[str, bool]
    user_id: str


active_connections = set()


@sit_blink_router.post("/start-camera-session")
async def start_camera_session(request: StreamRequest):
    try:
        session = await queue_manager.create_session(
            request.user_id,
            request.settings
        )
        return {
            "status": "success",
            "session": session
        }
    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Session service temporarily unavailable. Please try again later."
        )


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

    await websocket.accept()
    await queue_manager.update_session(user_id, "active")
    
    try:
        pipeline.validate(
            session['settings']['posture'],
            session['settings']['eye_blink']
        )
        pipeline.start()

        while True:
            frame_data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

            frame_response = {}

            if session['settings']['eye_blink']:
                processed_frame_eye_blink, ear, blink = pipeline.blink_detector.process_frame(frame)
                if processed_frame_eye_blink is not None:
                    frame_response["eye_blink_image"] = await convert_frame_to_webp_base64(processed_frame_eye_blink)
                    frame_response["eye_blink_data"] = {
                        "ear": ear,
                        "blink": blink
                    }

            if session['settings']['posture']:
                processed_frame_posture, head_tilt, displacement_ratio, posture_status = pipeline.posture_detector.process_frame(frame)
                if processed_frame_posture is not None:
                    frame_response["posture_image"] = await convert_frame_to_webp_base64(processed_frame_posture)
                    frame_response["posture_data"] = {
                        "head_tilt": head_tilt,
                        "displacement_ratio": displacement_ratio,
                        "status": posture_status
                    }

            if frame_response:
                await websocket.send_json(frame_response)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
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
async def get_posture_data(minutes: int = 20):
    try:
        data = db.get_recent_posture_data(minutes=minutes)

        return {"data": data}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sit_blink_router.get("/get_eye_data")
async def get_eye_data(minutes: int = 20):
    try:
        data = db.get_recent_eye_data(minutes=minutes)

        return {"data": data}
    except Exception as e:
        return {"message": f"Error: {e}"}


@sit_blink_router.get("/session/{session_id}")
async def get_session(session_id: str):
    try:
        session = await queue_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "status": "success",
            "session": session
        }
    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Session service unavailable"
        )
