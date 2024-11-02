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
from fastapi import WebSocket
from starlette.responses import JSONResponse

from src.models.models import initialize_database
from src.pipeline.main_pipeline import SitBlinkSipPipeline
from fastapi.responses import StreamingResponse

from src.utils.utils import config_reader

pipeline = SitBlinkSipPipeline()
db = initialize_database()
sit_blink_router = APIRouter(prefix="/sitblink", tags=["SitBlink"])

config = config_reader()

eye_blink_det_dir = config['frame_save']['eye_blink_det_dir']
posture_det_dir = config['frame_save']['posture_det_dir']
output_dir = config['frame_save']['output_dir']

active_connections = set()


@sit_blink_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, posture: bool = False, eye_blink: bool = False):
    await websocket.accept()
    active_connections.add(websocket)
    frame_count = 0

    if posture and eye_blink:
        upload_dir = eye_blink_det_dir
    elif posture:
        upload_dir = posture_det_dir

    elif eye_blink:
        upload_dir = eye_blink_det_dir

    else:
        return {"message": "Please select either posture or eye_blink or both."}

    try:
        while True:
            frame_data = await websocket.receive_bytes()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}.jpg"
            filepath = os.path.join(upload_dir, filename)

            # Save original frame
            with open(filepath, "wb") as f:
                f.write(frame_data)

            print(f"Saved and processed frame: {filename}")
            frame_count += 1

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        active_connections.remove(websocket)


@sit_blink_router.post("/clear-frames")
async def clear_frames():
    try:
        for file in os.listdir(output_dir):
            if file.endswith('.jpg'):
                os.remove(os.path.join(output_dir, file))
                response = {"message": "All frames cleared",
                            "status": 200}
            else:
                response = {"message": "No frames to clear",
                            "status": 200}

            return JSONResponse(content=response, status_code=200)
    except Exception as e:
        response = {"message": f"Error: {str(e)}", "status": 500}
        return JSONResponse(content=response, status_code=500)


async def eye_blink_detection():
    while True:
        processed_frame = pipeline.eye_blink_detection()

        return processed_frame


async def posture_detection():
    while True:
        processed_frame = pipeline.posture_detection()

        return processed_frame


async def convert_frame_to_webp_base64(frame: np.ndarray) -> str:
    if frame is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb_frame)

        buffer = io.BytesIO()

        pil_image.save(buffer, format="WebP", quality=80, optimize=True)

        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f"data:image/webp;base64,{img_base64}"
    return None


async def generate_frames(posture: bool, eye_blink: bool):
    try:
        while True:
            frame_data = {}

            if posture and eye_blink:
                processed_frame_eye_blink = await eye_blink_detection()
                processed_frame_posture = await posture_detection()

                if processed_frame_eye_blink is not None:
                    frame_data["processed_frame_eye_blink"] = await convert_frame_to_webp_base64(
                        processed_frame_eye_blink)
                if processed_frame_posture is not None:
                    frame_data["processed_frame_posture"] = await convert_frame_to_webp_base64(processed_frame_posture)

            elif posture:
                processed_frame_posture = await posture_detection()
                if processed_frame_posture is not None:
                    frame_data["processed_frame_posture"] = await convert_frame_to_webp_base64(processed_frame_posture)

            elif eye_blink:
                processed_frame_eye_blink = await eye_blink_detection()
                if processed_frame_eye_blink is not None:
                    frame_data["processed_frame_eye_blink"] = await convert_frame_to_webp_base64(
                        processed_frame_eye_blink)
            else:
                frame_data = {
                    "message": "Please select either posture or eye_blink or both."
                }
                break

            frame_data["timestamp"] = str(asyncio.get_event_loop().time())

            # Format as SSE data
            data = json.dumps(frame_data)
            yield f"data: {data}\n\n"

            # Add a small delay to control frame rate
            await asyncio.sleep(0.033)  # ~30 FPS

    except asyncio.CancelledError:
        print("Client disconnected")
        raise
    except Exception as e:
        print(f"Error in stream: {str(e)}")
        raise


@sit_blink_router.post("/start_sitblink_stream")
async def eye_blink_stream(posture: bool = False, eye_blink: bool = False):
    pipeline.validate(posture=posture, eye_blink=eye_blink)

    return StreamingResponse(
        generate_frames(posture, eye_blink),
        media_type="text/event-stream"
    )


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
