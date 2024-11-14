"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""

from fastapi import FastAPI

from src.apis.sip_api import sip_router
from src.apis.sitblink_api import sit_blink_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router( sit_blink_router)
app.include_router(sip_router)


@app.get("/status")
async def status():
    return {"status": "running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
