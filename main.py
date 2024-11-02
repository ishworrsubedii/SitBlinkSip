"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""

from fastapi import FastAPI

from src.apis.sip_api import sip_router
from src.apis.sitblink_api import sit_blink_router

app = FastAPI()
app.include_router(sit_blink_router)
app.include_router(sip_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
