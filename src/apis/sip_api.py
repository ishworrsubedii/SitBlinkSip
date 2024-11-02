"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from datetime import datetime
import asyncio
import sqlite3

sip_db = "sip.db"

sip_router = APIRouter(prefix="/sip", tags=["SIP"])


async def send_water_notification(interval):
    while True:
        await asyncio.sleep(interval * 60)
        conn = sqlite3.connect(sip_db)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO notifications (notification_time) VALUES (CURRENT_TIMESTAMP)")
        conn.commit()
        conn.close()


@sip_router.post("/set_water_break_interval")
async def set_water_break_interval(interval: int, background_tasks: BackgroundTasks):
    conn = sqlite3.connect(sip_db)
    if interval <= 0:
        raise HTTPException(status_code=400, detail="Interval must be a positive integer.")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM intervals")
    cursor.execute("INSERT INTO intervals (interval) VALUES (?)", (interval,))
    conn.commit()
    conn.close()
    background_tasks.add_task(send_water_notification, interval)
    return {"message": f"Water break notifications set to every {interval} minutes."}


@sip_router.post("/stop_water_break_notifications")
async def stop_water_break_notifications():
    conn = sqlite3.connect(sip_db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM intervals")
    conn.commit()
    conn.close()
    return {"message": "Water break notifications stopped successfully."}


@sip_router.get("/notification_history")
async def notification_history():
    conn = sqlite3.connect(sip_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notifications ORDER BY notification_time DESC")
    notifications = cursor.fetchall()
    conn.close()
    return {"notifications": notifications}
