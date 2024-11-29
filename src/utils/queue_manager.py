from redis import Redis, ConnectionError
from rq import Queue
from datetime import timedelta, datetime
import json
import logging

class CameraQueueManager:
    def __init__(self):
        try:
            self.redis = Redis(
                host='localhost',
                port=6379,
                db=0,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            self.queue = Queue('camera_jobs', connection=self.redis)
        except ConnectionError:
            logging.error("Failed to connect to Redis. Ensure Redis server is running.")
            raise

    async def create_session(self, user_id: str, settings: dict):
        try:
            job_data = {
                'user_id': user_id,
                'settings': settings,
                'status': 'created',
                'created_at': str(datetime.now())
            }
            
            self.redis.setex(
                f"camera_session:{user_id}",
                1800,  # 30 minutes in seconds
                json.dumps(job_data)
            )
            return job_data
        except ConnectionError:
            logging.error("Redis connection failed during session creation")
            # Fallback to in-memory storage or raise error
            raise
        
    async def get_session(self, user_id: str):
        """Get session details"""
        session_data = self.redis.get(f"camera_session:{user_id}")
        return json.loads(session_data) if session_data else None
        
    async def update_session(self, user_id: str, status: str, websocket=None):
        """Update session status"""
        session_data = await self.get_session(user_id)
        if session_data:
            session_data['status'] = status
            session_data['last_active'] = str(datetime.now())
            self.redis.setex(
                f"camera_session:{user_id}",
                timedelta(minutes=30),
                json.dumps(session_data)
            ) 