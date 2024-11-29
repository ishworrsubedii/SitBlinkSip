from datetime import datetime
from typing import Dict, Optional
from redis import Redis


class SessionManager:
    def __init__(self):
        self.redis = Redis(host='localhost', port=6379, db=0)
        self.active_sessions: Dict[str, Dict] = {}

    async def create_session(self, user_id: str, settings: Dict[str, bool]) -> Dict:
        session_data = {
            "user_id": user_id,
            "settings": settings,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        }

        # Store in Redis with 30-minute expiration
        self.redis.setex(
            f"session:{user_id}",
            1800,  # 30 minutes
            str(session_data)
        )

        self.active_sessions[user_id] = session_data
        return session_data

    async def get_session(self, user_id: str) -> Optional[Dict]:
        if user_id in self.active_sessions:
            return self.active_sessions[user_id]

        session_data = self.redis.get(f"session:{user_id}")
        if session_data:
            session = eval(session_data.decode('utf-8'))
            self.active_sessions[user_id] = session
            return session
        return None

    async def update_session(self, user_id: str, status: str):
        if session := await self.get_session(user_id):
            session["status"] = status
            session["last_active"] = datetime.now().isoformat()
            self.redis.setex(
                f"session:{user_id}",
                1800,
                str(session)
            )
            self.active_sessions[user_id] = session
