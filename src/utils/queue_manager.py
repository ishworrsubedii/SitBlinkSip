from datetime import datetime, timedelta
from typing import Dict, Optional


class CameraQueueManager:
    """In-memory, single-process camera session store with TTL expiry.

    Replaces the previous Redis-backed implementation: this app is a local,
    single-user tool, so an external session store is unnecessary overhead.
    """

    SESSION_TTL = timedelta(minutes=30)

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._expires_at: Dict[str, datetime] = {}

    def _is_expired(self, user_id: str) -> bool:
        expires_at = self._expires_at.get(user_id)
        return expires_at is None or datetime.now() >= expires_at

    def _touch(self, user_id: str):
        self._expires_at[user_id] = datetime.now() + self.SESSION_TTL

    async def create_session(self, user_id: str, settings: dict, person_id: Optional[int] = None):
        session_data = {
            'user_id': user_id,
            'person_id': person_id,
            'settings': settings,
            'status': 'created',
            'created_at': str(datetime.now())
        }
        self._sessions[user_id] = session_data
        self._touch(user_id)
        return session_data

    async def get_session(self, user_id: str) -> Optional[dict]:
        """Get session details"""
        if user_id not in self._sessions or self._is_expired(user_id):
            self._sessions.pop(user_id, None)
            self._expires_at.pop(user_id, None)
            return None
        return self._sessions[user_id]

    async def update_session(self, user_id: str, status: str, websocket=None):
        """Update session status"""
        session_data = await self.get_session(user_id)
        if session_data:
            session_data['status'] = status
            session_data['last_active'] = str(datetime.now())
            self._sessions[user_id] = session_data
            self._touch(user_id)
