import { v4 as uuidv4 } from 'uuid';
import { personService } from './personService';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';

export interface CameraSession {
  user_id: string;
  person_id?: number | null;
  settings: {
    posture: boolean;
    eye_blink: boolean;
  };
  status: string;
  created_at: string;
}

export const cameraService = {
  async startSession(settings: { posture: boolean; eye_blink: boolean }): Promise<CameraSession> {
    const user_id = uuidv4();
    const person_id = personService.getStoredPerson()?.id ?? null;

    // Store session ID in localStorage for persistence
    localStorage.setItem('camera_session_id', user_id);

    const response = await fetch(`${API_URL}/start-camera-session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id,
        person_id,
        settings
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to start camera session: ${response.statusText}`);
    }

    const data = await response.json();
    return data.session;
  },

  async resumeSession(): Promise<CameraSession | null> {
    const session_id = localStorage.getItem('camera_session_id');
    if (!session_id) return null;
    
    try {
      const response = await fetch(`${API_URL}/session/${session_id}`);
      if (!response.ok) {
        localStorage.removeItem('camera_session_id');
        return null;
      }
      const data = await response.json();
      return data.session;
    } catch (error) {
      localStorage.removeItem('camera_session_id');
      return null;
    }
  },

  createWebSocket(user_id: string): WebSocket {
    const wsUrl = new URL(`${WS_URL}/ws/${user_id}`);
    const ws = new WebSocket(wsUrl.toString());
    return ws;
  },

  async startBackgroundMonitoring(settings: { 
    posture: boolean; 
    eye_blink: boolean;
  }): Promise<CameraSession> {
    const session = await this.startSession(settings);
    
    // Store monitoring settings
    localStorage.setItem('monitoring_active', 'true');
    localStorage.setItem('monitor_settings', JSON.stringify({
      posture: settings.posture,
      eye_blink: settings.eye_blink
    }));
    
    return session;
  },

  async stopBackgroundMonitoring() {
    localStorage.removeItem('monitoring_active');
    localStorage.removeItem('monitor_settings');
    localStorage.removeItem('camera_session_id');
  }
}; 