import { apiClient } from '@/lib/api-client';

const STORAGE_KEY = 'sbs_person';

export interface Person {
  id: number;
  name: string;
  water_break_interval: number;
  water_break_started_at?: string;
  ear_threshold: number;
  posture_angle_threshold: number;
  posture_displacement_threshold: number;
  created_at?: string;
}

export interface DetectionThresholds {
  ear_threshold: number;
  posture_angle_threshold: number;
  posture_displacement_threshold: number;
}

export const personService = {
  getStoredPerson(): Person | null {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  },

  async createPerson(name: string): Promise<Person> {
    const data = await apiClient.fetch('persons', {
      method: 'POST',
      body: { name },
    });
    const person: Person = data.person;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(person));
    return person;
  },

  async listPersons(): Promise<Person[]> {
    const data = await apiClient.fetch('persons');
    return data.persons ?? [];
  },

  async updateWaterBreakInterval(personId: number, interval: number): Promise<Person> {
    const data = await apiClient.fetch(`persons/${personId}/water-break-interval`, {
      method: 'PUT',
      body: { interval },
    });
    const person: Person = data.person;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(person));
    return person;
  },

  async updateDetectionThresholds(personId: number, thresholds: DetectionThresholds): Promise<Person> {
    const data = await apiClient.fetch(`persons/${personId}/detection-thresholds`, {
      method: 'PUT',
      body: thresholds,
    });
    const person: Person = data.person;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(person));
    return person;
  },
};
