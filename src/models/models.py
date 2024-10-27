"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""

import sqlite3
import threading
import os


class SitBlinkSipDB:

    def __init__(self, db_name="sitblinksip.db"):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.lock = threading.Lock()  # For concurrency when accessing database

    def initialize(self):
        try:
            os.makedirs('data', exist_ok=True)

            self.connection = sqlite3.connect(f'data/{self.db_name}', check_same_thread=False)
            self.cursor = self.connection.cursor()

            self.connection.execute('PRAGMA journal_mode=WAL')

            self.create_tables()
            print("Database initialized successfully")

        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise

    def create_tables(self):
        """Create necessary tables for health monitoring"""
        with self.lock:
            # Posture Detection Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS posture_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    head_tilt REAL,
                    displacement_ratio REAL,
                    posture_status BOOLEAN
                )
            """)

            # Eye Metrics Table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS eyeblink_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    EAR REAL,
                    blink BOOLEAN
                )
            ''')

            self.connection.commit()

    def insert_posture_data(self, head_tilt, displacement_ratio, posture_status):
        """Insert new posture detection data"""
        with self.lock:
            try:
                self.cursor.execute('''
                    INSERT INTO posture_db 
                    (head_tilt, displacement_ratio, posture_status)
                    VALUES (?, ?, ?)
                ''', (head_tilt, displacement_ratio, posture_status))
                self.connection.commit()
            except sqlite3.Error as e:
                print(f"Error inserting posture data: {e}")
                raise

    def insert_eye_data(self, ear, blink):
        """Insert new eye tracking data"""
        with self.lock:
            try:
                self.cursor.execute('''
                    INSERT INTO eyeblink_db 
                    (EAR, blink)
                    VALUES (?, ?)
                ''', (ear, blink))
                self.connection.commit()
            except sqlite3.Error as e:
                print(f"Error inserting eye data: {e}")
                raise

    def get_recent_posture_data(self, minutes=30):
        """Get posture data from last 30 minutes"""
        with self.lock:
            self.cursor.execute(f'''
                SELECT * FROM posture_db
                WHERE timestamp >= datetime('now', '-{minutes} minutes')
                ORDER BY timestamp DESC
            ''')
            return self.cursor.fetchall()

    def get_recent_eye_data(self, minutes=30):
        """Get eye metrics from last 30 minutes"""
        with self.lock:
            self.cursor.execute(f'''
                SELECT * FROM eyeblink_db
                WHERE timestamp >= datetime('now', '-{minutes} minutes')
                ORDER BY timestamp DESC
            ''')
            return self.cursor.fetchall()

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class SipDB:
    def __init__(self):
        self.sip_db = "data/sip.db"

    def initialize_database(self):
        conn = sqlite3.connect(self.sip_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intervals (
                id INTEGER PRIMARY KEY,
                interval INTEGER NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY,
                notification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()


def initialize_database():
    db = SitBlinkSipDB()
    sdb = SipDB()
    db.initialize()
    sdb.initialize_database()
    return db
