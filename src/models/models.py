"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""

import sqlite3
import threading
import os
from typing import Optional


class SitBlinkSipDB:

    def __init__(self, db_name="sitblinksip.db"):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.sip_db = "data/sip.db"

        self.lock = threading.Lock()  # For concurrency when accessing database

    def initialize(self):
        try:
            os.makedirs('data', exist_ok=True)

            self.connection = sqlite3.connect(f'data/{self.db_name}', check_same_thread=False)
            self.cursor = self.connection.cursor()

            self.connection.execute('PRAGMA journal_mode=WAL')

            self.create_tables()
            self._migrate_person_columns()
            print("Database initialized successfully")

        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise

    def create_tables(self):
        """Create necessary tables for health monitoring"""
        with self.lock:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    water_break_interval INTEGER NOT NULL DEFAULT 30,
                    water_break_started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ear_threshold REAL NOT NULL DEFAULT 0.15,
                    posture_angle_threshold REAL NOT NULL DEFAULT 145.0,
                    posture_displacement_threshold REAL NOT NULL DEFAULT 0.65,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS waitlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    profession TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Posture Detection Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS posture_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    head_tilt REAL,
                    displacement_ratio REAL,
                    posture_status BOOLEAN,
                    person_id INTEGER
                )
            """)

            # Eye Metrics Table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS eyeblink_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    EAR REAL,
                    blink BOOLEAN,
                    person_id INTEGER
                )
            ''')

            self.connection.commit()

    def _migrate_person_columns(self):
        """Defensively add person_id to tables created before this column existed."""
        with self.lock:
            for table in ("posture_db", "eyeblink_db"):
                self.cursor.execute(f"PRAGMA table_info({table})")
                columns = {row[1] for row in self.cursor.fetchall()}
                if "person_id" not in columns:
                    try:
                        self.cursor.execute(f"ALTER TABLE {table} ADD COLUMN person_id INTEGER")
                    except sqlite3.OperationalError:
                        pass

            self.cursor.execute("PRAGMA table_info(persons)")
            person_columns = {row[1] for row in self.cursor.fetchall()}
            if "water_break_interval" not in person_columns:
                try:
                    self.cursor.execute(
                        "ALTER TABLE persons ADD COLUMN water_break_interval INTEGER NOT NULL DEFAULT 30"
                    )
                except sqlite3.OperationalError:
                    pass
            if "water_break_started_at" not in person_columns:
                try:
                    self.cursor.execute(
                        "ALTER TABLE persons ADD COLUMN water_break_started_at DATETIME"
                    )
                    self.cursor.execute(
                        "UPDATE persons SET water_break_started_at = CURRENT_TIMESTAMP WHERE water_break_started_at IS NULL"
                    )
                except sqlite3.OperationalError:
                    pass
            if "ear_threshold" not in person_columns:
                try:
                    self.cursor.execute(
                        "ALTER TABLE persons ADD COLUMN ear_threshold REAL NOT NULL DEFAULT 0.15"
                    )
                except sqlite3.OperationalError:
                    pass
            if "posture_angle_threshold" not in person_columns:
                try:
                    self.cursor.execute(
                        "ALTER TABLE persons ADD COLUMN posture_angle_threshold REAL NOT NULL DEFAULT 145.0"
                    )
                except sqlite3.OperationalError:
                    pass
            if "posture_displacement_threshold" not in person_columns:
                try:
                    self.cursor.execute(
                        "ALTER TABLE persons ADD COLUMN posture_displacement_threshold REAL NOT NULL DEFAULT 0.65"
                    )
                except sqlite3.OperationalError:
                    pass

            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_posture_person ON posture_db(person_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_eyeblink_person ON eyeblink_db(person_id)")
            self.connection.commit()

    @staticmethod
    def _parse_posture_status(value) -> Optional[bool]:
        """Normalize posture_status, which historically stored the string
        "Good Posture"/"Bad Posture" instead of a real boolean, into a bool."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower() == "good posture"
        return bool(value)

    def create_waitlist_entry(self, email: str, full_name: str, profession: str) -> dict:
        """Add a waitlist signup. Raises sqlite3.IntegrityError if the email already exists."""
        with self.lock:
            self.cursor.execute(
                "INSERT INTO waitlist (email, full_name, profession) VALUES (?, ?, ?)",
                (email, full_name, profession)
            )
            self.connection.commit()
            entry_id = self.cursor.lastrowid
            self.cursor.execute(
                "SELECT id, email, full_name, profession, created_at FROM waitlist WHERE id = ?",
                (entry_id,)
            )
            row = self.cursor.fetchone()
            return {"id": row[0], "email": row[1], "full_name": row[2], "profession": row[3], "created_at": row[4]}

    def list_waitlist(self) -> list:
        """List all waitlist signups"""
        with self.lock:
            self.cursor.execute(
                "SELECT id, email, full_name, profession, created_at FROM waitlist ORDER BY created_at DESC"
            )
            rows = self.cursor.fetchall()
            return [
                {"id": r[0], "email": r[1], "full_name": r[2], "profession": r[3], "created_at": r[4]}
                for r in rows
            ]

    _PERSON_COLUMNS = (
        "id, name, water_break_interval, water_break_started_at, "
        "ear_threshold, posture_angle_threshold, posture_displacement_threshold, created_at"
    )

    @classmethod
    def _person_row_to_dict(cls, row) -> dict:
        return {
            "id": row[0],
            "name": row[1],
            "water_break_interval": row[2],
            "water_break_started_at": row[3],
            "ear_threshold": row[4],
            "posture_angle_threshold": row[5],
            "posture_displacement_threshold": row[6],
            "created_at": row[7],
        }

    def create_person(self, name: str) -> dict:
        """Create a new person profile and return it"""
        with self.lock:
            try:
                self.cursor.execute(
                    "INSERT INTO persons (name, water_break_started_at) VALUES (?, CURRENT_TIMESTAMP)",
                    (name,)
                )
                self.connection.commit()
                person_id = self.cursor.lastrowid
                self.cursor.execute(
                    f"SELECT {self._PERSON_COLUMNS} FROM persons WHERE id = ?",
                    (person_id,)
                )
                return self._person_row_to_dict(self.cursor.fetchone())
            except sqlite3.Error as e:
                print(f"Error creating person: {e}")
                raise

    def list_persons(self) -> list:
        """List all known persons"""
        with self.lock:
            self.cursor.execute(
                f"SELECT {self._PERSON_COLUMNS} FROM persons ORDER BY created_at DESC"
            )
            return [self._person_row_to_dict(row) for row in self.cursor.fetchall()]

    def get_person(self, person_id: int) -> Optional[dict]:
        """Fetch a single person by id"""
        with self.lock:
            self.cursor.execute(
                f"SELECT {self._PERSON_COLUMNS} FROM persons WHERE id = ?",
                (person_id,)
            )
            row = self.cursor.fetchone()
            return self._person_row_to_dict(row) if row else None

    def update_water_break_interval(self, person_id: int, interval: int) -> Optional[dict]:
        """Update a person's water-break reminder interval (minutes) and restart its schedule anchor.
        Returns None if person doesn't exist."""
        with self.lock:
            self.cursor.execute(
                "UPDATE persons SET water_break_interval = ?, water_break_started_at = CURRENT_TIMESTAMP WHERE id = ?",
                (interval, person_id)
            )
            self.connection.commit()
            if self.cursor.rowcount == 0:
                return None
            self.cursor.execute(
                f"SELECT {self._PERSON_COLUMNS} FROM persons WHERE id = ?",
                (person_id,)
            )
            return self._person_row_to_dict(self.cursor.fetchone())

    def update_detection_thresholds(
        self,
        person_id: int,
        ear_threshold: float,
        posture_angle_threshold: float,
        posture_displacement_threshold: float,
    ) -> Optional[dict]:
        """Update a person's detection sensitivity thresholds. Returns None if person doesn't exist."""
        with self.lock:
            self.cursor.execute(
                """UPDATE persons
                   SET ear_threshold = ?, posture_angle_threshold = ?, posture_displacement_threshold = ?
                   WHERE id = ?""",
                (ear_threshold, posture_angle_threshold, posture_displacement_threshold, person_id)
            )
            self.connection.commit()
            if self.cursor.rowcount == 0:
                return None
            self.cursor.execute(
                f"SELECT {self._PERSON_COLUMNS} FROM persons WHERE id = ?",
                (person_id,)
            )
            return self._person_row_to_dict(self.cursor.fetchone())

    def insert_posture_data(self, head_tilt, displacement_ratio, posture_status, person_id: Optional[int] = None):
        """Insert new posture detection data"""
        with self.lock:
            try:
                self.cursor.execute('''
                    INSERT INTO posture_db
                    (head_tilt, displacement_ratio, posture_status, person_id)
                    VALUES (?, ?, ?, ?)
                ''', (head_tilt, displacement_ratio, posture_status, person_id))
                self.connection.commit()
            except sqlite3.Error as e:
                print(f"Error inserting posture data: {e}")
                raise

    def insert_eye_data(self, ear, blink, person_id: Optional[int] = None):
        """Insert new eye tracking data"""
        with self.lock:
            try:
                self.cursor.execute('''
                    INSERT INTO eyeblink_db
                    (EAR, blink, person_id)
                    VALUES (?, ?, ?)
                ''', (ear, blink, person_id))
                self.connection.commit()
            except sqlite3.Error as e:
                print(f"Error inserting eye data: {e}")
                raise

    def get_recent_posture_data(self, minutes=30, person_id: Optional[int] = None):
        """Get posture data from last N minutes, optionally scoped to a person"""
        with self.lock:
            query = '''
                SELECT id, timestamp, head_tilt, displacement_ratio, posture_status, person_id
                FROM posture_db
                WHERE timestamp >= datetime('now', ?)
            '''
            params = [f'-{minutes} minutes']
            if person_id is not None:
                query += ' AND person_id = ?'
                params.append(person_id)
            query += ' ORDER BY timestamp DESC'

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "head_tilt": r[2],
                    "displacement_ratio": r[3],
                    "posture_status": self._parse_posture_status(r[4]),
                    "person_id": r[5],
                }
                for r in rows
            ]

    def get_recent_eye_data(self, minutes=30, person_id: Optional[int] = None):
        """Get eye metrics from last N minutes, optionally scoped to a person"""
        with self.lock:
            query = '''
                SELECT id, timestamp, EAR, blink, person_id
                FROM eyeblink_db
                WHERE timestamp >= datetime('now', ?)
            '''
            params = [f'-{minutes} minutes']
            if person_id is not None:
                query += ' AND person_id = ?'
                params.append(person_id)
            query += ' ORDER BY timestamp DESC'

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "ear": r[2],
                    "blink": bool(r[3]) if r[3] is not None else None,
                    "person_id": r[4],
                }
                for r in rows
            ]

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

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


def initialize_database():
    db = SitBlinkSipDB()
    db.initialize()
    db.initialize_database()
    return db
