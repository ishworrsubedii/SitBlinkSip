"""
project @ SitBlinkSip
created @ 2024-10-27
author  @ github/ishworrsubedii
"""
from src.models.models import initialize_database

if __name__ == "__main__":
    db = initialize_database()

    try:
        # db.insert_posture_data(
        #     head_tilt=15.5,
        #     displacement_ratio=0.8,
        #     posture_status=True
        # )
        #
        # db.insert_eye_data(
        #     ear=0.25,
        #     blink=True
        # )

        recent_posture_data = db.get_recent_posture_data(minutes=10000)
        recent_eye_data = db.get_recent_eye_data(minutes=10000)
        with open("posture_data.txt", "w") as f:
            f.write(str(recent_posture_data))
        with open("eye_data.txt", "w") as f:
            f.write(str(recent_eye_data))

    finally:
        db.close()
