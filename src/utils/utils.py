"""
project @ SitBlinkSip
created @ 2024-10-18
author  @ github/ishworrsubedii
"""
import configparser
import subprocess


def send_blink_warning_notification(message):
    title = "Low Blink Alert!"
    message = message

    subprocess.run(["notify-send", title, message])
    sound_file = "/usr/share/sounds/Yaru/stereo/complete.oga"  # Default message sound
    subprocess.run(["paplay", sound_file])


def config_reader():
    config = configparser.ConfigParser()
    config.read('config.ini')

    return config
