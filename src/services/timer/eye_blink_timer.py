"""
project @ SitBlinkSip
created @ 2024-10-17
author  @ github/ishworrsubedii
"""
from datetime import datetime, timedelta


class BlinkMonitor:
    def __init__(self):
        self.last_blink_time = datetime.now()
        self.blinks_in_last_minute = []
        self.warning_status = {
            '20_sec': False,
            '30_sec': False,
            'frequency': False
        }

    def check_blink_health(self, current_time, total_blinks):
        warnings = []

        time_since_last_blink = (current_time - self.last_blink_time).total_seconds()

        # 20-second warning
        if time_since_last_blink >= 20 and not self.warning_status['20_sec']:
            warnings.append("Warning: No blinks detected for 20 seconds!")
            self.warning_status['20_sec'] = True

        # 30-second warning
        if time_since_last_blink >= 30 and not self.warning_status['30_sec']:
            warnings.append("Alert: No blinks detected for 30 seconds!")
            self.warning_status['30_sec'] = True

        # Reset warnings if person blinks
        if time_since_last_blink < 20:
            self.warning_status['20_sec'] = False
            self.warning_status['30_sec'] = False

        # Track blinks in the last minute
        self.blinks_in_last_minute.append({
            'time': current_time,
            'total_blinks': total_blinks
        })

        # Remove blinks older than 1 minute
        one_minute_ago = current_time - timedelta(minutes=1)
        self.blinks_in_last_minute = [
            blink for blink in self.blinks_in_last_minute
            if blink['time'] > one_minute_ago
        ]

        # Calculate blink frequency
        if len(self.blinks_in_last_minute) >= 2:
            blinks_per_minute = (
                    self.blinks_in_last_minute[-1]['total_blinks'] -
                    self.blinks_in_last_minute[0]['total_blinks']
            )

            if blinks_per_minute < 20 and not self.warning_status['frequency']:
                warnings.append(f"Warning: Low blink rate! Only {blinks_per_minute} blinks in last minute")
                self.warning_status['frequency'] = True
            elif blinks_per_minute >= 20:
                self.warning_status['frequency'] = False

        return warnings

    def update_last_blink(self, blink_time):
        self.last_blink_time = blink_time
