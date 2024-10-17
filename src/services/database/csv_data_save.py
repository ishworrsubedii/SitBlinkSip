"""
project @ SitBlinkSip
created @ 2024-10-17
author  @ github/ishworrsubedii
"""

import csv
import os


class CSVDatabase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fieldnames = ['id', 'time', 'eyeBlink', 'ear_value']

        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def insert_record(self, record):
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(record)

    def read_records(self):
        with open(self.file_path, mode='r') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]
