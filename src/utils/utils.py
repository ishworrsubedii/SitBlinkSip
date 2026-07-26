"""
project @ SitBlinkSip
created @ 2024-10-18
author  @ github/ishworrsubedii
"""
import configparser
import shutil


def config_reader():
    config = configparser.ConfigParser()
    config.read('config.ini')

    return config


def del_directory(directory_path):
    shutil.rmtree(directory_path)


def move_file(file_path, destination_path):
    shutil.move(file_path, destination_path)
