"""
project @ SitBlinkSip
created @ 2024-10-21
author  @ github/ishworrsubedii
"""
from setuptools import setup, find_packages

HYPER_E_DOT = "-e ."


def getRequirements(requirementsPath: str) -> list[str]:
    with open(requirementsPath) as file:
        requirements = file.read().split("\n")
    requirements.remove(HYPER_E_DOT)
    return requirements


setup(
    name="SitBlinkSip",
    author="Ishwor Subedi",
    author_email="ishworr.subedi@gmail.com",
    version="0.1",
    packages=find_packages(),
    install_requires=getRequirements(requirementsPath="./requirements.txt")
)
