from setuptools import setup
import pathlib

requirements = Path("requirements.txt").read_text().splitlines()

setup(
    install_requires=requirements,
)
