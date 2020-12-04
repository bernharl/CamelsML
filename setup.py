from setuptools import setup
import pip

requirements = pip.req.parse_requirements("requirements.txt")

setup(
    install_requires=requirements,
)
