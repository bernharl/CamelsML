from pathlib import Path
from setuptools import setup, find_packages


PACKAGE_NAME = "camelsml"

folder = Path(__file__).parent

version = {}
with open(folder / PACKAGE_NAME / "version.py") as fp:
    exec(fp.read(), version)

packs = find_packages(where=folder, exclude=("tests*",))

with open(folder / "requirements.txt") as fp:
    requirements = fp.read().split("\n")[8:]

with open("README.md") as fp:
    readme = fp.read()

with open("LICENSE") as fp:
    license = fp.read()
setup(
    name=PACKAGE_NAME,
    version=version["__version__"],
    description="Machine learning for CAMELS_GB (support for more in the future maybe)",
    long_description=readme,
    url="https://github.com/epigramai/videostream",
    author="Frederik Kratzert (original code), Bernhard Nornes Lotsberg (forked code)",
    author_email="bernhardnorneslotsberg@gmail.com",
    license=license,
    packages=packs,
    #package_dir={"": "camelsml"},
    install_requires=requirements,
)
