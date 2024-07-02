from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

setup(
    name="servir",
    version="0.1.0",
    packages=find_packages(),
    description="Precipitation Forecasting in West Africa using Deep Learning Models",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "attention mechanism",
        "metnet",
        "forecasting",
        "remote-sensing",
        "gan",
    ],
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
