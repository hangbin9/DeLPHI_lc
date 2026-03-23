from setuptools import setup, find_packages
import re

# Read version from lc_pipeline/version.py without importing the module
# (avoids triggering imports of numpy/scipy during build)
with open("lc_pipeline/version.py") as f:
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
    __version__ = version_match.group(1) if version_match else "1.0.0"

setup(
    name="lc_pipeline",
    version=__version__,
    description="End-to-end pipeline for asteroid pole determination from lightcurve observations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Asteroid Lightcurve Pipeline Team",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests_*", "scripts", "data", "artifacts", "results", "pole_synth", "axisnet"]),
    package_data={
        "lc_pipeline": ["checkpoints/*.pt"],
    },
    python_requires=">=3.8",
    url="https://github.com/hangbin9/lc_dl",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=1.12",
        "pandas",
        "tqdm",
        "pydantic>=1.8",
        "astropy>=5.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "plot": ["matplotlib>=3.4"],
    },
)
