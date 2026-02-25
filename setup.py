#!/usr/bin/env python
"""Setup script for KeyPoint-MoSeq Kumar Lab package."""

from setuptools import setup, find_packages
import pathlib

# Read the contents of README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open(here / "requirements.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="kpms-kumarlab",
    version="0.1.0",
    description="Unsupervised behavioral analysis using KeyPoint-MoSeq AR-HMM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anshu957/kpms_kumarlab",
    author="Kumar Lab",
    author_email="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="behavioral analysis, keypoint tracking, pose estimation, machine learning",
    packages=find_packages(exclude=["tests", "notebooks", "examples"]),
    python_requires=">=3.9, <4",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "kpms-train=scripts.train_kpms:main",
            "kpms-preprocess=scripts.preprocess_poses:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/anshu957/kpms_kumarlab/issues",
        "Source": "https://github.com/anshu957/kpms_kumarlab",
    },
)
