"""Setup configuration for Project Aura

Installation and packaging configuration.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aura-ai",
    version="0.1.0",
    author="Project Aura Team",
    description="Independent AI Assistant - Built from Scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bisalsoreng329-cmyk/MyAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.11.2",
        "pandas>=2.0.3",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
        "sqlalchemy>=2.0.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "pylint>=2.17.5",
            "mypy>=1.4.1",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.1",
        ],
        "vision": [
            "pillow>=10.0.0",
            "opencv-python>=4.8.0.74",
        ],
    },
)
