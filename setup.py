"""Setup configuration for QualityAssessor package."""

from setuptools import setup, find_packages

setup(
    name="quality-assessor",
    version="1.0.0",
    description="Unified quality assessment module for evidence-based medicine (GRADE, Cochrane RoB 2.0, ROBINS-I)",
    author="blue-penguin-123",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
