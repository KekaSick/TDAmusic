"""
Setup script for the project
"""
from setuptools import setup, find_packages

setup(
    name="3rdCourseWork",
    version="0.1.0",
    description="Music topology analysis project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "matplotlib",
        "scikit-learn",
        "umap-learn",
        "hdbscan",
        "ripser",
        "persim",
        "plotly",
        "pandas",
        "tqdm",
        "antropy",
        "ordpy",
    ],
)

