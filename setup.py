from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
BUILD_EXT = os.environ.get("TSCV_BUILD_EXT", "").lower() in {"1", "true", "yes", "on"}

extensions: list[Extension] = []
if BUILD_EXT:
    import numpy

    try:  # pragma: no cover - build time only
        from Cython.Build import cythonize

        USE_CYTHON = True
    except Exception:  # pragma: no cover - build time only
        USE_CYTHON = False

    ext = ".pyx" if USE_CYTHON else ".c"
    extensions = [
        Extension(
            "tscv_vision._encoders_cy",
            [f"src/tscv_vision/_encoders_cy{ext}"],
            include_dirs=[numpy.get_include()],
        )
    ]
    if USE_CYTHON:
        extensions = cythonize(extensions, language_level="3")

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="tscv-vision",
    version="0.4.0",
    description="Structured representation engineering for time series (NumPy-first).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Diogo Ribeiro",
    author_email="dfr@esmad.ipp.pt",
    url="https://github.com/DiogoRibeiro7/tscv-vision",
    project_urls={
        "Documentation": "https://github.com/DiogoRibeiro7/tscv-vision#readme",
        "Source": "https://github.com/DiogoRibeiro7/tscv-vision",
        "Issues": "https://github.com/DiogoRibeiro7/tscv-vision/issues",
    },
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"tscv_vision": ["py.typed"]},
    python_requires=">=3.10,<3.13",
    install_requires=["numpy>=1.24"],
    # Keep in sync with [tool.poetry.extras] in pyproject.toml;
    # tests/test_docs_sync.py enforces it.
    extras_require={
        "torch": ["torch>=2.2", "torchvision"],
        "mlops": [
            "fastapi>=0.110",
            "prometheus-client>=0.20",
            "feast>=0.42",
            "uvicorn>=0.29",
        ],
        "analytics": [
            "shap",
            "lime",
            "scikit-learn>=1.3",
            "umap-learn",
            "matplotlib",
            "seaborn",
            "pywavelets",
        ],
        "ml": ["scikit-learn>=1.3"],
        "research": ["scikit-learn>=1.3", "pyts>=0.13"],
        "domains": ["scikit-learn>=1.3"],
        "cli": ["pyyaml"],
        "gpu": ["cupy"],
        "speed": ["numba>=0.59"],
        "io": ["pyarrow>=14", "h5py"],
        "streaming": ["redis>=5", "kafka-python", "pika"],
        "distributed": ["dask>=2024.1"],
        "onnx": ["onnx"],
        "spectral": ["scipy>=1.10"],
        "scattering": ["kymatio>=0.3", "scipy>=1.10,<1.17"],
    },
    entry_points={
        "console_scripts": [
            "tscv-features=tscv_vision.cli:main",
        ],
    },
    ext_modules=extensions,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    zip_safe=False,
)
