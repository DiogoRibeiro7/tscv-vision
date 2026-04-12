from __future__ import annotations

import numpy
from setuptools import Extension, find_packages, setup

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

setup(
    name="tscv-vision",
    version="0.1.1",
    description="Computer-vision feature engineering for 1D time series (NumPy-first).",
    author="Diogo Ribeiro",
    author_email="dfr@esmad.ipp.pt",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.24"],
    extras_require={
        "torch": ["torch>=2.2"],
        "mlops": ["fastapi>=0.110", "prometheus-client>=0.20", "feast>=0.42"],
        "analytics": [
            "shap",
            "lime",
            "scikit-learn>=1.3",
            "umap-learn",
            "matplotlib",
            "seaborn",
            "pywavelets",
        ],
        "domains": ["scikit-learn>=1.3"],
        "cli": ["pyyaml"],
        "gpu": ["cupy"],
        "research": [],
    },
    ext_modules=extensions,
    zip_safe=False,
)
