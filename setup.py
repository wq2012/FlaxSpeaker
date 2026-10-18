"""Setup script for the FlaxSpeaker package."""

import setuptools

VERSION = "0.2.0"

with open("README.md", "r", encoding="utf-8") as file_object:
    LONG_DESCRIPTION = file_object.read()

setuptools.setup(
    name="flaxspeaker",
    version=VERSION,
    author="Quan Wang",
    author_email="quanw@google.com",
    description=(
        "A modernized, research- and production-ready speaker recognition "
        "library in JAX and Flax."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/FlaxSpeaker",
    project_urls={
        "Repository": "https://github.com/wq2012/FlaxSpeaker",
        "Hugging Face Models Collection": (
            "https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595"
        ),
        "Bug Tracker": "https://github.com/wq2012/FlaxSpeaker/issues",
    },
    packages=setuptools.find_packages(include=["flaxspeaker", "flaxspeaker.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "librosa",
        "soundfile",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "pyyaml",
        "munch",
        "transformers",
        "safetensors",
        "huggingface-hub",
    ],
    extras_require={
        "export": ["tensorflow", "ai-edge-litert"],
        "demo": ["gradio"],
        "dev": ["pytest", "pytest-cov", "flake8", "pytype", "pdoc"],
    },
    entry_points={
        "console_scripts": [
            "flaxspeaker=flaxspeaker.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
