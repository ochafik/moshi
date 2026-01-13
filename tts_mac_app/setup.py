"""
py2app setup script for Moshi TTS Mac App.

Build instructions:
    # Install dependencies
    pip install -r requirements.txt

    # Build standalone app
    python setup.py py2app

    # For development/testing (creates alias mode app)
    python setup.py py2app -A

The app will be created in the `dist/` directory.
"""

import sys
from pathlib import Path

from setuptools import setup

# App metadata
APP_NAME = "Moshi TTS"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "MLX-based Text-to-Speech for Apple Silicon"

# Main script
APP = ["tts_app.py"]

# Data files to include
DATA_FILES = []

# py2app options
OPTIONS = {
    "argv_emulation": False,  # Disable for better compatibility
    "iconfile": None,  # Add path to .icns file if you have one
    "plist": {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "ai.kyutai.moshi-tts",
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,  # Support dark mode
        "LSMinimumSystemVersion": "12.0",  # macOS Monterey minimum
        "NSMicrophoneUsageDescription": "This app may need microphone access for voice cloning features.",
    },
    # Packages to include
    "packages": [
        "mlx",
        "mlx.nn",
        "mlx.core",
        "numpy",
        "huggingface_hub",
        "sentencepiece",
        "pydantic",
        "moshi_mlx",
        "sphn",
        "safetensors",
        "tqdm",
        "filelock",
        "fsspec",
        "requests",
        "urllib3",
        "certifi",
        "charset_normalizer",
        "idna",
        "packaging",
        "yaml",
        "typing_extensions",
    ],
    # Modules to include
    "includes": [
        "tkinter",
        "tkinter.ttk",
        "tkinter.scrolledtext",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ],
    # Modules to exclude (reduce app size)
    "excludes": [
        "matplotlib",
        "PIL",
        "cv2",
        "scipy.spatial.cKDTree",  # Large, unused
        "test",
        "unittest",
        "email",
        "html",
        "http",
        "xml",
        "pydoc",
        "doctest",
        "argparse",  # Not needed at runtime
        "optparse",
        "pickle",
    ],
    # Optimize bytecode
    "optimize": 2,
    # Include __pycache__
    "strip": True,
    # Semi-standalone: assumes Python installed (smaller app)
    # Set to False for fully standalone app
    "semi_standalone": False,
}

# Check for moshi_mlx in parent directory
moshi_mlx_path = Path(__file__).parent.parent / "moshi_mlx"
if moshi_mlx_path.exists():
    # Add to path for build
    sys.path.insert(0, str(moshi_mlx_path))
    # Include moshi_mlx package path
    OPTIONS["packages"].append("moshi_mlx")

setup(
    name=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    author="Kyutai",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
