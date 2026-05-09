"""
Helpers for resolving and downloading MediaPipe Tasks model bundles.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import ssl
import urllib.request

import config


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a MediaPipe Tasks model bundle."""

    filename: str
    url: str


POSE_MODEL_SPECS = {
    0: ModelSpec(
        filename="pose_landmarker_lite.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/"
            "pose_landmarker_lite.task"
        ),
    ),
    1: ModelSpec(
        filename="pose_landmarker_full.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/1/"
            "pose_landmarker_full.task"
        ),
    ),
    2: ModelSpec(
        filename="pose_landmarker_heavy.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/1/"
            "pose_landmarker_heavy.task"
        ),
    ),
}

HAND_MODEL_SPEC = ModelSpec(
    filename="hand_landmarker.task",
    url=(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
)


def get_pose_model_spec(complexity=None):
    """Return the pose model spec for the configured or requested complexity."""
    if complexity is None:
        complexity = config.POSE_MODEL_COMPLEXITY
    return POSE_MODEL_SPECS.get(int(complexity), POSE_MODEL_SPECS[0])


def get_hand_model_spec():
    """Return the hand model spec."""
    return HAND_MODEL_SPEC


def resolve_pose_model_path():
    """Return the local path where the pose model should live."""
    override = config.MEDIAPIPE_POSE_MODEL_PATH
    if override:
        return Path(override)
    return Path(config.MEDIAPIPE_MODELS_DIR) / get_pose_model_spec().filename


def resolve_hand_model_path():
    """Return the local path where the hand model should live."""
    override = config.MEDIAPIPE_HAND_MODEL_PATH
    if override:
        return Path(override)
    return Path(config.MEDIAPIPE_MODELS_DIR) / get_hand_model_spec().filename


def ensure_pose_model(auto_download=None):
    """Ensure the pose model exists locally and return its path."""
    return ensure_model(
        resolve_pose_model_path(),
        get_pose_model_spec(),
        auto_download=auto_download,
        required=True,
        model_name="pose",
    )


def ensure_hand_model(auto_download=None, required=False):
    """Ensure the hand model exists locally and return its path or `None`."""
    return ensure_model(
        resolve_hand_model_path(),
        get_hand_model_spec(),
        auto_download=auto_download,
        required=required,
        model_name="hand",
    )


def ensure_model(path, spec, auto_download=None, required=True, model_name="model"):
    """Ensure a model file exists locally, optionally downloading it."""
    path = Path(path)
    if path.exists():
        return path

    if auto_download is None:
        auto_download = config.MEDIAPIPE_AUTO_DOWNLOAD_MODELS

    if auto_download:
        _download_model(spec.url, path)
        if path.exists():
            return path

    if required:
        raise FileNotFoundError(_build_missing_model_message(model_name, path, spec.url))
    return None


def _download_model(url, path):
    """Download a model bundle into the repo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    timeout = int(config.MEDIAPIPE_MODEL_DOWNLOAD_TIMEOUT)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gesture-presenter/1.0"},
    )
    try:
        print(f"[MediaPipe] Downloading model: {path.name}")
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=_download_ssl_context(),
        ) as response:
            with temp_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        temp_path.replace(path)
        print(f"[MediaPipe] Saved model to {path}")
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"[MediaPipe] Model download failed for {path.name}: {exc}")


def _download_ssl_context():
    """Use certifi's CA bundle when available for macOS Python installs."""
    try:
        import certifi
    except Exception:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _build_missing_model_message(model_name, path, url):
    return (
        "MediaPipe Tasks requires a local model bundle.\n"
        f"Missing {model_name} model: {path}\n"
        f"Download it from: {url}\n"
        "Or run `python3 scripts/setup_mediapipe_models.py`."
    )
