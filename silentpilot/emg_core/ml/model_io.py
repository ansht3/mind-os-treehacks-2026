"""Model I/O utilities for saving and loading trained classifiers."""

import os
import joblib
from typing import Any

from emg_core import config


def get_model_path(user_id: str) -> str:
    """Get the path to a user's model file."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    return os.path.join(config.MODELS_DIR, f"{user_id}_model.joblib")


def save_model(model: Any, user_id: str) -> str:
    """Save a trained model to disk.

    Returns the path to the saved model.
    """
    path = get_model_path(user_id)
    joblib.dump(model, path)
    return path


def load_model(user_id: str) -> Any:
    """Load a trained model from disk.

    Raises FileNotFoundError if the model doesn't exist.
    """
    path = get_model_path(user_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found for user '{user_id}' at {path}")
    return joblib.load(path)


def model_exists(user_id: str) -> bool:
    """Check if a model exists for a user."""
    return os.path.exists(get_model_path(user_id))
