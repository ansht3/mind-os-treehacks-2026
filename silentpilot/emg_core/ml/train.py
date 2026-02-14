"""Training pipeline for sEMG command classifier.

Per-user, per-session model using StandardScaler -> optional LDA -> classifier.
Supports RandomForest (default, research showed RF >> LR) and LogisticRegression.
TD10 feature space: 840 dims for 4 channels -> LDA -> 32 dims -> classifier.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Optional

from emg_core.dsp.features import extract_features
from emg_core.dsp.filters import preprocess_multichannel
from emg_core.ml.model_io import save_model
from emg_core.api.schemas import TrainResponse
from emg_core import config


def load_dataset(user_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the calibration dataset for a user.

    Returns:
        X: feature matrix (num_samples, num_features)
        y: label indices (num_samples,)
        labels: list of label names
    """
    data_path = os.path.join(config.DATA_DIR, f"{user_id}_calib.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No calibration data for user '{user_id}'")

    data = np.load(data_path, allow_pickle=True)
    segments = data["segments"]  # list of (fixed_length, num_channels)
    labels_raw = data["labels"]  # list of string labels

    # Get unique labels
    unique_labels = sorted(set(labels_raw))

    # Extract features
    X_list = []
    y_list = []
    for seg, label in zip(segments, labels_raw):
        seg = np.array(seg, dtype=np.float64)
        # Preprocess segment with bandpass (1.3-50 Hz)
        seg = preprocess_multichannel(seg, fs=config.SAMPLE_RATE, apply_bandpass=True)
        features = extract_features(seg, sample_rate=config.SAMPLE_RATE)
        X_list.append(features)
        y_list.append(unique_labels.index(label))

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, unique_labels


def _build_pipeline(n_classes: int) -> Pipeline:
    """Build the classification pipeline based on config.

    Pipeline: StandardScaler -> LDA (optional) -> Classifier (RF or LR).
    """
    steps = [('scaler', StandardScaler())]

    # LDA dimensionality reduction
    # n_components must be <= min(n_features, n_classes - 1)
    if n_classes > 2:
        lda_components = min(config.LDA_COMPONENTS, n_classes - 1)
        steps.append(('lda', LinearDiscriminantAnalysis(n_components=lda_components)))

    # Classifier
    if config.CLASSIFIER_TYPE == "rf":
        steps.append(('clf', RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )))
    else:
        steps.append(('clf', LogisticRegression(
            max_iter=5000,
            C=1.0,
            solver='lbfgs',
        )))

    return Pipeline(steps)


def train_model(user_id: str, test_size: float = 0.2) -> TrainResponse:
    """Train a classifier for a user and save it.

    Args:
        user_id: User identifier.
        test_size: Fraction of data for validation.

    Returns:
        TrainResponse with accuracy metrics.
    """
    X, y, labels = load_dataset(user_id)
    n_classes = len(labels)

    # Train/test split (stratified if possible)
    if len(X) < 10 or n_classes < 2:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

    pipeline = _build_pipeline(n_classes)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation score if enough data
    cv_acc = None
    if len(X) >= 20 and n_classes >= 2:
        try:
            cv_scores = cross_val_score(pipeline, X, y, cv=min(5, len(X) // n_classes),
                                         scoring='accuracy')
            cv_acc = float(np.mean(cv_scores))
        except Exception:
            cv_acc = None

    # Per-class accuracy
    per_class: dict[str, float] = {}
    for i, label in enumerate(labels):
        mask = y_test == i
        if mask.sum() > 0:
            per_class[label] = float(accuracy_score(y_test[mask], y_pred[mask]))
        else:
            per_class[label] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))

    # Save model pipeline (includes scaler + LDA + classifier) along with labels
    model_data = {
        "model": pipeline,
        "labels": labels,
    }
    save_model(model_data, user_id)

    return TrainResponse(
        accuracy=float(acc),
        per_class_accuracy=per_class,
        confusion_matrix=cm.tolist(),
        labels=labels,
        num_samples=len(X),
    )
