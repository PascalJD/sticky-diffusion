"""InceptionV3 feature extraction for FID.

This implementation uses `tf.keras.applications.InceptionV3` with ImageNet weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


def to_uint8_0_255(images: np.ndarray) -> np.ndarray:
    if images.dtype == np.uint8:
        return images

    imgs = images.astype(np.float32)
    mn = float(np.min(imgs))
    mx = float(np.max(imgs))

    if mn >= 0.0 and mx <= 1.0 + 1e-6:
        imgs = imgs * 255.0
    elif mn >= -1.0 - 1e-6 and mx <= 1.0 + 1e-6:
        imgs = (imgs + 1.0) * 127.5
    # else: assume already in [0,255] but float

    imgs = np.clip(imgs, 0.0, 255.0)
    return imgs.astype(np.uint8)


def preprocess_for_inception(images: np.ndarray, resize_to: Tuple[int, int]):
    tf = _require_tf()
    images_u8 = to_uint8_0_255(images)
    x = tf.convert_to_tensor(images_u8, dtype=tf.float32)
    x = tf.image.resize(x, resize_to, method="bilinear")
    return tf.keras.applications.inception_v3.preprocess_input(x)


def _require_tf():
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError(
            "TensorFlow is required for FID/Inception features. "
            "Install the project's environment (see environment.yml)."
        ) from e
    return tf


@dataclass
class InceptionV3FeatureExtractor:
    """Extract pooled InceptionV3 features (2048-D).
    """
    resize_to: Tuple[int, int] = (299, 299)
    batch_size: int = 128
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self._model: Optional[Any] = None  # tf.keras.Model

    def _require_tf(self):
        return _require_tf()

    def _build(self):
        tf = self._require_tf()
        return tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(self.resize_to[0], self.resize_to[1], 3),
        )

    @property
    def model(self):
        if self._model is None:
            tf = self._require_tf()
            if self.device is None:
                self._model = self._build()
            else:
                with tf.device(self.device):
                    self._model = self._build()
        return self._model

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Return Inception pooled features for a batch of images.

        Args:
          images: [N, H, W, 3] array.

        Returns:
          [N, 2048] float32 numpy array.
        """
        x = preprocess_for_inception(images, self.resize_to)
        feats = self.model(x, training=False)
        return feats.numpy()


@dataclass
class InceptionV3ProbabilityPredictor:
    """Predict InceptionV3 class probabilities (1000-way softmax)."""

    resize_to: Tuple[int, int] = (299, 299)
    batch_size: int = 128
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self._model: Optional[Any] = None  # tf.keras.Model

    def _require_tf(self):
        return _require_tf()

    def _build(self):
        tf = self._require_tf()
        return tf.keras.applications.InceptionV3(
            include_top=True,
            weights="imagenet",
            classifier_activation="softmax",
            input_shape=(self.resize_to[0], self.resize_to[1], 3),
        )

    @property
    def model(self):
        if self._model is None:
            tf = self._require_tf()
            if self.device is None:
                self._model = self._build()
            else:
                with tf.device(self.device):
                    self._model = self._build()
        return self._model

    def __call__(self, images: np.ndarray) -> np.ndarray:
        x = preprocess_for_inception(images, self.resize_to)
        probs = self.model(x, training=False)
        return probs.numpy()
