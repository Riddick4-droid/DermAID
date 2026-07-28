
"""
Vision module: loads a ViT‑based dermatology classifier.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from .logger import logger
from .exceptions import (
    VisionModelError,
    VisionModelLoadError,
    VisionImageProcessingError,
)
from .config import settings


class VisionClassifier:
    """
    Skin‑condition classifier based on a ViT model.

    1. Tries to load a fine‑tuned model from the local path specified in
       config/agent_config.yaml (key `vision.local_model_path`).
    2. If that path is not set or not found, falls back to a pre‑trained
       Hugging Face model (key `vision.pre_trained_model`).
    """

    def __init__(self):
        vision_config = settings.vision
        self.confidence_threshold = vision_config["confidence_threshold"]
        self.local_path = vision_config.get("local_model_path")  # optional
        self.pre_trained_id = vision_config.get("pre_trained_model", "Anwarkh1n/ViT-Base-Skin-Cancer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.model, self.processor = self._load_model()

        # Class label mapping (from the loaded model's config)
        self.id2label = self.model.config.id2label
        logger.info(
            "VisionClassifier ready on %s with %d classes.",
            self.device,
            len(self.id2label),
        )

    def _load_model(self):
        """Attempt to load from local fine‑tuned dir, else from HF Hub."""
        # --- try local fine‑tuned model first ---
        if self.local_path:
            local_dir = Path(self.local_path)
            if local_dir.exists() and (local_dir / "config.json").exists():
                logger.info("Loading fine‑tuned vision model from %s", self.local_path)
                try:
                    model = ViTForImageClassification.from_pretrained(str(local_dir)).to(self.device)
                    processor = ViTImageProcessor.from_pretrained(str(local_dir))
                    return model, processor
                except Exception as exc:
                    raise VisionModelLoadError(f"Local model at {self.local_path} failed: {exc}") from exc
            else:
                logger.warning("Local model path '%s' not found or incomplete. Falling back to HF Hub.", self.local_path)

        # --- fallback: pre‑trained model from Hugging Face ---
        logger.info("Loading pre‑trained vision model from HF Hub: %s", self.pre_trained_id)
        try:
            model = ViTForImageClassification.from_pretrained(self.pre_trained_id).to(self.device)
            processor = ViTImageProcessor.from_pretrained(self.pre_trained_id)
            return model, processor
        except Exception as exc:
            raise VisionModelLoadError(f"Failed to load HF model {self.pre_trained_id}: {exc}") from exc

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Classify a skin image.

        Args:
            image_path: path to a readable image file (JPEG, PNG, etc.)

        Returns:
            (predicted_class_name, confidence) where confidence in [0,1].

        Raises:
            VisionImageProcessingError: if the image cannot be opened or processed.
        """
        # 1. Load and validate image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise VisionImageProcessingError(image_path, f"Cannot open image: {exc}") from exc

        # 2. Preprocess
        try:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        except Exception as exc:
            raise VisionImageProcessingError(image_path, f"Preprocessing failed: {exc}") from exc

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)

        confidence = confidence.item()
        predicted_class = self.id2label[predicted_idx.item()]

        logger.debug(
            "Predicted '%s' with confidence %.3f for image '%s'",
            predicted_class,
            confidence,
            os.path.basename(image_path),
        )
        return predicted_class, confidence



# Singleton (eager loading) – can be disabled in tests by mocking
try:
    classifier = VisionClassifier()
except VisionModelLoadError as exc:
    logger.error("Could not initialise vision classifier: %s", exc)
    classifier = None