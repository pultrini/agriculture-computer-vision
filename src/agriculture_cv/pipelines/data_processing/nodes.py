from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import time

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_file_index(dataset_dir: Path | str) -> pd.DataFrame:
    """
    Build the dataset csv with a path for a image dataset
    Works only with the PlantVillage-Dataset
    Args:
        dataset_dir: Path to the dataset directory

   Returns:
       Dataframe with a path for the image dataset
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if "___" in class_name:
            crop, disease = class_name.rsplit("___", 1)
        else:
            crop, disease = class_name, "unknown"
        healthy = disease.lower() == "healthy"

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            rows.append({
                'path': str(img_path),
                'filename': img_path.name,
                'class_name': class_name,
                'crop': crop,
                'disease': disease,
                'is_healthy': healthy,
            })
    logger.info(f"Built {len(rows)} rows")
    return pd.DataFrame(rows)

def extract_image_metrics(path: str) -> dict:
    """Read the image and extract image metrics
    medium brightness, statistical histogram from colors
    :return is_corrupted if image is corrupted"""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return {"is_corrupted": True}

        h, w, c = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())

        b_mean, g_mean, r_mean = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()

        return {
            "is_corrupted": False,
            "height": h,
            "width": w,
            "channels": c,
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "mean_R": float(r_mean),
            "mean_G": float(g_mean),
            "mean_B": float(b_mean),
        }
    except Exception as e:
        return {"is_corrupted": True}

def process_dataset(df: pd.DataFrame, checkpoint_every: int = 5000) -> pd.DataFrame:
    """
    Process the dataset to add the image metrics.
    Args:
        df: Dataframe with a path for the image dataset and information
        checkpoint_every: how mutch iterations needs to save point dataset

    Returns:
        Dataframe with image metrics
    """
    metrics = []
    t0 = time.time()
    n = len(df)
    for i, path in enumerate(df["path"].values):
        metrics.append(extract_image_metrics(path))
        if (i + 1) % checkpoint_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate
            logger.info(f"[{i + 1}/{n}] {rate:.1f} imgs/s, ETA {eta / 60:.1f} min")

    metrics_df = pd.DataFrame(metrics)
    full_df = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
    return full_df