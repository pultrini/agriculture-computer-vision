import cv2
import numpy as np
import pytest

from src.agriculture_cv.pipelines.data_processing.nodes import (
    build_file_index,
    extract_image_metrics,
    process_dataset,
)


@pytest.fixture
def mock_plantvillage_dir(tmp_path):
    dataset_dir = tmp_path / "plantvillage"

    disease_dir = dataset_dir / "Tomato___Late_blight"
    disease_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0]  # Azul puro em BGR
    cv2.imwrite(str(disease_dir / "valid_leaf.jpg"), img)

    corrupt_file = disease_dir / "corrupt_leaf.jpg"
    corrupt_file.write_text("not an image content")

    healthy_dir = dataset_dir / "Apple___healthy"
    healthy_dir.mkdir(parents=True)
    cv2.imwrite(str(healthy_dir / "healthy_leaf.png"), img)

    return dataset_dir


def test_build_file_index(mock_plantvillage_dir):
    df = build_file_index(mock_plantvillage_dir)

    assert len(df) == 3
    assert set(df.columns) == {
        "path",
        "filename",
        "class_name",
        "crop",
        "disease",
        "is_healthy",
    }

    apple_row = df[df["crop"] == "Apple"].iloc[0]
    assert apple_row["disease"] == "healthy"
    assert bool(apple_row["is_healthy"]) is True

    tomato_row = df[df["crop"] == "Tomato"].iloc[0]
    assert tomato_row["disease"] == "Late_blight"
    assert bool(tomato_row["is_healthy"]) is False


def test_extract_image_metrics(mock_plantvillage_dir):
    valid_path = str(mock_plantvillage_dir / "Tomato___Late_blight" / "valid_leaf.jpg")
    corrupt_path = str(
        mock_plantvillage_dir / "Tomato___Late_blight" / "corrupt_leaf.jpg"
    )

    metrics_valid = extract_image_metrics(valid_path)
    assert metrics_valid["is_corrupted"] is False
    assert metrics_valid["height"] == 100
    assert metrics_valid["width"] == 100
    assert metrics_valid["channels"] == 3
    assert metrics_valid["mean_B"] == 254.0
    assert metrics_valid["mean_R"] == 0.0

    metrics_corrupt = extract_image_metrics(corrupt_path)
    assert metrics_corrupt["is_corrupted"] is True
    assert "height" not in metrics_corrupt


def test_process_dataset(mock_plantvillage_dir):
    df_index = build_file_index(mock_plantvillage_dir)
    df_processed = process_dataset(df_index, checkpoint_every=2)

    assert "is_corrupted" in df_processed.columns
    assert "brightness_mean" in df_processed.columns
    assert len(df_processed) == len(df_index)

    assert df_processed["is_corrupted"].sum() == 1
