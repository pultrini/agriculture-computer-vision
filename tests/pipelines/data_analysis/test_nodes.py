import pandas as pd
import plotly.graph_objects as go
import pytest

from src.agriculture_cv.pipelines.data_analysis.node import (
    clean_metadata,
    generate_plots,
)


@pytest.fixture
def mock_uncleaned_df():
    """DataFrame com registros válidos e casos de borda (corrompido, dimensão fora do padrão, brilho extremo)."""
    return pd.DataFrame(
        {
            "path": [
                "img1.jpg",
                "img2.jpg",
                "img3.jpg",
                "img4.jpg",
                "img5.jpg",
                "img6.jpg",
            ],
            "class_name": [
                "Tomato___Late_blight",
                "Tomato___Late_blight",
                "Apple___healthy",
                "Apple___healthy",
                "Apple___healthy",
                "Apple___healthy",
            ],
            "crop": ["Tomato", "Tomato", "Apple", "Apple", "Apple", "Apple"],
            "is_corrupted": [False, True, False, False, False, False],
            "height": [256, 256, 100, 256, 256, 256],
            "width": [256, 256, 100, 256, 256, 256],
            "brightness_mean": [120.0, 120.0, 120.0, 2.0, 252.0, 150.0],
            "mean_R": [100.0, 100.0, 100.0, 2.0, 250.0, 130.0],
            "mean_G": [110.0, 110.0, 110.0, 2.0, 250.0, 140.0],
            "mean_B": [120.0, 120.0, 120.0, 2.0, 250.0, 150.0],
        }
    )


def test_clean_metadata(mock_uncleaned_df):
    df_clean = clean_metadata(mock_uncleaned_df)

    assert len(df_clean) == 2
    assert not df_clean["is_corrupted"].any()
    assert (df_clean["height"] == 256).all()
    assert (df_clean["width"] == 256).all()
    assert (
        (df_clean["brightness_mean"] >= 5) & (df_clean["brightness_mean"] <= 250)
    ).all()


def test_generate_plots(mock_uncleaned_df):
    df_clean = clean_metadata(mock_uncleaned_df)
    figures = generate_plots(df_clean)
    assert len(figures) == 4
    for fig in figures:
        assert isinstance(fig, go.Figure)
