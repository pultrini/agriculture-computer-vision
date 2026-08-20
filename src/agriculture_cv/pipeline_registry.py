"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from agriculture_cv.pipelines import data_processing
from agriculture_cv.pipelines import data_analysis


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_analysis_pipeline = data_analysis.create_pipeline()
    return {
        "__default__": data_processing_pipeline+data_analysis_pipeline,
        "data_processing": data_processing_pipeline,
        "data_analysis": data_analysis_pipeline,
    }
