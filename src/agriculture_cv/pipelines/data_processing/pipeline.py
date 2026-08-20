from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_file_index, process_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        nodes=[
            node(
                func=build_file_index,
                inputs="params:dataset_dir",
                outputs="dataset_index",
                name="build_file_index_node",
                tags=["processing_data"]
            ),
            node(
                func=process_dataset,
                inputs="dataset_index",
                outputs="dataframe_with_metadata",
                name="process_dataset_node",
                tags=["processing_images", "images_metrics"]
            )
        ]
    )