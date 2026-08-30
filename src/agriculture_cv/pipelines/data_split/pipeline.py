from kedro.pipeline import Pipeline, node, pipeline

from .node import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="plantvillage_metadata_clean",
                outputs=["train_data", "test_data", "val_data", "label_mapping"],
                name="split_data_node",
            ),
        ]
    )
