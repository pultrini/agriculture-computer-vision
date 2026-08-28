from kedro.pipeline import Pipeline, node

from .node import evaluate_model, train_model_and_track_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_model_and_track_model,
                inputs=["train_data", "val_data", "params:model_training"],
                outputs="trained_model_checkpoint",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "trained_model_checkpoint",
                    "test_data",
                    "params:model_training",
                ],
                outputs=["evaluation_metrics", "confusion_matrix_plot"],
                name="evaluate_model_node",
            ),
        ]
    )
