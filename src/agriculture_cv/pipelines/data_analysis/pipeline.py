from kedro.pipeline import Pipeline, node, pipeline

from .node import clean_metadata, generate_plots, generate_statistic_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_metadata,
                inputs="dataframe_with_metadata",
                outputs="plantvillage_metadata_clean",
                name="clean_metadata_node",
                tags=["clean_metadata"],
            ),
            node(
                func=generate_statistic_report,
                inputs=["dataframe_with_metadata", "plantvillage_metadata_clean"],
                outputs="relatorio_fase1",
                name="generate_statistic_report_node",
                tags=["generate_statistic_report"],
            ),
            node(
                func=generate_plots,
                inputs="plantvillage_metadata_clean",
                outputs=[
                    "plot_class_distribution",
                    "plot_crop_distribution",
                    "plot_brightness_histogram",
                    "plot_rgb_histogram",
                ],
                name="generate_plots_node",
                tags=["data_visualization"],
            ),
        ]
    )
