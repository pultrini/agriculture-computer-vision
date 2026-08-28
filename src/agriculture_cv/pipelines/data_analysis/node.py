import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the corrupted images, with outside brightness or not standard"""
    n = len(df)

    n_corrupted = int(df["is_corrupted"].sum())
    df_clean = df[~df["is_corrupted"]].copy()
    logger.info(f"Removed {n_corrupted} corrupted images")

    dim_counts = (
        df_clean.groupby(["height", "width"]).size().sort_values(ascending=False)
    )
    mode_h, mode_w = dim_counts.index[0]

    inconsistent_mask = (df_clean["height"] != mode_h) | (df_clean["width"] != mode_w)
    n_inconsistent = int(inconsistent_mask.sum())
    df_clean = df_clean[~inconsistent_mask].copy()
    logger.info(
        f"Images with inconsistent dimension (≠ {mode_h}x{mode_w}) removed: {n_inconsistent}"
    )

    extreme_mask = (df_clean["brightness_mean"] < 5) | (
        df_clean["brightness_mean"] > 250
    )
    n_extreme = int(extreme_mask.sum())
    df_clean = df_clean[~extreme_mask].copy()
    logger.info(f"Images with extreme brightness removed: {n_extreme}")

    n_removed_total = n - len(df_clean)
    return df_clean


def generate_statistic_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> str:
    """Generate statistic report"""

    lines = []
    lines.append("=" * 70)
    lines.append("RELATÓRIO FASE 1 — CURADORIA E ANÁLISE DE DADOS (PlantVillage)")
    lines.append("=" * 70)
    lines.append(f"\nTotal de imagens indexadas: {len(df_raw)}")
    lines.append(f"Total de classes: {df_raw['class_name'].nunique()}")
    lines.append(f"Total de culturas (crops): {df_raw['crop'].nunique()}")

    lines.append("\n" + "-" * 70)
    lines.append("DISTRIBUIÇÃO DE CLASSES (após limpeza)")
    lines.append("-" * 70)

    class_dist = df_clean["class_name"].value_counts()
    class_pct = (class_dist / len(df_clean) * 100).round(2)
    dist_table = pd.DataFrame({"contagem": class_dist, "percentual(%)": class_pct})
    lines.append(dist_table.to_string())

    lines.append(
        f"\nClasse mais frequente: {class_dist.idxmax()} ({class_dist.max()} imagens)"
    )
    lines.append(
        f"Classe menos frequente: {class_dist.idxmin()} ({class_dist.min()} imagens)"
    )
    lines.append(f"Razão desbalanceamento: {class_dist.max() / class_dist.min():.1f}x")

    counts_arr = class_dist.values.astype(float)
    lines.append(
        f"\nEstatísticas da contagem: Média {np.mean(counts_arr):.1f} | Mediana {np.median(counts_arr):.1f} | Desvio {np.std(counts_arr):.1f}"
    )

    lines.append("\n" + "-" * 70)
    lines.append("DISTRIBUIÇÃO POR CULTURA (crop)")
    lines.append("-" * 70)
    lines.append(df_clean["crop"].value_counts().to_string())

    lines.append("\n" + "-" * 70)
    lines.append("BRILHO MÉDIO E COR POR CLASSE")
    lines.append("-" * 70)
    brightness = df_clean.groupby("class_name")["brightness_mean"].mean().sort_values()
    lines.append("Classes mais escuras:\n" + brightness.head(5).round(1).to_string())
    lines.append("\nClasses mais claras:\n" + brightness.tail(5).round(1).to_string())

    lines.append(f"\nBrilho médio global: {df_clean['brightness_mean'].mean():.2f}")
    lines.append(
        f"RGB global médio -> R: {df_clean['mean_R'].mean():.1f} G: {df_clean['mean_G'].mean():.1f} B: {df_clean['mean_B'].mean():.1f}"
    )

    logger.info("Relatório de estatísticas gerado com sucesso.")

    return "\n".join(lines)


def generate_plots(
    df_clean: pd.DataFrame,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Generate the distributions graphs using Plotly"""

    class_dist = df_clean["class_name"].value_counts().reset_index()
    class_dist.columns = ["Classe", "Quantidade"]
    fig_classes = px.bar(
        class_dist.sort_values("Quantidade"),
        x="Quantidade",
        y="Classe",
        orientation="h",
        title="Distribuição de imagens por classe",
        color_discrete_sequence=["#4c8c4a"],
    )
    fig_classes.update_layout(height=800)

    crop_dist = df_clean["crop"].value_counts().reset_index()
    crop_dist.columns = ["Cultura", "Quantidade"]
    fig_crop = px.bar(
        crop_dist.sort_values("Quantidade"),
        x="Quantidade",
        y="Cultura",
        orientation="h",
        title="Distribuição de imagens por cultura",
        color_discrete_sequence=["#2f6f4f"],
    )
    fig_bright = px.histogram(
        df_clean,
        x="brightness_mean",
        nbins=60,
        title="Distribuição do brilho médio",
        labels={"brightness_mean": "Brilho Médio (0-255)"},
        color_discrete_sequence=["#3a6ea5"],
    )

    fig_rgb = go.Figure()
    fig_rgb.add_trace(
        go.Histogram(
            x=df_clean["mean_R"], name="R", marker_color="red", opacity=0.5, nbinsx=60
        )
    )
    fig_rgb.add_trace(
        go.Histogram(
            x=df_clean["mean_G"], name="G", marker_color="green", opacity=0.5, nbinsx=60
        )
    )
    fig_rgb.add_trace(
        go.Histogram(
            x=df_clean["mean_B"], name="B", marker_color="blue", opacity=0.5, nbinsx=60
        )
    )
    fig_rgb.update_layout(
        title="Distribuição das médias de cor por canal (R, G, B)",
        barmode="overlay",
        xaxis_title="Intensidade média (0-255)",
    )

    logger.info("Gráficos Plotly gerados com sucesso.")
    return fig_classes, fig_crop, fig_bright, fig_rgb
