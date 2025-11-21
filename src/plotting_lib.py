"""
Custom plotting helpers tailored for the Syria-before-war notebooks.

The module configures a serif-focused Matplotlib style on import and provides
helpers for consistent color usage across multiple figures.
"""

from __future__ import annotations

import textwrap
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from collections import Counter, defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

# Pastel, print-friendly palette centered on oranges, reds, maroons, and greens.
PALETTE: List[str] = [
    "#855C75FF",
    "#D9AF6BFF",
    "#AF6458FF",
    "#736F4CFF",
    "#526A83FF",
    "#625377FF",
    "#68855CFF",
    "#9C9C5EFF",
    "#A06177FF",
    "#8C785DFF",
    "#467378FF",
    "#7C7C7CFF",
]


def _configure_style() -> None:
    """Apply the shared Matplotlib look-and-feel for the notebooks."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#4a2f2f",
            "axes.labelcolor": "#331c1c",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "font.family": "serif",
            "font.serif": ["Libertinus Serif"],
            "font.size": 12,
            "text.color": "#2c1e1e",
            "xtick.color": "#533131",
            "xtick.labelsize": 11,
            "ytick.color": "#533131",
            "ytick.labelsize": 11,
            "grid.color": "#eadbd0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.fontsize": 11,
            "figure.autolayout": True,
            "figure.dpi": 170,
            "savefig.dpi": 170,
        }
    )


_configure_style()


def _format_percentage(value: float) -> str:
    """Return a French-style percentage with one decimal place."""
    percentage = value * 100
    formatted = f"{percentage:.1f}".replace(".", ",")
    return f"{formatted} %"


def plot_analysis_pipeline(
    ax: Optional[plt.Axes] = None,
    *,
    figsize: tuple[float, float] = (10, 12),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a high-level pipeline summarizing the article analysis process.

    The diagram highlights the successive phases applied to the RTBF corpus:
    preparation, pre-analysis, stereotype detection, and the final plotting step.
    """

    phases = [
        {
            "title": "RTBF Corpus",
            "items": ["Corpus complet des articles RTBF"],
            "color": PALETTE[0],
        },
        {
            "title": "Phase 1 — Préparation des données",
            "items": [
                "Filtrer les articles mentionnant « Syrie »",
                "Filtrer par catégories (MONDE, SOCIETE, MEDIAS, ...)",
            ],
            "color": PALETTE[1],
        },
        {
            "title": "Phase 2 — Pré-analyse",
            "items": [
                "Notes de lecture -> liste de mots-clés liée à l'idée reçue",
                "Envoi des notes + mots-clés à Mistral pour amélioration",
                "Filtrage du corpus avec mots-clés obligatoires/optionnels (>= 3 matches)",
            ],
            "color": PALETTE[2],
        },
        {
            "title": "Phase 3 — Détection des stéréotypes",
            "items": [
                "Création d'un agent LLM (Mistral Small) avec consignes de détection",
                "Envoi de chaque article à l'agent et collecte des réponses formatées",
            ],
            "color": PALETTE[4],
        },
        {
            "title": "Phase 4 — Analyse des résultats",
            "items": ["Analyse et visualisation en Python + Matplotlib"],
            "color": PALETTE[6],
        },
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    center_x = 0.5
    spacing = 2.3
    start_y = (len(phases) - 1) * spacing

    boxes: list[dict[str, Any]] = []
    for idx, phase in enumerate(phases):
        y_pos = start_y - idx * spacing
        height = 0.55 + 0.35 * max(len(phase["items"]), 1)
        width = 0.88
        left = center_x - width / 2
        bottom = y_pos - height / 2

        boxes.append(
            {
                "phase": phase,
                "y": y_pos,
                "height": height,
                "width": width,
                "left": left,
                "bottom": bottom,
            }
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, start_y + 1.2)
    ax.axis("off")

    for idx, box in enumerate(boxes):
        phase = box["phase"]
        rounded_box = FancyBboxPatch(
            (box["left"], box["bottom"]),
            box["width"],
            box["height"],
            boxstyle="round,pad=0.35",
            facecolor=phase["color"],
            edgecolor="#2b1f1f",
            linewidth=1.1,
            alpha=0.9,
            mutation_scale=6,
        )
        ax.add_patch(rounded_box)

        wrapped_items = [
            textwrap.fill(f"• {item}", width=55, subsequent_indent="  ")
            for item in phase["items"]
        ]
        text_block = "\n".join([phase["title"]] + wrapped_items)
        ax.text(
            center_x,
            box["y"],
            text_block,
            ha="center",
            va="center",
            fontsize=12,
            color="#0f0f0f",
            fontweight="bold",
        )

        if idx < len(boxes) - 1:
            next_box = boxes[idx + 1]
            ax.annotate(
                "",
                xy=(center_x, next_box["y"] + next_box["height"] / 2),
                xytext=(center_x, box["y"] - box["height"] / 2),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#5d452f",
                    "linewidth": 2.0,
                    "shrinkA": 3,
                    "shrinkB": 5,
                },
            )

    title_text = title or "Pipeline d'analyse des articles RTBF"
    fig.suptitle(title_text, fontsize=16, fontweight="bold", y=1.02)

    return ax


def plot_article_distribution_by_year(
    df: pd.DataFrame,
    keywords: Optional[Iterable[str]] = None,
    *,
    pub_date_col: str = "pub_date",
    text_columns: Sequence[str],
    match_all_keywords: bool = True,
    title: Optional[str] = None,
    periods: Optional[Sequence[Mapping[str, str]]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Axes:
    """
    Plot article counts per publication year and highlight keyword coverage.

    Parameters
    ----------
    df:
        DataFrame that contains at least the publication date column.
    keywords:
        Iterable of keywords (case-insensitive) that must all be present in an
        article for it to be counted toward the highlighted proportion.
    pub_date_col:
        Name of the column with publication timestamps (object or datetime).
    text_columns:
        Columns to scan for keywords. All specified columns must exist in ``df``.
    match_all_keywords:
        When True (default) an article must contain every keyword. When False it
        is counted if it contains at least one keyword.
    title:
        Optional override for the chart title (French text recommended).
    periods:
        Optional list of dicts defining custom periods with ``title``,
        ``start_date`` and ``end_date``. When provided, bars are aggregated by
        these spans instead of calendar years.
    ax:
        Optional Matplotlib Axes to reuse. When None a new figure is created.
    figsize:
        Figure size to use when creating a new Axes.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the bar plot.
    """

    df = df.copy()

    if pub_date_col not in df.columns:
        raise ValueError(f"'{pub_date_col}' column not found in DataFrame.")

    provided_keywords = [kw.strip() for kw in (keywords or []) if kw and kw.strip()]
    normalized_keywords = [kw.lower() for kw in provided_keywords]
    pub_dates = pd.to_datetime(df[pub_date_col], errors="coerce")
    valid_mask = pub_dates.notna()

    if not valid_mask.any():
        raise ValueError("No valid publication dates found to plot.")

    working_df = df.loc[valid_mask].copy()
    timestamps = pub_dates.loc[valid_mask]

    def _parse_periods(period_defs: Sequence[Mapping[str, str]]) -> List[dict]:
        parsed: List[dict] = []
        for period in period_defs:
            title = period.get("title")
            start = pd.to_datetime(period.get("start_date"), errors="coerce")
            end_val = period.get("end_date")
            if isinstance(end_val, str) and end_val.lower() in {"present", "actuel"}:
                end = pd.Timestamp.max
            elif end_val is None:
                end = pd.Timestamp.max
            else:
                end = pd.to_datetime(end_val, errors="coerce")
            if title is None or pd.isna(start) or pd.isna(end):
                raise ValueError(
                    "Chaque période doit définir un titre, un start_date et un end_date valides."
                )
            if start > end:
                raise ValueError(
                    f"La période '{title}' a un start_date postérieur à son end_date."
                )
            parsed.append({"title": title, "start": start, "end": end})
        parsed.sort(key=lambda item: item["start"])
        return parsed

    use_periods = periods is not None
    if use_periods:
        parsed_periods = _parse_periods(periods or [])
        timeline_labels = [period["title"] for period in parsed_periods]
        assigned_labels: List[Optional[str]] = []
        unmatched_dates: List[pd.Timestamp] = []
        for ts in timestamps:
            label = None
            for period in parsed_periods:
                if period["start"] <= ts <= period["end"]:
                    label = period["title"]
                    break
            assigned_labels.append(label)
            if label is None:
                unmatched_dates.append(ts)
        if unmatched_dates:
            formatted = ", ".join(
                sorted({ts.strftime("%Y-%m-%d") for ts in unmatched_dates})
            )
            raise ValueError(
                "Certaines dates de publication ne correspondent à aucune période fournie: "
                f"{formatted}"
            )
        working_df["_timeline_label"] = pd.Categorical(
            assigned_labels, categories=timeline_labels, ordered=True
        )
    else:
        years = timestamps.dt.year.astype(int)
        timeline_labels = sorted(years.unique())
        working_df["_timeline_label"] = pd.Categorical(
            years, categories=timeline_labels, ordered=True
        )

    timeline_counts = (
        working_df.groupby("_timeline_label")
        .size()
        .reindex(timeline_labels, fill_value=0)
    )

    if not text_columns:
        raise ValueError("text_columns must include at least one column name.")

    missing = [col for col in text_columns if col not in working_df.columns]
    if missing:
        raise ValueError(
            f"text_columns contains columns not found in DataFrame: {missing}"
        )

    text_frame = working_df[text_columns].fillna("").astype(str)
    concatenated_text = text_frame.agg(" ".join, axis=1).str.lower()
    if normalized_keywords:
        if match_all_keywords:
            keyword_mask = concatenated_text.apply(
                lambda text: all(kw in text for kw in normalized_keywords)
            )
        else:
            keyword_mask = concatenated_text.apply(
                lambda text: any(kw in text for kw in normalized_keywords)
            )
    else:
        keyword_mask = pd.Series(False, index=working_df.index, dtype=bool)

    keyword_counts = (
        working_df.loc[keyword_mask]
        .groupby("_timeline_label")
        .size()
        .reindex(timeline_labels, fill_value=0)
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    base_color = PALETTE[1]
    highlight_color = PALETTE[2]

    positions = np.arange(len(timeline_labels))
    ax.bar(
        positions,
        timeline_counts.values,
        color=base_color,
        edgecolor=PALETTE[0],
        label="Articles",
    )

    if normalized_keywords:
        keywords_text = f'"{", ".join(provided_keywords)}"' if provided_keywords else ""
        label_text = f"Mentionnant {keywords_text}"
        ax.bar(
            positions,
            keyword_counts.values,
            width=0.55,
            color=highlight_color,
            label=label_text,
        )

        proportions = keyword_counts / timeline_counts.replace(0, pd.NA)
        max_height = timeline_counts.max()
        for x, count, proportion in zip(positions, keyword_counts.values, proportions):
            if pd.isna(proportion):
                continue
            ax.text(
                x,
                count + max(max_height * 0.015, 0.5),
                _format_percentage(proportion),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="#2f3b1c",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(label) for label in timeline_labels], rotation=0, ha="center"
    )
    ax.set_xlabel("Période" if use_periods else "Année de publication")
    ax.set_ylabel("Nombre d'articles")
    ax.set_title(title or "Répartition des articles par année")
    ax.grid(axis="y", alpha=0.4)
    ax.legend()

    return ax


def plot_keyword_category_distribution(
    df: pd.DataFrame,
    keywords: Optional[Iterable[str]] = None,
    *,
    text_columns: Sequence[str],
    category_column: str,
    match_all_keywords: bool = True,
    title: Optional[str] = None,
    min_share: float = 0.01,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8, 8),
) -> plt.Axes:
    """
    Plot a horizontal histogram showing keyword-matching articles per category.

    Parameters
    ----------
    df:
        DataFrame holding article metadata.
    keywords:
        Iterable of keywords to search for in the specified text columns.
    text_columns:
        Columns to scan for keywords. Every supplied column must exist in ``df``.
    category_column:
        Column containing the category label for each article.
    match_all_keywords:
        When True (default) an article must contain every keyword. When False it
        is counted if it contains at least one keyword.
    title:
        Optional chart title override (French text recommended).
    min_share:
        Minimum share (between 0 and 1) a category must represent to avoid being
        grouped into the "Autres" bucket. Defaults to 1%.
    ax:
        Optional Matplotlib Axes to reuse.
    figsize:
        Figure size used when creating a new Axes.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the histogram.
    """
    df = df.copy()

    if not text_columns:
        raise ValueError("text_columns must include at least one column.")

    if not 0 <= min_share < 1:
        raise ValueError("min_share must be between 0 (inclusive) and 1 (exclusive).")

    missing = [col for col in text_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"text_columns contains columns not found in DataFrame: {missing}"
        )

    if category_column not in df.columns:
        raise ValueError(f"category_column '{category_column}' not found in DataFrame.")

    provided_keywords = [kw.strip() for kw in (keywords or []) if kw and kw.strip()]
    normalized_keywords = [kw.lower() for kw in provided_keywords]

    text_frame = df[text_columns].fillna("").astype(str)
    concatenated_text = text_frame.agg(" ".join, axis=1).str.lower()

    if normalized_keywords:
        if match_all_keywords:
            keyword_mask = concatenated_text.apply(
                lambda text: all(kw in text for kw in normalized_keywords)
            )
        else:
            keyword_mask = concatenated_text.apply(
                lambda text: any(kw in text for kw in normalized_keywords)
            )
    else:
        keyword_mask = pd.Series(True, index=df.index, dtype=bool)

    filtered = df.loc[keyword_mask]
    if filtered.empty:
        raise ValueError(
            "Aucun article ne correspond aux mots-clés fournis dans les colonnes données."
        )

    category_series = filtered[category_column].fillna("Catégorie inconnue")
    raw_counts = category_series.value_counts()

    total = raw_counts.sum()
    shares = raw_counts / total

    major = raw_counts[shares >= min_share]
    minor = raw_counts[shares < min_share]

    major_sorted = major.sort_values(ascending=True)
    if minor.empty:
        category_counts = major_sorted
    else:
        category_counts = pd.concat(
            [
                pd.Series({"Autres": minor.sum()}),
                major_sorted,
            ],
            ignore_index=False,
        )

    shares = category_counts / category_counts.sum()
    categories = category_counts.index.tolist()
    num_categories = len(categories)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = (PALETTE * (num_categories // len(PALETTE) + 1))[:num_categories]

    spacing_factor = 0.8 if num_categories > 1 else 1.0
    positions = np.arange(num_categories) * spacing_factor
    bar_height = 0.35 if num_categories > 12 else 0.45

    bars = ax.barh(
        positions,
        category_counts.values,
        height=bar_height,
        color=colors,
        zorder=2,
        edgecolor="#2b1f1f",
        linewidth=0.4,
    )

    ax.grid(axis="x", linestyle="-", alpha=0.4, color="#b9b1a5", zorder=1)

    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b9b1a5")

    ax.set_yticks(positions)
    ax.set_yticklabels(categories, fontsize=12, color="#111111")
    ax.tick_params(axis="x", labelsize=11, colors="#111111")
    ax.xaxis.tick_top()
    ax.set_xlabel("Nombre d'articles mentionnant les mots-clés")
    ax.set_ylabel("")
    ax.margins(y=0.02)
    if num_categories:
        ax.set_ylim(
            positions[0] - bar_height * 1.2,
            positions[-1] + bar_height * 1.2,
        )

    max_count = category_counts.max()
    share_values = shares.values
    for bar, share in zip(bars, share_values):
        ax.text(
            bar.get_width() + max(max_count * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            _format_percentage(share),
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )

    qualifier = "tous les mots-clés" if match_all_keywords else "au moins un mot-clé"
    keywords_text = f" ({', '.join(provided_keywords)})" if provided_keywords else ""

    ax.set_title(
        title or f"Répartition des articles mentionnant {qualifier}{keywords_text}",
        pad=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return ax


__all__ = [
    "plot_analysis_pipeline",
    "plot_article_distribution_by_year",
    "plot_keyword_category_distribution",
    "plot_stereotype_stance_over_time",
    "plot_actor_mentions",
    "plot_actor_category_trends",
    "plot_prism_trends",
    "PALETTE",
]


def plot_stereotype_stance_over_time(
    df: pd.DataFrame,
    *,
    pub_date_col: str = "pub_date",
    presence_col: str = "st_present",
    mode_col: str = "st_mode",
    stereotype_name: Optional[str] = None,
    title: Optional[str] = None,
    periods: Optional[Sequence[Mapping[str, str]]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (16, 6),
    show_percentages: bool = True,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot yearly stance percentages for stereotypes along with article counts.

    Returns
    -------
    (Axes, Axes)
        Axes for percentages and the twin Axes for counts.
    """

    df = df.copy()

    missing_cols = [
        col for col in (pub_date_col, presence_col, mode_col) if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")

    pub_dates = pd.to_datetime(df[pub_date_col], errors="coerce")
    valid_mask = pub_dates.notna()
    if not valid_mask.any():
        raise ValueError("Aucune date de publication valide n'a été trouvée.")

    df = df.loc[valid_mask].copy()
    timestamp_series = pub_dates.loc[valid_mask]

    def _parse_periods(period_defs: Sequence[Mapping[str, str]]) -> List[dict]:
        parsed: List[dict] = []
        for period in period_defs:
            title = period.get("title")
            start = pd.to_datetime(period.get("start_date"), errors="coerce")
            end_val = period.get("end_date")
            if isinstance(end_val, str) and end_val.lower() in {"present", "actuel"}:
                end = pd.Timestamp.max
            elif end_val is None:
                end = pd.Timestamp.max
            else:
                end = pd.to_datetime(end_val, errors="coerce")
            if title is None or pd.isna(start) or pd.isna(end):
                raise ValueError(
                    "Chaque période doit inclure un titre, un start_date et un end_date valides."
                )
            if start > end:
                raise ValueError(
                    f"La période '{title}' a un start_date postérieur à son end_date."
                )
            parsed.append({"title": title, "start": start, "end": end})
        parsed.sort(key=lambda item: item["start"])
        return parsed

    use_periods = periods is not None
    if use_periods:
        parsed_periods = _parse_periods(periods or [])
        timeline_labels = [period["title"] for period in parsed_periods]
        assigned_labels: List[Optional[str]] = []
        unmatched = []
        for ts in timestamp_series:
            label = None
            for period in parsed_periods:
                if period["start"] <= ts <= period["end"]:
                    label = period["title"]
                    break
            assigned_labels.append(label)
            if label is None:
                unmatched.append(ts)
        if unmatched:
            formatted = ", ".join(sorted({ts.strftime("%Y-%m-%d") for ts in unmatched}))
            raise ValueError(
                "Certaines dates ne correspondent à aucune période fournie: "
                f"{formatted}"
            )
        df["_timeline_label"] = pd.Categorical(
            assigned_labels, categories=timeline_labels, ordered=True
        )
    else:
        years = timestamp_series.dt.year.astype(int)
        timeline_labels = sorted(years.unique())
        df["_timeline_label"] = pd.Categorical(
            years, categories=timeline_labels, ordered=True
        )

    presence_mask = df[presence_col].astype(str).str.strip().str.lower().eq("yes")
    stereotype_df = df.loc[presence_mask]
    if stereotype_df.empty:
        raise ValueError("Aucun article avec stéréotype présent n'a été trouvé.")

    timeline_counts = (
        stereotype_df.groupby("_timeline_label")
        .size()
        .reindex(timeline_labels, fill_value=0)
    )

    normalized_mode = (
        stereotype_df[mode_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "critique": "critiqué",
                "critiquee": "critiqué",
                "nuance": "nuancé",
                "nuancee": "nuancé",
            }
        )
    )
    stereotype_df = stereotype_df.assign(_mode=normalized_mode)

    modes = {
        "reproduit": "Reproduit",
        "nuancé": "Nuancé",
        "critiqué": "Critiqué",
    }

    stance_counts = (
        stereotype_df.groupby(["_timeline_label", "_mode"])
        .size()
        .unstack("_mode")
        .reindex(timeline_labels, fill_value=0)
    )

    for key in modes:
        if key not in stance_counts:
            stance_counts[key] = 0

    stance_counts = stance_counts[list(modes.keys())]
    if show_percentages:
        stance_values = (
            stance_counts.div(timeline_counts, axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            * 100.0
        )
    else:
        stance_values = stance_counts

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax_lines = ax
    ax_bars = ax_lines.twinx()

    if use_periods:
        x_coords = np.arange(len(timeline_labels))
        ax_lines.set_xticks(x_coords)
        ax_lines.set_xticklabels(timeline_labels, rotation=0, ha="center")
    else:
        x_coords = np.array(timeline_labels, dtype=float)
    color_cycle = PALETTE

    for idx, (mode_key, label) in enumerate(modes.items()):
        ax_lines.plot(
            x_coords,
            stance_values[mode_key].values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=color_cycle[idx % len(color_cycle)],
            label=label,
        )

    bar_color = color_cycle[4 % len(color_cycle)]
    ax_bars.bar(
        x_coords,
        timeline_counts.values,
        color=bar_color,
        alpha=0.35,
        label="Total d'articles",
        width=0.65,
        edgecolor="#2b1f1f",
        linewidth=0.4,
    )

    if show_percentages:
        ax_lines.set_ylabel("Part des articles (stéréotype présent)")
        ax_lines.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        ax_lines.set_ylabel("Nombre d'articles (stéréotype présent)")
        ax_lines.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax_lines.set_xlabel("Période" if use_periods else "Année de publication")
    ax_bars.set_ylabel("Nombre d'articles (stéréotype présent)")

    ax_lines.grid(axis="y", alpha=0.25)

    metric_label = "part" if show_percentages else "nombre"
    default_title = f"Évolution du {metric_label} de positions face au stéréotype"
    main_title = title or default_title
    ax_lines.set_title(main_title, pad=34)

    legend_lines = ax_lines.legend(loc="upper left", title="Mode du stéréotype")
    legend_bars = ax_bars.legend(loc="upper right")
    for legend in (legend_lines, legend_bars):
        for text in legend.get_texts():
            text.set_color("#2c1e1e")

    if stereotype_name:
        ax_lines.text(
            0.5,
            1.22,
            f"« {stereotype_name} »",
            transform=ax_lines.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            color="#4a2f2f",
        )

    if use_periods:
        fig.tight_layout(rect=(0, 0, 1, 1))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    return ax_lines, ax_bars


def plot_actor_mentions(
    df: pd.DataFrame,
    actor_categories: dict[str, list[str]],
    *,
    actor_column: str = "st_acteurs",
    group_by_category: bool = False,
    category_color_map: Optional[dict[str, str]] = None,
    stereotype_name: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 8),
) -> plt.Axes:
    """
    Plot a horizontal bar chart showing actor mention counts grouped by category.

    Parameters
    ----------
    df:
        DataFrame containing a column with actor names (lists of strings).
    actor_categories:
        Mapping of category names to the actors it contains. Only actors present
        in this dictionary are counted.
    actor_column:
        Column containing actors per article (iterable of strings).
    group_by_category:
        When True the chart aggregates counts per category instead of listing
        each actor individually.
    category_color_map:
        Optional mapping of category names to explicit colors. When omitted a
        deterministic palette is applied alphabetically across categories so
        both detailed and aggregated views line up.
    stereotype_name:
        Optional label displayed above the title, similar to a quotation.
    title:
        Optional override for the chart title.
    ax:
        Optional Matplotlib Axes to reuse.
    figsize:
        Figure size used when creating a new Axes.
    """

    df = df.copy()

    if actor_column not in df.columns:
        raise ValueError(f"'{actor_column}' column not found in DataFrame.")

    if not actor_categories:
        raise ValueError("actor_categories must contain at least one entry.")

    actor_to_category: dict[str, str] = {}
    for category, actors in actor_categories.items():
        for actor in actors:
            normalized = actor.strip()
            actor_to_category[normalized] = category

    counter: Counter[str] = Counter()
    series = df[actor_column]
    for value in series:
        if isinstance(value, (list, tuple, set)):
            actors = value
        elif pd.isna(value) or value is None:
            continue
        else:
            actors = [value]

        for actor in actors:
            if pd.isna(actor):
                continue
            normalized_actor = str(actor).strip()
            if normalized_actor in actor_to_category:
                counter[normalized_actor] += 1

    if not counter:
        raise ValueError(
            "Aucun acteur du dictionnaire n'a été trouvé dans la colonne fournie."
        )

    sorted_actors = counter.most_common()
    category_display_name = lambda c: c.replace("_", " ").title()

    category_palette: dict[str, str] = {}
    if category_color_map:
        for category, color in category_color_map.items():
            category_palette[category] = color

    alphabetical_categories = sorted(actor_categories.keys())
    palette_index = 0
    for category in alphabetical_categories:
        if category not in category_palette:
            category_palette[category] = PALETTE[palette_index % len(PALETTE)]
            palette_index += 1

    categories_sequence: list[str] = []

    if group_by_category:
        category_counts: Counter[str] = Counter()
        for actor_name, count in sorted_actors:
            category_counts[actor_to_category[actor_name]] += count
        sorted_items = category_counts.most_common()
        labels = [category_display_name(cat) for cat, _ in sorted_items]
        counts = np.array([count for _, count in sorted_items])
        colors = [category_palette.get(cat, PALETTE[0]) for cat, _ in sorted_items]
    else:
        labels = [name for name, _ in sorted_actors]
        counts = np.array([count for _, count in sorted_actors])
        colors = [category_palette[actor_to_category[name]] for name in labels]
        categories_sequence = [actor_to_category[name] for name in labels]

    positions = np.arange(len(labels))
    num_bars = len(labels)
    bar_height = 0.75 if num_bars < 12 else 0.45
    bar_height = min(bar_height, 0.6)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(
        positions,
        counts,
        color=colors,
        height=bar_height,
        edgecolor="#2b1f1f",
        linewidth=0.5,
        zorder=2,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=11, color="#111111")
    ax.tick_params(axis="x", labelsize=11, colors="#111111")
    ax.xaxis.tick_top()
    ax.set_xlabel("Nombre d'occurrences")
    ax.set_ylabel("")

    ax.grid(axis="x", linestyle="-", alpha=0.3, color="#b9b1a5", zorder=1)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b9b1a5")
    ax.margins(y=0.01)
    ax.invert_yaxis()

    max_count = counts.max()
    for bar in bars:
        ax.text(
            bar.get_width() + max(max_count * 0.01, 0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#111111",
        )

    if not group_by_category:
        seen = []
        for category in categories_sequence:
            if category not in seen:
                seen.append(category)
        used_categories = seen
        legend_handles = [
            Patch(
                color=category_palette.get(category, PALETTE[0]),
                label=category_display_name(category),
            )
            for category in used_categories
        ]
        ax.legend(
            handles=legend_handles,
            title="Catégories",
            loc="lower right",
            frameon=False,
        )

    default_title = (
        "Occurrences des catégories d'acteurs mentionnées"
        if group_by_category
        else "Occurrences des acteurs mentionnés"
    )
    ax.set_title(title or default_title, pad=12)
    if stereotype_name:
        ax.text(
            0.5,
            1.12,
            f"« {stereotype_name} »",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            color="#4a2f2f",
        )

    fig.tight_layout(rect=(0, 0, 0.82, 0.9))
    return ax


def plot_actor_category_trends(
    df: pd.DataFrame,
    actor_categories: dict[str, list[str]],
    *,
    actor_column: str = "st_acteurs",
    pub_date_col: str = "pub_date",
    show_percentages: bool = False,
    periods: Optional[Sequence[Mapping[str, str]]] = None,
    stereotype_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (14, 6),
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot the evolution of actor category mentions over publication years.

    Parameters
    ----------
    df:
        DataFrame containing publication dates and actor mentions.
    actor_categories:
        Mapping of category names to lists of actor names.
    actor_column:
        Column that stores the list of actors mentioned per article.
    pub_date_col:
        Column containing the publication date (parsable to datetime).
    show_percentages:
        When True, shows the share of mentions per category (per year) instead
        of absolute counts.
    periods:
        Optional ordered list of dicts describing custom time periods with
        ``title``, ``start_date``, and ``end_date`` (inclusive). Provide dates in
        ISO formats such as ``\"2011-03-01\"`` or ``\"2018\"``. Use ``None`` or
        ``\"present\"`` for open-ended end dates. All article dates must fall
        into one of the periods when this argument is provided.
    stereotype_name:
        Optional label displayed above the chart as a quote.
    ax:
        Optional Matplotlib Axes to reuse. A new figure is created otherwise.
    figsize:
        Figure size used when creating a new Axes.
    """

    df = df.copy()

    if actor_column not in df.columns:
        raise ValueError(f"'{actor_column}' column not found in DataFrame.")
    if pub_date_col not in df.columns:
        raise ValueError(f"'{pub_date_col}' column not found in DataFrame.")
    if not actor_categories:
        raise ValueError("actor_categories must contain at least one entry.")

    actor_to_category: dict[str, str] = {}
    for category, actors in actor_categories.items():
        for actor in actors:
            actor_to_category[actor.strip()] = category

    pub_dates = pd.to_datetime(df[pub_date_col], errors="coerce")
    valid_mask = pub_dates.notna()
    if not valid_mask.any():
        raise ValueError("Aucune date de publication valide n'a été trouvée.")

    df = df.loc[valid_mask].copy()
    timestamp_series = pub_dates.loc[valid_mask]

    def _parse_periods(period_defs: Sequence[Mapping[str, str]]) -> List[dict]:
        parsed: List[dict] = []
        for period in period_defs:
            title = period.get("title")
            start = pd.to_datetime(period.get("start_date"), errors="coerce")
            end_value = period.get("end_date")
            if isinstance(end_value, str) and end_value.lower() in {
                "present",
                "actuel",
            }:
                end = pd.Timestamp.max
            elif end_value is None:
                end = pd.Timestamp.max
            else:
                end = pd.to_datetime(end_value, errors="coerce")
            if title is None or pd.isna(start) or pd.isna(end):
                raise ValueError(
                    "Chaque période doit inclure un titre, un start_date et un end_date "
                    "parsables (format ISO recommandé, ex: '2011-03-01')."
                )
            if start > end:
                raise ValueError(
                    f"La période '{title}' a un start_date postérieur à son end_date."
                )
            parsed.append({"title": title, "start": start, "end": end})
        parsed.sort(key=lambda item: item["start"])
        return parsed

    use_periods = periods is not None
    if use_periods:
        parsed_periods = _parse_periods(periods or [])
        timeline_labels = [item["title"] for item in parsed_periods]
        assigned_labels: List[Optional[str]] = []
        unmatched_dates: List[pd.Timestamp] = []
        for ts in timestamp_series:
            label = None
            for period in parsed_periods:
                if period["start"] <= ts <= period["end"]:
                    label = period["title"]
                    break
            assigned_labels.append(label)
            if label is None:
                unmatched_dates.append(ts)
        if unmatched_dates:
            formatted = ", ".join(
                sorted({ts.strftime("%Y-%m-%d") for ts in unmatched_dates})
            )
            raise ValueError(
                "Certaines dates de publication ne correspondent à aucune période fournie: "
                f"{formatted}"
            )
        df["_timeline_label"] = assigned_labels
    else:
        df["_timeline_label"] = timestamp_series.dt.year.astype(int)
        timeline_labels = sorted(df["_timeline_label"].unique())

    if not timeline_labels:
        raise ValueError("Aucun point temporel n'a été créé pour ce graphique.")

    timeline_article_counts = (
        df.groupby("_timeline_label").size().reindex(timeline_labels, fill_value=0)
    )

    timeline_mentions: dict[Any, Counter[str]] = {
        label: Counter() for label in timeline_labels
    }

    for label, value in zip(df["_timeline_label"], df[actor_column]):
        if isinstance(value, (list, tuple, set)):
            actors = value
        elif pd.isna(value) or value is None:
            continue
        else:
            actors = [value]

        for actor in actors:
            if pd.isna(actor):
                continue
            normalized = str(actor).strip()
            category = actor_to_category.get(normalized)
            if category:
                timeline_mentions[label][category] += 1

    if not any(timeline_mentions[label] for label in timeline_labels):
        raise ValueError(
            "Aucune mention d'acteur correspondant aux catégories fournies n'a été trouvée."
        )

    categories = sorted(actor_categories.keys())

    counts_matrix = pd.DataFrame(
        {
            category: [
                timeline_mentions[label].get(category, 0) for label in timeline_labels
            ]
            for category in categories
        },
        index=timeline_labels,
    )

    if show_percentages:
        counts_sum = counts_matrix.sum(axis=1).replace(0, np.nan)
        display_values = counts_matrix.div(counts_sum, axis=0).fillna(0) * 100.0
    else:
        display_values = counts_matrix

    category_palette: dict[str, str] = {}
    for idx, category in enumerate(categories):
        category_palette[category] = PALETTE[idx % len(PALETTE)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax_lines = ax
    ax_bars = ax_lines.twinx()

    if use_periods:
        x_coords = np.arange(len(timeline_labels))
        ax_lines.set_xticks(x_coords)
        ax_lines.set_xticklabels(timeline_labels, rotation=0, ha="center")
        ax_bars.set_xticks(x_coords)
    else:
        x_coords = np.array(timeline_labels, dtype=float)

    for category in categories:
        ax_lines.plot(
            x_coords,
            display_values[category].values,
            marker="o",
            linewidth=2,
            markersize=4.5,
            color=category_palette[category],
            label=category.replace("_", " ").title(),
        )

    bar_color = PALETTE[min(len(PALETTE) - 1, len(categories))]
    ax_bars.bar(
        x_coords,
        timeline_article_counts.values,
        width=0.6,
        color=bar_color,
        alpha=0.35,
        edgecolor="#2b1f1f",
        linewidth=0.4,
        label="Articles analysés",
    )

    if show_percentages:
        ax_lines.set_ylabel("Part des mentions d'acteurs")
        ax_lines.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        ax_lines.set_ylabel("Nombre de mentions d'acteurs")
        ax_lines.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax_lines.set_xlabel("Période" if use_periods else "Année de publication")
    ax_bars.set_ylabel("Nombre d'articles analysés")
    ax_lines.grid(axis="y", alpha=0.25)

    default_title = (
        "Évolution de la part des mentions par catégorie d'acteurs"
        if show_percentages
        else "Évolution du nombre de mentions par catégorie d'acteurs"
    )
    ax_lines.set_title(default_title, pad=18)

    if stereotype_name:
        ax_lines.text(
            0.5,
            1.15,
            f"« {stereotype_name} »",
            transform=ax_lines.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#4a2f2f",
        )

    legend_lines = ax_lines.legend(
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        borderaxespad=0.0,
        title="Catégories d'acteurs",
        frameon=False,
    )
    legend_bars = ax_bars.legend(
        loc="upper left", bbox_to_anchor=(1.12, 0.3), borderaxespad=0.0, frameon=False
    )
    for legend in (legend_lines, legend_bars):
        for text in legend.get_texts():
            text.set_color("#2c1e1e")

    fig.tight_layout(rect=(0, 0, 0.82, 0.9))
    return ax_lines, ax_bars


def plot_prism_trends(
    df: pd.DataFrame,
    *,
    prism_column: str = "st_prismes",
    pub_date_col: str = "pub_date",
    show_percentages: bool = False,
    periods: Optional[Sequence[Mapping[str, str]]] = None,
    stereotype_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (18, 6),
    max_prisms: Optional[int] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot the evolution of prism usage over time (absolute or percentage).

    Parameters
    ----------
    df:
        DataFrame containing publication dates and prism annotations.
    prism_column:
        Column that lists the prism(s) associated with each article.
    pub_date_col:
        Column containing publication dates (parsable by pandas).
    show_percentages:
        When True displays percentages; when False displays absolute counts.
    periods:
        Optional ordered list of dicts describing custom periods
        (``title``, ``start_date``, ``end_date``). Dates must be ISO parseable.
    stereotype_name:
        Optional text displayed above the title as a quote.
    max_prisms:
        Optional integer limiting how many prismes are displayed (top-N overall).
    ax / figsize:
        Matplotlib axes/figure configuration.
    """

    df = df.copy()

    if prism_column not in df.columns:
        raise ValueError(f"'{prism_column}' column not found in DataFrame.")
    if pub_date_col not in df.columns:
        raise ValueError(f"'{pub_date_col}' column not found in DataFrame.")

    pub_dates = pd.to_datetime(df[pub_date_col], errors="coerce")
    valid_mask = pub_dates.notna()
    if not valid_mask.any():
        raise ValueError("Aucune date de publication valide n'a été trouvée.")

    df = df.loc[valid_mask].copy()
    timestamp_series = pub_dates.loc[valid_mask]

    def _parse_periods(period_defs: Sequence[Mapping[str, str]]) -> List[dict]:
        parsed: List[dict] = []
        for period in period_defs:
            title = period.get("title")
            start = pd.to_datetime(period.get("start_date"), errors="coerce")
            end_value = period.get("end_date")
            if isinstance(end_value, str) and end_value.lower() in {
                "present",
                "actuel",
            }:
                end = pd.Timestamp.max
            elif end_value is None:
                end = pd.Timestamp.max
            else:
                end = pd.to_datetime(end_value, errors="coerce")
            if title is None or pd.isna(start) or pd.isna(end):
                raise ValueError(
                    "Chaque période doit inclure un titre, un start_date et un end_date "
                    "parsables (format ISO recommandé)."
                )
            if start > end:
                raise ValueError(
                    f"La période '{title}' a un start_date postérieur à son end_date."
                )
            parsed.append({"title": title, "start": start, "end": end})
        parsed.sort(key=lambda item: item["start"])
        return parsed

    use_periods = periods is not None
    if use_periods:
        parsed_periods = _parse_periods(periods or [])
        timeline_labels = [
            "\n".join(textwrap.wrap(item["title"], width=28)) for item in parsed_periods
        ]
        assigned_labels: List[Optional[str]] = []
        unmatched_dates: List[pd.Timestamp] = []
        label_mapping = {
            period["title"]: timeline_labels[idx]
            for idx, period in enumerate(parsed_periods)
        }
        for ts in timestamp_series:
            label = None
            for period in parsed_periods:
                if period["start"] <= ts <= period["end"]:
                    label = label_mapping[period["title"]]
                    break
            assigned_labels.append(label)
            if label is None:
                unmatched_dates.append(ts)
        if unmatched_dates:
            formatted = ", ".join(
                sorted({ts.strftime("%Y-%m-%d") for ts in unmatched_dates})
            )
            raise ValueError(
                "Certaines dates de publication ne correspondent à aucune période fournie: "
                f"{formatted}"
            )
        df["_timeline_label"] = assigned_labels
    else:
        df["_timeline_label"] = timestamp_series.dt.year.astype(int)
        timeline_labels = sorted(df["_timeline_label"].unique())

    if not timeline_labels:
        raise ValueError("Aucun point temporel n'a été créé pour ce graphique.")

    timeline_article_counts = (
        df.groupby("_timeline_label").size().reindex(timeline_labels, fill_value=0)
    )

    timeline_prism_counts: dict[Any, Counter[str]] = {
        label: Counter() for label in timeline_labels
    }

    observed_prisms: set[str] = set()

    for label, value in zip(df["_timeline_label"], df[prism_column]):
        if isinstance(value, (list, tuple, set)):
            prisms = value
        elif pd.isna(value) or value is None:
            continue
        else:
            prisms = [value]

        for prism in prisms:
            if pd.isna(prism):
                continue
            normalized = str(prism).strip()
            if not normalized:
                continue
            timeline_prism_counts[label][normalized] += 1
            observed_prisms.add(normalized)

    if not observed_prisms:
        raise ValueError(
            "Aucun prisme exploitable n'a été trouvé dans la colonne fournie."
        )

    categories = sorted(observed_prisms)

    counts_matrix = pd.DataFrame(
        {
            category: [
                timeline_prism_counts[label].get(category, 0)
                for label in timeline_labels
            ]
            for category in categories
        },
        index=timeline_labels,
    )

    if show_percentages:
        counts_sum = counts_matrix.sum(axis=1).replace(0, np.nan)
        display_values = counts_matrix.div(counts_sum, axis=0).fillna(0) * 100.0
    else:
        display_values = counts_matrix

    if max_prisms is not None:
        if max_prisms <= 0:
            raise ValueError("max_prisms doit être un entier strictement positif.")
        prism_totals = counts_matrix.sum(axis=0).sort_values(ascending=False)
        selected = prism_totals.index[:max_prisms].tolist()
        counts_matrix = counts_matrix[selected]
        display_values = display_values[selected]
        categories = selected

    category_palette: dict[str, str] = {}
    for idx, category in enumerate(categories):
        category_palette[category] = PALETTE[idx % len(PALETTE)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax_lines = ax
    ax_bars = ax_lines.twinx()

    if use_periods:
        x_coords = np.arange(len(timeline_labels))
        ax_lines.set_xticks(x_coords)
        ax_lines.set_xticklabels(timeline_labels, rotation=0, ha="center")
        ax_bars.set_xticks(x_coords)
    else:
        x_coords = np.array(timeline_labels, dtype=float)

    for category in categories:
        ax_lines.plot(
            x_coords,
            display_values[category].values,
            marker="o",
            linewidth=2,
            markersize=4.5,
            color=category_palette[category],
            label=category,
        )

    bar_color = PALETTE[min(len(PALETTE) - 1, len(categories))]
    ax_bars.bar(
        x_coords,
        timeline_article_counts.values,
        width=0.6,
        color=bar_color,
        alpha=0.35,
        edgecolor="#2b1f1f",
        linewidth=0.4,
        label="Articles analysés",
    )

    if show_percentages:
        ax_lines.set_ylabel("Part des prismes mobilisés")
        ax_lines.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        ax_lines.set_ylabel("Nombre de prismes mobilisés")
        ax_lines.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax_lines.set_xlabel("Période" if use_periods else "Année de publication")
    ax_bars.set_ylabel("Nombre d'articles analysés")
    ax_lines.grid(axis="y", alpha=0.25)

    default_title = (
        "Évolution de la part des prismes mobilisés"
        if show_percentages
        else "Évolution du nombre de prismes mobilisés"
    )
    ax_lines.set_title(default_title, pad=18)

    if stereotype_name:
        ax_lines.text(
            0.5,
            1.15,
            f"« {stereotype_name} »",
            transform=ax_lines.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#4a2f2f",
        )

    legend_lines = ax_lines.legend(
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        borderaxespad=0.0,
        title="Prismes",
        frameon=False,
    )
    for text in legend_lines.get_texts():
        text.set_color("#2c1e1e")

    fig.tight_layout(rect=(0, 0, 0.86, 0.9))
    return ax_lines, ax_bars
