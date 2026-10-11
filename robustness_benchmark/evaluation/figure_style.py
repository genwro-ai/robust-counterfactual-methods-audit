"""One visual style for every figure: ink, fonts, method styles, and colormap.

Colour follows the method in every figure. Non-robust baselines are neutral
greys, each robust method (or pair of variants) has one of seven categorical
hues, and every method has its own marker, so identity never rests on colour
alone. The hues are seven slots of a colour-vision-deficiency-validated
categorical palette, kept in their validated order (adjacent pairs: protan/deutan
ΔE >= 16, normal-vision ΔE >= 30). No seven hues stay pairwise distinct in a
scatter for every reader, so the markers are part of the encoding.
"""

import functools
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EMPTY = "#f0efec"  # cells without a value
SURFACE = "white"

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 9
LABEL_SIZE = 7.5
TICK_SIZE = 7
SMALL_SIZE = 6.5
MARKER_AREA = 40

RC = {
    "font.family": FONT_FAMILY,
    "font.size": LABEL_SIZE,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.labelsize": LABEL_SIZE,
    "axes.linewidth": 0.8,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": SECONDARY,
    "ytick.labelcolor": SECONDARY,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": TICK_SIZE,
    "legend.labelcolor": INK,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Sequential: one hue (the reference palette's blue ramp), light to dark.
ROBUSTNESS_CMAP = LinearSegmentedColormap.from_list(
    "robustness",
    ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    marker: str


GREY = SECONDARY
AQUA, VIOLET, MAGENTA, GREEN = "#1baf7a", "#4a3aa7", "#e87ba4", "#008300"
YELLOW, BLUE, RED = "#eda100", "#2a78d6", "#e34948"
MARGIN_LABEL = "Confidence-margin baseline"

# Legend order: baselines, then the hues in their validated order.
METHOD_STYLES = {
    "wachter": MethodStyle("Wachter", GREY, "o"),
    "kdtree": MethodStyle("KD-tree", GREY, "s"),
    "growing_spheres": MethodStyle("Growing spheres", MUTED, "D"),
    **{
        f"gs_margin_{margin}": MethodStyle(f"GS margin 0.{margin}", MUTED, ".")
        for margin in (60, 70, 80, 90)
    },
    "rbr": MethodStyle("RBR", AQUA, "h"),
    "roar": MethodStyle("ROAR", VIOLET, "v"),
    "roar_lime": MethodStyle("ROAR-LIME", VIOLET, "^"),
    "betarce": MethodStyle("BetaRCE", MAGENTA, "p"),
    "rnce": MethodStyle("RNCE", GREEN, "d"),
    "apas": MethodStyle("APΔS", YELLOW, "*"),
    "robx_balanced": MethodStyle("RobX balanced", BLUE, "P"),
    "robx_robust": MethodStyle("RobX robust-first", BLUE, "X"),
    "trex": MethodStyle("T:Rex", RED, ">"),
}
CATEGORICAL_HUES = (AQUA, VIOLET, MAGENTA, GREEN, YELLOW, BLUE, RED)


def method_style(method: str) -> MethodStyle:
    """The method's style; unknown methods are drawn in ink with their own name."""

    return METHOD_STYLES.get(method, MethodStyle(method, INK, "o"))


def styled(plot):
    """Run a plotting function under the shared rcParams."""

    @functools.wraps(plot)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC):
            return plot(*args, **kwargs)

    return wrapper


def scatter_method(axis, method: str, x, y, *, size: float = MARKER_AREA, **kwargs):
    style = method_style(method)
    return axis.scatter(
        x,
        y,
        s=size * (1.6 if style.marker == "*" else 1),
        color=style.color,
        marker=style.marker,
        edgecolor=SURFACE,
        linewidth=0.6,
        label=style.label,
        **kwargs,
    )


def method_handle(method: str) -> Line2D:
    style = method_style(method)
    return Line2D(
        [],
        [],
        linestyle="",
        marker=style.marker,
        markersize=7.5 if style.marker == "*" else 6,
        markerfacecolor=style.color,
        markeredgecolor=SURFACE,
        markeredgewidth=0.5,
        label=style.label,
    )


def legend_methods(methods: Iterable[str]) -> list[str]:
    """``methods`` in legend order: the style table's, then unknown ones."""

    present = set(methods)
    return [method for method in METHOD_STYLES if method in present] + sorted(
        present.difference(METHOD_STYLES)
    )


def style_axis(axis, grid: str | None = "both") -> None:
    """Hairline axes with a recessive solid grid along ``grid`` (x, y, both)."""

    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    if grid:
        axis.grid(True, axis=grid, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)


def panel_title(axis, index: int, title: str) -> None:
    """Bold panel letter followed by the panel title, left-aligned."""

    axis.set_title(f"{chr(ord('A') + index)}   {title}", pad=6)


def mark_method_ticks(axis, methods: list[str]) -> None:
    """Draw each method's marker between its y tick label and the axis."""

    beside_axis = offset_copy(
        axis.get_yaxis_transform(), axis.figure, x=-5.5, units="points"
    )
    for y, method in zip(axis.get_yticks(), methods, strict=True):
        style = method_style(method)
        axis.plot(
            0,
            y,
            marker=style.marker,
            markersize=5.5 if style.marker == "*" else 4.2,
            color=style.color,
            markeredgecolor=SURFACE,
            markeredgewidth=0.4,
            transform=beside_axis,
            clip_on=False,
        )
    axis.tick_params(axis="y", length=0, pad=11)


def save_figure(figure, path: Path) -> None:
    """Write ``path`` as PDF and 300 dpi PNG, then close the figure."""

    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        figure.savefig(
            path.with_suffix(suffix), bbox_inches="tight", pad_inches=0.06, **options
        )
    plt.close(figure)
