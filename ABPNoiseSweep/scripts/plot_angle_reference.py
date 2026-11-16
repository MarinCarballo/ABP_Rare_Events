"""Draw the angle convention e(theta)=(cos(theta),sin(theta)); no simulation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch
import numpy as np


def main():
    output = Path(__file__).resolve().parents[1] / "docs"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.2, 7.3), facecolor="white")
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.19, top=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")
    ink, muted, blue, red = "#18394d", "#75838c", "#1679ad", "#b43f3f"

    ax.add_patch(Circle((0, 0), 1, facecolor="#f7fafc", edgecolor="#b1c0ca", lw=1.5))
    for start, end in (((-1.19, 0), (1.19, 0)), ((0, -1.19), (0, 1.19))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->", mutation_scale=12,
                                    color=muted, lw=1.25, zorder=2))

    # A pi orientation points exactly along the negative x axis.
    ax.add_patch(FancyArrowPatch((0, 0), (-1, 0), arrowstyle="-|>",
                                mutation_scale=21, lw=3, color=red, zorder=4))
    ax.text(-1.23, 0, "$\\pi = 180^\\circ$\n$-x$ (left)",
            ha="right", va="center", fontsize=13, color=red, linespacing=1.6)
    ax.text(1.23, 0, "$0$ or $2\\pi$\n$0^\\circ$ or $360^\\circ$\n$+x$ (right)",
            ha="left", va="center", fontsize=12, color=ink, linespacing=1.45)
    ax.text(0, 1.24, "$\\pi/2 = 90^\\circ$\n$+y$ (up)",
            ha="center", va="bottom", fontsize=12.5, color=ink, linespacing=1.5)
    ax.text(0, -1.24, "$3\\pi/2 = 270^\\circ$\n$-y$ (down)",
            ha="center", va="top", fontsize=12.5, color=ink, linespacing=1.5)

    # A separate example shows where theta is measured from and its sign.
    theta = np.pi / 4
    direction = (np.cos(theta), np.sin(theta))
    ax.add_patch(FancyArrowPatch((0, 0), direction, arrowstyle="-|>",
                                mutation_scale=21, color=blue, lw=2.7, zorder=4))
    ax.plot([direction[0], direction[0]], [0, direction[1]], ls=":", lw=1.1, color=blue)
    ax.plot([0, direction[0]], [direction[1], direction[1]], ls=":", lw=1.1, color=blue)
    ax.text(0.76, 0.79, "$\\mathbf{e}(\\theta)$\nexample: $\\theta=\\pi/4$",
            ha="left", va="center", color=blue, fontsize=11.5, linespacing=1.5)
    ax.add_patch(Arc((0, 0), 0.65, 0.65, theta1=0, theta2=45, color=blue, lw=1.8))
    ax.add_patch(FancyArrowPatch((0.325*np.cos(theta-0.09), 0.325*np.sin(theta-0.09)),
                                (0.325*np.cos(theta), 0.325*np.sin(theta)),
                                arrowstyle="-|>", mutation_scale=11, color=blue, lw=1.2))
    ax.text(0.40, 0.13, "$\\theta$", fontsize=15, color=blue)
    ax.plot(0, 0, "o", color=ink, ms=4, zorder=5)
    ax.text(-0.045, -0.055, "O", ha="right", va="top", color=muted, fontsize=10)

    fig.suptitle("Angle reference in the ABP model", y=0.965,
                 fontsize=19, fontweight="bold", color=ink)
    fig.text(0.5, 0.906, "$\\mathbf{e}(\\theta)=(\\cos\\theta,\\sin\\theta)$",
             ha="center", fontsize=15, color=ink)
    fig.text(0.5, 0.865, "Measured counterclockwise from +x, in radians",
             ha="center", fontsize=11, color=muted)
    fig.text(0.5, 0.108, "The angle describes propulsion orientation, not position or displacement.",
             ha="center", fontsize=10, color=ink)
    fig.text(0.5, 0.070, "Histograms wrap angles into [0, 2π): 0 and 2π point the same way.",
             ha="center", fontsize=10, color=muted)

    for extension in ("png", "pdf"):
        path = output / f"angle_reference.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
