import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")

plt.rcParams.update({"font.family": "serif", "font.size": 9})

BLUE, GREEN, ORANGE, RED, PURPLE, GREY = "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#6a3d9a", "#888888"
INK = "#1a1a1a"


def box(ax, xy, w, h, text, fc, ec=INK, fontsize=8.5, fontweight="normal", zorder=3):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=INK, zorder=zorder + 1, linespacing=1.35)
    return b


def arrow(ax, p0, p1, color=INK, lw=1.3, style="-|>", connectionstyle="arc3,rad=0.0", zorder=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                         connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(a)


fig = plt.figure(figsize=(13, 8.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.55], hspace=0.14)

# =============================================================================
# PANEL A -- macro pipeline: the four real-time modules
# =============================================================================
axA = fig.add_subplot(gs[0])
axA.set_xlim(0, 10)
axA.set_ylim(0, 3.2)
axA.axis("off")
axA.patch.set_alpha(0)
axA.set_title("(A) End-to-End Real-Time Pipeline", fontsize=11, fontweight="bold", loc="left")

mod_w, mod_h, gap = 1.95, 1.55, 0.55
xs = [0.25, 0.25 + (mod_w + gap), 0.25 + 2 * (mod_w + gap), 0.25 + 3 * (mod_w + gap)]
y0 = 1.0

box(axA, (xs[0], y0), mod_w, mod_h,
    "Module 1\nData Quality\nControl\n(§2.1; NB03)", "#eaf1fb", ec=BLUE)
box(axA, (xs[1], y0), mod_w, mod_h,
    "Module 2\nOperating-State\nClassification\n(§2.1; NB04)", "#eaf6ea", ec=GREEN)
box(axA, (xs[2], y0), mod_w, mod_h,
    "Module 3\nForward-Model\nDispatch (PI / MBE)\n(§2.1–2.2; NB05)", "#fdf1e3", ec=ORANGE)
box(axA, (xs[3], y0), mod_w, mod_h,
    "Module 4\nPCE / Smolyak\nUQ  (this paper)\n(§3–4; detailed in B)", "#fbe9ea", ec=RED, fontweight="bold")

for i in range(3):
    arrow(axA, (xs[i] + mod_w, y0 + mod_h / 2), (xs[i + 1], y0 + mod_h / 2))

axA.text(xs[0] + mod_w / 2, y0 + mod_h + 0.32, "Pwf(t), q(t)\n(raw PDHMS/MPFM)",
          ha="center", va="bottom", fontsize=7.3, color=GREY)
axA.text(xs[1] + mod_w / 2, y0 + mod_h + 0.32, "Pwf, q\n(QC'd, lag-corrected)",
          ha="center", va="bottom", fontsize=7.3, color=GREY)
axA.text(xs[2] + mod_w / 2, y0 + mod_h + 0.32, "state ∈ {0,1,2,3}",
          ha="center", va="bottom", fontsize=7.3, color=GREY)
axA.text(xs[3] + mod_w / 2, y0 + mod_h + 0.32, "Pwf, q, Jeff, ct, state",
          ha="center", va="bottom", fontsize=7.3, color=GREY)

axA.annotate("", xy=(xs[3] + mod_w + 0.35, y0 + mod_h / 2), xytext=(xs[3] + mod_w, y0 + mod_h / 2),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
axA.text(xs[3] + mod_w + 0.45, y0 + mod_h / 2, "P̄(t), CI95,\nSobol' indices",
          ha="left", va="center", fontsize=7.8, color=INK)

axA.text(0.25, 0.35,
          "Modules 1–3 are the upstream deterministic estimator (§2.1); implementation details in the "
          "accompanying code repository (Section 6). Module 4, detailed in panel B, is the stochastic "
          "layer developed in this paper and attaches to Module 3's output without altering it.",
          fontsize=8, color=INK, ha="left", va="bottom", style="italic", wrap=True)

# =============================================================================
# PANEL B -- Module 4 internal detail, offline vs online as the primary structure
# =============================================================================
axB = fig.add_subplot(gs[1])
axB.set_xlim(0, 10.9)
axB.set_ylim(0, 5.7)
axB.axis("off")
axB.patch.set_alpha(0)
axB.set_title("(B) Module 4 Detail — PCE/Smolyak UQ Pipeline", fontsize=11, fontweight="bold", loc="left")

# offline swimlane
axB.add_patch(mpatches.FancyBboxPatch((0.2, 4.15), 9.6, 1.4, boxstyle="round,pad=0.02,rounding_size=0.04",
                                       linewidth=1.3, edgecolor=PURPLE, facecolor="#f5f0fa", zorder=1))
axB.text(0.45, 5.3, "OFFLINE — constructed once at initialisation", fontsize=9.3, fontweight="bold",
          color=PURPLE, va="top", zorder=10, clip_on=False)
box(axB, (0.5, 4.3), 2.6, 1.0, "Basis {Ψα}\n(Hermite × Legendre)\n(§3.1, Fig. 1)", "#ffffff", ec=PURPLE, fontsize=8)
box(axB, (3.5, 4.3), 3.0, 1.0, "Smolyak sparse grid\nnodes {ξ(j)}, weights {wj}\nNq=165  (§3.3, Fig. 3)", "#ffffff", ec=PURPLE, fontsize=8)
box(axB, (7.0, 4.3), 2.5, 1.0, "d=4 joint\ndistribution ρ(ξ)\n(§2.3, Fig. 2)", "#ffffff", ec=PURPLE, fontsize=8)
arrow(axB, (3.1, 4.8), (3.5, 4.8), color=PURPLE)
arrow(axB, (6.5, 4.8), (7.0, 4.8), color=PURPLE)

axB.annotate("", xy=(2.0, 4.15), xytext=(2.0, 3.75),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
axB.text(2.15, 3.95, "available to every timestep", fontsize=7.3, color=GREY, va="center")

# online swimlane
axB.add_patch(mpatches.FancyBboxPatch((0.2, 0.2), 9.6, 3.35, boxstyle="round,pad=0.02,rounding_size=0.04",
                                       linewidth=1.3, edgecolor=RED, facecolor="#fdf3f3", zorder=1))
axB.text(0.45, 3.36, "ONLINE — repeated every timestep (≈ 1.04 ms, §3.6)", fontsize=9.3, fontweight="bold",
          color=RED, va="top")

step_w, step_h = 2.05, 1.35
sx = [0.55, 0.55 + (step_w + 0.35), 0.55 + 2 * (step_w + 0.35), 0.55 + 3 * (step_w + 0.35)]
sy = 1.55

box(axB, (sx[0], sy), step_w, step_h,
    "1. Evaluate active\nforward model\n(PI, Eq. 1, or MBE,\nEq. 2) at 165 nodes", "#ffffff", ec=RED, fontsize=7.8)
box(axB, (sx[1], sy), step_w, step_h,
    "2. Project onto\nbasis → spectral\ncoefficients cα\n(Eq. 14)", "#ffffff", ec=RED, fontsize=7.8)
box(axB, (sx[2], sy), step_w, step_h,
    "3. Sample surrogate,\nclip to [0,Pi]\n(Eq. 16, post hoc,\nN=10,000)", "#ffffff", ec=RED, fontsize=7.8)
box(axB, (sx[3], sy), step_w, step_h,
    "4. Mean, CI95,\nSobol' S1/ST\n(Eqs. 17–18, 24)", "#ffffff", ec=RED, fontsize=7.8)

for i in range(3):
    arrow(axB, (sx[i] + step_w, sy + step_h / 2), (sx[i + 1], sy + step_h / 2), color=RED)

axB.annotate("", xy=(sx[0] - 0.32, sy + step_h - 0.2), xytext=(0.35, sy + step_h + 0.68),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, connectionstyle="arc3,rad=-0.2"))
axB.text(0.35, sy + step_h + 0.78, "Pwf, q, Jeff, ct, state\n(from Module 3)",
          fontsize=7.3, color=GREY, ha="left", va="bottom", zorder=10, clip_on=False)

axB.annotate("", xy=(sx[3] + step_w + 0.5, sy + step_h / 2), xytext=(sx[3] + step_w, sy + step_h / 2),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
axB.text(sx[3] + step_w + 0.6, sy + step_h / 2, "P̄(t),\nCI95,\nSobol'",
          fontsize=7.5, color=INK, ha="left", va="center")

# repeat loop -- clean L-shaped path entirely below the boxes, inside the online swimlane
loop_y = sy - 0.42
x_start = sx[3] + step_w / 2
x_end = sx[0] + step_w / 2
axB.plot([x_start, x_start], [sy - 0.06, loop_y], color=GREY, lw=1.2, ls="dashed", zorder=2)
axB.plot([x_start, x_end], [loop_y, loop_y], color=GREY, lw=1.2, ls="dashed", zorder=2)
axB.annotate("", xy=(x_end, sy - 0.06), xytext=(x_end, loop_y),
             arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2, linestyle="dashed"))
axB.text((x_start + x_end) / 2, loop_y - 0.1, "repeat for next timestep",
          fontsize=7.8, color=GREY, ha="center", va="top", style="italic")

fig.savefig(os.path.join(out_dir, "fig04_workflow.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig04_workflow.png")
