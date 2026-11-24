"""Regenerate and fix all figures identified in the IEEE Access audit:
1. fig_inference_ablation.pdf:
   - Footer note: 'DeepGlobe test split, 623 tiles' (not validation split)
   - Labels on left/right margins not clipped: bbox_inches='tight', pad_inches=0.1
   - Ground truth label moved to top of vertical dotted line in panel (d)
2. fig_network.pdf and fig_network_v3.pdf:
   - Panel (b) channel shift labels top-to-bottom / upright ('up', not 'dn')
   - Panel (a) strip conv: lowercase w_{ij}, colon in 3x1:
   - Saved as both fig_network.pdf and fig_network_v3.pdf
3. fig_graph_analysis.pdf:
   - Panel (c) bottom legend not clipped: bbox_extra_artists, extra bottom margin
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Patch
import cv2
import networkx as nx
import glob
import random

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
OUT_DIR = os.path.join(REPO, "docs", "paper", "figures")
RESULTS_JSON = os.path.join(REPO, "figures", "real", "v4", "paper_results", "results.json")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================================
# 1. Regenerate fig_inference_ablation.pdf
# =========================================================================
def fix_inference_ablation():
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    results_final = data["native_resolution"]["final"]
    
    abl = ["single_raw", "single_hyst", "single_hyst_close", "single_full_orig", "single_full",
           "tta_raw", "tta_full_orig", "tta_full"]
    variant_names = {
        "single_raw": "single pass, p > 0.5",
        "single_hyst": "hysteresis",
        "single_hyst_close": "hysteresis + closing",
        "single_full_orig": "+ gap bridging (original)",
        "single_full": "+ gap bridging (border fix)",
        "tta_raw": "TTA, p > 0.5",
        "tta_full_orig": "TTA + post-proc. (original bridging)",
        "tta_full": "TTA + post-proc. (deployed)"
    }
    abl_labels = [variant_names[v] for v in abl]
    
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 0.35 * len(abl) + 1.05), sharey=True)
    panels = [
        ("iou", "IoU (%)", 100, "{:.1f}"),
        ("relaxed_f1", "Relaxed F1 (%)", 100, "{:.1f}"),
        ("cldice", "clDice (%)", 100, "{:.1f}"),
        ("median_components_per_tile", "Components / tile (median)", 1, "{:.0f}")
    ]
    
    y = np.arange(len(abl))[::-1]
    
    for ax, (k, title, sc, fmt), letter in zip(axes, panels, "abcd"):
        vals = [sc * results_final[v][k] for v in abl]
        cols = ["#1f4e9c" if v == "tta_full" else "#86b6ef" for v in abl]
        cis = [[sc * c for c in results_final[v]["ci95"][k]] for v in abl] if k in results_final[abl[0]]["ci95"] else None
        
        ax.barh(y, vals, height=0.64, color=cols, edgecolor="white", linewidth=1.0)
        if cis:
            for yi, (lo, hi) in zip(y, cis):
                ax.plot([lo, hi], [yi, yi], color="#1a1a1a", lw=0.8, solid_capstyle="butt")
        
        reach = max((ci[1] if ci else v) for v, ci in zip(vals, cis or [None] * len(vals)) if v == v)
        top = 22 if k == "median_components_per_tile" else reach * 1.20
        for yi, v, ci in zip(y, vals, cis or [None] * len(vals)):
            xpos = (ci[1] if ci else v) + top * 0.02
            ax.text(xpos, yi, fmt.format(v), va="center", fontsize=6.8, color="#202124")
            
        ax.set_xlim(0, top)
        ax.set_yticks(y)
        ax.set_yticklabels(abl_labels, fontsize=7.2)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=7.8, fontweight="bold")
        ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, fontweight="bold", color="#1a1a1a")

    gt_cc = results_final["single_raw"]["median_gt_components_per_tile"]
    axes[3].axvline(gt_cc, color="#5f6368", lw=0.9, ls=":")
    # Fix: place ground truth text at the top of the vertical line to prevent crowding tick 10
    axes[3].text(gt_cc + 0.4, y[0] + 0.4, f"ground truth: {gt_cc:.0f}", fontsize=6.8,
                 color="#202124", fontweight="bold", va="bottom", ha="left")

    # Fix: Correct caption footer from "validation split" to "test split"
    foot_text = "DeepGlobe test split, 623 tiles, native 1024² resolution. Proposed model; dark bar = deployed configuration."
    fig.text(0.01, 0.02, foot_text, fontsize=6.8, color="#5f6368", va="bottom")
    
    fig.subplots_adjust(left=0.34, right=0.98, top=0.92, bottom=0.12, wspace=0.18)
    
    out_pdf = os.path.join(OUT_DIR, "fig_inference_ablation.pdf")
    out_png = os.path.join(OUT_DIR, "fig_inference_ablation.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("Saved fixed fig_inference_ablation to", out_pdf)


# =========================================================================
# 2. Regenerate fig_network.pdf & fig_network_v3.pdf
# =========================================================================
def fix_network():
    NAVY, BLUE, TEAL, PURPLE = "#1b2a49", "#1f4e9c", "#1d6b52", "#3d2b7a"
    GOLD, RED, MAROON, GREY = "#b8860b", "#c0392b", "#8e1b1b", "#5f6368"
    LIGHT = {NAVY: "#eef1f7", BLUE: "#edf3fc", TEAL: "#ecf6f1", PURPLE: "#f1eef9",
             GOLD: "#fbf6e9", RED: "#fcefee", MAROON: "#f8ecec", GREY: "#f3f4f5"}

    fig = plt.figure(figsize=(7.16, 4.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.16); ax.set_ylim(0, 4.45); ax.axis("off")
    r = fig.canvas.get_renderer()

    def fit(t, x0, x1, y0=None, y1=None):
        inv = ax.transData.inverted()
        for _ in range(40):
            bb = inv.transform(t.get_window_extent(r))
            ok = bb[0][0] >= x0 and bb[1][0] <= x1
            if y0 is not None:
                ok = ok and bb[0][1] >= y0 and bb[1][1] <= y1
            if ok:
                return t
            t.set_fontsize(t.get_fontsize() * 0.95)
        return t

    def arrow(x0, y0, x1, y1, color=BLUE, label=None, lpos=None, lw=1.4, ls="-", rad=0.0, fs=5.6):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=color, lw=lw,
                                    linestyle=ls, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
        if label:
            lx, ly = lpos if lpos else ((x0 + x1) / 2, (y0 + y1) / 2 + 0.06)
            ax.text(lx, ly, label, color=color, fontsize=fs, ha="center", va="bottom", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.5))

    # ---- panel (a): strip convolution ----
    px, py, pw, ph = 0.02, 2.62, 3.5, 1.8
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=BLUE, lw=0.9, ls="--"))
    ax.text(px + pw / 2, py + ph - 0.13, "(a) Strip convolution, adopted from HPLNet", color=BLUE, fontsize=7.2,
            weight="bold", ha="center", va="center")
    s = 0.22
    def grid(x0, y0, nr, nc, labels, fc):
        for i in range(nr):
            for j in range(nc):
                ax.add_patch(Rectangle((x0 + j * s, y0 - (i + 1) * s), s, s, fc=fc, ec=NAVY, lw=0.6))
                ax.text(x0 + j * s + s / 2, y0 - (i + 0.5) * s, labels[i][j], fontsize=5.2, ha="center", va="center")
    gy = py + ph - 0.42
    # All lowercase w_{ij}
    grid(px + 0.25, gy, 3, 3, [[f"$w_{{{i}{j}}}$" for j in range(1, 4)] for i in range(1, 4)], "#dbe7f7")
    ax.text(px + 0.25 + 1.5 * s, gy - 3 * s - 0.12, r"$3{\times}3$: $9\,C_{in}C_{out}$", fontsize=5.6, ha="center", va="top")
    ax.text(px + 1.18, gy - 1.5 * s, r"$\rightarrow$", fontsize=10, ha="center", va="center", color=BLUE)
    grid(px + 1.45, gy - s, 1, 3, [[f"$a_{j}$" for j in range(1, 4)]], "#cfe0f5")
    ax.text(px + 1.45 + 1.5 * s, gy - 3 * s - 0.12, r"$1{\times}3$: $3\,C_{in}C_{out}$", fontsize=5.6, ha="center", va="top")
    ax.text(px + 2.28, gy - 1.5 * s, r"then", fontsize=5.8, ha="center", va="center", color=BLUE)
    grid(px + 2.6, gy, 3, 1, [[f"$b_{i}$"] for i in range(1, 4)], "#cfe0f5")
    # Colon in 3x1:
    ax.text(px + 2.6 + s / 2, gy - 3 * s - 0.12, r"$3{\times}1$: $3\,C_{in}C_{out}$", fontsize=5.6, ha="center", va="top")
    t = ax.text(px + pw / 2, py + 0.2, "Factorised kernel: 6 instead of 9 weights per channel pair,\nstrong response to horizontal and vertical line segments",
                fontsize=5.6, ha="center", va="center", color="#202124")
    fit(t, px + 0.05, px + pw - 0.05)

    # ---- panel (b): channel shift ----
    qx = 3.64
    ax.add_patch(FancyBboxPatch((qx, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=TEAL, lw=0.9, ls="--"))
    ax.text(qx + pw / 2, py + ph - 0.13, "(b) Channel shift, adopted from HPLNet", color=TEAL, fontsize=7.2,
            weight="bold", ha="center", va="center")
    cols = ["#9ec5e8", "#f2c38b", "#b9dcb0", "#d7b8e6", "#e6e6e6"]
    names = ["up", "down", "left", "right", "unchanged"]
    bx, by = qx + 0.22, py + 0.5
    for k in range(5):
        wk = 0.18 if k < 4 else 0.5
        x0 = bx + (0.2 * k if k < 4 else 0.8)
        ax.add_patch(Rectangle((x0, by), wk, 0.8, fc=cols[k], ec=NAVY, lw=0.6))
        # Use rotation=-90 so text reads naturally top-to-bottom ('u' then 'p', never 'dn')
        ax.text(x0 + wk / 2, by + 0.4, names[k], fontsize=5.0, ha="center", va="center",
                rotation=0 if k == 4 else -90)
    ax.text(bx + 0.66, by + 0.9, r"Features $Z\in\mathbb{R}^{C\times h\times w}$", fontsize=5.6, ha="center")
    ax.annotate("", xy=(bx + 1.72, by + 0.4), xytext=(bx + 1.38, by + 0.4),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
    cx, cy = qx + 2.55, by + 0.4
    ax.add_patch(Rectangle((cx - 0.25, cy - 0.25), 0.5, 0.5, fc="#f7f7f7", ec=NAVY, lw=0.6))
    
    # 4 directional shift arrows with clear labels
    shift_dirs = [((0, 1), "up", (cx, cy + 0.55)),
                  ((0, -1), "down", (cx, cy - 0.55)),
                  ((-1, 0), "left", (cx - 0.65, cy)),
                  ((1, 0), "right", (cx + 0.65, cy))]
    for (dx, dy), lbl, (lx, ly) in shift_dirs:
        col = cols[0] if dy > 0 else (cols[1] if dy < 0 else (cols[2] if dx < 0 else cols[3]))
        ax.annotate("", xy=(cx + dx * 0.50, cy + dy * 0.44), xytext=(cx + dx * 0.22, cy + dy * 0.22),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        ax.text(lx, ly, lbl, fontsize=4.8, ha="center", va="center", color=TEAL, weight="bold")
    ax.text(cx, cy, "2 px", fontsize=5.4, ha="center", va="center")
    t = ax.text(qx + pw / 2, py + 0.2, "25% of channels in four groups, each moved by 2 pixels;\nno weights and no multiply operations",
                fontsize=5.6, ha="center", va="center", color="#202124")
    fit(t, qx + 0.05, qx + pw - 0.05)

    # ---- panel (c): encoder and decoder ----
    ax.text(0.02, 2.47, "(c) Encoder and decoder for a 256 x 256 input", color=NAVY, fontsize=7.2, weight="bold", ha="left", va="center")
    bw, bh = 1.22, 0.62
    xs = [0.55 + i * 1.33 for i in range(5)]
    ye = 1.62
    enc = [("Stem", "Strip conv, s 2", "32 x 128 x 128"),
           ("Encoder 1", "Inverted residual, s 2", "64 x 64 x 64"),
           ("Encoder 2", "MobileViT v2 (2), IR s 2", "96 x 32 x 32"),
           ("Encoder 3", "MobileViT v2 (2), IR s 2", "128 x 16 x 16"),
           ("Bottleneck", "MobileViT v2 (3)", "128 x 16 x 16")]
    def block(x, y, title, op, shape, color):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05", fc=LIGHT[color], ec=color, lw=1.1))
        t1 = ax.text(x + bw / 2, y + bh - 0.12, title, fontsize=6.2, weight="bold", color=color, ha="center", va="center")
        t2 = ax.text(x + bw / 2, y + bh / 2 - 0.02, op, fontsize=5.3, ha="center", va="center")
        ax.plot([x + 0.08, x + bw - 0.08], [y + 0.17, y + 0.17], color=color, lw=0.5, alpha=0.5)
        t3 = ax.text(x + bw / 2, y + 0.085, shape, fontsize=5.3, ha="center", va="center", color="#202124")
        for t in (t1, t2, t3):
            fit(t, x + 0.04, x + bw - 0.04)
    ax.text(0.02, ye + bh / 2, "Input\n3 x 256\nx 256", fontsize=5.2, ha="left", va="center")
    arrow(0.4, ye + bh / 2, xs[0], ye + bh / 2, NAVY, lw=1.0)
    for i, e in enumerate(enc):
        block(xs[i], ye, *e, BLUE)
        if i:
            arrow(xs[i - 1] + bw, ye + bh / 2, xs[i], ye + bh / 2, BLUE, lw=1.0)
    yg, gw, gh = 0.9, 0.95, 0.36
    yd = 0.02
    dec_x = [xs[3], xs[2], xs[1]]
    src = [xs[3], xs[2], xs[0]]
    skip = ["96 x 32 x 32", "64 x 64 x 64", "32 x 128 x 128"]
    dec = [("Decoder 3", "Up x2, concat, strip conv", "96 x 32 x 32"),
           ("Decoder 2", "Up x2, concat, strip conv", "64 x 64 x 64"),
           ("Decoder 1", "Up x2, concat, strip conv", "32 x 128 x 128")]
    for i, x in enumerate(dec_x):
        gx = x + (bw - gw) / 2
        ax.add_patch(FancyBboxPatch((gx, yg), gw, gh, boxstyle="round,pad=0,rounding_size=0.04", fc=LIGHT[GOLD], ec=GOLD, lw=1.1))
        ax.text(gx + gw / 2, yg + gh / 2, f"Gate {3 - i}", fontsize=5.8, weight="bold", color=GOLD, ha="center", va="center")
        if src[i] == x:
            arrow(x + 0.3, ye, x + 0.3, yg + gh, GOLD, lw=0.9, ls="--")
            ax.text(x + 0.36, (ye + yg + gh) / 2, "skip\n" + skip[i], fontsize=4.8, color=GOLD, ha="left", va="center")
        else:
            arrow(src[i] + bw / 2, ye, gx + 0.15, yg + gh, GOLD, lw=0.9, ls="--")
            ax.text((src[i] + bw / 2 + gx + 0.15) / 2 - 0.08, (ye + yg + gh) / 2 - 0.1, "skip " + skip[i], fontsize=4.8, color=GOLD, ha="right", va="center")
        block(x, yd, *dec[i], TEAL)
        arrow(gx + gw / 2, yg, gx + gw / 2, yd + bh, GOLD, lw=0.9)
    arrow(xs[4] + bw / 2, ye, xs[3] + bw, yd + bh * 0.75, TEAL, lw=1.0)
    ax.text((xs[4] + bw / 2 + xs[3] + bw) / 2 + 0.08, 1.05, "up x2,\ngating\nsignal", fontsize=4.8, color=TEAL, ha="left", va="center")
    arrow(xs[3], yd + bh / 2, xs[2] + bw, yd + bh / 2, TEAL, lw=1.0)
    arrow(xs[2], yd + bh / 2, xs[1] + bw, yd + bh / 2, TEAL, lw=1.0)
    block(xs[0] - 0.53, yd, "Head", "Up x2, 1x1 conv, sigmoid", "1 x 256 x 256", PURPLE)
    arrow(xs[1], yd + bh / 2, xs[0] - 0.53 + bw, yd + bh / 2, TEAL, lw=1.0)

    # Save to both fig_network.pdf and fig_network_v3.pdf
    for name in ("fig_network", "fig_network_v3"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.03)
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved fixed fig_network and fig_network_v3 to", OUT_DIR)


# =========================================================================
# 3. Regenerate fig_graph_analysis.pdf
# =========================================================================
def fix_graph_analysis():
    # Load measurements saved by graph_analysis.py
    json_path = os.path.join(OUT_DIR, "measurements", "graph_analysis.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(REPO, "figures", "real", "measurements", "graph_analysis.json")
    
    with open(json_path) as f:
        data = json.load(f)
    
    FRACS = np.array(data["fractions"])
    results = data["tiles"]
    
    # Load sample image for tile 100892
    tile_name = "100892"
    sample_path = os.path.join(REPO, "data", "samples", f"{tile_name}_sat.jpg")
    img = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
    
    # Run skeleton graph on the prediction or use precomputed nodes
    from scripts.paper_figures.graph_analysis import predict, skeleton_graph
    G = skeleton_graph(predict(img))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    bc = nx.betweenness_centrality(G, weight="length", normalized=True)
    
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.65), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    
    # Panel (a)
    ax = axs[0]; ax.imshow(img, alpha=0.6)
    for u, v, d in G.edges(data=True):
        p = np.array(d["pixels"]); ax.plot(p[:, 1], p[:, 0], color="white", lw=1.2)
    ys, xs = zip(*G.nodes()); c = [bc[n] for n in G.nodes()]
    sc = ax.scatter(xs, ys, c=c, cmap="plasma", s=14, edgecolors="black", linewidths=0.3, zorder=3)
    top = max(bc, key=bc.get); ax.scatter([top[1]], [top[0]], s=80, facecolors="none", edgecolors="red", lw=1.2, zorder=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"(a) Betweenness, tile {tile_name}", fontsize=8, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
    
    # Panel (b)
    ax = axs[1]; pct = FRACS * 100
    T = np.mean([r["targeted"] for r in results.values()], axis=0)
    Rr = np.mean([r["random"] for r in results.values()], axis=0)
    ax.plot(pct, T, "o-", color="#c0392b", ms=3, label="Highest betweenness first")
    ax.plot(pct, Rr, "s--", color="#2c3e50", ms=3, label="Random")
    ax.set_xlabel("Nodes removed (%)"); ax.set_ylabel("Relative efficiency"); ax.set_ylim(0, 1.05)
    ax.set_title(f"(b) Mean over {len(results)} tiles", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False); ax.grid(alpha=0.3)
    
    # Panel (c)
    ax = axs[2]; names = list(results); y = np.arange(len(names))
    ax.barh(y - 0.2, [results[n]["R_targeted_10pct"] for n in names], 0.4, color="#c0392b", label="Targeted")
    ax.barh(y + 0.2, [results[n]["R_random_10pct"] for n in names], 0.4, color="#7f8c8d", label="Random")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5); ax.set_xlim(0, 1); ax.set_xlabel("Resilience index R")
    ax.set_title("(c) R after removing 10% of nodes", fontsize=8, fontweight="bold")
    # Position legend cleanly below panel with proper spacing
    leg = ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.invert_yaxis()
    
    fig.subplots_adjust(bottom=0.22, top=0.90, wspace=0.28)
    out_pdf = os.path.join(OUT_DIR, "fig_graph_analysis.pdf")
    out_png = os.path.join(OUT_DIR, "fig_graph_analysis.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1, bbox_extra_artists=[leg])
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1, bbox_extra_artists=[leg])
    plt.close(fig)
    print("Saved fixed fig_graph_analysis to", out_pdf)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fix_inference_ablation()
    fix_network()
    fix_graph_analysis()
    print("All figures successfully fixed and exported!")
