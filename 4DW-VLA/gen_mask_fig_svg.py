#!/usr/bin/env python3
"""Generate SVG version of the UBCA attention mask for easy manual editing."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


blocks = ["VLM\nPrefix", "4D Future\nExpert", "World Knowledge\nExpert", "Action\nExpert"]
relations = [
    ["Causal\nself", "Masked", "Masked", "Masked"],
    ["Full\ncross", "Causal\nself", "Causal\nUBCA", "Masked"],
    ["Full\ncross", "Causal\nUBCA", "Causal\nself", "Masked"],
    ["Full\ncross", "Full\ncross", "Full\ncross", "Causal\nself"],
]

colors = {
    "Causal\nself": "#DCEAF7",
    "Full\ncross": "#E8F4E8",
    "Causal\nUBCA": "#FFF0D9",
    "Masked": "#34495E",
}

fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=220)
fig.patch.set_facecolor("white")
ax.set_xlim(-1.55, 4.1)
ax.set_ylim(4.35, -1.45)
ax.axis("off")

ax.text(1.55, -1.25, "Key (attended stream)", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#263238")
ax.text(-1.42, 1.65, "Query\nstream", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#263238")

for col, label in enumerate(blocks):
    ax.text(col + 0.5, -0.72, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="#263238", linespacing=1.2)

for row, label in enumerate(blocks):
    ax.text(-0.18, row + 0.5, label, ha="right", va="center",
            fontsize=9, fontweight="bold", color="#263238", linespacing=1.2)
    for col, relation in enumerate(relations[row]):
        text_color = "white" if relation == "Masked" else "#263238"
        rect = patches.Rectangle(
            (col, row), 1, 1, facecolor=colors[relation],
            edgecolor="white", linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(col + 0.5, row + 0.5, relation, ha="center", va="center",
                fontsize=8.3, color=text_color, fontweight="bold", linespacing=1.15)

ax.add_patch(patches.Rectangle((0, 0), 4, 4, fill=False,
                               edgecolor="#263238", linewidth=1.8))

legend = [
    ("#DCEAF7", "Causal self-attention"),
    ("#E8F4E8", "Full cross-attention"),
    ("#FFF0D9", "Causal UBCA"),
    ("#34495E", "Masked"),
]
for idx, (color, label) in enumerate(legend):
    x = 0.05 + (idx % 2) * 2.15
    y = 3.95 + (idx // 2) * 0.42
    ax.add_patch(patches.Rectangle((x, y), 0.18, 0.18, facecolor=color,
                                   edgecolor="#78909C", linewidth=0.7))
    ax.text(x + 0.26, y + 0.09, label, ha="left", va="center", fontsize=8,
            color="#455A64")

plt.tight_layout(pad=0.35)
out_dir = "/home/l/文档/Research/InternVLA-A-series/4DW-VLA/figures"
plt.savefig(f"{out_dir}/ubca_mask.svg", bbox_inches="tight", pad_inches=0.12)
print("Saved: figures/ubca_mask.svg")
